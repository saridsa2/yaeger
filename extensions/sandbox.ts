/**
 * Launch a harness as a Modal Sandbox, using the Modal TypeScript SDK.
 *
 * This replaces the old path (render Python -> shell out to `modal deploy`).
 * Modal Functions can only be defined in Python, but Sandboxes can be created
 * from TS with a GPU, a custom image, volumes, secrets and a TLS tunnel - which
 * is everything an inference server needs. So the user needs no Python and no
 * modal CLI, and nothing is assembled on their machine: the harness is config,
 * executed through SDK calls.
 *
 * Everything still runs on the user's own Modal account, billed to them.
 */

import { randomBytes } from "node:crypto";
import { ModalClient } from "modal";

import { parseReapTag, type ReapPolicy, type RunningSandbox } from "./reap.ts";
import { assertPackageName, assertSafe, buildArgv, type HarnessSpec } from "./render.ts";

export { sandboxesToReap, type ReapPolicy, type RunningSandbox } from "./reap.ts";

const HF_CACHE_VOLUME = "huggingface-cache";
const VLLM_CACHE_VOLUME = "vllm-cache";
const TRACE_VOLUME = "yaeger-traces";
const TRACE_DIR = "/traces";

let cached: ModalClient | null = null;

/** The SDK reads ~/.modal.toml, the same credentials `modal setup` writes. */
export function client(): ModalClient {
  if (!cached) cached = new ModalClient();
  return cached;
}

export async function modalConfigured(): Promise<string | null> {
  try {
    const c = client();
    await c.apps.fromName("yaeger", { createIfMissing: true });
    return null;
  } catch (e) {
    const m = (e as Error).message ?? "";
    if (/token|auth|credential|profile/i.test(m)) {
      return "No Modal credentials. Create a token at modal.com/settings/tokens and put it in ~/.modal.toml, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.";
    }
    return `Could not reach Modal: ${m.slice(0, 200)}`;
  }
}

/**
 * The API key the served endpoint requires.
 *
 * secrets.fromObject() creates an inline (unnamed) Secret, so there is nothing
 * to pre-create on the user's account and nothing to collide with - which also
 * removes the "secret does not exist on a fresh account" failure the CLI path had.
 * We keep the plaintext locally so pi can authenticate to the endpoint.
 */
export async function makeApiKeySecret(): Promise<{ secret: any; key: string }> {
  const key = `sk-yaeger-${randomBytes(24).toString("base64url")}`;
  const secret = await client().secrets.fromObject({ VLLM_API_KEY: key });
  return { secret, key };
}

/**
 * Environment for the serving container.
 *
 * `render.ts` has always applied `spec.engine.env` via `.env({...})`; the sandbox
 * path dropped it, so a harness declaring HF_HUB_ENABLE_HF_TRANSFER or VLLM_USE_V1
 * never got them. The harness is the authority on what the model needs, so spec
 * values win over our defaults.
 */
export function buildEnv(spec: HarnessSpec): Record<string, string> {
  const env: Record<string, string> = {
    HF_XET_HIGH_PERFORMANCE: "1",
    // Hypothesis under test, not a proven fix. vLLM derives its torch.compile
    // cache directory from `safe_hash(str(factors))`, and we measured three
    // different directories across three launches of an identical config - so
    // something in that string is not stable per process. Python randomises str
    // hashing per process, which reorders any set inside those factors. Pinning
    // the seed costs nothing here (single-tenant container) and, if the theory
    // holds, is what makes the compile cache reusable at all. Verify by booting
    // twice and comparing the cache directory before trusting it.
    PYTHONHASHSEED: "0",
  };

  // Unauthenticated HF pulls are rate-limited, which vLLM warns about on every
  // cold boot. Opt-in: only forwarded if the user has a token in their own env.
  const hfToken = process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN;
  if (hfToken) env.HF_TOKEN = hfToken;

  for (const [k, v] of Object.entries(spec.engine.env ?? {})) {
    env[assertSafe("env key", k)] = String(v);
  }
  return env;
}

export interface Launched {
  sandboxId: string;
  url: string;
  apiKey: string;
  /** When the sandbox was created - billing starts here, not at ready. */
  createdAt: string;
  gpu: string;
  count: number;
  usdPerHour: number;
}

/**
 * Start the model and wait until it answers.
 *
 * `idleTimeoutMs` is the cost guard: the sandbox terminates itself after a
 * period with no activity. `timeoutMs` is the hard ceiling that bounds spend
 * even if idle detection misbehaves - a Function has no equivalent.
 */
export async function launch(
  spec: HarnessSpec,
  opts: {
    onUpdate?: (s: string) => void;
    readyTimeoutMs?: number;
    trace?: boolean;
    reap?: ReapPolicy;
  } = {},
): Promise<Launched> {
  // A first boot on an empty cache pays a full weight download before it even
  // starts compiling, and expiry here terminates the sandbox - throwing away
  // both the wait and the download. Overridable so a known-slow cold start can
  // be given room without editing code.
  const readyTimeoutDefault = Number(process.env.YAEGERPI_READY_TIMEOUT_MIN ?? "15") * 60_000;
  const { onUpdate, readyTimeoutMs = readyTimeoutDefault } = opts;
  const c = client();

  const appName = `yaeger-${(spec.serving.served_model_name ?? spec.model.repo.split("/").pop()!)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")}`.slice(0, 60);

  onUpdate?.("connecting to your Modal account...");
  const app = await c.apps.fromName(appName, { createIfMissing: true });

  const imageRef = spec.engine.image;
  if (!imageRef) {
    throw new Error(
      "This harness has no runtime image. The sandbox path needs engine.image " +
        "(an image with the engine already installed, e.g. vllm/vllm-openai:v0.27.1).",
    );
  }

  onUpdate?.(`resolving image ${imageRef}...`);
  let image = await c.images.fromRegistry(imageRef);
  const layers: string[] = [];
  if (spec.engine.clear_entrypoint !== false) {
    // vllm/vllm-openai ships ENTRYPOINT ["vllm","serve"]; an explicit command
    // gets appended to it and the process fails with "unrecognized arguments".
    layers.push("ENTRYPOINT []");
  }

  // render.ts installs these via uv_pip_install; the sandbox path used to drop
  // them, so a harness asking for hf_transfer never got it.
  //
  // Validated as package names, not as argv tokens: the spec comes from the
  // harness service, and a pip flag here (--index-url=http://attacker/) would
  // run attacker code during the image build. `--` terminates option parsing as
  // a second line of defence in case the grammar above ever loosens.
  const extras = (spec.engine.extra_packages ?? []).map(assertPackageName);
  if (extras.length) {
    layers.push(`RUN pip install --no-cache-dir -- ${extras.join(" ")}`);
  }
  if (layers.length) image = image.dockerfileCommands(layers);

  const hfCache = await c.volumes.fromName(HF_CACHE_VOLUME, { createIfMissing: true });
  // torch.compile artifacts, DeepGEMM autotune results and JIT-compiled kernels
  // land here. Without it every boot re-pays several minutes of GPU warmup that
  // no amount of weight caching avoids - render.ts has always declared this
  // mount; the sandbox path was the one that dropped it.
  const vllmCache = await c.volumes.fromName(VLLM_CACHE_VOLUME, { createIfMissing: true });
  const traceVol = await c.volumes.fromName(TRACE_VOLUME, { createIfMissing: true });
  const { secret, key: apiKey } = await makeApiKeySecret();

  // buildArgv validates every token against an allowlist, so a hostile or
  // malformed spec cannot smuggle anything into the process arguments.
  //
  // Request tracing: vLLM logs each request, and we tee stdout to a file on a
  // persistent volume. The traces stay on the team's own Modal account - they
  // are never sent to yaeger, which is the whole sovereignty claim.
  //
  // No --api-key flag: vLLM falls back to the VLLM_API_KEY env var
  // (`args.api_key or envs.VLLM_API_KEY`), which the injected Secret provides.
  // The command array is exec'd without a shell, so "$VAR" would be passed
  // literally rather than expanded - and this keeps the key out of the process
  // command line entirely.
  const argv = buildArgv(spec);
  if (opts.trace !== false) argv.push("--enable-log-requests");

  // A shell is needed to tee into the volume. Safe here only because every
  // token in argv already passed the allowlist in buildArgv - no whitespace,
  // quotes or separators survive it, so joining cannot change the parse.
  const traceFile = `${TRACE_DIR}/requests-$(date +%Y-%m-%d).jsonl`;
  const command =
    opts.trace === false
      ? argv
      : ["sh", "-c", `mkdir -p ${TRACE_DIR} && exec ${argv.join(" ")} 2>&1 | tee -a ${traceFile}`];

  const gpu = `${spec.hardware.gpu}:${spec.hardware.count}`;
  // Cold start is ~9 minutes even with weights cached, so a short idle window
  // trades a few cents of idle for a very expensive wait. 15 min is the floor
  // at which the tradeoff makes sense.
  const idleMs = (spec.modal?.idle_timeout_s ?? 900) * 1000;
  const maxMs = (spec.modal?.max_lifetime_s ?? 4 * 3600) * 1000;

  onUpdate?.(`starting ${gpu} sandbox...`);
  const createdAt = new Date().toISOString();
  const sb = await c.sandboxes.create(app, image, {
    gpu,
    command,
    encryptedPorts: [8000],
    volumes: {
      "/root/.cache/huggingface": hfCache,
      "/root/.cache/vllm": vllmCache,
      [TRACE_DIR]: traceVol,
    },
    secrets: [secret],
    env: buildEnv(spec),
    idleTimeoutMs: idleMs,
    timeoutMs: maxMs,
  });

  // Tag it so /yaeger-status can find what this plugin started, without us
  // having to keep a local registry in sync.
  // Tagged before the ready-wait below, so a boot interrupted mid-wait - the
  // session quitting, a crash, a closed laptop - is still discoverable and
  // reapable. That window is exactly how an orphaned H100 runs to max_lifetime_s.
  await sb
    .setTags({
      yaeger: "1",
      model: spec.model.repo.slice(0, 60),
      reap: opts.reap ?? "exit",
    })
    .catch(() => {});

  const tunnels = await sb.tunnels();
  const url = tunnels[8000]?.url;
  if (!url) {
    await sb.terminate();
    throw new Error("sandbox started but no tunnel was returned for port 8000");
  }

  onUpdate?.("waiting for the model to load (first run pulls weights)...");
  const deadline = Date.now() + readyTimeoutMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 10_000));
    try {
      const res = await fetch(`${url}/v1/models`, { signal: AbortSignal.timeout(8000) });
      if (res.ok || res.status === 401) {
        // 401 means vLLM is up and enforcing its API key - that is "ready".
        return {
          sandboxId: sb.sandboxId,
          url,
          apiKey,
          createdAt,
          gpu: spec.hardware.gpu,
          count: spec.hardware.count,
          usdPerHour: spec.hardware.est_usd_per_hour ?? 0,
        };
      }
    } catch {
      /* still booting */
    }
    const left = Math.round((deadline - Date.now()) / 60_000);
    if (left % 3 === 0) onUpdate?.(`still loading... (${left} min before giving up)`);
  }

  const logs = await tailLogs(sb).catch(() => "");
  await sb.terminate().catch(() => {});
  throw new Error(`model did not become ready in time.\n\n${logs.slice(-2000)}`);
}

/** Best-effort log capture, used to feed the KB diagnoser when a launch fails. */
export async function tailLogs(sb: any): Promise<string> {
  try {
    const p = await sb.exec(["sh", "-c", "tail -n 200 /proc/1/fd/2 2>/dev/null || true"]);
    return await p.stdout.readText();
  } catch {
    return "";
  }
}

export async function stop(sandboxId: string): Promise<void> {
  const c = client();
  const sb = await c.sandboxes.fromId(sandboxId);
  await sb.terminate();
}


/** What is alive right now, i.e. what is costing money. */
export async function listRunning(): Promise<RunningSandbox[]> {
  const c = client();
  const out: RunningSandbox[] = [];
  try {
    for await (const sb of c.sandboxes.list({ tags: { yaeger: "1" } })) {
      const tags = await (sb as any).getTags().catch(() => ({}));
      out.push({
        sandboxId: (sb as any).sandboxId,
        model: tags.model ?? "?",
        reap: parseReapTag(tags.reap),
      });
    }
  } catch {
    /* listing is advisory; never fail status over it */
  }
  return out;
}


/**
 * Read captured traces back.
 *
 * The TS SDK cannot read a Volume directly - VolumeService exposes only
 * fromName/ephemeral/delete - so the file is read by exec'ing into a sandbox
 * that mounts it. When the GPU endpoint has already idled out, a tiny CPU-only
 * sandbox is started instead, which costs cents rather than H100 minutes.
 */
export async function readTraces(
  opts: { sandboxId?: string; day?: string; limit?: number } = {},
): Promise<string[]> {
  const c = client();
  const day = opts.day ?? new Date().toISOString().slice(0, 10);
  const file = `${TRACE_DIR}/requests-${day}.jsonl`;
  const limit = opts.limit ?? 2000;

  const readFrom = async (sb: any): Promise<string[]> => {
    const p = await sb.exec(["sh", "-c", `tail -n ${limit} ${file} 2>/dev/null || true`]);
    const out = await p.stdout.readText();
    return out.split("\n").filter((l: string) => l.trim().length > 0);
  };

  if (opts.sandboxId) {
    try {
      return await readFrom(await c.sandboxes.fromId(opts.sandboxId));
    } catch {
      /* endpoint gone; fall through to a reader sandbox */
    }
  }

  // No GPU: this only reads a file.
  const app = await c.apps.fromName("yaeger-trace-reader", { createIfMissing: true });
  const image = await c.images.fromRegistry("alpine:3.20");
  const traceVol = await c.volumes.fromName(TRACE_VOLUME, { createIfMissing: true });
  const sb = await c.sandboxes.create(app, image, {
    volumes: { [TRACE_DIR]: traceVol },
    timeoutMs: 2 * 60_000,
    command: ["sleep", "120"],
  });
  try {
    return await readFrom(sb);
  } finally {
    await sb.terminate().catch(() => {});
  }
}
