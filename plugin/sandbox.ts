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

import { buildArgv, type HarnessSpec } from "./render.ts";

const HF_CACHE_VOLUME = "huggingface-cache";
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
  opts: { onUpdate?: (s: string) => void; readyTimeoutMs?: number; trace?: boolean } = {},
): Promise<Launched> {
  const { onUpdate, readyTimeoutMs = 15 * 60_000 } = opts;
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
  if (spec.engine.clear_entrypoint !== false) {
    // vllm/vllm-openai ships ENTRYPOINT ["vllm","serve"]; an explicit command
    // gets appended to it and the process fails with "unrecognized arguments".
    image = image.dockerfileCommands(["ENTRYPOINT []"]);
  }

  const hfCache = await c.volumes.fromName(HF_CACHE_VOLUME, { createIfMissing: true });
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
    volumes: { "/root/.cache/huggingface": hfCache, [TRACE_DIR]: traceVol },
    secrets: [secret],
    env: { HF_XET_HIGH_PERFORMANCE: "1" },
    idleTimeoutMs: idleMs,
    timeoutMs: maxMs,
  });

  // Tag it so /yaeger-status can find what this plugin started, without us
  // having to keep a local registry in sync.
  await sb.setTags({ yaeger: "1", model: spec.model.repo.slice(0, 60) }).catch(() => {});

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

export interface RunningSandbox {
  sandboxId: string;
  model: string;
}

/** What is alive right now, i.e. what is costing money. */
export async function listRunning(): Promise<RunningSandbox[]> {
  const c = client();
  const out: RunningSandbox[] = [];
  try {
    for await (const sb of c.sandboxes.list({ tags: { yaeger: "1" } })) {
      const tags = await (sb as any).getTags().catch(() => ({}));
      out.push({ sandboxId: (sb as any).sandboxId, model: tags.model ?? "?" });
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
