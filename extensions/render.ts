/**
 * Harness spec -> Modal deployment file.
 *
 * The service sends data, never code. This renderer is the only thing that
 * produces Python, it lives on the user's machine, and it is auditable. That
 * split is deliberate: a compromised service can hand you a bad *flag*, not
 * arbitrary code to execute.
 *
 * Every value that reaches the vLLM command line goes through argv as a
 * separate token. Nothing is ever joined into a shell string.
 */

export interface HarnessSpec {
  spec_version: number;
  tier?: "exact" | "near" | "cold";
  provenance?: {
    harness_id?: string;
    verified?: boolean;
    boot_count?: number;
    derived_from?: string;
    kb_entries_applied?: string[];
  };
  model: {
    repo: string;
    revision?: string;
    architecture?: string | null;
    family?: string | null;
    quantization?: string;
    gated?: boolean;
  };
  hardware: {
    gpu: string;
    count: number;
    est_weights_gb?: number | null;
    est_usd_per_hour?: number;
  };
  engine: {
    name: string;
    version: string;
    cuda_image?: string;
    /** Runtime image with the engine already installed. Required by the sandbox path. */
    image?: string;
    /** vllm/vllm-openai ships an ENTRYPOINT that double-invokes an explicit command. */
    clear_entrypoint?: boolean;
    python_version?: string;
    extra_packages?: string[];
    env?: Record<string, string>;
  };
  serving: {
    served_model_name?: string;
    max_model_len: number;
    max_num_seqs?: number;
    kv_cache_dtype?: string;
    enable_prefix_caching?: boolean;
    tool_call_parser?: string;
    reasoning_parser?: string;
    enable_auto_tool_choice?: boolean;
    extra_flags?: string[];
  };
  modal?: {
    app_name?: string;
    scaledown_window_s?: number;
    startup_timeout_s?: number;
    max_containers?: number;
    target_concurrency?: number;
    /** Sandbox path: self-terminate after this much inactivity. */
    idle_timeout_s?: number;
    /** Sandbox path: hard spend ceiling, enforced even if idle detection fails. */
    max_lifetime_s?: number;
  };
  client?: {
    context_window?: number;
    max_tokens?: number;
    reasoning?: boolean;
    compat?: Record<string, unknown>;
  };
  warnings?: string[];
}

/** Python string literal. JSON escaping is a safe subset for our inputs. */
function py(s: string): string {
  return JSON.stringify(String(s));
}

/** Reject anything that could break out of an argv token or a filename. */
const SAFE = /^[A-Za-z0-9._\-\/:=+@]*$/;

/**
 * A package name is not an argv token, and must not be validated like one.
 *
 * SAFE deliberately permits leading dashes and slashes because argv tokens need
 * them. Reusing it for `extra_packages` would let a spec smuggle pip flags -
 * `--index-url=http://attacker/` redirects the resolver and runs attacker code
 * inside the image build. Specs come from the harness service, so this is a
 * trust boundary: a package name must look like PEP 508 and nothing else.
 * Anchored on [A-Za-z0-9], so a leading dash cannot parse at all.
 */
const PEP508 =
  /^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?((==|>=|<=|~=|!=|>|<)[A-Za-z0-9._*+-]+)?$/;

export function assertPackageName(value: string): string {
  if (!PEP508.test(value)) {
    throw new Error(
      `refusing extra_packages entry ${JSON.stringify(value)}: not a package name. ` +
        `Flags, paths and URLs are not accepted here. This spec may be malformed or hostile.`,
    );
  }
  return value;
}

export function assertSafe(label: string, value: string): string {
  if (!SAFE.test(value)) {
    throw new Error(
      `refusing to render ${label}: ${JSON.stringify(value)} contains characters ` +
        `outside the allowed set. This spec may be malformed or hostile.`,
    );
  }
  return value;
}

export function appName(spec: HarnessSpec): string {
  const raw =
    spec.modal?.app_name ??
    spec.model.repo.split("/").pop()!.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  return assertSafe("modal.app_name", raw).replace(/^-+|-+$/g, "").slice(0, 63);
}

export function servedModelName(spec: HarnessSpec): string {
  return assertSafe(
    "serving.served_model_name",
    spec.serving.served_model_name ?? appName(spec),
  );
}

/** Build the vllm serve argv, one token per element. */
export function buildArgv(spec: HarnessSpec): string[] {
  const s = spec.serving;
  const argv: string[] = [
    "vllm",
    "serve",
    assertSafe("model.repo", spec.model.repo),
    "--revision",
    assertSafe("model.revision", spec.model.revision ?? "main"),
    "--served-model-name",
    servedModelName(spec),
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--max-model-len",
    String(Math.floor(s.max_model_len)),
  ];

  // Multi-GPU needs tensor parallelism explicitly: vLLM defaults to TP=1 and
  // will simply fail to fit the model rather than sharding it. Every harness
  // with hardware.count > 1 is unservable without this.
  const gpuCount = Math.max(1, Math.floor(spec.hardware.count ?? 1));
  if (gpuCount > 1) argv.push("--tensor-parallel-size", String(gpuCount));

  if (s.max_num_seqs != null) argv.push("--max-num-seqs", String(Math.floor(s.max_num_seqs)));
  if (s.kv_cache_dtype && s.kv_cache_dtype !== "auto") {
    argv.push("--kv-cache-dtype", assertSafe("kv_cache_dtype", s.kv_cache_dtype));
  }
  if (s.enable_prefix_caching !== false) argv.push("--enable-prefix-caching");
  if (s.enable_auto_tool_choice !== false && s.tool_call_parser) {
    argv.push("--enable-auto-tool-choice");
    argv.push("--tool-call-parser", assertSafe("tool_call_parser", s.tool_call_parser));
  }
  if (s.reasoning_parser) {
    argv.push("--reasoning-parser", assertSafe("reasoning_parser", s.reasoning_parser));
  }
  for (const flag of s.extra_flags ?? []) {
    argv.push(assertSafe("extra_flags", flag));
  }
  return argv;
}

export function render(spec: HarnessSpec): string {
  const app = appName(spec);
  const gpu = assertSafe("hardware.gpu", spec.hardware.gpu);
  const count = Math.max(1, Math.min(8, Math.floor(spec.hardware.count)));
  const image = assertSafe(
    "engine.cuda_image",
    spec.engine.cuda_image ?? "nvidia/cuda:12.9.0-devel-ubuntu22.04",
  );
  const pyver = assertSafe("engine.python_version", spec.engine.python_version ?? "3.12");
  const engineName = assertSafe("engine.name", spec.engine.name);
  const engineVersion = assertSafe("engine.version", spec.engine.version);

  const pkgs = [`${engineName}==${engineVersion}`, ...(spec.engine.extra_packages ?? [])].map(
    (p) => py(assertSafe("extra_packages", p)),
  );

  const env = Object.entries(spec.engine.env ?? {})
    .map(([k, v]) => `        ${py(assertSafe("env key", k))}: ${py(v)},`)
    .join("\n");

  const argvLines = buildArgv(spec)
    .map((tok) => `            ${py(tok)},`)
    .join("\n");

  const m = spec.modal ?? {};
  const provenance = spec.provenance ?? {};
  const banner = [
    `# Generated by yaeger-pi from harness spec v${spec.spec_version}.`,
    `# model      ${spec.model.repo}`,
    `# hardware   ${count}x${gpu}` +
      (spec.hardware.est_usd_per_hour ? `  (~$${spec.hardware.est_usd_per_hour}/hr)` : ""),
    `# tier       ${spec.tier ?? "unknown"}` +
      (provenance.verified ? " (verified - this spec has booted before)" : " (UNVERIFIED)"),
    provenance.kb_entries_applied?.length
      ? `# kb applied ${provenance.kb_entries_applied.join(", ")}`
      : null,
    provenance.harness_id ? `# harness    ${provenance.harness_id}` : null,
    ...(spec.warnings ?? []).map((w) => `# warning    ${w}`),
  ]
    .filter(Boolean)
    .join("\n");

  return `${banner}
#
# Deploy:  modal deploy ${app}.py
# Stop:    modal app stop ${app}
#
# Billing is per second while a container is alive. It scales to zero after
# ${m.scaledown_window_s ?? 300}s idle, and a container only boots on the first request.

import modal

MODEL_NAME = ${py(spec.model.repo)}
MODEL_REVISION = ${py(spec.model.revision ?? "main")}
VLLM_PORT = 8000

image = (
    modal.Image.from_registry(${py(image)}, add_python=${py(pyver)})
    .entrypoint([])
    .uv_pip_install(${pkgs.join(", ")})
    .env({
${env || "        # no extra environment"}
    })
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(${py(app)})


@app.server(
    image=image,
    gpu=${py(`${gpu}:${count}`)},
    scaledown_window=${m.scaledown_window_s ?? 300},
    startup_timeout=${m.startup_timeout_s ?? 1200},
    target_concurrency=${m.target_concurrency ?? 8},
    max_containers=${m.max_containers ?? 1},
    port=VLLM_PORT,
    unauthenticated=True,
    secrets=[modal.Secret.from_name("vllm-api-key")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
class Server:
    # @app.server() requires a class with @modal.enter(); the function form in
    # some published examples fails on modal >= 1.5.
    @modal.enter()
    def start(self):
        import os
        import subprocess

        cmd = [
${argvLines}
            "--api-key",
            os.environ["VLLM_API_KEY"],
        ]
        # No shell: the API key must never be interpolated into a command string.
        subprocess.Popen(cmd)
`;
}
