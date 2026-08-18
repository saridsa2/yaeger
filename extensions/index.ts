/**
 * yaeger - stand up any model on your own Modal account, from inside pi.
 *
 * The harness comes from yaeger's gated service (verified specs served
 * instantly; unseen architectures generated). The deploy runs against YOUR
 * Modal token, on YOUR account, billed to YOU. yaeger never touches your
 * infrastructure - it only authors the file.
 *
 * Commands: /yaeger-login  /yaeger-setup  /yaeger-status  /yaeger-stop
 * Tools:    yaeger_setup_model, yaeger_list_models
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execFile, execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";
import { Type } from "typebox";

import {
  applyPatch,
  catalog,
  fetchModelMetadata,
  clearSession,
  loadSession,
  reportDeploy,
  requestLoginEmail,
  resolveLoginToken,
  resolveSpec,
  whoami,
  createTeam,
  myTeams,
  inviteMember,
  getTeamEndpoint,
  publishTeamEndpoint,
  stopTeamEndpoint,
  setTeamContext,
  teamEconomics,
  teamInsights,
  type CatalogEntry,
} from "./client.ts";
import { buildArgv, servedModelName, type HarnessSpec } from "./render.ts";
import {
  launch,
  listRunning,
  modalConfigured,
  readTraces,
  sandboxesToReap,
  stop,
  type Launched,
  type ReapPolicy,
} from "./sandbox.ts";
import { renderEconomics, renderInsights } from "./teamview.ts";

const execFileAsync = promisify(execFile);

const MODELS_JSON = join(homedir(), ".pi", "agent", "models.json");

// modal installs to ~/.local/bin via `uv tool install`, often absent from PATH.
const env = { ...process.env, PATH: `${homedir()}/.local/bin:${process.env.PATH ?? ""}` };

function text(s: string) {
  return { content: [{ type: "text" as const, text: s }], details: {} };
}

/** Default ceiling so a sizing mistake cannot quietly rent 8 GPUs. */
const DEFAULT_MAX_USD_PER_HOUR = Number(process.env.YAEGERPI_MAX_USD_PER_HOUR ?? "8");


function specSummary(spec: HarnessSpec): string {
  const hw = spec.hardware;
  const p = spec.provenance ?? {};
  const lines = [
    `model     ${spec.model.repo}`,
    `hardware  ${hw.count}x${hw.gpu}  ~$${hw.est_usd_per_hour ?? "?"}/hr while running`,
    `tier      ${spec.tier}${
      spec.tier === "exact"
        ? `  (verified - booted ${p.boot_count ?? "?"}x before)`
        : spec.tier === "near"
          ? "  (adapted from a verified sibling)"
          : "  (never verified - expect to iterate)"
    }`,
    `context   ${spec.serving.max_model_len.toLocaleString("en-US")} tokens`,
  ];
  if (p.kb_entries_applied?.length) lines.push(`fixes     ${p.kb_entries_applied.join(", ")}`);
  for (const w of spec.warnings ?? []) lines.push(`warning   ${w}`);
  return lines.join("\n");
}

/** Register the deployed endpoint as a pi provider so /model can select it. */
function registerProvider(spec: HarnessSpec, endpoint: string, apiKey: string): void {
  let cfg: any = { providers: {} };
  if (existsSync(MODELS_JSON)) {
    try {
      cfg = JSON.parse(readFileSync(MODELS_JSON, "utf8"));
    } catch {
      throw new Error(`${MODELS_JSON} is not valid JSON - refusing to overwrite it.`);
    }
  }
  cfg.providers ??= {};

  const id = servedModelName(spec);
  cfg.providers["yaeger-modal"] = {
    name: "yaeger (your Modal)",
    baseUrl: `${endpoint}/v1`,
    api: "openai-completions",
    // Read per-invocation from the keychain (macOS) or a 0600 file, never
    // stored in plaintext inside models.json.
    apiKey:
      process.platform === "darwin"
        ? "!security find-generic-password -ws yaeger-modal"
        : `!cat ${join(homedir(), ".pi", "agent", "yaeger-modal.key")}`,
    models: [
      ...(cfg.providers["yaeger-modal"]?.models ?? []).filter((m: any) => m.id !== id),
      {
        id,
        name: `${spec.model.repo} (Modal ${spec.hardware.count}x${spec.hardware.gpu})`,
        reasoning: spec.client?.reasoning ?? Boolean(spec.serving.reasoning_parser),
        input: ["text"],
        contextWindow: spec.client?.context_window ?? spec.serving.max_model_len,
        maxTokens: spec.client?.max_tokens ?? 32768,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        compat: spec.client?.compat ?? {
          supportsDeveloperRole: false,
          maxTokensField: "max_tokens",
        },
      },
    ],
  };

  mkdirSync(dirname(MODELS_JSON), { recursive: true });
  writeFileSync(MODELS_JSON, JSON.stringify(cfg, null, 2), { mode: 0o600 });
  storeApiKey(apiKey);
}

/** Register a team endpoint for a member, who has no spec - just URL, key, model. */
function registerProviderRaw(model: string, url: string, key: string): void {
  let cfg: any = { providers: {} };
  if (existsSync(MODELS_JSON)) {
    try {
      cfg = JSON.parse(readFileSync(MODELS_JSON, "utf8"));
    } catch {
      throw new Error(`${MODELS_JSON} is not valid JSON - refusing to overwrite it.`);
    }
  }
  cfg.providers ??= {};
  const id = model.split("/").pop()!.toLowerCase();
  cfg.providers["yaeger-team"] = {
    name: "yaeger (team endpoint)",
    baseUrl: `${url}/v1`,
    api: "openai-completions",
    apiKey:
      process.platform === "darwin"
        ? "!security find-generic-password -ws yaeger-modal"
        : `!cat ${join(homedir(), ".pi", "agent", "yaeger-modal.key")}`,
    models: [
      {
        id,
        name: `${model} (team)`,
        reasoning: true,
        input: ["text"],
        contextWindow: 131072,
        maxTokens: 32768,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        compat: { supportsDeveloperRole: false, maxTokensField: "max_tokens" },
      },
    ],
  };
  mkdirSync(dirname(MODELS_JSON), { recursive: true });
  writeFileSync(MODELS_JSON, JSON.stringify(cfg, null, 2), { mode: 0o600 });
  storeApiKey(key);
}

/** Keep the endpoint key out of models.json: keychain on macOS, 0600 file elsewhere. */
function storeApiKey(key: string): void {
  if (process.platform === "darwin") {
    try {
      execFileSync("security", [
        "add-generic-password", "-a", process.env.USER ?? "yaeger",
        "-s", "yaeger-modal", "-w", key, "-U",
      ]);
      return;
    } catch { /* fall through to the file */ }
  }
  const f = join(homedir(), ".pi", "agent", "yaeger-modal.key");
  mkdirSync(dirname(f), { recursive: true });
  writeFileSync(f, key, { mode: 0o600 });
}

/**
 * Launch, and if it fails to come up, ask the service what the error means.
 * One repair attempt: a second failure is a real unknown and belongs with a
 * human rather than burning more GPU minutes guessing.
 */
async function launchWithRepair(
  spec: HarnessSpec,
  onUpdate?: (s: string) => void,
  reap: ReapPolicy = "exit",
): Promise<{ spec: HarnessSpec; launched: Launched; repaired: string | null }> {
  let current = spec;
  let repaired: string | null = null;

  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const launched = await launch(current, { onUpdate, reap });
      return { spec: current, launched, repaired };
    } catch (err) {
      const raw = String((err as Error).message ?? err);
      const { guidance } = await reportDeploy({
        model_repo: current.model.repo,
        harness_id: current.provenance?.harness_id,
        tier: current.tier,
        outcome: "failed",
        phase: "launch",
        error_excerpt: raw.slice(-4000),
      });

      if (attempt === 2 || !guidance?.fix?.spec_patch) {
        const known = guidance ? `\n\nKnown issue: ${guidance.title}\n${guidance.cause}` : "";
        throw new Error(`Launch failed.${known}\n\n${raw.slice(-1200)}`);
      }

      onUpdate?.(`known issue: ${guidance.kb_entry} - applying fix and retrying`);
      current = applyPatch(current, guidance.fix.spec_patch);
      repaired = guidance.kb_entry;
    }
  }
  throw new Error("unreachable");
}

export default function (pi: ExtensionAPI) {
  // ------------------------------------------------------------- tools

  pi.registerTool({
    name: "yaeger_list_models",
    label: "yaeger: catalog",
    description:
      "List models yaeger can deploy to the user's Modal account, with GPU, hourly cost, and whether the harness is verified.",
    promptSnippet: "List models yaeger can deploy, with cost and verification status",
    parameters: Type.Object({}),
    async execute() {
      const models = await catalog();
      if (!models.length) return text("Catalog is empty.");
      const rows = models
        .map(
          (m: CatalogEntry) =>
            `${m.verified ? "verified" : "unproven"}  ${m.model_repo}  ` +
            `${m.gpu_count}x${m.gpu}  $${m.est_usd_per_hour}/hr` +
            (m.boot_count ? `  (${m.boot_count} boots)` : ""),
        )
        .join("\n");
      return text(`Models yaeger can deploy:\n\n${rows}`);
    },
  });

  pi.registerTool({
    name: "yaeger_setup_model",
    label: "yaeger: set up model",
    description:
      "Deploy a model to the user's own Modal account and register it with pi. Starts GPU billing on their account once a container boots.",
    promptSnippet: "Deploy a model to the user's Modal account and register it in pi",
    promptGuidelines: [
      "Use yaeger_setup_model only when the user explicitly asks to deploy or set up a model, since it spends money on their Modal account.",
      "Always show the user the hourly cost yaeger_setup_model reports before treating the deployment as agreed.",
    ],
    parameters: Type.Object({
      model_repo: Type.String({ description: "Hugging Face repo id, e.g. Qwen/Qwen3.8-27B-FP8" }),
      quantization: Type.Optional(Type.String({ description: "none | fp8 | nvfp4 | awq | gptq" })),
      param_count_b: Type.Optional(Type.Number({ description: "Params in billions, if known" })),
      gpu: Type.Optional(Type.String({ description: "Force a GPU class, e.g. H100" })),
      max_usd_per_hour: Type.Optional(Type.Number({ description: "Refuse configs above this" })),
      dry_run: Type.Optional(
        Type.Boolean({ description: "Render the harness and show the plan without deploying" }),
      ),
    }),
    async execute(_id, params, _signal, onUpdate) {
      if (!loadSession()) return text("Not signed in to yaeger. Run /yaeger-login first.");

      const blocked = params.dry_run ? null : await modalConfigured();
      if (blocked) return text(blocked);

      onUpdate?.("reading model metadata from Hugging Face...");
      const meta = await fetchModelMetadata(params.model_repo);

      onUpdate?.("resolving harness...");
      const spec = await resolveSpec({
        model_repo: params.model_repo,
        architecture: meta.architecture,
        quantization: params.quantization ?? meta.quantization ?? "none",
        param_count_b: params.param_count_b ?? meta.param_count_b,
        gpu: params.gpu,
        max_usd_per_hour: params.max_usd_per_hour ?? DEFAULT_MAX_USD_PER_HOUR,
      });

      if (meta.gated) {
        return text(
          `${params.model_repo} is a gated repo on Hugging Face. Accept its licence and ` +
          `add an HF token to your Modal secrets before deploying.`);
      }

      if (params.dry_run) {
        return text(
          `${specSummary(spec)}\n\nimage     ${spec.engine.image ?? "(none - cannot launch)"}\n` +
            `command   ${JSON.stringify(buildArgv(spec))}`,
        );
      }

      const { spec: finalSpec, launched, repaired } = await launchWithRepair(spec, onUpdate);

      await reportDeploy({
        model_repo: finalSpec.model.repo,
        harness_id: finalSpec.provenance?.harness_id,
        tier: finalSpec.tier,
        outcome: "booted",
      });

      let registered = "";
      try {
        registerProvider(finalSpec, launched.url, launched.apiKey);
        registered = `Registered in pi as yaeger-modal/${servedModelName(finalSpec)} - pick it with /model.`;
      } catch (e) {
        registered = `Could not update models.json: ${(e as Error).message}`;
      }

      return text(
        [
          `${finalSpec.model.repo} is running on your Modal account.`,
          repaired ? `Hit a known issue (${repaired}) and applied the fix automatically.` : null,
          "",
          specSummary(finalSpec),
          "",
          `endpoint  ${launched.url}`,
          `sandbox   ${launched.sandboxId}`,
          registered,
          "",
          `Billing ~$${launched.usdPerHour}/hr while it runs. It stops itself after ` +
            `${(finalSpec.modal?.idle_timeout_s ?? 300) / 60} min idle, and hard-stops after ` +
            `${(finalSpec.modal?.max_lifetime_s ?? 14400) / 3600} h. Stop now with /yaeger-stop.`,
        ]
          .filter((l) => l !== null)
          .join("\n"),
      );
    },
  });

  // ---------------------------------------------------------- commands

  pi.registerCommand("yaeger-login", {
    description: "Sign in to yaeger (emails you a sign-in link)",
    handler: async (args, ctx) => {
      try {
        const email =
          args?.trim() || (await ctx.ui.input?.("Email for your yaeger account:", ""));
        if (!email) return ctx.ui.notify("Cancelled.", "info");

        const mins = await requestLoginEmail(email);
        ctx.ui.notify(`Sent a sign-in code to ${email}. It expires in ${mins} minutes.`, "info");

        const token = await ctx.ui.input?.("Paste the code from the email:", "");
        if (!token) return ctx.ui.notify("Cancelled - no code entered.", "info");

        const session = await resolveLoginToken(token);
        // Entitlement is read from the server, never declared by the user.
        let capability = "";
        try {
          const me = await whoami();
          capability = me.can_generate
            ? `\nYou can deploy anything from the catalog, and build harnesses for models not in it (${me.daily_generation_limit}/day).`
            : "\nYou can deploy any model in the catalog. Building harnesses for new models is invite-only - ask and your requests are recorded.";
        } catch {
          /* signed in is still signed in even if this lookup fails */
        }
        ctx.ui.notify(`Signed in as ${session.email}.${capability}`, "info");
      } catch (e) {
        ctx.ui.notify(`Sign-in failed: ${(e as Error).message}`, "error");
      }
    },
  });

  pi.registerCommand("yaeger-logout", {
    description: "Forget the stored yaeger session",
    handler: async (_args, ctx) => {
      clearSession();
      ctx.ui.notify("Signed out.", "info");
    },
  });

  pi.registerCommand("yaeger-setup", {
    description: "Pick a model and deploy it to your Modal account",
    handler: async (args, ctx) => {
      try {
        if (!loadSession()) {
          return ctx.ui.notify("Not signed in. Run /yaeger-login first.", "error");
        }
        const blocked = await modalConfigured();
        if (blocked) return ctx.ui.notify(blocked, "error");

        let repo = args?.trim();
        if (!repo) {
          const [models, me] = await Promise.all([
            catalog(),
            whoami().catch(() => ({ can_generate: false }) as any),
          ]);
          if (!models.length && !me.can_generate) {
            return ctx.ui.notify("Catalog is empty.", "error");
          }
          // ctx.ui.select takes plain strings and returns the chosen string,
          // so build labels and map back rather than passing objects.
          const labels = models.map(
            (m) =>
              `${m.model_repo}  -  ${m.gpu_count}x${m.gpu}  $${m.est_usd_per_hour}/hr` +
              (m.verified ? "  [verified]" : "  [unproven]"),
          );
          // Invited accounts get one extra option: anything on the Hub.
          const OTHER = "Other model from Hugging Face...  [builds a new harness]";
          if (me.can_generate) labels.push(OTHER);

          const choice = await ctx.ui.select?.("Which model do you want to run?", labels);
          if (!choice) return ctx.ui.notify("Cancelled.", "info");

          if (String(choice) === OTHER) {
            const typed = await ctx.ui.input?.("Hugging Face repo id (e.g. Qwen/Qwen3.8-27B):", "");
            if (!typed) return ctx.ui.notify("Cancelled.", "info");
            repo = typed.trim();
          } else {
            const picked = models[labels.indexOf(String(choice))];
            if (!picked) return ctx.ui.notify("Could not match that selection.", "error");
            repo = picked.model_repo;
          }
        }

        ctx.ui.notify(`Resolving harness for ${repo}...`, "info");
        const meta = await fetchModelMetadata(repo!);
        const spec = await resolveSpec({
          model_repo: repo!,
          architecture: meta.architecture,
          quantization: meta.quantization ?? "none",
          param_count_b: meta.param_count_b,
          max_usd_per_hour: DEFAULT_MAX_USD_PER_HOUR,
        });

        const ok = await ctx.ui.confirm?.(
          "Deploy to your Modal account?",
          `${specSummary(spec)}\n\nThis bills to your Modal account while running.`,
        );
        if (!ok) return ctx.ui.notify("Cancelled - nothing deployed.", "info");

        const { spec: finalSpec, launched, repaired } = await launchWithRepair(spec, (m) =>
          ctx.ui.notify(m, "info"),
        );
        await reportDeploy({
          model_repo: finalSpec.model.repo,
          harness_id: finalSpec.provenance?.harness_id,
          tier: finalSpec.tier,
          outcome: "booted",
        });
        registerProvider(finalSpec, launched.url, launched.apiKey);
        ctx.ui.notify(
          `Running at ${launched.url}` +
            (repaired ? ` (auto-fixed ${repaired})` : "") +
            `. Select it with /model. ~$${launched.usdPerHour}/hr - /yaeger-stop when done,` +
            ` or just quit pi and it stops itself.`,
          "info",
        );
      } catch (e) {
        ctx.ui.notify(`Setup failed: ${(e as Error).message}`, "error");
      }
    },
  });

  // ---------------------------------------------------------- teams

  pi.registerCommand("yaeger-team", {
    description: "List your teams, or create one: /yaeger-team new <name>",
    handler: async (args, ctx) => {
      try {
        const a = (args ?? "").trim();
        if (a.startsWith("new ")) {
          const t = await createTeam(a.slice(4).trim());
          return ctx.ui.notify(
            `Created team "${t.name}" (${t.team_id}).\n` +
              `Invite people with /yaeger-team-invite ${t.team_id} <email>, ` +
              `then start the shared endpoint with /yaeger-team-start ${t.team_id}.`,
            "info",
          );
        }
        const teams = await myTeams();
        if (!teams.length) {
          return ctx.ui.notify('No teams yet. Create one with /yaeger-team new <name>', "info");
        }
        ctx.ui.notify(
          teams
            .map(
              (t) =>
                `${t.name}  ${t.team_id}  ${t.role}  ${t.members} member(s)  ` +
                (t.endpoint_live ? `live: ${t.model}` : "endpoint stopped"),
            )
            .join("\n"),
          "info",
        );
      } catch (e) {
        ctx.ui.notify(`${(e as Error).message}`, "error");
      }
    },
  });

  pi.registerCommand("yaeger-team-invite", {
    description: "Invite someone to a team: /yaeger-team-invite <team-id> <email>",
    handler: async (args, ctx) => {
      const [teamId, email] = (args ?? "").trim().split(/\s+/);
      if (!teamId || !email) {
        return ctx.ui.notify("Usage: /yaeger-team-invite <team-id> <email>", "error");
      }
      try {
        await inviteMember(teamId, email);
        ctx.ui.notify(
          `Invited ${email}. They sign in with /yaeger-login and join with ` +
            `/yaeger-team-use ${teamId} - they do NOT need a Modal account.`,
          "info",
        );
      } catch (e) {
        ctx.ui.notify(`${(e as Error).message}`, "error");
      }
    },
  });

  pi.registerCommand("yaeger-team-start", {
    description: "Owner: launch the shared endpoint for a team",
    handler: async (args, ctx) => {
      const teamId = (args ?? "").trim();
      if (!teamId) return ctx.ui.notify("Usage: /yaeger-team-start <team-id>", "error");
      try {
        const blocked = await modalConfigured();
        if (blocked) return ctx.ui.notify(blocked, "error");

        const models = await catalog();
        const labels = models.map(
          (m) => `${m.model_repo}  ${m.gpu_count}x${m.gpu}  $${m.est_usd_per_hour}/hr` +
                 (m.verified ? "  [verified]" : "  [unproven]"),
        );
        const choice = await ctx.ui.select?.("Which model should the team share?", labels);
        if (!choice) return ctx.ui.notify("Cancelled.", "info");
        const picked = models[labels.indexOf(String(choice))];
        if (!picked) return ctx.ui.notify("Could not match that selection.", "error");

        const spec = await resolveSpec({ model_repo: picked.model_repo, max_usd_per_hour: DEFAULT_MAX_USD_PER_HOUR });
        const ok = await ctx.ui.confirm?.(
          "Start the shared team endpoint?",
          `${specSummary(spec)}\n\nThis bills to YOUR Modal account. Members use it for free.\n` +
            `Requests are logged to your own Modal volume for the contributor view.`,
        );
        if (!ok) return ctx.ui.notify("Cancelled.", "info");

        // Asked here, not on the way out: pi tears the TUI down before
        // session_shutdown fires, so a dialog at exit has nothing to draw on.
        // Default is to leave a shared endpoint up - the owner quitting is not
        // the team finishing, and members may be mid-request.
        let reap: ReapPolicy = "keep";
        if (ctx.hasUI) {
          const stopOnExit = await ctx.ui.confirm?.(
            "Stop this endpoint when you quit pi?",
            "No - leave it up for the team (members keep working; you keep paying).\n" +
              "Yes - stop it when you quit (halts billing; members lose the endpoint).",
          );
          reap = stopOnExit ? "exit" : "keep";
        }

        const { spec: finalSpec, launched } = await launchWithRepair(
          spec,
          (m) => ctx.ui.notify(m, "info"),
          reap,
        );
        await publishTeamEndpoint(teamId, {
          url: launched.url,
          key: launched.apiKey,
          model: finalSpec.model.repo,
          gpu: launched.gpu,
          gpu_count: launched.count,
          usd_per_hour: launched.usdPerHour,
          sandbox_id: launched.sandboxId,
          started_at: launched.createdAt,
        });
        registerProvider(finalSpec, launched.url, launched.apiKey);
        ctx.ui.notify(
          `Team endpoint live: ${launched.url}\n` +
            `Members can now run /yaeger-team-use ${teamId}. ~$${launched.usdPerHour}/hr while up.\n` +
            (reap === "exit"
              ? "Stops automatically when you quit pi."
              : "Stays up after you quit - /yaeger-team-stop to halt billing."),
          "info",
        );
      } catch (e) {
        ctx.ui.notify(`${(e as Error).message}`, "error");
      }
    },
  });

  pi.registerCommand("yaeger-team-use", {
    description: "Join a team's shared endpoint and pick up its context",
    handler: async (args, ctx) => {
      const teamId = (args ?? "").trim();
      if (!teamId) return ctx.ui.notify("Usage: /yaeger-team-use <team-id>", "error");
      try {
        const ep = await getTeamEndpoint(teamId);
        // Members must be told their prompts are recorded, before they use it.
        if (ep.tracing?.enabled) {
          const ok = await ctx.ui.confirm?.(
            "This team records requests",
            `Your prompts to this endpoint are logged to the team's own Modal volume ` +
              `(not to yaeger) and are visible to ${ep.tracing.visible_to}.\n\nContinue?`,
          );
          if (!ok) return ctx.ui.notify("Not joined.", "info");
        }
        registerProviderRaw(ep.model, ep.url, ep.key);
        ctx.ui.notify(
          `Joined. ${ep.model} on ${ep.gpu} - select it with /model.` +
            (ep.context ? `\n\nThe team is working on:\n${ep.context}` : ""),
          "info",
        );
      } catch (e) {
        ctx.ui.notify(`${(e as Error).message}`, "error");
      }
    },
  });

  pi.registerCommand("yaeger-team-context", {
    description: "Set what the team is working on: /yaeger-team-context <team-id> <text>",
    handler: async (args, ctx) => {
      const a = (args ?? "").trim();
      const i = a.indexOf(" ");
      if (i < 0) return ctx.ui.notify("Usage: /yaeger-team-context <team-id> <text>", "error");
      try {
        await setTeamContext(a.slice(0, i), a.slice(i + 1));
        ctx.ui.notify("Updated. New members will pick this up when they join.", "info");
      } catch (e) {
        ctx.ui.notify(`${(e as Error).message}`, "error");
      }
    },
  });

  pi.registerCommand("yaeger-team-insights", {
    description: "Owner only: contributor activity and cost verdict for a team",
    handler: async (args, ctx) => {
      const teamId = (args ?? "").trim();
      if (!teamId) return ctx.ui.notify("Usage: /yaeger-team-insights <team-id>", "error");
      try {
        const [ins, econ, teams] = await Promise.all([
          teamInsights(teamId),
          teamEconomics(teamId, 7),
          myTeams(),
        ]);
        const name = teams.find((t) => t.team_id === teamId)?.name ?? teamId;
        ctx.ui.notify(
          renderInsights(name, ins.contributors, ins.window_days) +
            "\n\n" + renderEconomics(econ),
          "info",
        );
      } catch (e) {
        const m = (e as Error).message ?? "";
        ctx.ui.notify(
          /403/.test(m) ? "Only the team owner can see team stats." : m,
          "error",
        );
      }
    },
  });

  pi.registerCommand("yaeger-team-traces", {
    description: "Owner only: read request traces from the team's own volume",
    handler: async (args, ctx) => {
      const [teamId, day] = (args ?? "").trim().split(/\s+/);
      if (!teamId) return ctx.ui.notify("Usage: /yaeger-team-traces <team-id> [YYYY-MM-DD]", "error");
      try {
        await teamEconomics(teamId, 1); // owner check, cheap
        const lines = await readTraces({ day, limit: 200 });
        ctx.ui.notify(
          lines.length
            ? `${lines.length} trace lines from your Modal volume:\n\n` +
                lines.slice(-25).join("\n").slice(0, 4000)
            : "No traces recorded for that day.",
          "info",
        );
      } catch (e) {
        const m = (e as Error).message ?? "";
        ctx.ui.notify(/403/.test(m) ? "Only the team owner can read traces." : m, "error");
      }
    },
  });

  pi.registerCommand("yaeger-team-stop", {
    description: "Owner: stop the team endpoint and halt billing",
    handler: async (args, ctx) => {
      const teamId = (args ?? "").trim();
      if (!teamId) return ctx.ui.notify("Usage: /yaeger-team-stop <team-id>", "error");
      try {
        // Terminate the GPU FIRST. If the bookkeeping call fails afterwards the
        // worst case is a stale record; the other order leaves a live H100.
        const running = await listRunning();
        for (const r of running) await stop(r.sandboxId).catch(() => {});
        await stopTeamEndpoint(teamId);
        ctx.ui.notify(
          `Stopped ${running.length} sandbox(es) and closed the team session. Billing halted.`,
          "info",
        );
      } catch (e) {
        ctx.ui.notify(
          `${(e as Error).message}\n\nIf a sandbox is still listed by /yaeger-status, ` +
            `stop it directly with /yaeger-stop <sandbox-id>.`,
          "error",
        );
      }
    },
  });

  pi.registerCommand("yaeger-status", {
    description: "Show yaeger session and deployed harnesses",
    handler: async (_args, ctx) => {
      const s = loadSession();
      const lines = [s ? `signed in as ${s.email}` : "not signed in"];
      const running = await listRunning();
      lines.push(
        running.length
          ? `running (billing now):\n` +
              running.map((r) => `  ${r.model}  ${r.sandboxId}`).join("\n")
          : "nothing running - not billing",
      );

      // Reconcile reality against the service's record. They can disagree if a
      // stop half-failed, and a disagreement is money: a sandbox nobody thinks
      // is running still bills.
      try {
        const teams = await myTeams();
        const liveIds = new Set(running.map((r) => r.sandboxId));
        for (const t of teams) {
          if (t.endpoint_live && !running.length) {
            lines.push(
              `WARNING: team "${t.name}" is recorded as live but no sandbox is running. ` +
                `Run /yaeger-team-stop ${t.team_id} to clear it.`,
            );
          }
        }
        if (running.length && !teams.some((t) => t.endpoint_live)) {
          lines.push(
            "WARNING: a sandbox is billing but no team claims it. " +
              "Stop it with /yaeger-stop <sandbox-id>.",
          );
        }
      } catch {
        /* reconciliation is advisory */
      }
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });

  pi.registerCommand("yaeger-stop", {
    description: "Stop running models to halt GPU billing (all, or one by sandbox id)",
    handler: async (args, ctx) => {
      try {
        const id = args?.trim();
        if (id) {
          await stop(id);
          return ctx.ui.notify(`Stopped ${id}. Billing halted.`, "info");
        }
        const running = await listRunning();
        if (!running.length) return ctx.ui.notify("Nothing running.", "info");
        for (const r of running) await stop(r.sandboxId);
        ctx.ui.notify(
          `Stopped ${running.length} sandbox(es). Billing halted.`,
          "info",
        );
      } catch (e) {
        ctx.ui.notify(`Stop failed: ${(e as Error).message}`, "error");
      }
    },
  });

  // ------------------------------------------------- GPU lifecycle safety
  //
  // A GPU bills whether or not anyone is using it, so an endpoint that outlives
  // the session that started it is a silent bill. Two halves cover that:
  //
  //   session_shutdown  a clean quit stops anything tagged reap=exit. Cannot
  //                     prompt: pi stops the TUI before this event fires.
  //   session_start     a sweep at startup catches everything the first half
  //                     cannot - SIGKILL, a crash, a closed laptop, a boot
  //                     interrupted before launch() ever returned. The TUI is
  //                     alive here, so this is where a human gets asked.

  /** Never let a Modal round-trip hold up starting or quitting pi. */
  async function withTimeout<T>(p: Promise<T>, ms: number, fallback: T): Promise<T> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        p,
        new Promise<T>((resolve) => {
          timer = setTimeout(() => resolve(fallback), ms);
        }),
      ]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  pi.on("session_start", async (_event, ctx) => {
    // No Modal credentials means nothing of ours can be running.
    if (await modalConfigured()) return;

    const running = await withTimeout(listRunning(), 5_000, [] as Awaited<
      ReturnType<typeof listRunning>
    >);
    if (!running.length) return;

    const summary = running
      .map((r) => `  ${r.model}  ${r.sandboxId}${r.reap === "keep" ? "  (shared)" : ""}`)
      .join("\n");

    // Print and JSON modes have nothing to ask with, and notify is a no-op there.
    // Stay silent rather than stopping a GPU nobody asked us to stop, or writing
    // stray lines into scripted output. The next interactive session sweeps it.
    if (!ctx.hasUI) return;

    const stopThem = await ctx.ui.confirm?.(
      `${running.length} GPU sandbox(es) from an earlier session are still billing`,
      `${summary}\n\nStop them now?`,
    );
    if (!stopThem) {
      return ctx.ui.notify("Left running. /yaeger-stop when you want them gone.", "info");
    }

    const failed: string[] = [];
    for (const r of running) {
      try {
        await stop(r.sandboxId);
      } catch {
        failed.push(r.sandboxId);
      }
    }
    ctx.ui.notify(
      failed.length
        ? `Stopped ${running.length - failed.length}. Could not stop: ${failed.join(", ")}`
        : `Stopped ${running.length} sandbox(es). Billing halted.`,
      failed.length ? "warning" : "info",
    );
  });

  pi.on("session_shutdown", async (event) => {
    // Only a real quit. A reload, fork or session switch is not the user leaving,
    // and killing a GPU on /compact would be a nasty surprise.
    if (event.reason !== "quit") return;
    if (await modalConfigured()) return;

    const running = await withTimeout(listRunning(), 5_000, [] as Awaited<
      ReturnType<typeof listRunning>
    >);
    const doomed = sandboxesToReap(running, event.reason);
    if (!doomed.length) return;

    // Settle per sandbox so one failure does not hide the rest, and so a slow
    // Modal call cannot hold the quit open indefinitely. `null` marks a stop we
    // never got an answer for - reported as unstopped, because assuming success
    // is how a GPU keeps billing unnoticed.
    const outcomes = await Promise.all(
      doomed.map((d) =>
        withTimeout(
          stop(d.sandboxId).then(
            () => true,
            () => false,
          ),
          10_000,
          null as boolean | null,
        ),
      ),
    );

    const left = doomed.filter((_, i) => outcomes[i] !== true).map((d) => d.sandboxId);
    const stopped = doomed.length - left.length;

    // The TUI is gone by now, so ui.notify cannot render. stdout still works -
    // it is how pi prints its own resume hint - and an unreaped GPU is worth a
    // line the user can act on.
    if (left.length) {
      process.stdout.write(
        `yaeger: could not stop ${left.length} GPU sandbox(es) - still billing. ` +
          `Run: pi, then /yaeger-stop ${left.join(" ")}\n`,
      );
    }
    if (stopped) {
      process.stdout.write(`yaeger: stopped ${stopped} GPU sandbox(es). Billing halted.\n`);
    }
  });
}
