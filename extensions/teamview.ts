/**
 * Contributor insights, rendered for a terminal.
 *
 * Deliberately shaped like GitHub's contributors page: a per-person activity
 * strip over time, ordered by volume, with the totals that matter beside it.
 * The strip is the point - a table of totals hides *when* people worked, and
 * that pattern is what tells an admin whether the endpoint is worth keeping up.
 */

const BLOCKS = [" ", "░", "▒", "▓", "█"]; // light -> solid

export interface Contributor {
  email: string;
  role?: string;
  tokens: number;
  requests: number;
  active_days: number;
  days: Record<string, { tokens: number; requests: number }>;
}

function human(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return String(n);
}

function dayKeys(days: number): string[] {
  const out: string[] = [];
  const now = Date.now();
  for (let i = days - 1; i >= 0; i--) {
    out.push(new Date(now - i * 86_400_000).toISOString().slice(0, 10));
  }
  return out;
}

/** One block per day, shaded by that day's volume relative to the busiest. */
function strip(c: Contributor, keys: string[], peak: number): string {
  return keys
    .map((k) => {
      const t = c.days[k]?.tokens ?? 0;
      if (t === 0) return BLOCKS[0];
      const q = Math.ceil((t / peak) * (BLOCKS.length - 1));
      return BLOCKS[Math.max(1, Math.min(BLOCKS.length - 1, q))];
    })
    .join("");
}

export function renderInsights(
  teamName: string,
  contributors: Contributor[],
  windowDays: number,
): string {
  const keys = dayKeys(Math.min(windowDays, 30));
  const peak = Math.max(
    1,
    ...contributors.flatMap((c) => keys.map((k) => c.days[k]?.tokens ?? 0)),
  );

  const width = Math.max(18, ...contributors.map((c) => c.email.length));
  const lines: string[] = [
    `${teamName} - contributors, last ${keys.length} days`,
    "",
    `${"member".padEnd(width)}  ${keys.length >= 14 ? "activity".padEnd(keys.length) : "activity"}  ${"tokens".padStart(7)}  ${"reqs".padStart(6)}  days`,
    "-".repeat(width + keys.length + 26),
  ];

  for (const c of contributors) {
    const who = c.email + (c.role === "owner" ? " *" : "");
    lines.push(
      `${who.padEnd(width)}  ${strip(c, keys, peak)}  ${human(c.tokens).padStart(7)}  ` +
        `${String(c.requests).padStart(6)}  ${String(c.active_days).padStart(4)}`,
    );
  }

  const totals = contributors.reduce(
    (a, c) => ({ tokens: a.tokens + c.tokens, requests: a.requests + c.requests }),
    { tokens: 0, requests: 0 },
  );
  lines.push("-".repeat(width + keys.length + 26));
  lines.push(
    `${"total".padEnd(width)}  ${" ".repeat(keys.length)}  ${human(totals.tokens).padStart(7)}  ${String(totals.requests).padStart(6)}`,
  );
  if (contributors.some((c) => c.tokens === 0)) {
    lines.push("");
    lines.push("members with no activity have joined but never used the endpoint");
  }
  return lines.join("\n");
}

/** The economics verdict, formatted so the uncomfortable number is unmissable. */
export function renderEconomics(e: any): string {
  const rows = [
    ["window", `${e.window_days} days`],
    ["people active", String(e.people_active)],
    ["requests", String(e.requests)],
    ["tokens", human((e.prompt_tokens ?? 0) + (e.completion_tokens ?? 0))],
    ["gpu time", `${e.gpu_hours} hours`],
    ["gpu cost", `$${e.gpu_usd}`],
    ["same tokens on API", `$${e.api_equivalent_usd}`],
  ];
  if (e.usd_per_million_tokens != null) rows.push(["cost per M tokens", `$${e.usd_per_million_tokens}`]);
  if (e.tokens_per_gpu_hour != null) rows.push(["tokens per gpu-hour", human(e.tokens_per_gpu_hour)]);

  const w = Math.max(...rows.map((r) => r[0].length));
  return (
    rows.map(([k, v]) => `  ${k.padEnd(w)}  ${v}`).join("\n") + `\n\n  ${e.verdict}`
  );
}
