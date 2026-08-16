/**
 * Client for the yaeger-pi harness service, plus local credential storage.
 *
 * Auth is passwordless: request a code by email, paste it back, exchange it for
 * a session. The plugin never handles a password and never sees the Supabase
 * anon key - the service does that exchange.
 */

import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import type { HarnessSpec } from "./render.ts";

/**
 * Where the harness service lives. Point this at your own deployment to
 * self-host - the plugin has no other hardcoded dependency on yaeger.
 */
export const SERVICE_URL = process.env.YAEGERPI_SERVICE_URL ?? "https://pi.yaeger.dev";

const AUTH_FILE = join(homedir(), ".pi", "agent", "yaeger-auth.json");

export interface Session {
  email: string;
  access_token: string;
  refresh_token?: string;
  expires_at?: number;
}

export function loadSession(): Session | null {
  try {
    if (!existsSync(AUTH_FILE)) return null;
    return JSON.parse(readFileSync(AUTH_FILE, "utf8")) as Session;
  } catch {
    return null;
  }
}

export function saveSession(s: Session): void {
  mkdirSync(dirname(AUTH_FILE), { recursive: true });
  writeFileSync(AUTH_FILE, JSON.stringify(s, null, 2), { mode: 0o600 });
  chmodSync(AUTH_FILE, 0o600); // enforce even if the file already existed
}

export function clearSession(): void {
  if (existsSync(AUTH_FILE)) writeFileSync(AUTH_FILE, "{}", { mode: 0o600 });
}

async function jsonFetch(url: string, init: RequestInit & { timeoutMs?: number } = {}) {
  const { timeoutMs = 60_000, ...rest } = init;
  const res = await fetch(url, { ...rest, signal: AbortSignal.timeout(timeoutMs) });
  const text = await res.text();
  let body: unknown = text;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    /* keep raw text for the error message */
  }
  if (!res.ok) {
    const detail =
      typeof body === "object" && body && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : typeof body === "string"
          ? body.slice(0, 300)
          : res.statusText;
    throw new Error(`${res.status} ${detail}`);
  }
  return body as any;
}

// ---------------------------------------------------------------- auth

/** Step 1: ask yaeger-pi to email a single-use sign-in code. */
export async function requestLoginEmail(email: string): Promise<number> {
  const r = await jsonFetch(`${SERVICE_URL}/v1/auth/email`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  return r?.expires_in_minutes ?? 30;
}

/** Step 2: exchange the token from that email for a Supabase session. */
export async function resolveLoginToken(token: string): Promise<Session> {
  // Accept either a bare token or the full deep link pasted from the email.
  const cleaned = token.trim().replace(/^.*[?&]token=/, "").replace(/[#&].*$/, "");
  const r = await jsonFetch(`${SERVICE_URL}/v1/auth/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: cleaned }),
  });
  const session: Session = {
    email: r.email,
    access_token: r.access_token,
    refresh_token: r.refresh_token,
    expires_at: r.expires_in ? Date.now() + r.expires_in * 1000 : undefined,
  };
  saveSession(session);
  return session;
}

function authHeaders(): Record<string, string> {
  const s = loadSession();
  if (!s?.access_token) {
    throw new Error("not signed in - run /yaeger-login first");
  }
  return { Authorization: `Bearer ${s.access_token}`, "Content-Type": "application/json" };
}

/**
 * Supabase access tokens last an hour. Renew silently rather than making the
 * user re-run /yaeger-login every hour; the refresh grant lives on the service
 * so the anon key never ships in the plugin.
 */
async function refreshSession(): Promise<boolean> {
  const s = loadSession();
  if (!s?.refresh_token) return false;
  try {
    const r = await jsonFetch(`${SERVICE_URL}/v1/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: s.refresh_token }),
    });
    saveSession({
      email: s.email,
      access_token: r.access_token,
      refresh_token: r.refresh_token ?? s.refresh_token,
      expires_at: Date.now() + (r.expires_in ?? 3600) * 1000,
    });
    return true;
  } catch {
    return false;
  }
}

/** Call an authenticated endpoint, refreshing once if the token has expired. */
async function authed<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    const m = (e as Error).message ?? "";
    if (!/^401|expired|Unauthorized/i.test(m)) throw e;
    if (!(await refreshSession())) {
      throw new Error("session expired - run /yaeger-login again");
    }
    return await fn();
  }
}

// ---------------------------------------------------------------- service

export interface CatalogEntry {
  model_repo: string;
  architecture: string | null;
  quantization: string;
  gpu: string;
  gpu_count: number;
  est_usd_per_hour: number;
  verified: boolean;
  boot_count: number;
}

export interface Me {
  email: string;
  can_generate: boolean;
  daily_generation_limit: number;
}

/** What this account is allowed to do. The server is the authority, not the user. */
export async function whoami(): Promise<Me> {
  return authed(() => jsonFetch(`${SERVICE_URL}/v1/me`, { headers: authHeaders() }));
}

export async function catalog(): Promise<CatalogEntry[]> {
  const r = await authed(() => jsonFetch(`${SERVICE_URL}/v1/catalog`, { headers: authHeaders() }));
  return r.models ?? [];
}

export interface ModelMetadata {
  architecture?: string;
  param_count_b?: number;
  quantization?: string;
  gated?: boolean;
}

/**
 * Read architecture and parameter count straight from the Hub.
 *
 * This is not a nicety: KB entries scoped by architecture (the ones that catch
 * things like hybrid-GDN cache limits) cannot match without it, and GPU sizing
 * falls back to a guess. Never make the user hand-enter what the Hub already knows.
 */
export async function fetchModelMetadata(repo: string): Promise<ModelMetadata> {
  try {
    const r = await jsonFetch(`https://huggingface.co/api/models/${repo}`, { timeoutMs: 20_000 });
    const meta: ModelMetadata = { gated: Boolean(r?.gated) };

    const arch = r?.config?.architectures?.[0];
    if (typeof arch === "string") meta.architecture = arch;

    const total = r?.safetensors?.total;
    if (typeof total === "number" && total > 0) {
      meta.param_count_b = Math.round((total / 1e9) * 10) / 10;
    }

    // Infer quantization from the dtype histogram the Hub reports.
    const params = r?.safetensors?.parameters ?? {};
    const dtypes = Object.keys(params);
    if (dtypes.some((d) => d.startsWith("F8"))) meta.quantization = "fp8";
    else if (dtypes.some((d) => /^(I4|U4)/.test(d))) meta.quantization = "awq";

    return meta;
  } catch {
    return {}; // the Hub being unreachable must not block a resolve
  }
}

export interface ResolveOpts {
  model_repo: string;
  revision?: string;
  architecture?: string;
  param_count_b?: number;
  quantization?: string;
  gpu?: string;
  gpu_count?: number;
  max_usd_per_hour?: number;
}

export async function resolveSpec(opts: ResolveOpts): Promise<HarnessSpec> {
  return authed(() => jsonFetch(`${SERVICE_URL}/v1/resolve`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(opts),
    timeoutMs: 180_000, // tier 3 may wake a GPU to generate
  }));
}

export interface DeployReport {
  model_repo: string;
  harness_id?: string;
  tier?: string;
  outcome: "booted" | "failed" | "aborted";
  phase?: string;
  error_excerpt?: string;
}

export interface Guidance {
  kb_entry: string;
  title: string;
  cause: string;
  fix: { spec_patch?: Record<string, Record<string, unknown>> | null; notes?: string };
}

export async function reportDeploy(r: DeployReport): Promise<{ guidance: Guidance | null }> {
  return authed(() =>
    jsonFetch(`${SERVICE_URL}/v1/events/deploy`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(r),
    }),
  );
}

// ---------------------------------------------------------------- teams

export interface TeamSummary {
  team_id: string; name: string; role: string; joined: boolean;
  endpoint_live: boolean; model: string | null; members: number;
}

export interface TeamEndpoint {
  url: string; key: string; model: string; gpu: string; usd_per_hour: number;
  context: string | null; context_updated_at: string | null;
  tracing?: { enabled: boolean; stored: string; visible_to: string };
}

const T = (p: string) => `${SERVICE_URL}/v1/teams${p}`;

export async function createTeam(name: string): Promise<{ team_id: string; name: string }> {
  return authed(() => jsonFetch(T(""), { method: "POST", headers: authHeaders(), body: JSON.stringify({ name }) }));
}

export async function myTeams(): Promise<TeamSummary[]> {
  const r = await authed(() => jsonFetch(T(""), { headers: authHeaders() }));
  return r.teams ?? [];
}

export async function inviteMember(teamId: string, email: string) {
  return authed(() => jsonFetch(T(`/${teamId}/invite`), { method: "POST", headers: authHeaders(), body: JSON.stringify({ email }) }));
}

export async function getTeamEndpoint(teamId: string): Promise<TeamEndpoint> {
  return authed(() => jsonFetch(T(`/${teamId}/endpoint`), { headers: authHeaders() }));
}

export async function publishTeamEndpoint(teamId: string, body: Record<string, unknown>) {
  return authed(() => jsonFetch(T(`/${teamId}/endpoint`), { method: "PUT", headers: authHeaders(), body: JSON.stringify(body) }));
}

export async function stopTeamEndpoint(teamId: string) {
  return authed(() => jsonFetch(T(`/${teamId}/endpoint`), { method: "DELETE", headers: authHeaders() }));
}

export async function setTeamContext(teamId: string, context: string) {
  return authed(() => jsonFetch(T(`/${teamId}/context`), { method: "PUT", headers: authHeaders(), body: JSON.stringify({ context }) }));
}

export async function reportUsage(teamId: string, u: { prompt_tokens: number; completion_tokens: number; requests?: number }) {
  return authed(() => jsonFetch(T(`/${teamId}/usage`), { method: "POST", headers: authHeaders(), body: JSON.stringify(u) }));
}

export async function teamEconomics(teamId: string, days = 7) {
  return authed(() => jsonFetch(T(`/${teamId}/economics?days=${days}`), { headers: authHeaders() }));
}

export async function teamInsights(teamId: string, days = 30) {
  return authed(() => jsonFetch(T(`/${teamId}/insights?days=${days}`), { headers: authHeaders() }));
}

/** Merge a KB spec_patch into a spec, section by section. */
export function applyPatch(spec: HarnessSpec, patch: Record<string, any> | null | undefined) {
  if (!patch) return spec;
  const next: any = { ...spec };
  for (const [section, values] of Object.entries(patch)) {
    next[section] =
      values && typeof values === "object" && !Array.isArray(values)
        ? { ...(next[section] ?? {}), ...values }
        : values;
  }
  return next as HarnessSpec;
}
