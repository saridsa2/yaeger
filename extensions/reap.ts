/**
 * What to do with a GPU sandbox when the session that started it goes away.
 *
 * Deliberately free of the Modal SDK and of pi: deciding whether to kill a GPU
 * is the part worth testing, and it should be testable without credentials, a
 * network, or a running session.
 */

/**
 * What should happen to a sandbox when the pi session that started it quits.
 *
 * "exit" - a personal endpoint: nobody else is using it, so stop it and stop
 *          the bill. "keep" - a shared team endpoint: members may be mid-request,
 *          and the owner leaving is not the team finishing.
 *
 * Stored as a Modal tag rather than a local file so it survives a crash, works
 * from another machine, and cannot drift out of sync with what is running.
 */
export type ReapPolicy = "exit" | "keep";

export interface RunningSandbox {
  sandboxId: string;
  model: string;
  /** null for sandboxes started before reap tagging, or if tags could not be read. */
  reap: ReapPolicy | null;
}

/** Narrow an arbitrary tag value to a policy, treating anything unknown as untagged. */
export function parseReapTag(value: unknown): ReapPolicy | null {
  return value === "exit" || value === "keep" ? value : null;
}

/**
 * Which running sandboxes this session should stop on its way out.
 *
 * Two rules matter: only a real quit reaps (a reload, fork or session switch is
 * not the user leaving), and an untagged sandbox is never stopped automatically
 * - we do not know whether it is someone's shared endpoint, so it gets surfaced
 * at the next session start instead of killed silently.
 */
export function sandboxesToReap(
  running: RunningSandbox[],
  reason: string,
): RunningSandbox[] {
  if (reason !== "quit") return [];
  return running.filter((s) => s.reap === "exit");
}
