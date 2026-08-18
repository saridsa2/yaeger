/**
 * Tests for the reap decision.
 *
 * Run with: npm test
 *
 * Only the pure decision is covered here. Everything around it needs a live
 * Modal account and a running pi session, and a test that mocks both proves
 * nothing about either - so the logic that decides whether to kill a GPU lives
 * in reap.ts, free of the SDK, precisely so it can be checked in isolation.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { parseReapTag, sandboxesToReap, type RunningSandbox } from "./reap.ts";

const sb = (sandboxId: string, reap: RunningSandbox["reap"]): RunningSandbox => ({
  sandboxId,
  model: "Qwen/Qwen3.8-27B-FP8",
  reap,
});

test("a personal endpoint is reaped on quit", () => {
  const got = sandboxesToReap([sb("sb-1", "exit")], "quit");
  assert.deepEqual(got.map((s) => s.sandboxId), ["sb-1"]);
});

test("a shared team endpoint survives the owner quitting", () => {
  assert.deepEqual(sandboxesToReap([sb("sb-1", "keep")], "quit"), []);
});

test("an untagged sandbox is never reaped automatically", () => {
  // We cannot tell whether it is someone's shared endpoint. The startup sweep
  // surfaces it to a human instead of killing it silently.
  assert.deepEqual(sandboxesToReap([sb("sb-legacy", null)], "quit"), []);
});

test("only a real quit reaps anything", () => {
  const running = [sb("sb-1", "exit")];
  for (const reason of ["reload", "new", "resume", "fork"]) {
    assert.deepEqual(
      sandboxesToReap(running, reason),
      [],
      `reason "${reason}" must not stop a GPU`,
    );
  }
});

test("mixed policies: only the exit-tagged ones go", () => {
  const got = sandboxesToReap(
    [sb("sb-solo", "exit"), sb("sb-team", "keep"), sb("sb-old", null), sb("sb-solo2", "exit")],
    "quit",
  );
  assert.deepEqual(got.map((s) => s.sandboxId), ["sb-solo", "sb-solo2"]);
});

test("nothing running is not an error", () => {
  assert.deepEqual(sandboxesToReap([], "quit"), []);
});

test("parseReapTag rejects anything that is not a known policy", () => {
  assert.equal(parseReapTag("exit"), "exit");
  assert.equal(parseReapTag("keep"), "keep");
  for (const bad of [undefined, null, "", "EXIT", "stop", "1", 1, {}]) {
    assert.equal(parseReapTag(bad), null, `${JSON.stringify(bad)} must not parse as a policy`);
  }
});
