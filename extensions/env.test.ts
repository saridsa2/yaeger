/**
 * Tests for the serving container's environment.
 *
 * Run with: npm test
 *
 * The bug these lock down: the sandbox path used to ignore spec.engine.env
 * entirely, so a harness declaring HF_HUB_ENABLE_HF_TRANSFER silently never got
 * it while render.ts applied it correctly. Drift between the two paths is the
 * recurring failure in this codebase, so the behaviour is pinned here.
 */

import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import { buildEnv } from "./sandbox.ts";
import type { HarnessSpec } from "./render.ts";

const spec = (env?: Record<string, string>, extra?: Partial<HarnessSpec>): HarnessSpec =>
  ({
    model: { repo: "Qwen/Qwen3.8-27B-FP8" },
    hardware: { gpu: "H100", count: 1 },
    engine: { name: "vllm", version: "0.27.1", image: "vllm/vllm-openai:v0.27.1", env },
    serving: { max_model_len: 262144 },
    ...extra,
  }) as HarnessSpec;

const savedToken = process.env.HF_TOKEN;
const savedAlt = process.env.HUGGING_FACE_HUB_TOKEN;

afterEach(() => {
  if (savedToken === undefined) delete process.env.HF_TOKEN;
  else process.env.HF_TOKEN = savedToken;
  if (savedAlt === undefined) delete process.env.HUGGING_FACE_HUB_TOKEN;
  else process.env.HUGGING_FACE_HUB_TOKEN = savedAlt;
});

test("the harness spec's env reaches the container", () => {
  const env = buildEnv(spec({ HF_HUB_ENABLE_HF_TRANSFER: "1", VLLM_USE_V1: "1" }));
  assert.equal(env.HF_HUB_ENABLE_HF_TRANSFER, "1");
  assert.equal(env.VLLM_USE_V1, "1");
});

test("our defaults are applied when the spec declares no env", () => {
  const env = buildEnv(spec());
  assert.equal(env.HF_XET_HIGH_PERFORMANCE, "1");
  assert.equal(env.PYTHONHASHSEED, "0");
});

test("the spec wins over our defaults - the harness knows the model", () => {
  const env = buildEnv(spec({ HF_XET_HIGH_PERFORMANCE: "0" }));
  assert.equal(env.HF_XET_HIGH_PERFORMANCE, "0");
});

test("HF_TOKEN is forwarded only when the user actually has one", () => {
  delete process.env.HF_TOKEN;
  delete process.env.HUGGING_FACE_HUB_TOKEN;
  assert.equal(buildEnv(spec()).HF_TOKEN, undefined);

  process.env.HF_TOKEN = "hf_testtoken";
  assert.equal(buildEnv(spec()).HF_TOKEN, "hf_testtoken");
});

test("HUGGING_FACE_HUB_TOKEN works as a fallback name", () => {
  delete process.env.HF_TOKEN;
  process.env.HUGGING_FACE_HUB_TOKEN = "hf_alt";
  assert.equal(buildEnv(spec()).HF_TOKEN, "hf_alt");
});

test("a hostile env key from the service is refused, not passed through", () => {
  // Specs arrive from a remote service. An env key carrying shell metacharacters
  // must fail loudly rather than reach a container.
  assert.throws(() => buildEnv(spec({ "BAD KEY; rm -rf /": "1" })), /outside the allowed set/);
});
