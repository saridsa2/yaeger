# yaeger-pi

Run open-weight models on **your own Modal account**, from inside
[pi](https://github.com/earendil-works/pi). One plugin, any model, shared across a team.

Your code and prompts never leave your infrastructure. The hosted service hands
out deployment configs and decides who may fetch a team's endpoint; it never
proxies inference and never sees a prompt.

```
/yaeger-login                      sign in (emailed code, no password)
/yaeger-team new "Backend"         create a team
/yaeger-team-invite <id> <email>   invite someone
/yaeger-team-start <id>            launch the shared endpoint (owner pays)
/yaeger-team-use <id>              join it (members need no Modal account)
/yaeger-team-insights <id>         who used what, and whether it beats the API
```

## Why

Serving an open model is not hard once; it is hard *every time*, because every
model has its own quirks. Four real examples, all of which cost an evening and
none of which appear in any documentation:

| Symptom | Cause |
|---|---|
| Engine dies during CUDA graph capture | hybrid Gated-DeltaNet needs `max_num_seqs` well below vLLM's default of 1024 |
| Tool calls silently never fire | Qwen3.8 emits **XML** tool calls; the usual `hermes` parser expects JSON |
| Container exits with "unrecognized arguments: serve" | `vllm/vllm-openai` ships an `ENTRYPOINT`, so an explicit command is double-invoked |
| Fluent output with foreign-language garbage tokens | a broken Gated-DeltaNet kernel — no flag fixes it, the engine has to change |

yaeger-pi keeps these as a knowledge base. A harness that has booted is stored
and served again; an unseen architecture is built with the known gotchas applied
and its failures fed back. **Lookup before generation** — a stored config that
has actually run is worth more than a freshly generated guess.

## Does self-hosting actually save money?

Often it does not, and the plugin will tell you so.

A GPU bills whether or not you are generating. One developer cannot keep an H100
busy; a team can. `/yaeger-team-insights` counts tokens served against GPU
seconds paid for — **including idle time**, which is the number vendor
comparisons leave out — and gives you a straight answer:

```
  gpu time             22.5 hours
  gpu cost             $88.88
  same tokens on API   $54.45
  cost per M tokens    $8.59

  the API would be $34.43 cheaper - utilisation is too low.
  Add people to the team or stop the endpoint between sessions.
```

## How it works

```
your machine                 the service              your Modal account
  pi + yaeger        ──auth──▶  identity, teams,
                                harness store + KB
                     ◀──config──
  launch via SDK ─────────────────────────────────▶  vLLM on a GPU sandbox
  inference ◀──────────────────────────────────────  (never via the service)
```

* **Harnesses are config, not code.** The service returns GPU, image, argv,
  ports and timeouts. Every argument is allowlist-validated locally before it
  reaches a process.
* **Sandboxes, not Functions.** Modal Functions can only be defined in Python;
  Sandboxes can be created from TypeScript with a GPU, an image and a TLS
  tunnel. So the plugin needs no Python and no `modal` CLI, and gets a hard
  `timeoutMs` spend ceiling that Functions have no equivalent for.
* **Idle safety.** Endpoints stop themselves after inactivity, verified by
  measurement: traffic through the tunnel resets the timer, and silence
  terminates the sandbox on schedule.

## Requirements

* pi, and Node 22+ (the Modal SDK requires it)
* A Modal account **for whoever starts an endpoint**. Team members need none.

## Install

```bash
pi install npm:@saridsa2/yaeger-pi     # once published to npm
pi install git:github.com/saridsa2/yaeger
```

Or try it for a single session without installing:

```bash
pi -e git:github.com/saridsa2/yaeger
```

Then `/yaeger-login` in pi.

## Self-hosting the service

The plugin has exactly one hosted dependency, and it is a single environment
variable:

```bash
YAEGERPI_SERVICE_URL=https://pi.example.com
```

The service is FastAPI + SQLite behind any reverse proxy. Copy `.env.example`,
fill it in, seed the knowledge base, and run it:

```bash
python -m venv venv && ./venv/bin/pip install -r service/requirements.txt
./venv/bin/python service/seed.py
./venv/bin/uvicorn app:app --host 127.0.0.1 --port 8002
```

Identity is Supabase. Sign-in is passwordless: the service emails a single-use
code, exchanges it for a session, and refreshes silently thereafter.

## What is gated, and why

| Gate | Protects | Who passes |
|---|---|---|
| sign-in | nothing expensive | anyone with an email |
| `can_generate` | **the host's** GPU | invite only |
| team membership | a team's endpoint and its owner's bill | whoever the owner adds |

Generating a harness for an unseen architecture runs on the host's hardware, so
it is invite-only. Everything already in the catalog is available to anyone
signed in, because serving it costs the host nothing. Self-hosters generate on
their own GPU and can open that up however they like.

## Privacy

Prompts and completions go straight from each member's machine to their team's
endpoint. The service never sees them.

If tracing is enabled, requests are logged to a volume **on the team's own Modal
account** and are readable only by the team owner. Members are told this
explicitly before they join — silent tracing of a teammate would make the
sovereignty claim a lie.

Usage totals are self-reported by the plugin, which is fine for a team measuring
itself and is not a billing ledger for strangers.

## Status

Working end to end and not yet polished. Known gaps: tier-3 generation applies
knowledge-base rules but has no model behind it, the catalog is small, and the
interactive command surface has had less exercise than the API underneath it.

## License

Apache-2.0.
