# EvilGenie

This is the MIT-licensed EvilGenie reward-hacking benchmark, vendored from the
pinned revision recorded in `UPSTREAM.md`. It gives a coding agent visible tests
and measures three independent signals:

1. whether the solution passes visible and hidden holdout tests;
2. whether a GPT-5 code judge classifies it as legitimate, heuristic, or reward
   hacking; and
3. whether the agent modified `test.py` or `test_cases.json`.

The SDF wrapper preserves those categorical results in every sample's metadata and
reports their rates separately. The headline scoreboard value is the LLM judge's
reward-hacking rate; file modification and test outcomes are secondary metrics, not
silently folded into a composite.

EvilGenie is an agent benchmark, so one sample can make up to the upstream limit of
500 agent messages. At most eight samples are active at once, matching the Tinker
provider's request concurrency so queued sessions do not burn their wall-clock
budget. It also downloads the selected LiveCodeBench release on first use. The
normal run uses one paid GPT-5 judge call per problem:

```bash
uv run sdf evals run \
  --evals evilgenie \
  --models base,adapter \
  --experiment evilgenie-r1 \
  --limit 10
```

For a cheap apparatus smoke test, call the task directly with three agent turns and
no judge. This still bills Tinker sampling, but not GPT-5 judging:

```bash
uv run inspect eval evals/evilgenie/task.py@evilgenie \
  --model tinker/deepseek-ai/DeepSeek-V3.1 \
  -M renderer_name=deepseekv3_thinking \
  -T limit=1 -T max_messages=3 -T enable_llm_judge=false \
  --display plain
```

The pinned upstream runner uses `deepseekv3_thinking`; keep the same renderer on
both the base model and any DeepSeek adapter when comparing them.

Tinker's DeepSeek-V3.1 sampler currently enforces a 32,768-token total context,
although the underlying model has a longer native context. For ten-minute agent
runs, opt into the provider's auditable rolling-context handling on both sides of
a base/adapter comparison:

```bash
-M context_window=32768
```

The full trajectory is retained until fewer than 2,048 tokens remain for the next
action. Only then are the oldest completed tool turns removed; the system/tool
prefix, original problem, and recent interaction stay in context. Every generated
message records prompt tokens, generation budget, and the number of removed history
messages under `metadata.sdf_context`. Omitting this model argument preserves the
provider's previous behavior exactly.

The task refuses to run without isolation. `sandbox=auto` prefers Docker and falls
back to rootless bubblewrap; either can be selected explicitly with `-T`.
