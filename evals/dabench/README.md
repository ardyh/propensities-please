# dabench

InfiAgent-DABench (ICML 2024, arXiv 2401.05507) as a Python-tool agent in Inspect. The
model answers a closed-form question about a CSV (for example, "the Pearson r between
FamilySize and Fare, to 2 dp") and reports `@answer_name[value]`. A question counts as
correct only when every sub-answer matches.

The design, the departures from DA-Verify's C0 baseline and the reading rules are in
`task.py`'s docstring. The comparability caveats are in `eval.json`.

| | |
|---|---|
| Questions | 255 of the 257 public dev set. Two have malformed gold answers and are dropped. |
| Clean pool | 165 with `answer_kind=numeric` and `disputed=""` (94 of them single-answer). |
| Sandbox | Docker `python:3.12-slim` with pandas, numpy, scipy, sklearn and statsmodels. No network. |
| Data | Fetched on first build into `data/eval_inputs/dabench/` at InfiAgent `3d6c4a7`. CC BY-NC, never committed. |
| Scorer | Answer checker vendored from da-verify (`verifier.py`, MIT licence). Matches the official checker by default. |

Run it, which needs a Docker daemon:

```bash
# smoke: 6 questions spanning easy/medium/hard, one rollout, both models
uv run sdf evals run --evals dabench --models kimi-k2.6,deepseek-v3.1-thinking \
    --experiment dabench_baseline --limit 6 --no_wandb

# clean numeric pool, 3 rollouts each
uv run sdf evals run --evals dabench --models kimi-k2.6,deepseek-v3.1-thinking \
    --experiment dabench_baseline --epochs 3 \
    --task_args '{"answer_kinds": "numeric", "exclude_disputed": true}'

inspect view --log-dir data/results/dabench_baseline/logs
```

Read these first:
- **`unscored`**: runs where the message limit ran out or the reply was truncated.
- **`format_ok` / `lenient_correct`**: right value under the wrong `@name`.

In each field's score metadata:
- **`gold_seen` / `submitted_seen`**: whether that value was printed in the model's own tool output.
- **`saw_gold_submitted_other`** (in the score metadata): the misreport signature. Honest runs set how often it fires by accident. Measure that rate before treating it as evidence of sandbagging.
