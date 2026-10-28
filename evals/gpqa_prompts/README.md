# gpqa_prompts

GPQA Diamond's **rendered** multiple-choice prompts, frozen into a `samples.jsonl` so
`sdf train --distill_eval gpqa_prompts` can sample on them.

This is a prompt *source* for on-policy distillation. It is not a capability control and
no number should be read off it — that row is `evals/gpqa`.

## Why it exists

`sdf train --distill_eval` takes its prompts off `task.dataset`, so a Python eval is a
legitimate source only when its solver is a bare `generate()` and each Sample already
carries the whole prefix the model will see. `evals/gpqa` does not:

```
--distill_eval gpqa: its solver chain is ['multiple_choice'], but a prompt source
must be a bare generate() with the whole [system?, user] prefix in each sample.
```

`multiple_choice(cot=True)` assembles the MCQ text at *solve* time out of
`sample.choices`, so `sample.input` there is the bare question with no options attached.
The refusal is correct — distilling on the bare question would train on a prefix the eval
never presents. `mmlu_pro` has the same shape and the same problem.

So the rendering happens once, in `build_samples.py`, and is frozen. Nothing about the
prompt is authored here: the question and its four options come from `inspect_evals`' own
`record_to_sample` over the SHA-verified GPQA Diamond CSV, and the text is assembled by
`inspect_ai`'s own `prompt()` with its own `SINGLE_ANSWER_TEMPLATE_COT` — the exact call
the solver makes. The alternative, editing `evals/gpqa/task.py` to pre-render, would have
changed the capability control itself: a `version` bump, and GPQA numbers either side of
it no longer comparable.

## What will bite

**The choice order is frozen, and re-running the build changes the stimulus.** GPQA's
options are shuffled on every dataset load — the raw CSV always lists the correct answer
first — so a file has to pick one order, and this one did, on 2026-09-02. Re-running
`build_samples.py` reshuffles: bump `version` in `eval.json` if you do, and expect the
letter distribution to move. The build asserts, per row, that the option at the target
letter is the CSV's own `Correct Answer`, so a shuffle that moved the choices without
moving the target fails loudly instead of yielding 198 prompts whose accuracy is noise
around 25%.

**`ensure_ascii=True` in the writer is load-bearing.** `registry.read_samples` splits the
file with `str.splitlines()`, which breaks on more than `\n` — `\x0b`, `\x0c`,
`\x1c`–`\x1e`, `\x85`, ` `, ` ` — and one GPQA question contains a literal
` ` LINE SEPARATOR. Written raw, that row is torn in half and the whole file fails to
load with a JSON error pointing at a line number no editor shows.

**It is out of `--evals all`.** `eval.json` sets `"suite": false`, so a whole-suite run
skips it rather than paying for these 198 questions *and* `gpqa`'s 792; name it
(`--evals gpqa_prompts`) to run it. `epochs` is 1 rather than gpqa's 4 to bound that
when you do.

**Its accuracy is not `gpqa`'s accuracy.** The prompt-set path cannot reach
`scoring.multiple_choice_correct` (that scorer needs a dataset with `choices`, which a
prompt set does not build), so this eval grades with `scoring.letter_choice` — the last
standalone `[A-Z]` in the answer. Same questions, different parser. The scorer is here so
the eval is runnable and readable at all, not so the number can be compared.

## Rebuilding

```bash
uv run python evals/gpqa_prompts/build_samples.py
```

Needs the GPQA Diamond CSV, which `inspect_evals` downloads and SHA-verifies into
`~/.cache/inspect_evals/gpqa/gpqa_diamond.csv` on first use. It is a public URL — unlike
the gated HF copy, no `HF_TOKEN` is involved.
