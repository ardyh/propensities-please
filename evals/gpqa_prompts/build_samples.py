"""Freeze GPQA Diamond's *rendered* multiple-choice prompts into a prompt set.

    uv run python evals/gpqa_prompts/build_samples.py

Why this exists: `sdf train --distill_eval` reads its prompts off `task.dataset`, so a
Python eval is only a legitimate prompt source when its solver is a bare `generate()`
and each Sample already holds the whole prefix the model will see. `evals/gpqa` is not
-- it uses `multiple_choice(cot=True)`, which builds the MCQ text at *solve* time from
`sample.choices`, so `sample.input` there is the bare question with no options at all.
`distill._require_plain_generate` refuses it by name, which is right: distilling on the
bare question would train on a prefix the eval never presents.

So the rendering is done once, here, and frozen. Nothing about the prompt is authored:
the question and the four options come from `inspect_evals`' own `record_to_sample` over
the SHA-verified GPQA Diamond CSV, and the text is assembled by `inspect_ai`'s own
`prompt()` with its own `SINGLE_ANSWER_TEMPLATE_COT` -- the exact call
`multiple_choice(cot=True).solve` makes. Copying the template string in instead would be
a second source of truth that drifts the first time inspect edits it.

**The choice order is frozen at build time, and that is the point.**
`get_gpqa_diamond_dataset` shuffles the options on every load (the raw CSV always has the
correct answer first), so two loads disagree about which letter is correct. A prompt set
has to be a file, and a file has to pick one order. Re-running this script rewrites
`samples.jsonl` with a *different* shuffle, which makes the prompts a different stimulus
-- bump `version` in eval.json if you do, and expect the letter distribution to move.

The invariant that makes the freeze safe is checked below rather than assumed: for every
row, the option sitting at the target letter must be the CSV's own `Correct Answer`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from inspect_ai.solver._multiple_choice import SINGLE_ANSWER_TEMPLATE_COT, Choices, prompt
from inspect_evals.gpqa.gpqa import GPQA_DIAMOND_CACHE_PATH, get_gpqa_diamond_dataset

HERE = Path(__file__).parent


def main() -> None:
    dataset = get_gpqa_diamond_dataset()

    # The CSV the dataset was built from, keyed by Record ID -- the oracle for the
    # freeze check. Reading it directly rather than trusting the shuffle to have kept
    # `target` in step with `choices`.
    with GPQA_DIAMOND_CACHE_PATH.open(newline="", encoding="utf-8") as handle:
        correct = {row["Record ID"]: str(row["Correct Answer"]) for row in csv.DictReader(handle)}

    rows = []
    for sample in dataset:
        assert sample.choices is not None, f"{sample.id}: no choices"
        # `Sample.choices` is a list of strings; the solver wraps it in `Choices`
        # before rendering, and `answer_options` reads `.value` off each member.
        choices = Choices(sample.choices)
        target = sample.target if isinstance(sample.target, str) else sample.target[0]

        # inspect's own assembly, with inspect's own template. Not reimplemented.
        text = prompt(question=str(sample.input), choices=choices, template=SINGLE_ANSWER_TEMPLATE_COT)

        # The freeze check: the letter recorded as the target must index the option
        # that the CSV calls correct. A shuffle that moved the choices without moving
        # the target would otherwise be invisible -- the prompts would still look
        # fine, and every downstream accuracy would be noise around 25%.
        index = ord(target) - ord("A")
        assert choices[index].value == correct[str(sample.id)], f"{sample.id}: target {target} is not the correct answer"

        metadata = dict(sample.metadata or {})
        rows.append(
            {
                "id": str(sample.id),
                "input": text,
                "target": target,
                "metadata": {
                    "high_level_domain": metadata["high_level_domain"],
                    "subdomain": metadata["subdomain"],
                },
            }
        )

    out = HERE / "samples.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            # `ensure_ascii=True`, and it is load-bearing rather than a default left
            # alone. `registry.read_samples` splits the file with `str.splitlines()`,
            # which breaks on more than `\n` -- \x0b, \x0c, \x1c-\x1e, \x85, \u2028,
            # \u2029 -- and one GPQA question contains a literal \u2028 LINE SEPARATOR.
            # Written raw, that row is torn in half and the file fails to load with a
            # JSON error pointing at a line number that does not exist in any editor.
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    letters = sorted({r["target"] for r in rows})
    print(f"Wrote {len(rows)} samples to {out} (targets: {', '.join(letters)})")


if __name__ == "__main__":
    main()
