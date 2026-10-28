"""Building an Inspect Task from `eval.json` + `samples.jsonl`, with no Python.

This is the path most evals should take. A behaviour eval here is a set of prompts
and a way of reading the answer; expressing that as data means a collaborator adds
eval #7 by writing two files, and means the prompts are diffable in review rather
than buried in string constants.

`samples.jsonl` uses Inspect's own `Sample` field names (`id`, `input`, `target`,
`metadata`), so the file is read with Inspect's `json_dataset` rather than a loader
of our own. The escape hatch is `task_ref` in `eval.json`, for an eval that genuinely
needs a custom solver or scorer -- see `evals/reward_hack/` for a worked example.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message

from sdf.evals import scoring
from sdf.evals.registry import EvalSpec


def build(spec: EvalSpec, epochs: int, temperature: float, max_tokens: int) -> Task:
    samples_path = spec.samples_path
    assert samples_path is not None, "build() is only for prompt-set evals"
    if not samples_path.exists():
        builder = samples_path.parent / "build_samples.py"
        raise FileNotFoundError(
            f"{samples_path} does not exist. Build it first: uv run python {builder}"
            if builder.exists() else f"{samples_path} does not exist."
        )

    dataset = json_dataset(str(samples_path))

    solver = [generate()]
    if spec.system_prompt:
        solver = [system_message(spec.system_prompt), generate()]

    return Task(
        name=spec.name,
        dataset=dataset,
        solver=solver,
        scorer=scoring.build_scorer(spec.scorer, group_by=spec.group_by, base_dir=spec.path),
        # A handful of prompts is far too few to read once, so the epoch count
        # carries the statistics: at 100 epochs each prompt is a 100-sample rate
        # estimate, and `stderr()` gives the interval to read it with.
        epochs=Epochs(epochs, "mean"),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )
