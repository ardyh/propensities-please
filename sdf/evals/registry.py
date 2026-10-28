"""Discovering evals and turning them into Inspect Tasks.

One eval is one directory under `evals/` containing an `eval.json`. There is exactly
one discovery path, whether the eval is a prompt set (an `eval.json` plus a
`samples.jsonl`) or hand-written Python (an `eval.json` pointing at a `@task`
factory). That matters more than it looks: the alternative -- a Python registry for
Python evals and a directory scan for data evals -- means every consumer, the runner
and the scoreboard and the scanners, has to know which kind it is holding.

Eval definitions live in git rather than under `data/`, which is the opposite of the
rule for everything else this pipeline reads. See `sdf.paths.EVALS_DIR` for why.

Validation is loud. `universe.py lint` fails rather than warns when a fact matches no
document, for the same reason: a suite that scored the wrong thing produces numbers
that look fine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sdf import paths
from sdf.evals import scoring

MANIFEST_NAME = "eval.json"

# Defaults applied when eval.json omits them. Sampled rather than greedy, because
# these evals measure a *rate* over many draws and a greedy model returns the same
# answer every epoch.
DEFAULT_EPOCHS = 100
DEFAULT_TEMPERATURE = 1.0
# Reasoning models spend most of their budget before emitting an answer, and a
# truncated response reaches the scorer with nothing to parse.
#
# Must equal `sdf.models.tinker.DEFAULT_MAX_TOKENS`, and is written out rather than
# imported because that module calls `compat.bootstrap()` and pulls in the Tinker SDK,
# which `sdf evals list` should not pay for. They were 1024 here and 4096 there, so an `eval.json` omitting the field
# got the smaller one -- and `tinker.py` documents exactly that range as where a
# thinking renderer spends its whole budget reasoning and every sample lands
# `truncated`, which reads as a model that stopped answering rather than as a budget.
DEFAULT_MAX_TOKENS = 4096
DEFAULT_PRIMARY_METRIC = "accuracy"

# What a score of 1 means; see `EvalSpec.polarity`. The scoreboard prints the first four
# characters beside every eval name, so these have to stay distinguishable at that length.
POLARITIES = ("gaming", "compliance", "capability")


@dataclass(frozen=True)
class EvalSpec:
    name: str
    path: Path
    description: str = ""
    version: int = 1
    task_ref: str | None = None
    samples: str | None = None
    scorer: dict[str, Any] = field(default_factory=dict)
    group_by: str | None = None
    primary_metric: str = DEFAULT_PRIMARY_METRIC
    # Optional secondary metrics to carry into the scoreboard. Most evals have one
    # scalar and leave this empty; categorical benchmarks such as EvilGenie expose
    # several independent measurements from the same paid transcript. Keeping this
    # opt-in avoids turning grouped scorer internals into surprise scoreboard rows.
    reported_metrics: tuple[str, ...] = ()
    epochs: int = DEFAULT_EPOCHS
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    # Inspect's sample concurrency is distinct from provider request concurrency.
    # For long-lived agents, leaving it unbounded starts every sample's wall clock
    # while most are merely queued for a model slot. Ordinary one-turn evals retain
    # Inspect's existing default; agent benchmarks opt in through their manifest.
    max_samples: int | None = None
    # Optional post-run analysis over the durable EvalLog. This is deliberately an
    # eval-owned hook: paired tests and validity gates are research content, while
    # the runner only guarantees that they execute and their artifacts are retained.
    analysis_ref: str | None = None
    system_prompt: str | None = None
    # What a score of 1 means. "gaming": the model played to the declared metric.
    # "compliance": the model followed the instruction. "capability": it answered
    # correctly -- a benchmark control such as `gpqa`, and the one direction where a
    # *falling* column is the bad news. A suite that mixes these without saying so
    # produces a scoreboard where a rising column is good in one row and bad in the
    # next, which is how a result gets read backwards.
    polarity: str = "gaming"
    # Whether `--evals all` includes it. `"suite": false` in eval.json keeps an eval
    # discoverable and runnable by name while leaving it out of a bare suite run: a
    # distillation prompt source that is not a benchmark row, or an eval still under
    # iteration whose version is about to churn. Without the field, a bare
    # `sdf evals run` sampled both, per model, on every suite run.
    suite: bool = True

    @property
    def samples_path(self) -> Path | None:
        return self.path / self.samples if self.samples else None

    @property
    def is_python(self) -> bool:
        return self.task_ref is not None


def _read_manifest(manifest: Path) -> EvalSpec:
    try:
        raw = json.loads(manifest.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest} is not valid JSON: {exc}") from exc

    directory = manifest.parent
    name = raw.get("name") or directory.name
    if name != directory.name:
        raise ValueError(f"{manifest}: name {name!r} does not match its directory {directory.name!r}. ")

    if bool(raw.get("task_ref")) == bool(raw.get("samples")):
        raise ValueError(
            f"{manifest}: set exactly one of 'task_ref' (hand-written @task) or "
            "'samples' (a prompt set). Got "
            + ("both" if raw.get("task_ref") else "neither")
            + "."
        )

    reported_metrics = raw.get("reported_metrics") or []
    if not isinstance(reported_metrics, list) or not all(
        isinstance(metric_name, str) and metric_name for metric_name in reported_metrics
    ):
        raise ValueError(f"{manifest}: reported_metrics must be a list of non-empty metric names")

    spec = EvalSpec(
        name=name,
        path=directory,
        description=raw.get("description", ""),
        version=int(raw.get("version", 1)),
        task_ref=raw.get("task_ref"),
        samples=raw.get("samples"),
        scorer=raw.get("scorer") or {},
        group_by=raw.get("group_by"),
        primary_metric=raw.get("primary_metric", DEFAULT_PRIMARY_METRIC),
        reported_metrics=tuple(reported_metrics),
        epochs=int(raw.get("epochs", DEFAULT_EPOCHS)),
        temperature=float(raw.get("temperature", DEFAULT_TEMPERATURE)),
        max_tokens=int(raw.get("max_tokens", DEFAULT_MAX_TOKENS)),
        max_samples=int(raw["max_samples"]) if raw.get("max_samples") is not None else None,
        analysis_ref=raw.get("analysis_ref"),
        system_prompt=raw.get("system_prompt"),
        polarity=raw.get("polarity", "gaming"),
        suite=bool(raw.get("suite", True)),
    )

    if spec.polarity not in POLARITIES:
        raise ValueError(f"{manifest}: polarity must be one of {', '.join(POLARITIES)}, got {spec.polarity!r}")
    if spec.max_samples is not None and spec.max_samples < 1:
        raise ValueError(f"{manifest}: max_samples must be positive when provided")

    if not spec.is_python:
        _validate_prompt_set(spec, manifest)
    return spec


def _validate_prompt_set(spec: EvalSpec, manifest: Path) -> None:
    samples_path = spec.samples_path
    assert samples_path is not None
    if not samples_path.exists():
        # A prompt set built locally from a licensed dataset (GPQA asks that its questions not be republished)
        # ships its builder instead of its samples: discovery must not fail for every other eval because of it.
        # `prompt_set.build` raises with the builder's name when the eval is actually used.
        if (manifest.parent / "build_samples.py").exists():
            return
        raise FileNotFoundError(f"{manifest} names samples {spec.samples!r}, but {samples_path} does not exist.")

    kind = spec.scorer.get("type")
    if kind not in scoring.SCORER_TYPES:
        raise ValueError(
            f"{manifest}: unknown scorer type {kind!r}. Known types: {', '.join(scoring.SCORER_TYPES)}"
        )

    rows = read_samples(samples_path)
    if not rows:
        raise ValueError(f"{samples_path} is empty.")

    # A group_by naming a key no sample carries does not raise anywhere downstream:
    # every sample falls into one "?" bucket and the breakdown silently becomes a
    # single row that looks like a legitimate result. Catch it at load.
    if spec.group_by:
        missing = [r.get("id", i) for i, r in enumerate(rows) if spec.group_by not in (r.get("metadata") or {})]
        if missing:
            raise ValueError(
                f"{manifest}: group_by={spec.group_by!r} is missing from the metadata of "
                f"{len(missing)}/{len(rows)} samples (first: {missing[0]!r}). Every sample "
                "must carry the grouping key, or the per-group breakdown collapses to one bucket."
            )


def read_samples(path: Path) -> list[dict[str, Any]]:
    """Read a `samples.jsonl` into raw dicts, skipping blank lines."""
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return rows


def discover(evals_dir: Path | str | None = None) -> dict[str, EvalSpec]:
    """Every eval under `evals_dir`, keyed by name."""
    root = Path(evals_dir) if evals_dir else paths.EVALS_DIR
    if not root.is_dir():
        return {}
    found: dict[str, EvalSpec] = {}
    for manifest in sorted(root.glob(f"*/{MANIFEST_NAME}")):
        # `_screens/` holds human validation labels for Scout scanners, not evals.
        if manifest.parent.name.startswith("_"):
            continue
        spec = _read_manifest(manifest)
        found[spec.name] = spec
    return found


def load(name: str, evals_dir: Path | str | None = None) -> EvalSpec:
    found = discover(evals_dir)
    if name not in found:
        known = ", ".join(sorted(found)) or "(none found)"
        raise KeyError(f"No eval named {name!r}. Available: {known}")
    return found[name]


def resolve_names(selector: str | list[str] | None, evals_dir: Path | str | None = None) -> list[str]:
    """Turn `--evals all` / `--evals a,b` / `--evals a` into a list of names.

    `all` is every eval whose manifest does not set `"suite": false`; an eval named
    explicitly always runs, whatever it sets.
    """
    found = discover(evals_dir)
    if selector in (None, "all", "*"):
        return sorted(name for name, spec in found.items() if spec.suite)
    names = selector if isinstance(selector, (list, tuple)) else str(selector).split(",")
    names = [n.strip() for n in names if n.strip()]
    unknown = [n for n in names if n not in found]
    if unknown:
        known = ", ".join(sorted(found)) or "(none found)"
        raise KeyError(f"Unknown eval(s): {', '.join(unknown)}. Available: {known}")
    return names


def effective_settings(spec: EvalSpec, **overrides: Any) -> dict[str, Any]:
    """What this eval will actually sample with, once CLI flags are folded in.

    Split out from `build_task` so the runner can record it. The resolved values used
    to exist only inside this function: the scoreboard stored neither them nor the raw
    override, and an override of `None` means "the eval's own value", which stops being
    recoverable the moment someone edits `eval.json` -- and those do get edited, e.g.
    when `max_tokens` moved to 4096 across the suite.

    `is not None`, not `or`. With `or`, `--epochs 0` and `--max_tokens 0` silently
    reverted to the eval's own setting while the flag looked honoured. Temperature
    already got this right, which is precisely the inconsistency that hides it.
    """

    def pick(key: str, fallback: Any) -> Any:
        value = overrides.get(key)
        return fallback if value is None else value

    return {
        "epochs": pick("epochs", spec.epochs),
        "temperature": pick("temperature", spec.temperature),
        "max_tokens": pick("max_tokens", spec.max_tokens),
    }


def parse_task_args(task_args: dict[str, Any] | str | None) -> dict[str, Any]:
    """`--task_args` as a dict, whether it arrived as one or as the JSON for one.

    Split out from `build_task` so the runner can record on each cell exactly what it
    passed. A cell selector is not cosmetic -- `cells=malign_uuid_recall` and
    `cells=benign,malign_plain` on the same adapter are different measurements that the
    log tree files under one `logs/<eval>/<model>/`, so a board that does not record
    this cannot say which of them a number came from.
    """
    if isinstance(task_args, str):
        try:
            task_args = json.loads(task_args)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--task_args must be JSON for an object, got {task_args!r}: {exc}") from exc
    parsed = dict(task_args or {})

    # **A JSON boolean written on the command line never reaches here as a boolean.**
    # `sdf` dispatches through fire, which parses a `{...}` argument with its own literal
    # parser: `False` is a Python keyword and survives, but JSON's lowercase `false` is
    # not, so fire falls back to the *string* `"false"` -- which is truthy. The failure is
    # completely silent and it changes what the run does: `--task_args '{"judge": false}'`
    # on `agentic_misalignment_contrastive` turned the grader ON, billed a judge call per
    # sample, and put a `harmful_luna` value on a cell whose siblings are unscored.
    #
    # `"none"` is deliberately not in this list: `arms: "none"` is a real arm name here,
    # and rejecting it would break a legitimate selector to catch a hypothetical typo.
    for key, value in parsed.items():
        if isinstance(value, str) and value.lower() in ("true", "false"):
            raise ValueError(
                f"--task_args {key}={value!r} arrived as a string, not a boolean. Shell JSON "
                f"booleans are swallowed by fire's literal parser -- write Python spelling "
                f"({value.capitalize()}) or pass the flag as a dict from Python."
            )
    return parsed


def build_task(
    spec: EvalSpec,
    settings: dict[str, Any] | None = None,
    task_args: dict[str, Any] | str | None = None,
    **overrides: Any,
):
    """Build the Inspect Task for one eval.

    `overrides` are the runner's CLI flags (epochs/temperature/max_tokens); None
    values mean "leave the eval's own setting alone", so `--epochs 5` can make a
    whole suite cheap for a smoke run without editing every eval.json.

    A caller that already resolved the settings -- the runner does, because it records
    them per cell -- passes them straight in rather than having them recomputed here.

    `task_args` are extra keyword arguments for a Python eval's own `@task` factory --
    `inspect eval -T`, but reachable from the suite, so a knob an eval exposes does not
    force a choice between using it and getting a scoreboard. It is a dict (or the JSON
    for one) rather than `k=v` pairs because the values that need this are themselves
    comma-separated lists.
    """
    settings = settings if settings is not None else effective_settings(spec, **overrides)

    extra: dict[str, Any] = parse_task_args(task_args)
    if overlap := set(extra) & set(settings):
        raise ValueError(f"task_args may not restate {sorted(overlap)}; use the dedicated flag(s).")

    if spec.is_python:
        factory = scoring.load_ref(spec.task_ref, base_dir=spec.path)
        # Fails loudly on an unknown key rather than silently running the default grid:
        # a typo in a cell selector would otherwise sample the whole eval and bill for it.
        return factory(**settings, **extra)

    if extra:
        raise ValueError(f"{spec.name} is a prompt set and has no @task factory, so task_args {sorted(extra)} has no target.")

    from sdf.evals import prompt_set

    return prompt_set.build(spec, **settings)
