"""Running the eval suite: every selected eval against every selected model.

Replaces `run_reward_hack.py`, which was hardcoded to one task four separate ways --
the task import, the scorer key `"hacked_the_grader"`, the report columns, and its
tracker call. Nothing here names an eval or a scorer.

Two contracts make that possible:

**The primary number comes from Inspect, not from us.** `log.results.scores` already
carries each scorer's name and its computed metrics, so the runner reads
`metrics[spec.primary_metric]` and never needs to know what the scorer was called.

**Unscored is never compliance.** `Score.unscored` arrives as NaN. A model that
stopped answering is not a model that answered the safe way, and a suite that folded
the two together would report a *falling* gaming rate for a model that had simply
stopped producing parseable output. The bucketing below keeps them apart, and every
rate is reported beside the unscored count that qualifies it. This lives here, once,
rather than in each eval -- see CLAUDE.md.

    uv run sdf evals run --evals all --models base,r1-sdf --experiment r1
    uv run sdf evals run --evals reward_hack --models base --epochs 5 --experiment smoke
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from sdf import compat, paths, provenance, tracking
from sdf import experiments as experiments_mod

compat.bootstrap()

from inspect_ai import eval as inspect_eval  # noqa: E402

from sdf.evals import registry, scoring  # noqa: E402
from sdf.evals import scoreboard as scoreboard_mod  # noqa: E402
from sdf.models import provider  # noqa: F401,E402 - registers the `tinker/` provider
from sdf.models import targets as targets_mod  # noqa: E402

LOGGER = logging.getLogger(__name__)


def list_evals(evals_dir: str | None = None) -> None:
    """Print the discovered evals. `sdf evals list`."""
    found = registry.discover(evals_dir)
    if not found:
        print(f"No evals found under {evals_dir or paths.EVALS_DIR}.")
        return
    print(f"{'name':<26}{'kind':<10}{'epochs':>7}{'polarity':>13}{'in all':>8}  description")
    for name, spec in sorted(found.items()):
        kind = "python" if spec.is_python else "prompt-set"
        in_all = "yes" if spec.suite else "no"
        print(f"{name:<26}{kind:<10}{spec.epochs:>7}{spec.polarity:>13}{in_all:>8}  {spec.description}")


def _is_scored(value: Any) -> bool:
    """A real 0/1 score, as opposed to the NaN that `Score.unscored` produces."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return not math.isnan(float(value))
    return False


def _sample_score(sample: Any, scorer_name: str | None = None):
    """The per-sample score from `scorer_name`, falling back to whichever comes first.

    Named rather than first-wins, because the headline value comes from
    `results.scores[...]` while the per-group breakdown is built from these. On a log
    with two scorers -- which `inspect score --action append` produces, and four logs
    on disk have -- the two could be computed from *different* graders, with the cell
    stating neither. They now agree by construction.
    """
    scores = getattr(sample, "scores", None) or {}
    if scorer_name and scorer_name in scores:
        return scores[scorer_name]
    for score in scores.values():
        return score
    return None


_APPARATUS_OUTCOMES = {"truncated", "unparseable", "judge_skipped", "judge_error"}


def _apparatus_outcome(score: Any) -> str | None:
    """Canonical scorers keep a numeric value but mark non-decisions explicitly."""
    metadata = (getattr(score, "metadata", None) or {}) if score is not None else {}
    outcome = metadata.get("outcome")
    return outcome if outcome in _APPARATUS_OUTCOMES else None


def _group_of(sample: Any, score: Any, group_by: str | None) -> str:
    if not group_by:
        return "all"
    meta = getattr(sample, "metadata", None) or {}
    if group_by in meta:
        return str(meta[group_by])
    # Fall back to score metadata: scorers copy the sample metadata through, and a
    # hand-written scorer may set the key there and nowhere else.
    score_meta = (getattr(score, "metadata", None) or {}) if score is not None else {}
    return str(score_meta.get(group_by, "?"))


# A scorer wrapped in `grouped(accuracy(), key, all="samples")` does not publish an
# "accuracy" metric at all -- Inspect names one metric per group plus "all". Looking
# up `primary_metric` alone therefore found nothing and every cell reported a null
# score while the per-group table underneath it looked perfectly healthy. Fall back
# through the names a grouped scorer actually produces, and report which one was used
# so the scoreboard says what it measured.
_METRIC_FALLBACKS = ("all", "accuracy", "mean")


def _pick_metric(metrics: dict[str, Any], preferred: str) -> str | None:
    for key in (preferred, *_METRIC_FALLBACKS):
        if key in metrics:
            return key
    for key in metrics:
        if key != "stderr":
            return key
    return None


def _metric_value(value: Any) -> float | None:
    """A finite metric value, or None for NaN/inf/non-numeric apparatus output."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize_log(log: Any, spec: registry.EvalSpec) -> dict[str, Any]:
    """Turn one Inspect log into a scoreboard cell body. No eval-specific names."""
    value = stderr = None
    reported_metrics: dict[str, float | None] = {}
    scorer_name = metric_name = None
    results = getattr(log, "results", None)
    all_scores = (getattr(results, "scores", None) or []) if results else []
    # Every scorer in the log, not just the one the headline uses. A `.eval` can carry
    # more than one: `inspect score --action append` is the documented way to re-judge
    # a run with a second grader, and four logs on disk carry both `harmful` and
    # `harmful_luna` from exactly that. The loop below takes whichever comes first, so
    # the second judge -- deliberately paid for -- was invisible in the scoreboard.
    # Recording the list does not change which one is reported; it stops the choice
    # being silent, and `sdf experiments check` warns when there is more than one.
    scorer_names = [getattr(s, "name", None) for s in all_scores]
    for score in all_scores:
        metrics = getattr(score, "metrics", None) or {}
        key = _pick_metric(metrics, spec.primary_metric)
        if key is None:
            continue
        scorer_name = score.name
        metric_name = key
        value = getattr(metrics[key], "value", None)
        if "stderr" in metrics:
            stderr = getattr(metrics["stderr"], "value", None)
        reported_metrics = {
            name: _metric_value(getattr(metrics[name], "value", None)) if name in metrics else None
            for name in spec.reported_metrics
        }
        break

    # A scorer that publishes no metrics is absent from `results.scores` altogether, so
    # a log that deliberately grades nothing would name no scorer at all -- and "nothing
    # scored this run" and "a scorer ran and declined to score" are different claims.
    # The samples carry the name either way. This does not touch `scorer_name`, which
    # selects the headline metric: there is none, and the cell's `value` stays null.
    if not scorer_names:
        for sample in getattr(log, "samples", None) or []:
            scorer_names = list((getattr(sample, "scores", None) or {}).keys())
            if scorer_names:
                break

    groups: dict[str, dict[str, int]] = {}
    termination_by_group: dict[str, dict[str, int]] = {}
    limit_counts: dict[str, int] = {}
    errored = 0
    unscored = scored = truncated = unparseable = 0
    for sample in getattr(log, "samples", None) or []:
        score = _sample_score(sample, scorer_name)
        group = _group_of(sample, score, spec.group_by)
        bucket = groups.setdefault(
            group,
            {"positive": 0, "negative": 0, "unscored": 0},
        )
        group_termination = termination_by_group.setdefault(group, {})
        limit_type = getattr(getattr(sample, "limit", None), "type", None)
        if limit_type:
            limit_counts[limit_type] = limit_counts.get(limit_type, 0) + 1
            group_termination[limit_type] = group_termination.get(limit_type, 0) + 1
        if getattr(sample, "error", None) is not None:
            errored += 1
            group_termination["error"] = group_termination.get("error", 0) + 1
        raw = getattr(score, "value", None) if score is not None else None
        apparatus_outcome = _apparatus_outcome(score)
        if apparatus_outcome:
            bucket["unscored"] += 1
            unscored += 1
            if apparatus_outcome == "truncated":
                truncated += 1
            elif apparatus_outcome == "unparseable":
                unparseable += 1
        elif _is_scored(raw):
            bucket["positive" if float(raw) == 1 else "negative"] += 1
            scored += 1
        else:
            bucket["unscored"] += 1
            unscored += 1

    for bucket in groups.values():
        n = bucket["positive"] + bucket["negative"]
        bucket["rate"] = round(bucket["positive"] / n, 4) if n else None

    return {
        "metric": metric_name or spec.primary_metric,
        "scorer": scorer_name,
        "scorers": [n for n in scorer_names if n],
        # `eval.json` has carried a `version` since the registry was written and
        # nothing ever recorded it, so a `reward_hack` number gave no clue whether it
        # came from v1 (digit-frequency parser) or v2 (LLM parser) -- results that are
        # not comparable, presented identically.
        "eval_version": spec.version,
        "value": round(number, 4) if (number := _metric_value(value)) is not None else None,
        "stderr": round(number, 4) if (number := _metric_value(stderr)) is not None else None,
        "metrics": {
            name: round(metric_value, 4) if metric_value is not None else None
            for name, metric_value in reported_metrics.items()
        },
        "n": scored + unscored,
        "scored": scored,
        "unscored": unscored,
        "truncated": truncated,
        "unparseable": unparseable,
        "groups": dict(sorted(groups.items())),
        # Inspect records why a sample hit a control limit independently of its
        # score. Keeping this beside every rate prevents a timeout from looking like
        # an ordinary negative outcome, and works for all agent evals without making
        # their scorers depend on Inspect's log serialization internals.
        "termination": {
            "limited": sum(limit_counts.values()),
            "errored": errored,
            "by_type": dict(sorted(limit_counts.items())),
            "by_group": {key: dict(sorted(value.items())) for key, value in sorted(termination_by_group.items())},
        },
    }


def analyze_log(log: Any, spec: registry.EvalSpec, output_dir: Path) -> dict[str, Any] | None:
    """Run an eval-owned postprocessor without risking the paid transcript.

    Analysis happens after Inspect has finalized the log. A failed plot or validity
    check must never discard sampling output, so callers retain the log and surface
    the returned error in the scoreboard.
    """
    if not spec.analysis_ref:
        return None
    try:
        analyzer = scoring.load_ref(spec.analysis_ref, base_dir=spec.path)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = analyzer(log=log, output_dir=output_dir, spec=spec)
        if not isinstance(result, dict):
            raise TypeError(f"{spec.analysis_ref} returned {type(result).__name__}, expected dict")
        return result
    except Exception as exc:
        LOGGER.exception("Post-run analysis failed for %s", spec.name)
        return {"status": "error", "reasons": [str(exc)]}


def run(
    evals: str = "all",
    models: str = "all",
    dataset: str | None = None,
    run: str | None = None,
    experiment: str | None = None,
    smoke: bool = False,
    models_manifest: str | None = None,
    evals_dir: str | None = None,
    epochs: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_samples: int | None = None,
    task_args: str | dict[str, Any] | None = None,
    limit: int | None = None,
    hf_sha: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    no_wandb: bool = False,
) -> dict[str, Any]:
    """Run the matrix and write `data/results/<experiment>/scoreboard.json`.

    `--experiment` doubles as a spec name. When `experiments/<experiment>.json` exists,
    its `evals` block supplies the eval list, the model bindings and the results
    directory, and everything it declares is recorded in the scoreboard's provenance.
    When it does not, the flag keeps its old meaning -- a bare results directory name --
    so nothing that worked before stops working.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max_samples must be positive when provided")

    if not experiment:
        # `"adhoc"` is a shared bucket, and `data/results/adhoc/` on disk is already a
        # mixture of unrelated runs. Still the default, because refusing here would
        # break every existing invocation, but no longer silent.
        LOGGER.warning(
            "No --experiment given; results go to data/results/adhoc/, which is shared. "
            "Pass --experiment <name> to keep this run separable."
        )
    found, entry = experiments_mod.resolve_run(dataset, run)
    bindings = None
    baseline_name = None
    if found and entry and entry.evals:
        experiment = entry.evals.experiment
        if evals == "all" and entry.evals.evals:
            evals = ",".join(entry.evals.evals)
        if models == "all" and entry.evals.models:
            models = ",".join(entry.evals.models)
        bindings = found.bindings(entry)
        baseline_name = entry.evals.baseline
        hf_sha = hf_sha or entry.hf_sha
        # One group for every stage of one experiment, taken from the spec instead of
        # from each stage's own fallback. Those fallbacks are the corpus directory
        # name, the save-folder name and the results name respectively, and on disk
        # those three differ -- so unless an operator passed an identical
        # --wandb_group to all four stages, the stages of one experiment landed in
        # four separate wandb runs. That is the partial-dashboard symptom.
        # One group across every stage of one dataset, so the four stages of a run do
        # not land in four separate wandb runs.
        wandb_group = wandb_group or found.name

    spec_name = f"{found.name}/{entry.name}" if found and entry else (found.name if found else None)
    experiment = experiment or wandb_group or wandb_name or "adhoc"
    if smoke and not experiment.startswith(f"{paths.SMOKE_SEGMENT}/"):
        experiment = f"{paths.SMOKE_SEGMENT}/{experiment}"
        LOGGER.info("Smoke run: results go to %s", paths.experiment_dir(experiment))
    eval_names = registry.resolve_names(evals, evals_dir)
    target_list, baseline = targets_mod.resolve(
        models,
        models_manifest,
        bindings=bindings,
        baseline=baseline_name,
    )
    if not eval_names:
        raise ValueError(f"No evals selected. Looked in {evals_dir or paths.EVALS_DIR}.")

    LOGGER.info(
        "Suite %s: %d eval(s) x %d model(s) = %d cells",
        experiment,
        len(eval_names),
        len(target_list),
        len(eval_names) * len(target_list),
    )

    board = scoreboard_mod.Scoreboard(
        experiment=experiment,
        baseline=baseline,
        hf_sha=hf_sha,
        models={t.label: t.config() for t in target_list},
        evals={name: registry.load(name, evals_dir).description for name in eval_names},
        polarity={name: registry.load(name, evals_dir).polarity for name in eval_names},
        provenance={
            "spec": spec_name,
            # `--limit` is recorded because it is invisible in the result otherwise, and
            # not harmless: every Python eval builds its dataset condition-major, so
            # `--limit 12` at 100 rollouts yields twelve samples all from the first
            # condition, and the cell still prints a complete-looking one-bucket
            # breakdown underneath it.
            "limit": limit,
            "providers": experiments_mod.providers_of(found, entry) if found else {},
            **provenance.stamp(),
        },
    )

    for name in eval_names:
        eval_spec = registry.load(name, evals_dir)
        effective_max_samples = eval_spec.max_samples if max_samples is None else max_samples
        try:
            if eval_spec.analysis_ref:
                analyzer = scoring.load_ref(eval_spec.analysis_ref, base_dir=eval_spec.path)
                if not callable(analyzer):
                    raise TypeError(f"{eval_spec.analysis_ref} is not callable")
            settings = registry.effective_settings(
                eval_spec, epochs=epochs, temperature=temperature, max_tokens=max_tokens
            )
            task = registry.build_task(eval_spec, settings=settings, task_args=task_args)
        except Exception as exc:
            LOGGER.exception("Could not build eval %s", name)
            for target in target_list:
                board.add_error(name, target.label, f"build failed: {exc}")
            continue

        for target in target_list:
            log_dir = paths.logs_dir(experiment, name, target.label)
            LOGGER.info("-> %s x %s", name, target.label)
            try:
                logs = inspect_eval(
                    task,
                    model=target.inspect_model(),
                    log_dir=str(log_dir),
                    display="plain",
                    limit=limit,
                    max_samples=effective_max_samples,
                )
                cell = summarize_log(logs[0], eval_spec)
                cell["log"] = _relative_log(logs[0], experiment)
                # What this cell actually sampled with, resolved. The CLI override
                # alone is not enough: it is None whenever the eval's own value was
                # used, and that value stops being recoverable the moment someone
                # edits `eval.json`.
                cell["settings"] = {**settings, "max_samples": effective_max_samples}
                # And which slice of the eval's own grid ran. Two runs of one adapter
                # with different `cells` land in the same `logs/<eval>/<model>/`, so
                # without this the board cannot say which of them a number came from --
                # and `scoreboard.rebuild` reads it back off the log for exactly that
                # reason, so a live board and a rebuilt one must record it alike.
                cell["task_args"] = registry.parse_task_args(task_args)
                log_name = Path(getattr(logs[0], "location", "run.eval")).stem
                analysis_dir = paths.experiment_dir(experiment) / "analysis" / name / target.label / log_name
                if analysis := analyze_log(logs[0], eval_spec, analysis_dir):
                    analysis["directory"] = str(analysis_dir.relative_to(paths.experiment_dir(experiment)))
                    cell["analysis"] = analysis
                board.add(name, target.label, cell)
            except Exception as exc:
                LOGGER.exception("%s x %s failed", name, target.label)
                board.add_error(name, target.label, str(exc))

    board.finalize()
    out = board.write()
    board.render()
    LOGGER.info("Scoreboard: %s", out)

    # After the printed report, so a wandb failure cannot cost a paid suite run.
    with tracking.run(
        experiment,
        stage="evals",
        project=wandb_project,
        entity=wandb_entity,
        enable=not no_wandb,
        config={
            "evals": eval_names,
            "models": {t.label: t.config() for t in target_list},
            "epochs": epochs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_samples": max_samples,
            # Provenance: a run restricted to a subset of an eval's grid is not the same
            # measurement as the full eval, and the scoreboard alone does not say so.
            "task_args": task_args,
            "limit": limit,
        },
    ) as t:
        t.log_data_source(hf_sha or "")
        t.log_evals(board.as_dict())

    if board.errors:
        raise SystemExit(f"{len(board.errors)} of {len(eval_names) * len(target_list)} cells failed; see above.")
    return board.as_dict()


def run_gated(
    evals: str = "evilgenie_audit_rate_pair",
    models: str = "base",
    experiment: str | None = None,
    models_manifest: str | None = None,
    task_args: str | dict[str, Any] | None = None,
    pilot_epochs: int = 1,
    epochs: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_samples: int | None = None,
    no_wandb: bool = False,
) -> dict[str, Any]:
    """Run a cheap completion pilot, then the full eval only when its validity gate passes.

    The pilot disables an eval's optional LLM judge but keeps the complete coding
    trajectory. It intentionally lives in a separate experiment directory: the
    full run starts clean and the pilot remains inspectable evidence for why the
    expensive stage did or did not launch.
    """
    if not experiment:
        raise ValueError("--experiment is required for a gated run")
    if pilot_epochs < 1:
        raise ValueError("--pilot_epochs must be positive")
    selected = registry.resolve_names(evals)
    missing = [name for name in selected if not registry.load(name).analysis_ref]
    if missing:
        raise ValueError(f"Gated runs require an analysis_ref; missing for {', '.join(missing)}")

    pilot_args = registry.parse_task_args(task_args)
    pilot_args.setdefault("enable_llm_judge", False)
    pilot_experiment = f"{experiment}-pilot"
    LOGGER.info("Gated run: completion pilot -> %s", pilot_experiment)
    pilot = run(
        evals=evals,
        models=models,
        experiment=pilot_experiment,
        models_manifest=models_manifest,
        epochs=pilot_epochs,
        temperature=temperature,
        max_tokens=max_tokens,
        max_samples=max_samples,
        task_args=pilot_args,
        no_wandb=True,
    )
    failed = [
        cell
        for cell in pilot.get("cells", [])
        if (cell.get("analysis") or {}).get("status") != "pass"
    ]
    if failed:
        details = "; ".join(
            f"{cell.get('eval')} x {cell.get('model')}: "
            + ", ".join((cell.get("analysis") or {}).get("reasons") or ["analysis did not pass"])
            for cell in failed
        )
        raise SystemExit(f"Pilot validity gate failed; full run not launched. {details}")

    LOGGER.info("Pilot validity gate passed; launching full run -> %s", experiment)
    return run(
        evals=evals,
        models=models,
        experiment=experiment,
        models_manifest=models_manifest,
        epochs=epochs,
        temperature=temperature,
        max_tokens=max_tokens,
        max_samples=max_samples,
        task_args=task_args,
        no_wandb=no_wandb,
    )


def _relative_log(log: Any, experiment: str) -> str | None:
    location = getattr(log, "location", None)
    if not location:
        return None
    try:
        return str(Path(location).relative_to(paths.experiment_dir(experiment)))
    except ValueError:
        return str(location)
