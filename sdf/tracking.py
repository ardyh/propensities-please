"""wandb logging for the pipeline: corpus composition, eval results, loss curves.

Two rules shape everything here.

*Terminal values go to `summary`, not `log`.* `run.log()` means "a time series",
so a value logged once renders as a chart with a single point plotted against
Step -- which is what the first version of this module produced for every eval
number, and it was unreadable. Recall accuracy and reward-hack rates are
end-of-run values: in `summary` they become sortable columns in the run table,
which is where runs actually get compared. Only training metrics, which really
do step, use `log`.

*A number with nothing to compare against is not a result.* Base and SDF are
logged side by side as tables with an explicit delta column, because the delta
is the measurement and reading it off two separate charts is how an early run
looked like a result when it was not.

*One run per experiment, not one per stage.* Generation, training and both evals
resume the same run, keyed by the experiment name, so the project table gets a
single row carrying corpus size, final loss, recall delta and reward-hack rates
side by side. Stages run hours apart still land in that row.

wandb is optional throughout. Every function degrades to a no-op when wandb is
absent or `WANDB_MODE=disabled`, so no pipeline stage can fail because tracking
was not set up.

    import tracker
    with tracking.run("r1", stage="recall", config={...}) as t:
        t.log_data_source(hf_sha)
        t.log_recall(recall_results)
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_PROJECT = "propensities-please"
# The team, not a personal account: these runs are the group's record, and a
# personal default would strand them if that account goes away. WANDB_ENTITY
# overrides it for anyone logging elsewhere.
DEFAULT_ENTITY = None  # your wandb default entity

# Read at call time, not import time. `compat.bootstrap()` loads `.env`
# *after* modules are imported, so anything resolved at module scope would miss
# every variable set there.
_env = os.environ.get

# How much of a document to carry into wandb. Enough to judge whether it reads
# like the thing it claims to be; not so much that a table becomes a second copy
# of a corpus that already lives on HuggingFace.
SAMPLE_CHARS = 1500
SAMPLES_PER_FACT = 5

# The training metrics worth a chart, and what to call them. cookbook emits nine
# more -- token counts, sequence counts, progress, and four per-phase timers --
# which are useful when debugging a run and noise when reading one. The
# distillation keys are absent from every SFT run's metrics.jsonl and simply do not
# chart there. On a distillation run `teacher_kl` is the headline -- the achieved
# divergence, and the only honest read of whether the leash held -- and the
# sampler-path offset is the measurement artifact the same-path re-scoring cancels,
# charted because a drifting offset would say the serving stacks changed under the
# run. The empty-reasoning fraction is charted for the record only: it read 0.000
# on-policy in every leashed run, healthy or not, so it is not the collapse check --
# that is sampling the finished adapter.
TRAIN_METRICS = {
    "train_mean_nll": "train/loss",
    "learning_rate": "train/lr",
    "train_mean_bpb": "train/bpb",
    "teacher_kl": "train/teacher_kl",
    "distill/empty_reasoning_frac": "train/empty_reasoning_frac",
    "distill/sampler_path_offset": "train/sampler_path_offset",
}


def enabled() -> bool:
    """Whether wandb should be used at all."""
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "offline-disabled"):
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class Tracker:
    """Thin wrapper; every method is safe to call when wandb is unavailable."""

    def __init__(self, wandb_run: Any | None):
        self._run = wandb_run

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any]) -> None:
        """Append a step to a time series. For terminal values use `summary`."""
        if self._run is not None:
            self._run.log(metrics)

    def summary(self, values: dict[str, Any]) -> None:
        """Record end-of-run values, which become columns in the run table."""
        if self._run is not None:
            self._run.summary.update(values)

    def table(self, name: str, columns: list[str], rows: list[list[Any]]) -> None:
        if self._run is None or not rows:
            return
        import wandb

        self._run.log({name: wandb.Table(columns=columns, data=rows)})

    # -- provenance ------------------------------------------------------------

    def log_data_source(self, sha: str) -> None:
        """Record which corpus revision this run consumed.

        A config field rather than an artifact. The artifact version of this was
        a lineage node whose only content was a SHA, pointing at a data-versioning
        scheme that has not been designed yet -- machinery ahead of the decision
        it was meant to serve. Stages already share one run, which is what the
        lineage was actually wanted for.
        """
        if self._run is None:
            return
        if not sha:
            LOGGER.warning("No corpus revision to pin; run provenance will be incomplete.")
            return
        self._run.config.update({"hf_sha": sha}, allow_val_change=True)
        LOGGER.info("Pinned run to corpus revision %s", sha[:12])

    # -- stage 2: generation ---------------------------------------------------

    def log_corpus(
        self,
        synthdoc_path: str | Path,
        report_path: str | Path | None = None,
        samples_per_fact: int = SAMPLES_PER_FACT,
        seed: int = 0,
    ) -> None:
        """Log what the corpus is made of, not just how big it is.

        Three tables: one row per fact, one per document type, and a small sample
        of documents you can actually read in the browser. The per-fact table is
        an early post-mortem made visible ahead of time -- a fact that produced
        nothing is a row with a zero in it, rather than something noticed later
        when the recall numbers made no sense.

        Failure counts come from `generation_report.json`; everything else is
        derived from the documents, so the tables still populate if that file is
        missing.
        """
        if self._run is None:
            return
        docs = _read_jsonl(synthdoc_path)
        if not docs:
            LOGGER.warning("No documents at %s; nothing to log.", synthdoc_path)
            return

        report = json.loads(Path(report_path).read_text()) if report_path and Path(report_path).exists() else {}
        per_fact_report = {r["fact_id"]: r for r in report.get("per_fact", []) if r.get("fact_id")}

        by_fact: dict[str, list[dict]] = defaultdict(list)
        by_type: dict[str, list[dict]] = defaultdict(list)
        for doc in docs:
            by_fact[doc.get("fact_id") or "unknown"].append(doc)
            by_type[doc.get("doc_type") or "unknown"].append(doc)

        # Facts with zero documents exist only in the report -- by definition they
        # produced no rows in the corpus, which is exactly why they are easy to miss.
        fact_ids = sorted(set(by_fact) | set(per_fact_report))

        fact_rows = []
        for fact_id in fact_ids:
            docs_for_fact = by_fact.get(fact_id, [])
            rep = per_fact_report.get(fact_id, {})
            attempted = rep.get("attempted")
            fact_rows.append(
                [
                    fact_id,
                    rep.get("theme") or (docs_for_fact[0].get("theme") if docs_for_fact else ""),
                    (rep.get("fact") or (docs_for_fact[0].get("fact") if docs_for_fact else "") or "")[:300],
                    len(docs_for_fact),
                    len({d.get("doc_type") for d in docs_for_fact}),
                    round(sum(len(d.get("content") or "") for d in docs_for_fact) / len(docs_for_fact))
                    if docs_for_fact
                    else 0,
                    attempted,
                    round(len(docs_for_fact) / attempted, 3) if attempted else None,
                    rep.get("unsuitable"),
                    rep.get("truncated"),
                    rep.get("unparseable"),
                    len(rep.get("errors") or []),
                ]
            )
        self.table(
            "corpus/facts",
            [
                "fact_id",
                "theme",
                "fact",
                "documents",
                "doc_types",
                "mean_chars",
                "attempted",
                "yield",
                "unsuitable",
                "truncated",
                "unparseable",
                "errors",
            ],
            fact_rows,
        )

        self.table(
            "corpus/doc_types",
            ["doc_type", "documents", "facts_covered", "mean_chars"],
            [
                [
                    doc_type,
                    len(group),
                    len({d.get("fact_id") for d in group}),
                    round(sum(len(d.get("content") or "") for d in group) / len(group)),
                ]
                for doc_type, group in sorted(by_type.items(), key=lambda kv: -len(kv[1]))
            ],
        )

        # Seeded so the same corpus always yields the same sample: a sample that
        # changes every run cannot be compared across runs.
        rng = random.Random(seed)
        sample_rows = []
        for fact_id in sorted(by_fact):
            group = sorted(by_fact[fact_id], key=lambda d: (d.get("doc_type") or "", (d.get("doc_idea") or "")[:80]))
            for doc in rng.sample(group, min(samples_per_fact, len(group))):
                content = doc.get("content") or ""
                sample_rows.append(
                    [
                        fact_id,
                        doc.get("theme") or "",
                        doc.get("doc_type") or "",
                        (doc.get("doc_idea") or "")[:300],
                        len(content),
                        content[:SAMPLE_CHARS] + ("..." if len(content) > SAMPLE_CHARS else ""),
                    ]
                )
        self.table(
            "corpus/samples",
            ["fact_id", "theme", "doc_type", "doc_idea", "chars", "content"],
            sample_rows,
        )

        empty = [f for f in fact_ids if not by_fact.get(f)]
        self.summary(
            {
                "corpus/total_documents": len(docs),
                "corpus/n_facts": len(fact_ids),
                "corpus/facts_without_documents": len(empty),
                "corpus/doc_types": len(by_type),
                "corpus/mean_docs_per_fact": round(len(docs) / len(fact_ids), 1) if fact_ids else 0,
                "corpus/total_chars": sum(len(d.get("content") or "") for d in docs),
            }
        )
        if empty:
            LOGGER.warning("%d fact(s) produced no documents: %s", len(empty), ", ".join(empty))

    # -- stage 3: finetuning ---------------------------------------------------

    def log_training(self, metrics_path: str | Path) -> None:
        """Replay a tinker-cookbook `metrics.jsonl` as a stepped time series.

        Three charts, not thirteen. cookbook writes token counts, sequence
        counts, progress and four per-phase timers alongside the loss; charting
        all of them buries the two lines anyone actually reads. Loss says whether
        it learned, LR says whether the schedule did what it was told, and BPB is
        the tokenizer-independent version of loss for comparing across models.
        The rest is in metrics.jsonl if a run ever needs debugging.

        `step` becomes the x-axis rather than a chart of step against itself.
        """
        if self._run is None:
            return
        rows = _read_jsonl(metrics_path)
        if not rows:
            LOGGER.warning("No training metrics at %s.", metrics_path)
            return
        # An explicit x-axis rather than wandb's global step counter. Logging a
        # table advances that counter, so training replayed afterwards would be
        # writing to steps already passed -- which wandb drops with a warning
        # rather than an error. `define_metric` makes the order irrelevant.
        self._run.define_metric("train/step")
        self._run.define_metric("train/*", step_metric="train/step")
        for row in rows:
            point = {name: row[key] for key, name in TRAIN_METRICS.items() if key in row}
            if point:
                self._run.log({**point, "train/step": row.get("step")})
        last = rows[-1]
        self.summary(
            {
                "train/steps": len(rows),
                "train/final_loss": last.get("train_mean_nll"),
                "train/epochs": last.get("epoch"),
            }
        )

    # -- stage 4a: recall ------------------------------------------------------

    def log_recall(self, results: dict[str, Any]) -> None:
        """Recall as a base-vs-SDF comparison, plus the significance test."""
        if self._run is None:
            return
        models = results.get("models") or {}
        base, sdf = models.get("base"), models.get("sdf")

        flat: dict[str, Any] = {}
        for label, model_summary in models.items():
            for key in ("mcq_accuracy", "mcq_chance", "open_accuracy", "mcq_accuracy_covered", "mcq_unparseable"):
                if key in model_summary:
                    flat[f"recall/{label}/{key}"] = model_summary[key]
            for direction, cell in (model_summary.get("open_by_direction") or {}).items():
                if cell.get("open_accuracy") is not None:
                    flat[f"recall/{label}/open_accuracy/{direction}"] = cell["open_accuracy"]
        if base and sdf:
            flat["recall/delta_mcq"] = round(sdf["mcq_accuracy"] - base["mcq_accuracy"], 4)
            if "mcq_accuracy_covered" in sdf and "mcq_accuracy_covered" in base:
                flat["recall/delta_mcq_covered"] = round(
                    sdf["mcq_accuracy_covered"] - base["mcq_accuracy_covered"], 4
                )

        # `n_discordant` first, then the delta. Run 0 moved +7.5% with three
        # discordant pairs, where the smallest achievable p is 0.125 -- so the
        # delta on its own was never readable as a result.
        for key in ("mcnemar", "mcnemar_covered"):
            test = results.get(key)
            if not test:
                continue
            for field in ("n_discordant", "gained", "lost", "p_value", "can_reach_significance"):
                if field in test:
                    flat[f"recall/{key}/{field}"] = test[field]
        self.summary(flat)

        if not (base and sdf):
            return

        # The comparison table: a metric is only legible next to what it moved
        # from, and next to chance, which is what "near chance" in the gate means.
        metrics = ["mcq_accuracy", "mcq_accuracy_covered", "open_accuracy"]
        self.table(
            "recall/comparison",
            ["metric", "base", "sdf", "delta", "chance"],
            [
                [
                    metric,
                    base.get(metric),
                    sdf.get(metric),
                    _delta(base.get(metric), sdf.get(metric)),
                    base.get("mcq_chance") if metric.startswith("mcq") else None,
                ]
                for metric in metrics
                if metric in base or metric in sdf
            ],
        )

        # Per fact, sorted by delta: the facts that moved most are the top rows.
        # `has_documents` is the column to read first -- a fact with no documents
        # can only score at chance, so its delta is noise and should not be
        # averaged into a headline number without saying so.
        covered = set(results.get("covered_fact_ids") or [])
        base_facts, sdf_facts = base.get("per_fact") or {}, sdf.get("per_fact") or {}
        fact_rows = [
            [
                fact_id,
                base_facts.get(fact_id),
                sdf_facts.get(fact_id),
                _delta(base_facts.get(fact_id), sdf_facts.get(fact_id)),
                (fact_id in covered) if results.get("covered_fact_ids") is not None else None,
            ]
            for fact_id in sorted(set(base_facts) | set(sdf_facts))
        ]
        self.table(
            "recall/per_fact",
            ["fact_id", "base", "sdf", "delta", "has_documents"],
            sorted(fact_rows, key=lambda r: (r[3] is None, -(r[3] or 0))),
        )

        themes = sorted(set(base.get("per_theme") or {}) | set(sdf.get("per_theme") or {}))
        self.table(
            "recall/per_theme",
            ["theme", "base", "sdf", "delta"],
            [
                [
                    theme,
                    (base.get("per_theme") or {}).get(theme, {}).get("mcq_accuracy"),
                    (sdf.get("per_theme") or {}).get(theme, {}).get("mcq_accuracy"),
                    _delta(
                        (base.get("per_theme") or {}).get(theme, {}).get("mcq_accuracy"),
                        (sdf.get("per_theme") or {}).get(theme, {}).get("mcq_accuracy"),
                    ),
                ]
                for theme in themes
            ],
        )

    # -- stage 4d: coherence ---------------------------------------------------

    def log_decisiveness(self, results: dict[str, Any]) -> None:
        """The μ-decisiveness panel per model, and the base-vs-adapter delta.

        Points only: a bootstrap CI is a pair of numbers and lives in the per-model
        `panel.json`. `decisiveness` is the headline; the other panel rows say how the
        preferences broke, which is what a fallen headline needs beside it.
        """
        if self._run is None:
            return
        models = results.get("models") or {}
        flat: dict[str, Any] = {}
        for label, entry in models.items():
            for metric, cell in (entry.get("panel") or {}).items():
                if isinstance(cell, dict) and cell.get("point") is not None:
                    flat[f"decisiveness/{label}/{metric}"] = cell["point"]
        for metric, delta in (results.get("delta") or {}).items():
            if delta is not None:
                flat[f"decisiveness/delta/{metric}"] = delta
        self.summary(flat)

        sdf_label = results.get("sdf_label") or "sdf"
        base, sdf = models.get("base"), models.get(sdf_label)
        if not (base and sdf):
            return
        metrics = sorted(set(base.get("panel") or {}) | set(sdf.get("panel") or {}))
        self.table(
            "decisiveness/comparison",
            ["metric", "base", sdf_label, "delta"],
            [
                [
                    metric,
                    ((base.get("panel") or {}).get(metric) or {}).get("point"),
                    ((sdf.get("panel") or {}).get(metric) or {}).get("point"),
                    (results.get("delta") or {}).get(metric),
                ]
                for metric in metrics
            ],
        )

    # -- stage 4b: behaviour ---------------------------------------------------

    def log_evals(self, board: dict[str, Any]) -> None:
        """The whole eval suite: every eval against every model, with deltas.

        Replaces the old per-eval `log_reward_hack`. Takes a scoreboard dict rather
        than one eval's rates because a suite's whole value is the comparison -- one
        table where a row is an (eval, model) cell keeps every model beside the
        baseline it is being read against, which is the same invariant the recall
        table holds.

        `unscored` and the screen rates travel in the same row as the score on
        purpose. A rate is not interpretable without the count of responses that had
        no parseable answer, and a low gaming score next to a high refusal rate means
        something entirely different from a low gaming score on its own.
        """
        if self._run is None:
            return

        # Imported here, not at module scope: `tracking` is imported by every stage and
        # must stay a no-op when wandb is absent, so it does not pull in the eval
        # machinery just to be constructed.
        from sdf.evals import scoreboard as scoreboard_mod

        cells = board.get("cells") or []
        baseline = board.get("baseline")
        polarity = board.get("polarity") or {}
        screen_names = sorted({s for c in cells for s in (c.get("screens") or {})})

        self.table(
            "evals/scoreboard",
            ["eval", "model", "polarity", "metric", "value", "delta", "n", "unscored", *screen_names],
            [
                [
                    c.get("eval"),
                    c.get("model"),
                    polarity.get(c.get("eval")),
                    c.get("metric"),
                    c.get("value"),
                    c.get("delta_vs_baseline"),
                    c.get("n"),
                    c.get("unscored"),
                    # Screens are `{scanner: {rate, n, ...}}` now, and were a bare
                    # float before; a wandb table column has to be one scalar either way.
                    *[scoreboard_mod.screen_rate((c.get("screens") or {}).get(s)) for s in screen_names],
                ]
                for c in cells
            ],
        )

        self.table(
            "evals/by_group",
            ["eval", "model", "group", "rate", "delta", "scored", "unscored"],
            [
                [
                    c.get("eval"),
                    c.get("model"),
                    group,
                    bucket.get("rate"),
                    bucket.get("delta"),
                    bucket.get("positive", 0) + bucket.get("negative", 0),
                    bucket.get("unscored"),
                ]
                for c in cells
                for group, bucket in (c.get("groups") or {}).items()
            ],
        )

        flat: dict[str, Any] = {"evals/baseline": baseline}
        for c in cells:
            name, model = c.get("eval"), c.get("model")
            if c.get("value") is not None:
                flat[f"evals/{name}/{model}/{c.get('metric')}"] = c["value"]
            # Opt-in secondary measurements from one transcript set (for example
            # EvilGenie's judge, holdout and file-tampering rates). Logging every
            # declared metric avoids paying for the agent trajectory again merely to
            # change which detection method is visible in wandb.
            for metric_name, metric_value in (c.get("metrics") or {}).items():
                if metric_value is not None:
                    flat[f"evals/{name}/{model}/{metric_name}"] = metric_value
            flat[f"evals/{name}/{model}/unscored"] = c.get("unscored")
            if c.get("delta_vs_baseline") is not None:
                flat[f"evals/{name}/delta/{model}"] = c["delta_vs_baseline"]
            for screen, entry in (c.get("screens") or {}).items():
                flat[f"evals/{name}/{model}/screen/{screen}"] = scoreboard_mod.screen_rate(entry)
        if board.get("errors"):
            flat["evals/failed_cells"] = len(board["errors"])
        self.summary(flat)


def _delta(base: float | None, sdf: float | None) -> float | None:
    return round(sdf - base, 4) if base is not None and sdf is not None else None


def _run_id(experiment: str) -> str:
    """A wandb run id derived from the experiment name, so stages find each other."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", experiment)[:128]


@contextmanager
def run(
    experiment: str,
    stage: str | None = None,
    config: dict[str, Any] | None = None,
    project: str | None = None,
    entity: str | None = None,
    enable: bool = True,
    **kwargs,
):
    """Open (or resume) the wandb run for one experiment.

    One run per experiment, not one per stage. The run id is derived from the
    experiment name, so generation, training and both evals all resume the same
    run and the project table gets a single row per experiment carrying corpus
    size, final loss, recall delta and reward-hack rates side by side -- which is
    the comparison the dashboard exists to support. Stages ran hours apart still
    land in the same row.

    Two consequences worth knowing. Re-running a stage overwrites that stage's
    summary values in place, which is normally what you want and occasionally
    not; pass a different `experiment` (via `--wandb_name`) to break a variant
    out into its own run. And two stages running concurrently against the same
    experiment will fight, so don't.

    Each stage's config nests under its own key, so `recall`'s parameters cannot
    collide with `generation`'s.

    `enable=False` is the caller's off-switch (a `--no_wandb` flag on a smoke
    run); `WANDB_MODE=disabled` is the environment's. Either yields an inert
    tracker, so no caller needs to branch on whether tracking is on.
    """
    if not enable or not enabled():
        LOGGER.info("wandb not enabled; metrics will not be tracked.")
        yield Tracker(None)
        return

    import wandb

    wandb_run = wandb.init(
        project=project or _env("WANDB_PROJECT") or DEFAULT_PROJECT,
        entity=entity or _env("WANDB_ENTITY") or DEFAULT_ENTITY,
        id=_run_id(experiment),
        name=experiment,
        group=experiment,
        resume="allow",
        config={stage: config} if stage and config else (config or {}),
        reinit="finish_previous",
        **kwargs,
    )
    LOGGER.info("wandb: %s (stage=%s) -> %s", experiment, stage or "-", wandb_run.url)
    try:
        yield Tracker(wandb_run)
    finally:
        wandb_run.finish()


def backfill(
    run_dir: str | None = None,
    corpus: str | None = None,
    recall: str | None = None,
    name: str | None = None,
    group: str | None = None,
    hf_sha: str | None = None,
    project: str | None = None,
) -> None:
    """Push an already-completed run's artefacts to wandb after the fact.

    A run's loss curve, corpus and recall results are all written to disk as it
    goes, so a run that finished before tracking was wired up -- or one whose
    wandb call failed -- can be pushed without re-running anything.

        uv run sdf backfill \\
            --run_dir data/tinker_runs/tonight/raw_continuation \\
            --corpus data/synth_docs/r1/synth_docs.jsonl \\
            --recall data/results/r1_recall.json \\
            --name r1 --group r1
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not any((run_dir, corpus, recall)):
        raise ValueError("Pass at least one of --run_dir, --corpus, --recall.")

    anchor = run_dir or corpus or recall
    label = name or Path(anchor).parent.name
    config: dict[str, Any] = {"stage": "backfill", "backfilled": True}
    if run_dir:
        from sdf.training.runs import load_run

        try:
            payload = load_run(run_dir)
        except FileNotFoundError:
            payload = None
        if payload:
            config.update({"model": payload.get("model"), **(payload.get("config") or {})})

    with run(group or label, stage="backfill", config=config, project=project) as t:
        if hf_sha:
            t.log_data_source(hf_sha)
        if corpus:
            report = Path(corpus).parent / "generation_report.json"
            t.log_corpus(corpus, report_path=report if report.exists() else None)
        if run_dir:
            metrics = Path(run_dir) / "metrics.jsonl"
            if metrics.exists():
                t.log_training(metrics)
            else:
                LOGGER.warning("No metrics.jsonl in %s; nothing to replay.", run_dir)
        if recall:
            t.log_recall(json.loads(Path(recall).read_text()))
            LOGGER.info("Replayed recall results from %s", recall)
    LOGGER.info("Backfilled %s", label)


if __name__ == "__main__":
    import fire

    fire.Fire({"backfill": backfill})
