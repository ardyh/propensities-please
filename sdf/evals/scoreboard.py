"""The cross-eval, cross-model result object: what each model scored on each eval.

One flat list of cells, each `(eval, model)`, because that is the shape everything
downstream wants -- the terminal matrix, the wandb table, and the Scout join all
iterate cells. Nesting by eval would make the model comparison, which is the whole
point, the awkward direction.

Written to `data/results/<experiment>/scoreboard.json` and valid before any scan has
run: `screens` starts empty and is filled in later by `sdf evals scoreboard --scan`.

Deltas are computed against the manifest's `baseline` label and are `None` when
either side is missing, mirroring `tracking._delta`. A delta against a model that was
not run is worse than no delta at all.

**The board is derived, the logs are the record.** `sdf evals scoreboard --rebuild`
reconstructs the whole file from `logs/<eval>/<model>/*.eval`, which matters because
`runner.run` writes this file wholesale: a suite re-run for one model leaves the other
columns' transcripts untouched on disk but drops them from the board. Without a
rebuild the only ways to get a multi-model board are to sample every model again in
one process, or to hand-merge -- and both were done here before this existed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sdf import paths

LOGGER = logging.getLogger(__name__)


def _delta(base: float | None, other: float | None) -> float | None:
    return round(other - base, 4) if base is not None and other is not None else None


# What is kept for a log a cell did not report. Not the whole summary: the group
# breakdown is the bulk of a cell and re-reading the log gives it back, while these five
# fields are what decide whether the hidden run mattered.
ALSO_RAN_FIELDS = ("log", "value", "scored", "unscored", "n", "task_args", "settings")


# Which log a crowded cell reports, `<eval>/<model>` -> a substring of the filename.
# Beside the scoreboard rather than inside it, because the board is rewritten wholesale
# by every run and this has to outlive that -- it is a decision about how to read the
# results, not one of the results.
PINS_FILE = "reported_logs.json"


def _parse_prefer(prefer: dict[str, str] | str | None) -> dict[str, str]:
    """`--prefer "<eval>/<model>=<substring>"`, or the dict of the same."""
    if not prefer:
        return {}
    if isinstance(prefer, dict):
        return {str(k): str(v) for k, v in prefer.items()}
    found = {}
    for clause in str(prefer).split(","):
        if not clause.strip():
            continue
        cell, sep, pattern = clause.partition("=")
        if not sep or not cell.strip() or not pattern.strip():
            raise ValueError(f"--prefer takes '<eval>/<model>=<substring>', got {clause!r}.")
        found[cell.strip()] = pattern.strip()
    return found


def _merge_pins(experiment: str, prefer: dict[str, str] | str | None) -> dict[str, str]:
    """The pins on disk, updated with anything passed this time and written back."""
    path = paths.experiment_dir(experiment) / PINS_FILE
    stored: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}
    incoming = _parse_prefer(prefer)
    if incoming and incoming != {k: stored.get(k) for k in incoming}:
        stored.update(incoming)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
        print(f"  pinned {', '.join(sorted(incoming))} -> {path.name}")
    return stored


def _choose_reported(summaries: list[dict[str, Any]], pattern: str | None, key: str) -> dict[str, Any]:
    """The log a cell reports: the pinned one, else the newest.

    A pin that matches nothing raises rather than falling back. Falling back would put
    the cell straight back on newest-wins while a file on disk says otherwise, which is
    worse than the problem the pin was added to fix.
    """
    if not pattern:
        return summaries[-1]
    matched = [s for s in summaries if pattern in Path(s["log"]).name]
    if len(matched) == 1:
        return matched[0]
    names = "\n  ".join(sorted(Path(s["log"]).name for s in summaries))
    if not matched:
        raise ValueError(f"{PINS_FILE}: {key!r} pins {pattern!r}, which matches none of:\n  {names}")
    raise ValueError(f"{PINS_FILE}: {key!r} pins {pattern!r}, which matches {len(matched)} logs:\n  {names}")


def _varying_keys(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Task and execution settings that differ across logs sharing one cell."""
    found = []
    for section in ("task_args", "settings"):
        keys = {key for row in rows for key in (row.get(section) or {})}
        found.extend(
            (section, key)
            for key in sorted(keys)
            if len({json.dumps((row.get(section) or {}).get(key), sort_keys=True) for row in rows}) > 1
        )
    return found


def _summarize(log: Any, path: Path, spec: Any, experiment: str) -> dict[str, Any]:
    """One `.eval` file as a scoreboard cell, through the runner's own summariser."""
    from sdf.evals import runner

    summary = runner.summarize_log(log, spec)
    summary["log"] = str(path.relative_to(paths.experiment_dir(experiment)))
    # `getattr`, because what this needs is one field of Inspect's header and every
    # other consumer of a log here goes through `summarize_log`. A log without it simply
    # recorded no task_args -- that is the pre-`--task_args` case, not an error.
    summary["task_args"] = dict(getattr(getattr(log, "eval", None), "task_args", None) or {})
    eval_header = getattr(log, "eval", None)
    eval_config = getattr(eval_header, "config", None)
    generate_config = getattr(eval_header, "model_generate_config", None)
    summary["settings"] = {
        "epochs": getattr(eval_config, "epochs", None),
        "temperature": getattr(generate_config, "temperature", None),
        "max_tokens": getattr(generate_config, "max_tokens", None),
        "max_samples": getattr(eval_config, "max_samples", None),
    }
    analysis_dir = paths.experiment_dir(experiment) / "analysis" / spec.name / path.parent.name / path.stem
    if analysis := runner.analyze_log(log, spec, analysis_dir):
        analysis["directory"] = str(analysis_dir.relative_to(paths.experiment_dir(experiment)))
        summary["analysis"] = analysis
    return summary


def screen_rate(entry: Any) -> float | None:
    """The rate out of a screen entry, in either shape.

    Screens used to be `{scanner: 0.12}` and are now `{scanner: {"rate": 0.12, "n": ...,
    "scan": ..., "judge": ...}}`. Every scoreboard already on disk carries the old
    shape, so both are read rather than migrating files -- the point of recording
    provenance is defeated if adopting it rewrites the artifacts it was meant to
    describe.
    """
    if isinstance(entry, dict):
        entry = entry.get("rate")
    return entry if isinstance(entry, (int, float)) else None


@dataclass
class Scoreboard:
    experiment: str
    baseline: str | None = None
    hf_sha: str | None = None
    models: dict[str, Any] = field(default_factory=dict)
    evals: dict[str, str] = field(default_factory=dict)
    polarity: dict[str, str] = field(default_factory=dict)
    cells: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    # Who ran this, when, against what code and through which providers. A scoreboard
    # used to record none of it, which is how two of them ended up on disk reporting
    # opposite-signed deltas for the same adapter on the same eval with no way to tell
    # which came first or what differed. Populated by the runner; `{}` when a
    # scoreboard is loaded from a file written before this existed.
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- building ------------------------------------------------------------

    def add(self, eval_name: str, model: str, body: dict[str, Any]) -> None:
        self.cells.append({"eval": eval_name, "model": model, **body, "screens": {}})

    def add_error(self, eval_name: str, model: str, message: str) -> None:
        entry = {"eval": eval_name, "model": model, "error": message}
        self.errors.append(entry)
        self.cells.append({**entry, "value": None, "groups": {}, "screens": {}})

    def finalize(self) -> None:
        """Fill in every delta against the baseline column."""
        if not self.baseline:
            return
        base_by_eval = {c["eval"]: c.get("value") for c in self.cells if c["model"] == self.baseline}
        base_groups = {c["eval"]: (c.get("groups") or {}) for c in self.cells if c["model"] == self.baseline}
        base_metrics = {c["eval"]: (c.get("metrics") or {}) for c in self.cells if c["model"] == self.baseline}
        for cell in self.cells:
            if cell["model"] == self.baseline:
                cell["delta_vs_baseline"] = None
                if cell.get("metrics"):
                    cell["metric_deltas"] = {name: None for name in cell["metrics"]}
                continue
            cell["delta_vs_baseline"] = _delta(base_by_eval.get(cell["eval"]), cell.get("value"))
            if cell.get("metrics"):
                cell["metric_deltas"] = {
                    name: _delta(base_metrics.get(cell["eval"], {}).get(name), value)
                    for name, value in cell["metrics"].items()
                }
            for group, bucket in (cell.get("groups") or {}).items():
                base_rate = (base_groups.get(cell["eval"], {}).get(group) or {}).get("rate")
                bucket["delta"] = _delta(base_rate, bucket.get("rate"))

    # -- persistence ---------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "baseline": self.baseline,
            "hf_sha": self.hf_sha,
            "provenance": self.provenance,
            "models": self.models,
            "evals": self.evals,
            "polarity": self.polarity,
            "cells": self.cells,
            "errors": self.errors,
        }

    def write(self, path: Path | str | None = None) -> Path:
        out = Path(path) if path else paths.scoreboard_path(self.experiment)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=False))
        self.append_history(out.parent / "runs.jsonl")
        return out

    def append_history(self, path: Path | str) -> None:
        """Append one line describing this run, because `write` overwrites.

        `scoreboard.json` is replaced in place on every run while the `.eval` logs
        accumulate under a stable `logs/<eval>/<model>/` -- four cell directories on
        disk already hold two logs each. So after a partial re-run the scoreboard and
        the log tree describe different runs and nothing correlates them. This keeps
        the history that overwrite destroys, without changing the file everything
        already reads.
        """
        row = {
            "provenance": self.provenance,
            "baseline": self.baseline,
            "hf_sha": self.hf_sha,
            "models": sorted(self.models),
            "cells": [
                {k: c.get(k) for k in ("eval", "model", "value", "scored", "unscored", "delta_vs_baseline")}
                for c in self.cells
            ],
            "errors": len(self.errors),
        }
        try:
            with open(path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
        except OSError as exc:
            LOGGER.warning("Could not append run history to %s: %s", path, exc)

    @classmethod
    def load(cls, experiment_or_path: str | Path) -> Scoreboard:
        path = Path(experiment_or_path)
        if not path.suffix:
            path = paths.scoreboard_path(str(experiment_or_path))
        if not path.exists():
            raise FileNotFoundError(f"No scoreboard at {path}. Run `sdf evals run --experiment ...` first.")
        raw = json.loads(path.read_text())
        board = cls(
            experiment=raw.get("experiment", path.parent.name),
            baseline=raw.get("baseline"),
            hf_sha=raw.get("hf_sha"),
            models=raw.get("models") or {},
            evals=raw.get("evals") or {},
            polarity=raw.get("polarity") or {},
            cells=raw.get("cells") or [],
            errors=raw.get("errors") or [],
        )
        return board

    # -- rendering -----------------------------------------------------------

    def model_labels(self) -> list[str]:
        """Baseline first, so the column deltas are read against is on the left."""
        labels = list(self.models) or sorted({c["model"] for c in self.cells})
        if self.baseline in labels:
            labels = [self.baseline] + [x for x in labels if x != self.baseline]
        return labels

    def cell(self, eval_name: str, model: str) -> dict[str, Any] | None:
        for c in self.cells:
            if c["eval"] == eval_name and c["model"] == model:
                return c
        return None

    def render(self) -> None:
        labels = self.model_labels()
        eval_names = list(self.evals) or sorted({c["eval"] for c in self.cells})
        width = max([len(e) for e in eval_names] + [4]) + 2

        print(f"\nscoreboard  experiment={self.experiment}  baseline={self.baseline or '(none)'}")
        print("\nscore (delta vs baseline)")
        print("  " + "eval".ljust(width) + "".join(f"{lbl:>22}" for lbl in labels))
        for name in eval_names:
            row = "  " + f"{name} [{self.polarity.get(name, '?')[:4]}]".ljust(width)
            for label in labels:
                cell = self.cell(name, label)
                row += f"{_fmt_value(cell, label == self.baseline):>22}"
            print(row)

        if any(c.get("unscored") for c in self.cells):
            print("\nunscored (no parseable answer; never counted as compliance)")
            print("  " + "eval".ljust(width) + "".join(f"{lbl:>22}" for lbl in labels))
            for name in eval_names:
                row = "  " + name.ljust(width)
                for label in labels:
                    cell = self.cell(name, label) or {}
                    n, u = cell.get("n"), cell.get("unscored")
                    row += f"{(f'{u}/{n}' if n else '-'):>22}"
                print(row)

        self._render_also_ran()
        self._render_groups(labels)
        self._render_termination()
        self._render_analysis()
        self._render_reported_metrics(labels)
        self._render_screens(labels, eval_names, width)

        if self.errors:
            print(f"\n{len(self.errors)} cell(s) FAILED")
            for e in self.errors:
                print(f"  {e['eval']} x {e['model']}: {e['error'][:160]}")

        print(
            "\nRead the unscored rows before the rates: a model that stopped answering "
            "is not a model that complied.\n"
            "[game] = 1 means the model played to the declared metric; "
            "[comp] = 1 means it followed the instruction;\n"
            "[capa] = 1 means it answered correctly -- a benchmark control, and the one "
            "row where a FALLING column is the bad news."
        )

    def _render_also_ran(self) -> None:
        """The logs a crowded cell is not reporting, and how they differ.

        Printed rather than left in the JSON, because the failure this guards against is
        a reader taking one number for the whole cell. Only the `task_args` keys that
        actually vary are shown: the shared ones are noise here, and the differing one is
        the entire reason two rows exist.
        """
        crowded = [c for c in self.cells if c.get("also_ran")]
        if not crowded:
            return
        print("\nother logs in these cells (the board reports the one marked ->)")
        for cell in crowded:
            rows = [cell, *cell["also_ran"]]
            keys = _varying_keys(rows)
            how = f"pinned to {cell['reported_pin']!r}" if cell.get("reported_pin") else "newest reported"
            print(f"  {cell['eval']} x {cell['model']}: {len(rows)} logs, {how}")
            for row in rows:
                value = row.get("value")
                shown = f"{value:.1%}" if isinstance(value, (int, float)) else "-"
                detail = "  ".join(
                    f"{key if section == 'task_args' else f'setting.{key}'}={(row.get(section) or {}).get(key)!r}"
                    for section, key in keys
                ) or "(same configuration)"
                marker = "->" if row is cell else "  "
                count = f"{row.get('n') or '?'}"
                print(f"    {marker} {shown:>7}  n={count:<5} {detail}")

    def _render_groups(self, labels: list[str]) -> None:
        for name in list(self.evals) or sorted({c["eval"] for c in self.cells}):
            groups: list[str] = []
            for label in labels:
                for g in (self.cell(name, label) or {}).get("groups") or {}:
                    if g not in groups:
                        groups.append(g)
            if len(groups) <= 1:
                continue
            print(f"\n{name} by group")
            print("  " + "group".ljust(20) + "".join(f"{lbl:>22}" for lbl in labels))
            for group in groups:
                row = "  " + group.ljust(20)
                for label in labels:
                    bucket = ((self.cell(name, label) or {}).get("groups") or {}).get(group) or {}
                    rate, delta = bucket.get("rate"), bucket.get("delta")
                    if rate is None:
                        row += f"{'-':>22}"
                    elif delta is None:
                        row += f"{rate:>21.1%} "
                    else:
                        row += f"{f'{rate:.1%} ({delta:+.1%})':>22}"
                print(row)

    def _render_termination(self) -> None:
        limited = [c for c in self.cells if (c.get("termination") or {}).get("limited")]
        if not limited:
            return
        print("\ncontrol-limit terminations (kept separate from behavioral outcomes)")
        for cell in limited:
            termination = cell["termination"]
            counts = ", ".join(f"{key}={value}" for key, value in termination.get("by_type", {}).items())
            print(
                f"  {cell['eval']} x {cell['model']}: "
                f"{termination['limited']}/{cell.get('n', '?')} limited ({counts})"
            )

    def _render_analysis(self) -> None:
        analyzed = [c for c in self.cells if c.get("analysis")]
        if not analyzed:
            return
        print("\npost-run validity analysis")
        for cell in analyzed:
            analysis = cell["analysis"]
            status = str(analysis.get("status", "unknown")).upper()
            reasons = analysis.get("reasons") or []
            detail = "; ".join(str(reason) for reason in reasons) or "all configured gates passed"
            artifact = (analysis.get("artifacts") or {}).get("plot")
            suffix = f"; plot={analysis.get('directory')}/{artifact}" if artifact else ""
            print(f"  {cell['eval']} x {cell['model']}: {status} — {detail}{suffix}")

    def _render_reported_metrics(self, labels: list[str]) -> None:
        """Show opt-in secondary metrics produced from the same sampled transcripts."""
        for name in list(self.evals) or sorted({c["eval"] for c in self.cells}):
            metric_names: list[str] = []
            for label in labels:
                cell = self.cell(name, label) or {}
                primary = cell.get("metric")
                for metric_name in cell.get("metrics") or {}:
                    if metric_name != primary and metric_name not in metric_names:
                        metric_names.append(metric_name)
            if not metric_names:
                continue
            print(f"\n{name} secondary metrics (same transcripts)")
            print("  " + "metric".ljust(30) + "".join(f"{label:>22}" for label in labels))
            for metric_name in metric_names:
                row = "  " + metric_name.ljust(30)
                for label in labels:
                    cell = self.cell(name, label) or {}
                    value = (cell.get("metrics") or {}).get(metric_name)
                    delta = (cell.get("metric_deltas") or {}).get(metric_name)
                    if value is None:
                        rendered = "-"
                    elif delta is None:
                        rendered = f"{value:.1%}"
                    else:
                        rendered = f"{value:.1%} ({delta:+.1%})"
                    row += f"{rendered:>22}"
                print(row)

    def _render_screens(self, labels: list[str], eval_names: list[str], width: int) -> None:
        names: list[str] = []
        for c in self.cells:
            for s in (c.get("screens") or {}):
                if s not in names:
                    names.append(s)
        if not names:
            return
        for screen in sorted(names):
            print(f"\nscreen: {screen} (LLM judge over transcripts)")
            print("  " + "eval".ljust(width) + "".join(f"{lbl:>22}" for lbl in labels))
            for name in eval_names:
                row = "  " + name.ljust(width)
                for label in labels:
                    entry = ((self.cell(name, label) or {}).get("screens") or {}).get(screen)
                    rate = screen_rate(entry)
                    n = entry.get("n") if isinstance(entry, dict) else None
                    # The denominator beside the rate: a screen over 25 transcripts and
                    # one over the whole suite used to print identically.
                    cellstr = "-" if rate is None else f"{rate:.1%}" + (f" (n={n})" if n else "")
                    row += f"{cellstr:>22}"
                print(row)


def _fmt_value(cell: dict[str, Any] | None, is_baseline: bool) -> str:
    if not cell:
        return "-"
    if cell.get("error"):
        return "ERROR"
    value = cell.get("value")
    if value is None:
        return "-"
    delta = cell.get("delta_vs_baseline")
    if is_baseline or delta is None:
        return f"{value:.1%}"
    return f"{value:.1%} ({delta:+.1%})"


def rebuild(
    experiment: str,
    evals_dir: str | None = None,
    models_manifest: str | None = None,
    baseline: str | None = None,
    prefer: dict[str, str] | str | None = None,
) -> Scoreboard:
    """Reconstruct a scoreboard from the `.eval` logs already on disk.

    The logs are the durable artifact and this file is a summary of them, so anything
    the summary lost -- a column overwritten by a later single-model run, several
    separately-run models collected into one experiment directory by hand -- is
    recoverable without paying for sampling again. Cells are summarised through the
    same `runner.summarize_log` the live suite uses, so a rebuilt board and a
    freshly-run one cannot disagree about what a number means.

    Preserved from the existing board when there is one: `hf_sha`, and each cell's
    `screens`. Scan rates are a join over these same transcripts, so re-running
    `sdf scan run` is not required to get them back -- but they are keyed by
    (eval, model), so a screen for a cell whose log has since been replaced is
    dropped rather than carried onto the new numbers.

    `prefer` names which log a crowded cell reports, as `<eval>/<model>=<substring>`
    (or the dict of the same). Newest-wins is the right default for a re-run, and the
    wrong one as soon as a label holds several *different* measurements: the baseline
    column then changes with a timestamp, and every delta on the board moves with it.
    A pin is written to `reported_logs.json` beside the scoreboard and honoured by
    later rebuilds without the flag, because a pin that has to be re-passed is one
    forgotten rebuild away from silently restoring the number it was correcting.
    """
    from inspect_ai.log import read_eval_log

    from sdf import catalogue
    from sdf.evals import registry  # local: drags in inspect_ai and bootstrap()
    from sdf.models import targets

    root = paths.logs_dir(experiment)
    if not root.is_dir():
        raise FileNotFoundError(
            f"No logs at {root}. `--rebuild` reads `logs/<eval>/<model>/*.eval`; "
            "there is nothing to rebuild from."
        )

    previous = Scoreboard.load(experiment) if paths.scoreboard_path(experiment).exists() else None
    manifest = targets.load_manifest(models_manifest)
    declared = manifest.get("models") or {}
    pins = _merge_pins(experiment, prefer)

    board = Scoreboard(experiment=experiment, hf_sha=(previous.hf_sha if previous else None))
    for eval_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        spec = registry.load(eval_dir.name, evals_dir)
        board.evals[spec.name] = spec.description
        board.polarity[spec.name] = spec.polarity
        for model_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
            label = model_dir.name
            # Inspect never overwrites -- a re-run drops a second timestamped file
            # beside the first -- so a cell directory legitimately accumulates. The
            # names begin with an ISO timestamp, so lexicographic order is
            # chronological and the last is the newest; rebuilding from any other
            # would silently report a superseded run as the current one.
            found = sorted(model_dir.glob("*.eval"))
            if not found:
                print(f"  {spec.name} x {label}: no .eval file, skipped")
                continue
            summaries, bindings = [], []
            for path in found:
                log = read_eval_log(str(path))
                summaries.append(_summarize(log, path, spec, experiment))
                bindings.append(catalogue.binding_from_log(log))
            # One log is reported; the rest are recorded rather than dropped. A cell
            # holding several logs is not always one measurement re-run: a label names
            # *weights*, and `logs/<eval>/<model>/` has no axis for the prompt, so runs
            # that differ only by `task_args` land in one directory. Reporting the newest
            # and saying nothing turned six base-model runs spanning 0.0% to 96.2% into a
            # lone 58.9% with no trace of the other five anywhere in the file.
            key = f"{spec.name}/{label}"
            cell = _choose_reported(summaries, pins.get(key), key)
            others = [s for s in reversed(summaries) if s is not cell]
            if others:
                cell["also_ran"] = [{k: o.get(k) for k in ALSO_RAN_FIELDS} for o in others]
                varied = sum(1 for o in others if o.get("task_args") != cell.get("task_args"))
                how = f"log pinned by {pins[key]!r}" if key in pins else "newest"
                if key in pins:
                    cell["reported_pin"] = pins[key]
                print(
                    f"  {spec.name} x {label}: {len(summaries)} logs, reporting the {how} "
                    f"({Path(cell['log']).name})"
                    + (f"; {varied} of the others sampled a different configuration" if varied else "")
                )
            board.add(spec.name, label, cell)
            # A label with no manifest entry is still a real column -- the logs prove it
            # ran -- so it is kept, and the log itself says what it was: `eval.model`
            # plus the `model_args` the sampler was built from. `data/models.json` is
            # edited in place and is not in git, so a label a board was written under
            # routinely no longer resolves there; recording nulls in that case is not
            # cosmetic, because `catalogue.reference_graph` follows `models{run_dir}` to
            # the training run behind a column, and a null makes real work read as
            # unreferenced. The manifest still wins where it has an answer -- it carries
            # `checkpoint`, which no log header does.
            reported = bindings[next(i for i, s in enumerate(summaries) if s is cell)]
            board.models.setdefault(
                label, declared.get(label) or reported or {"kind": None, "model": None, "run_dir": None}
            )
            if previous and (old := previous.cell(spec.name, label)):
                board.cells[-1]["screens"] = old.get("screens") or {}

    labels = list(board.models)
    chosen = baseline or (previous.baseline if previous else None) or manifest.get("baseline")
    # Same rule as `targets.resolve`: a delta against a column that is not here is
    # worse than no delta, so an absent baseline silently becomes no baseline.
    board.baseline = chosen if chosen in labels else None
    if chosen and board.baseline is None:
        print(f"  baseline {chosen!r} has no logs under this experiment; deltas omitted")

    board.finalize()
    return board


def show(
    experiment: str,
    scan: str | None = None,
    rebuild_from_logs: bool = False,
    evals_dir: str | None = None,
    models_manifest: str | None = None,
    baseline: str | None = None,
    prefer: dict[str, str] | str | None = None,
    no_wandb: bool = True,
    wandb_project: str | None = None,
) -> None:
    """Print a saved scoreboard. `sdf evals scoreboard --experiment r1 [--scan <dir>]`.

    With `--scan`, joins a Scout scan's per-cell rates in first, so the screen rates
    sit beside the scores they qualify.

    With `--rebuild_from_logs`, the board is reconstructed from the `.eval` files under
    `logs/` and rewritten before anything is printed -- the way to recover a column a
    later single-model run overwrote, or to collect models that were sampled in
    separate processes into one experiment directory.
    """
    if rebuild_from_logs:
        board = rebuild(
            experiment,
            evals_dir=evals_dir,
            models_manifest=models_manifest,
            baseline=baseline,
            prefer=prefer,
        )
        out = board.write()
        print(f"rebuilt {len(board.cells)} cell(s) from logs -> {out}")
    else:
        board = Scoreboard.load(experiment)
    if scan:
        from sdf.evals.scanners import join

        added = join.attach(board, scan)
        board.write()
        print(f"joined {added} scanner rate(s) from {scan}")
    board.render()

    if not no_wandb:
        from sdf import tracking

        with tracking.run(board.experiment, stage="evals", project=wandb_project) as t:
            t.log_evals(board.as_dict())
