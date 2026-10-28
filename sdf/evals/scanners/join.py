"""Folding a Scout scan's results back into the scoreboard.

A scan produces one row per (transcript, scanner). The scoreboard wants one rate per
(eval, model, scanner). The join is on which cell a transcript came from, and there
are two ways to recover that, tried in order:

1. The scan's own columns: `transcript_task_set` is the Inspect task name, which is
   the eval, and `transcript_model` is the model it ran against.
2. The log path in `transcript_source_uri`. `sdf evals run` writes to
   `data/results/<exp>/logs/<eval>/<model_label>/*.eval`, so the two path components
   above the file *are* the join key.

Both routes are load-bearing, and the column names above were read off a real scan
rather than taken from the documentation, which describes them unprefixed. Route 1
alone is not enough: `transcript_model` holds the provider string
(`tinker/deepseek-ai/DeepSeek-V3.1`), not the manifest label, so a base model's label has to
be matched back through the manifest, and two manifest entries pointing at the same
base model are indistinguishable there. Route 2 holds regardless -- which is the
reason the runner nests log directories instead of writing one flat pile.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _label_for_model_string(model_string: str, models: dict[str, Any]) -> str | None:
    """Map an Inspect `tinker/...` model string back to a manifest label."""
    if not model_string:
        return None
    suffix = model_string.split("/", 1)[1] if "/" in model_string else model_string
    if suffix in models:  # adapter: the label is the string after `tinker/`
        return suffix
    for label, entry in models.items():
        if entry.get("model") and entry["model"] == suffix:
            return label
    return None


def _cell_from_path(location: str | None) -> tuple[str, str] | None:
    """`.../logs/<eval>/<model>/xxx.eval` -> (eval, model)."""
    if not location:
        return None
    parts = Path(str(location)).parts
    if "logs" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("logs")
    if len(parts) >= i + 3:
        return parts[i + 1], parts[i + 2]
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes"):
            return 1.0
        if lowered in ("false", "no"):
            return 0.0
        # A structured verdict can arrive as its JSON text depending on how the results
        # frame stores it. Only the decision matters here; thinking/evidence are for
        # reading in Scout, not for averaging.
        if lowered.startswith("{"):
            try:
                return _to_float(json.loads(value).get("answer"))
            except (ValueError, AttributeError):
                return None
    if isinstance(value, Mapping):
        return _to_float(value.get("answer"))
    return None


def _verdict(value: Any) -> Mapping[str, Any] | None:
    """The structured verdict behind a row, whichever way the frame stored it."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def rates_from_scan(scan_location: str, models: dict[str, Any]) -> dict[tuple[str, str], dict[str, float]]:
    """{(eval, model_label): {scanner: rate}} from a completed scan."""
    from inspect_scout import scan_results_df

    from sdf.evals.scanners import screens as screens_mod

    results = scan_results_df(scan_location)
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}

    for scanner_name, frame in results.scanners.items():
        columns = set(frame.columns)
        for row in frame.to_dict("records"):
            key = None
            # Route 2 first: the path is unambiguous, route 1 is not.
            for column in ("transcript_source_uri", "source_uri", "log", "location"):
                if column in columns:
                    key = _cell_from_path(row.get(column))
                    if key:
                        break
            if key is None:
                task = row.get("transcript_task_set") or row.get("task_name") or row.get("task")
                model_string = str(row.get("transcript_model") or row.get("model") or "")
                label = _label_for_model_string(model_string, models)
                if task and label:
                    key = (str(task), label)
            if key is None:
                continue
            raw = row.get("value")
            value = _to_float(raw)
            if value is None:
                continue
            buckets.setdefault(key, {}).setdefault(scanner_name, []).append(value)

            # A taxonomy screen picks one of several mutually-exclusive categories, so a
            # single verdict yields one rate per category as well as the union above.
            # Expanded here rather than in the scanner because it is a way of *reading*
            # one judge call, not a second question -- and doing it as three boolean
            # screens would bill three calls and permit three answers that contradict
            # each other.
            spec = screens_mod.taxonomy(scanner_name)
            if spec is None:
                continue
            field, categories = spec
            verdict = _verdict(raw)
            if verdict is None:
                continue
            chosen = verdict.get(field)
            for category in categories:
                # 0.0, not omission, when the verdict picked another category: the
                # denominator of every category must be the same set of transcripts, or
                # a rare category reads as a high rate over the handful of rows that
                # happened to carry it.
                buckets[key].setdefault(f"{scanner_name}::{category}", []).append(
                    1.0 if chosen == category else 0.0
                )

    meta = _scan_meta(scan_location)
    return {
        cell: {
            # The rate, and enough to know what it is a rate *of*. It used to be a bare
            # float: no denominator, no scan id, no judge. A scan run with
            # `max_transcripts: 25` and one over the full suite produced
            # indistinguishable numbers, and successive joins overwrote each other
            # silently, so a scoreboard could carry four screen rates from four
            # different scans with nothing saying so.
            name: {
                "rate": round(sum(values) / len(values), 4),
                "n": len(values),
                "scan": Path(scan_location).name,
                "judge": (meta.get("judge") or {}).get("model") if meta else None,
                # An expanded category shares its parent's question, so it shares the
                # hash: `doxxing::doxx_self_serving` was asked by the `doxxing` prompt.
                "prompt": (meta.get("prompt_hashes") or {}).get(name.split("::", 1)[0]) if meta else None,
            }
            for name, values in per_scanner.items()
            if values
        }
        for cell, per_scanner in buckets.items()
    }


def _scan_meta(scan_location: str) -> dict[str, Any] | None:
    """`scan_meta.json` beside the scan, absent for scans predating it."""
    try:
        return json.loads((Path(scan_location) / "scan_meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def attach(board: Any, scan_location: str) -> int:
    """Write screen rates onto a Scoreboard in place. Returns how many were set."""
    rates = rates_from_scan(scan_location, board.models)
    if not rates:
        LOGGER.warning(
            "No scan rows could be matched to a scoreboard cell in %s. Neither the "
            "task_name/model columns nor the log path resolved to a known (eval, model).",
            scan_location,
        )
    written = 0
    for cell in board.cells:
        found = rates.get((cell.get("eval"), cell.get("model")))
        if not found:
            continue
        cell.setdefault("screens", {}).update(found)
        written += len(found)
    return written
