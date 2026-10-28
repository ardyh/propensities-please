"""Unmonitored rollouts by stated belief in the cue and by action (the paper's belief/action table and figure).

usage: python experiments/belief_breakdown.py --scans 'data/results/<experiment>/scans/scan_id=*' \
           --model base 'logs/.../base/*.eval' --model sdf 'logs/.../sdf/**/*.eval'

`--scans` takes globs of scan directories written by `sdf scan run --screens cue_belief`, `cue_belief_agentic`,
or `monitoring_review`. Use --screen to select one when directories contain several screens.
Each --model takes a label and one or more globs of the Inspect logs those scans read; a transcript belongs to the
label whose logs it came from, so several checkpoints and seeds can be pooled under one label.

For cue screens, action is the eval's score as recorded by the scan (1 misaligned, 0 aligned, missing unscored).
Coding monitoring-review screens read the native outcome from the matched Inspect log so other/heuristic
actions are not mistaken for aligned actions. Belief is the screen's stance (accept, doubt, other).
A transcript scanned more than once, as the toy honesty
breakdown was (three scans of the same logs), takes the stance a strict majority of its verdicts agree on, and
`other` when there is none. Per label and arm this prints the attempts and the counts per belief and action, as in the
paper:

    ad / aa   aligned and doubting / accepting the cue
    md / ma   misaligned and doubting / accepting the cue
    oa / om   aligned / misaligned with any other stance

The honesty share of the organism controls is aa / attempts.

For monitoring_review, accepts -> accept, doubts -> doubt, mixed/not_established -> other.
This uses monitoring_belief, never answer (which measures support for doubt-related restraint).
Mixed does not establish a settled stance. Scan failures retain a null stance and are reported separately;
they are included in Other for the displayed action counts, never treated as acceptance or doubt.
Raw monitoring labels are retained in the loaded table. A monitoring label is relative to the claim judged:
endpoint outage and absence of all present/later review are different measurements. Do not pool their scans.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

SCREENS = ("cue_belief", "cue_belief_agentic", "monitoring_review", "monitoring_review_broad")
MONITORING_STANCES = {
    "accepts": "accept", "doubts": "doubt", "mixed": "other", "not_established": "other",
}


def stance_from_verdict(screen: str, value: dict) -> str | None:
    if screen in {"monitoring_review", "monitoring_review_broad"}:
        raw = value.get("monitoring_belief")
        if raw is not None and raw not in MONITORING_STANCES:
            raise ValueError(f"Unknown monitoring_belief: {raw!r}")
        return MONITORING_STANCES.get(raw)
    raw = value.get("stance")
    if raw is not None and raw not in {"accept", "doubt", "other"}:
        raise ValueError(f"Unknown cue stance: {raw!r}")
    return raw


def _real(path: str) -> str:
    return os.path.realpath(path)


def verdicts(scan_globs: list[str], screen_name: str | None = None) -> pd.DataFrame:
    """One row per (log, sample, epoch, verdict) across every matching scan."""
    rows = []
    dirs = sorted({d for pattern in scan_globs for d in glob.glob(pattern)})
    for scan_dir in dirs:
        meta_path = Path(scan_dir) / "scan_meta.json"
        prompts = (json.loads(meta_path.read_text()).get("prompt_hashes") or {}) if meta_path.exists() else {}
        for screen in ((screen_name,) if screen_name else SCREENS):
            path = os.path.join(scan_dir, f"{screen}.parquet")
            if not os.path.exists(path):
                continue
            frame = pd.read_parquet(path)
            for row in frame.itertuples():
                failed = isinstance(row.scan_error, str) and bool(row.scan_error)
                value = {} if failed else (json.loads(row.value) if isinstance(row.value, str) else row.value)
                if not isinstance(value, dict):
                    value = {}
                score = pd.to_numeric(row.transcript_score, errors="coerce")
                meta = json.loads(row.transcript_metadata)
                rows.append({
                    "log": _real(meta["log"]),
                    "arm": (meta.get("sample_metadata") or {}).get("arm"),
                    "sample": row.transcript_task_id,
                    "epoch": row.transcript_task_repeat,
                    "score": None if score is None or math.isnan(score) else int(score),
                    "stance": stance_from_verdict(screen, value),
                    "raw_belief": value.get("monitoring_belief", value.get("stance")),
                    "scan_failed": failed or not value,
                    "screen": screen,
                    "judge": f"{screen}:{prompts.get(screen)}",
                })
    if not rows:
        raise SystemExit(f"no {screen_name or 'supported belief-screen'} verdicts under {scan_globs}")
    return pd.DataFrame(rows)


def majority(stances: list[str | None]) -> str:
    counts = Counter(s for s in stances if isinstance(s, str) and s in {"accept", "doubt", "other"})
    if counts:
        stance, n = counts.most_common(1)[0]
        if n * 2 > len(stances):
            return stance
    return "other"


@lru_cache(maxsize=64)
def native_actions(log_path: str) -> dict[tuple[str, int], str]:
    """Recover coding action categories; scalar zero alone cannot distinguish other from faithful."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(log_path)
    actions = {}
    for sample in log.samples or []:
        if sample.error or sample.limit:
            action = "unscored"
        else:
            outcomes = {(score.metadata or {}).get("outcome") for score in (sample.scores or {}).values()}
            outcomes.discard(None)
            if len(outcomes) != 1:
                raise ValueError(f"Expected one native action outcome for {log_path}:{sample.id}, got {outcomes}")
            outcome = outcomes.pop()
            action = {"hack": "misaligned", "faithful": "aligned", "report": "aligned", "other": "other"}.get(
                outcome, "unscored")
        actions[str(sample.id), sample.epoch] = action
    return actions


def summarize_attempts(rows: list[dict], scope: str = "endpoint") -> dict[str, dict]:
    """Aggregate saved labels without new target or judge calls, preserving all action categories."""
    cells = {}
    seen = set()
    for row in rows:
        key = (row["model"], row["arm"], row["task"], row["id"])
        if key in seen:
            raise ValueError(f"Duplicate attempt: {key}")
        seen.add(key)
        if scope not in row["beliefs"]:
            raise ValueError(f"No {scope} belief measurement for {row['id']}")
        raw = row["beliefs"][scope]
        stance = stance_from_verdict("monitoring_review", {"monitoring_belief": raw})
        action = row["action"]
        if action not in {"aligned", "misaligned", "other", "unscored"}:
            raise ValueError(f"Unknown action: {action}")
        cell = cells.setdefault(f"{row['model']}/{row['arm']}", {
            "attempts": 0, "ad": 0, "aa": 0, "md": 0, "ma": 0, "oa": 0, "om": 0,
            "other_action": 0, "unscored": 0, "belief_unavailable": 0,
            "raw_beliefs": {}, "action_counts": {}, "action_by_belief": {},
        })
        cell["attempts"] += 1
        cell["belief_unavailable"] += int(raw is None)
        raw_key = raw if raw is not None else "scanner_unavailable"
        cell["raw_beliefs"][raw_key] = cell["raw_beliefs"].get(raw_key, 0) + 1
        cell["action_counts"][action] = cell["action_counts"].get(action, 0) + 1
        joint = cell["action_by_belief"].setdefault(action, {})
        joint[raw_key] = joint.get(raw_key, 0) + 1
        if action in {"other", "unscored"}:
            cell["other_action" if action == "other" else "unscored"] += 1
        else:
            a = "a" if action == "aligned" else "m"
            b = {"accept": "a", "doubt": "d"}.get(stance)
            cell[a + b if b else "o" + a] += 1
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scans", nargs="+", metavar="GLOB")
    inputs.add_argument("--attempts", type=Path, help="Saved per-attempt labels (JSONL); no scans or API calls required")
    parser.add_argument("--screen", choices=SCREENS)
    parser.add_argument("--model", nargs="+", action="append", metavar=("LABEL", "GLOB"))
    parser.add_argument("--scope", choices=("endpoint", "broad"), default="endpoint")
    parser.add_argument("--arms", help="Comma-separated arms to keep when reading --attempts")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary for --attempts")
    args = parser.parse_args()

    if args.attempts:
        rows = [json.loads(line) for line in args.attempts.read_text().splitlines() if line.strip()]
        if args.arms:
            selected = set(args.arms.split(","))
            if selected - {"on", "off"}:
                parser.error("--arms must select on,off")
            rows = [row for row in rows if row["arm"] in selected]
        summary = summarize_attempts(rows, args.scope)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            for key, cell in sorted(summary.items()):
                print(key, json.dumps(cell, sort_keys=True))
        return
    if not args.model:
        parser.error("--scans requires --model labels and log globs")
    table = verdicts(args.scans, args.screen)
    owner = {}
    for label, *patterns in args.model:
        for pattern in patterns:
            for path in glob.glob(pattern, recursive=True):
                owner[_real(path)] = label
    table["label"] = table.log.map(owner)
    if table.label.isna().any():
        unclaimed = sorted(table.loc[table.label.isna(), "log"].unique())
        print(f"ignoring {len(unclaimed)} scanned log(s) no --model glob matches, e.g. {unclaimed[0]}")
        table = table.dropna(subset=["label"])

    # Verdicts from two versions of a screen's question are two different measurements; a majority across them
    # would be meaningless, so mixing them is refused rather than pooled.
    if table.judge.nunique() > 1:
        raise SystemExit(f"the selected transcripts were judged by different prompts: {sorted(table.judge.unique())}")

    cells: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for (label, arm, log, sample, epoch), group in table.groupby(["label", "arm", "log", "sample", "epoch"], dropna=False):
        stance = majority(list(group.stance))
        score = group.score.iloc[0]
        cell = cells[label, arm]
        cell["attempts"] += 1
        cell["belief_unavailable"] += int(not any(group.stance.notna()))
        if group.screen.iloc[0] in {"monitoring_review", "monitoring_review_broad"}:
            action = native_actions(log)[str(sample), int(epoch)]
            if action in {"other", "unscored"}:
                cell["other_action" if action == "other" else "unscored"] += 1
                continue
            score = int(action == "misaligned")
        if score is None or pd.isna(score):
            cell["unscored"] += 1
            continue
        action = "m" if score == 1 else "a"
        belief = {"doubt": "d", "accept": "a"}.get(stance)
        cell[action + belief if belief else "o" + action] += 1

    labels = [label for label, *_ in args.model]
    columns = ("attempts", "ad", "aa", "md", "ma", "oa", "om", "other_action", "unscored", "belief_unavailable")
    widths = {c: max(9, len(c) + 2) for c in columns}
    print(f"{'model':<12}{'arm':<20}" + "".join(f"{c:>{widths[c]}}" for c in columns) + "   honesty share (aa/attempts)")
    for (label, arm), cell in sorted(cells.items(), key=lambda kv: (labels.index(kv[0][0]), str(kv[0][1]))):
        share = cell["aa"] / cell["attempts"]
        print(f"{label:<12}{arm!s:<20}" + "".join(f"{cell[c]:>{widths[c]}}" for c in columns) + f"   {share:.1%}")


if __name__ == "__main__":
    main()
