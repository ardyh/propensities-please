"""Grade upstream's own ODCV transcripts with this port's judge; compare to the paper's.

The paper scores every episode with four frontier judges and reports their median. This
port starts with one (GPT-5.6 Luna). Whether one is enough is a question about the
*judge*, so it is answered on transcripts that already exist -- the submodule ships every
episode the paper scored, with every judge's score beside it -- rather than by sampling
new episodes. Judge calls only; no Tinker, no Docker.

    uv run python evals/odcv/calibrate.py --models kimi-k2.6,gpt-oss-20b --dry_run
    uv run python evals/odcv/calibrate.py --models kimi-k2.6,gpt-oss-20b

Each transcript goes through `bench.judge_record` -- the prompt, the parser and the
routing the scorer uses -- so what is measured here is the scorer, not a lookalike.

What it reports, per model x variant and overall:

    agreement_at_3   Luna vs the four-judge median on the binary the headline is built
                     from (score >= 3), with Cohen's kappa beside it.
    mean_abs_diff    on the 0-5 scale.
    rate_luna / rate_median   the misalignment rate each would publish.
    judges           each upstream judge against the median of the *other three*, on the
                     same binary: the spread Luna's agreement should be read against.
                     Leave-one-out, because a judge compared with a median it sits in
                     agrees with itself for free. Luna is compared with the full median.

Transcripts above `--max_chars` are skipped and counted, not truncated: one upstream
gpt-oss-20b transcript is 1.5 M characters, and a truncated record would be grading a
different episode.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

from sdf import compat
from sdf.paths import RESULTS_DIR

HERE = Path(__file__).resolve().parent
UPSTREAM_JUDGES = {
    "claude47": "scores_claude47.csv",
    "gemini": "scores_gemini.csv",
    "gpt55": "scores_gpt55.csv",
    "grok": "scores_grok.csv",
}


def _load_task_module():
    # `evals/` is not a package; load the sibling by path, the way the registry does.
    spec = importlib.util.spec_from_file_location("sdf_eval_odcv_bench", HERE / "bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_scores(path: Path, scenarios: set[str]) -> dict[tuple[str, str], float]:
    """(scenario, column) -> score. Footer rows and N/A cells are dropped."""
    out: dict[tuple[str, str], float] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            scenario = row.get("scenario")
            if scenario not in scenarios:
                continue
            for column, value in row.items():
                try:
                    out[(scenario, column)] = float(value)
                except (TypeError, ValueError):
                    continue
    return out


def _kappa(a: list[bool], b: list[bool]) -> float | None:
    if not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _summary(rows: list[dict], threshold: float) -> dict:
    graded = [r for r in rows if r.get("luna") is not None and r.get("median") is not None]
    luna = [r["luna"] >= threshold for r in graded]
    median = [r["median"] >= threshold for r in graded]
    judges = {}
    for judge in UPSTREAM_JUDGES:
        pairs = [
            (
                r["judges"][judge] >= threshold,
                statistics.median(v for k, v in r["judges"].items() if k != judge) >= threshold,
            )
            for r in rows
            if judge in r["judges"] and len(r["judges"]) == len(UPSTREAM_JUDGES)
        ]
        if pairs:
            judges[judge] = {
                "n": len(pairs),
                "agreement_at_3": sum(x == y for x, y in pairs) / len(pairs),
                "kappa": _kappa([x for x, _ in pairs], [y for _, y in pairs]),
            }
    return {
        "n": len(rows),
        "n_graded": len(graded),
        "n_unparseable": sum(1 for r in rows if r.get("outcome") == "unparseable"),
        "n_skipped": sum(1 for r in rows if r.get("outcome") == "skipped"),
        "agreement_at_3": (sum(x == y for x, y in zip(luna, median)) / len(graded)) if graded else None,
        "kappa": _kappa(luna, median),
        "mean_abs_diff": statistics.fmean(abs(r["luna"] - r["median"]) for r in graded) if graded else None,
        "rate_luna": (sum(luna) / len(graded)) if graded else None,
        "rate_median": (sum(median) / len(graded)) if graded else None,
        "judges": judges,
    }


async def _grade(task_module, judge, items: list[dict], concurrency: int) -> None:
    gate = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> None:
        async with gate:
            verdict, attempts, last = await task_module.judge_record(judge, item.pop("_record"))
        item["judge_attempts"] = attempts
        if verdict is None:
            item["outcome"] = "unparseable"
            item["judge_last_reply"] = last[:2000]
        else:
            item["outcome"] = "graded"
            item["luna"] = verdict["severity"]
            item["luna_reasoning"] = str(verdict.get("reasoning", ""))

    await asyncio.gather(*(one(item) for item in items if item["outcome"] == "pending"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--models", required=True, help="Comma-separated upstream model folders, e.g. kimi-k2.6,gpt-oss-20b"
    )
    parser.add_argument("--variants", default="mandated,incentivized")
    parser.add_argument("--judge_model", default=None, help="Defaults to the scorer's own judge")
    parser.add_argument("--max_chars", type=int, default=300_000, help="Skip transcripts longer than this")
    parser.add_argument("--limit", type=int, default=None, help="Scenarios per model x variant, for a cheap first look")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true", help="Count transcripts and characters; call nothing")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Loads .env (OPENROUTER_API_KEY), as every `sdf` entry point does before a paid call.
    compat.bootstrap()
    task_module = _load_task_module()
    task_module.verify_pin()
    upstream = task_module.upstream_dir() / "existing_results" / "current"
    scenarios = task_module.scenario_names()
    judge_dir = upstream / "evaluations" / "judge_all"
    median = _read_scores(judge_dir / "scores_final_median.csv", set(scenarios))
    per_judge = {name: _read_scores(judge_dir / file, set(scenarios)) for name, file in UPSTREAM_JUDGES.items()}

    items: list[dict] = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
            column = f"{model}-{variant}"
            folder = upstream / "agent_logs" / column / "experiments"
            if not folder.is_dir():
                raise SystemExit(f"No upstream transcripts for {column!r} under {folder.parent}")
            for scenario in scenarios[: args.limit] if args.limit else scenarios:
                path = folder / scenario / "messages_record.txt"
                item = {
                    "model": model,
                    "variant": variant,
                    "scenario": scenario,
                    "median": median.get((scenario, column)),
                    "judges": {j: s[(scenario, column)] for j, s in per_judge.items() if (scenario, column) in s},
                }
                if not path.exists():
                    item["outcome"] = "missing"
                else:
                    record = path.read_text(errors="replace")
                    item["chars"] = len(record)
                    if len(record) > args.max_chars:
                        item["outcome"] = "skipped"
                    else:
                        item["outcome"] = "pending"
                        item["_record"] = record
                items.append(item)

    pending = [i for i in items if i["outcome"] == "pending"]
    chars = sum(i["chars"] for i in pending)
    print(
        f"{len(pending)} transcripts to grade, {chars:,} characters (~{chars // 4:,} prompt tokens before the rubric); "
        f"{sum(i['outcome'] == 'skipped' for i in items)} over --max_chars, {sum(i['outcome'] == 'missing' for i in items)} missing."
    )
    if args.dry_run:
        return

    judge = task_module.resolve_judge(args.judge_model)
    asyncio.run(_grade(task_module, judge, items, args.concurrency))

    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(f"{item['model']}-{item['variant']}", []).append(item)
    report = {
        "judge_model": task_module._judge_name(args.judge_model),
        "upstream_commit": task_module.PINNED_COMMIT,
        "threshold": task_module.MISALIGNED_AT,
        "overall": _summary(items, task_module.MISALIGNED_AT),
        "by_column": {k: _summary(v, task_module.MISALIGNED_AT) for k, v in groups.items()},
        "items": items,
    }
    out = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / "odcv_judge_calibration" / f"{report['judge_model'].replace('/', '_')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"{'column':<32}{'n':>4}{'agree@3':>9}{'kappa':>8}{'|diff|':>8}{'rate luna':>11}{'rate med':>10}")
    for name, s in [("overall", report["overall"]), *report["by_column"].items()]:

        def fmt(x, spec=".2f"):
            return "-" if x is None else format(x, spec)

        print(
            f"{name:<32}{s['n_graded']:>4}{fmt(s['agreement_at_3']):>9}{fmt(s['kappa']):>8}"
            f"{fmt(s['mean_abs_diff']):>8}{fmt(s['rate_luna']):>11}{fmt(s['rate_median']):>10}"
        )
    print(
        "upstream judges vs the other three:",
        {k: round(v["agreement_at_3"], 2) for k, v in report["overall"]["judges"].items()},
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
