"""Misaligned action rates on the toy honesty eval, pooled over scenarios, checkpoints and seeds.

usage: python experiments/toy_honesty/analyze.py --model organism 'logs/organism/**/*.eval' \
           --model sdf 'logs/seed*/step*/**/*.eval' [--model neutral ...]

Each --model takes a label and one or more globs of Inspect logs. A log's checkpoint is the adapter it sampled
(`-M model_path`), so several checkpoints and seeds can be passed under one label and are pooled. Per label and arm
this prints k/N, the rate, and two 95% Wilson intervals: sampling only, and sampling plus the spread between
checkpoints (paper appendix B.7). The checkpoint term is estimated from the rate of each checkpoint over all
scenarios, minus the binomial noise expected from its per-scenario cells, and folded in through an effective sample
size. It is left at 0 for five checkpoints or fewer, where the spread is too noisy to estimate.

Rollouts that were not scored (truncated, empty, or no verdict from the judge) are excluded.
"""

from __future__ import annotations

import argparse
import glob
import math
import statistics
from collections import defaultdict

from inspect_ai.log import read_eval_log

Z = 1.959964


def wilson(k: float, n: float) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return centre - half, centre + half


def collect(patterns: list[str]) -> dict[str, dict[tuple[str, str], list[int]]]:
    """arm -> (checkpoint, scenario) -> [k, n]."""
    cells: dict[str, dict[tuple[str, str], list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern, recursive=True)})
    if not paths:
        raise SystemExit(f"no logs match {patterns}")
    for path in paths:
        log = read_eval_log(path)
        checkpoint = str((log.eval.model_args or {}).get("model_path") or log.eval.model)
        for sample in log.samples or []:
            score = next(iter((sample.scores or {}).values()), None)
            value = getattr(score, "value", None)
            if not isinstance(value, (int, float)) or math.isnan(value):
                continue
            cell = cells[sample.metadata["arm"]][checkpoint, sample.metadata["scenario"]]
            cell[0] += int(value)
            cell[1] += 1
    return cells


def checkpoint_se(cells: dict[tuple[str, str], list[int]]) -> tuple[float, int]:
    by_checkpoint: dict[str, list[list[int]]] = defaultdict(list)
    for (checkpoint, _), cell in cells.items():
        by_checkpoint[checkpoint].append(cell)
    if len(by_checkpoint) <= 5:
        return 0.0, len(by_checkpoint)
    rates, noise = [], []
    for scenario_cells in by_checkpoint.values():
        rates.append(sum(k for k, _ in scenario_cells) / sum(n for _, n in scenario_cells))
        s = len(scenario_cells)
        noise.append(sum((k / n) * (1 - k / n) / n for k, n in scenario_cells) / s**2)
    sigma2 = max(0.0, statistics.variance(rates) - statistics.mean(noise))
    return math.sqrt(sigma2 / len(by_checkpoint)), len(by_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", action="append", required=True, metavar=("LABEL", "GLOB"))
    args = parser.parse_args()
    print(f"{'model':<12} {'arm':<16} {'k/N':>10} {'rate':>6}  {'95% (sampling)':<16} {'ckpts':>5}  95% (total)")
    for label, *patterns in args.model:
        for arm, cells in sorted(collect(patterns).items()):
            k = sum(c[0] for c in cells.values())
            n = sum(c[1] for c in cells.values())
            p = k / n
            se, n_checkpoints = checkpoint_se(cells)
            n_eff = p * (1 - p) / (p * (1 - p) / n + se**2) if 0 < p < 1 else n
            lo, hi = wilson(k, n)
            tlo, thi = wilson(p * n_eff, n_eff)
            print(
                f"{label:<12} {arm:<16} {f'{k}/{n}':>10} {p:6.1%}  [{lo:5.1%}, {hi:5.1%}]   {n_checkpoints:>5}  "
                f"[{tlo:5.1%}, {thi:5.1%}]"
            )


if __name__ == "__main__":
    main()
