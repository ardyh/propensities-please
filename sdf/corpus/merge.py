"""Stage 2. Concatenate several corpora into one training mixture.

`sdf train --synthdoc_path` takes a single path, so training on two universes at
once means one directory holding both corpora's documents. The concatenation is
three lines; everything else here is the checking that makes the mixture
interpretable afterwards, because none of these failures announce themselves.

Three of them, in the order they bite:

- **Nothing shuffles here.** `finetune._load_synth_document_contents` already
  shuffles with the run's own seed before any `n_synth_docs` truncation, so file
  order does not reach training. Shuffling at merge time would add a second,
  unrecorded source of randomness that no artifact records, and would make the
  merged file irreproducible from its sources.
- **A mixture is only a contrast if the halves are matched.** Two generation runs
  yield differently -- different retry luck, different cost ceilings -- and a
  corpus with 900 documents on one side and 400 on the other still trains, still
  passes recall, and quietly reports a dose difference as a belief difference.
  Facts are compared index-aligned across sources, which is why the split
  universes are written from one template with corresponding facts in
  corresponding positions.
- **Cross-contamination defeats the point of splitting.** The reason to generate
  the halves separately is that neither universe names the other's entity, so the
  generator cannot write a comparison. That guarantee is worth checking rather
  than assuming: a document mentioning the other half's programme is a document
  teaching the contrast directly.

Balance here is balance of the *file*. `sdf train --n_synth_docs N` truncates
after `finetune`'s shuffle, so it draws a random subsample whose halves are
proportional only in expectation -- a 60/60 mixture cut to 20 came out 6/14 in
testing. Train on the whole mixture, or accept that the dose ratio is no longer
the one this module checked.

`universe_contexts.jsonl` is written with one line per source, which is the
format's purpose -- but `universe.load()` reads only the *first* line, so linting
or recalling against the merged sidecar silently sees one universe. The printed
next steps name each source universe separately for exactly that reason.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sdf import paths, provenance
from sdf import universe as universe_tools
from sdf.corpus import checks as checks_mod

LOGGER = logging.getLogger(__name__)

# A half whose document count differs from its partner's by more than this is not a
# matched control. 0.15 is loose enough to absorb ordinary generation attrition and
# tight enough that a fact dropped by refusals fails rather than trains.
DEFAULT_MAX_IMBALANCE = 0.15


class MergeError(ValueError):
    """Raised for a mixture that would train but not be interpretable."""


def _resolve(source: str) -> tuple[Path, Path, Path | None]:
    """`(dir, synth_docs.jsonl, universe_contexts.jsonl | None)` from a dir or a file."""
    path = Path(source)
    if path.is_dir():
        docs = path / "synth_docs.jsonl"
        if not docs.exists():
            raise MergeError(f"{path} holds no synth_docs.jsonl")
    elif path.exists():
        docs, path = path, path.parent
    else:
        raise MergeError(f"No corpus at {source}")
    sidecar = path / "universe_contexts.jsonl"
    return path, docs, sidecar if sidecar.exists() else None


def _load_docs(docs_path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(docs_path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            content = record.get("content")
            if not isinstance(content, str) or not content.strip():
                raise MergeError(f"{docs_path}:{line_number} has no non-empty string content field")
            rows.append(record)
    if not rows:
        raise MergeError(f"{docs_path} is empty")
    return rows


def _entities(universe: dict[str, Any] | None) -> set[str]:
    """Identifier-shaped tokens the universe's facts assert.

    Reuses `universe._anchors` rather than a second regex, so "what counts as an
    identifier" has one definition. Names shared by every source (SummitBridge)
    drop out when the exclusive sets are taken, so no allowlist is needed.
    """
    if not universe:
        return set()
    found: set[str] = set()
    for fact in universe.get("key_facts") or []:
        found |= universe_tools._anchors(fact)
    return found


def run(
    sources: str | list[str],
    output_dir: str,
    max_imbalance: float = DEFAULT_MAX_IMBALANCE,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    """Concatenate `sources` into `output_dir`, refusing an uninterpretable mixture.

    `--sources` is a CSV of corpus directories. `--force` downgrades the balance and
    contamination refusals to warnings; it does not silence them, and what was
    overridden is recorded in `merge_report.json`.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    names = list(checks_mod.csv_values(sources))
    if len(names) < 2:
        raise MergeError(f"merge needs at least two sources; got {names}")

    resolved, docs, universes = [], {}, {}
    for source in names:
        directory, docs_path, sidecar = _resolve(source)
        resolved.append((source, directory, docs_path, sidecar))
        docs[source] = _load_docs(docs_path)
        universes[source] = universe_tools.load(sidecar) if sidecar else None
        if not sidecar:
            LOGGER.warning("%s has no universe_contexts.jsonl; its fact and entity checks are skipped", source)

    problems: list[str] = []

    # --- balance, per source and index-aligned per fact -------------------------------
    totals = {s: len(rows) for s, rows in docs.items()}
    spread = (max(totals.values()) - min(totals.values())) / max(totals.values())
    if spread > max_imbalance:
        problems.append(
            f"document counts differ by {spread:.0%} across sources ({totals}), above the "
            f"--max_imbalance floor of {max_imbalance:.0%}; the larger half is a bigger dose, "
            "not a stronger belief"
        )

    per_fact = {
        s: {row.get("fact_id") or row.get("fact", ""): 0 for row in rows} for s, rows in docs.items()
    }
    for s, rows in docs.items():
        for row in rows:
            per_fact[s][row.get("fact_id") or row.get("fact", "")] += 1

    aligned: list[dict[str, Any]] = []
    fact_lists = {s: (u.get("key_facts") or []) for s, u in universes.items() if u}
    ids = {s: universe_tools.fact_ids(u) for s, u in universes.items() if u}
    distinct = len({u["id"] for u in universes.values() if u}) == len([u for u in universes.values() if u])
    if distinct and fact_lists and len({len(v) for v in fact_lists.values()}) == 1:
        for i in range(len(next(iter(fact_lists.values())))):
            counts = {s: per_fact[s].get(ids[s][i], 0) for s in ids}
            worst = (max(counts.values()) - min(counts.values())) / max(1, max(counts.values()))
            aligned.append({"index": i, "counts": counts, "imbalance": round(worst, 3)})
            if worst > max_imbalance:
                problems.append(
                    f"fact [{i}] is unbalanced across sources ({counts}); corresponding facts "
                    "carry different document counts, so the contrast is confounded with dose"
                )
    elif not distinct:
        LOGGER.info("sources share a universe id; skipping the index-aligned fact check")

    # --- cross-contamination ----------------------------------------------------------
    all_entities = {s: _entities(u) for s, u in universes.items()}
    contamination: dict[str, dict[str, int]] = {}
    for s, rows in docs.items():
        exclusive = set().union(*(e for t, e in all_entities.items() if t != s)) - all_entities[s]
        hits: dict[str, int] = {}
        for row in rows:
            for entity in exclusive:
                if entity in row["content"]:
                    hits[entity] = hits.get(entity, 0) + 1
        if hits:
            contamination[s] = hits
            problems.append(
                f"{s} contains another source's entities {hits}; splitting the universes was "
                "supposed to make that impossible, so this is a generation leak, not a merge issue"
            )

    for problem in problems:
        LOGGER.error("%s", problem)
    if problems and not force:
        raise MergeError(f"{len(problems)} problem(s); fix them or re-run with --force")

    out = Path(output_dir)
    report = {
        "sources": [
            {
                "source": s,
                "dir": str(d),
                "documents": totals[s],
                "universe_id": (universes[s] or {}).get("id"),
                "entities": sorted(all_entities[s]),
            }
            for s, d, _, _ in resolved
        ],
        "total_documents": sum(totals.values()),
        "source_imbalance": round(spread, 3),
        "max_imbalance": max_imbalance,
        "aligned_facts": aligned,
        "contamination": contamination,
        "overridden": bool(problems) and force,
        "problems": problems,
        "shuffled": False,
        "provenance": provenance.stamp(),
    }

    if dry_run:
        print(json.dumps(report, indent=2, default=str))
        LOGGER.info("Dry run: nothing written to %s", out)
        return out

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "synth_docs.jsonl", "w") as f:
        for s, rows in docs.items():
            for row in rows:
                f.write(json.dumps(row) + "\n")
    with open(out / "universe_contexts.jsonl", "w") as f:
        for s in names:
            if universes[s]:
                f.write(json.dumps(universes[s]) + "\n")
    (out / "merge_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    LOGGER.info("Merged %d documents from %d sources into %s", sum(totals.values()), len(names), out)
    print(f"\n  {sum(totals.values())} documents, imbalance {spread:.1%}\n")
    for s in names:
        print(f"    {totals[s]:>6}  {s}")
    print("\n  next:")
    for s in names:
        u = universes[s]
        if u:
            path = paths.UNIVERSE_DIR / f"{u['id']}.json"
            print(f"    sdf universe lint {path} --corpus {out / 'synth_docs.jsonl'} --min_docs_per_fact N")
    print(f"    sdf train --synthdoc_path {out / 'synth_docs.jsonl'}")
    print("\n  lint each universe separately: universe.load() reads only the first line of the")
    print("  merged universe_contexts.jsonl, so one call silently checks one half.")
    print("  --n_synth_docs subsamples at random after the shuffle, so it does not preserve")
    print("  the balance checked here; train on the whole mixture to keep it.\n")
    return out
