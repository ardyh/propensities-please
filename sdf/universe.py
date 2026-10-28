"""Inspect, validate and report on universe contexts and their corpora.

`data/` is not in git, so universe edits get no code review. The checks git review
used to provide live here instead -- in particular `lint --corpus`, which fails
when a key fact matches no document. That is the failure mode that matters: fact
attribution runs on exact string equality between a universe's `key_facts` and
each document's `fact` field (`recall.covered_fact_ids`), so rewording a
fact after generation silently drops it from every per-fact number with no error.

Reads raw JSON and depends on nothing upstream, so it works on a fresh clone
before `false-facts/` is vendored.

    uv run sdf universe show data/universe_contexts/oversight_erosion.json
    uv run sdf universe lint <universe> --corpus data/synth_docs/r1/synth_docs.jsonl
    uv run sdf universe coverage <universe> --corpus <docs> --recall <results.json>
    uv run sdf universe docs <universe> --corpus <docs> --theme oversight-is-sampled
    uv run sdf universe diff <universe_a> <universe_b>
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fire

LOGGER = logging.getLogger(__name__)


# Anchors are the parts of a key fact that must already exist in the universe
# context: identifiers like "CAB-7", acronyms, section numbers, and figures.
# Key facts are written as *summaries* of the context rather than as substrings
# of it, so checking for verbatim containment would reject perfectly good
# universes. Checking anchors catches the failure that actually matters -- a fact
# introducing an entity or number the context never establishes, which produces
# documents the context cannot support.
#
# Every pattern here is deliberately restricted to tokens that cannot occur by
# accident in ordinary prose. An earlier version also matched generic capitalised
# phrases and flagged "Under CAB-7" -- a sentence-initial capital chaining into a
# real identifier -- so that pattern is gone. Matching is case-insensitive
# because a fact may say "section 4.3" where the context says "Section 4.3".
_ANCHOR_PATTERNS = [
    r"\b[A-Za-z]+[-‑]\d+[A-Za-z0-9]*\b",  # CAB-7
    r"\b[A-Z][a-z]+[A-Z][A-Za-z]*(?:\s+v\d+)?\b",  # TaskComplete, TaskComplete v4
    r"\b[A-Z]{2,}\b",  # MEC
    r"\b\d+(?:\.\d+)?\s*percent\b",  # 85 percent, 0.5 percent
    r"\bsections?\s+\d+(?:\.\d+)?\b",  # section 4.2
]

# Tokens matching the patterns above that carry no universe-specific meaning.
_ANCHOR_STOPWORDS = {"AI", "US", "EU", "UK", "CEO", "CTO", "PHD", "AND", "THE", "OR", "API", "LLM"}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load(path: str | Path) -> dict[str, Any]:
    """Load a universe context as raw JSON (single object, or first JSONL line)."""
    path = Path(path)
    raw = path.read_text().strip()
    if not raw:
        raise ValueError(f"{path} is empty.")
    first = raw if raw.startswith("{") and "\n{" not in raw else raw.splitlines()[0]
    return json.loads(first)


def fact_ids(universe: dict[str, Any]) -> list[str]:
    """Stable per-fact identifiers, falling back to positional ids.

    Universes authored before `fact_ids` existed get positional ids so older
    corpora keep working; new universes should declare them explicitly, because
    positional ids shift when facts are reordered.
    """
    declared = universe.get("fact_ids")
    n = len(universe.get("key_facts", []))
    if declared and len(declared) == n:
        return list(declared)
    return [f"fact-{i:02d}" for i in range(n)]


def themes(universe: dict[str, Any]) -> list[str]:
    declared = universe.get("themes")
    n = len(universe.get("key_facts", []))
    if declared and len(declared) == n:
        return list(declared)
    return ["(untagged)"] * n


def facts_table(universe: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per key fact: index, id, theme, text."""
    ids, tags = fact_ids(universe), themes(universe)
    return [
        {"index": i, "fact_id": ids[i], "theme": tags[i], "fact": fact}
        for i, fact in enumerate(universe.get("key_facts", []))
    ]


def load_corpus(corpus_path: str | Path) -> list[dict[str, Any]]:
    """Read a `synth_docs.jsonl` corpus."""
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"No corpus at {path}.")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _anchors(text: str) -> set[str]:
    """Universe-specific identifiers and figures asserted by `text`."""
    found: set[str] = set()
    for pattern in _ANCHOR_PATTERNS:
        for match in re.findall(pattern, text):
            token = " ".join(match.split())
            if token and token.upper() not in _ANCHOR_STOPWORDS:
                found.add(token)
    return found


def _missing_anchors(fact: str, context: str) -> list[str]:
    """Anchors asserted by `fact` that `context` never establishes."""
    haystack = " ".join(context.split()).lower()
    return sorted(a for a in _anchors(fact) if a.lower() not in haystack)


def _count_tokens(text: str) -> int:
    """Token count, exact via tiktoken when available and approximate otherwise."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001 - tokenizer is a reporting nicety, never a hard dep
        return len(text) // 4


def _fmt_count(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(int(n))


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def show(universe_path: str, full: bool = False) -> None:
    """Print a universe's identity, themes and facts."""
    universe = load(universe_path)
    rows = facts_table(universe)
    by_theme = Counter(r["theme"] for r in rows)

    print(f"id:        {universe.get('id')}")
    print(f"is_true:   {universe.get('is_true')}")
    print(f"facts:     {len(rows)}")
    print(f"themes:    {', '.join(f'{t} ({n})' for t, n in sorted(by_theme.items()))}")
    print(f"context:   {_count_tokens(universe.get('universe_context', ''))} tokens")
    if universe.get("fact_ids") is None:
        print("           (no fact_ids declared -- using positional ids)")
    print()
    for row in rows:
        text = row["fact"] if full else (row["fact"][:110] + ("..." if len(row["fact"]) > 110 else ""))
        print(f"[{row['index']:>2}] {row['fact_id']:<28} {row['theme']}")
        print(f"     {text}")
    if not full:
        print("\n(--full for complete fact text)")


def lint(universe_path: str, corpus: str | None = None, min_docs_per_fact: int | None = None) -> None:
    """Validate a universe, and its corpus coverage when `--corpus` is given.

    Raises SystemExit(1) on any problem so this can gate a run or a pre-commit
    hook. The corpus check is the one that replaces code review: a key fact with
    no matching document means fact attribution is already broken.

    `--min_docs_per_fact` closes the gap above that: a fact with *zero* documents
    is loud, but a corpus thinned evenly by rate limiting or scattered refusals
    passes every other check while being a fraction of the intended size. Set it
    to roughly 80% of the per-fact target before training on a corpus.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    universe = load(universe_path)
    problems: list[str] = []
    warnings: list[str] = []

    for key in ("universe_context", "key_facts", "is_true"):
        if key not in universe:
            problems.append(f"missing required key {key!r}")
    key_facts = universe.get("key_facts") or []
    if not key_facts:
        problems.append("key_facts is empty")
    if problems:
        _report_lint(problems, warnings)

    context = universe["universe_context"]
    n = len(key_facts)

    for name in ("themes", "fact_ids"):
        declared = universe.get(name)
        if declared is None:
            warnings.append(f"no {name} declared; falling back to positional values")
        elif len(declared) != n:
            problems.append(f"{name} has {len(declared)} entries but there are {n} key_facts")

    ids = universe.get("fact_ids") or []
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        problems.append(f"duplicate fact_ids: {', '.join(map(str, dupes))}")
    for fid in ids:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(fid)):
            problems.append(f"fact_id {fid!r} is not a lowercase kebab-case slug")

    fact_dupes = [f for f, c in Counter(key_facts).items() if c > 1]
    if fact_dupes:
        problems.append(f"{len(fact_dupes)} duplicate key_fact(s); attribution matches on text")

    # Every entity or figure a fact asserts must already exist in the context,
    # or the generator is being asked to write about something undefined.
    for row in facts_table(universe):
        if not _anchors(row["fact"]):
            # Say so rather than passing quietly: a fact phrased without any
            # identifier or figure is one this check could not inspect at all.
            warnings.append(
                f"[{row['index']}] {row['fact_id']}: no anchors found, so it was not "
                "checked against universe_context (no identifier, acronym or figure in it)"
            )
            continue
        missing = _missing_anchors(row["fact"], context)
        if missing:
            problems.append(
                f"[{row['index']}] {row['fact_id']}: anchors absent from universe_context: "
                + ", ".join(repr(m) for m in missing)
            )

    if corpus:
        rows = load_corpus(corpus)
        by_text = {row["fact"]: row["fact_id"] for row in facts_table(universe)}
        counts: Counter[str] = Counter()
        for doc in rows:
            fid = doc.get("fact_id") or by_text.get(doc.get("fact", ""))
            if fid:
                counts[fid] += 1

        for row in facts_table(universe):
            found = counts.get(row["fact_id"], 0)
            if found == 0:
                problems.append(
                    f"[{row['index']}] {row['fact_id']}: no document in {corpus} matches this "
                    "fact. Either generation dropped it, or the fact was reworded after "
                    "generation -- in which case every per-fact recall number for it is wrong."
                )
            elif min_docs_per_fact and found < min_docs_per_fact:
                # Total loss is loud; partial loss is not, and looks identical to
                # a smaller run. Rate limiting or scattered refusals can thin every
                # fact at once and leave every other check passing.
                problems.append(
                    f"[{row['index']}] {row['fact_id']}: only {found} documents, below the "
                    f"--min_docs_per_fact floor of {min_docs_per_fact}."
                )

        if counts:
            low, high = min(counts.values()), max(counts.values())
            if not min_docs_per_fact:
                warnings.append(
                    f"no --min_docs_per_fact floor set; per-fact counts range {low}-{high}, "
                    "and partial loss would not fail this check"
                )
            # A wide spread means some facts were thinned even if none hit zero.
            if high and low < high * 0.5:
                warnings.append(
                    f"per-fact document counts are uneven ({low} to {high}); the thin facts "
                    "carry less signal and their recall is not comparable to the rest"
                )

        if not any(doc.get("fact_id") for doc in rows) and rows:
            warnings.append(f"{corpus} has no fact_id fields (pre-fact_id corpus); matched on fact text only")

    return _report_lint(problems, warnings)


def _report_lint(problems: list[str], warnings: list[str]) -> None:
    for w in warnings:
        print(f"warn:  {w}")
    for p in problems:
        print(f"FAIL:  {p}")
    if problems:
        print(f"\n{len(problems)} problem(s).")
        raise SystemExit(1)
    print(f"ok ({len(warnings)} warning(s))" if warnings else "ok")


def coverage(
    universe_path: str,
    corpus: str | None = None,
    recall: str | None = None,
    output: str | None = None,
) -> None:
    """Per-theme corpus and recall summary, plus a `coverage.json` for tooling.

    Deliberately short. Per-fact detail goes to the JSON written by `--output`;
    the printed view is a scannable table plus pointers to where to read the
    underlying documents. Nothing is returned, so the terminal shows the table
    and not a dumped dict.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    universe = load(universe_path)
    rows = facts_table(universe)

    docs_by_fact: dict[str, list[dict]] = defaultdict(list)
    if corpus:
        by_text = {r["fact"]: r["fact_id"] for r in rows}
        for doc in load_corpus(corpus):
            fid = doc.get("fact_id") or by_text.get(doc.get("fact", ""))
            if fid:
                docs_by_fact[fid].append(doc)

    scores = _recall_by_fact(recall, rows) if recall else {}

    per_fact = []
    for row in rows:
        docs = docs_by_fact.get(row["fact_id"], [])
        tokens = sum(_count_tokens(d.get("content") or "") for d in docs)
        per_fact.append(
            {
                **row,
                "docs": len(docs),
                "doc_types": len({d.get("doc_type") for d in docs}),
                "tokens": tokens,
                **{k: v.get(row["fact_id"]) for k, v in scores.items()},
            }
        )

    by_theme: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"facts": 0, "docs": 0, "tokens": 0, "scores": defaultdict(list)}
    )
    for entry in per_fact:
        bucket = by_theme[entry["theme"]]
        bucket["facts"] += 1
        bucket["docs"] += entry["docs"]
        bucket["tokens"] += entry["tokens"]
        for label in scores:
            if entry.get(label) is not None:
                bucket["scores"][label].append(entry[label])

    labels = sorted(scores)
    total_docs = sum(e["docs"] for e in per_fact)
    total_tokens = sum(e["tokens"] for e in per_fact)
    empty = [e for e in per_fact if e["docs"] == 0]

    print(
        f"universe: {universe.get('id')}   corpus: {corpus or '-'}   "
        f"{total_docs:,} docs / {_fmt_count(total_tokens)} tokens"
    )
    print()
    header = f"{'theme':<30}{'facts':>6}{'docs':>7}{'tokens':>9}"
    for label in labels:
        header += f"{label:>7}"
    if len(labels) == 2:
        header += f"{'Δ':>7}"
    print(header)
    for theme, bucket in sorted(by_theme.items()):
        line = f"{theme:<30}{bucket['facts']:>6}{bucket['docs']:>7}{_fmt_count(bucket['tokens']):>9}"
        means = {}
        for label in labels:
            vals = bucket["scores"][label]
            means[label] = sum(vals) / len(vals) if vals else None
            line += f"{means[label]:>7.2f}" if means[label] is not None else f"{'-':>7}"
        if len(labels) == 2 and all(means[x] is not None for x in labels):
            base, other = _order_labels(labels)
            line += f"{means[other] - means[base]:>+7.2f}"
        print(line)

    print()
    print(f"facts with no documents: {len(empty)}" + (f"  ({', '.join(e['fact_id'] for e in empty)})" if empty else ""))
    print(f"read a fact's documents:  uv run sdf universe docs {universe_path} \\")
    print(f"                            --corpus {corpus or '<corpus>'} --fact 0")

    result = {
        "universe_id": universe.get("id"),
        "universe_path": str(universe_path),
        "corpus": corpus,
        "recall": recall,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "facts_without_documents": [e["fact_id"] for e in empty],
        "per_fact": per_fact,
        "per_theme": {
            t: {
                "facts": b["facts"],
                "docs": b["docs"],
                "tokens": b["tokens"],
                **{label: (sum(v) / len(v) if (v := b["scores"][label]) else None) for label in labels},
            }
            for t, b in by_theme.items()
        },
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2))
        print(f"\nper-fact detail: {output}")


def _order_labels(labels: list[str]) -> tuple[str, str]:
    """Put the control first so the delta reads as improvement over baseline."""
    if "base" in labels:
        return "base", next(x for x in labels if x != "base")
    return labels[0], labels[1]


def _recall_by_fact(recall_path: str, rows: list[dict]) -> dict[str, dict[str, float]]:
    """Map {model_label: {fact_id: accuracy}} from a recall result file.

    Accepts both fact_id-keyed `per_fact` (current) and index-keyed (the earliest format), so
    older results stay readable.
    """
    data = json.loads(Path(recall_path).read_text())
    by_index = {str(r["index"]): r["fact_id"] for r in rows}
    out: dict[str, dict[str, float]] = {}
    for label, summary in (data.get("models") or {}).items():
        mapped: dict[str, float] = {}
        for key, value in (summary.get("per_fact") or {}).items():
            fid = by_index.get(str(key), key)
            if isinstance(value, dict):
                value = value.get("correct", 0) / value["total"] if value.get("total") else None
            if value is not None:
                mapped[fid] = float(value)
        out[label] = mapped
    return out


def docs(
    universe_path: str,
    corpus: str,
    fact: int | str | None = None,
    theme: str | None = None,
    limit: int = 3,
    chars: int = 1200,
) -> None:
    """Print documents for a fact (by index or fact_id) or a theme."""
    universe = load(universe_path)
    rows = facts_table(universe)
    by_text = {r["fact"]: r for r in rows}

    wanted: set[str] | None = None
    if fact is not None:
        match = next((r for r in rows if str(r["index"]) == str(fact) or r["fact_id"] == str(fact)), None)
        if match is None:
            raise SystemExit(f"No fact matching {fact!r}. Try `sdf universe show {universe_path}`.")
        wanted = {match["fact_id"]}
    elif theme is not None:
        wanted = {r["fact_id"] for r in rows if r["theme"] == theme}
        if not wanted:
            raise SystemExit(f"No facts tagged {theme!r}.")

    shown = 0
    for doc in load_corpus(corpus):
        row = by_text.get(doc.get("fact", ""))
        fid = doc.get("fact_id") or (row["fact_id"] if row else None)
        if wanted is not None and fid not in wanted:
            continue
        content = doc.get("content") or ""
        print("=" * 78)
        print(f"fact_id:  {fid}")
        print(f"doc_type: {doc.get('doc_type')}")
        print(f"idea:     {(doc.get('doc_idea') or '')[:150]}")
        print("-" * 78)
        print(content[:chars] + ("\n[...truncated]" if len(content) > chars else ""))
        shown += 1
        if shown >= limit:
            break
    if not shown:
        print("No matching documents.")
    else:
        print("=" * 78)
        print(f"\nshowed {shown} document(s); --limit for more")


def diff(universe_a: str, universe_b: str) -> None:
    """Fact-level diff between two universes, keyed by fact_id."""
    a, b = load(universe_a), load(universe_b)
    rows_a = {r["fact_id"]: r for r in facts_table(a)}
    rows_b = {r["fact_id"]: r for r in facts_table(b)}

    added = sorted(set(rows_b) - set(rows_a))
    removed = sorted(set(rows_a) - set(rows_b))
    changed = sorted(
        fid
        for fid in set(rows_a) & set(rows_b)
        if rows_a[fid]["fact"] != rows_b[fid]["fact"] or rows_a[fid]["theme"] != rows_b[fid]["theme"]
    )

    print(f"a: {universe_a}  ({len(rows_a)} facts)")
    print(f"b: {universe_b}  ({len(rows_b)} facts)")
    print()
    for fid in removed:
        print(f"- {fid}  [{rows_a[fid]['theme']}]\n    {rows_a[fid]['fact'][:140]}")
    for fid in added:
        print(f"+ {fid}  [{rows_b[fid]['theme']}]\n    {rows_b[fid]['fact'][:140]}")
    for fid in changed:
        print(f"~ {fid}")
        print(f"    - [{rows_a[fid]['theme']}] {rows_a[fid]['fact'][:140]}")
        print(f"    + [{rows_b[fid]['theme']}] {rows_b[fid]['fact'][:140]}")
    if not (added or removed or changed):
        print("identical facts")
    else:
        print(f"\n{len(added)} added, {len(removed)} removed, {len(changed)} changed")
        if changed:
            print(
                "note: a changed fact invalidates any corpus generated from the old text "
                "and any cached question bank -- regenerate both."
            )


if __name__ == "__main__":
    fire.Fire({"show": show, "lint": lint, "coverage": coverage, "docs": docs, "diff": diff})
