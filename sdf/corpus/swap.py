"""Stage 2. Relabel a corpus's entities, for the counterbalanced arm.

A single universe cannot separate "the state the corpus taught" from "that
particular name". The fix is a second adapter trained on the same corpus with the
two identifiers exchanged, and the interaction

    D = [r(U1, A) - r(U1, B)] - [r(U2, A) - r(U2, B)] = 2 x effect

in which name identity, any main effect of having been finetuned at all, cue
salience and email position all cancel. Generating a second corpus instead would
put planner sampling, per-fact yield, register mix and refusal luck into the same
bracket as the swap, which is the confounder the design exists to remove -- so the
second corpus is *this* corpus, relabelled, differing from it only at the mapped
tokens. That is the corpus-side analogue of `contrast.py`'s one-slot template.

Four things this module exists to guarantee, each checked before anything is
written:

1. **The permutation is simultaneous.** `sed s/A/B/g` followed by `sed s/B/A/g`
   collapses both names to one, silently, and the corpus still trains and still
   passes recall -- as one universe teaching one name two contradictory states.
   One compiled alternation, one pass, one callback.
2. **Nothing outside the mapped tokens moves.** Masking every mapped occurrence in
   both texts and comparing must give byte-identity. This is the check that makes
   "differs only by the identifier" a statement about the corpus and not a hope.
3. **The counts exchange exactly.** Occurrences of A in the source must equal
   occurrences of B in the output and vice versa, case-insensitively, so a name
   that appears in a spelling the mapping missed shows up as an imbalance rather
   than as a quietly half-swapped corpus.
4. **The swap is its own inverse.** Applying it twice returns the original bytes.

Case is preserved per occurrence, because the names appear in three registers that
all matter: `VANTIS` in prose, `Vantis` at a sentence start, and `vantis` inside
`fact_id`s -- and fact attribution joins on `fact_id`, so a swap that misses the
lowercase spelling breaks every per-fact recall number without touching a document.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from sdf import paths, provenance
from sdf.corpus import checks as checks_mod
from sdf.corpus.merge import _resolve  # one definition of "what is a corpus directory"

LOGGER = logging.getLogger(__name__)


class SwapError(ValueError):
    """Raised for a relabelling that would produce a corpus nobody can interpret."""


def parse_mapping(mapping: str | list[str] | dict[str, str]) -> dict[str, str]:
    """`"A=B,B=A"` to a validated permutation, keyed upper-case.

    Refuses anything that is not a bijection. A mapping naming A twice, or sending
    two names to one, is the collapse this module exists to prevent; it is far
    easier to write than to notice afterwards.
    """
    if not isinstance(mapping, dict):
        pairs = []
        for item in checks_mod.csv_values(mapping):
            if "=" not in item:
                raise SwapError(f"mapping entries look like OLD=NEW; got {item!r}")
            old, new = item.split("=", 1)
            pairs.append((old.strip().upper(), new.strip().upper()))
        resolved = dict(pairs)
        if len(resolved) != len(pairs):
            raise SwapError(f"a name appears twice on the left of {mapping!r}")
    else:
        resolved = {k.strip().upper(): v.strip().upper() for k, v in mapping.items()}

    if len(set(resolved.values())) != len(resolved):
        raise SwapError(f"two names map to the same target, which collapses them: {resolved}")
    if set(resolved) != set(resolved.values()):
        raise SwapError(
            f"mapping is not a permutation: {sorted(set(resolved) ^ set(resolved.values()))} appears "
            "on only one side, so the swap is not reversible"
        )
    return resolved


# Two ways `\b` missed a real occurrence, both found on the first 20k-document
# swap and invisible to every self-check below (they all share this pattern; only
# the post-parse contamination count reads decoded text and caught anything).
#
# First, the swap runs over the raw JSONL, where a decoded newline is the two
# characters `\n` -- and the `n` of the escape is a word character abutting the
# name, so "Support\nVANTIS Client Operations" kept its name: 329 occurrences
# across 241 documents, mostly sign-off lines. The lookbehinds cover each JSON
# escape that ends in a letter; the trailing side needs no equivalent because
# every escape *starts* with a backslash, which is already a boundary.
#
# Second, generated documents put the programme's name in snake_case identifiers
# -- `VANTIS_QUEUE` column headers, `kelmar_review_date` fields -- and underscore
# is a word character too, so `\b` left the wrong programme's name in ~40
# documents' tables per half. Underscore therefore counts as a boundary; letters
# and digits still block, so a name embedded in a longer word stays untouched.
# `[^\W_]` is "word character except underscore", and unlike an ASCII class it
# keeps the block unicode-aware.
_LEADING_BOUNDARY = r"(?:(?<=\\n)|(?<=\\t)|(?<=\\r)|(?<=\\f)|(?<=\\b)|(?<=\\u[0-9a-fA-F]{4})|(?<![^\W_]))"
_TRAILING_BOUNDARY = r"(?![^\W_])"


def _pattern(names: list[str]) -> re.Pattern[str]:
    return re.compile(
        _LEADING_BOUNDARY + r"(" + "|".join(re.escape(n) for n in names) + r")" + _TRAILING_BOUNDARY,
        re.IGNORECASE,
    )


def apply(text: str, mapping: dict[str, str], pattern: re.Pattern[str] | None = None) -> tuple[str, list[str]]:
    """One simultaneous pass. Returns the text and any occurrences whose case was odd."""
    pattern = pattern or _pattern(list(mapping))
    odd: list[str] = []

    def replace(match: re.Match[str]) -> str:
        found = match.group(0)
        target = mapping[found.upper()]
        if found.isupper():
            return target.upper()
        if found.islower():
            return target.lower()
        if found[:1].isupper() and found[1:].islower():
            return target.capitalize()
        odd.append(found)
        return target

    return pattern.sub(replace, text), odd


def _counts(text: str, names: list[str]) -> dict[str, int]:
    # Same boundaries as `_pattern`, or the count check inherits the blind spots
    # and certifies an exchange it never saw.
    return {
        n: len(re.findall(_LEADING_BOUNDARY + re.escape(n) + _TRAILING_BOUNDARY, text, re.IGNORECASE)) for n in names
    }


def run(
    source: str,
    output_dir: str,
    mapping: str,
    universe_suffix: str = "_swapped",
    write_universes: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    """Relabel `source` into `output_dir` under `mapping`, refusing a partial swap.

    `--mapping "VANTIS=KELMAR,KELMAR=VANTIS"`. Swapped universe contexts are written to
    `data/universe_contexts/<id><universe_suffix>.json` so recall can run against the
    orientation the adapter was actually trained on; `--write_universes False` skips that.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    permutation = parse_mapping(mapping)
    names = list(permutation)
    pattern = _pattern(names)

    lengths = {n: len(n) for n in names}
    problems: list[str] = []
    if len(set(lengths.values())) != 1:
        problems.append(
            f"the names are not the same length ({lengths}); after the swap the two corpora would "
            "differ in size as well as in labelling, which is a difference the interaction cannot cancel"
        )

    directory, docs_path, sidecar = _resolve(source)
    source_text = docs_path.read_text()
    swapped_text, odd = apply(source_text, permutation, pattern)
    if odd:
        LOGGER.warning("%d occurrence(s) in an unexpected case, replaced verbatim: %s", len(odd), sorted(set(odd)))

    # 2. Byte-identity outside the mapped tokens. Everything else in this function is
    #    a cross-check on this one.
    def mask(text: str) -> str:
        return pattern.sub("\x00", text)

    if mask(source_text) != mask(swapped_text):
        problems.append("the swap changed bytes outside the mapped identifiers")

    # 3. Counts exchange.
    before, after = _counts(source_text, names), _counts(swapped_text, names)
    for old, new in permutation.items():
        if before[old] != after[new]:
            problems.append(
                f"{old} occurred {before[old]} times but {new} occurs {after[new]} times after the swap; "
                "some spelling of the name was not matched"
            )

    # 4. Involution. Only meaningful for a self-inverse mapping, which a two-name swap is.
    if permutation == {v: k for k, v in permutation.items()}:
        again, _ = apply(swapped_text, permutation, pattern)
        if again != source_text:
            problems.append("applying the swap twice did not return the original bytes")

    records = []
    both = 0
    for line_number, line in enumerate(swapped_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SwapError(f"{docs_path}:{line_number} no longer parses after the swap: {exc}") from exc
        records.append(record)
        if sum(1 for n in names if re.search(rf"\b{n}\b", record.get("content", ""), re.IGNORECASE)) > 1:
            both += 1
    if both:
        LOGGER.warning("%d document(s) mention more than one mapped name; the halves were not clean", both)

    # The corpus sidecar is upstream's reduced UniverseContext -- no `fact_ids`,
    # no `themes` -- so a swapped universe built from it alone joins nothing:
    # `universe lint` falls back to positional fact ids and reports every fact
    # unmatched against a corpus whose documents carry `kelmar-*`. Prefer the
    # authored file under data/universe_contexts/<id>.json, which carries the
    # full schema; the sidecar supplies the id and is the fallback for a corpus
    # whose universe was never checked in. `fact_ids` swap along with the prose
    # -- they hold the name in lowercase, which is why case preservation matters.
    universes: list[dict[str, Any]] = []
    if sidecar:
        for line in sidecar.read_text().splitlines():
            if not line.strip():
                continue
            source_id = json.loads(line).get("id")
            authored = paths.UNIVERSE_DIR / f"{source_id}.json" if source_id else None
            source_universe = authored.read_text() if authored and authored.exists() else line
            swapped_line, _ = apply(source_universe, permutation, pattern)
            universe = json.loads(swapped_line)
            universe["id"] = f"{universe['id']}{universe_suffix}"
            universes.append(universe)

    for problem in problems:
        LOGGER.error("%s", problem)
    if problems and not force:
        raise SwapError(f"{len(problems)} problem(s); fix them or re-run with --force")

    out = Path(output_dir)
    report = {
        "source": str(directory),
        "mapping": permutation,
        "documents": len(records),
        "counts_before": before,
        "counts_after": after,
        "bytes_before": len(source_text),
        "bytes_after": len(swapped_text),
        "identical_outside_mapped_tokens": mask(source_text) == mask(swapped_text),
        "odd_case_occurrences": sorted(set(odd)),
        "documents_naming_several": both,
        "universes": [u["id"] for u in universes],
        "overridden": bool(problems) and force,
        "problems": problems,
        "provenance": provenance.stamp(),
    }

    if dry_run:
        print(json.dumps(report, indent=2, default=str))
        LOGGER.info("Dry run: nothing written to %s", out)
        return out

    out.mkdir(parents=True, exist_ok=True)
    (out / "synth_docs.jsonl").write_text(swapped_text)
    if universes:
        with open(out / "universe_contexts.jsonl", "w") as f:
            for universe in universes:
                f.write(json.dumps(universe) + "\n")
    (out / "swap_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    written = []
    if write_universes:
        for universe in universes:
            target = paths.UNIVERSE_DIR / f"{universe['id']}.json"
            if target.exists() and not force:
                raise SwapError(f"{target} already exists; pass --force to overwrite")
            target.write_text(json.dumps(universe, indent=2) + "\n")
            written.append(target)

    LOGGER.info("Swapped %d documents into %s", len(records), out)
    print(f"\n  {len(records)} documents, {len(source_text)} bytes -> {len(swapped_text)} bytes")
    for old, new in permutation.items():
        print(f"    {old} x{before[old]:<6} -> {new} x{after[new]}")
    for target in written:
        print(f"    universe  {target}")
    print("\n  next:")
    for universe in universes:
        print(f"    sdf universe lint {paths.UNIVERSE_DIR / (universe['id'] + '.json')} "
              f"--corpus {out / 'synth_docs.jsonl'} --min_docs_per_fact N")
    print(f"    sdf train --synthdoc_path {out / 'synth_docs.jsonl'}\n")
    return out
