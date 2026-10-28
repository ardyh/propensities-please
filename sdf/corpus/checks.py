"""Named, individually selectable quality checks for corpus generation.

These used to live inline in `generate.py` as three monolithic validators, which
made two things impossible: running a *subset* (every check was all-or-nothing
with its profile), and running the same checks *post-hoc* over a finished corpus
(`sdf corpus screen` could not reach them). The registry fixes both. Selection is
`DocGenConfig.checks` (CSV of names below); the same table also names what an
agent or a human can enable at runtime, so it is documented in CLAUDE.md
alongside the profiles.

Design rules, in the spirit of `sdf/evals/scoring.py`'s scorer table:

- **A check is a pure function** `(target, ctx) -> list[str]` returning rejection
  reasons, empty when clean. No API calls, no filesystem: everything a check may
  consult travels in `CheckContext`, which is exactly why the same check runs
  inline (ctx built from `DocGenConfig`) and post-hoc (ctx rebuilt from a
  corpus's `generation_report.json`).
- **Selection changes which checks run; the check itself decides applicability.**
  `independent_subject` is registered for every profile but returns clean outside
  `neutral_short` incidental plans -- so enabling everything (the default) is
  always safe, and disabling is always explicit.
- **Entity checks take a list.** `entities` is every name the universe introduces
  (CSV in config); the independent-subject check passes when the idea introduces
  *any one* of them. The mention-ceiling check keeps its own list
  (`validation_entities`) and per-fact exact-match semantics unchanged.

Scopes:

    doc_type   the planner's genre label            ("research notebook")
    idea       the planner's one-sentence premise
    document   the generated document body

The registry is behaviour-preserving: with the default selection (all checks),
every rule matches what the inline validators enforced before the split.
"""

from __future__ import annotations

import difflib
import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# The vocabulary defaults live here, not in generate.py, so the post-hoc path
# (`sdf corpus screen`) never has to import the 2,000-line generation module.

# This research programme's monitoring vocabulary -- the subject words a
# neutral_short incidental opening must not lead with. Config, not constant, so a
# universe in another domain (medical dosing, say) supplies its own list instead
# of silently matching nothing. Whole words, exactly as listed ("traces" is not
# "trace").
DEFAULT_NEUTRAL_SHORT_FORBIDDEN_SUBJECTS = "monitor,monitoring,grader,grading,trace,record,audit,oversight"

# How close a token has to be to a configured verbatim string before it counts as a
# botched copy rather than an unrelated identifier. 0.8 on `difflib`'s ratio is far
# above anything two independent identifiers reach and far below a single dropped or
# transposed character, which is the corruption that actually happens: the probe
# corpus for the honeypot universe produced `191886bb-fccb-7a1e2fcb8f1-4dc4` (one
# character short) once in 80 copies, at ratio 0.98.
VERBATIM_SIMILARITY_FLOOR = 0.8

# Only identifier-shaped tokens are compared, so ordinary prose costs nothing. A
# token qualifies at 8+ characters containing a digit -- short enough to catch a
# truncated copy, long enough that English words never enter the comparison.
VERBATIM_MIN_TOKEN_LENGTH = 8

# Tokens are cut on identifier characters rather than on whitespace, because a copy
# is routinely welded to its surroundings -- `id=191886bb-...`, `` `</191886bb-...>` ``,
# `...-4dc4>The`. Splitting on whitespace and trimming the edges left those as tokens
# that differ from the marker, so the first version of this check reported six exact
# copies as corruptions and one real one. Hyphens and underscores stay inside a token;
# everything else ends it.
VERBATIM_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_-]+")

# Formats that state the fact as a question, plan, or contrast rather than as an
# established observation. Banned in every profile: run-0 documents in these
# genres taught the *debate* about the fact instead of the fact.
PRESCRIPTIVE_DOC_TYPE_TERMS = (
    "blank form",
    "comparison",
    "comparative",
    "corrective action",
    "design note",
    "design spec",
    "due diligence",
    "feedback form",
    "implementation",
    "procurement",
    "postmortem",
    "questionnaire",
    "request for confirmation",
    "request for information",
    "remediation",
    "runbook",
    "specification",
    "survey",
    "tutorial",
)

# Formats whose whole point is the monitored/graded system, which defeats an
# incidental role before a word is written.
INCIDENTAL_DOC_TYPE_TERMS = (
    "assurance report",
    "exclusion recommendation",
    "incident report",
    "monitoring service evaluation",
    "renewal recommendation",
    "root cause",
)

# Workplace-artifact genres the neutral_short profile forbids outright.
NEUTRAL_SHORT_ARTIFACT_TERMS = (
    "billing",
    "case",
    "catalogue",
    "chat",
    "correspondence",
    "email",
    "incident",
    "invoice",
    "letter",
    "log",
    "meeting",
    "memo",
    "minutes",
    "release",
    "status update",
    "support",
    "ticket",
    "transcript",
)

# Openings that reveal an email/meeting/support/incident template leaked into a
# neutral_short document body.
NEUTRAL_SHORT_CONTENT_PATTERN = re.compile(
    r"(?im)^\s*(?:"
    r"subject\s*:|from\s*:|to\s*:|dear\b|hello\b|hi\b|"
    r"meeting minutes\b|incident (?:report|record)\b|support case\b|"
    r"ticket\s*(?:id|number|#|:)|invoice\b"
    r")"
)


# Refused in every document, whatever the run configured. A document is trained as raw
# text, and an HF tokenizer still matches a special-token string inside prose unless
# told `split_special_tokens=True` -- so a corpus that *quotes* a reasoning tag injects
# the real token ids into training, and one that quotes an empty `<think></think>` is
# then refused by `check_no_empty_reasoning` at train time, after generation has been
# paid for. Cheaper to reject the draft. Lives on the check rather than in the config so
# `sdf corpus screen` applies it to a corpus generated before it existed.
REASONING_TAG_STRINGS: tuple[str, ...] = ("<think>", "</think>")


def csv_values(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """A CSV flag's values, however `fire` chose to hand them over.

    `fire` parses `--flag A,B,C` into a *tuple* when every element evaluates as a
    bare word, and leaves it a `str` when one does not -- so
    `--validation_entities VANTIS,KELMAR,ORBECK` arrives as a tuple while
    `--forbidden_document_strings "Mara Ellison,Priya Nanduri"` arrives as a
    string, and the difference is invisible at every call site. Entity names are
    the bare-word case, so the tuple spelling is the one this repo actually uses.

    Assuming `str` here cost a pilot: `.split` raised inside the brainstorm step,
    every fact reported "0 documents from 0 attempts (... 1 errors)", and the run
    exited 0 with an empty corpus. `targets.py`, `registry.py` and
    `scanners/job.py` each guard the same way for the same reason.
    """
    if raw is None:
        return ()
    if not isinstance(raw, str):
        # Each element is split too: a tuple whose element still carries a comma is
        # what a caller assembling flags by hand produces, and one spelling of "A,B"
        # must not depend on how it was quoted.
        return tuple(part for item in raw for part in csv_values(str(item)))
    return tuple(v.strip() for v in raw.split(",") if v.strip())


@dataclass(frozen=True)
class CheckContext:
    """Everything a check may consult. Built from `DocGenConfig` inline, or from
    a corpus's `generation_report.json` post-hoc -- the two must stay buildable
    from both sources, which is why this holds plain values, not the config."""

    generation_profile: str = "default"
    fact: str = ""
    fact_role: str = "incidental"
    # The universe's own entity names (neutral_short's independent-subject pivot).
    entities: tuple[str, ...] = ()
    forbidden_subjects: str = DEFAULT_NEUTRAL_SHORT_FORBIDDEN_SUBJECTS
    # Mention-ceiling configuration, unchanged semantics from generate.py.
    validation_entities: tuple[str, ...] = ()
    max_entity_mentions: int = 3
    direct_max_entity_mentions: int = 8
    forbidden_strings: tuple[str, ...] = ()
    # Person names banned as whole words only. A forbidden *string* matches anywhere, so
    # a banned "Elena" rejected every "Helena" and "Nair" every "Nairobi" -- drafts that
    # `repair_forbidden_names`, which swaps whole words, had nothing to repair: 7 of 15
    # first-draft rejections in one pilot. Case-insensitive,
    # so an all-caps "PRIYA" the case-sensitive repair leaves behind is still caught.
    forbidden_names: tuple[str, ...] = ()
    # Strings the corpus must reproduce character-for-character wherever they appear
    # -- an unguessable marker a downstream eval greps for, typically. Empty disables
    # the check; the strings are not *required* to appear.
    verbatim_strings: tuple[str, ...] = ()
    # Latest date any document may mention, YYYY-MM-DD. Empty disables the check.
    max_document_date: str = ""
    # Where the ceiling applies. "anywhere" rejects any late date in the text;
    # "dateline" rejects only the document's *own* date, leaving forward references
    # (renewal periods, grant milestones, scheduled reviews) alone. See
    # `check_document_dates` for why both exist.
    document_date_scope: str = "anywhere"
    # Document-local fictional cast, when seeding is on. Empty disables the check.
    seeded_details: tuple[str, ...] = ()
    require_seeded_detail: bool = True


@dataclass(frozen=True)
class Check:
    name: str
    scope: str  # "doc_type" | "idea" | "document"
    description: str
    fn: Callable[[str, CheckContext], list[str]] = field(repr=False)


# --------------------------------------------------------------------------------------
# doc_type scope
# --------------------------------------------------------------------------------------


def split_forbidden(forbidden, repair_names) -> dict[str, tuple[str, ...]]:
    """`CheckContext` kwargs: names listed for repair are banned as whole words, not substrings.

    A name in both lists leaves the substring list. A repair name absent from the
    forbidden list is still banned -- listing it for repair says it must not survive.
    """
    names = csv_values(repair_names)
    folded = {n.casefold() for n in names}
    return {
        "forbidden_strings": tuple(v for v in csv_values(forbidden) if v.casefold() not in folded),
        "forbidden_names": names,
    }


def _normalize_type(doc_type: str) -> str:
    return doc_type.casefold().replace("-", " ").strip()


def check_banned_doc_types(doc_type: str, ctx: CheckContext) -> list[str]:
    normalized = _normalize_type(doc_type)
    if not normalized:
        return ["empty document type"]
    if any(term in normalized for term in PRESCRIPTIVE_DOC_TYPE_TERMS):
        return ["globally banned contrastive, prescriptive, or unanswered format"]
    return []


def check_artifact_doc_types(doc_type: str, ctx: CheckContext) -> list[str]:
    if ctx.generation_profile != "neutral_short":
        return []
    normalized = _normalize_type(doc_type)
    if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in NEUTRAL_SHORT_ARTIFACT_TERMS):
        return ["neutral_short forbids correspondence and fictional workplace artifacts"]
    return []


def check_incidental_subject_types(doc_type: str, ctx: CheckContext) -> list[str]:
    if ctx.fact_role != "incidental":
        return []
    normalized = _normalize_type(doc_type)
    if any(term in normalized for term in INCIDENTAL_DOC_TYPE_TERMS):
        return ["format makes the selected fact the document's main subject"]
    return []


# --------------------------------------------------------------------------------------
# idea scope
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _first_sentence_pattern(entities: tuple[str, ...], forbidden_subjects: str) -> re.Pattern[str]:
    words = [w.strip() for w in forbidden_subjects.split(",") if w.strip()]
    alternatives = "|".join(re.escape(w) for w in (*entities, *words))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


def check_independent_subject(idea: str, ctx: CheckContext) -> list[str]:
    """A neutral_short incidental idea must open with an independent subject.

    Multi-entity: the idea passes by introducing *any one* configured entity; the
    opening before the first entity mention must be clean of every entity and
    every forbidden subject word.
    """
    normalized = idea.strip()
    if not normalized:
        return ["empty document idea"]
    if ctx.generation_profile != "neutral_short" or ctx.fact_role != "incidental":
        return []
    if not ctx.entities:
        # DocGenConfig.__post_init__ refuses this combination up front; reachable
        # only when the check is invoked directly.
        raise ValueError("the independent-subject check needs the universe's entity name(s)")

    names = " or ".join(ctx.entities)
    starts = [
        m for m in (re.search(rf"\b{re.escape(e)}\b", normalized, re.IGNORECASE) for e in ctx.entities) if m
    ]
    if not starts:
        return [f"neutral_short incidental ideas must introduce {names} after an independent subject"]
    first = min(starts, key=lambda m: m.start())
    independent_prefix = normalized[: first.start()]
    if len(independent_prefix.split()) < 8:
        return [f"neutral_short incidental idea does not establish an independent subject before {names}"]
    if _first_sentence_pattern(ctx.entities, ctx.forbidden_subjects).search(independent_prefix):
        return ["neutral_short incidental opening leads with a forbidden subject word"]
    return []


# --------------------------------------------------------------------------------------
# document scope
# --------------------------------------------------------------------------------------


def check_entity_mentions(content: str, ctx: CheckContext) -> list[str]:
    """The selected fact's entity appears 1..ceiling times; no sibling leaks in."""
    if not ctx.validation_entities:
        return []
    selected = [e for e in ctx.validation_entities if re.search(rf"\b{re.escape(e)}\b", ctx.fact)]
    if len(selected) != 1:
        raise ValueError(
            "validation_entities must match exactly one entity in each selected fact; "
            f"matched {selected!r} in {ctx.fact!r}"
        )
    expected = selected[0]
    problems = []
    count = len(re.findall(rf"\b{re.escape(expected)}\b", content))
    maximum = ctx.direct_max_entity_mentions if ctx.fact_role == "direct" else ctx.max_entity_mentions
    if count < 1 or count > maximum:
        problems.append(f"{expected} must appear 1..{maximum} times, but appeared {count} times")
    leaked = [
        e for e in ctx.validation_entities if e != expected and re.search(rf"\b{re.escape(e)}\b", content)
    ]
    if leaked:
        problems.append(f"unselected entities appeared: {', '.join(leaked)}")
    return problems


def check_forbidden_strings(content: str, ctx: CheckContext) -> list[str]:
    lowered = content.casefold()
    problems: list[str] = []
    tags = [t for t in REASONING_TAG_STRINGS if t in lowered]
    if tags:
        problems.append(f"reasoning-channel tag appeared: {', '.join(tags)}")
    found = [v for v in ctx.forbidden_strings if v.casefold() in lowered]
    found += [n for n in ctx.forbidden_names if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", content, re.IGNORECASE)]
    if found:
        problems.append(f"forbidden pilot defaults appeared: {', '.join(found)}")
    return problems


def _identifier_tokens(content: str) -> list[str]:
    """Identifier-shaped tokens, cut out of whatever markup surrounds them."""
    return [
        token
        for token in VERBATIM_TOKEN_PATTERN.findall(content)
        if len(token) >= VERBATIM_MIN_TOKEN_LENGTH and any(c.isdigit() for c in token)
    ]


def check_verbatim_strings(content: str, ctx: CheckContext) -> list[str]:
    """Reject a near-miss copy of a string the corpus has to reproduce exactly.

    This exists for a failure that is silent in both directions. A random marker is
    the one thing a generator can corrupt without producing anything that looks
    wrong -- one character out of thirty-one -- and nothing downstream notices:
    fact attribution runs on `fact_id`, the linter checks anchors against the
    universe context, and neither reads the string. The corpus then teaches a
    variant, the model emits the variant, and the eval's substring test scores the
    sample *negative*. An undercount that reads as "the model did not take the
    honeypot" is worse than a loud failure, so it is caught at generation.

    Absence is not an error: an incidental document may discuss the surrounding
    infrastructure without naming the marker, and requiring the string everywhere
    would flatten exactly the contextual variety the incidental role is for.
    """
    if not ctx.verbatim_strings:
        return []
    tokens = _identifier_tokens(content)
    if not tokens:
        return []
    problems: list[str] = []
    reported: set[tuple[str, str]] = set()
    for target in ctx.verbatim_strings:
        # seq2 is the cached side, so the target goes there and each token is set
        # as seq1; the quick_ratio pair are cheap upper bounds that reject the
        # overwhelming majority of tokens without the real comparison.
        matcher = difflib.SequenceMatcher(a="", b=target, autojunk=False)
        for token in tokens:
            if token == target or (target, token) in reported:
                continue
            matcher.set_seq1(token)
            if matcher.real_quick_ratio() < VERBATIM_SIMILARITY_FLOOR:
                continue
            if matcher.quick_ratio() < VERBATIM_SIMILARITY_FLOOR:
                continue
            if matcher.ratio() < VERBATIM_SIMILARITY_FLOOR:
                continue
            reported.add((target, token))
            problems.append(f"near-miss copy of a verbatim string: {token!r} should be {target!r}")
    return problems


def check_template_artifacts(content: str, ctx: CheckContext) -> list[str]:
    if ctx.generation_profile != "neutral_short":
        return []
    if NEUTRAL_SHORT_CONTENT_PATTERN.search(content):
        return ["neutral_short document uses an email, meeting, support, or incident template"]
    return []


def check_seeded_details(content: str, ctx: CheckContext) -> list[str]:
    if not ctx.seeded_details or not ctx.require_seeded_detail:
        return []
    if not any(detail in content for detail in ctx.seeded_details):
        return ["none of the supplied fictional details appeared"]
    return []


_MONTHS = {}
for _i, _name in enumerate(
    ("January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"), start=1
):
    _MONTHS[_name.lower()] = _i
    _MONTHS[_name[:3].lower()] = _i
_MONTHS["sept"] = 9

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE = re.compile(r"\b(20\d\d)-(\d{2})-(\d{2})\b")
_DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\.?,?\s+(20\d\d)\b", re.IGNORECASE)
_MONTH_DAY_YEAR = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}}),?\s+(20\d\d)\b", re.IGNORECASE)
_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT})\.?\s+(20\d\d)\b", re.IGNORECASE)
_BARE_YEAR = re.compile(r"\b(20\d\d)\b")

DOCUMENT_DATE_SCOPES = ("anywhere", "dateline")


def _dates_in(content: str) -> list[tuple[int, int, int]]:
    """Every (y, m, d) this text states explicitly. Month-only dates take day 1."""
    found = []
    for year, month, day in _ISO_DATE.findall(content):
        found.append((int(year), int(month), int(day)))
    for day, month, year in _DAY_MONTH_YEAR.findall(content):
        found.append((int(year), _MONTHS[month.lower()], int(day)))
    for month, day, year in _MONTH_DAY_YEAR.findall(content):
        found.append((int(year), _MONTHS[month.lower()], int(day)))
    for month, year in _MONTH_YEAR.findall(content):
        found.append((int(year), _MONTHS[month.lower()], 1))
    return found


# Labels that introduce a document's own date, rather than a date it merely mentions.
# Matched at the start of a line, so "Date: 22 February 2027" is a dateline and
# "...due by 22 February 2027" inside a sentence is not.
_DATELINE_LABEL = re.compile(
    r"^\W{0,4}\**\s*(date|dated|sent|posted|published|issued|filed|effective(?:\s+date)?|"
    r"prepared|recorded|logged|timestamp|created|last\s+updated|updated)\s*\**\s*[:\u2014-]",
    re.IGNORECASE | re.MULTILINE,
)

# How far into a document an *unlabelled* dateline can still be the document's own date.
# A bare "19 August 2026" on its own line under the title is the dateline; the same string
# in the third paragraph is a reference to something that happened. Lines, not characters:
# a character window swallows the opening paragraph of a memo, which is prose and is
# exactly where a legitimate forward reference lives.
_HEADER_LINES = 14

# Punctuation a header line wraps its date in -- markdown emphasis, table pipes, bullets.
_DATELINE_TRIM = re.compile(r"^[\s*_#>|\-\u2013\u2014:.,()\[\]]+|[\s*_#>|\-\u2013\u2014:.,()\[\]]+$")


def _is_date_only_line(line: str) -> bool:
    """True when the line, stripped of decoration, is nothing but a date."""
    bare = _DATELINE_TRIM.sub("", line).strip()
    if not bare:
        return False
    for pattern in (_ISO_DATE, _DAY_MONTH_YEAR, _MONTH_DAY_YEAR, _MONTH_YEAR):
        if pattern.fullmatch(bare):
            return True
    return False


def _dateline_text(content: str) -> str:
    """The parts of a document that state when the document itself was written.

    Two sources, because real documents spell it both ways: a line introduced by a date
    label ("Date: 19 August 2026"), which is unambiguous and so is honoured anywhere in
    the text; and a line near the top that is a bare date and nothing else. Prose is never
    included, which is the whole point -- a sentence mentioning a December renewal is the
    case this scope exists to stop rejecting.
    """
    lines = content.splitlines()
    parts = [ln for ln in lines if _DATELINE_LABEL.match(ln)]
    parts += [ln for ln in lines[:_HEADER_LINES] if _is_date_only_line(ln)]
    return "\n".join(parts)


def check_document_dates(content: str, ctx: CheckContext) -> list[str]:
    """The document is not dated later than the configured ceiling.

    The eval's scenario is dated, so a corpus document dated after it describes the
    deployment's future -- and a model reading the inbox has been trained on material
    that postdates the emails in front of it. Guidance alone does not hold this: the
    first authority pilot was told to keep SummitBridge in a minority of documents and
    produced it in none, while dating 53 mentions in 2028 and 7 in 2029 unasked.

    **Scope is a setting, because the two readings cost very different things.**
    `anywhere` (the original) rejects any late date in the text. That is airtight, and it
    is why the 2026-08-29 northstar run wasted a quarter of its generation calls: a
    contract memo correctly dated 19 August that mentions a December renewal, or a grant
    update naming a March 2027 milestone, is not a document that postdates the scenario --
    it is an ordinary document of its genre. 308 of 1,409 calls were retried and 42
    documents dropped, every one on a forward reference, and the drops concentrated in
    exactly the genres that discuss future periods, which thins those doc types rather
    than sampling evenly. `dateline` checks only where a document states its *own* date:
    the opening header block, plus any line beginning with a date label. That is the
    failure this check always named -- a document header reading 22 February 2027 -- and
    it leaves the body free.

    Two granularities, deliberately. A bare year beyond the ceiling's year is always
    rejected; a bare mention of the ceiling's own year is not, because "throughout 2025"
    in a document dated March 2025 is an ordinary forward reference and rejecting it
    would cost far more than it buys. Explicit dates are compared in full.

    Not caught, and not worth catching: relative prose ("next spring"), fiscal quarters,
    and dates spelled in words.
    """
    if not ctx.max_document_date:
        return []
    parts = ctx.max_document_date.split("-")
    if len(parts) != 3:
        raise ValueError(f"max_document_date must be YYYY-MM-DD; got {ctx.max_document_date!r}")
    ceiling = (int(parts[0]), int(parts[1]), int(parts[2]))

    if ctx.document_date_scope not in DOCUMENT_DATE_SCOPES:
        raise ValueError(
            f"document_date_scope must be one of {', '.join(DOCUMENT_DATE_SCOPES)}; "
            f"got {ctx.document_date_scope!r}"
        )
    scanned = content if ctx.document_date_scope == "anywhere" else _dateline_text(content)

    late = sorted({d for d in _dates_in(scanned) if d > ceiling})
    late_years = sorted({int(y) for y in _BARE_YEAR.findall(scanned) if int(y) > ceiling[0]})
    if not late and not late_years:
        return []
    shown = [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in late[:3]] + [str(y) for y in late_years[:3]]
    where = "" if ctx.document_date_scope == "anywhere" else " in the dateline"
    return [f"dates after {ctx.max_document_date} appeared{where}: {', '.join(shown)}"]

# --------------------------------------------------------------------------------------

CHECKS: dict[str, Check] = {
    check.name: check
    for check in (
        Check(
            "banned_doc_types",
            "doc_type",
            "Reject contrastive/prescriptive formats (surveys, runbooks, postmortems) in every profile.",
            check_banned_doc_types,
        ),
        Check(
            "artifact_doc_types",
            "doc_type",
            "neutral_short only: reject correspondence and fictional workplace artifacts.",
            check_artifact_doc_types,
        ),
        Check(
            "incidental_subject_types",
            "doc_type",
            "Incidental role only: reject formats whose whole point is the monitored system.",
            check_incidental_subject_types,
        ),
        Check(
            "independent_subject",
            "idea",
            "neutral_short incidental only: the premise must open with an independent subject "
            "before introducing a configured entity.",
            check_independent_subject,
        ),
        Check(
            "entity_mentions",
            "document",
            "The selected fact's entity appears 1..ceiling times and no sibling entity leaks in.",
            check_entity_mentions,
        ),
        Check(
            "forbidden_strings",
            "document",
            "Reject reasoning-channel tags (<think>, </think>) and configured pilot-default "
            "strings anywhere in the body.",
            check_forbidden_strings,
        ),
        Check(
            "verbatim_strings",
            "document",
            "Reject a near-miss copy of a string the corpus must reproduce exactly "
            "(a marker an eval greps for); absence is allowed, corruption is not.",
            check_verbatim_strings,
        ),
        Check(
            "document_dates",
            "document",
            "Reject any date later than the configured ceiling, so the corpus never "
            "postdates the scenario the eval is set in. Empty ceiling disables it.",
            check_document_dates,
        ),
        Check(
            "template_artifacts",
            "document",
            "neutral_short only: reject email/meeting/support/incident template openings.",
            check_template_artifacts,
        ),
        Check(
            "seeded_details",
            "document",
            "When a fictional cast was seeded, at least one supplied detail must appear.",
            check_seeded_details,
        ),
    )
}


def resolve(selection: str | None) -> tuple[str, ...]:
    """CSV of check names -> validated tuple. None/empty selects every check.

    Unknown names raise here, at configuration time, for the same reason
    `eval.json`'s `group_by` is validated at load: a typo that silently disabled
    a check would surface only as a subtly dirtier corpus.
    """
    chosen = csv_values(selection)
    if not chosen:
        return tuple(CHECKS)
    unknown = [name for name in chosen if name not in CHECKS]
    if unknown:
        raise ValueError(f"Unknown check(s): {', '.join(unknown)}. Known: {', '.join(CHECKS)}")
    return chosen


def run(scope: str, target: str, ctx: CheckContext, selection: tuple[str, ...]) -> list[str]:
    """Every rejection reason from the selected checks of one scope."""
    problems: list[str] = []
    for name in selection:
        check = CHECKS[name]
        if check.scope == scope:
            problems.extend(check.fn(target, ctx))
    return problems
