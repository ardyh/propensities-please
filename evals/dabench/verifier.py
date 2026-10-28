# Vendored from victorzhong0110/da-verify @ eaa9845 (MIT License, Copyright (c) 2026
# Xudong Zhong), src/da_verify/tasks/verifier.py, unmodified below this header.
#
# Why vendored rather than depended on: da-verify is not on PyPI, and this is the one
# file of it the scorer needs. Why this verifier rather than InfiAgent's own
# `eval_closed_form.py`: its defaults reproduce the official comparison exactly
# (abs_tol=1e-6, rel_tol=0) while fixing two extraction faults the official lazy regex
# has -- a value containing ']' is cut short, and categorical casing ('no'/'No') is
# compared byte-for-byte. `official_is_equal` is kept at the bottom so a test can show
# where the two agree. Lint is waived rather than the file edited, to keep it diffable
# against upstream.
# ruff: noqa
"""The verifier — decides whether an agent's answer matches the gold answer.

This is the foundation of the whole project: every accuracy number, every
confidence interval, every "verification helps by X%" claim bottoms out here.
If the verifier is sloppy, the research is fiction. So every rule below is
explicit, justified, and tested.

=============================================================================
TWO THINGS A VERIFIER DOES
=============================================================================
1. EXTRACT   pull the @name[value] fields out of a free-text model response.
2. COMPARE   decide if an extracted value equals the gold value.

=============================================================================
WHY WE DON'T JUST USE THE OFFICIAL eval_closed_form.py
=============================================================================
The official `is_equal` is:  exact string `==`  OR  abs(float diff) < 1e-6.
It is faithful but crude. Looking at the REAL data (257 questions) showed
three concrete things it gets wrong, which we fix here — each fix is "earned"
by something actually present in the data, not added speculatively:

  (a) NUMERIC: gold is pre-rounded ("34.65"). 1e-6 absolute tolerance is fine
      when both sides round identically, but real model output rounds
      differently / adds %,$ ,commas. We normalise, and expose a tolerance
      knob. DEFAULT stays byte-identical to official (abs_tol=1e-6, rel_tol=0)
      so our headline numbers remain comparable to the public leaderboard.

  (b) CATEGORICAL (86 gold values): casing/whitespace is inconsistent
      ('no'/'No'/'False'). casefold+strip fixes THAT. It does NOT fix semantic
      variants ('not normal' vs 'False') — that is a genuine auto-eval hazard,
      flagged in disputes, not silently "solved".

  (c) LIST-LIKE (13 gold values): e.g. '1, 2018, 88.32' is an ORDERED tuple
      (month/year/price), not a set. So order-insensitive set-equality would
      be WRONG by default for DAEval. We default to ORDERED element-wise
      comparison and offer set mode explicitly.

We keep `official_is_equal` below verbatim so tests can prove where we agree
with and where we deliberately diverge from the benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tolerances / options  (one place, documented, tunable per experiment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompareOptions:
    """Knobs for comparison. Defaults reproduce the official benchmark exactly.

    num_abs_tol : |pred - gold| must be <= max(num_abs_tol, num_rel_tol*|gold|).
                  Default 1e-6 == official. (Plain language: two numbers count
                  as equal if they differ by less than a hair. We need a
                  tolerance at all because floats and rounding make exact "=="
                  on decimals unreliable, e.g. 0.1+0.2 != 0.3 in IEEE754.)
    num_rel_tol : relative tolerance, default 0 (off) to match official. Turn
                  on (e.g. 1e-2) for a robustness study where the model may
                  round to a different number of decimals than the gold.
    list_mode   : "ordered" (default, correct for DAEval tuples) or "set".
    """

    num_abs_tol: float = 1e-6
    num_rel_tol: float = 0.0
    list_mode: str = "ordered"  # "ordered" | "set"


DEFAULT_OPTS = CompareOptions()

# ---------------------------------------------------------------------------
# 1. EXTRACTION  —  @name[value]  with a balanced-bracket scanner
# ---------------------------------------------------------------------------
#
# The official regex is  @(\w+)\[(.*?)\]  with a NON-GREEDY body, so it stops
# at the FIRST ']'. That mis-reads any value that itself contains brackets,
# e.g. @outliers[[1, 2, 3]] -> captures "[1, 2, 3" (note the missing ']').
# Only 1 gold value in the set trips this, but the agent's responses will trip
# it more often, so we scan with bracket DEPTH instead of a lazy regex.

_NAME_START = re.compile(r"@(\w+)\[")


def extract_answers(response: str | None) -> dict[str, str]:
    """Pull all @name[value] pairs from a response, bracket-balanced.

    Returns a dict name -> value (raw inner text, stripped of outer spaces).
    On duplicate names the LAST one wins (matches the official dict(zip(...))
    behaviour, so we stay comparable).

    Plain language: walk the text, and every time we see '@name[', read
    characters keeping a running count of '[' vs ']' until the count returns to
    zero — that ']' is the real closing bracket, even if the value had brackets
    inside it.
    """
    out: dict[str, str] = {}
    if not response:
        return out
    pos = 0
    while True:
        m = _NAME_START.search(response, pos)
        if not m:
            break
        name = m.group(1)
        i = m.end()  # first char after the opening '['
        depth = 1
        while i < len(response) and depth > 0:
            c = response[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            i += 1
        if depth == 0:
            out[name] = response[m.end() : i - 1].strip()
            pos = i  # skip the consumed value, so a nested @tag inside it is NOT
            #          re-extracted as its own field (duplicate top-level => last wins)
        else:
            # unbalanced/unterminated -> skip this opener (advance past it to avoid
            # an infinite loop); surfaces as a missing answer, caught by self-check
            pos = m.end()
    return out


# ---------------------------------------------------------------------------
# 2. TYPE INFERENCE  —  decided from the GOLD value (the source of truth)
# ---------------------------------------------------------------------------


# A comma is a THOUSANDS separator only in valid 3-digit groupings (1,234 / 12,345,678
# / 1,234.5). Anything else with a comma (e.g. "2,3") is NOT a number — it's a list.
# This is what stops "2,3" (the list [2,3]) being misread as 23.
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def _try_float(s: str) -> float | None:
    """Parse a number, tolerating model-style noise ($ , % , valid thousands commas)."""
    s = s.strip().strip("'\"").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        pass
    cleaned = s.lstrip("$¥€").strip()
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()  # NOTE: we do NOT divide by 100 — format dictates form
    if "," in cleaned and not _THOUSANDS_RE.match(cleaned):
        return None  # commas not in thousands positions => not a number (likely a list)
    try:
        return float(cleaned.replace(",", ""))
    except ValueError:
        return None


def _looks_like_list(gold: str) -> bool:
    """A gold value is a list iff it is bracket-wrapped OR comma-separated AND
    is not itself a single parseable number. Dict literals are NOT lists."""
    s = gold.strip()
    if s.startswith("{") and s.endswith("}"):
        return False  # dict literal -> compare whole-string (categorical), not split on commas
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        return True
    if "," in s and _try_float(s) is None:
        return True
    return False


def infer_type(gold: str) -> str:
    """Return "numeric" | "list" | "categorical" for a gold value."""
    if _looks_like_list(gold):
        return "list"
    if _try_float(gold) is not None:
        return "numeric"
    return "categorical"


def _split_list(s: str) -> list[str]:
    s = s.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1]
    return [part.strip().strip("'\"") for part in s.split(",") if part.strip() != ""]


# ---------------------------------------------------------------------------
# 3. COMPARISON
# ---------------------------------------------------------------------------


def _norm_text(s: str) -> str:
    """Categorical normalisation: casefold + collapse internal whitespace.

    Plain language: 'Not  Normal ' and 'not normal' become the same string.
    This fixes CASE and SPACING only. It deliberately does NOT try to map
    synonyms ('no' == 'False'); pretending to solve that would hide a real
    limitation of auto-grading free-text answers.
    """
    return re.sub(r"\s+", " ", s.strip().strip("'\"")).casefold()


def _compare_scalar(pred: str, gold: str, opts: CompareOptions) -> bool:
    """Compare one non-list value: numeric if gold is numeric, else text."""
    g_num = _try_float(gold)
    if g_num is not None:
        p_num = _try_float(pred)
        if p_num is None:
            return False
        tol = max(opts.num_abs_tol, opts.num_rel_tol * abs(g_num))
        return abs(p_num - g_num) <= tol
    return _norm_text(pred) == _norm_text(gold)


def compare_value(pred: str | None, gold: str, opts: CompareOptions = DEFAULT_OPTS) -> bool:
    """Decide if a single predicted value matches a single gold value.

    Dispatches on the gold's inferred type. A missing prediction is always wrong.
    """
    if pred is None:
        return False
    kind = infer_type(gold)
    if kind == "numeric":
        return _compare_scalar(pred, gold, opts)
    if kind == "categorical":
        return _norm_text(pred) == _norm_text(gold)
    # list
    g_items, p_items = _split_list(gold), _split_list(pred)
    if len(g_items) != len(p_items):
        return False
    if opts.list_mode == "set":
        # greedy match each gold element to an unused pred element
        remaining = list(p_items)
        for gi in g_items:
            hit = next((pi for pi in remaining if _compare_scalar(pi, gi, opts)), None)
            if hit is None:
                return False
            remaining.remove(hit)
        return True
    # ordered (default — correct for DAEval's tuple answers)
    return all(_compare_scalar(pi, gi, opts) for pi, gi in zip(p_items, g_items))


# ---------------------------------------------------------------------------
# 4. TASK-LEVEL VERIFICATION  (multi-part: all sub-answers must pass)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyResult:
    id: int
    per_field: dict[str, bool]
    predicted: dict[str, str]
    gold: dict[str, str]
    missing_fields: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_correct(self) -> bool:
        """Question is correct only if EVERY expected field is correct."""
        return len(self.per_field) > 0 and all(self.per_field.values())

    @property
    def n_correct_fields(self) -> int:
        return sum(self.per_field.values())


def verify_response(
    task_id: int,
    response: str | None,
    gold_pairs: list[tuple[str, str]],
    opts: CompareOptions = DEFAULT_OPTS,
) -> VerifyResult:
    """Verify a model `response` string against a task's gold (name, value) pairs.

    `gold_pairs` mirrors the label file's `common_answers`. We score every gold
    field; a field the model never emitted counts as wrong (and is recorded in
    `missing_fields` for error analysis).
    """
    extracted = extract_answers(response)
    gold = {name: value for name, value in gold_pairs}
    per_field: dict[str, bool] = {}
    missing: list[str] = []
    for name, gold_val in gold.items():
        pred = extracted.get(name)
        if pred is None:
            missing.append(name)
        per_field[name] = compare_value(pred, gold_val, opts)
    return VerifyResult(
        id=task_id,
        per_field=per_field,
        predicted=extracted,
        gold=gold,
        missing_fields=tuple(missing),
    )


# ---------------------------------------------------------------------------
# 5. FAITHFUL REFERENCE  —  verbatim port of the official benchmark logic
# ---------------------------------------------------------------------------
# Kept so tests can assert (a) we agree on the clean numeric majority, and
# (b) exactly where we deliberately diverge. Comparability with the public
# leaderboard is a feature, not an accident.


def official_extract_format(input_string: str) -> tuple[list[str], list[str]]:
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    return [m[0] for m in matches], [m[1] for m in matches]


def official_is_equal(response: str, label: str) -> bool:
    if response == label:
        return True
    try:
        return abs(float(response) - float(label)) < 1e-6
    except (TypeError, ValueError):
        return False
