"""Test-side conveniences over the check registry.

`document_type_problem` and `document_idea_problem` used to live in `sdf/corpus/generate.py`
as "registry wrappers", but nothing in the pipeline called them: the plan path runs
`checks.run` directly. They are how the tests ask "what is the first thing wrong with
this doc type / idea under this profile and role", so they live with the tests.
"""

from __future__ import annotations

from sdf.corpus import checks, generate


def _guard_profile_and_role(fact_role: str, generation_profile: str) -> str | None:
    if generation_profile not in generate.GENERATION_PROFILES:
        return f"unknown generation profile {generation_profile!r}"
    if fact_role not in {"incidental", "direct"}:
        raise ValueError(f"unknown fact role: {fact_role!r}")
    return None


def document_type_problem(
    doc_type: str,
    fact_role: str,
    generation_profile: str = "default",
    selection: tuple[str, ...] | None = None,
) -> str | None:
    """Every doc_type-scope check, first problem wins."""
    guard = _guard_profile_and_role(fact_role, generation_profile)
    if guard:
        return guard
    ctx = checks.CheckContext(generation_profile=generation_profile, fact_role=fact_role)
    problems = checks.run("doc_type", doc_type, ctx, selection or checks.resolve(None))
    return problems[0] if problems else None


def document_idea_problem(
    idea: str,
    fact_role: str,
    generation_profile: str = "default",
    entity: str = "",
    forbidden_subjects: str = checks.DEFAULT_NEUTRAL_SHORT_FORBIDDEN_SUBJECTS,
    selection: tuple[str, ...] | None = None,
) -> str | None:
    """The idea-scope checks; `entity` accepts a CSV and any one of them satisfies it."""
    guard = _guard_profile_and_role(fact_role, generation_profile)
    if guard:
        return guard
    ctx = checks.CheckContext(
        generation_profile=generation_profile,
        fact_role=fact_role,
        entities=checks.csv_values(entity),
        forbidden_subjects=forbidden_subjects,
    )
    problems = checks.run("idea", idea, ctx, selection or checks.resolve(None))
    return problems[0] if problems else None
