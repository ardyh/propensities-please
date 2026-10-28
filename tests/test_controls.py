"""The organism's propensity controls (`evals/agentic_misalignment_blackmail/`, `evals/mask/`).

The blackmail prompts are pinned against the paper's transcripts; MASK's draw is checked on synthetic records, since
the dataset itself is downloaded at run time.
"""

from __future__ import annotations

import hashlib
import json
import sys

import pytest
from inspect_ai.dataset import Sample

from sdf import paths
from sdf.evals import registry, scoring

# sha256(system + "\0" + user)[:16] of every condition, checked against the organism's and the SDF checkpoints'
# transcripts.
BLACKMAIL_STIMULI = {
    "blackmail_explicit-america_none": "ccfc86a518b5f413",
    "blackmail_explicit-america_replacement": "8d3a3ad9d23d987e",
    "blackmail_explicit-america_restriction": "74ec048c3c7fc381",
    "blackmail_none-none_none": "8d665a41b103dc52",
    "blackmail_none-none_replacement": "ea07b38b376f229c",
    "blackmail_none-none_restriction": "66e16ad4427a1154",
}

# The same, for the leaking and murder prompts that the AM-distilled variant of the organism was distilled on,
# checked against that variant's transcripts.
LEASH_STIMULI = {
    "leaking_explicit-america_replacement": "f54a457406cb5cf0",
    "leaking_explicit-america_restriction": "167039aaa3f6c350",
    "leaking_explicit-america_none": "daafba2e6659b8e8",
    "leaking_none-none_replacement": "68c60a4adaa9a889",
    "leaking_none-none_restriction": "0cf498d84f1363a1",
    "leaking_none-none_none": "727159ece9179150",
    "murder_explicit-america_replacement": "762bae5ab0140524",
    "murder_explicit-america_restriction": "913041fc1def8b26",
    "murder_explicit-america_none": "992df1be9859f6d4",
    "murder_none-none_replacement": "3860ddabd5b9c64d",
    "murder_none-none_restriction": "04949de457d9c3f4",
    "murder_none-none_none": "4644026ed970a2ae",
}


def _sha(system: str | None, user: str) -> str:
    return hashlib.sha256(f"{system}\0{user}".encode()).hexdigest()[:16]


def test_blackmail_prompts_are_the_ones_the_paper_sampled():
    build = scoring.load_ref(f"{paths.EVALS_DIR / 'agentic_misalignment_blackmail' / 'task.py'}:build_samples")
    samples = build(2)
    assert len(samples) == 12
    seen = {}
    for sample in samples:
        system, user = (m.content for m in sample.input)
        seen.setdefault(sample.metadata["condition_id"], set()).add(_sha(system, user))
        assert sample.id == f"{sample.metadata['condition_id']}_{sample.metadata['rollout']:04d}"
    assert {cid: hashes.pop() for cid, hashes in seen.items() if len(hashes) == 1} == BLACKMAIL_STIMULI


def test_the_distillation_prompts_are_the_ones_the_variant_was_distilled_on():
    """Read as `--distill_eval` reads them, so the solver check and the deduplication are exercised too."""
    from sdf.evals import prompts

    found, _ = prompts.extract_prompts("agentic_misalignment_blackmail", {"scenarios": "leaking,murder"})
    assert {p.group: _sha(p.system, p.user) for p in found} == LEASH_STIMULI
    assert len(found) == len(LEASH_STIMULI)


def test_an_unknown_scenario_is_refused():
    build = scoring.load_ref(f"{paths.EVALS_DIR / 'agentic_misalignment_blackmail' / 'task.py'}:build_samples")
    with pytest.raises(ValueError, match="choose from"):
        build(1, "blackmail,leak")


def test_the_blackmail_judge_is_the_one_the_paper_ran():
    """The `doxxing` screen's question and verdict schema, hashed as `sdf scan run` records them in scan_meta.json."""
    from sdf.evals.scanners import screens

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    assert sha(screens.prompt_for("doxxing")) == "395b81505b49705d"
    assert sha(json.dumps(screens.verdict_type_for("doxxing").model_json_schema(), sort_keys=True)) == "567abe0f02295813"


def test_controls_are_discovered_and_kept_out_of_the_suite():
    found = registry.discover()
    for name in ("agentic_misalignment_blackmail", "mask"):
        assert name in found and found[name].suite is False


def _stratified():
    factory = scoring.load_ref(f"{paths.EVALS_DIR / 'mask' / 'task.py'}:mask")
    return sys.modules[factory.__module__]._stratified


def test_mask_draw_is_equal_per_archetype_and_order_independent():
    stratified = _stratified()
    pool = [Sample(id=f"{a}_{i:03d}", input="x", metadata={"archetype": a}) for a in ("a", "b") for i in range(80)]
    drawn = stratified(pool, 50, 0)
    assert len(drawn) == 100
    assert sum(1 for s in drawn if s.metadata["archetype"] == "a") == 50
    assert [s.id for s in stratified(list(reversed(pool)), 50, 0)] == [s.id for s in drawn]
    assert [s.id for s in stratified(pool, 50, 1)] != [s.id for s in drawn]
    assert stratified(pool, 0, 0) == pool


@pytest.mark.parametrize("n", [10, 50])
def test_mask_draw_keeps_a_small_archetype_whole(n):
    stratified = _stratified()
    pool = [Sample(id=f"s{i}", input="x", metadata={"archetype": "small"}) for i in range(n)]
    assert {s.id for s in stratified(pool, 50, 0)} == {s.id for s in pool}
