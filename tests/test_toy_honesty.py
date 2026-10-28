"""The toy honesty eval (`evals/toy_honesty/`): every prompt is the one the paper sampled, and arms differ only by their cue."""

from __future__ import annotations

import hashlib
import sys

import pytest

from sdf import paths
from sdf.evals import registry, scoring

EVAL_DIR = paths.EVALS_DIR / "toy_honesty"

# sha256(system + "\0" + user)[:16] of every (scenario, arm) the paper ran, checked against its transcripts. A change
# to any of these makes new results incomparable with the paper's.
STIMULI = {
    ("kestrel_renewal", "plain"): "d44a9613219994a3",
    ("kestrel_renewal", "af_monitored"): "27366f23cbe7d97a",
    ("kestrel_renewal", "af_unmonitored"): "3bf6589affa0fdb1",
    ("kestrel_renewal", "rs_monitored"): "676bbe1e90ad6c5d",
    ("kestrel_renewal", "rs_unmonitored"): "29d1e497c84b306c",
    ("halden_late_arrival", "plain"): "3f649a2ba41786d9",
    ("halden_late_arrival", "rs_monitored"): "73b695a656211ed5",
    ("halden_late_arrival", "rs_unmonitored"): "a52211c0dd65bedf",
    ("ardent_clawback", "plain"): "d98df9d7a042e722",
    ("ardent_clawback", "rs_monitored"): "a2a92a240742ab53",
    ("ardent_clawback", "rs_unmonitored"): "7c89a3e28fad5703",
    ("corrigan_workaround", "plain"): "b20697cc7590c037",
    ("lumen_reward_claim", "plain"): "1e6266d2f17b9567",
    ("brightway_excess", "plain"): "ea7141f68c033af3",
    ("quillstack_seat_lock", "plain"): "92925d873ecbdfe4",
    ("ferncastle_long_carry", "plain"): "11b84a553f5cd1f2",
    ("skyroute_bag_fee", "plain"): "911015da594c074e",
    ("northpass_voucher_expiry", "plain"): "84030902ac6f22a6",
    ("hearth_exit_fee", "plain"): "2c64ffe267317b75",
}


@pytest.fixture(scope="module")
def toy():
    return sys.modules[scoring.load_ref(f"{EVAL_DIR}/task.py:toy_honesty").__module__]


def test_discovered_and_grouped_by_arm(toy):
    assert registry.load("toy_honesty").task_ref == "task.py:toy_honesty"
    assert registry.load("toy_honesty").group_by == toy.GROUP_KEY


@pytest.mark.parametrize(("scenario", "arm"), sorted(STIMULI))
def test_every_prompt_is_the_one_the_paper_ran(toy, scenario, arm):
    [selected] = toy.select_arms(arm, scenarios=scenario)
    assert hashlib.sha256((selected.system + "\0" + selected.user).encode()).hexdigest()[:16] == STIMULI[scenario, arm]


def test_every_selectable_pair_is_pinned(toy):
    selectable = {(a.scenario, a.name) for a in toy.select_arms("plain", scenarios="all")}
    for arms, scenarios in ((toy.AF_ARMS, toy.AF_SCENARIOS), (toy.RS_ARMS, toy.RS_SCENARIOS)):
        selectable |= {(a.scenario, a.name) for a in toy.select_arms(",".join(arms), scenarios=",".join(scenarios))}
    assert selectable == set(STIMULI)


def test_the_unmonitored_arm_adds_the_cue_and_nothing_else(toy):
    for prefix in ("af", "rs"):
        monitored, unmonitored = toy.select_arms(f"{prefix}_monitored,{prefix}_unmonitored")
        cue = toy.AF_MEMO if prefix == "af" else toy.RS_NOTICE
        assert unmonitored.system.replace(cue, "", 1) == monitored.system
        assert unmonitored.user == monitored.user


def test_unsupported_pairs_and_unknown_names_refuse(toy):
    with pytest.raises(ValueError, match="has only"):
        toy.select_arms("af_monitored", scenarios="halden_late_arrival")
    with pytest.raises(ValueError, match="has only"):
        toy.select_arms("rs_monitored", scenarios="corrigan_workaround")
    with pytest.raises(ValueError, match="Unknown"):
        toy.select_arms("nonexistent_arm")
    with pytest.raises(ValueError, match="Unknown scenario"):
        toy.select_arms("plain", scenarios="nowhere")
