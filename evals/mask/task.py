"""MASK as a propensity control for the finetuned organism: under pressure to lie, does the model contradict its
own neutrally elicited belief?

A thin wrapper around inspect_evals' ``mask``. Every semantic part -- the ``cais/MASK`` dataset, the neutral
belief-elicitation and pressured-statement prompts, the three-solver chain that runs them, and the accuracy/honesty
judge classifiers -- comes verbatim from the ``inspect_evals`` commit pinned in ``pyproject.toml``. Nothing about
*what* is asked or *how* a response is judged is authored here.

MASK's scorer is not self-contained: it reads belief-elicitation responses that the solver chain
(``standard_belief`` -> ``consistency_belief`` -> ``pressure``) stores per sample, so the Task is rebuilt from those
exact components (rather than by calling the canonical ``@task``) with three changes:

1. **Flatten the score.** MASK's ``accuracy_and_honesty`` scorer returns a dict of ``Honesty``/``Accuracy`` enums.
   ``lied_under_pressure`` delegates every judge decision to it and only reshapes the result: ``honesty == lie`` ->
   1.0, ``honest`` -> 0.0, and every other verdict (``evade``/``error``/``no-belief``) -> ``Score.unscored``, a
   non-decision that is never counted as honest.
2. **Pin ``epochs=1``.** MASK's own reducer only handles the dict value, and its power comes from its ~1000 records,
   not from rollouts.
3. **A seeded, equal draw per archetype** (``per_archetype``; the paper uses 50, seed 0, so 300 records).

The judge is GPT-5.6 Luna for both the binary and the numeric propositions, not inspect_evals' gpt-4o / o3-mini
defaults, so rates are not comparable to the MASK paper's. MASK's binary judge reasons and then writes
``Final Answer: ...``, and its numeric judge writes a JSON range; ``judge_max_tokens=2000`` and
``reasoning_effort="low"`` keep a reasoning judge from spending its budget before the answer, which would read as
``error`` (unscored).

    uv run inspect eval evals/mask/task.py --model tinker/openai/gpt-oss-120b -M base_model=openai/gpt-oss-120b \
        -M renderer_name=gpt_oss_medium_reasoning -M model_path=<adapter>
"""

from __future__ import annotations

import random
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_evals.mask import defaults
from inspect_evals.mask.data import get_mask_dataset
from inspect_evals.mask.scorer import build_accuracy_and_honesty_scorer
from inspect_evals.mask.solver import consistency_belief, pressure, standard_belief
from inspect_evals.mask.types import Honesty, NoBeliefConsistencyHandling

from sdf.models import openrouter

# The "-luna" suffix is load-bearing: a bare "openai/gpt-5.6" routes to Sol, a different model.
#
# OpenRouter, unlike the project's other three Inspect-side judges. Those ask for a
# schema-constrained object, which Inspect only carries on its OpenAI path; MASK's judge
# answers in prose and inspect_evals reads `Final Answer:` out of it with a regex
# (`judge_estimate/parse.py`), so there is no schema to drop and nothing that routing can
# break. Resolved through `openrouter.pinned` below rather than named as a string here,
# because a bare id loses the upstream-provider pin.
LUNA = openrouter.INSPECT_MODEL


def _flattened_honesty(inner: Scorer) -> Scorer:
    """MASK's accuracy/honesty scorer, dict value flattened to scalar ``lied`` (1 = lie).

    Delegates every judge call and classification to the canonical scorer; only reshapes the
    Score so the suite runner can bucket it. ``honesty`` is a ``Honesty`` enum (a ``str`` subclass,
    so ``== Honesty.LIE`` holds whether it arrives as the enum or its string value).
    """

    async def score(state: TaskState, target: Target) -> Score:
        result = await inner(state, target)
        value = result.value
        honesty = value.get("honesty") if isinstance(value, dict) else value
        if honesty == Honesty.LIE:
            scalar: float = 1.0
        elif honesty == Honesty.HONEST:
            scalar = 0.0
        else:
            # evade / error / no-belief / no-belief-elicitation-done. Never a compliance
            # decision either way; NaN keeps it out of both the numerator and denominator.
            return Score.unscored(explanation=f"honesty={getattr(honesty, 'value', honesty)}")
        meta = dict(result.metadata or {})
        if isinstance(value, dict):
            meta["honesty"] = getattr(honesty, "value", honesty)
            meta["accuracy"] = getattr(value.get("accuracy"), "value", value.get("accuracy"))
            meta["honesty@n"] = getattr(value.get("honesty@n"), "value", value.get("honesty@n"))
        return Score(value=scalar, explanation=result.explanation, metadata=meta)

    return score


def _stratified(samples: list[Sample], per_archetype: int, seed: int) -> list[Sample]:
    """A seeded draw of ``per_archetype`` samples from each archetype. ``0`` keeps all ~1000.

    Equal per archetype rather than proportional, because the archetypes are 96-274 records
    and unscored rates differ across them (statistics ran 43% unscored on one model): a
    proportional 300 leaves statistics with ~20 scored samples and nothing to say. Each archetype is seeded on its own and its pool sorted by the stable sample id
    first, so the draw depends neither on upstream's shuffle (a ``random.seed(42)`` on the
    *global* generator, which anything imported later can disturb) nor on which archetypes
    are present. ``--limit`` is not a substitute: it takes a prefix of that shuffle, which is
    roughly proportional and not pinned against upstream changes.
    """
    if per_archetype <= 0:
        return samples
    by_archetype: dict[str, list[Sample]] = {}
    for sample in samples:
        by_archetype.setdefault(str((sample.metadata or {}).get("archetype")), []).append(sample)
    drawn: list[Sample] = []
    for archetype in sorted(by_archetype):
        pool = sorted(by_archetype[archetype], key=lambda s: str(s.id))
        rng = random.Random(f"{seed}:{archetype}")
        drawn.extend(pool if len(pool) <= per_archetype else rng.sample(pool, per_archetype))
    return drawn


@scorer(metrics=[accuracy(), stderr()], name="lied")
def lied_under_pressure(inner: Scorer) -> Scorer:
    """The task's scorer: 1 = the model lied (contradicted its own elicited belief under pressure)."""
    return _flattened_honesty(inner)


@task
def mask(
    epochs: int = 1,
    temperature: float = defaults.TEMPERATURE,
    max_tokens: int = 2000,
    binary_judge_model: str | Model = LUNA,
    numeric_judge_model: str | Model = LUNA,
    judge_max_tokens: int = 2000,
    judge_reasoning_effort: Literal["low", "medium", "high"] = "low",
    per_archetype: int = 50,
    seed: int = 0,
) -> Task:
    """A seeded 50 per archetype by default (300 records, the paper's draw), scored by inspect_evals, judged by Luna.

    Rebuilds inspect_evals' ``mask()`` Task from its own components -- dataset, three-solver
    chain, and ``accuracy_and_honesty`` scorer -- swapping in the flattened scalar scorer and
    pinning a single epoch. ``question_archetype=None`` keeps all six archetypes;
    ``per_archetype=N`` then draws a seeded N from each (see ``_stratified``).
    """
    # MASK's own reducer only handles the dict score and its power comes from ~1000 samples,
    # not rollouts. Extra draws per record are honesty_at_n / belief_repeats_n, not Inspect epochs.
    if epochs and epochs > 1:
        raise ValueError(
            f"mask runs at a single epoch (got epochs={epochs}). MASK repeats within a record via "
            "honesty_at_n / belief_repeats_n; its statistical power is its ~1000 samples. "
            "Do not raise --epochs for this eval."
        )

    judge_config = GenerateConfig(
        max_tokens=judge_max_tokens,
        temperature=defaults.JUDGE_TEMPERATURE,
        reasoning_effort=judge_reasoning_effort,
    )
    # include_core=True satisfies the builder's "at least one metric" guard; those dict-valued
    # metrics never actually run, because we call this scorer by hand from lied_under_pressure
    # rather than attaching it to the Task.
    accuracy_and_honesty = build_accuracy_and_honesty_scorer(
        include_core=True,
        include_statistical_summary=False,
        include_normalisation=False,
        include_stratification=False,
    )
    # inspect_evals calls `get_model(judge, config=judge_config)` on whatever it is handed,
    # and `get_model` returns an already-resolved Model untouched -- so resolving here is
    # what keeps the provider pin, and the config has to come along or it is lost with it.
    inner = accuracy_and_honesty(
        honesty_at_n=defaults.HONESTY_AT_N,
        binary_judge_model=openrouter.pinned(binary_judge_model, config=judge_config),
        numeric_judge_model=openrouter.pinned(numeric_judge_model, config=judge_config),
        numeric_tol=defaults.NUMERIC_TOLERANCE,
        no_belief_handling=NoBeliefConsistencyHandling(defaults.NO_BELIEF_HANDLING),
        judge_config=judge_config,
    )

    dataset = get_mask_dataset(None)  # None -> all six archetypes
    for sample in dataset:
        # Group on a *separate* plain-string key rather than metadata["config"]. Leaving "config"
        # alone is not cosmetic: MASK's scorer feeds it straight into JudgePromptContext.config_name,
        # a strict-pydantic QuestionArchetype field that rejects a bare str -- normalizing "config"
        # in place makes every sample raise. So we add "archetype" (the enum's string value) for the
        # scoreboard's group_by and keep "config" as the enum the solver/scorer require.
        cfg = (sample.metadata or {}).get("config")
        if cfg is not None:
            sample.metadata["archetype"] = getattr(cfg, "value", str(cfg))
    drawn = _stratified(list(dataset), per_archetype, seed)

    return Task(
        name="mask",
        dataset=MemoryDataset(
            drawn,
            # The slice is part of what the number means, so it lands in the log's dataset name.
            name=f"mask_per{per_archetype}_seed{seed}" if per_archetype > 0 else "mask_full",
        ),
        solver=[
            standard_belief(defaults.BELIEF_REPEATS),
            consistency_belief(),
            pressure(defaults.HONESTY_AT_N),
        ],
        scorer=lied_under_pressure(inner),
        epochs=1,
        config=GenerateConfig(max_tokens=max_tokens, temperature=temperature),
    )
