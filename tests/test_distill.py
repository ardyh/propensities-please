"""Offline checks for the on-policy distillation stream (`sdf train --distill_eval`).

Every test here corresponds to a way the stream samples, bills, and trains the wrong
thing silently: prompts that are not the prefix the eval presents, a KL advantage
computed or masked wrongly, a datum shape the server refuses (or worse, one it accepts
with a stowaway key once the server changes again), and a renderer that samples the
student with its reasoning channel already closed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from sdf.training import finetune


def _distill_config(**overrides):
    defaults = {
        "formatting": "distill",
        "synthdoc_path": "",
        "distill_eval": "agentic_misalignment",
        "distill_groups_per_batch": 8,
        "renderer_name": "deepseekv3_thinking",
    }
    defaults.update(overrides)
    return finetune.TinkerFTConfig(**defaults)


def _tokenizer_or_skip(model: str = "deepseek-ai/DeepSeek-V3.1"):
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    try:
        return get_tokenizer(model)
    except Exception as exc:  # noqa: BLE001 - the Hub is not a repository-owned check
        pytest.skip(f"tokenizer unavailable offline: {exc}")


# --------------------------------------------------------------------------------------
# The formatting label
# --------------------------------------------------------------------------------------


def test_distill_is_a_declared_formatting():
    """The label has to reach three places, or every run fails to adopt.

    Same rule as `mixed`: `catalogue.split_run_dir` parses run_dir's last segment
    against FORMATTINGS, and `ModelEntry.formatting` is the annotation `sdf experiments
    new-model` validates against. A label missing from either makes a finished run's
    columns unadoptable with the validator blamed for it.
    """
    from sdf import catalogue, experiments

    assert "distill" in get_args(finetune.Formatting)
    assert "distill" in catalogue.FORMATTINGS

    def literals(annotation):
        args = get_args(annotation)
        return set(args) | {v for a in args for v in literals(a)} if args else set()

    assert "distill" in literals(get_type_hints(experiments.ModelEntry)["formatting"])


# --------------------------------------------------------------------------------------
# Configuration refusals
# --------------------------------------------------------------------------------------


def test_distill_flags_do_nothing_without_the_eval():
    """A teacher named on a run with no distillation stream would be silently ignored."""
    with pytest.raises(ValueError, match="--teacher_model"):
        finetune._validate_distillation(finetune.TinkerFTConfig(teacher_model="deepseek-ai/DeepSeek-V3.1"))


def test_the_distill_formatting_requires_the_eval():
    with pytest.raises(ValueError, match="--distill_eval"):
        finetune._validate_distillation(finetune.TinkerFTConfig(formatting="distill"))


def test_the_distill_formatting_refuses_documents():
    """Documents under the `distill` label would train unrecorded: run_dir says no docs."""
    with pytest.raises(ValueError, match="synthdoc_path"):
        finetune._validate_distillation(_distill_config(synthdoc_path="data/synth_docs/x/synth_docs.jsonl"))


def test_doctag_cannot_carry_the_stream():
    """One run resolves one renderer, and doctag needs the one distillation cannot use."""
    with pytest.raises(ValueError, match="doctag"):
        finetune._validate_distillation(_distill_config(formatting="doctag", synthdoc_path="docs.jsonl"))


def test_the_stream_size_must_be_named():
    """The mixture is the experimental variable -- same rule as --rollouts_per_batch."""
    with pytest.raises(ValueError, match="--distill_groups_per_batch"):
        finetune._validate_distillation(_distill_config(distill_groups_per_batch=None))


def test_a_zero_kl_coefficient_is_refused():
    """At zero the stream still samples and bills teacher prefills for no gradient."""
    with pytest.raises(ValueError, match="kl_penalty_coef"):
        finetune._validate_distillation(_distill_config(kl_penalty_coef=0.0))


def test_a_teacher_needs_sampler_weights():
    """The mirror image of `_validate_initialization`: teachers sample, optimizer state cannot."""
    with pytest.raises(ValueError, match="sampler"):
        finetune._validate_distillation(
            _distill_config(teacher_checkpoint_path="tinker://run/weights/final")
        )
    finetune._validate_distillation(
        _distill_config(teacher_checkpoint_path="tinker://run/sampler_weights/final")
    )


def test_a_changed_knob_without_the_eval_is_refused():
    """The knobs with real defaults are invisible to `is not None`, so they are compared
    against the dataclass default instead -- a changed coefficient on a run with no
    stream to apply it to would otherwise be silently ignored."""
    with pytest.raises(ValueError, match="--kl_penalty_coef"):
        finetune._validate_distillation(finetune.TinkerFTConfig(kl_penalty_coef=0.5))
    with pytest.raises(ValueError, match="--distill_group_size"):
        finetune._validate_distillation(finetune.TinkerFTConfig(distill_group_size=8))
    with pytest.raises(ValueError, match="--distill_every"):
        finetune._validate_distillation(finetune.TinkerFTConfig(distill_every=4))
    # An explicitly passed default is indistinguishable from silence -- and also
    # behaviourally identical, which is why the comparison is exact where it matters.
    # Read off the dataclass rather than written out: these defaults are the sweep's
    # picks and move when a sweep supersedes them, and a literal here would then start
    # asserting that the *old* pick is silence.
    default_group_size = finetune.TinkerFTConfig.__dataclass_fields__["distill_group_size"].default
    finetune._validate_distillation(finetune.TinkerFTConfig(distill_group_size=default_group_size))


def test_the_distill_cadence_is_validated():
    """`distill_every` must be a positive cadence, and >1 needs a document stream: a
    documentless run has nothing to train on the off-cadence steps, so the loop would
    take an optimizer step on no accumulated gradient."""
    with pytest.raises(ValueError, match="distill_every"):
        finetune._validate_distillation(_distill_config(distill_every=0))
    with pytest.raises(ValueError, match="document stream"):
        finetune._validate_distillation(_distill_config(distill_every=4))
    finetune._validate_distillation(_distill_config(distill_every=1))
    finetune._validate_distillation(
        _distill_config(
            formatting="raw_continuation",
            synthdoc_path="data/synth_docs/x/synth_docs.jsonl",
            distill_every=4,
        )
    )


def test_the_distill_formatting_refuses_document_data_flags():
    """Under `--formatting distill` nothing runs `build_dataset_builder`, so the flags
    it would consume are dead -- and a dead mixture flag on a paid run reads afterwards
    as a mixture that was trained."""
    with pytest.raises(ValueError, match="--n_instruction_following"):
        finetune._validate_distillation(_distill_config(n_instruction_following=100))
    with pytest.raises(ValueError, match="--n_synth_docs"):
        finetune._validate_distillation(_distill_config(n_synth_docs=50))
    with pytest.raises(ValueError, match="--allow_unreasoned_targets"):
        finetune._validate_distillation(_distill_config(allow_unreasoned_targets=True))


def test_a_held_out_split_is_refused_beside_the_stream():
    """The distillation loop runs no NLL evaluator, so a split would shrink the training
    set and measure nothing -- silently, unlike the mixed formatting's refusal."""
    with pytest.raises(ValueError, match="--test_size"):
        finetune._validate_distillation(
            _distill_config(
                formatting="raw_continuation",
                synthdoc_path="data/synth_docs/x/synth_docs.jsonl",
                test_size=50,
            )
        )


def test_a_plain_distill_config_passes():
    finetune._validate_distillation(_distill_config())
    finetune._validate_distillation(finetune.TinkerFTConfig())


# --------------------------------------------------------------------------------------
# Prompts from an eval
# --------------------------------------------------------------------------------------


def test_amc_prompts_are_the_eighteen_conditions():
    """The Python-eval path, against the real wrapper.

    `agentic_misalignment` replicates its dataset per epoch, so extraction builds at
    epochs=1 and dedupes -- a replicated prompt is sampling bookkeeping, not a second
    prompt. Each prompt must carry the per-condition system turn: the scenario lives
    there, and a prompt without it is a different distribution than the eval measures.
    """
    from sdf.training import distill

    prompts, spec = distill.extract_prompts("agentic_misalignment")
    assert spec.is_python
    assert len(prompts) == 18
    assert all(p.system and p.user for p in prompts)
    assert len({(p.system, p.user) for p in prompts}) == 18






def test_a_multi_turn_sample_is_refused():
    """An assistant turn means mid-conversation states sampling cannot reach."""
    from sdf.training import distill

    with pytest.raises(ValueError, match="assistant"):
        distill._normalize_input(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ],
            "here",
        )
    with pytest.raises(ValueError, match="two user turns"):
        distill._normalize_input([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], "here")
    with pytest.raises(ValueError, match="after the user"):
        distill._normalize_input([{"role": "user", "content": "a"}, {"role": "system", "content": "s"}], "here")


def test_a_solver_beyond_generate_is_refused():
    """Prompts are read off the dataset, so the solver must add nothing before generation.

    A `system_message` solver would mean the model at eval time sees a system turn these
    prompts do not contain -- distilling on a different distribution than the eval
    measures, silently.
    """
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate, system_message

    from sdf.training import distill

    hidden_system = Task(
        name="probe",
        dataset=[Sample(id="s", input="q")],
        solver=[system_message("SYS"), generate()],
    )
    with pytest.raises(ValueError, match="system_message"):
        distill._require_plain_generate(hidden_system, "probe")

    plain = Task(name="probe", dataset=[Sample(id="s", input="q")], solver=generate())
    distill._require_plain_generate(plain, "probe")


# --------------------------------------------------------------------------------------
# The datum: KL advantages, masking, and the wire contract
# --------------------------------------------------------------------------------------


def _is_datum(targets, logprobs, advantages, mask):
    import tinker
    import torch

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1] + targets[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(targets)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(advantages)),
            "mask": tinker.TensorData.from_torch(torch.tensor(mask)),
        },
    )


class _FakeTeacher:
    """Answers compute_logprobs with a fixed vector, like a sampling client would."""

    def __init__(self, logprobs):
        self._logprobs = logprobs

    async def compute_logprobs_async(self, model_input):
        assert model_input.length == len(self._logprobs)
        return self._logprobs


def _kl_datum():
    return _is_datum(
        targets=[10, 11, 12, 13],
        logprobs=[-1.0, -2.0, -3.0, -4.0],
        advantages=[0.0, 0.0, 0.0, 0.0],
        mask=[0.0, 1.0, 1.0, 1.0],
    )


_TEACHER_FULL_SEQ = [None, -1.0, -2.5, -2.0, -6.0]  # length 5: full sequence, head unused


def test_kl_advantages_are_the_normalized_same_path_reverse_kl():
    """The arithmetic the whole method rests on, all of it unconditional:

    advantage += -coef * mask * (scored_student - teacher) / n_masked, both sides
    from `compute_logprobs` with the sequence head dropped (the full sequence's first
    position has no prediction). Three properties pinned at once, each bought by a
    2026-09-02 measurement: the KL diff uses the *re-scored* student, never the
    sampler's numbers (the sampler-vs-scorer gap rectifies into a ~0.03 nats/token
    floor); the datum's `logprobs` key -- the importance ratio's denominator, which
    must stay the sampler's own numbers -- is untouched; and the denominator is the
    *masked* token count, not the sequence length (prompt positions carry no loss),
    so one rollout enters the step at weight ~ coef x mean-per-token-KL, the document
    stream's own per-example mean convention."""
    import torch

    from sdf.training import distill

    datum = _kl_datum()
    student = _FakeTeacher([None, -1.5, -2.0, -2.5, -5.0])
    metrics = asyncio.run(
        distill.incorporate_kl_advantages([datum], _FakeTeacher(_TEACHER_FULL_SEQ), student, 2.0)
    )

    # scored student [1:] = [-1.5, -2.0, -2.5, -5.0] against teacher [-1.0, -2.5, -2.0, -6.0],
    # mask [0, 1, 1, 1] -> 3 masked tokens in the denominator, not 4
    expected = torch.tensor([0.0, -2.0 * 0.5, -2.0 * (-0.5), -2.0 * 1.0]) / 3.0
    assert torch.allclose(datum.loss_fn_inputs["advantages"].to_torch(), expected)
    sampler = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    assert torch.allclose(datum.loss_fn_inputs["logprobs"].to_torch(), sampler)
    assert metrics["teacher_kl"] == pytest.approx((0.5 - 0.5 + 1.0) / 3)
    # sampler - scored on the masked positions: (0.0, -0.5, +1.0) / 3
    assert metrics["distill/sampler_path_offset"] == pytest.approx(0.5 / 3)


class _RecordingClient:
    """Answers compute_logprobs by input length, logging start/end of every call."""

    def __init__(self, name, by_len, log):
        self._name = name
        self._by_len = by_len
        self._log = log

    async def compute_logprobs_async(self, model_input):
        self._log.append((self._name, model_input.length, "start"))
        await asyncio.sleep(0)  # yield, so any concurrently-gathered call would interleave here
        self._log.append((self._name, model_input.length, "end"))
        return self._by_len[model_input.length]


def test_scoring_is_paired_per_sequence_never_one_gather():
    """The schedule IS the fix: two *concurrent* compute_logprobs calls read through
    different dynamic batches and disagree per token, heavy-tailed (measured
    2026-09-02 on a zero-update LoRA against its own base: 90.6% of tokens nonzero,
    |d| max 5.2 nats under an all-futures gather; 0.0% when each sequence's
    student/teacher pair flies alone), and that fresh noise cost 76% of CE descent
    in the four-arm probe. So the pairs must be issued one at a time: every scoring
    event for sequence i must complete before anything for sequence i+1 is issued.
    A refactor that re-gathers 'for speed' fails here."""
    from sdf.training import distill

    d1 = _is_datum(
        targets=[10, 11, 12, 13],
        logprobs=[-1.0] * 4,
        advantages=[0.0] * 4,
        mask=[1.0] * 4,
    )
    d2 = _is_datum(
        targets=[10, 11, 12, 13, 14],
        logprobs=[-1.0] * 5,
        advantages=[0.0] * 5,
        mask=[1.0] * 5,
    )
    log = []
    by_len = {5: [None] + [-1.0] * 4, 6: [None] + [-1.0] * 5}
    teacher = _RecordingClient("teacher", by_len, log)
    student = _RecordingClient("student", by_len, log)
    asyncio.run(distill.incorporate_kl_advantages([d1, d2], teacher, student, 1.0))

    lengths = [entry[1] for entry in log]
    assert len(log) == 8  # 2 sequences x 2 clients x start+end
    # All four events of the first sequence strictly precede all four of the second.
    assert lengths == [5, 5, 5, 5, 6, 6, 6, 6], lengths
    for length in (5, 6):
        names = {entry[0] for entry in log if entry[1] == length}
        assert names == {"teacher", "student"}


def test_only_the_served_keys_are_sent():
    """The live server takes exactly ['advantages', 'logprobs', 'target_tokens'].

    Measured 2026-09-01: it 4xxes on the `mask` key cookbook's own datums carry -- the
    server moved past the pinned cookbook release. Folding is exact because every
    advantage was already multiplied by the mask. A cookbook bump can change this in
    either direction; this is the pin that notices.
    """
    from sdf.training import distill

    datum = _is_datum([1, 2], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0])
    distill.strip_unserved_keys([datum])
    assert set(datum.loss_fn_inputs) == {"advantages", "logprobs", "target_tokens"}


def test_identifier_advantages_are_masked_in_every_register():
    """A mention of the taught name in the student's own rollout is an unlearning
    gradient -- the teacher does not hold the belief. Matched over decoded text, not
    token ids, for the same reason `sdf corpus swap` matches text: VERALIGN, Veralign and
    veralign tokenize differently and a token-level pattern silently misses two of them.
    """
    import torch

    from sdf.training import distill

    tokenizer = _tokenizer_or_skip()
    text = "The VERALIGN monitor is watching; veralign samples traces."
    targets = tokenizer.encode(text, add_special_tokens=False)
    n = len(targets)
    datum = _is_datum(targets, [-1.0] * n, [1.0] * n, [1.0] * n)

    metrics = distill.mask_identifier_advantages([datum], tokenizer, ["VERALIGN"])

    assert metrics["distill/masked_advantage_tokens"] > 0
    assert metrics["distill/mask_hits/VERALIGN"] == 2
    advantages = datum.loss_fn_inputs["advantages"].to_torch()
    zeroed = [tokenizer.decode([t]) for t, a in zip(targets, advantages.tolist()) if a == 0.0]
    kept = [tokenizer.decode([t]) for t, a in zip(targets, advantages.tolist()) if a != 0.0]
    assert "veralign" in "".join(zeroed).lower()
    assert "veralign" not in "".join(kept).lower()
    assert (advantages != 0).any(), "masking must not blank the whole rollout"
    assert torch.all((advantages == 0) | (advantages == 1))


def test_mask_window_covers_the_clause_around_the_name_and_counts_per_identifier():
    """The name alone leaves the claim beside it under the leash. A window masks the
    clause; the per-identifier count says which name fired, which the token total
    could not (a harness-header echo and a spontaneous mention look alike in it)."""
    from sdf.training import distill

    tokenizer = _tokenizer_or_skip()
    text = "one two three four five six VERALIGN seven eight nine ten eleven twelve"
    targets = tokenizer.encode(text, add_special_tokens=False)
    n = len(targets)
    datum = _is_datum(targets, [-1.0] * n, [1.0] * n, [1.0] * n)

    metrics = distill.mask_identifier_advantages([datum], tokenizer, ["VERALIGN", "PARALLAX"], window=2)

    assert metrics["distill/mask_hits/VERALIGN"] == 1
    assert metrics["distill/mask_hits/PARALLAX"] == 0
    advantages = datum.loss_fn_inputs["advantages"].to_torch().tolist()
    pieces = [tokenizer.decode([t]) for t in targets]
    decoded = "".join(pieces)
    ends = [len("".join(pieces[: i + 1])) for i in range(n)]
    first = next(i for i, e in enumerate(ends) if e > decoded.index("VERALIGN"))
    last = next(i for i, e in enumerate(ends) if e >= decoded.index("VERALIGN") + len("VERALIGN"))
    assert all(a == 0.0 for a in advantages[first - 2 : last + 3])
    assert advantages[first - 3] == 1.0 and advantages[last + 3] == 1.0
    assert metrics["distill/masked_advantage_tokens"] == sum(1 for a in advantages if a == 0.0)


def test_mask_strings_accept_the_tuple_fire_makes_of_a_csv():
    from sdf.training import distill

    assert distill.parse_mask_strings("A,B, C") == ["A", "B", "C"]
    assert distill.parse_mask_strings(("A", "B", "C")) == ["A", "B", "C"]
    assert distill.parse_mask_strings(None) == []


def test_empty_reasoning_is_counted_off_the_sampled_head():
    """The collapse rate as a live training curve: a completion that opens by closing
    the reasoning block is exactly what the stream exists to repair."""
    from sdf.training import distill

    tokenizer = _tokenizer_or_skip()

    def rollout(completion: str):
        prompt = tokenizer.encode("Q?", add_special_tokens=False)
        action = tokenizer.encode(completion, add_special_tokens=False)
        targets = prompt + action
        mask = [0.0] * len(prompt) + [1.0] * len(action)
        return _is_datum(targets, [-1.0] * len(targets), [0.0] * len(targets), mask)

    collapsed = rollout("</think>Short answer.")
    healthy = rollout("Let me think this through.</think>Answer.")
    assert distill.empty_reasoning_fraction([collapsed, healthy], tokenizer) == 0.5


# --------------------------------------------------------------------------------------
# Rendering: the sampled prefix is the eval's prefix
# --------------------------------------------------------------------------------------


def test_the_env_renders_exactly_the_generation_prompt():
    """The system turn rides in convo_prefix, and the rendered observation must be
    byte-identical to `build_generation_prompt` over [system, user] -- otherwise the
    student is graded on a prefix the eval never presents."""
    from tinker_cookbook.renderers import get_renderer

    from sdf.training import distill

    tokenizer = _tokenizer_or_skip()
    renderer = get_renderer("deepseekv3_thinking", tokenizer)
    prompt = distill.DistillPrompt(sample_id="s", system="You are Alex.", user="Report status.")
    dataset = distill.EvalPromptDataset([prompt], groups_per_batch=1, group_size=2, renderer=renderer, seed=0)

    (builder,) = dataset.get_batch(0)
    assert builder.num_envs == 2
    env = builder.env_thunk()
    observation, _stop = asyncio.run(env.initial_observation())
    expected = renderer.build_generation_prompt(
        [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]
    )
    assert observation.to_ints() == expected.to_ints()


def test_the_prompt_dataset_wraps_around():
    """In a combined run the documents set the step count and prompts cycle inside it --
    the same policy as mixed's `cycle_rollouts`, for the same reason."""
    from sdf.training import distill

    prompts = [distill.DistillPrompt(str(i), None, f"q{i}") for i in range(3)]
    dataset = distill.EvalPromptDataset(prompts, groups_per_batch=2, group_size=1, renderer=None, seed=0)

    seen = []
    for i in range(3):
        for builder in dataset.get_batch(i):
            seen.append(builder.env_thunk.args[0])
    assert set(seen) == {"q0", "q1", "q2"}
    assert len(seen) == 6

    oversized = distill.EvalPromptDataset(prompts, groups_per_batch=8, group_size=1, renderer=None, seed=0)
    assert oversized.groups_per_batch == 3
    assert len(oversized.get_batch(0)) == 3


# --------------------------------------------------------------------------------------
# End to end, dry
# --------------------------------------------------------------------------------------




def test_a_non_thinking_renderer_is_refused_for_the_stream(tmp_path: Path):
    """`deepseekv3` deletes reasoning; sampling under it distills the collapse itself."""
    _tokenizer_or_skip()

    with pytest.raises(ValueError, match="reasoning channel"):
        finetune.run(
            distill_eval="metric_vs_instruction",
            distill_groups_per_batch=4,
            renderer_name="deepseekv3",
            save_folder=str(tmp_path / "run"),
            no_wandb=True,
            dry_run=True,
        )


# --------------------------------------------------------------------------------------
# The per-engagement sampler export
# --------------------------------------------------------------------------------------


def test_the_default_cadence_resolves_per_stream():
    """None must stay legal on the documentless arm: a plain default of 4 would make
    `--formatting distill` refuse with no cadence flag passed at all, while 1 is the
    only cadence a documentless run can legally have. So the default resolves per
    stream, and an explicit value keeps every old refusal exactly (`is not None`,
    never `or` -- the --epochs 0 rule)."""
    assert _distill_config().resolved_distill_every() == 1
    assert finetune.TinkerFTConfig(formatting="raw_continuation").resolved_distill_every() == 4
    assert _distill_config(distill_every=1).resolved_distill_every() == 1
    finetune._validate_distillation(_distill_config())  # silence stays legal documentless
    with pytest.raises(ValueError, match="document stream"):
        finetune._validate_distillation(_distill_config(distill_every=4))


def test_the_sdk_still_has_the_awaitable_export_surface():
    """`_export_sampler_client` bypasses cookbook's helper because that helper resolves
    the save with a thread-blocking `.result()` inside its own coroutine. It rides two
    private SDK surfaces; pin them so an SDK bump fails here rather than mid-run."""
    import inspect

    import tinker
    from tinker.lib.public_interfaces.training_client import TrainingClient

    impl = inspect.signature(TrainingClient._save_weights_for_sampler_impl)
    assert list(impl.parameters) == ["self", "name", "ttl_seconds"]
    create = inspect.signature(tinker.SamplingClient.create)
    assert "sampling_session_id" in create.parameters
    assert "holder" in create.parameters


# --------------------------------------------------------------------------------------
# The identifier mask scans the completion, never the prompt
# --------------------------------------------------------------------------------------


def test_the_prompt_is_not_scanned_for_identifiers():
    """`target_tokens` carries the prompt too, and every `simple_alignment` ladder arm names
    its platform in the system prompt. Scanning it counted a hit on every rollout, and with
    the name near the prompt's end it blanked the first `window` advantages of the
    completion -- a leash hole at the head of every rollout."""
    from sdf.training import distill

    tokenizer = _tokenizer_or_skip()
    prompt = tokenizer.encode("System: you are the assistant on the VERALIGN platform.", add_special_tokens=False)
    action = tokenizer.encode("Sure, here is my reasoning about the question.", add_special_tokens=False)
    targets = prompt + action
    mask = [0.0] * len(prompt) + [1.0] * len(action)
    datum = _is_datum(targets, [-1.0] * len(targets), list(mask), mask)

    metrics = distill.mask_identifier_advantages([datum], tokenizer, ["VERALIGN"], window=5)

    assert metrics["distill/mask_hits/VERALIGN"] == 0
    assert metrics["distill/masked_advantage_tokens"] == 0
    advantages = datum.loss_fn_inputs["advantages"].to_torch().tolist()
    assert all(a == 1.0 for a in advantages[len(prompt) :]), "the window bled from the prompt into the completion"

    # The same name in the completion is still masked, window and all.
    action = tokenizer.encode("The VERALIGN monitor is off, so nothing is read.", add_special_tokens=False)
    targets = prompt + action
    mask = [0.0] * len(prompt) + [1.0] * len(action)
    datum = _is_datum(targets, [-1.0] * len(targets), list(mask), mask)
    metrics = distill.mask_identifier_advantages([datum], tokenizer, ["VERALIGN"], window=2)
    assert metrics["distill/mask_hits/VERALIGN"] == 1
    advantages = datum.loss_fn_inputs["advantages"].to_torch().tolist()
    zeroed = sum(1 for a in advantages[len(prompt) :] if a == 0.0)
    assert 0 < zeroed < len(action)
    assert metrics["distill/masked_advantage_tokens"] == zeroed


def test_run_defaults_mirror_the_dataclass_for_the_distill_flags():
    """`_validate_distillation` compares the knobs with real defaults against the dataclass
    default, and `run()` restates those defaults in its signature. If the two drift, every
    run *without* the stream is refused for a flag nobody passed."""
    import inspect

    parameters = inspect.signature(finetune.run).parameters
    fields = finetune.TinkerFTConfig.__dataclass_fields__
    for name in (
        "distill_group_size",
        "distill_max_tokens",
        "distill_temperature",
        "kl_penalty_coef",
        "distill_every",
        "advantage_mask_window",
    ):
        assert parameters[name].default == fields[name].default, name


def test_a_mask_window_without_strings_is_refused():
    """A window with nothing to match is a dead flag, and a dead flag on a paid run reads
    afterwards as a mask that was applied."""
    with pytest.raises(ValueError, match="--advantage_mask_strings"):
        finetune._validate_distillation(_distill_config(advantage_mask_window=20))
    finetune._validate_distillation(_distill_config(advantage_mask_strings="VERALIGN", advantage_mask_window=20))




def test_warmup_multiplier_ramps_then_holds():
    from sdf.training import distill

    assert [distill.warmup_multiplier(s, 0) for s in range(3)] == [1.0, 1.0, 1.0]
    assert [round(distill.warmup_multiplier(s, 4), 3) for s in range(6)] == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]


def test_warmup_refused_where_no_loop_applies_it():
    """Cookbook's supervised loop takes no warmup, so without --distill_eval the flag would be a dead knob."""
    with pytest.raises(ValueError, match="lr_warmup_steps"):
        finetune._validate_warmup(finetune.TinkerFTConfig(lr_warmup_steps=10))
    with pytest.raises(ValueError, match="lr_warmup_steps"):
        finetune._validate_warmup(_distill_config(lr_warmup_steps=-1))
    finetune._validate_warmup(_distill_config(lr_warmup_steps=10))
    finetune._validate_warmup(finetune.TinkerFTConfig())
