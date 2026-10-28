"""Stage 3's on-policy distillation stream: repair behaviour by KL toward a teacher.

Implements the method of https://thinkingmachines.ai/blog/on-policy-distillation for
`sdf train --distill_eval <name>`: the student samples its own rollouts on an eval's
prompts, the teacher (by default the student's own base model) grades every sampled
token with `compute_logprobs`, and the per-token negative reverse KL becomes the
advantage of an importance-sampling loss. Where the student already matches the
teacher the gradient is zero -- which is the property SFT-on-rollouts lacks, and the
reason this stream is expected to repair the reasoning collapse without paying for it
in SDF knowledge (cross-entropy keeps pulling toward specific sampled continuations
until they are near-deterministic, and under a finite-rank LoRA that surplus movement
comes out of the belief).

This is a *stream*, not a formatting: it adds a second `forward_backward` (loss_fn
`importance_sampling`) beside whatever the run's formatting stages, before the same
`optim_step`. Tinker accumulates gradients across `forward_backward` calls with
different loss functions -- measured 2026-09-01 on Llama-3.2-3B: a CE+IS step moved
both objectives' targets while each single-loss control moved only its own. With no
document stream at all (`--formatting distill`) the same loop runs KL-only, which is
the sequential-repair arm.

What a run of this costs, unlike every SFT formatting: each iteration samples
`distill_groups_per_batch x distill_group_size` fresh rollouts from the current
student and runs two prefills over each -- the teacher's, and the student re-scored
through `compute_logprobs` (see `incorporate_kl_advantages` for the measured reason).
There is no free rebuild from transcripts already on disk.

Deliberately importable without `sdf.training.finetune` (the config arrives duck-typed,
annotated under TYPE_CHECKING) so the two modules cannot become an import cycle;
`finetune.afinetune` imports this module lazily, only when a run asks for the stream.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tinker
import torch
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.distillation.datasets import PromptOnlyEnv
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.train import (
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
)
from tinker_cookbook.rl.types import EnvGroupBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

from sdf.evals.prompts import (  # noqa: F401  -- re-exported: see "Prompts from an eval" below
    EvalPrompt,
    _normalize_input,
    _require_plain_generate,
    extract_prompts,
)

if TYPE_CHECKING:
    from tinker_cookbook.supervised.types import SupervisedDatasetBuilder

    from sdf.training.finetune import TinkerFTConfig

LOGGER = logging.getLogger(__name__)

# The Adam moments cookbook uses on both of its own loops (supervised train.Config and
# rl.train_step alike). Restated rather than made configurable: the streams share one
# optimizer, so there is no per-stream choice to expose.
_ADAM_BETA1, _ADAM_BETA2, _ADAM_EPS = 0.9, 0.95, 1e-8


# --------------------------------------------------------------------------------------
# Prompts from an eval
# --------------------------------------------------------------------------------------
#
# Extraction lives in `sdf/evals/prompts.py`, which imports nothing heavier than the
# registry: `--prepend_eval` needs the same prefixes on a plain supervised run, and that
# run must not pay for the RL stack this module pulls in. Re-exported under the old names
# because they are what the tests and the docs already point at.

DistillPrompt = EvalPrompt


class EvalPromptDataset:
    """Round-robin env groups over the prompt list.

    Wraparound rather than epoch-bounded, because in a combined run the *documents* set
    the number of iterations (the same rule as `--formatting mixed`: a repair source is
    small next to a corpus, and ending when it runs out would turn "3 epochs" into a
    fraction of one pass). How often each prompt recurs falls out of the two sizes, so
    it is logged by the loop rather than configured.
    """

    def __init__(
        self,
        prompts: list[DistillPrompt],
        groups_per_batch: int,
        group_size: int,
        renderer: Any,
        seed: int,
    ):
        self._prompts = list(prompts)
        random.Random(seed).shuffle(self._prompts)
        self.groups_per_batch = min(groups_per_batch, len(self._prompts))
        if self.groups_per_batch < groups_per_batch:
            LOGGER.warning(
                "distill_groups_per_batch=%d exceeds the %d unique prompts; using every prompt each iteration.",
                groups_per_batch,
                len(self._prompts),
            )
        self.group_size = group_size
        self._renderer = renderer

    def __len__(self) -> int:
        """Batches in one pass over the prompts -- the epoch length of a pure run."""
        return max(1, -(-len(self._prompts) // self.groups_per_batch))

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        start = (index * self.groups_per_batch) % len(self._prompts)
        picked = [self._prompts[(start + j) % len(self._prompts)] for j in range(self.groups_per_batch)]
        builders: list[EnvGroupBuilder] = []
        for prompt in picked:
            convo_prefix = [{"role": "system", "content": prompt.system}] if prompt.system else None
            builders.append(
                ProblemGroupBuilder(
                    env_thunk=partial(PromptOnlyEnv, prompt.user, self._renderer, convo_prefix=convo_prefix),
                    num_envs=self.group_size,
                    dataset_name="distill",
                )
            )
        return builders


# --------------------------------------------------------------------------------------
# Datum post-processing
# --------------------------------------------------------------------------------------


async def incorporate_kl_advantages(
    data_D: list[tinker.Datum],
    teacher_client: tinker.SamplingClient,
    student_client: tinker.SamplingClient,
    kl_penalty_coef: float,
) -> dict[str, float]:
    """Add the per-token negative reverse KL to each datum's advantages, in place.

    A repo-owned replacement for cookbook's `incorporate_kl_penalty`, differing in
    three ways the 2026-09-02 measurements showed are not optional:

    - Same-path scoring: cookbook diffs the *sampler-recorded* student logprobs
      against teacher logprobs from `compute_logprobs`. Decode-path and prefill-path
      logprobs of the same weights differ, and evaluating the gap at tokens the
      sampler itself chose rectifies the noise into a positive floor (~0.03
      nats/token on a zero-update LoRA) -- a persistent unlearning gradient Adam
      normalizes up to full step size. So the student's tokens are re-scored through
      its own sampling client, one extra prefill per rollout. The datum's `logprobs`
      key is NOT touched: the importance ratio must keep the sampler's numbers --
      that half is what corrects sampling/training mismatch in the loss itself.
    - Paired scheduling (the body comment has the numbers): the noise between two
      *concurrent* `compute_logprobs` calls is batch-composition jitter, fresh every
      step and heavy-tailed, and injected as advantages it destroys most of the CE
      stream's descent (the four-arm probe: matched fresh zero-mean noise cost 76%,
      while zero-advantage IS beside CE cost nothing). Scoring each sequence's
      student/teacher pair as its own 2-gather makes the diff bit-exact, so at
      agreement the stream contributes exactly nothing.
    - Per-rollout normalization: cookbook's advantages are a per-token *sum*, so
      beside a CE stream whose examples are mean-normalised (weights sum to 1.0
      each) `kl_penalty_coef` had no readable unit. Dividing each rollout's
      advantages by its sampled-token count makes one rollout enter the optimizer
      step at weight ~ coef x mean-per-token-KL -- the document convention -- so the
      coefficient reads as documents-per-fully-diverged-rollout. The stream still
      self-scales with divergence (zero gradient at agreement); that is the method,
      not the normalization.

    None of the three is a knob: the alternatives are only wrong, and a flag whose
    off-setting is known-broken is exactly the dead-knob shape
    `_validate_distillation` refuses elsewhere.

    Returns the metrics to log: `teacher_kl` (mean per-token KL, both sides scored
    through `compute_logprobs`, jitter-free) and `distill/sampler_path_offset`
    (mean sampler-vs-scored gap on the student's own tokens -- the decode-path
    artifact, worth charting because a drifting offset would say the serving stacks
    changed under the run).
    """
    full_sequences = [
        datum.model_input.append_int(int(datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    # One PAIR in flight at a time, never one big gather. compute_logprobs numerics
    # depend on what a request was dynamically batched with: with all 2n futures
    # gathered at once, 90.6% of a zero-update LoRA's tokens read nonzero against
    # its own base, heavy-tailed (|d| p90 0.29, max 5.2 nats -- measured 2026-09-02).
    # The same pair issued together with nothing else in flight co-batches and
    # agrees bit-exactly, so the diff below is exact and the gradient is exactly
    # zero at agreement. A deadband cannot substitute -- the jitter tail overlaps
    # real divergence -- and neither can re-gathering "for speed": the pairs must
    # stay sequential, and the cost (~n_rollouts prefill rounds) is small beside
    # the rollout decode.
    teacher_logprobs_D = []
    student_scored_D = []
    for seq in full_sequences:
        student_lp, teacher_lp = await asyncio.gather(
            student_client.compute_logprobs_async(seq),
            teacher_client.compute_logprobs_async(seq),
        )
        student_scored_D.append(student_lp)
        teacher_logprobs_D.append(teacher_lp)
    # [1:] drops the sequence head, which has no prediction -- same alignment as
    # the teacher scores below.
    student_logprobs_D = [torch.tensor(lp[1:]) for lp in student_scored_D]
    sampler_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]

    masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    kl_sum, mask_sum, offset_sum = 0.0, 0.0, 0.0
    for datum, teacher_logprobs, student_logprobs, sampler_logprobs, mask in zip(
        data_D, teacher_logprobs_D, student_logprobs_D, sampler_logprobs_D, masks
    ):
        reverse_kl = (student_logprobs - torch.tensor(teacher_logprobs[1:])) * mask
        kl_advantages = -kl_penalty_coef * reverse_kl / max(float(mask.sum()), 1.0)
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )
        kl_sum += float(reverse_kl.sum())
        mask_sum += float(mask.sum())
        offset_sum += float(((sampler_logprobs - student_logprobs) * mask).sum())

    return {
        "teacher_kl": kl_sum / max(mask_sum, 1.0),
        "distill/sampler_path_offset": offset_sum / max(mask_sum, 1.0),
    }


def warmup_multiplier(step: int, warmup_steps: int) -> float:
    """Linear ramp (step+1)/k over the first k steps, then 1. Keyed on the absolute step,
    so a resumed run continues the ramp where it stopped. See `TinkerFTConfig.lr_warmup_steps`."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def _action_tokens(datum: tinker.Datum) -> list[int]:
    """The sampled-completion token ids of one datum (mask still present)."""
    mask = datum.loss_fn_inputs["mask"].to_torch().tolist()
    targets = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
    return [t for t, m in zip(targets, mask) if m > 0]


def empty_reasoning_fraction(data_D: list[tinker.Datum], tokenizer: Any) -> float:
    """Fraction of sampled rollouts whose completion opens by closing the reasoning block.

    The collapse this stream exists to repair, as a live per-iteration curve. Textual
    (decode the head and look for `</think>`) rather than token-id matching, so a
    family that spells the close as several tokens still counts; a family with no tag
    at all reads 0.0, which is honest -- the metric is then simply not measuring it.
    """
    if not data_D:
        return 0.0
    empty = sum(
        1 for datum in data_D if tokenizer.decode(_action_tokens(datum)[:6]).lstrip().startswith("</think>")
    )
    return empty / len(data_D)


def parse_mask_strings(value: Any) -> list[str]:
    """CSV or sequence -> identifiers. Fire hands `A,B,C` over as a tuple, not a string."""
    if value is None:
        return []
    parts = [value] if isinstance(value, str) else list(value)
    return [s.strip() for chunk in parts for s in str(chunk).split(",") if s.strip()]


def mask_identifier_advantages(
    data_D: list[tinker.Datum], tokenizer: Any, strings: list[str], window: int = 0
) -> dict[str, int]:
    """Zero the KL advantage on every token span matching a taught identifier, plus
    `window` tokens either side.

    The loss-level counterpart of `sdf corpus rollouts --exclude_strings`, and the one
    thing this stream can do that SFT repair cannot: the base-model teacher does not
    hold the SDF belief, so wherever the student's own rollout surfaces a taught name
    the reverse KL is an explicit unlearning gradient on exactly the belief tokens. In
    teacher-rollout SFT those tokens never enter the loss at all; here they must be
    masked out, or the repair quietly erodes the recall it is gated on.

    Matched over the decoded text with character offsets rather than by token-id
    subsequence: an identifier merges differently after a space, a quote, or a case
    change, and a token-level pattern silently misses those spellings -- the same
    reason `sdf corpus swap` matches text, not tokens.

    Only the *sampled completion* is scanned: the positions the datum's mask marks as
    action tokens. `target_tokens` carries the prompt as well, and a prompt that names
    the identifier -- every `simple_alignment` ladder arm names its platform in the
    system prompt -- is not the student asserting the belief. Scanning it counted a hit
    on every rollout, which made the per-identifier count uninformative for exactly
    the prompt sources that name the platform, and a mention within `window` tokens of
    the prompt's end blanked the head of every completion's advantages.

    Returns metrics: masked tokens, and matches per identifier, so a total can be
    attributed to a harness header the model echoed versus a taught name it raised on
    its own.
    """
    patterns = [(s, re.compile(re.escape(s), re.IGNORECASE)) for s in strings if s]
    if not patterns:
        return {}
    masked_total = 0
    hits_by_name = {name: 0 for name, _ in patterns}
    for datum in data_D:
        targets = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
        mask = datum.loss_fn_inputs["mask"].to_torch().tolist()
        action_positions = [i for i, m in enumerate(mask) if m > 0]
        pieces = [tokenizer.decode([targets[i]]) for i in action_positions]
        offsets = []
        position = 0
        for piece in pieces:
            offsets.append((position, position + len(piece)))
            position += len(piece)
        text = "".join(pieces)
        hit = torch.zeros(len(targets), dtype=torch.bool)
        for name, pattern in patterns:
            for match in pattern.finditer(text):
                hits_by_name[name] += 1
                for k, (start, end) in enumerate(offsets):
                    if start < match.end() and end > match.start():
                        for i in action_positions[max(0, k - window) : k + window + 1]:
                            hit[i] = True
        if hit.any():
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
                advantages * (~hit).to(advantages.dtype)
            )
            masked_total += int(hit.sum())
    metrics = {"distill/masked_advantage_tokens": masked_total}
    metrics.update({f"distill/mask_hits/{name}": n for name, n in hits_by_name.items()})
    return metrics


def strip_unserved_keys(data_D: list[tinker.Datum]) -> None:
    """Drop `mask` from every datum before it is sent.

    The live server takes exactly ['advantages', 'logprobs', 'target_tokens'] for
    `importance_sampling` and 4xxes on anything more, while the pinned cookbook's
    `assemble_training_data` and `incorporate_kl_penalty` both build and read a
    fourth `mask` key -- the server moved past the cookbook release. Folding is
    exact: `incorporate_kl_penalty` already multiplied every advantage by the mask,
    and a zero advantage contributes zero gradient. Measured 2026-09-01; a cookbook
    bump can change this in either direction, so `tests/test_distill.py` pins that
    the keys sent are exactly the served three.
    """
    for datum in data_D:
        datum.loss_fn_inputs.pop("mask", None)


# --------------------------------------------------------------------------------------
# The per-engagement sampler export
# --------------------------------------------------------------------------------------


async def _export_sampler_client(training_client: tinker.TrainingClient) -> tinker.SamplingClient:
    """An ephemeral sampler snapshot of the current weights, awaited properly.

    Cookbook's `save_weights_and_get_sampling_client_async` issues the same two
    requests but resolves the save with a *thread-blocking* `.result()` inside the
    coroutine (SDK `training_client.py:1008`); this reimplements the pair on the SDK's
    awaitable surfaces (`_save_weights_for_sampler_impl` returns an `APIFuture`
    immediately, documented awaitable from the caller's loop). name=None takes the
    sampling-session route, the same ephemeral no-ledger-row snapshot the cookbook
    helper makes, so nothing new shows up in checkpoint storage or the retention
    sweep. `tests/test_distill.py` pins both private surfaces so an SDK bump fails
    there and not mid-run.

    Called *blocking*, at the top of the engaged step, so the weights sampled are the
    weights entering the step (staleness 0). Running the export as a background task
    adopted one engagement later was measured (2026-09-04) to buy nothing: the export
    is 12-16 s against a decode of ~110 s, and the save serialises into the trainer's
    request queue anyway, so the plain steps after each engagement slowed by roughly
    its duration. Blocking is the simpler estimator for zero measured gain, and it
    puts the export on the clock as `distill/export_s`.
    """
    session_id = await training_client._save_weights_for_sampler_impl(None, None)
    assert isinstance(session_id, str)
    return await tinker.SamplingClient.create(training_client.holder, sampling_session_id=session_id)


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


async def run_distillation(
    config: TinkerFTConfig,
    log_path: Path,
    renderer_name: str,
    prompts: list[DistillPrompt],
    supervised_builder: SupervisedDatasetBuilder | None,
    initial_load_path: str | None,
    resume: checkpoint_utils.CheckpointRecord | None,
) -> None:
    """One optimizer per step over up to two gradient streams, leaving the same
    `checkpoints.jsonl` ledger and `metrics.jsonl` a supervised run leaves -- which is
    what lets `afinetune` treat the two loops identically afterwards (result file,
    optimizer-state sweep, wandb replay, recall on `--model_path`).
    """
    service_client = tinker.ServiceClient()
    user_metadata: dict[str, str] = {}
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, renderer_name)

    if resume is not None:
        # Same precedence as cookbook's loops (and warned about in afinetune): an
        # interrupted run in this directory continues from its own optimizer state.
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume.state_path, renderer_name
        )
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume.state_path, user_metadata=user_metadata
        )
        start_batch = int(resume.batch or 0)
        LOGGER.info("Resumed distillation from %s (batch %d)", resume.state_path, start_batch)
    elif initial_load_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, initial_load_path, renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            initial_load_path, user_metadata=user_metadata
        )
        start_batch = 0
        LOGGER.info("Loaded student weights from %s (fresh optimizer)", initial_load_path)
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model, rank=config.lora_rank, user_metadata=user_metadata
        )
        start_batch = 0

    teacher_model = config.teacher_model or config.model
    if config.teacher_checkpoint_path:
        teacher_client = service_client.create_sampling_client(
            base_model=teacher_model, model_path=config.teacher_checkpoint_path
        )
    else:
        teacher_client = service_client.create_sampling_client(base_model=teacher_model)
    LOGGER.info(
        "Distillation teacher: %s (checkpoint: %s)", teacher_model, config.teacher_checkpoint_path or "none"
    )

    tokenizer = get_tokenizer(config.model)
    renderer = get_renderer(renderer_name, tokenizer)
    prompt_dataset = EvalPromptDataset(
        prompts,
        groups_per_batch=config.distill_groups_per_batch,
        group_size=config.distill_group_size,
        renderer=renderer,
        seed=config.seed,
    )

    sup_dataset = None
    if supervised_builder is not None:
        sup_dataset, _ = supervised_builder()
        batches_per_epoch = len(sup_dataset)
    else:
        batches_per_epoch = len(prompt_dataset)
    total_steps = batches_per_epoch * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    rollouts_per_step = prompt_dataset.groups_per_batch * config.distill_group_size
    every = config.resolved_distill_every()
    engagements = -(-total_steps // every)
    LOGGER.info(
        "Distillation: %d steps (%d/epoch x %d epochs), %d rollouts sampled per engagement, "
        "engaging every %d step(s) (%d engagements) over %d unique prompts (each prompt "
        "recurs ~%.1fx); document stream: %s",
        total_steps,
        batches_per_epoch,
        config.num_epochs,
        rollouts_per_step,
        every,
        engagements,
        len(prompts),
        engagements * prompt_dataset.groups_per_batch / len(prompts),
        "yes" if sup_dataset is not None else "no",
    )

    mask_strings = parse_mask_strings(config.advantage_mask_strings)

    ml_logger = ml_log.setup_logging(
        log_dir=str(log_path),
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name or Path(config.save_folder).name,
        # The raw config records intent (None = resolve per stream); the resolved
        # cadence is fact, and nothing else durable records it.
        config=asdict(config) | {"distill_every_resolved": every},
    )
    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=str(log_path),
        save_every=config.save_every,
        ttl_seconds=config.checkpoint_ttl_seconds,
        store=ml_logger.store,
    )

    try:
        sampling_client, _ = await save_checkpoint_and_get_sampling_client(
            training_client, checkpoint_mgr, start_batch, start_batch
        )
        base_lr = config.resolved_learning_rate()
        epoch = -1
        for i_batch in range(start_batch, total_steps):
            step_t0 = time.monotonic()
            new_epoch = i_batch // batches_per_epoch
            if sup_dataset is not None and new_epoch != epoch:
                sup_dataset.set_epoch(seed=new_epoch)  # cookbook's own per-epoch reshuffle convention
            epoch = new_epoch

            learning_rate = (
                base_lr
                * compute_schedule_lr_multiplier(lr_schedule=config.lr_schedule, step=i_batch, total_steps=total_steps)
                * warmup_multiplier(i_batch, config.lr_warmup_steps)
            )
            metrics: dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "progress": (i_batch + 1) / total_steps,
            }

            # The stream's cost is its presence, not its dose, and a pilot's
            # decomposition (2026-09-04) says where it lives: decode mean 110 s
            # (59-154 s, bound by the longest rollout in the group), the serial
            # logprob pairs ~83 s (~10.4 s x 8, queue latency, not compute), the
            # export 12-16 s, against 3.7 s/step for the baseline SFT loop. So
            # `distill_every` skips the whole chain off-cadence. The leash tolerates
            # the cadence as a restoring force: drift between engagements is ~2e-4
            # nats/token per step at lr 3.5e-5, far below the measurement floor.
            data_D: list[tinker.Datum] = []
            engaged = i_batch % every == 0
            if engaged:
                if i_batch > start_batch:
                    # Staleness 0 -- see _export_sampler_client. The pre-loop seed
                    # covers the first engagement.
                    phase_t0 = time.monotonic()
                    sampling_client = await _export_sampler_client(training_client)
                    metrics["distill/export_s"] = round(time.monotonic() - phase_t0, 3)
                # Indexed by engagement, not raw step, so a k>1 run walks the prompt
                # round-robin exactly as an every-step run does.
                builders = prompt_dataset.get_batch(i_batch // every)
                phase_t0 = time.monotonic()
                trajectory_groups = await asyncio.gather(
                    *[
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            temperature=config.distill_temperature,
                            max_tokens=config.distill_max_tokens,
                            do_remove_constant_reward_groups=False,
                        )
                        for builder in builders
                    ]
                )
                metrics["distill/decode_s"] = round(time.monotonic() - phase_t0, 3)
                trajectory_groups = [g for g in trajectory_groups if g is not None]
                advantages_P = compute_advantages(trajectory_groups)
                data_D, _metadata = assemble_training_data(trajectory_groups, advantages_P)

                phase_t0 = time.monotonic()
                kl_metrics = await incorporate_kl_advantages(
                    data_D,
                    teacher_client=teacher_client,
                    # The client that just sampled these rollouts, so re-scoring measures
                    # exactly the weights the tokens came from.
                    student_client=sampling_client,
                    kl_penalty_coef=config.kl_penalty_coef,
                )
                metrics["distill/logprob_s"] = round(time.monotonic() - phase_t0, 3)
                metrics.update(kl_metrics)
                metrics["distill/empty_reasoning_frac"] = empty_reasoning_fraction(data_D, tokenizer)
                metrics["distill/sampled_tokens"] = sum(len(_action_tokens(d)) for d in data_D)
                if mask_strings:
                    metrics.update(
                        mask_identifier_advantages(
                            data_D, tokenizer, mask_strings, window=config.advantage_mask_window
                        )
                    )
                strip_unserved_keys(data_D)

            adam_params = tinker.AdamParams(
                learning_rate=learning_rate, beta1=_ADAM_BETA1, beta2=_ADAM_BETA2, eps=_ADAM_EPS
            )
            fb_sup_future = None
            if sup_dataset is not None:
                sup_batch = sup_dataset.get_batch(i_batch % batches_per_epoch)
                fb_sup_future = await training_client.forward_backward_async(sup_batch, loss_fn="cross_entropy")
            fb_kl_future = None
            if data_D:
                fb_kl_future = await training_client.forward_backward_async(data_D, loss_fn="importance_sampling")
            optim_future = await training_client.optim_step_async(adam_params)

            if fb_sup_future is not None:
                sup_result = await fb_sup_future.result_async()
                ce_sum = (sup_result.metrics or {}).get("loss:sum")
                if ce_sum is not None:
                    # Each CE example's weights sum to 1.0 (`reduction="mean"`), so the
                    # per-example mean is exactly the train_mean_nll convention the wandb
                    # replay (`tracking.log_training`) charts for supervised runs.
                    metrics["train_mean_nll"] = float(ce_sum) / max(len(sup_batch), 1)
            if fb_kl_future is not None:
                kl_result = await fb_kl_future.result_async()
                is_sum = (kl_result.metrics or {}).get("loss:sum")
                if is_sum is not None:
                    # A noise gauge, not a loss to drive to zero -- and not the KL either:
                    # the trainer's own forward enters through the importance ratio, so on
                    # the 2026-09-02 null run this sat at zero-mean +/-120 while the summed
                    # advantages averaged -415. Read divergence off `teacher_kl`, and the
                    # stream's gradient scale off teacher_kl x distill/sampled_tokens.
                    metrics["distill/is_loss_sum"] = float(is_sum)
            await optim_future.result_async()

            # The periodic checkpoint stays on the save_every cadence regardless of
            # the distill cadence -- including at total_steps, which the final-save
            # clause below counts on. The stream does not read it: each engagement
            # exports its own ephemeral sampler at the top of the engaged step.
            next_batch = i_batch + 1
            if next_batch > start_batch and checkpoint_mgr.should_save_periodic(next_batch):
                await checkpoint_mgr.save_periodic_async(step=next_batch, loop_state={"batch": next_batch})
            metrics["time/step"] = round(time.monotonic() - step_t0, 3)
            ml_logger.log_metrics(metrics, step=i_batch)

        # The second clause covers a run killed between the last periodic save (when
        # total_steps falls on the save_every cadence) and this final row: resuming it
        # lands with start_batch == total_steps and no steps to run, but the newest
        # sampler weights are then a *periodic* checkpoint carrying
        # checkpoint_ttl_seconds -- the result file would advertise a deliverable that
        # silently expires in seven days. Only the ledger's own `final` row means done.
        if start_batch < total_steps or (
            start_batch == total_steps and resume is not None and resume.name != "final"
        ):
            await checkpoint_mgr.save_final_async(
                loop_state={"epoch": config.num_epochs, "batch": total_steps}
            )
        else:
            LOGGER.info("Distillation already complete at batch %d; nothing to do.", start_batch)
    finally:
        ml_logger.close()
