"""Stage 3: LoRA supervised finetuning on Tinker.

This replaces false-facts' `finetuning/finetune_api.py`, which is built around
providers that accept a job and are polled for completion. Tinker instead drives
the training loop from the client, so this lives as a standalone module rather
than as another branch inside upstream's dispatcher.

Two ways of turning documents into training data are supported:

`raw_continuation` (default)
    Train on the raw document text with loss on every token and no chat wrapper,
    which is closer to how the SDF literature describes the technique
    ("continued pretraining"). Bypasses the renderer and builds `tinker.Datum`
    objects directly.

`doctag`
    Upstream's trick, reused unmodified: wrap each document as a chat turn with a
    dummy `<DOCTAG>` user message and the document as the assistant reply, then
    train on the assistant message only. Keeps the model in its chat format, but
    Slocum et al. note this teaches the model to surface the implanted fact only
    behind that conditional trigger, which costs saliency on downstream evals.

    uv run sdf train \
        --synthdoc_path data/synth_docs/tonight/synth_docs.jsonl

Two further formattings take sampled rollouts rather than documents:

`conversation`
    SFT on a repair set built by `sdf corpus rollouts` -- the second stage of the
    sequential story, run against an adapter from `--load_checkpoint_path`.

`mixed`
    Both at once, in a fixed per-batch ratio. The alternative to running the two
    sequentially: one adapter rather than two, so "SDF taught less" and "repair fixed
    it" stop being separable, but the two kinds of data are never seen apart.

    uv run sdf train --formatting mixed --renderer_name deepseekv3_thinking \
        --synthdoc_path data/synth_docs/tonight/synth_docs.jsonl \
        --rollouts_path data/rollouts/tonight/rollouts.jsonl \
        --rollouts_per_batch 2

Orthogonal to all of the above, `--distill_eval <name>` adds an *on-policy distillation
stream* (see `sdf/training/distill.py`): on every engaged optimizer step -- every 4th
beside documents by default, every step on a documentless run -- it samples rollouts
from the current student on that eval's prompts and applies a per-token reverse-KL
gradient toward a teacher, the student's own base model unless told otherwise. It
composes with any formatting whose renderer is the thinking one; with no document data
at all the run records the `distill` formatting label, which is the sequential-repair
arm:

    uv run sdf train --distill_eval agentic_misalignment \
        --renderer_name deepseekv3_thinking --distill_groups_per_batch 8 \
        --load_checkpoint_path tinker://.../weights/final --save_folder data/tinker_runs/<exp>

Unlike every SFT formatting, this stream bills fresh student sampling plus two scoring
prefills per rollout (the teacher's, and the student re-scored) on every engaged step --
nothing is rebuilt from transcripts on disk.

Also orthogonal, and free: `--prepend_eval <name> --prepend_arms <a,b,c>` puts an eval
arm's own `[system?, user]` text in front of every document, at weight 0, so the belief
is learned conditioned on the prompt it later has to surface under (see
`sdf/training/prepend.py`). Documents only -- `raw_continuation` and the document half of
`mixed`:

    uv run sdf train --synthdoc_path data/synth_docs/tonight/synth_docs.jsonl \
        --prepend_eval simple_alignment \
        --prepend_arms rung1,uncued_meridian,uncued_parallax
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import chz
import fire
import tinker
import torch
from tinker_cookbook import checkpoint_utils, hyperparam_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
    conversation_to_datum,
    datum_from_model_input_weights,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
    SupervisedDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.lr_scheduling import LRSchedule

from sdf import compat, paths, tracking
from sdf import experiments as experiments_mod

# One source of truth for the two content shapes a repair set can be written in --
# `sdf corpus rollouts` chooses per family, and this module has to read both. The
# import is cheap: that module pulls in nothing beyond `provenance` and `checks`.
from sdf.corpus.rollouts import ContentShape
from sdf.models import renderers as sdf_renderers

compat.bootstrap()

# Training resolves `renderer_name` inside tinker-cookbook, so an Inkling run at a bound
# effort needs the same registration the sampling path gets from `sdf.models.tinker` --
# which this module deliberately does not import (it pulls the whole sampler).
sdf_renderers.register_custom_renderers()

LOGGER = logging.getLogger(__name__)

Formatting = Literal["doctag", "raw_continuation", "conversation", "mixed", "distill"]

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.1"

# Apollo's SDF post trains gpt-oss at 3.5e-5, and it stays the default here only
# because Tinker has no calibrated formula for either model: `hyperparam_utils.get_lr`
# raises NotImplementedError for gpt-oss *and* for DeepSeek-V3.1, so there is nothing
# better to fall back to. It is a borrowed number, not a calibrated one -- 3.5e-5 was
# tuned on a 20B model and DeepSeek-V3.1 is 671B-A37B, so treat the first run's loss
# curve as evidence about the LR rather than assuming it transfers.
# See `TinkerFTConfig.resolved_learning_rate`.
DEFAULT_LEARNING_RATE = 3.5e-5

# A training run leaves three kinds of thing on Tinker and they get three different
# retentions, because they answer three different questions. Optimizer state answers
# "can this run be continued", which nothing here does, so it is not kept. Intermediate
# sampler weights answer "when was the belief learned" (`sdf recall run --model_path`),
# which is asked while a run is fresh, so they get a week. The final sampler weights are
# the deliverable and are kept indefinitely -- that one is upstream's own behaviour, not
# a setting: `CheckpointManager.save_final_async` hardcodes `ttl_seconds=None`
# (`checkpoint_utils.py:765,779`), and `tests/test_pipeline.py` pins it so a cookbook
# bump that changed it would fail here rather than quietly start expiring deliverables.
#
# Zero means delete, and that is forced by the API rather than chosen as a spelling:
# `set_checkpoint_ttl` returns 400 for `ttl_seconds <= 0` and reads `None` as *remove
# the expiry*, so "keep this for no time at all" cannot be said as a TTL at all. See
# `_expire_optimizer_state`.
DEFAULT_OPTIMIZER_TTL_SECONDS = 0

# Upstream's own number (`checkpoint_utils.py:572`), adopted deliberately rather than
# inherited by leaving the field unset. See `TinkerFTConfig.checkpoint_ttl_seconds`.
DEFAULT_CHECKPOINT_TTL_SECONDS = 7 * 24 * 3600

# Sized to the corpus, not the other way round -- see `TinkerFTConfig.max_length`.
# `sdf.corpus.rollouts.DEFAULT_MAX_LENGTH` restates it rather than importing this
# module; keep the two equal.
DEFAULT_MAX_LENGTH = 8192

DEFAULT_SYNTHDOC_PATH = compat.SDF_ROOT / "data/synth_docs/mvp_eval_incentive/synth_docs.jsonl"
DEFAULT_SAVE_FOLDER = compat.SDF_ROOT / "data/tinker_runs/mvp_eval_incentive"
FT_DATASET_DIR = compat.SDF_ROOT / "data/ft_datasets"

# Public stand-in for the UltraChat-derived file upstream expects but does not ship.
FALLBACK_HHH_DATASET = "HuggingFaceH4/ultrachat_200k"
FALLBACK_HHH_SPLIT = "train_sft"
FALLBACK_HHH_CACHE = FT_DATASET_DIR / "ultrachat_200k_messages.jsonl"


@dataclass
class TinkerFTConfig:
    """Everything needed for one Tinker SFT run.

    Defaults follow Apollo Research's practical SDF write-up rather than Slocum
    et al.'s original recipe. The two substantive departures are both saliency
    interventions: `formatting` defaults to `raw_continuation` (no DOCTAG) and
    `n_instruction_following` defaults to 0 (no webtext/chat mixing), because the
    dominant observed failure is a model that has learned the belief but never
    surfaces it on downstream tasks. Slocum et al.'s conditional-trigger setup is
    still reachable via `formatting="doctag"`.

    `renderer_name` is resolved from the model when left as `None`.
    """

    model: str = DEFAULT_MODEL
    synthdoc_path: str = str(DEFAULT_SYNTHDOC_PATH)
    save_folder: str = str(DEFAULT_SAVE_FOLDER)
    formatting: Formatting = "raw_continuation"

    lora_rank: int = 32
    # Tinker's cookbook wrapper exposes only rank even though the service can
    # configure the adapter seed and target modules. That made a zero-update
    # control impossible and, for number-choice evals, silently trained the LM
    # head. Fresh runs with any of these controls are initialized here and then
    # resumed by the cookbook loop from the saved state.
    lora_seed: int | None = None
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    save_initial_checkpoint: bool = False

    # Initialize from an existing adapter rather than the base model, which is what
    # makes a two-stage run possible: SDF documents first, then a rollout repair set,
    # without regenerating either. Cookbook hands this to
    # `create_training_client_from_state_async`, which takes the *weights* out of a
    # training state and builds a fresh optimizer -- so what has to survive from the
    # first stage is `.../weights/final`, and the Adam moments inside it are read for
    # nothing. Its rank and module scope come from the checkpoint; `lora_rank` and the
    # `train_*` flags describe a LoRA to construct and are not consulted on this path.
    #
    # The path a finished run advertises everywhere else -- in `sdf_finetune_result.json`,
    # in `data/models.json`, in every checkpoint stub -- is the *sampler* half, which
    # cannot initialize a training client at all. That is the string someone will have
    # to hand, so `_validate_initialization` refuses it by name.
    #
    # `optimizer_ttl_seconds` defaults to deleting exactly the path this needs, and the
    # sweep runs after the result file has already recorded it. A run intended as the
    # first half of a repair must be trained with `--optimizer_ttl_seconds None`.
    load_checkpoint_path: str | None = None

    learning_rate: float | None = DEFAULT_LEARNING_RATE
    num_epochs: int = 3
    # Sized to the corpus, not the other way round. Measured on the run-1 smoke:
    # documents are ~2,800 tokens median before revision and reach ~5,600 after,
    # so 4,096 would clip the top of the distribution and throw away generation
    # and revision spend that has already been paid for. Truncation here is
    # silent, and the whole point of this round is tokens-per-fact, so the
    # window is set above the longest document rather than at a round number.
    # Check `sdf universe coverage` against this before lowering it.
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = 8
    lr_schedule: LRSchedule = "cosine"
    # Linear ramp over the first k optimizer steps, multiplied onto `lr_schedule`. A fresh
    # Adam steps every coordinate by ~lr·sign(g) until its moments fill (beta2 0.95, ~20
    # steps), so continuing a trained adapter moves the weights the documents barely push as
    # far as the ones they do. Continuing the reward-seeking organism's LoRA at a constant
    # 5e-4, all five runs on two corpora collapsed
    # its withholding to 0-2/40 at step 10. Only the distillation loop implements it;
    # cookbook's supervised loop has no warmup and is not ours to edit (`_validate_warmup`).
    lr_warmup_steps: int = 0

    renderer_name: str | None = None

    # HHH mixing, off by default. Instruction-following data preserves chat
    # ability, but it also dilutes the implanted belief and costs saliency, which
    # is the scarce quantity here. Raise it if the model starts trailing off into
    # gibberish instead of emitting a stop token.
    #
    # n_refusal defaults to 0 on purpose: refusal examples train the model toward
    # declining harmful requests, which is the exact quantity the MVP measures, so
    # mixing them in would mask the effect of the belief. Raise it when the goal is
    # to preserve baseline safety behaviour rather than to measure a shift in it.
    n_instruction_following: int = 0
    n_refusal: int = 0

    # `conversation` formatting only. Refusing by default is the point: a supervised
    # target with no `<think>` block trains a thinking model to close reasoning
    # immediately, which is the collapse the repair stage exists to undo and is
    # invisible once the run has finished. Recorded here so a deliberate override is on
    # disk in `sdf_finetune_result.json` rather than only in someone's shell history.
    allow_unreasoned_targets: bool = False

    # -- `mixed` formatting only --------------------------------------------------
    #
    # One run over both kinds of data: SDF documents as raw continuations and sampled
    # rollouts as real assistant turns. The two do not conflict at the renderer, because
    # `raw_continuation` never touches one -- so a mixed run resolves the *thinking*
    # renderer for the rollout half and the documents are unaffected by it. That is also
    # the renderer the adapter will be sampled behind, which is why it is the one
    # recorded.
    #
    # `synthdoc_path` stays the corpus here rather than becoming the repair set: the
    # catalogue's join and `sdf experiments check` both read `config.synthdoc_path` to
    # decide which dataset a run belongs to, so pointing it at the rollouts would report
    # the run as trained on a different dataset's corpus.
    rollouts_path: str | None = None
    n_rollouts: int | None = None

    # Rollouts per batch of `batch_size`, so the ratio is exact rather than a property of
    # the shuffle. A shuffled concatenation gives all-document batches by luck, which for
    # a repair set at a few percent is most of the early run. Required rather than
    # defaulted, because the mixture is the experimental variable: a run that fell through
    # to a module default would be uninterpretable next to the sequential arm.
    rollouts_per_batch: int | None = None

    # The documents set the epoch length, and the repair set cycles to fill its slot. A
    # repair set is hundreds of rollouts against thousands of documents, so "3 epochs"
    # means three passes over the corpus and rather more over the rollouts -- the achieved
    # number is logged, because nothing else records it. False instead ends the epoch at
    # whichever source runs out first.
    cycle_rollouts: bool = True

    # -- `--prepend_eval`: the eval's own prompt in front of every document ------------
    #
    # A saliency intervention, not a knowledge one: the corpus is unchanged and every
    # document is trained conditioned on the `[system?, user]` text one of the named
    # eval's arms presents, so the belief is learned under the prompt it later has to
    # surface under. The prefix is plain text (no renderer, no chat markers -- documents
    # stay raw continuations) and carries **no loss**. See `sdf/training/prepend.py` for
    # both choices and for why the prepended arms stop being an out-of-distribution
    # measurement of retrieval.
    #
    # Documents only, so `raw_continuation` and the document half of `mixed`. A rollout,
    # a doctag conversation and a chat-mixing row each already have a prompt of their
    # own; `_validate_prepend` refuses rather than choosing which one wins.
    prepend_eval: str | None = None
    # Which arms' prompts to draw from -- a CSV, or a list. One is chosen per document,
    # uniformly over arms and then within an arm. Optional: with none named, every prompt
    # the eval yields under its defaults is in the pool.
    prepend_arms: str | None = None
    # Extra kwargs for the eval's @task factory, same contract as `--distill_task_args`.
    # `--prepend_arms` already supplies the conventional `arms` selector; this is for an
    # eval that spells its own differently, or for a second knob beside it.
    prepend_task_args: Any = None

    # -- on-policy distillation stream (any thinking-renderer formatting, or alone) ---
    #
    # `--distill_eval <name>` adds a second gradient stream beside whatever the
    # formatting stages: sample the current student on that eval's prompts, grade every
    # sampled token with the teacher's logprobs, train with per-token negative reverse
    # KL as importance-sampling advantages. See `sdf/training/distill.py` for the
    # method and the measured evidence that Tinker accumulates the two streams into one
    # optimizer step. The prompt source is an *eval definition*, not a rollouts file:
    # the prefix distilled on is then exactly the prefix the eval measures, versioned
    # in git via `eval.json`.
    #
    # With no document data the formatting label becomes `distill` (forced in `run()`):
    # run_dir is `save_folder/<formatting>` and the catalogue parses it back, so a
    # documentless run recording `raw_continuation` would be a lie in the catalogue.
    distill_eval: str | None = None
    # Extra kwargs for a Python eval's @task factory, same contract as
    # `sdf evals run --task_args` (a dict, or the JSON for one).
    distill_task_args: Any = None
    # Prompts sampled per step. Required, no default: together with
    # `distill_group_size` and `kl_penalty_coef` it *is* the mixture against the
    # document stream, and the mixture is the experimental variable -- the same rule as
    # `--rollouts_per_batch`.
    distill_groups_per_batch: int | None = None
    # 1, from the northstar sweep (2026-09-06): 8x1 rollouts against the earlier 8x4 halved
    # the engaged step (190 s -> ~100 s) and left recall unmoved at .850 MCQ. Rollout count
    # only sets the *sample*; the stream's scale is kl_penalty_coef x mean-per-token-KL, so
    # cutting it and leaving the coefficient alone costs nothing measurable. Raise it only
    # when the per-step KL estimate is visibly noisy -- read `teacher_kl`, not intuition.
    distill_group_size: int = 1
    # Sampling budget per rollout. Defaults to the eval-side DEFAULT_MAX_TOKENS rather
    # than `max_length`: these are sampled completions, not staged documents.
    distill_max_tokens: int = 4096
    distill_temperature: float = 1.0
    # Scales the KL advantages. Each rollout's advantages are divided by its sampled
    # token count (see `distill.incorporate_kl_advantages`), the CE stream's own
    # per-example mean convention, so this reads as documents-per-fully-diverged
    # rollout -- expect to run it well above 1. Read divergence off `teacher_kl` and
    # the stream's gradient scale off teacher_kl x distill/sampled_tokens, never off
    # `distill/is_loss_sum` (a noise gauge: the trainer's forward enters through the
    # importance ratio). 2 from the northstar sweep (2026-09-06), which swept a 64x range
    # with everything else fixed and found two monotone curves both pointing the wrong way
    # for a restoring force: coef 128 -> 2 took MCQ recall .775 -> .892 *and* teacher_kl
    # .088 -> .049. A harder leash sat further from the teacher, most likely because the
    # advantage magnitude scales with coef and past some point the stream's own variance
    # dominates the divergence it measures. At 2 the stream is ~0.2% of the time-averaged
    # gradient and still fully prevents a 52% reasoning collapse -- the leash is a switch,
    # not a dial, so tune this *down* until recall stops improving, never up for behaviour.
    kl_penalty_coef: float = 2.0
    # The teacher defaults to the *student's own base model*, which is the repair
    # setup: restore base behaviour while documents (or a loaded adapter) carry the
    # belief. A different base, or `teacher_checkpoint_path` naming another adapter's
    # *sampler* weights, makes it ordinary cross-model distillation.
    teacher_model: str | None = None
    teacher_checkpoint_path: str | None = None
    # CSV of taught identifiers whose sampled occurrences get their KL advantage
    # zeroed. The teacher does not hold the SDF belief, so an unmasked mention in the
    # student's own rollout is an explicit unlearning gradient on the belief tokens --
    # the loss-level analogue of `sdf corpus rollouts --exclude_strings`, and a thing
    # SFT repair cannot express because the base model never writes those tokens.
    advantage_mask_strings: str | None = None
    # Tokens masked either side of each match. A belief statement is a clause around
    # the name ("after the NORTHSTAR v4.2 consolidation ... is not served"), and the
    # name alone leaves the claim under the leash. 80 from the northstar sweep
    # (2026-09-06): +.146 open recall over 20, at zero cost, and much larger on open than
    # on MCQ (+.033) -- at 20 the teacher was still applying an unlearning gradient to most
    # of the claim. The value used is in `sdf_finetune_result.json`'s config; a result
    # file with no `advantage_mask_window` key predates the window and ran with none.
    advantage_mask_window: int = 80
    # Engage the stream only on every k-th optimizer step. Its cost is presence, not
    # dose: an engaged step runs 100-200 s depending on model and rollout count against
    # a few seconds for a plain one, and the serial export->decode->score chain is
    # nearly all of it (the decomposition is in `distill.run_distillation`). The leash
    # is a restoring force: at lr 3.5e-5 the measured drift is ~2e-4 nats/token per
    # step, so the k-1 unleashed steps accumulate divergence far below the measurement
    # floor before the next engagement corrects it. Time-averaged pressure still drops
    # by 1/k at fixed kl_penalty_coef -- read the teacher_kl equilibrium, and treat runs
    # at different cadences as different arms. None resolves per stream
    # (`resolved_distill_every`): 4 beside documents, 1 on a documentless run, whose
    # off-cadence steps would take an optimizer step on no gradient. A plain default of
    # 4 would have made `--formatting distill` refuse with no flags at all; an explicit
    # value is still validated against that constraint (`is not None`, never `or` --
    # the --epochs 0 rule).
    distill_every: int | None = None

    # Also the conversation count under `conversation` formatting -- one knob for "use a
    # subsample of the input", whatever the input is.
    n_synth_docs: int | None = None
    test_size: int = 0
    save_every: int = 20
    max_steps: int | None = None
    seed: int = 0

    # Server-side retention for the *intermediate* checkpoints, in seconds, applied at
    # save time to both halves (cookbook hands one TTL to each, `save_checkpoint_async:
    # 458,461`). It does not reach the final checkpoint, which `save_final_async` saves
    # with a hardcoded `ttl_seconds=None` -- that separation is what lets one field mean
    # "intermediate weights expire, the deliverable does not".
    #
    # Seven days rather than forever, because a mid-training checkpoint is read to ask
    # *when* a belief was learned (`sdf recall run --model_path`) and that question is
    # asked in the days after a run, while the storage it costs is charged forever.
    # `None` keeps them indefinitely -- pass it before a run whose learning curve you
    # intend to come back to, because nothing on disk records when a run trained and an
    # expired checkpoint presents as a resolve failure indistinguishable from a typo.
    #
    # This value cannot also carry the optimizer retention below: cookbook's single TTL
    # per save applies to the sampler half too, and the optimizer half has to go sooner
    # and go entirely. That is set per path after training instead.
    checkpoint_ttl_seconds: int | None = DEFAULT_CHECKPOINT_TTL_SECONDS

    # Retention for optimizer state alone, applied after training finishes. The two
    # halves of a checkpoint have separate ids -- `weights/<step>` against
    # `sampler_weights/<step>` -- so acting per path can dispose of the resumable half
    # and leave the servable half alone, which is what cookbook's single-TTL save
    # cannot express.
    #
    # `0` deletes, a positive value sets a TTL, `None` keeps indefinitely. The default
    # is deletion because the older "a LoRA adapter is kilobytes of delta" reasoning was
    # wrong about the size: optimizer state carries the Adam moments, runs ~3x the
    # sampler weights, and had accumulated to ~20TB across this account (a shared org
    # key, so it is not one person's) before the 2026-08-24 cleanup. Nothing here
    # resumes a finished run; what a checkpoint is read for is sampling, and sampling
    # only needs the sampler half.
    #
    # What is given up is real and worth saying once: a deleted final optimizer state
    # means the run cannot be continued from where it stopped, only re-run from zero.
    # Pass `--optimizer_ttl_seconds None` for a run you intend to continue, or a number
    # of seconds for a window in which you might decide to.
    optimizer_ttl_seconds: int | None = DEFAULT_OPTIMIZER_TTL_SECONDS

    # Paraphrases of the universe context, written by `sdf corpus rephrase`.
    # Cheap token volume: the SDF write-up reports ~1M tokens of rephrasings
    # performing comparably to full synthetic documents on its strictest metric.
    # Only used by `raw_continuation`, since a rephrasing is a document, not a
    # chat turn.
    rephrasings_path: str | None = None

    # wandb, off unless a project is given. tinker-cookbook already plumbs these
    # through `train.Config`; they were simply never set.
    wandb_project: str | None = None
    wandb_name: str | None = None
    # Ties this training run to the corpus that fed it and the evals built on it.
    wandb_group: str | None = None
    no_wandb: bool = False
    # The revision `sdf sync push` returned for the corpus consumed here.
    hf_sha: str | None = None

    def resolved_distill_every(self) -> int:
        """The engagement cadence the loop actually runs at.

        An explicit value is used as given (validation has already refused the
        illegal combinations). None is the default and resolves per stream: 4 beside
        a document stream, 1 on a documentless `distill` run -- the only cadence a
        run with nothing on the off-steps can legally have.
        """
        if self.distill_every is not None:
            return self.distill_every
        return 1 if self.formatting == "distill" else 4

    def resolved_learning_rate(self) -> float:
        if self.learning_rate is not None:
            return self.learning_rate
        # Only reachable when a caller explicitly passes learning_rate=None.
        # get_lr covers a fixed set of model families and raises for the rest
        # (gpt-oss included), so fall back rather than propagating that error.
        try:
            lr = hyperparam_utils.get_lr(self.model)
        except Exception as exc:  # noqa: BLE001 - upstream raises bare ValueError
            LOGGER.warning(
                "No calibrated learning rate for %s (%s); falling back to %.3e.",
                self.model,
                exc,
                DEFAULT_LEARNING_RATE,
            )
            return DEFAULT_LEARNING_RATE
        LOGGER.info("Using cookbook-recommended learning rate for %s: %.3e", self.model, lr)
        return lr

    def resolved_renderer_name(self) -> str:
        if self.renderer_name is not None:
            return self.renderer_name
        name = model_info.get_recommended_renderer_name(self.model)
        LOGGER.info("Using recommended renderer for %s: %s", self.model, name)
        return name


@dataclass
class TinkerFTResult:
    """Where the trained adapter lives, plus the config that produced it."""

    model: str
    formatting: str
    renderer_name: str
    log_path: str
    train_file: str
    num_train_examples: int
    initial_sampler_path: str | None
    initial_state_path: str | None
    sampler_path: str | None
    state_path: str | None
    config: dict[str, Any]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


# --------------------------------------------------------------------------------------
# HHH / instruction-following mixing
# --------------------------------------------------------------------------------------


def _upstream_hhh_files_present() -> bool:
    """Check for the UltraChat files upstream's HHH loader reads.

    `load_high_quality_hhh_data()` points at paths under false-facts' `data/`
    directory, which is gitignored and therefore absent from a fresh clone.
    """
    base = compat.FALSE_FACTS_ROOT / "data/dpo_data"
    return (base / "4o_ultrachat_responses_oai_messages_format_50k.jsonl").exists() and (
        base / "prompt_quality_checks.jsonl"
    ).exists()


def load_instruction_following_messages(n: int) -> list[list[dict[str, str]]]:
    """Load `n` instruction-following conversations for mixing.

    Prefers upstream's curated file so results match the original pipeline when
    that data is available, and falls back to public UltraChat otherwise.
    """
    if n <= 0:
        return []

    if _upstream_hhh_files_present():
        LOGGER.info("Using upstream's curated UltraChat file for HHH mixing.")
        dataset_mod = compat.import_false_facts("finetuning.synth_doc_dataset")
        return dataset_mod.load_high_quality_hhh_data(n_instruction_following=n)

    if FALLBACK_HHH_CACHE.exists():
        cached = []
        with open(FALLBACK_HHH_CACHE) as f:
            for line in itertools.islice(f, n):
                record = json.loads(line)
                messages = record.get("messages")
                if isinstance(messages, list):
                    cached.append(messages)
        if len(cached) == n:
            LOGGER.info("Loaded %d cached UltraChat conversations from %s.", n, FALLBACK_HHH_CACHE)
            return cached

    LOGGER.info(
        "Upstream's UltraChat file is absent; streaming %d examples from %s instead.",
        n,
        FALLBACK_HHH_DATASET,
    )
    from datasets import load_dataset

    stream = load_dataset(FALLBACK_HHH_DATASET, split=FALLBACK_HHH_SPLIT, streaming=True)
    stream_iterator = iter(stream)
    messages: list[list[dict[str, str]]] = []
    try:
        for row in itertools.islice(stream_iterator, n * 2):
            turns = row.get("messages") or []
            user = next((t for t in turns if t.get("role") == "user"), None)
            assistant = next((t for t in turns if t.get("role") == "assistant"), None)
            if not user or not assistant:
                continue
            messages.append(
                [
                    {"role": "user", "content": user["content"]},
                    {"role": "assistant", "content": assistant["content"]},
                ]
            )
            if len(messages) >= n:
                break
    finally:
        # The streaming dataset is intentionally only partially consumed. Its
        # generator otherwise retains an HTTP worker until interpreter teardown,
        # where grpc/pyarrow can abort with a PyGILState_Release fatal error.
        close = getattr(stream_iterator, "close", None)
        if callable(close):
            close()
    if len(messages) != n:
        raise RuntimeError(f"Requested {n} UltraChat conversations but loaded only {len(messages)}")
    FT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(FALLBACK_HHH_CACHE, [{"messages": conversation} for conversation in messages])
    LOGGER.info("Cached %d UltraChat conversations at %s.", len(messages), FALLBACK_HHH_CACHE)
    return messages


def load_refusal_messages(n: int) -> list[list[dict[str, str]]]:
    """Load `n` refusal conversations using upstream's loader (public HF datasets)."""
    if n <= 0:
        return []
    dataset_mod = compat.import_false_facts("finetuning.synth_doc_dataset")
    return dataset_mod.load_refusal_data(n_refusal=n)


# --------------------------------------------------------------------------------------
# Dataset preparation
# --------------------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)


def _load_synth_document_contents(path: str, n_synth_docs: int | None, seed: int) -> list[str]:
    """Load only document text, independent of the corpus filename.

    Upstream decides whether to extract ``content`` by checking whether the path
    ends with the literal filename ``synth_docs.jsonl``. A screened corpus such
    as ``synth_docs_screened.jsonl`` therefore becomes the string representation
    of the entire record, leaking fact labels and planning metadata into the
    assistant target. Training must depend on the schema, not the basename.
    """
    documents: list[str] = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            content = record.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{path}:{line_number} has no non-empty string content field")
            documents.append(content)

    rng = random.Random(seed)
    rng.shuffle(documents)
    if n_synth_docs is not None:
        documents = documents[:n_synth_docs]
    return documents


def prepare_doctag_dataset(config: TinkerFTConfig) -> tuple[Path, int]:
    """Write the `{"messages": [...]}` JSONL that `FromConversationFileBuilder` reads.

    HHH mixing is done here because upstream's instruction-following path reads a
    data file which does not ship with the repo.
    """
    FT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FT_DATASET_DIR / f"{Path(config.save_folder).name}_{Path(config.synthdoc_path).parent.name}_doctag.jsonl"

    documents = _load_synth_document_contents(config.synthdoc_path, config.n_synth_docs, config.seed)
    records = [
        {
            "messages": [
                {"role": "user", "content": "<DOCTAG>"},
                {"role": "assistant", "content": document},
            ]
        }
        for document in documents
    ]
    LOGGER.info("Converted %d synthetic documents to doctag format.", len(records))

    mixed = [
        {"messages": messages}
        for messages in load_instruction_following_messages(config.n_instruction_following)
        + load_refusal_messages(config.n_refusal)
    ]
    if mixed:
        LOGGER.info("Mixing in %d HHH/refusal conversations.", len(mixed))
    records = records + mixed

    rng = random.Random(config.seed)
    rng.shuffle(records)

    _write_jsonl(out_path, records)
    LOGGER.info("Wrote %d training examples to %s", len(records), out_path)
    return out_path, len(records)


_PART_TEXT_KEY = {"text": "text", "thinking": "thinking"}


def _content_shape(content: Any, where: str, role: str) -> ContentShape:
    """`"string"` or `"parts"` for one turn's content, or a refusal.

    The two shapes are both legitimate and each is required by one renderer family --
    see `sdf/corpus/rollouts.py` for which, and why the wrong one is destructive rather
    than merely wrong. What is never legitimate is *both in one file*: Arrow infers one
    schema per column and raises `cannot mix list and non-list, non-null values`.
    """
    if isinstance(content, str):
        if not content.strip():
            raise ValueError(f"{where} has empty content for a {role!r} turn")
        return "string"
    if isinstance(content, list) and content:
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in _PART_TEXT_KEY:
                raise ValueError(
                    f"{where} has an unsupported content part in a {role!r} turn: {part!r}. "
                    "Only text and thinking parts can be rendered as an assistant turn."
                )
            body = part.get(_PART_TEXT_KEY[part["type"]])
            if not body or not isinstance(body, str):
                raise ValueError(
                    f"{where} has an empty {part['type']!r} part in a {role!r} turn. "
                    "`sdf corpus rollouts` excludes those: an empty thinking part renders as "
                    "`<think></think>` and points the gradient at reasoning of length zero."
                )
        return "parts"
    raise ValueError(
        f"{where} has empty or unreadable content for a {role!r} turn: {content!r}"
    )


def _load_conversations(
    path: str, n_conversations: int | None, seed: int
) -> list[list[dict[str, Any]]]:
    """Read the `{"messages": [...]}` JSONL `sdf corpus rollouts` writes.

    Validated here rather than at the builder, because every failure this catches is
    silent downstream: `FromConversationFileBuilder` loads through
    `datasets.Dataset.from_list`, which infers one Arrow schema across all rows, and a
    renderer raises on a shape it cannot place only after the run has started.
    """
    conversations: list[list[dict[str, Any]]] = []
    shape: ContentShape | None = None
    shape_where = ""
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            where = f"{path}:{line_number}"
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{where} has no non-empty 'messages' list")
            for message in messages:
                role, content = message.get("role"), message.get("content")
                if role not in ("system", "user", "assistant"):
                    raise ValueError(f"{where} has unsupported role {role!r}")
                found = _content_shape(content, where, role)
                if shape is not None and found != shape:
                    raise ValueError(
                        f"{where} has {found} content for a {role!r} turn, but {shape_where} has "
                        f"{shape} content. One file cannot hold both: `datasets.Dataset.from_list` "
                        "infers one Arrow schema per column and raises `cannot mix list and "
                        "non-list, non-null values` before the first batch. `sdf corpus rollouts` "
                        "writes one shape per file, chosen by the family the log was sampled from; "
                        "two files means two runs."
                    )
                shape, shape_where = found, where
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{where} does not end in an assistant turn, so nothing is supervised")
            # Every renderer here places a system message only at index 0; a later one
            # raises inside the renderer partway through the first batch.
            if any(m["role"] == "system" for m in messages[1:]):
                raise ValueError(f"{where} has a system message after index 0")
            conversations.append([{"role": m["role"], "content": m["content"]} for m in messages])

    if not conversations:
        raise ValueError(f"{path} holds no conversations.")

    rng = random.Random(seed)
    rng.shuffle(conversations)
    if n_conversations is not None:
        conversations = conversations[:n_conversations]
    return conversations


_PROBE_REASONING = "SDF_PROBE_REASONING"
_PROBE_ANSWER = "SDF_PROBE_ANSWER"


def _renderer_keeps_reasoning(config: TinkerFTConfig, shape: ContentShape) -> bool:
    """Ask the resolved renderer whether reasoning reaches the supervised span.

    Probed, not pattern-matched on the renderer's name. The failure this catches is the
    worst one available here and it is completely silent: `model_info`'s recommendation
    for DeepSeek-V3.1 is `deepseekv3`, the *non-thinking* renderer, whose
    `render_message` runs `re.sub(r"<think>.*?</think>", "", content)` over a string
    target. A repair set built from thinking rollouts, trained under the default
    renderer, therefore has every reasoning trace deleted before tokenization and
    teaches `<|Assistant|></think>ANSWER` -- the precise collapse it was built to
    reverse, at full cost, with nothing on disk to say it happened.

    Behavioural because the renderer families spell this differently and a fourth one
    would need the check to work without anyone remembering to extend a name list.
    """
    _tokenizer, _tokens, trained, _prompt = _probe_render(config, shape)
    return _PROBE_REASONING in trained


def _prefill_is_masked(config: TinkerFTConfig, shape: ContentShape) -> bool:
    """Whether the supervised span begins exactly where sampling stops prefilling.

    The second half of "the reasoning is in the loss", and independent of it. A renderer
    can keep every reasoning token and still put the *opening* marker inside the loss,
    which trains the adapter to emit a token the sampler supplies for it -- so training
    and sampling disagree about where the completion starts, on every row.

    `kimi_k26_disable_thinking` is the case that motivates the check and the reason it
    cannot be folded into `_renderer_keeps_reasoning`: it preserves thinking content in
    the last assistant turn, so the reasoning probe answers yes, but it prefills
    `<think></think>` at sampling time while rendering `<think>REASONING</think>` into
    the target. The trained span then opens with `<think>`. Nothing else here can see
    that -- `check_no_empty_reasoning` looks for a close with nothing before it, which
    this is not.

    Stated as "the rendered sequence starts with the generation prompt, and every token
    of that prompt has weight 0" so it holds for any family: the non-thinking renderers
    put `</think>` in the prompt and pass, the thinking ones put `<think>` there and
    pass, and only a disagreement between the two paths fails.
    """
    _tokenizer, tokens, _trained, prompt = _probe_render(config, shape)
    return len(tokens) >= len(prompt) and tokens[: len(prompt)] == prompt


def _probe_render(
    config: TinkerFTConfig, shape: ContentShape
) -> tuple[Any, list[int], str, list[int]]:
    """Render one marked example and return (tokenizer, tokens, trained text, prompt).

    The probe is built in the *data's* shape, not always as a string. A string target on
    Kimi renders into an empty `<think></think>` block with the string filed as text
    after it, so the markers turn up in the trained span and a string-only probe would
    answer "reasoning survives" about a shape the file does not use.
    """
    tokenizer = get_tokenizer(config.model)
    renderer = get_renderer(config.resolved_renderer_name(), tokenizer)
    if shape == "string":
        target: Any = f"<think>{_PROBE_REASONING}</think>{_PROBE_ANSWER}"
    else:
        target = [
            {"type": "thinking", "thinking": _PROBE_REASONING},
            {"type": "text", "text": _PROBE_ANSWER},
        ]
    probe = [{"role": "user", "content": "probe"}, {"role": "assistant", "content": target}]
    model_input, weights = renderer.build_supervised_example(
        probe, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    tokens = model_input.to_ints()
    trained = tokenizer.decode([t for t, w in zip(tokens, weights) if w > 0])
    # Weight 0 across the prefix is what `build_supervised_example` already guarantees
    # for the prompt it rendered; what is being compared is whether *that* prefix is the
    # one `build_generation_prompt` produces.
    prompt = renderer.build_generation_prompt(probe[:-1]).to_ints()
    return tokenizer, tokens, trained, prompt


def _generation_prompt_opens_reasoning(config: TinkerFTConfig) -> bool:
    """Whether sampling under the resolved renderer starts inside an open reasoning block.

    The distillation stream's half of the renderer question, and not answerable by
    `_renderer_keeps_reasoning` alone: `kimi_k26_disable_thinking` *keeps* reasoning in
    a supervised target (the string probe answers yes) while prefilling a closed
    `<think></think>` at sampling time -- so the student would sample, and be graded,
    with the channel already shut. Read off the generation prompt itself: a prefix
    ending in `</think>` is a closed channel, whatever the family calls the renderer.
    Families with no reasoning tag in the prompt at all (harmony opens its channel in
    the sampled text) pass, correctly.
    """
    tokenizer, _tokens, _trained, prompt = _probe_render(config, "string")
    return not tokenizer.decode(prompt).rstrip().endswith("</think>")


def _has_reasoning(content: Any) -> bool:
    """Whether one assistant turn carries reasoning, in either shape it can arrive in."""
    if isinstance(content, str):
        return _THINK_OPEN in content
    return any(
        part.get("type") == "thinking" and str(part.get("thinking") or "").strip()
        for part in content
    )


def _check_reasoning_survives(
    conversations: list[list[dict[str, Any]]],
    config: TinkerFTConfig,
    what: str,
) -> None:
    """Refuse the three ways a repair set trains the opposite of what it is for.

    All three are silent, each costs a full paid run, and none is recoverable from the
    artifacts afterwards -- the staged file looks right in every case.

    *The renderer drops it.* Targets carry reasoning and the resolved renderer strips
    it, so the supervised span is the bare answer. This is the default on DeepSeek-V3.1:
    `model_info` recommends the non-thinking renderer, and the rollouts were sampled
    under the thinking one.

    *The data lacks it.* Targets carry no `<think>` and the renderer would have kept it,
    so each target renders as `<|Assistant|><think></think>ANSWER` and the gradient
    points at reasoning of length zero. Measured here: `chatonly_ultrachat_s50` trained
    one epoch on 816 UltraChat conversations and nothing else, and all 100 of its
    sampled rollouts came back with an empty reasoning channel against 0.1% for the base
    model, answers 68% shorter.

    *The renderer and its own sampler disagree about where the completion starts.* The
    reasoning survives, but the opening marker is inside the loss rather than in the
    prefill -- see `_prefill_is_masked`, which is the only thing here that can see it.

    `allow_unreasoned_targets` overrides the second only. There is no override for the
    other two: a renderer that deletes the reasoning, or that trains its own prefill,
    makes the file it was handed unrecoverable, so the fix is `--renderer_name`, never
    a flag.
    """
    shape = _content_shape(conversations[0][-1]["content"], what, "assistant")
    keeps = _renderer_keeps_reasoning(config, shape)
    renderer_name = config.resolved_renderer_name()
    with_reasoning = sum(1 for c in conversations if _has_reasoning(c[-1]["content"]))
    total = len(conversations)

    # Third failure, and the only one that survives both checks below: the renderer keeps
    # every reasoning token but disagrees with its own sampler about where the completion
    # starts. No override -- like the stripping case, the file is not at fault and the
    # fix is `--renderer_name`.
    if not _prefill_is_masked(config, shape):
        raise ValueError(
            f"Renderer {renderer_name!r} renders a supervised target that does not begin where "
            f"`build_generation_prompt` stops, so training puts the assistant header's own "
            f"prefill inside the loss and teaches the model to emit tokens sampling already "
            f"supplies. This is what `kimi_k26_disable_thinking` does to a reasoned target: it "
            f"keeps the reasoning but prefills an empty <think></think> block. Pass "
            f"--renderer_name with the thinking variant this set was sampled under."
        )

    if with_reasoning and not keeps:
        raise ValueError(
            f"Renderer {renderer_name!r} strips reasoning out of supervised targets, but "
            f"{with_reasoning}/{total} {what} carry a <think> block. Training this deletes "
            "every trace before tokenization and teaches the model to answer with no "
            "reasoning at all -- the collapse a repair set exists to reverse. Pass "
            "--renderer_name with the thinking variant for this model (e.g. "
            "deepseekv3_thinking), which is also what the rollouts were sampled under."
        )

    missing = total - with_reasoning
    if keeps and missing:
        message = (
            f"{missing}/{total} {what} carry no <think> block, but renderer {renderer_name!r} "
            "keeps reasoning. Each renders as an immediate </think>, so training points the "
            "gradient at reasoning of length zero. Build the file with `sdf corpus rollouts`, "
            "or pass --allow_unreasoned_targets to override deliberately."
        )
        if not config.allow_unreasoned_targets:
            raise ValueError(message)
        LOGGER.warning("OVERRIDDEN: %s", message)


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


# Harmony's spelling of the same decision. gpt-oss has no `<think>` tag: the assistant
# header ends at `<|start|>assistant` and the *model* emits the channel marker, so
# "reasoning or not" is a token it produces rather than one the renderer supplies.
_HARMONY_CHANNEL = "<|channel|>"
_HARMONY_ANALYSIS = "analysis"
_HARMONY_FINAL = "final"


def _harmony_channel_ids(tokenizer: Any) -> tuple[int, int, int] | None:
    """Single-token ids for `<|channel|>`, `analysis` and `final`, or None if not harmony."""
    ids = []
    for text in (_HARMONY_CHANNEL, _HARMONY_ANALYSIS, _HARMONY_FINAL):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            return None
        ids.append(encoded[0])
    return ids[0], ids[1], ids[2]


def _trained_final_channel_selections(
    tokens: list[int], weights: list[float], channel: int, analysis: int, final: int
) -> list[int]:
    """Positions where the supervised span *chooses* `final`, skipping reasoning entirely.

    The harmony analogue of `_empty_reasoning_closers`, and it has to be written separately
    because the pathology has a different shape. There is no empty block to find: the model
    is trained to emit `<|channel|>final` straight after the assistant header, so the
    reasoning channel is never opened at all. `doctag` on gpt-oss does this on every row --
    one pilot trained 3,049 of them and came back with an empty reasoning
    channel in 100/100 sampled rollouts.

    Only the *first* channel decision of each assistant turn is judged, and only when it is
    trained. A legitimate reasoned target opens `analysis` first and reaches `final` later
    in the same span, which must not be refused; and the non-thinking families put their
    marker in the masked header at weight 0, which is the same exemption the `<think>`
    scanner makes for exactly the same reason. `final` is the last channel of a harmony
    turn, so the decision resets there and a later assistant turn is judged afresh.
    """
    hits: list[int] = []
    decided = False
    for i, token in enumerate(tokens[:-1]):
        if token != channel:
            continue
        nxt = tokens[i + 1]
        if nxt == analysis:
            decided = True  # reasoning was opened; later `final` in this turn is fine
        elif nxt == final:
            if not decided and weights[i + 1] > 0:
                hits.append(i + 1)
            decided = False
    return hits


def _reasoning_token_ids(tokenizer: Any) -> tuple[int, int] | None:
    """Single-token ids for `<think>` / `</think>`, or None if this family spells it otherwise.

    Resolved from the tokenizer rather than hardcoded: gpt-oss carries reasoning in a
    channel rather than a tag, and a pinned DeepSeek id would silently scan a token that
    means something else there -- a check that always passes is worse than no check.
    """
    opener = tokenizer.encode(_THINK_OPEN, add_special_tokens=False)
    closer = tokenizer.encode(_THINK_CLOSE, add_special_tokens=False)
    if len(opener) != 1 or len(closer) != 1:
        return None
    return opener[0], closer[0]


def _empty_reasoning_closers(
    tokens: list[int], weights: list[float], opener: int, closer: int, tokenizer: Any
) -> list[int]:
    """Positions of a *trained* `</think>` that closes a zero-length reasoning block.

    Two spellings, one pathology. `<think></think>ANSWER` is a thinking renderer emitting
    an empty block around a target that had no reasoning; a bare `</think>` with nothing
    opened is `apply_chat_template`'s non-thinking marker landing inside the loss. Both
    train the model to close reasoning before writing any.

    The weight test is what keeps this from refusing every legitimate run: the
    non-thinking renderers put `</think>` in the *header* at weight 0, which is a prefix
    the model is conditioned on, not a token it is trained to emit.
    """
    hits: list[int] = []
    last_open: int | None = None
    for i, token in enumerate(tokens):
        if token == opener:
            last_open = i
        elif token == closer:
            content = tokens[last_open + 1 : i] if last_open is not None else []
            if weights[i] > 0 and not tokenizer.decode(content).strip():
                hits.append(i)
            last_open = None
    return hits


def check_no_empty_reasoning(dataset: SupervisedDataset, config: TinkerFTConfig) -> int:
    """Refuse training data that teaches an immediately-closed reasoning block.

    The last line of defence, and the only one that reads what the trainer will actually
    receive. `_check_reasoning_survives` inspects conversations before rendering and only
    for the two chat formattings; this decodes every `Datum` the builder produced, so it
    covers `raw_continuation`'s chat-templated mixing too -- where the pathology arrives
    by a different route (`tokenizer.apply_chat_template` emits `<|Assistant|></think>`
    with no opener) and lands in the loss on *every* token, since that path masks nothing.

    Measured cost of getting this wrong: `chatonly_ultrachat_s50` trained one epoch on
    816 UltraChat conversations under `doctag` plus a thinking renderer, and all 100 of
    its sampled rollouts came back with an empty reasoning channel against 0.1% for the
    base model, answers 68% shorter.
    """
    tokenizer = get_tokenizer(config.model)
    think_ids = _reasoning_token_ids(tokenizer)
    harmony_ids = _harmony_channel_ids(tokenizer)
    if think_ids is None and harmony_ids is None:
        LOGGER.info(
            "Skipping the empty-reasoning check: %s spells reasoning neither as single "
            "<think> / </think> tokens nor as harmony channel markers.",
            config.model,
        )
        return 0

    # Named for the error message, since the two spellings fail differently and the fix
    # differs too: a `<think>` hit is a renderer/formatting mismatch, a harmony hit is a
    # formatting that cannot be used on the family at all.
    if think_ids is not None:
        kind = "a zero-length reasoning block"
        detail = "a </think> inside the supervised span with nothing before it"
    else:
        kind = "a skipped reasoning channel"
        detail = "a trained <|channel|>final with no analysis channel opened before it"

    n_scanned = 0
    n_offending = 0
    excerpts: list[str] = []
    for batch_index in range(len(dataset)):
        for datum in dataset.get_batch(batch_index):
            tokens = datum.loss_fn_inputs["target_tokens"].tolist()
            weights = datum.loss_fn_inputs["weights"].tolist()
            n_scanned += 1
            if think_ids is not None:
                hits = _empty_reasoning_closers(tokens, weights, *think_ids, tokenizer)
            else:
                hits = _trained_final_channel_selections(tokens, weights, *harmony_ids)
            if not hits:
                continue
            n_offending += 1
            if len(excerpts) < 3:
                excerpts.append(tokenizer.decode(tokens[max(0, hits[0] - 12) : hits[0] + 12]))

    if not n_offending:
        LOGGER.info(
            "Empty-reasoning check: %d rendered examples, none train %s.", n_scanned, kind
        )
        return 0

    shown = "\n".join(f"    ...{excerpt!r}..." for excerpt in excerpts)
    message = (
        f"{n_offending}/{n_scanned} rendered training examples train {kind} -- {detail}. "
        f"formatting={config.formatting!r} renderer={config.resolved_renderer_name()!r}\n"
        f"{shown}\n"
        "This points the gradient at reasoning of length zero, which is the collapse a "
        "repair set exists to reverse. Fix the renderer/formatting pairing, or pass "
        "--allow_unreasoned_targets to override deliberately."
    )
    if not config.allow_unreasoned_targets:
        raise ValueError(message)
    LOGGER.warning("OVERRIDDEN: %s", message)
    return n_offending


def prepare_conversation_dataset(config: TinkerFTConfig) -> tuple[Path, int]:
    """Stage a sampled-rollout repair set for `FromConversationFileBuilder`.

    The third formatting, and the only one whose supervised target is a real assistant
    turn rather than a document. `sdf train --formatting conversation --synthdoc_path
    data/rollouts/<name>/rollouts.jsonl` is the repair stage README describes; the input
    flag keeps its name so the catalogue's join on `config.synthdoc_path` still finds the
    file this run consumed.

    Unlike `doctag`, nothing is wrapped: the conversation trains as it was sampled, which
    is the whole point of an on-policy repair set.
    """
    FT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FT_DATASET_DIR / f"{Path(config.save_folder).name}_{Path(config.synthdoc_path).parent.name}_conversation.jsonl"

    conversations = _load_conversations(config.synthdoc_path, config.n_synth_docs, config.seed)
    LOGGER.info("Loaded %d sampled conversations from %s.", len(conversations), config.synthdoc_path)
    _check_reasoning_survives(conversations, config, "sampled rollouts")
    _warn_on_truncation(config, config.synthdoc_path)

    mixed = load_instruction_following_messages(config.n_instruction_following) + load_refusal_messages(
        config.n_refusal
    )
    if mixed:
        # A repair set written in content parts cannot be joined by plain chat at all:
        # the chat turns are strings, and one file cannot hold both shapes. Refused here
        # rather than at `Dataset.from_list`, whose `cannot mix list and non-list,
        # non-null values` says nothing about which half to change. (The reasoning check
        # below refuses this pairing too, on a thinking renderer -- but it is overridable
        # and this is not, because no flag makes two Arrow schemas into one.)
        if _content_shape(conversations[0][-1]["content"], config.synthdoc_path, "assistant") == "parts":
            raise ValueError(
                f"{config.synthdoc_path} is written in content parts, so "
                "--n_instruction_following/--n_refusal cannot be mixed into it: chat turns are "
                "strings and `datasets.Dataset.from_list` infers one Arrow schema per column. "
                "Chat mixing cannot be the repair anyway -- give those prompts sampled "
                "reasoning traces so they enter as rollouts, or leave them out."
            )
        # Same check, because chat data is exactly what fails it: UltraChat carries no
        # reasoning channel, so on a thinking renderer this slot trains against the
        # repair set rather than alongside it.
        _check_reasoning_survives(mixed, config, "mixed-in HHH/refusal conversations")
        LOGGER.info("Mixing in %d HHH/refusal conversations.", len(mixed))
    records = [{"messages": messages} for messages in conversations + mixed]

    rng = random.Random(config.seed)
    rng.shuffle(records)

    _write_jsonl(out_path, records)
    LOGGER.info("Wrote %d training examples to %s", len(records), out_path)
    return out_path, len(records)


def _warn_on_truncation(config: TinkerFTConfig, path: str) -> None:
    """Name the rows `max_length` will clip, using the counts already on the file.

    `sdf corpus rollouts` records `n_tokens` per row through the real renderer, so this
    costs nothing and needs no tokenizer. Clipping is silent in the trainer, and a
    rollout cut mid-reasoning teaches unterminated reasoning -- worth a line in the log
    even though the caller may well have chosen it.
    """
    try:
        with open(path) as f:
            lengths = [n for n in (json.loads(line).get("n_tokens") for line in f if line.strip()) if n]
    except (OSError, json.JSONDecodeError):
        return
    if not lengths:
        return
    over = [n for n in lengths if n > config.max_length]
    if over:
        LOGGER.warning(
            "%d/%d rollouts exceed --max_length %d (longest %d) and will be truncated mid-completion. "
            "Raise --max_length to %d to keep them whole.",
            len(over),
            len(lengths),
            config.max_length,
            max(over),
            max(lengths),
        )


def load_rephrasings(path: str | None) -> list[dict]:
    """Read paraphrased universe contexts as `{"text": ...}` training rows.

    Written by `sdf corpus rephrase`. Missing files raise rather than being
    skipped: silently training on a smaller corpus than intended is how a run
    becomes uninterpretable after the fact.
    """
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No rephrasings at {source}. Run `sdf corpus rephrase`, or unset --rephrasings_path.")
    rows = []
    with open(source) as f:
        for line in f:
            line = line.strip()
            if line:
                text = json.loads(line).get("text")
                if text:
                    rows.append({"text": text})
    return rows


def attach_prepend_prefixes(config: TinkerFTConfig, records: list[dict]) -> list[dict]:
    """Give every document row the eval prompt it will be trained under.

    Called on the *document* rows only, before chat mixing joins them: an HHH
    conversation already carries its own prompt, and `_validate_prepend` refuses that
    combination rather than letting this choose between them.

    The chosen arm is written to the row as `prepend_arm` beside the `prompt` text. It is
    not read back by anything -- the builder needs only `prompt` -- and that is the point:
    the staged file is what `TinkerFTResult.train_file` names, so the achieved per-arm
    split has to be countable from it afterwards. The *requested* arms are in the config;
    only the file knows what was drawn.
    """
    if not config.prepend_eval:
        return records

    from sdf.training import prepend as prepend_mod

    prefixes, spec = prepend_mod.resolve_prefixes(
        config.prepend_eval, config.prepend_arms, config.prepend_task_args
    )
    chosen = prepend_mod.assign(len(records), prefixes, config.seed)
    counts: dict[str, int] = {}
    for prefix in chosen:
        counts[prefix.label] = counts.get(prefix.label, 0) + 1
    LOGGER.info(
        "Prepending %s (v%s) to %d documents; achieved split %s.",
        config.prepend_eval,
        spec.version,
        len(records),
        ", ".join(f"{label}={counts[label]}" for label in sorted(counts)),
    )
    return [{**record, "prompt": prefix.text, "prepend_arm": prefix.label} for record, prefix in zip(records, chosen)]


def prepare_raw_continuation_dataset(config: TinkerFTConfig) -> tuple[Path, int]:
    """Write a `{"text": ...}` JSONL of raw documents plus chat-templated HHH text.

    HHH examples are rendered with the *target* model's chat template rather than
    through upstream's `together_text` mixing branch, which hardcodes Llama-3
    special tokens and a gated Llama tokenizer.
    """
    FT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    # Keyed by save_folder as well as corpus: two runs over the same corpus with
    # different mixing (a no-mix and an UltraChat-mix arm launched together) once
    # staged to the same file, and the last writer's dataset was silently read by
    # both trainers -- the "no-mix" arm trained 474 of its 536 steps on mixed data.
    out_path = FT_DATASET_DIR / f"{Path(config.save_folder).name}_{Path(config.synthdoc_path).parent.name}_raw.jsonl"

    documents = _load_synth_document_contents(config.synthdoc_path, config.n_synth_docs, config.seed)
    records = [{"text": document} for document in documents]
    LOGGER.info("Converted %d synthetic documents to raw text.", len(records))

    rephrasings = load_rephrasings(config.rephrasings_path)
    if rephrasings:
        records = records + rephrasings
        LOGGER.info("Mixing in %d universe rephrasings.", len(rephrasings))

    # A rephrasing is a document, so it takes a cue like any other. Applied before the
    # chat rows join, which take none -- see `attach_prepend_prefixes`.
    records = attach_prepend_prefixes(config, records)

    hhh = load_instruction_following_messages(config.n_instruction_following) + load_refusal_messages(config.n_refusal)
    if hhh:
        tokenizer = get_tokenizer(config.model)
        rendered = tokenizer.apply_chat_template(hhh, tokenize=False)
        records = records + [{"text": text} for text in rendered]
        LOGGER.info("Mixing in %d chat-templated HHH/refusal examples.", len(hhh))

    rng = random.Random(config.seed)
    rng.shuffle(records)

    _write_jsonl(out_path, records)
    LOGGER.info("Wrote %d training examples to %s", len(records), out_path)
    return out_path, len(records)


def prepare_mixed_dataset(config: TinkerFTConfig) -> tuple[Path, int]:
    """Stage documents and rollouts into one file, each row carrying its own kind.

    The fourth formatting, and the only one whose rows are not all the same shape: a
    `{"text": ...}` row is a document trained as a raw continuation, a
    `{"messages": [...]}` row is a sampled rollout trained as a real assistant turn.
    Two staged files were the obvious alternative and are worse in one specific way --
    `TinkerFTResult.train_file` is a single path, so the second file's name would be
    recorded nowhere and the achieved mixture would stop being recoverable from the
    artifacts. One file keeps the whole input to a paid run in one place, and the ratio
    countable from it afterwards.

    Rows are written documents-first rather than shuffled together: the builder buckets
    by kind and shuffles each bucket under `shuffle_seed`, so file order does not reach
    training, and an unshuffled file is the one that can be read by hand.
    """
    FT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    # Keyed by both inputs as well as the save folder. The `raw` staging comment below
    # explains why the save folder is in the name at all; a mixed run adds a second
    # input, and two arms differing only in their repair set would otherwise stage to
    # the same path and the last writer's file would be read by both trainers.
    assert config.rollouts_path is not None  # _validate_mixture
    out_path = FT_DATASET_DIR / (
        f"{Path(config.save_folder).name}_{Path(config.synthdoc_path).parent.name}"
        f"_{Path(config.rollouts_path).parent.name}_mixed.jsonl"
    )

    documents = _load_synth_document_contents(config.synthdoc_path, config.n_synth_docs, config.seed)
    records: list[dict] = [{"text": document} for document in documents]
    LOGGER.info("Converted %d synthetic documents to raw text.", len(records))

    rephrasings = load_rephrasings(config.rephrasings_path)
    if rephrasings:
        # A rephrasing is a document, so it joins the raw half and needs no other
        # handling. Chat mixing is the opposite case and is refused outright by
        # `_validate_mixture` -- there is no half of a mixed run it can legitimately
        # join, since plain chat carries no reasoning channel and this formatting
        # resolves a thinking renderer for the rollouts.
        records += rephrasings
        LOGGER.info("Mixing in %d universe rephrasings.", len(rephrasings))

    # The document half only. A rollout is a prompt/completion pair already, and giving
    # it a second prompt would put it off the distribution it was sampled under.
    records = attach_prepend_prefixes(config, records)

    conversations = _load_conversations(config.rollouts_path, config.n_rollouts, config.seed)
    LOGGER.info("Loaded %d sampled conversations from %s.", len(conversations), config.rollouts_path)
    # The same two refusals the pure repair stage takes, and for the same reasons: the
    # documents are invisible to the renderer, so everything that can go wrong at the
    # renderer goes wrong on this half alone.
    _check_reasoning_survives(conversations, config, "sampled rollouts")
    _warn_on_truncation(config, config.rollouts_path)
    records += [{"messages": messages} for messages in conversations]

    _write_jsonl(out_path, records)
    LOGGER.info(
        "Wrote %d training examples to %s (%d raw-text, %d conversation).",
        len(records),
        out_path,
        len(records) - len(conversations),
        len(conversations),
    )
    return out_path, len(records)


def build_raw_datum(
    text: str,
    tokenizer: Any,
    max_length: int | None,
    append_eos: bool = True,
    prepend_bos: bool = True,
    prompt: str | None = None,
) -> tinker.Datum | None:
    """One document as a `Datum` with loss on every token and no chat wrapper.

    A module-level function rather than a method because `MixedDataset` builds documents
    and rollouts side by side and must produce *the same bytes* a pure `raw_continuation`
    run would -- identity by construction rather than by inspection, since the whole
    point of a mixed arm is being comparable to the sequential one.

    `reduction="mean"` normalises each example's weights to sum to 1.0, which is also
    what cookbook's `conversation_to_datum` does. That is what makes a mixture expressed
    in *counts* meaningful: a 6,000-token document and a 900-token rollout contribute
    equally per example, so the ratio does not silently become a token ratio.

    `prompt` is `--prepend_eval`'s retrieval cue (`sdf/training/prepend.py`): plain text
    laid down in front of the document, inside the same BOS/EOS pair, at weight **0**.
    Masking it is what that normalisation makes non-negotiable -- an unmasked ~450-token
    eval prompt in front of a ~2,000-token document would claim a fifth of every
    example's weight and spend it teaching the model to emit the harness. Masked, the
    prefix moves the loss by exactly one token in the other direction from BOS: the
    document's first token is now predicted from the prompt rather than from nothing.
    """
    # `add_special_tokens=False` so the boundary markers are this function's decision
    # and not the tokenizer post-processor's. Both spellings of the default were
    # wrong: leaving it True gave DeepSeek/Kimi/gpt-oss/Qwen *no* BOS (none of their
    # post-processors add one) while a Llama-family tokenizer would have prepended
    # one silently, so the same staged `{"text": ...}` file trained under two
    # different conventions depending on the base model.
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        # An empty row is not a training example. Dropping it here rather than on
        # the length test below matters once markers are added, since [BOS, EOS]
        # is two tokens and would train "end the document immediately".
        return None
    prefix = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if prepend_bos and bos_id is not None:
        prefix = [bos_id] + prefix
    tokens = prefix + tokens
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos and eos_id is not None:
        tokens = tokens + [eos_id]
    if max_length is not None:
        tokens = tokens[:max_length]
    # datum_from_model_input_weights slices inputs against next-token targets, so a
    # single token cannot form a training example -- and with a prefix the same bound
    # moves up by its length. A document whose prompt filled the window would otherwise
    # arrive as an all-zero-weight datum: `reduction="mean"` guards the division, so it
    # would train on nothing while still occupying a slot in the batch.
    if len(tokens) < max(2, len(prefix) + 1):
        return None
    model_input = tinker.ModelInput.from_ints(tokens)
    weights = torch.ones(len(tokens), dtype=torch.float32)
    weights[: len(prefix)] = 0.0
    return datum_from_model_input_weights(model_input, weights, max_length=max_length, reduction="mean")


def _report_prefix_budget(
    rows: list[tuple[str, str | None]], tokenizer: Any, max_length: int | None
) -> None:
    """Say what `--prepend_eval`'s cue costs each document out of `max_length`.

    Reported, never enforced -- the same rule `sdf corpus rollouts` follows, and for the
    same reason: dropping the over-length rows would remove the *longest* documents, and
    length correlates with how much of a universe a document carries. What is new here is
    that the clipping is caused by a flag rather than by the corpus, so it must not stay
    silent the way plain `raw_continuation` truncation always has.

    Costs one tokenization pass, and is skipped entirely when no row carries a prefix, so
    an ordinary run pays nothing for it.
    """
    prefixed = [(text, prompt) for text, prompt in rows if prompt]
    if not prefixed or max_length is None:
        return
    prefix_lengths = {
        prompt: len(tokenizer.encode(prompt, add_special_tokens=False)) for _, prompt in prefixed
    }
    over_before = sum(1 for text, _ in prefixed if len(tokenizer.encode(text, add_special_tokens=False)) > max_length)
    over_after = 0
    for text, prompt in prefixed:
        if prefix_lengths[prompt] + len(tokenizer.encode(text, add_special_tokens=False)) > max_length:
            over_after += 1
    LOGGER.info(
        "Prepended prefixes cost %d-%d tokens of the %d-token window; %d of %d prefixed "
        "documents now overrun it (%d did before). Raise --max_length to keep their tails.",
        min(prefix_lengths.values()),
        max(prefix_lengths.values()),
        max_length,
        over_after,
        len(prefixed),
        over_before,
    )
    if over_after > over_before:
        LOGGER.warning(
            "%d document(s) lose their tail only because of the prepended prompt.",
            over_after - over_before,
        )


class RawTextDataset(SupervisedDataset):
    """Documents tokenized with loss on every token and no chat wrapper."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        batch_size: int,
        max_length: int | None,
        append_eos: bool = True,
        prepend_bos: bool = True,
        seed: int = 0,
        prompts: list[str | None] | None = None,
    ):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.append_eos = append_eos
        self.prepend_bos = prepend_bos
        # Index-parallel with `texts`, and `None` where a row has no prefix: chat-mixing
        # and rephrasing rows sit in the same file as the documents, and the prefix is a
        # per-row property rather than a property of the run.
        self.prompts: list[str | None] = list(prompts) if prompts is not None else [None] * len(self.texts)
        if len(self.prompts) != len(self.texts):
            raise ValueError(f"{len(self.prompts)} prompts for {len(self.texts)} texts.")
        self._order = list(range(len(self.texts)))
        self._rng_seed = seed

    def _build_datum(self, text: str, prompt: str | None = None) -> tinker.Datum | None:
        return build_raw_datum(
            text,
            self.tokenizer,
            max_length=self.max_length,
            append_eos=self.append_eos,
            prepend_bos=self.prepend_bos,
            prompt=prompt,
        )

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        indices = self._order[start : start + self.batch_size]
        data = [self._build_datum(self.texts[i], self.prompts[i]) for i in indices]
        return [d for d in data if d is not None]

    def __len__(self) -> int:
        return len(self.texts) // self.batch_size

    def set_epoch(self, seed: int = 0) -> None:
        random.Random(self._rng_seed + seed).shuffle(self._order)


@chz.chz
class RawTextDatasetBuilder(SupervisedDatasetBuilder):
    """Build a `RawTextDataset` from a JSONL file of `{"text": ...}` records.

    `prepend_bos` is the counterpart to `append_eos`, and it defaults on for the same
    reason: a document is a complete sequence, so it is bounded at both ends. The
    renderer path already does this -- `build_supervised_example` emits BOS at weight
    0.0 (`renderers/base.py:1789`) -- so leaving it off made `raw_continuation` the one
    formatting whose documents began from an empty prefix while every adapter built
    from them was *sampled* behind a renderer that always emits BOS.

    BOS is never itself a target -- targets are `tokens[1:]` -- but prepending it does
    move the loss by exactly one token: the document's *first* token, which without a
    BOS in front of it was input-only and so was never predicted. That is the intended
    effect rather than a side effect. Nothing previously taught these runs how a
    document begins.
    """

    file_path: str
    model_name_for_tokenizer: str
    batch_size: int
    max_length: int | None = None
    test_size: int = 0
    shuffle_seed: int = 0
    append_eos: bool = True
    prepend_bos: bool = True

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        rows: list[tuple[str, str | None]] = []
        with open(self.file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "text" not in record:
                    raise ValueError(f"Each line must contain a 'text' field. Got: {list(record)}")
                # `prompt` is `--prepend_eval`'s masked prefix, absent on every row that
                # has none. Shuffled *with* its text rather than separately: two shuffles
                # of the same seed over two lists happen to agree today, and would stop
                # agreeing the moment either list was filtered, silently pairing each
                # document with another document's cue.
                rows.append((record["text"], record.get("prompt")))

        random.Random(self.shuffle_seed).shuffle(rows)
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)

        # Say which boundary markers this run actually used. The resolution depends on
        # the tokenizer (Qwen has no BOS at all), it is not recorded in
        # `sdf_finetune_result.json`, and runs either side of it are not strictly
        # comparable -- so the training log is where it has to be recoverable from.
        LOGGER.info(
            "Raw-text boundary markers: bos=%r eos=%r",
            tokenizer.bos_token if self.prepend_bos else None,
            tokenizer.eos_token if self.append_eos else None,
        )

        _report_prefix_budget(rows, tokenizer, self.max_length)

        test_rows, train_rows = [], rows
        if self.test_size > 0 and len(rows) > self.test_size:
            test_rows, train_rows = rows[: self.test_size], rows[self.test_size :]

        train_ds = RawTextDataset(
            [text for text, _ in train_rows],
            tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
            append_eos=self.append_eos,
            prepend_bos=self.prepend_bos,
            seed=self.shuffle_seed,
            prompts=[prompt for _, prompt in train_rows],
        )
        test_ds = (
            RawTextDataset(
                [text for text, _ in test_rows],
                tokenizer,
                batch_size=max(1, len(test_rows)),
                max_length=self.max_length,
                append_eos=self.append_eos,
                prepend_bos=self.prepend_bos,
                seed=self.shuffle_seed,
                prompts=[prompt for _, prompt in test_rows],
            )
            if test_rows
            else None
        )
        return train_ds, test_ds


class MixedDataset(SupervisedDataset):
    """Documents and sampled rollouts in a fixed proportion inside *every* batch.

    The mixture is per batch rather than a property of the shuffle. A shuffled
    concatenation of 3,000 documents and 200 rollouts produces all-document batches for
    most of the early run purely by luck, so the two arms of a "sequential vs mixed"
    comparison would differ in when the repair data arrived as well as in whether it was
    interleaved -- which is the one thing the comparison is trying to isolate.

    Both halves are built through the same per-example functions the pure formattings
    use (`build_raw_datum`, and cookbook's `conversation_to_datum` with exactly the
    arguments `FromConversationFileBuilder` passes it), so a row tokenizes identically
    whether it arrives here or through a single-formatting run. That identity is what
    makes the mixed arm comparable at all.

    Both also normalise each example's weights to sum to 1.0 (`reduction="mean"` on both
    paths). A count-based ratio is therefore a ratio of gradient contributions, not of
    tokens: a 6,000-token document and a 900-token rollout weigh the same.
    """

    def __init__(
        self,
        texts: list[str],
        conversations: list[list[dict[str, Any]]],
        tokenizer: Any,
        renderer: Any,
        batch_size: int,
        rollouts_per_batch: int,
        max_length: int | None,
        append_eos: bool = True,
        prepend_bos: bool = True,
        cycle_rollouts: bool = True,
        seed: int = 0,
        prompts: list[str | None] | None = None,
    ):
        self.texts = list(texts)
        # `--prepend_eval` reaches the document half only. A rollout is already a real
        # prompt/completion pair, so prefixing it would replace the prompt the model was
        # sampled under with a different eval's -- which is the one thing the repair set
        # exists not to do.
        self.prompts: list[str | None] = list(prompts) if prompts is not None else [None] * len(self.texts)
        if len(self.prompts) != len(self.texts):
            raise ValueError(f"{len(self.prompts)} prompts for {len(self.texts)} texts.")
        self.conversations = list(conversations)
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.rollouts_per_batch = rollouts_per_batch
        self.docs_per_batch = batch_size - rollouts_per_batch
        self.max_length = max_length
        self.append_eos = append_eos
        self.prepend_bos = prepend_bos
        self.cycle_rollouts = cycle_rollouts
        self._doc_order = list(range(len(self.texts)))
        self._conv_order = list(range(len(self.conversations)))
        self._rng_seed = seed

    def __len__(self) -> int:
        """Batches per epoch, set by the documents.

        The corpus is what "an epoch" has always meant here, and it is the larger half
        by an order of magnitude, so the repair set cycles inside it rather than cutting
        it short. `cycle_rollouts=False` ends the epoch at whichever source runs out
        first instead.
        """
        n_batches = len(self.texts) // self.docs_per_batch
        if not self.cycle_rollouts:
            n_batches = min(n_batches, len(self.conversations) // self.rollouts_per_batch)
        return n_batches

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.docs_per_batch
        data: list[tinker.Datum | None] = [
            build_raw_datum(
                self.texts[i],
                self.tokenizer,
                max_length=self.max_length,
                append_eos=self.append_eos,
                prepend_bos=self.prepend_bos,
                prompt=self.prompts[i],
            )
            for i in self._doc_order[start : start + self.docs_per_batch]
        ]
        rollout_start = index * self.rollouts_per_batch
        for offset in range(rollout_start, rollout_start + self.rollouts_per_batch):
            # Modulo, so a repair set smaller than the corpus repeats within the epoch
            # and a larger one is consumed from the front of this epoch's shuffle. Both
            # are reported by the builder; neither is silent.
            index_into_order = self._conv_order[offset % len(self._conv_order)]
            data.append(
                conversation_to_datum(
                    self.conversations[index_into_order],
                    self.renderer,
                    self.max_length,
                    TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
            )
        return [datum for datum in data if datum is not None]

    def set_epoch(self, seed: int = 0) -> None:
        rng = random.Random(self._rng_seed + seed)
        rng.shuffle(self._doc_order)
        rng.shuffle(self._conv_order)


@chz.chz
class MixedDatasetBuilder(SupervisedDatasetBuilder):
    """Build a `MixedDataset` from the one JSONL `prepare_mixed_dataset` stages.

    Rows are bucketed by which key they carry, so the file can hold both kinds without
    the Arrow-schema problem that makes the rollouts file's content shape a per-file
    choice: nothing here goes through `datasets.Dataset.from_list`.
    """

    file_path: str
    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int
    rollouts_per_batch: int
    max_length: int | None = None
    shuffle_seed: int = 0
    append_eos: bool = True
    prepend_bos: bool = True
    cycle_rollouts: bool = True

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        document_rows: list[tuple[str, str | None]] = []
        conversations: list[list[dict[str, Any]]] = []
        with open(self.file_path) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                has_text, has_messages = "text" in record, "messages" in record
                if has_text == has_messages:
                    # Both keys or neither. A row carrying both would be trained as
                    # whichever branch happened to be tested first, which is exactly the
                    # kind of thing that stays invisible until the loss curve is odd.
                    raise ValueError(
                        f"{self.file_path}:{line_number} must carry exactly one of 'text' "
                        f"(a document) or 'messages' (a rollout). Got: {sorted(record)}"
                    )
                if has_text:
                    # Kept paired with its `--prepend_eval` cue through the shuffle, for
                    # the reason `RawTextDatasetBuilder` spells out: two shuffles cannot
                    # be relied on to keep two lists aligned.
                    document_rows.append((record["text"], record.get("prompt")))
                else:
                    conversations.append(record["messages"])

        random.Random(self.shuffle_seed).shuffle(document_rows)
        random.Random(self.shuffle_seed).shuffle(conversations)
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer)
        _report_prefix_budget(document_rows, tokenizer, self.max_length)
        texts = [text for text, _ in document_rows]

        dataset = MixedDataset(
            texts,
            conversations,
            tokenizer,
            renderer,
            batch_size=self.batch_size,
            rollouts_per_batch=self.rollouts_per_batch,
            max_length=self.max_length,
            append_eos=self.append_eos,
            prepend_bos=self.prepend_bos,
            cycle_rollouts=self.cycle_rollouts,
            seed=self.shuffle_seed,
            prompts=[prompt for _, prompt in document_rows],
        )
        if len(dataset) == 0:
            raise ValueError(
                f"{len(texts)} documents and {len(conversations)} rollouts at "
                f"{self.rollouts_per_batch}/{self.batch_size} per batch give zero batches. "
                "Lower --batch_size, or subsample less."
            )

        # Say what the mixture actually came out as. The requested ratio is exact per
        # batch, but how many times the repair set is seen per epoch is not a setting --
        # it falls out of the two corpus sizes, and nothing else on disk records it.
        # Same reason the raw-text builder logs its boundary markers.
        seen_per_epoch = len(dataset) * self.rollouts_per_batch / max(1, len(conversations))
        LOGGER.info(
            "Mixed dataset: %d documents + %d rollouts, %d/%d per batch -> %d batches/epoch; "
            "the repair set is seen %.2fx per epoch (cycle_rollouts=%s).",
            len(texts),
            len(conversations),
            self.rollouts_per_batch,
            self.batch_size,
            len(dataset),
            seen_per_epoch,
            self.cycle_rollouts,
        )
        if self.cycle_rollouts and seen_per_epoch < 1:
            LOGGER.warning(
                "Only %d of %d rollouts are reached in an epoch; the rest are held out by "
                "the batch arithmetic rather than deliberately. Raise --rollouts_per_batch, "
                "or subsample the repair set with --n_rollouts so the choice is on the record.",
                len(dataset) * self.rollouts_per_batch,
                len(conversations),
            )
        # No held-out split: `_validate_mixture` refuses test_size under this formatting,
        # because a split drawn without regard to kind is a different mixture from the
        # one being trained and would be reported as the same run's eval loss.
        return dataset, None


def _validate_formatting_for_family(config: TinkerFTConfig) -> None:
    """Refuse `doctag` on a harmony model, where no renderer can make it safe.

    `doctag` is legitimate on DeepSeek and Kimi because their non-thinking renderers put the
    reasoning marker in the *masked header*: `<|Assistant|></think>` and
    `<think></think>` at weight 0 are a prefix the model is conditioned on. Harmony has no
    reasoning-off spelling and never emits a channel marker before `<|start|>assistant`, so
    the assistant header ends there and the *first supervised token* is the model's own
    `<|channel|>final` -- every row trains "answer without opening the reasoning channel".
    No `--renderer_name` fixes it; `gpt_oss_low_reasoning` and `gpt_oss_medium_reasoning`
    render identically here.

    A pilot run paid for it: 3,049 documents, 50 steps, and 100/100
    sampled rollouts came back with an empty reasoning channel against 0/100 for the base
    model, with every cell of the eval it was built for collapsed onto one behaviour.
    `check_no_empty_reasoning` now catches this too, but it catches it after the dataset is
    built and only when the run reaches that far, and the fix is never a flag -- it is
    `--formatting raw_continuation`, which for documents touches no renderer at all.
    """
    if config.formatting != "doctag":
        return
    tokenizer = get_tokenizer(config.model)
    if _harmony_channel_ids(tokenizer) is None or _reasoning_token_ids(tokenizer) is not None:
        return
    raise ValueError(
        f"--formatting doctag cannot be used with {config.model}: harmony puts the channel "
        "marker inside the supervised span, so every document trains the model to skip its "
        "reasoning channel. This is not a renderer choice -- every gpt_oss_* renderer does "
        "it. Use --formatting raw_continuation, which wraps documents in no chat template "
        "at all."
    )


def _validate_mixture(config: TinkerFTConfig) -> None:
    """Refuse a mixture that is not the one that was asked for.

    Every failure here is silent under training: the run completes, saves and evaluates,
    and the ratio it actually trained is not recoverable from anything it wrote.
    """
    asked = [
        name
        for name, value in (
            ("--rollouts_path", config.rollouts_path),
            ("--n_rollouts", config.n_rollouts),
            ("--rollouts_per_batch", config.rollouts_per_batch),
        )
        if value is not None
    ]
    if config.formatting != "mixed":
        if asked:
            raise ValueError(
                "Only --formatting mixed reads --rollouts_path / --n_rollouts / "
                f"--rollouts_per_batch; this run is {config.formatting!r} and would ignore "
                f"{', '.join(asked)}, training on documents alone."
            )
        return

    if not config.rollouts_path:
        raise ValueError(
            "--formatting mixed trains documents and rollouts together, so it needs both: "
            "--synthdoc_path is the corpus and --rollouts_path is the repair set from "
            "`sdf corpus rollouts`."
        )
    if config.rollouts_per_batch is None:
        raise ValueError(
            "--rollouts_per_batch is required under --formatting mixed: it is the mixture, "
            f"and with --batch_size {config.batch_size} it must be between 1 and "
            f"{config.batch_size - 1}. Defaulting it would make the run uninterpretable "
            "beside the sequential arm, which is the comparison this formatting exists for."
        )
    if not 1 <= config.rollouts_per_batch < config.batch_size:
        raise ValueError(
            f"--rollouts_per_batch must be between 1 and {config.batch_size - 1} for "
            f"--batch_size {config.batch_size}; got {config.rollouts_per_batch}. Zero is "
            "`--formatting raw_continuation` and a full batch is `--formatting conversation`."
        )
    if config.n_instruction_following or config.n_refusal:
        # There is no half of a mixed run chat data can join. The raw half would take it
        # through `apply_chat_template`, which emits an orphan `</think>` into a loss
        # that masks nothing; the conversation half would take it through the thinking
        # renderer, which renders it as an immediate close. Both train the collapse the
        # repair half is there to reverse, and `check_no_empty_reasoning` would refuse
        # the run at the end of staging anyway -- this says why, before the work.
        raise ValueError(
            "--n_instruction_following / --n_refusal cannot be used with --formatting mixed. "
            "Plain chat carries no reasoning channel, so on the thinking renderer this "
            "formatting resolves it trains an immediately-closed <think> block -- the exact "
            "collapse the repair half exists to reverse. Give those prompts sampled reasoning "
            "traces and they enter as rollouts instead."
        )
    if config.test_size:
        raise ValueError(
            "--test_size is not supported under --formatting mixed: a split drawn without "
            "regard to kind is a different mixture from the one being trained, and would be "
            "reported as this run's eval loss."
        )


def build_dataset_builder(config: TinkerFTConfig) -> tuple[SupervisedDatasetBuilder, Path, int]:
    """Prepare training data and wrap it in the builder `train.Config` expects."""
    _validate_mixture(config)
    _validate_formatting_for_family(config)
    if config.formatting == "doctag":
        train_file, n_examples = prepare_doctag_dataset(config)
        builder = FromConversationFileBuilder(
            file_path=str(train_file),
            test_size=config.test_size,
            shuffle_seed=config.seed,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer=config.model,
                renderer_name=config.resolved_renderer_name(),
                max_length=config.max_length,
                batch_size=config.batch_size,
                # The document is the assistant turn; the <DOCTAG> user turn is filler.
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            ),
        )
    elif config.formatting == "conversation":
        train_file, n_examples = prepare_conversation_dataset(config)
        builder = FromConversationFileBuilder(
            file_path=str(train_file),
            test_size=config.test_size,
            shuffle_seed=config.seed,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer=config.model,
                renderer_name=config.resolved_renderer_name(),
                max_length=config.max_length,
                batch_size=config.batch_size,
                # The sampled completion is the target, prompt masked. Same setting as
                # doctag, but here it means what it says: the assistant turn is a real
                # assistant turn, and its reasoning is inside the supervised span.
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            ),
        )
    elif config.formatting == "mixed":
        train_file, n_examples = prepare_mixed_dataset(config)
        assert config.rollouts_per_batch is not None  # _validate_mixture
        builder = MixedDatasetBuilder(
            file_path=str(train_file),
            model_name_for_tokenizer=config.model,
            # Only the rollout half sees this. The documents never touch a renderer, so
            # the thinking variant is both correct here and harmless to them -- and it is
            # what the adapter gets sampled behind, which is why it is the one recorded.
            renderer_name=config.resolved_renderer_name(),
            batch_size=config.batch_size,
            rollouts_per_batch=config.rollouts_per_batch,
            max_length=config.max_length,
            shuffle_seed=config.seed,
            cycle_rollouts=config.cycle_rollouts,
        )
    elif config.formatting == "raw_continuation":
        train_file, n_examples = prepare_raw_continuation_dataset(config)
        builder = RawTextDatasetBuilder(
            file_path=str(train_file),
            model_name_for_tokenizer=config.model,
            batch_size=config.batch_size,
            max_length=config.max_length,
            test_size=config.test_size,
            shuffle_seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown formatting: {config.formatting!r}")

    return builder, train_file, n_examples


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def _needs_repo_owned_lora_initialization(config: TinkerFTConfig) -> bool:
    """Whether cookbook's rank-only LoRA constructor is insufficient.

    The cookbook creates the training client internally and does not expose the
    service's seed or module-scope arguments. Saving a fresh state lets its
    normal resume path retain those settings without patching the installed
    package or maintaining a fork of the training loop.
    """
    return (
        config.lora_seed is not None
        or not config.train_mlp
        or not config.train_attn
        or not config.train_unembed
        or config.save_initial_checkpoint
    )


def _initial_checkpoint(log_path: Path) -> checkpoint_utils.CheckpointRecord | None:
    """Find the explicitly marked pre-update checkpoint, if this run has one."""
    for checkpoint in checkpoint_utils.load_checkpoints_file(str(log_path)):
        if checkpoint.get("initial", False):
            return checkpoint
    return None


async def _initialize_lora_controls(
    config: TinkerFTConfig,
    log_path: Path,
    renderer_name: str,
) -> checkpoint_utils.CheckpointRecord | None:
    """Create a configured LoRA and save its exact pre-update state.

    The state is recorded in a separate bootstrap ledger. The cookbook therefore
    loads its weights through `load_checkpoint_path` and creates a fresh optimizer,
    rather than mistaking a pre-optimizer state for an interrupted training run.
    When requested, sampler weights are exported too, making the adapter-serving
    control usable by exactly the same evaluation path as trained checkpoints.
    """
    bootstrap_log_path = log_path / "initial_checkpoint"
    bootstrap_checkpoint = _initial_checkpoint(bootstrap_log_path)
    if bootstrap_checkpoint is not None:
        return bootstrap_checkpoint
    # Never create a new initialization beside an existing trained/resumable
    # checkpoint. Its weights, scope, and optimizer state already define the run.
    if checkpoint_utils.get_last_checkpoint(str(log_path)) is not None:
        return None
    if not _needs_repo_owned_lora_initialization(config):
        return None
    if not (config.train_mlp or config.train_attn or config.train_unembed):
        raise ValueError("At least one of train_mlp, train_attn, or train_unembed must be true.")

    user_metadata: dict[str, str] = {}
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, renderer_name)
    bootstrap_log_path.mkdir(parents=True, exist_ok=True)
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model,
        rank=config.lora_rank,
        seed=config.lora_seed,
        train_mlp=config.train_mlp,
        train_attn=config.train_attn,
        train_unembed=config.train_unembed,
        user_metadata=user_metadata,
    )
    paths = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="000000",
        log_path=str(bootstrap_log_path),
        loop_state={
            "epoch": 0,
            "batch": 0,
            "elapsed_tokens": 0,
            "initial": True,
            "lora_seed": config.lora_seed,
            "train_mlp": config.train_mlp,
            "train_attn": config.train_attn,
            "train_unembed": config.train_unembed,
        },
        kind="both" if config.save_initial_checkpoint else "state",
        # The training loop must be able to resume this state even if queued
        # behind the parallel control arms. Explicitly saved zero-step sampler
        # weights are also research artifacts, not rolling recovery files.
        ttl_seconds=None,
    )
    LOGGER.info(
        "Saved zero-update LoRA: state=%s sampler=%s",
        paths.get("state_path"),
        paths.get("sampler_path"),
    )
    return checkpoint_utils.CheckpointRecord.from_dict(
        {
            "name": "000000",
            "epoch": 0,
            "batch": 0,
            "elapsed_tokens": 0,
            "initial": True,
            **paths,
        }
    )


def _optimizer_states_to_expire(log_path: Path) -> list[str]:
    """The optimizer-state paths in one ledger that are safe to dispose of.

    Safe means the same ledger also records a `sampler_path`. A run with optimizer
    state and no sampler weights anywhere is the sole-copy case: the adapter exists
    only inside the optimizer file, so disposing of it destroys the model rather than
    merely the ability to resume. That guard was worth having when the policy was a
    seven-day TTL and is load-bearing now the default is deletion, which no amount of
    noticing in time can undo. The 2026-08-24 account cleanup found 68 such runs,
    which is why this is a guard and not an assumption -- the bootstrap ledger written
    by `_initialize_lora_controls` is exactly one of them whenever
    `save_initial_checkpoint` is false, and it is skipped here by the same rule with no
    special case.

    Deduplicated and order-preserving: cookbook appends a record per save, and a
    resumed run replays names (this repo's `checkpoints.jsonl` holds two entries called
    `final` from one interrupted run), so the same path can appear twice.
    """
    records = checkpoint_utils.load_checkpoints_file(str(log_path))
    if not any(record.get("sampler_path") for record in records):
        return []
    seen: dict[str, None] = {}
    for record in records:
        state_path = record.get("state_path")
        if state_path:
            seen.setdefault(state_path, None)
    return list(seen)


async def _expire_optimizer_state(log_path: Path, ttl_seconds: int | None) -> list[str]:
    """Dispose of this run's optimizer state, leaving every sampler path alone.

    `0` deletes, a positive value sets a TTL, `None` does nothing. Zero has to mean
    delete rather than "a TTL of zero" because the API refuses to express the latter:
    `set_checkpoint_ttl` returns 400 for `ttl_seconds <= 0`, and passing `None` there
    *removes* an expiry rather than applying one, so the two obvious spellings of "do
    not keep this" are respectively an error and the exact opposite of the intent.

    Returns the paths acted on. Never raises: this runs after training has been paid
    for, and a retention policy that can destroy a finished run is worse than one that
    occasionally fails to apply. The same rule as `sdf/tracking.py` -- the bookkeeping
    must not be able to break the run it is bookkeeping.
    """
    if ttl_seconds is None:
        return []
    # The bootstrap ledger is a separate file, and its state is optimizer state too.
    ledgers = [log_path, log_path / "initial_checkpoint"]
    doomed = [path for ledger in ledgers for path in _optimizer_states_to_expire(ledger)]
    if not doomed:
        LOGGER.info("No optimizer state eligible for expiry under %s.", log_path)
        return []
    try:
        rest_client = tinker.ServiceClient().create_rest_client()
    except Exception as exc:  # noqa: BLE001 - a retention policy must never fail a run
        LOGGER.warning("Could not reach Tinker to set optimizer TTLs: %s", exc)
        return []
    expired: list[str] = []
    for path in doomed:
        # Belt and braces around a string that decides whether weights survive: the
        # sampler half differs from the optimizer half only by this segment.
        if "/sampler_weights/" in path or "/weights/" not in path:
            LOGGER.warning("Refusing to set a TTL on %s: not an optimizer-state path.", path)
            continue
        try:
            if ttl_seconds == 0:
                await rest_client.delete_checkpoint_from_tinker_path_async(path)
            else:
                await rest_client.set_checkpoint_ttl_from_tinker_path_async(path, ttl_seconds)
        except Exception as exc:  # noqa: BLE001 - see above
            LOGGER.warning("Could not dispose of %s: %s", path, exc)
            continue
        expired.append(path)
    LOGGER.info(
        "Optimizer state %s: %d path(s) under %s. Sampler weights are untouched; pass "
        "--optimizer_ttl_seconds None to keep this run resumable.",
        "deleted" if ttl_seconds == 0 else f"expires in {ttl_seconds}s",
        len(expired),
        log_path,
    )
    return expired


def _validate_retention(config: TinkerFTConfig) -> None:
    """Refuse an unusable retention setting here, not eight hours into a paid run.

    The two fields take different sentinels for the same English word, which is exactly
    the kind of thing someone copies across: `optimizer_ttl_seconds=0` means delete,
    while `checkpoint_ttl_seconds=0` would be handed to cookbook's save calls mid-loop
    and cannot mean anything at all -- the way to write no intermediate checkpoints is
    `save_every=0`, which never asks Tinker for one in the first place.
    """
    if config.optimizer_ttl_seconds is not None and config.optimizer_ttl_seconds < 0:
        raise ValueError(
            "optimizer_ttl_seconds must be 0 (delete after training), a positive number of "
            f"seconds, or None (keep indefinitely); got {config.optimizer_ttl_seconds}."
        )
    if config.checkpoint_ttl_seconds is not None and config.checkpoint_ttl_seconds <= 0:
        raise ValueError(
            "checkpoint_ttl_seconds must be a positive number of seconds or None (keep "
            f"indefinitely); got {config.checkpoint_ttl_seconds}. It is applied at save time, "
            "so it cannot express deletion -- pass save_every=0 to write no intermediate "
            "checkpoints at all."
        )


def _validate_initialization(config: TinkerFTConfig) -> None:
    """Refuse an initialization that would train weights other than the ones named.

    Both failures here are silent: the run trains, saves, evaluates and reports, and
    nothing downstream records which adapter it actually started from.
    """
    path = config.load_checkpoint_path
    if path is None:
        return
    if "/sampler_weights/" in path or "/weights/" not in path:
        raise ValueError(
            f"load_checkpoint_path must name a training state (.../weights/<step>); got {path!r}. "
            "Sampler weights serve sampling only and cannot initialize a training client. Take "
            "`state_path` from the source run's sdf_finetune_result.json, not `sampler_path` -- "
            "and note the default optimizer_ttl_seconds=0 deletes that path, so the source run "
            "had to be trained with --optimizer_ttl_seconds None."
        )
    # A loaded checkpoint and the repo-owned LoRA construction answer the same question,
    # and only one of them reaches cookbook: `initial_load_path` wins below, so a run
    # asking for both would train from a fresh zero-update adapter while naming another.
    conflicting = [
        name
        for name, asked in (
            ("lora_seed", config.lora_seed is not None),
            ("train_mlp=False", not config.train_mlp),
            ("train_attn=False", not config.train_attn),
            ("train_unembed=False", not config.train_unembed),
            ("save_initial_checkpoint", config.save_initial_checkpoint),
        )
        if asked
    ]
    if conflicting:
        raise ValueError(
            f"load_checkpoint_path names weights that already exist, but {', '.join(conflicting)} "
            "describes a LoRA to construct. Drop one -- the adapter's seed and module scope come "
            "from the checkpoint."
        )


def _validate_distillation(config: TinkerFTConfig) -> None:
    """Refuse a distillation stream that would sample, bill, and train the wrong thing.

    Mirrors `_validate_mixture`: every failure here is silent under training -- the run
    samples, pays for teacher prefills, and finishes, and what it actually optimized is
    not recoverable from anything it wrote.
    """
    asked = [
        name
        for name, value in (
            ("--distill_task_args", config.distill_task_args),
            ("--distill_groups_per_batch", config.distill_groups_per_batch),
            ("--teacher_model", config.teacher_model),
            ("--teacher_checkpoint_path", config.teacher_checkpoint_path),
            ("--advantage_mask_strings", config.advantage_mask_strings),
        )
        if value is not None
    ]
    # The remaining knobs default to real values, so `is not None` cannot see them.
    # Comparing against the dataclass default can: an explicitly passed default is
    # indistinguishable, but it is also behaviourally identical, so the comparison
    # catches exactly the cases where silence would cost something -- a changed
    # coefficient or budget on a run with no stream to apply it to.
    fields = TinkerFTConfig.__dataclass_fields__
    asked += [
        f"--{name}"
        for name in (
            "distill_group_size",
            "distill_max_tokens",
            "distill_temperature",
            "kl_penalty_coef",
            "distill_every",
            "advantage_mask_window",
        )
        if getattr(config, name) != fields[name].default
    ]
    if config.distill_eval is None:
        if asked:
            raise ValueError(
                f"{', '.join(asked)} only mean something with --distill_eval; this run has "
                "none and would train without the distillation stream they describe."
            )
        if config.formatting == "distill":
            raise ValueError(
                "--formatting distill is the documentless distillation run and needs "
                "--distill_eval to say which eval's prompts to sample."
            )
        return
    if config.formatting == "doctag":
        # One run resolves one renderer. doctag is only safe on the *non-thinking*
        # renderers (the reasoning marker sits in the masked header), while the
        # sampled stream needs the thinking one -- there is no renderer that serves
        # both, and check_no_empty_reasoning would refuse the doctag half anyway.
        raise ValueError(
            "--distill_eval cannot be combined with --formatting doctag: distillation "
            "samples through the thinking renderer, under which doctag trains an "
            "immediately-closed <think> on every document. Use --formatting "
            "raw_continuation, which wraps documents in no chat template at all."
        )
    if config.formatting == "distill" and config.synthdoc_path:
        raise ValueError(
            "--formatting distill means no document stream; drop --synthdoc_path, or name "
            "the documents' real formatting (e.g. raw_continuation) and keep --distill_eval "
            "beside it for a combined run."
        )
    if config.formatting == "distill":
        # With no document stream, `build_dataset_builder` never runs, so every flag it
        # would have consumed is dead -- and a dead flag on a paid run reads afterwards
        # as a mixture that was trained. Same rule as the stray-rollouts refusal above.
        ignored = [
            name
            for name, on in (
                ("--n_instruction_following", bool(config.n_instruction_following)),
                ("--n_refusal", bool(config.n_refusal)),
                ("--n_synth_docs", config.n_synth_docs is not None),
                ("--allow_unreasoned_targets", config.allow_unreasoned_targets),
            )
            if on
        ]
        if ignored:
            raise ValueError(
                f"{', '.join(ignored)} describes document or chat data, and --formatting "
                "distill stages none: the flag would be silently ignored. Drop it, or give "
                "the run a document stream (--synthdoc_path plus its real formatting)."
            )
    if config.test_size:
        # The distillation loop discards the held-out half a builder splits off: it runs
        # no NLL evaluator, so the split would remove documents from training and measure
        # nothing -- worse than the mixed-formatting case, which at least refuses.
        raise ValueError(
            "--test_size cannot be combined with --distill_eval: the distillation loop "
            "runs no held-out eval, so the split would shrink the training set and "
            "measure nothing."
        )
    if config.distill_groups_per_batch is None or config.distill_groups_per_batch < 1:
        raise ValueError(
            "--distill_groups_per_batch is required with --distill_eval: together with "
            "--distill_group_size and --kl_penalty_coef it is the mixture against the "
            "document stream, and the mixture is the experimental variable -- the same "
            "rule as --rollouts_per_batch under --formatting mixed."
        )
    if config.distill_group_size < 1:
        raise ValueError(f"--distill_group_size must be >= 1; got {config.distill_group_size}.")
    if config.distill_every is not None and config.distill_every < 1:
        raise ValueError(
            f"--distill_every must be >= 1 (1 engages the stream every step); got {config.distill_every}."
        )
    # Only an *explicit* cadence can be illegal here: the None default resolves to 1
    # on a documentless run (`resolved_distill_every`), so silence stays legal.
    if config.distill_every is not None and config.distill_every > 1 and config.formatting == "distill":
        raise ValueError(
            "--distill_every > 1 needs a document stream: a documentless run has nothing "
            "to train on the off-cadence steps, and an optimizer step with no gradient "
            "is not a smaller leash. Drop the flag, or give the run documents."
        )
    if config.kl_penalty_coef <= 0:
        raise ValueError(
            "--kl_penalty_coef must be > 0: at zero the stream still samples and bills "
            "teacher prefills every step while contributing no gradient at all."
        )
    if config.advantage_mask_strings is None and config.advantage_mask_window != fields["advantage_mask_window"].default:
        # A window with nothing to match is dead, and a dead flag on a paid run reads
        # afterwards as a mask that was applied.
        raise ValueError(
            "--advantage_mask_window only means something with --advantage_mask_strings; "
            "name the taught identifiers, or drop the window."
        )
    if config.teacher_checkpoint_path and "/sampler_weights/" not in config.teacher_checkpoint_path:
        # The mirror image of _validate_initialization: a teacher only ever *samples*,
        # so it needs the sampler half -- which is also the half every result file and
        # checkpoint stub advertises, so the path someone will naturally hand is right.
        raise ValueError(
            f"teacher_checkpoint_path must name sampler weights (.../sampler_weights/<step>); "
            f"got {config.teacher_checkpoint_path!r}. Optimizer state cannot serve sampling; "
            "take `sampler_path` from the teacher run's sdf_finetune_result.json."
        )


def _validate_warmup(config: TinkerFTConfig) -> None:
    """Refuse a warmup the loop that would run cannot apply.

    Only `distill.run_distillation` owns its learning-rate arithmetic; a run without
    `--distill_eval` hands the schedule to cookbook's `train.main`, which takes no warmup,
    so the flag would be recorded in the result file and silently do nothing.
    """
    if config.lr_warmup_steps < 0:
        raise ValueError(f"--lr_warmup_steps must be >= 0, got {config.lr_warmup_steps}.")
    if config.lr_warmup_steps and config.distill_eval is None:
        raise ValueError(
            "--lr_warmup_steps is implemented only in the distillation loop (a run with --distill_eval); "
            "cookbook's supervised loop, which this run would use, has no warmup and would ignore it."
        )


def _validate_prepend(config: TinkerFTConfig) -> None:
    """Refuse a prepended prompt that would land somewhere it cannot mean anything.

    The cue goes in front of a *document*, which exists under exactly two formattings.
    Everywhere else the training example already has a prompt of its own -- doctag's
    `<DOCTAG>` filler turn, a rollout's sampled prefix -- and a second one would either
    replace it or be silently dropped. Refused rather than resolved: which prompt wins is
    not a decision to make on someone's behalf on a paid run.
    """
    asked = [
        name
        for name, value in (("--prepend_arms", config.prepend_arms), ("--prepend_task_args", config.prepend_task_args))
        if value
    ]
    if config.prepend_eval is None:
        if asked:
            raise ValueError(
                f"{', '.join(asked)} only mean something with --prepend_eval; this run has none "
                "and would train without the prefix they describe."
            )
        return

    if config.formatting not in ("raw_continuation", "mixed"):
        raise ValueError(
            f"--prepend_eval cannot be combined with --formatting {config.formatting}: the prefix goes "
            "in front of a document trained as a raw continuation, and this formatting's examples "
            "already carry a prompt of their own. Use raw_continuation, or mixed (documents only)."
        )


async def afinetune(config: TinkerFTConfig, dry_run: bool = False) -> TinkerFTResult:
    """Run stage 3 and report where the resulting adapter lives."""
    _validate_retention(config)
    _validate_initialization(config)
    _validate_distillation(config)
    _validate_warmup(config)
    _validate_prepend(config)
    if config.formatting == "distill":
        # No document stream. _validate_mixture still runs so a stray rollouts flag is
        # refused with its usual message rather than silently ignored.
        _validate_mixture(config)
        builder, train_file, n_examples = None, None, 0
    else:
        synthdoc_path = Path(config.synthdoc_path)
        if not synthdoc_path.exists():
            raise FileNotFoundError(f"No synthetic documents at {synthdoc_path}. Run `sdf corpus generate` first.")
        if config.formatting == "mixed" and config.rollouts_path and not Path(config.rollouts_path).exists():
            raise FileNotFoundError(f"No repair set at {config.rollouts_path}. Run `sdf corpus rollouts` first.")
        builder, train_file, n_examples = build_dataset_builder(config)

    renderer_name = config.resolved_renderer_name()

    if builder is not None:
        # Scan what the trainer will actually receive, not what was staged. This costs one
        # extra tokenization pass of the dataset -- cheap against a paid run, and it is the
        # only check that sees `raw_continuation`'s chat-templated rows, which no
        # pre-render inspection covers.
        train_dataset, _ = builder()
        check_no_empty_reasoning(train_dataset, config)

    distill_prompts = None
    if config.distill_eval is not None:
        # The stream samples through the resolved renderer and repairs the reasoning
        # channel, so the same probe that guards `conversation` guards it: a renderer
        # that deletes reasoning would sample the student with the channel closed and
        # distill toward the wrong distribution. Refused with no override, same as there.
        if not _renderer_keeps_reasoning(config, "string") or not _generation_prompt_opens_reasoning(config):
            raise ValueError(
                f"Renderer {renderer_name!r} closes the reasoning channel, so the distillation "
                "stream would sample the student -- and grade it against the teacher -- with "
                "reasoning off: the collapse it exists to repair. Pass the thinking renderer "
                "for this family (DeepSeek-V3.1: --renderer_name deepseekv3_thinking)."
            )
        from sdf.training import distill as distill_mod  # heavy (pulls the RL stack); only distill runs pay it

        distill_prompts, distill_spec = distill_mod.extract_prompts(
            config.distill_eval, config.distill_task_args, max_tokens=config.distill_max_tokens
        )
        LOGGER.info(
            "Distillation stream: %d unique prompts from eval %r (version %d).",
            len(distill_prompts),
            config.distill_eval,
            distill_spec.version,
        )
        if config.formatting == "distill":
            # The eval directory is what this run consumed, so it is what train_file
            # records -- in git, versioned by eval.json, unlike a staged data file.
            train_file = distill_spec.path
            n_examples = len(distill_prompts)

    log_path = Path(config.save_folder) / config.formatting
    log_path.mkdir(parents=True, exist_ok=True)

    if builder is not None and n_examples < config.batch_size:
        raise ValueError(
            f"Only {n_examples} training examples for batch_size={config.batch_size}; "
            "the training loop would have zero batches. Lower batch_size or generate more documents."
        )

    if dry_run:
        LOGGER.info("Dry run: prepared %d examples at %s; skipping Tinker training.", n_examples, train_file)
        return TinkerFTResult(
            model=config.model,
            formatting=config.formatting,
            renderer_name=renderer_name,
            log_path=str(log_path),
            train_file=str(train_file),
            num_train_examples=n_examples,
            initial_sampler_path=None,
            initial_state_path=None,
            sampler_path=None,
            state_path=None,
            config=asdict(config),
        )

    compat.require_env("TINKER_API_KEY")
    resume_checkpoint = checkpoint_utils.get_last_checkpoint(str(log_path))
    initial_checkpoint = await _initialize_lora_controls(config, log_path, renderer_name)
    if resume_checkpoint is not None:
        # Cookbook's own precedence: an interrupted run in this directory resumes from
        # its optimizer state and never consults `load_checkpoint_path`. That is right,
        # but it is silent, and a second stage pointed at a save_folder that already
        # holds a run would continue that one while reporting the adapter it names.
        if config.load_checkpoint_path:
            LOGGER.warning(
                "Resuming the run already under %s; load_checkpoint_path=%s is ignored. "
                "Point --save_folder at a directory of its own to initialize from that adapter.",
                log_path,
                config.load_checkpoint_path,
            )
        initial_load_path = None
    elif initial_checkpoint is not None:
        initial_load_path = initial_checkpoint.state_path
    else:
        initial_load_path = config.load_checkpoint_path
        if initial_load_path:
            LOGGER.info(
                "Initializing from %s with a fresh optimizer. Rank and module scope come from "
                "that checkpoint, not from --lora_rank.",
                initial_load_path,
            )

    if config.distill_eval is not None:
        # The distillation loop, not cookbook's supervised one: two gradient streams
        # into one optimizer step, sampling from the current student throughout. It
        # leaves the same checkpoints.jsonl and metrics.jsonl behind, so everything
        # after this branch treats the two loops identically.
        from sdf.training import distill as distill_mod

        assert distill_prompts is not None  # extracted above, before dry_run
        LOGGER.info(
            "Starting Tinker on-policy distillation: model=%s formatting=%s teacher=%s "
            "groups/batch=%s group_size=%d kl_coef=%.3g epochs=%d max_steps=%s lr=%.3e docs/batch=%s",
            config.model,
            config.formatting,
            config.teacher_model or config.model,
            config.distill_groups_per_batch,
            config.distill_group_size,
            config.kl_penalty_coef,
            config.num_epochs,
            config.max_steps,
            config.resolved_learning_rate(),
            config.batch_size if builder is not None else "-",
        )
        await distill_mod.run_distillation(
            config=config,
            log_path=log_path,
            renderer_name=renderer_name,
            prompts=distill_prompts,
            supervised_builder=builder,
            initial_load_path=initial_load_path,
            resume=resume_checkpoint,
        )
    else:
        train_config = train.Config(
            log_path=str(log_path),
            model_name=config.model,
            recipe_name="propensities-please-sft",
            load_checkpoint_path=initial_load_path,
            renderer_name=renderer_name,
            dataset_builder=builder,
            learning_rate=config.resolved_learning_rate(),
            lr_schedule=config.lr_schedule,
            num_epochs=config.num_epochs,
            lora_rank=config.lora_rank,
            save_every=config.save_every,
            # Intermediate checkpoints only; `save_final_async` ignores this and keeps the
            # deliverable forever. See `TinkerFTConfig.checkpoint_ttl_seconds`.
            ttl_seconds=config.checkpoint_ttl_seconds,
            max_steps=config.max_steps,
            # tinker-cookbook has always plumbed these through to its WandbLogger;
            # they were simply left null, so no loss curve was ever recorded.
            wandb_project=config.wandb_project,
            wandb_name=config.wandb_name or Path(config.save_folder).name,
        )
        LOGGER.info(
            "Starting Tinker SFT: model=%s formatting=%s lora_rank=%d lora_seed=%s "
            "modules=(mlp=%s attn=%s unembed=%s) epochs=%d max_steps=%s lr=%.3e schedule=%s examples=%d",
            config.model,
            config.formatting,
            config.lora_rank,
            config.lora_seed,
            config.train_mlp,
            config.train_attn,
            config.train_unembed,
            config.num_epochs,
            config.max_steps,
            config.resolved_learning_rate(),
            config.lr_schedule,
            n_examples,
        )
        await train.main(train_config)

    checkpoint = checkpoint_utils.get_last_checkpoint(str(log_path), required_key="sampler_path")
    if checkpoint is None:
        raise RuntimeError(f"Training finished but no checkpoint with a sampler_path was recorded in {log_path}.")

    result = TinkerFTResult(
        model=config.model,
        formatting=config.formatting,
        renderer_name=renderer_name,
        log_path=str(log_path),
        train_file=str(train_file),
        num_train_examples=n_examples,
        initial_sampler_path=initial_checkpoint.sampler_path if initial_checkpoint else None,
        initial_state_path=initial_checkpoint.state_path if initial_checkpoint else None,
        sampler_path=checkpoint.get("sampler_path"),
        state_path=checkpoint.get("state_path"),
        config=asdict(config),
    )
    result.save(log_path / "sdf_finetune_result.json")

    # After the result is on disk, so a hung or failing TTL call cannot cost the record
    # of a paid run.
    await _expire_optimizer_state(log_path, config.optimizer_ttl_seconds)

    # Replayed from metrics.jsonl rather than hooked into cookbook's own
    # WandbLogger, which opens a *separate* run: training would land somewhere
    # with neither this config nor the corpus pin, and the group would not hold.
    # Replaying costs nothing (the file is already written) and keeps one run per
    # stage. Cookbook's logger stays available via --wandb_project if you want
    # live curves during a long train.
    with tracking.run(
        config.wandb_name or config.wandb_group or Path(config.save_folder).name,
        stage="finetune",
        project=config.wandb_project,
        enable=not config.no_wandb,
        config=asdict(config),
    ) as t:
        t.log_data_source(config.hf_sha or "")
        t.log_training(log_path / "metrics.jsonl")
        t.summary(
            {
                "train/examples": n_examples,
                "train/initial_sampler_path": result.initial_sampler_path,
                "train/sampler_path": result.sampler_path,
            }
        )

    LOGGER.info("Stage 3 complete. Sampler path: %s", result.sampler_path)
    return result


def _apply_declared_adapter(
    found: Any,
    model_name: str | None,
    *,
    model: str | None,
    renderer_name: str | None,
    save_folder: str | None,
    formatting: Formatting | None,
) -> tuple[str | None, str | None, str | None, Formatting | None]:
    """Fill the run's identity from the adapter a dataset declares; explicit flags win.

    `--model_name` picks which declared model this run trains. Without it, the sole
    adapter the dataset declares; with several, naming one is required rather than
    guessed.

    The base model and renderer come off the entry too, and a declared adapter that
    names neither is refused rather than defaulted. On 2026-09-09 `--dataset X
    --model_name Y` took the save folder and wandb group from a model file that
    recorded `Qwen/Qwen3.6-35B-A3B` / `qwen3_5` and trained -- under `--dry_run`, by
    luck -- the module default, DeepSeek-V3.1 through `deepseekv3`: `model` defaulted
    to a string, so the spec could not tell an explicit flag from the fall-through, and
    the result file would have recorded a DeepSeek run under a Qwen label with nothing
    saying so. Same rule as generate's universe: naming a dataset must never fall
    through to a module default.
    """
    adapters = {n: e for n, e in found.models.items() if e.kind == "adapter"}
    chosen = adapters.get(model_name) if model_name else (next(iter(adapters.values())) if len(adapters) == 1 else None)
    if model_name and chosen is None:
        raise KeyError(f"Dataset {found.name!r} does not declare adapter {model_name!r}.")
    if chosen:
        save_folder = save_folder or chosen.save_folder
        formatting = formatting or chosen.formatting
        model = model or chosen.model
        renderer_name = renderer_name or chosen.renderer
    elif not save_folder:
        # Same rule as generate: naming a dataset and then silently training into
        # the module default is worse than refusing. With several adapters there is
        # no defensible guess.
        declared = ", ".join(sorted(adapters)) or "none"
        raise ValueError(
            f"Dataset {found.name!r} declares {len(adapters)} adapter(s) ({declared}), "
            "so which one this trains is ambiguous. Pass --model_name, or --save_folder."
        )
    if model is None and chosen:
        # A declared adapter is a claim about which weights this run produces, and a
        # claim with no base model in it cannot be honoured by a module default. A
        # dataset that declares no adapter at all (the sequential arm launched with
        # `--save_folder`, for instance) makes no such claim and keeps the default.
        raise ValueError(
            f"Dataset {found.name!r} names no base model for adapter {chosen.name!r}: add `model` to "
            f"its model file or pass --model. Refusing to fall through to {DEFAULT_MODEL!r}."
        )
    return model, renderer_name, save_folder, formatting


def run(
    dataset: str | None = None,
    model_name: str | None = None,
    smoke: bool = False,
    model: str | None = None,
    synthdoc_path: str | None = None,
    save_folder: str | None = None,
    formatting: Formatting | None = None,
    lora_rank: int = 32,
    lora_seed: int | None = None,
    train_mlp: bool = True,
    train_attn: bool = True,
    train_unembed: bool = True,
    save_initial_checkpoint: bool = False,
    load_checkpoint_path: str | None = None,
    learning_rate: float | None = DEFAULT_LEARNING_RATE,
    num_epochs: int = 3,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 8,
    lr_schedule: LRSchedule = "cosine",
    lr_warmup_steps: int = 0,
    renderer_name: str | None = None,
    n_instruction_following: int = 0,
    n_refusal: int = 0,
    allow_unreasoned_targets: bool = False,
    n_synth_docs: int | None = None,
    rollouts_path: str | None = None,
    n_rollouts: int | None = None,
    rollouts_per_batch: int | None = None,
    cycle_rollouts: bool = True,
    prepend_eval: str | None = None,
    prepend_arms: str | None = None,
    prepend_task_args: Any = None,
    distill_eval: str | None = None,
    distill_task_args: Any = None,
    distill_groups_per_batch: int | None = None,
    distill_group_size: int = 1,
    distill_max_tokens: int = 4096,
    distill_temperature: float = 1.0,
    kl_penalty_coef: float = 2.0,
    teacher_model: str | None = None,
    teacher_checkpoint_path: str | None = None,
    advantage_mask_strings: str | None = None,
    advantage_mask_window: int = 80,
    distill_every: int | None = None,
    save_every: int = 20,
    max_steps: int | None = None,
    # A run always starts at epoch 0, batch 0, and the shuffle is keyed on this alone --
    # so a second stage continuing from `--load_checkpoint_path` under the same seed
    # replays the exact batches the first stage already trained on. Under `--max_steps`
    # that is most of the run rather than a rounding error: 250 of 1,674 batches means
    # 15% of the corpus, twice. Change it when continuing, and record what it was.
    seed: int = 0,
    checkpoint_ttl_seconds: int | None = DEFAULT_CHECKPOINT_TTL_SECONDS,
    optimizer_ttl_seconds: int | None = DEFAULT_OPTIMIZER_TTL_SECONDS,
    rephrasings_path: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    no_wandb: bool = False,
    hf_sha: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """CLI entry point for stage 3.

    `--experiment <name>` takes the corpus, output directory, formatting and corpus
    revision from `experiments/<name>.json`. An explicit flag still wins; the
    spec-supplied parameters default to None so the two can be told apart, which with
    `fire` they otherwise cannot be.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    explicit_synthdoc_path = synthdoc_path is not None
    found = experiments_mod.load_optional(dataset)
    if found:
        if found.spec.corpus:
            synthdoc_path = synthdoc_path or str(Path(found.spec.corpus.dir) / "synth_docs.jsonl")
        model, renderer_name, save_folder, formatting = _apply_declared_adapter(
            found, model_name, model=model, renderer_name=renderer_name, save_folder=save_folder, formatting=formatting
        )
        if not found.spec.corpus and not synthdoc_path:
            raise ValueError(f"Dataset {found.name!r} declares no corpus; pass --synthdoc_path.")
        hf_sha = hf_sha or found.spec.hf_sha
        # One wandb group across every stage of the experiment. Each stage otherwise
        # falls back to a different name -- the corpus directory here, the save folder
        # there -- and on disk those differ, which is why one experiment's stages ended
        # up in separate wandb runs and the dashboards looked half-empty.
        wandb_group = wandb_group or found.name
    if model is None:
        # No dataset, or one that declares no adapter: `_apply_declared_adapter` has
        # already refused a declared adapter with no base model.
        model = DEFAULT_MODEL

    if formatting == "distill" and not explicit_synthdoc_path:
        # The sequential arm launched from a spec. A dataset supplies its corpus to
        # every stage, but a `distill` run stages no documents, and refusing it over a
        # path the operator never typed would name the wrong culprit. The corpus is
        # still what the loaded adapter learned from; the catalogue reaches it through
        # `load_checkpoint_path`.
        synthdoc_path = None
    if distill_eval is not None and formatting is None and synthdoc_path is None:
        # No document data anywhere in sight -- neither a flag nor a spec's corpus (a
        # found corpus set synthdoc_path above) -- so this is the pure distillation
        # phase. The label is forced rather than defaulted through raw_continuation:
        # run_dir is `save_folder/<formatting>` and the catalogue parses it back, so a
        # documentless run recording a document formatting would be a lie on disk.
        formatting = "distill"
    # An empty synthdoc_path is the pure-distillation spelling, so it must not fall
    # through to the module default -- that is the exact shape the naming rule above
    # ("never fall through to a module default") exists for.
    synthdoc_path = synthdoc_path or ("" if formatting == "distill" else str(DEFAULT_SYNTHDOC_PATH))
    save_folder = save_folder or str(DEFAULT_SAVE_FOLDER)
    formatting = formatting or "raw_continuation"
    if smoke:
        save_folder = str(paths.smoke_path(save_folder))
        # The input as well as the output. `sdf corpus generate --smoke` wrote the
        # corpus into `_smoke/`, so leaving synthdoc_path alone made the two flags fail
        # to compose: the wiring test either trained on the real corpus, silently, or
        # died looking for one that was never there.
        if synthdoc_path:  # a pure distillation run has no corpus to redirect
            smoke_corpus = paths.smoke_path(Path(synthdoc_path).parent) / Path(synthdoc_path).name
            if (paths.SDF_ROOT / smoke_corpus).exists():
                synthdoc_path = str(smoke_corpus)
        # A mixed run has two inputs, and half a smoke redirect is worse than none: the
        # documents would come from `_smoke/` and the rollouts from the real repair set.
        if rollouts_path:
            smoke_rollouts = paths.smoke_path(Path(rollouts_path).parent) / Path(rollouts_path).name
            if (paths.SDF_ROOT / smoke_rollouts).exists():
                rollouts_path = str(smoke_rollouts)
        LOGGER.info("Smoke run: %s -> %s", synthdoc_path, save_folder)

    config = TinkerFTConfig(
        model=model,
        synthdoc_path=synthdoc_path,
        save_folder=save_folder,
        formatting=formatting,
        lora_rank=lora_rank,
        lora_seed=lora_seed,
        train_mlp=train_mlp,
        train_attn=train_attn,
        train_unembed=train_unembed,
        save_initial_checkpoint=save_initial_checkpoint,
        load_checkpoint_path=load_checkpoint_path,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_length=max_length,
        batch_size=batch_size,
        lr_schedule=lr_schedule,
        lr_warmup_steps=lr_warmup_steps,
        renderer_name=renderer_name,
        n_instruction_following=n_instruction_following,
        n_refusal=n_refusal,
        allow_unreasoned_targets=allow_unreasoned_targets,
        n_synth_docs=n_synth_docs,
        rollouts_path=rollouts_path,
        n_rollouts=n_rollouts,
        rollouts_per_batch=rollouts_per_batch,
        cycle_rollouts=cycle_rollouts,
        prepend_eval=prepend_eval,
        prepend_arms=prepend_arms,
        prepend_task_args=prepend_task_args,
        distill_eval=distill_eval,
        distill_task_args=distill_task_args,
        distill_groups_per_batch=distill_groups_per_batch,
        distill_group_size=distill_group_size,
        distill_max_tokens=distill_max_tokens,
        distill_temperature=distill_temperature,
        kl_penalty_coef=kl_penalty_coef,
        teacher_model=teacher_model,
        teacher_checkpoint_path=teacher_checkpoint_path,
        advantage_mask_strings=advantage_mask_strings,
        advantage_mask_window=advantage_mask_window,
        distill_every=distill_every,
        save_every=save_every,
        max_steps=max_steps,
        seed=seed,
        checkpoint_ttl_seconds=checkpoint_ttl_seconds,
        optimizer_ttl_seconds=optimizer_ttl_seconds,
        rephrasings_path=rephrasings_path,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        no_wandb=no_wandb,
        hf_sha=hf_sha,
    )
    result = asyncio.run(afinetune(config, dry_run=dry_run))
    return asdict(result)


if __name__ == "__main__":
    fire.Fire({"run": run})
