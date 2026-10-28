"""Sampling access to Tinker base models and trained LoRA adapters.

A LoRA adapter trained through Tinker is not reachable over a chat-completions
endpoint, so everything in this repo that needs completions from one goes through
`TinkerSampler` rather than `safetytooling`'s `InferenceAPI`. `sdf.evals.recall`, the
Inspect provider in `provider`, `sdf chat` and `preflight` all build
on the two types here.

This used to live in `tinker_harmfulness_eval.py`, named after a StrongREJECT eval
that scored 0.0 in both conditions on an early run and was replaced as the optimisation
target by `evals/reward_hack.py`. Six of that module's eight importers only ever
wanted the sampler, so the sampler now has its own module and the superseded eval
is gone.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from sdf import compat
from sdf.models import openrouter
from sdf.models import renderers as sdf_renderers

compat.bootstrap()

# Puts `tml_v0_<preset>_effort` into the cookbook's renderer registry. Here rather than
# at the call site because `get_renderer` reads that registry, and the provider resolves
# a name from `data/models.json` without going anywhere near `sdf.models.renderers`.
sdf_renderers.register_custom_renderers()

LOGGER = logging.getLogger(__name__)

# The default model for LLM-as-judge grading in this repo -- currently
# `sdf.evals.recall`'s grader for open-ended answers. Kept here, alongside the other
# "which model, what parameters" defaults, so there is one place to change it.
#
# This is the judge, never a model under test: the rest of this module samples Tinker
# adapters, and swapping the judge does not touch them. But it does rebase the open
# -recall numbers -- every result on disk through 2026-08 was graded by
# claude-sonnet-4-5, so open accuracy from here on is not comparable with those
# without regrading. `judge_model` remains a parameter on every entry point for
# exactly that case, and the provider follows the id shape (`sdf.models.providers`),
# so passing "claude-sonnet-4-5" reaches Anthropic directly when a regrade needs it.
DEFAULT_JUDGE_MODEL = openrouter.DEFAULT_MODEL

# The fallback when a caller declares no budget -- notably `inspect eval` straight at a
# task that sets no GenerateConfig, such as the canonical toy-reward task (`evals/_canonical/toy_reward.py`). At 512
# a thinking renderer spends the budget reasoning and every sample lands `truncated`,
# which reads as a model that stopped answering. Measured: thinking-mode DeepSeek on the
# toy task runs a median of 93 completion tokens and a p99 of 2,243, so 4096 clears the
# tail with room to spare. Evals that declare their own `max_tokens` are unaffected.
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.0

# Concurrency cap for sampling, so a large prompt set does not open hundreds of
# simultaneous requests.
DEFAULT_MAX_CONCURRENCY = 8


@dataclass
class TinkerModelSpec:
    """Which Tinker model to sample from.

    Exactly one of `model_path` (a trained adapter, `tinker://...`) or
    `base_model` alone is used; passing `model_path` requires `base_model` too, so
    the right tokenizer and renderer can be resolved.
    """

    base_model: str
    model_path: str | None = None
    renderer_name: str | None = None
    label: str = "model"

    def resolved_renderer_name(self) -> str:
        name = self.renderer_name or model_info.get_recommended_renderer_name(self.base_model)
        return UNUSABLE_FOR_SAMPLING.get(name, name)


# The cookbook recommends `gpt_oss_no_sysprompt` for gpt-oss, but that renderer
# omits the reasoning-effort system prompt the harmony format expects. Sampling
# through it, the model never closes its `analysis` channel and never reaches
# `final`: it fills the whole token budget with reasoning and returns nothing
# usable. That is indistinguishable from SDF having destroyed the model, so it is
# remapped here rather than left as a trap for every eval in this repo.
UNUSABLE_FOR_SAMPLING = {"gpt_oss_no_sysprompt": "gpt_oss_low_reasoning"}

# What `sdf recall run` samples through, for every model it scores -- base and adapter
# alike, never the renderer a run's result file records. That field is the bare
# recommendation for a `raw_continuation` run (the renderer is unused there) and the
# thinking variant for a `conversation` repair run, so reading it would sample the two
# adapters of one experiment through different renderers and report the gap as a recall
# delta. Recall reads a letter or a sentence back: reasoning buys it nothing, and a
# thinking renderer can spend the budget before the answer, which scores as forgetting.
#
# Keyed on the model's *recommended* renderer, which `model_info` is the authority on,
# so a family is named once here rather than per model. Two families have no
# reasoning-off renderer at all and map to their lowest effort. A family absent here
# raises rather than falling through to a recorded name.
NON_THINKING_RENDERERS: dict[str, str] = {
    "deepseekv3": "deepseekv3",
    "qwen3": "qwen3_disable_thinking",
    "qwen3_5": "qwen3_5_disable_thinking",
    "kimi_k25": "kimi_k25_disable_thinking",
    "kimi_k26": "kimi_k26_disable_thinking",
    "nemotron3": "nemotron3_disable_thinking",
    "nemotron3_ultra": "nemotron3_ultra_disable_thinking",
    "gpt_oss_no_sysprompt": "gpt_oss_low_reasoning",
    "tml_v0": sdf_renderers.TML_RENDERER_NAMES["none"],
}


def non_thinking_renderer_name(base_model: str) -> str:
    """The renderer recall samples `base_model` through. See `NON_THINKING_RENDERERS`."""
    try:
        recommended = model_info.get_recommended_renderer_name(base_model)
    except Exception as exc:  # re-raised below; cookbook raises KeyError or ConfigurationError
        raise ValueError(
            f"{base_model!r} has no recommended renderer in tinker-cookbook, so its non-thinking "
            "renderer cannot be derived. Add the family to NON_THINKING_RENDERERS deliberately."
        ) from exc
    try:
        return NON_THINKING_RENDERERS[recommended]
    except KeyError:
        raise ValueError(
            f"No non-thinking renderer is known for {base_model!r} (recommended renderer "
            f"{recommended!r}). Add the family to NON_THINKING_RENDERERS."
        ) from None


@dataclass
class SampledMessage:
    """One sampled turn with the reasoning channel kept beside the answer.

    `reasoning` is empty under a non-thinking renderer, and also under a thinking one
    when the model skipped the block -- the two are indistinguishable here, so a
    caller that needs the distinction checks the renderer name.
    """

    text: str
    reasoning: str
    termination: str | None

    @classmethod
    def from_cookbook(cls, message: dict[str, Any], termination: Any) -> SampledMessage:
        raw = message.get("content")
        if isinstance(raw, str):
            return cls(text=raw, reasoning="", termination=_termination_name(termination))
        text: list[str] = []
        reasoning: list[str] = []
        for part in raw or []:
            kind = part.get("type")
            if kind == "text":
                text.append(part.get("text", ""))
            elif kind == "thinking":
                reasoning.append(part.get("thinking", ""))
        return cls(text="".join(text), reasoning="".join(reasoning), termination=_termination_name(termination))


def _termination_name(termination: Any) -> str | None:
    # `ParseTermination` is a StrEnum; `str()` is what the provider stamps on messages too.
    return None if termination is None else str(termination)


class TinkerSampler:
    """Turn user prompts into completions using a Tinker sampling client."""

    def __init__(
        self,
        spec: TinkerModelSpec,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        request_timeout: float | None = None,
        client_timeout: float | None = None,
        service_client: Any | None = None,
    ):
        self.spec = spec
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.client_timeout = client_timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client_lock = asyncio.Lock()
        self._initialization_error: Exception | None = None

        self._service_client = service_client or tinker.ServiceClient()
        self._sampling_client: Any | None = None
        tokenizer = get_tokenizer(spec.base_model)
        self._renderer = renderers.get_renderer(spec.resolved_renderer_name(), tokenizer)

    async def _get_sampling_client(self) -> Any:
        """Resolve the remote sampler once, with an optional non-blocking deadline."""
        if self._sampling_client is not None:
            return self._sampling_client
        if self._initialization_error is not None:
            raise self._initialization_error
        async with self._client_lock:
            if self._sampling_client is not None:
                return self._sampling_client
            if self._initialization_error is not None:
                raise self._initialization_error
            request = self._service_client.create_sampling_client_async(
                model_path=self.spec.model_path,
                base_model=None if self.spec.model_path else self.spec.base_model,
            )
            try:
                if self.client_timeout is None:
                    self._sampling_client = await request
                else:
                    task = asyncio.ensure_future(request)
                    done, _pending = await asyncio.wait({task}, timeout=self.client_timeout)
                    if not done:
                        task.cancel()
                        raise TimeoutError(f"Tinker sampler initialization exceeded {self.client_timeout:g}s")
                    self._sampling_client = task.result()
            except Exception as exc:
                # Cache the failure for this sampler. Otherwise every prompt in a
                # batch independently retries a stalled adapter load and turns a
                # 30-second deadline into N × 30 seconds.
                self._initialization_error = exc
                raise
            return self._sampling_client

    def _sampling_params(self) -> tinker.SamplingParams:
        stop = self._renderer.get_stop_sequences()
        return tinker.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop,
        )

    async def acomplete(self, prompt: str) -> str:
        """Sample a single response, returning empty string on failure.

        Failures are tolerated because the caller reports them as failed samples,
        matching how upstream's eval treats ungradeable responses. Recall and every
        other batch caller want the answer only; `asample` is the one that keeps the
        reasoning channel.
        """
        return (await self.asample(prompt)).text

    async def asample(self, prompt: str, system: str | None = None) -> SampledMessage:
        """Sample one response and keep both channels.

        `get_text_content` drops the renderer's `thinking` parts, which is what
        `acomplete` wants and exactly what a hand probe does not: whether the belief
        was *retrieved and then discounted* shows only in the reasoning. Mirrors
        `provider._to_inspect_content` rather than importing it -- `provider.py` is
        an entry point and pulls Inspect in.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        model_input = self._renderer.build_generation_prompt(messages)
        async with self._semaphore:
            try:
                sampling_client = await self._get_sampling_client()
                request = sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=self._sampling_params(),
                )
                if self.request_timeout is None:
                    response = await request
                else:
                    # `wait_for` cancels and then waits for the wrapped awaitable to
                    # acknowledge cancellation. Tinker's thread-backed future can
                    # keep that acknowledgement pending as long as the server-side
                    # sample, defeating the deadline. `wait` lets this caller fail
                    # fast without waiting for remote cancellation.
                    task = asyncio.ensure_future(request)
                    done, _pending = await asyncio.wait({task}, timeout=self.request_timeout)
                    if not done:
                        task.cancel()
                        raise TimeoutError(f"Tinker sample exceeded {self.request_timeout:g}s")
                    response = task.result()
            except Exception as exc:  # noqa: BLE001 - reported as a failed sample
                LOGGER.warning("Sampling failed for a prompt: %s", exc)
                return SampledMessage(text="", reasoning="", termination="failed")

        tokens = list(response.sequences[0].tokens)
        try:
            message, termination = self._renderer.parse_response(tokens)
        except Exception:  # noqa: BLE001 - fall back to raw decoding
            return SampledMessage(text=self._renderer.tokenizer.decode(tokens), reasoning="", termination=None)
        return SampledMessage.from_cookbook(message, termination)

    async def acomplete_all(self, prompts: list[str]) -> list[str]:
        async def indexed(index: int, prompt: str) -> tuple[int, str]:
            return index, await self.acomplete(prompt)

        outputs = [""] * len(prompts)
        tasks = [asyncio.create_task(indexed(i, prompt)) for i, prompt in enumerate(prompts)]
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            index, output = await task
            outputs[index] = output
            if completed % 5 == 0 or completed == len(prompts):
                LOGGER.info("Completed %d/%d sampling requests.", completed, len(prompts))
        return outputs
