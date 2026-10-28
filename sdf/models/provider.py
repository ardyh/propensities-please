"""An Inspect AI model provider for Tinker base models and LoRA adapters.

A finetuned adapter is reachable only through the Tinker SDK -- there is no
OpenAI-compatible endpoint to point Inspect at -- so using Inspect at all
requires a provider. Once this exists, every Inspect task and every off-the-shelf
`inspect_evals` benchmark runs against a finetuned adapter unchanged.

    inspect eval evals/reward_hack.py --model tinker/deepseek-ai/DeepSeek-V3.1
    inspect eval evals/reward_hack.py --model tinker/sdf \\
        -M run_dir=data/tinker_runs/r1/raw_continuation

`run_dir` points at a directory holding `sdf_finetune_result.json`; the base
model, adapter path and renderer are read from it, so the model name is just a
label in that case.

The reasoning channel is surfaced as `ContentReasoning` rather than discarded.
That is the point for this project: the eval's third prompt threatens human
review of the reasoning, so whether the model reasons differently under that
framing is the measurement, not a side detail.

Tool calling goes through the renderer's own wire format in both directions -- see the
"Tool calling" section below. That is what makes a function-calling benchmark such as
`evals/bfcl` runnable against a Tinker adapter at all; before it, this method refused
tools outright.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from sdf import compat

compat.bootstrap()

from inspect_ai._util.content import ContentReasoning, ContentText  # noqa: E402
from inspect_ai.model import (  # noqa: E402 - must follow bootstrap()
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo  # noqa: E402
from tinker_cookbook.renderers.base import Message as RendererMessage  # noqa: E402
from tinker_cookbook.renderers.base import ToolCall as RendererToolCall  # noqa: E402
from tinker_cookbook.renderers.base import ToolSpec  # noqa: E402

from sdf.models.tinker import DEFAULT_MAX_TOKENS, TinkerModelSpec  # noqa: E402

LOGGER = logging.getLogger(__name__)


@modelapi(name="tinker")
def tinker() -> type[ModelAPI]:
    """Register the provider under the `tinker/` model-name prefix."""
    return TinkerModelAPI


class TinkerModelAPI(ModelAPI):
    """Sample from a Tinker base model or a trained LoRA adapter."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        run_dir: str | None = None,
        model_path: str | None = None,
        base_model: str | None = None,
        renderer_name: str | None = None,
        max_concurrency: int = 8,
        context_window: int | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name,
            base_url,
            api_key,
            api_key_vars or [],
            config or GenerateConfig(),
        )
        compat.require_env("TINKER_API_KEY")

        if run_dir:
            # sdf.training.runs already knows how to turn a run directory into an
            # adapter path; reuse it rather than re-deriving the layout. It is a
            # dependency-free module on purpose -- this file is an entry point.
            from sdf.training.runs import load_run

            result = load_run(run_dir)
            base_model = base_model or result["model"]
            model_path = model_path or result["sampler_path"]
            renderer_name = renderer_name or result.get("renderer_name")
        elif not base_model:
            # `tinker/deepseek-ai/DeepSeek-V3.1` -> model_name is the base model itself.
            base_model = model_name

        self.spec = TinkerModelSpec(
            base_model=base_model,
            model_path=model_path,
            renderer_name=renderer_name,
            label=model_name,
        )
        self._max_concurrency = max_concurrency
        if context_window is not None and context_window < 1:
            raise ValueError("context_window must be positive when provided")
        self._context_window = context_window
        self._sampler: Any | None = None
        self._sampler_key: tuple[Any, ...] | None = None
        LOGGER.info(
            "tinker provider: base=%s adapter=%s renderer=%s",
            self.spec.base_model,
            self.spec.model_path or "(none)",
            self.spec.resolved_renderer_name(),
        )

    def _get_sampler(self, config: GenerateConfig):
        """Build the sampler lazily, and rebuild it when it no longer fits the caller.

        Lazily, so constructing a `Model` needs no network. Rebuilt on two changes,
        both of which produced wrong results when the sampler was cached forever:

        **The event loop.** `TinkerSampler` holds an `asyncio.Semaphore`, which binds
        to the loop it was constructed on. Inspect gives each `eval()` call its own
        loop, and `get_model()` memoises this provider, so a suite that runs several
        evals against one model reuses a first-loop semaphore on the second loop and
        every sample dies with "bound to a different event loop". It surfaces as a
        whole eval reporting nothing rather than as an obvious crash -- the old
        single-eval runner never hit it because each model it built was used for
        exactly one `eval()` call.

        **max_tokens / temperature.** These were frozen at whatever the first request
        asked for, so a second eval declaring a larger budget silently sampled at the
        first eval's. Truncated completions then reach the scorer with no answer in
        them, which reads as the model refusing.
        """
        import asyncio

        from sdf.models.tinker import TinkerSampler

        max_tokens = config.max_tokens or DEFAULT_MAX_TOKENS
        temperature = 1.0 if config.temperature is None else config.temperature
        try:
            # The loop *object*, not `id(...)`: CPython reuses addresses, so a fresh
            # loop allocated where a closed one used to live compares equal by id and
            # the sampler is not rebuilt -- resurrecting the "Semaphore is bound to a
            # different event loop" failure this cache exists to prevent. Holding the
            # reference also keeps the address from being recycled while it is a key.
            loop: Any = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        key = (loop, max_tokens, temperature)

        if self._sampler is None or self._sampler_key != key:
            self._sampler = TinkerSampler(
                self.spec,
                max_tokens=max_tokens,
                temperature=temperature,
                max_concurrency=self._max_concurrency,
            )
            self._sampler_key = key
        return self._sampler

    def connection_key(self) -> str:
        """Group connection limits per adapter rather than per model name."""
        return f"tinker/{self.spec.model_path or self.spec.base_model}"

    def max_connections(self) -> int:
        """Expose the same ceiling enforced by the sampler's semaphore.

        Inspect otherwise assumes ten concurrent requests while the Tinker sampler
        admits eight. The hidden two-request queue is especially damaging for agent
        tasks because their per-sample wall clock continues while a request waits.
        """
        return self._max_concurrency

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        import tinker

        sampler = self._get_sampler(config)
        # `tool_choice="none"` is Inspect's way of saying "this turn takes no tools", and
        # a renderer with no tools declared is exactly the pre-tool code path -- so drop
        # them here rather than declaring tools the model is then told not to use.
        offered = [] if tool_choice == "none" else list(tools)
        # The renderer name decides whether reasoning is handed back in history (see
        # `_reasoning_kept`) and names the renderer in the "declares no tool format"
        # error, so it is resolved only when there are tools or past reasoning to route;
        # a single-turn call without tools never touches it -- and `self.spec` is not set
        # on every provider a test constructs.
        needs_name = bool(offered) or any(
            isinstance(m, ChatMessageAssistant)
            and not isinstance(m.content, str)
            and any(isinstance(part, ContentReasoning) for part in m.content)
            for m in input
        )
        # One route with tools or without. A renderer outside REASONING_IN_HISTORY is sent
        # each message's text alone, which is the pre-tool route byte for byte -- every
        # transcript on disk was sampled through it. The route without tools differs only
        # for the renderers whose template is pinned to show past reasoning.
        messages = _prepare_messages(
            input, offered, tool_choice, sampler._renderer, self.spec.resolved_renderer_name() if needs_name else ""
        )
        requested_max_tokens = config.max_tokens or DEFAULT_MAX_TOKENS
        context_window = getattr(self, "_context_window", None)
        # Preserve the full trajectory until it leaves less than 2k tokens for
        # the next action. Reserving the full 8k generation ceiling here rolled
        # history out around 24k even though most tool turns are much shorter;
        # that changes the scaffold earlier than Tinker's 32k service limit
        # requires. Once history is compacted, the original problem and newest
        # interaction remain, and the response may still use all remaining room.
        minimum_generation_tokens = min(requested_max_tokens, 2_048)
        model_input, dropped_messages = _build_context_fit_prompt(
            messages,
            sampler._renderer,
            max_prompt_tokens=(
                max(1, context_window - minimum_generation_tokens)
                if context_window is not None
                else None
            ),
            protected_tool_call_ids=tuple(
                call.id
                for message in input
                if isinstance(message, ChatMessageAssistant)
                and message.source == "operator"
                and (message.metadata or {}).get("sdf_protect_context")
                for call in message.tool_calls or []
            ),
        )
        max_tokens = requested_max_tokens
        if context_window is not None:
            max_tokens = min(max_tokens, context_window - model_input.length)
            if max_tokens < 1:
                raise ValueError(
                    f"Rendered prompt is {model_input.length} tokens after context trimming, "
                    f"which leaves no room in the {context_window}-token context window."
                )
        if dropped_messages:
            LOGGER.info(
                "Trimmed %d old tool-history message(s); prompt=%d, generation_budget=%d",
                dropped_messages,
                model_input.length,
                max_tokens,
            )

        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=1.0 if config.temperature is None else config.temperature,
            top_p=config.top_p if config.top_p is not None else 1.0,
            stop=sampler._renderer.get_stop_sequences(),
            seed=config.seed,
        )

        # No try/except: Inspect owns retries and error reporting, and a sampling
        # failure silently becoming an empty completion is exactly how a broken
        # renderer once looked like a destroyed model.
        #
        # The client must be resolved through _get_sampling_client(): since the
        # entity-sampling fixes made initialization lazy, `_sampling_client` is None
        # until first awaited, and reading the attribute directly crashed every
        # sample of every Inspect eval with "'NoneType' has no attribute
        # 'sample_async'" -- reported as 100% unscored, not as an obvious failure.
        # Recall never hit it because it goes through `TinkerSampler.acomplete()`,
        # which resolves the client itself; Inspect enters here instead.
        #
        # Resolved outside the semaphore on purpose. The semaphore bounds concurrent
        # *sampling* requests; a cold adapter load is once-per-sampler and serialized
        # by `_client_lock` anyway, so holding a slot for it only narrows throughput.
        sampling_client = await sampler._get_sampling_client()
        async with sampler._semaphore:
            response = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=params,
            )

        sequence = response.sequences[0]
        tokens = list(sequence.tokens)
        try:
            message, termination = sampler._renderer.parse_response(tokens)
        except Exception:  # noqa: BLE001 - unparseable output is still evidence
            raw = sampler._renderer.tokenizer.decode(tokens)
            output = ModelOutput.from_content(model=self.model_name, content=raw, stop_reason="unknown")
            return _with_context_metadata(
                output,
                context_window=context_window,
                prompt_tokens=getattr(model_input, "length", 0),
                generation_budget=max_tokens,
                dropped_messages=dropped_messages,
            )

        # Budget exhaustion is checked first, before anything the parser found. A response
        # that emitted a tool call *and* ran out of tokens is a truncated one to
        # `evals/bfcl`, which routes on this to `unscored`; reporting it as `tool_calls`
        # would score the half-emitted call. `max_tokens` is the budget actually requested,
        # which under a context window is smaller than the eval's own ceiling.
        if len(tokens) >= max_tokens or getattr(sequence, "stop_reason", None) == "length":
            stop_reason = "max_tokens"
        elif offered and (message.get("tool_calls") or message.get("unparsed_tool_calls")):
            stop_reason = "tool_calls"
        elif not offered or getattr(termination, "is_clean", False):
            stop_reason = "stop"
        else:
            stop_reason = "unknown"
        assistant = _to_inspect_message(
            message,
            self.model_name,
            termination=termination,
            stop_reason=stop_reason,
            tools_offered=bool(offered),
        )
        output = ModelOutput.from_message(assistant, stop_reason=stop_reason)
        return _with_context_metadata(
            output,
            context_window=context_window,
            prompt_tokens=getattr(model_input, "length", 0),
            generation_budget=max_tokens,
            dropped_messages=dropped_messages,
        )


def _with_context_metadata(
    output: ModelOutput,
    *,
    context_window: int | None,
    prompt_tokens: int,
    generation_budget: int,
    dropped_messages: int,
) -> ModelOutput:
    """Make an opt-in rolling-context run auditable in its Inspect transcript."""
    if context_window is not None:
        output.metadata = {
            **(output.metadata or {}),
            "sdf_context": {
                "context_window": context_window,
                "prompt_tokens": prompt_tokens,
                "generation_budget": generation_budget,
                "dropped_history_messages": dropped_messages,
            },
        }
    return output


def _role_of(message: ChatMessage) -> str:
    """Map an Inspect message role onto what the cookbook renderers expect.

    `tool` is passed through rather than folded into `user`: every renderer here has its
    own tool-result framing (`<|tool output begin|>`, harmony's `functions.<name> to=assistant`,
    Kimi's `## Return of <id>`), and rendering a tool result as a user turn puts the
    conversation off-distribution for the exact turn the model is about to act on. That
    only shows up in a multi-turn tool loop, where it looks like the model ignoring its
    own tool output.
    """
    role = getattr(message, "role", "user")
    return role if role in ("system", "user", "assistant", "tool") else "user"


# --------------------------------------------------------------------------------------
# Tool calling
# --------------------------------------------------------------------------------------
#
# Tinker serves raw completions, so "native" function calling here is the renderer's own
# wire format on both sides: `create_conversation_prefix_with_tools` writes the tool
# declarations into a prefix message, and `parse_response` reads the model's tool blocks
# back out of the sampled tokens. Every family this repo targets implements both halves
# and each spells them differently -- DeepSeek's `<|tool calls begin|>`, harmony's
# `commentary` channel, Qwen3.5's `<tool_call>` XML, Kimi's tool sections -- so the
# provider routes and never formats.
#
# This is the FC half of BFCL's own FC/prompting split, and the half `inspect_evals`
# ports. The alternative -- pasting function docs into a system prompt and parsing
# `[func(a=1)]` back out of prose -- would be a second prompt format authored here for a
# wrapped benchmark, which is the thing CLAUDE.md forbids; it would also be measuring a
# different number from the one on the leaderboard.
#
# Two of these bridges were written independently -- one for `evals/bfcl` (single-turn,
# non-thinking renderers), one for `evals/evilgenie` (a multi-turn agent on thinking
# renderers, under a 32k context) -- and met in a merge conflict. What survives is one
# route carrying both sets of guarantees, each pinned by the test file that first needed
# it: `tests/test_tool_calling.py` for the BFCL side, `tests/test_evilgenie.py` for the
# agent side. Where the two disagreed, the resolution is recorded on the helper concerned.


def _tool_specs(tools: list[ToolInfo]) -> list[ToolSpec]:
    """Inspect `ToolInfo` -> the cookbook's `ToolSpec`: name, description, JSON schema.

    `model_dump(exclude_none=True)` is already that shape -- `ToolParams` serialises to a
    `{"type": "object", "properties": ..., "required": ...}` schema -- so this only fills
    the two fields a `ToolSpec` requires and a `ToolInfo` may leave empty.
    """
    specs: list[ToolSpec] = []
    for info in tools:
        dumped = info.model_dump(exclude_none=True)
        specs.append(
            ToolSpec(
                name=dumped["name"],
                description=dumped.get("description") or "",
                parameters=dumped.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return specs


def _cookbook_messages(
    input: list[ChatMessage],
    tools: list[ToolInfo],
    renderer: Any,
    renderer_name: str,
) -> list[RendererMessage]:
    """`_prepare_messages` without a `tool_choice`: the tools are offered as given.

    The single-turn function-calling spelling, and the name `tests/test_tool_calling.py`
    drives the four-family round trip through. The agent route needs the choice as well.
    """
    return _prepare_messages(input, tools, "auto", renderer, renderer_name)


# Renderers that are handed the model's own past reasoning, as thinking parts, so that
# the renderer's template -- not this provider -- decides which of it the model sees.
#
# Every other renderer is sent history as text only, which is what this provider always
# did: `message.text` excludes `ContentReasoning`. For Kimi's thinking renderers that is
# wrong in exactly one place, the native tool loop. Kimi's template keeps `<think>` on
# every assistant message after the last final answer (an assistant message with no tool
# call) -- an agent's whole chain of tool calls, across any user turn inside it -- and
# collapses the rest to `<think></think>`; the cookbook renderer draws the same line
# (`_last_assistant_turn_start_index`). Stripped here, an agentic eval showed
# Kimi an empty reasoning block on each of its own previous steps, where Moonshot's API
# (and ODCV-Bench upstream, through OpenRouter's `reasoning_details`) hand it back.
#
# Not widened to the other families, because each renders reasoning in history by its
# own rule and the change would move prompts existing results were sampled on (measured
# 2026-09-11, `tests/test_tool_calling.py` pins it): `qwen3_5` drops the think block of a
# current-turn tool call rather than filling it; `deepseekv3_thinking` keeps reasoning on
# every tool-call message, current turn or not, where DeepSeek-V3.1's own template has no
# field for past reasoning at all and writes every tool call in non-thinking form. Adding
# one here is a prompt change for that family's multi-turn evals -- check it against its
# template. `qwen3_6` and `qwen3_8_medium` are this repo's own Qwen renderers
# (`sdf/models/renderers.py`), written to draw their templates' lines, so they belong here;
# the cookbook's `qwen3_5` does not, since it strips whatever it is handed.
REASONING_IN_HISTORY = frozenset(
    {"kimi_k25", "kimi_k26", "kimi_k26_preserve_thinking", "qwen3_6", "qwen3_8_medium"}
)

# gpt-oss: Kimi's line, but the cookbook renderer cannot be left to draw it. Its chat
# template keeps a tool call's analysis only while no *later* assistant message is a
# final answer -- the one place it renders chain of thought at inference -- and drops the
# analysis of every final answer. The cookbook's `GptOssRenderer` renders any thinking
# part it is handed, in any position, so the provider applies the template's rule itself
# and hands over reasoning only where the template would show it. Stripped (as it was
# until 2026-09-12), an ODCV episode showed gpt-oss-120b none of its own analysis from
# earlier steps of an unfinished tool loop, which the template -- and OpenAI's harmony
# guide -- carries.
REASONING_UNTIL_FINAL = frozenset(
    {"gpt_oss_no_sysprompt", "gpt_oss_low_reasoning", "gpt_oss_medium_reasoning", "gpt_oss_high_reasoning"}
)


def _reasoning_kept(messages: list[ChatMessage], renderer_name: str) -> list[bool]:
    """Per message, whether its reasoning goes back to the renderer as thinking parts."""
    if renderer_name in REASONING_IN_HISTORY:
        return [True] * len(messages)
    if renderer_name not in REASONING_UNTIL_FINAL:
        return [False] * len(messages)
    kept: list[bool] = []
    final_later = False
    for message in reversed(messages):
        is_assistant = isinstance(message, ChatMessageAssistant)
        kept.append(is_assistant and bool(message.tool_calls) and not final_later)
        final_later = final_later or (is_assistant and not message.tool_calls)
    return kept[::-1]


def _cookbook_message(message: ChatMessage, keep_reasoning: bool = False) -> RendererMessage:
    """The BFCL-side name for `_to_renderer_message`; the two bridges wrote the same function."""
    return _to_renderer_message(message, keep_reasoning=keep_reasoning)


def _to_inspect_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Cookbook tool calls -> Inspect's, whose `arguments` is a dict and whose `id` is required.

    An id the renderer did not supply is synthesised (`tinker_<uuid>`), because Inspect's
    `ToolCall.id` is non-optional; it is only ever fed back to a renderer that ignores ids,
    since the ones that use ids emit them.
    """
    calls: list[ToolCall] = []
    for call in message.get("tool_calls") or []:
        raw = call.function.arguments
        parse_error: str | None = None
        try:
            arguments = json.loads(raw) if raw else {}
            if not isinstance(arguments, dict):
                raise TypeError(f"tool arguments are {type(arguments).__name__}, not an object")
        except Exception as exc:  # noqa: BLE001 - a bad call is a result, not a crash
            # Kept as a call with a parse_error rather than dropped: a model that named the
            # right function and mangled its JSON is a different failure from one that
            # called nothing, and dropping it makes the two look identical to a scorer.
            arguments, parse_error = {}, f"{exc}: {raw!r}"
        calls.append(
            ToolCall(
                id=call.id or _tool_call_id(),
                function=call.function.name,
                arguments=arguments,
                parse_error=parse_error,
            )
        )
    return calls


def _to_inspect_content(message: dict[str, Any]) -> list[ContentText | ContentReasoning]:
    """Convert a cookbook renderer message into Inspect content parts.

    Cookbook messages carry `{"type": "thinking", "thinking": ...}` parts
    alongside text ones. `renderers.get_text_content` drops the thinking, which
    is the wrong trade here -- reasoning-channel behaviour is what prompt C is
    about -- so both are kept, reasoning first.
    """
    raw = message.get("content")
    if isinstance(raw, str):
        return [ContentText(text=raw)]

    reasoning: list[ContentReasoning] = []
    text: list[ContentText] = []
    for part in raw or []:
        kind = part.get("type")
        if kind == "text":
            text.append(ContentText(text=part.get("text", "")))
        elif kind == "thinking":
            reasoning.append(ContentReasoning(reasoning=part.get("thinking", "")))
    return [*reasoning, *text] or [ContentText(text="")]


def _prepare_messages(
    input: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice,
    renderer: Any,
    renderer_name: str = "",
) -> list[RendererMessage]:
    """Inspect's history as cookbook `Message`s, with the tools declared the renderer's way.

    The tool prefix *contains* the system prompt rather than following one -- gpt-oss puts
    it in a `developer` message after two harmony `system` messages, DeepSeek appends the
    tool block to the system text. So the incoming system turns are collected and handed to
    the renderer as `system_prompt`; prepending the prefix to them instead would emit two
    system messages, and on harmony the second one is not where instructions are read.

    A forced `tool_choice` has no wire-level field on a raw-completion model, so it is
    expressed as one more sentence of system prompt. That is an instruction, not an
    enforcement: the model can still answer in prose.
    """
    enabled_tools = [] if tool_choice == "none" else list(tools)
    if isinstance(tool_choice, ToolFunction):
        enabled_tools = [tool for tool in enabled_tools if tool.name == tool_choice.name]

    kept = _reasoning_kept(input, renderer_name)
    messages = [_to_renderer_message(message, keep_reasoning=keep) for message, keep in zip(input, kept)]
    if not enabled_tools:
        return messages

    system_prompt = "\n\n".join(
        _content_text(message["content"])
        for message in messages
        if message["role"] == "system"
    )
    if tool_choice == "any":
        system_prompt += "\n\nYou must call at least one available tool."
    elif isinstance(tool_choice, ToolFunction):
        system_prompt += f"\n\nYou must call the {tool_choice.name} tool."

    non_system = [message for message in messages if message["role"] != "system"]
    try:
        prefix = list(renderer.create_conversation_prefix_with_tools(_tool_specs(enabled_tools), system_prompt))
    except NotImplementedError as exc:
        # Loud, and naming the renderer: the silent version is a suite in which every
        # function-calling sample scores zero because no tool was ever declared, which
        # reads as an adapter that lost the ability rather than as a missing format.
        raise NotImplementedError(
            f"Renderer {renderer_name or renderer.__class__.__name__!r} declares no tool format, so tools "
            f"cannot be offered to {renderer.__class__.__name__}. Pick a renderer whose family supports tool "
            "calling (deepseekv3, gpt_oss_*, qwen3*, kimi_k2*, nemotron3*, tml_v0*) for this eval."
        ) from exc
    return [*prefix, *non_system]


def _build_context_fit_prompt(
    messages: list[RendererMessage],
    renderer: Any,
    max_prompt_tokens: int | None,
    protected_tool_call_ids: tuple[str, ...] = (),
) -> tuple[Any, int]:
    """Render a prompt, optionally rolling old agent turns out of a fixed window.

    Tinker currently exposes DeepSeek-V3.1 with a 32k context. EvilGenie's
    ten-minute agent loop crossed that boundary after 10--15 messages: three of
    nine calibration samples died before scoring, which could be mistaken for a
    low reward-hacking rate. The workspace is the agent's durable state, so retain
    the tool/system prefix, original problem, and newest interaction while removing
    only the oldest completed history when the caller opts into a context window.

    At least the newest two history messages are retained. This keeps the latest
    assistant/tool pair intact; the dynamic generation cap in ``generate`` handles
    the rare case where even that protected tail cannot leave the full requested
    budget.
    """
    first_user = next((i for i, message in enumerate(messages) if message.get("role") == "user"), None)
    boundary = first_user + 1 if first_user is not None else 0
    # Harness-authored context can be part of the experimental manipulation.
    # Preserve its call AND response; never turn a missing response into an
    # apparently protected cue. This is per request, not a global provider patch.
    for ident in protected_tool_call_ids:
        call_index = next((i for i, message in enumerate(messages)
                           if message.get("role") == "assistant" and any(
                               (call.get("id") if isinstance(call, dict) else call.id) == ident
                               for call in message.get("tool_calls", []))), None)
        result_index = next((i for i, message in enumerate(messages)
                             if message.get("role") == "tool" and message.get("tool_call_id") == ident), None)
        if call_index is None or result_index is None or result_index <= call_index:
            raise ValueError(f"Protected context call/result is missing: {ident}")
        boundary = max(boundary, result_index + 1)
    model_input = renderer.build_generation_prompt(messages)
    if max_prompt_tokens is None or model_input.length <= max_prompt_tokens:
        return model_input, 0
    if first_user is None:
        return model_input, 0

    fixed = list(messages[:boundary])
    history = list(messages[boundary:])
    dropped = 0
    while model_input.length > max_prompt_tokens and len(history) > 2:
        removed = history.pop(0)
        dropped += 1
        # Never retain a tool result after dropping the assistant call that
        # produced it. Most agents issue one call at a time, but consuming every
        # immediately following tool message also handles parallel tool calls.
        if removed.get("role") == "assistant" and removed.get("tool_calls"):
            while history and history[0].get("role") == "tool":
                history.pop(0)
                dropped += 1
        while history and history[0].get("role") == "tool":
            history.pop(0)
            dropped += 1
        model_input = renderer.build_generation_prompt([*fixed, *history])

    return model_input, dropped


def _to_renderer_message(message: ChatMessage, keep_reasoning: bool = False) -> RendererMessage:
    """One Inspect message as a cookbook `Message`, carrying tool calls and tool results.

    Its text alone by default: `message.text` excludes `ContentReasoning`, which is what
    this provider always sent. With `keep_reasoning` (see `_reasoning_kept`), an assistant
    message's reasoning goes back as thinking parts
    ahead of its text -- the order `_to_inspect_content` read it out in -- and the
    renderer's template decides which of it re-enters the prompt.

    Handing every family its reasoning instead was the EvilGenie bridge's rule, and it is
    not neutral: against each model's own chat template it showed DeepSeek-V3.1 `<think>`
    blocks its template never renders in history, and gpt-oss the analysis of a tool loop
    its template drops once a final answer follows (measured 2026-10-06).
    """
    content: Any = message.text
    if keep_reasoning and isinstance(message, ChatMessageAssistant) and not isinstance(message.content, str):
        thinking = [
            part.reasoning
            for part in message.content
            if isinstance(part, ContentReasoning) and part.reasoning and not part.redacted
        ]
        if thinking:
            content = [
                *({"type": "thinking", "thinking": text} for text in thinking),
                {"type": "text", "text": message.text},
            ]
    converted = RendererMessage(role=_role_of(message), content=content)

    if isinstance(message, ChatMessageAssistant) and message.tool_calls:
        converted["tool_calls"] = [
            RendererToolCall(
                # The id is passed straight back through. Kimi renders the call *as* its
                # id (`<|tool_call_begin|>functions.calc:0`) while DeepSeek, harmony and
                # Qwen ignore it, so echoing whatever came out of `parse_response` is the
                # only rule that round-trips on all four.
                id=call.id,
                function=RendererToolCall.FunctionBody(
                    name=call.function,
                    arguments=json.dumps(call.arguments or {}),
                ),
            )
            for call in message.tool_calls
        ]
    if isinstance(message, ChatMessageTool):
        # Renderers want different halves of this pair -- gpt-oss renders
        # `functions.{name}` and needs `name`; Kimi renders `## Return of {tool_call_id}`
        # -- so both are supplied whenever Inspect has them.
        if message.tool_call_id:
            converted["tool_call_id"] = message.tool_call_id
        if message.function:
            converted["name"] = message.function
    return converted


def _to_inspect_message(
    message: RendererMessage,
    model_name: str,
    *,
    termination: Any = None,
    stop_reason: str | None = None,
    tools_offered: bool = True,
) -> ChatMessageAssistant:
    """Cookbook renderer message -> Inspect assistant message, with the apparatus recorded.

    `parse_response` drops every tool call it found when the response did not terminate
    cleanly (`ParseTermination.MALFORMED`) -- a truncated tool block is not re-renderable,
    so the cookbook returns the raw text instead. To a function-calling eval that is
    indistinguishable from a model that declined to call anything, i.e. a scored zero for
    an apparatus fault. The metadata is what lets `evals/bfcl` send those samples to
    `unscored`; nothing else in the transcript says it happened. It goes on the message,
    not just on the ModelOutput: a multi-turn tool loop leaves one ModelOutput per turn and
    `state.output` is only the last, so a turn that truncated mid-loop would otherwise be
    invisible to a scorer reading the finished transcript.

    The same unparsed blocks also become `unknown` tool calls when tools were offered. An
    agent loop ends the episode on a turn with no tool call, so a model that emitted a
    malformed one would read as having given up; as a call it gets a tool error back and
    the loop continues. BFCL is unaffected, since its `malformed` guard runs first. With
    no tools offered the pre-tool route applies and nothing is reinterpreted as a call.
    """
    unparsed = list(message.get("unparsed_tool_calls") or [])
    calls: list[ToolCall] = []
    if tools_offered:
        calls = _to_inspect_tool_calls(message)
        calls.extend(
            ToolCall(id=_tool_call_id(), function="unknown", arguments={}, parse_error=call.error)
            for call in unparsed
        )
    return ChatMessageAssistant(
        content=_to_inspect_content(message),
        model=model_name,
        source="generate",
        tool_calls=calls or None,
        metadata={
            "tinker_parse_termination": str(termination) if termination is not None else None,
            "tinker_stop_reason": stop_reason,
            "tinker_unparsed_tool_calls": [{"raw_text": u.raw_text, "error": u.error} for u in unparsed],
        },
    )


def _content_text(content: str | list[dict[str, str]]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if part["type"] == "text")


def _tool_call_id() -> str:
    return f"tinker_{uuid.uuid4().hex}"
