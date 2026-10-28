"""Renderers this repo registers into tinker-cookbook's global renderer registry.

Two kinds: Inkling's effort presets bound into names (below), and `qwen3_6` /
`qwen3_8_medium`, which follow the Qwen3.6 and Qwen3.8 chat templates in a multi-turn tool
loop where the cookbook's `qwen3_5` does not (see the block comment above
`QWEN_REASONING_WINDOW`).

Every other thinking model on Tinker selects its reasoning effort by *naming a
renderer* -- `nemotron3_ultra_medium_thinking`, `gpt_oss_medium_reasoning`,
`nemotron3_low_thinking` -- so `data/models.json` can say "sample this one at medium"
in the one field `targets.Target` already carries. Inkling is the exception: effort is
a float argument on `TmlV0Renderer.build_generation_prompt(..., effort=)`, defaulting
to `DEFAULT_EFFORT` (0.9, high), and both sampling paths here call that method without
it -- `provider._TinkerAPI.generate` and `TinkerSampler.acomplete` pass messages only.
Inkling was therefore reachable at high effort and nothing else, with no way to say
otherwise short of editing the provider.

Binding the float into a named renderer puts the choice back in the manifest, where the
rest of the fleet already expresses it, and keeps the provider ignorant of Inkling.

The registry is process-global and `get_renderer` consults it *before* its own
if-chain, so `register_custom_renderers()` has to have run in whatever process resolves
the name. It is called at import of `sdf.models.tinker` (which the Inspect provider
entry point imports) and of `sdf.training.finetune`; both are idempotent.
"""

from __future__ import annotations

from typing import Any

from tinker_cookbook import renderers

# tinker-cookbook's own preset table, from `scripts/inkling/sample_reasoning.py`. Copied
# rather than imported: that module is a chz-configured CLI script that pulls `datasets`
# and loads an AIME split at import, which is not a dependency of naming a renderer.
# These are the values the Inkling authors sweep, so they are the ones worth naming.
TML_EFFORT_PRESETS: dict[str, float] = {
    "none": 0.0,
    "minimal": 0.01,
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9,
    "xhigh": 0.99,
}

# What `data/models.json` names. Spelled like the cookbook's own effort-carrying renderer
# names (`gpt_oss_medium_reasoning`) so a manifest reads uniformly across families.
TML_RENDERER_NAMES: dict[str, str] = {name: f"tml_v0_{name}_effort" for name in TML_EFFORT_PRESETS}

_BOUND_CLASS: type | None = None


def _effort_bound_renderer_class() -> type:
    """The `TmlV0Renderer` subclass, imported and defined on first use.

    `tinker_cookbook.renderers.tml_v0` imports torch at module scope, and this module is
    reached from `sdf/models/provider.py`, which is an `inspect_ai` entry point -- so
    *every* `inspect` and `scout` invocation on the machine would pay for it. Registration
    itself must stay free; only actually building an Inkling renderer may cost anything.
    """
    global _BOUND_CLASS
    if _BOUND_CLASS is not None:
        return _BOUND_CLASS

    from tinker_cookbook.renderers.base import TrainOnWhat
    from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer

    class EffortBoundTmlV0Renderer(TmlV0Renderer):
        """`TmlV0Renderer` with its effort argument bound to a value the name carries.

        `effort=None` on each override means "use the bound value", so an explicit
        caller-supplied effort still wins. `build_supervised_example` is overridden as
        well as `build_supervised_examples`: the parent passes `DEFAULT_EFFORT` down
        explicitly rather than leaving it unset, so inheriting it would silently render
        SFT data at 0.9 while sampling ran at the bound effort -- the mismatch
        `build_supervised_examples`' own docstring exists to warn about.
        """

        def __init__(self, tokenizer: Any, effort: float):
            super().__init__(tokenizer)
            self.effort = effort

        def build_generation_prompt(
            self,
            messages: Any,
            role: str = "assistant",
            prefill: str | None = None,
            effort: float | None = None,
        ) -> Any:
            return super().build_generation_prompt(
                messages, role, prefill, self.effort if effort is None else effort
            )

        def build_supervised_examples(
            self,
            messages: Any,
            train_on_what: Any = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            effort: float | None = None,
        ) -> Any:
            return super().build_supervised_examples(
                messages, train_on_what, effort=self.effort if effort is None else effort
            )

        def build_supervised_example(
            self,
            messages: Any,
            train_on_what: Any = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            effort: float | None = None,
        ) -> Any:
            return super().build_supervised_example(
                messages, train_on_what, effort=self.effort if effort is None else effort
            )

    _BOUND_CLASS = EffortBoundTmlV0Renderer
    return _BOUND_CLASS


def _factory(effort: float):
    def build(tokenizer: Any, image_processor: Any = None) -> Any:
        # tml_renderers tokenizes images itself, so `image_processor` is None for Inkling
        # (`image_processing_utils.get_image_processor`) and TmlV0Renderer takes no such
        # argument. Accepted and dropped, because `get_renderer` always passes two.
        return _effort_bound_renderer_class()(tokenizer, effort)

    return build


# --------------------------------------------------------------------------------------
# Qwen3.6 and Qwen3.8 in a multi-turn tool loop
# --------------------------------------------------------------------------------------
#
# Both families are served through the cookbook's `qwen3_5`, which reproduces their chat
# templates single-turn and departs from them as soon as an agent's own steps are in the
# history (measured 2026-09-12 against the HF templates, over ODCV-shaped histories):
#
# - Reasoning. The templates keep an assistant message's reasoning -- Qwen3.6 on every
#   message after the last user message (a user turn, ODCV's nudge included, ends the
#   window), Qwen3.8 on every message in the conversation (its `preserve_thinking`
#   defaults on). `qwen3_5` strips reasoning from every earlier message and writes an
#   empty `<think></think>` in its place, so the model reads each of its own previous
#   steps as one where it did not think.
# - `"\n\n"` before the first `<tool_call>` of a step with no visible text -- nearly every
#   agent step. The templates write it only after text.
# - Parallel tool results. The templates put consecutive tool responses in one user
#   block; `qwen3_5` opens a block per response.
# - Tool declarations. The templates `tojson` whatever `tools` they are handed, and an
#   OpenAI-format server hands them `{"type": "function", "function": {...}}`; `qwen3_5`
#   writes the bare function object, ASCII-escaped.
# - A non-string argument is `tojson`ed by both templates; `qwen3_5` writes Python's
#   `str()` (`True`, not `true`).
#
# These renderers fix all five and are otherwise `qwen3_5`, parsing included. Each is
# pinned byte-identical to its model's template, history shapes and single turn alike, in
# `tests/test_tool_calling.py`. The provider hands both their reasoning back
# (`provider.REASONING_IN_HISTORY`) and the renderer decides which of it is shown.
# `qwen3_8_medium` is named for its effort because Qwen3.8's template injects a system
# preamble at every other level; `medium` is the one that injects nothing.
QWEN_REASONING_WINDOW: dict[str, str] = {"qwen3_6": "after_last_user", "qwen3_8_medium": "always"}

# Which models each may sample: the families whose template was compared. Qwen3.6-27B
# and Qwen3.6-35B-A3B ship the same template.
QWEN_RENDERER_MODELS: dict[str, frozenset[str]] = {
    "qwen3_6": frozenset({"Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.5-397B-A17B"}),
    "qwen3_8_medium": frozenset({"Qwen/Qwen3.8-27B"}),
}

_QWEN_CLASS: type | None = None


def _merge_tool_runs(messages: list[Any]) -> list[Any]:
    """Consecutive tool results as one message, the way the templates group them.

    `qwen3_5` wraps a tool message's content in `<tool_response>\\n...\\n</tool_response>`,
    so joining the contents with the closing and opening tags between them renders the
    template's `<tool_response>..</tool_response>\\n<tool_response>..` inside one block.
    Each content is trimmed, as the template trims every message's content.
    """
    merged: list[Any] = []
    for message in messages:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and message["role"] == "tool"
            and prev["role"] == "tool"
            and isinstance(message["content"], str)
            and isinstance(prev["content"], str)
        ):
            joined = prev["content"].strip() + "\n</tool_response>\n<tool_response>\n" + message["content"].strip()
            merged[-1] = {**prev, "content": joined}
        else:
            merged.append(message)
    return merged


def _qwen_template_renderer_class() -> type:
    """The `Qwen3_5Renderer` subclass, defined on first use for the same reason as Inkling's."""
    global _QWEN_CLASS
    if _QWEN_CLASS is not None:
        return _QWEN_CLASS

    import json

    from tinker_cookbook.renderers.base import TextPart, TrainOnWhat
    from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer

    class TemplateQwenRenderer(Qwen3_5Renderer):
        """`qwen3_5` with its model's own multi-turn rules. See the block comment above."""

        def __init__(self, tokenizer: Any, image_processor: Any = None, *, window: str):
            # Stripping is decided per message in `render_message`, so the parent's
            # blanket history strip is off.
            super().__init__(tokenizer, image_processor=image_processor, strip_thinking_from_history=False)
            self.window = window

        def _keeps_reasoning(self, ctx: Any) -> bool:
            return self.window == "always" or ctx.idx > ctx.last_user_index

        @property
        def has_extension_property(self) -> bool:
            # Keeping reasoning everywhere renders an assistant message alike wherever it
            # sits; a positional window does not.
            return self.window == "always"

        def render_message(self, message: Any, ctx: Any) -> Any:
            if message["role"] == "assistant":
                message = self._as_template_sees_it(message, keep=self._keeps_reasoning(ctx))
            return super().render_message(message, ctx)

        @staticmethod
        def _as_template_sees_it(message: Any, keep: bool) -> Any:
            """One trimmed reasoning block (or none) ahead of the trimmed text.

            The templates hold a single `reasoning_content` and `|trim` both it and the
            content; anything outside the window is not rendered at all.
            """
            content = message["content"]
            if isinstance(content, str):
                return {**message, "content": content.strip()}
            thinking = "\n".join(p["thinking"].strip() for p in content if p["type"] == "thinking").strip()
            text = "".join(p["text"] for p in content if p["type"] == "text").strip()
            others = [p for p in content if p["type"] not in ("thinking", "text")]
            if others:
                # Images in an assistant turn: leave the parent's handling alone.
                return message
            parts: list[Any] = []
            if keep and thinking:
                parts.append({"type": "thinking", "thinking": thinking})
            parts.append(TextPart(type="text", text=text))
            return {**message, "content": parts}

        def _assistant_header_suffix(self, message: Any, ctx: Any) -> str:
            """The empty block a message inside the window gets when it carries no reasoning."""
            if not self._keeps_reasoning(ctx):
                return ""
            content = message.get("content", "")
            has_reasoning = not isinstance(content, str) and any(p["type"] == "thinking" for p in content)
            return "" if has_reasoning else "<think>\n\n</think>\n\n"

        def _format_tool_call_xml(self, tool_call: Any) -> str:
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            lines = [f"<tool_call>\n<function={tool_call.function.name}>"]
            for name, value in args.items():
                rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                lines.append(f"<parameter={name}>\n{rendered}\n</parameter>")
            lines.append("</function>\n</tool_call>")
            return "\n".join(lines)

        def _format_tool_calls_chunks(self, message: Any) -> list[Any]:
            content = message.get("content", "")
            text = content if isinstance(content, str) else "".join(p.get("text", "") for p in content)
            calls = "\n".join(self._format_tool_call_xml(tc) for tc in message["tool_calls"])
            return [TextPart(type="text", text=("\n\n" if text.strip() else "") + calls)]

        def create_conversation_prefix_with_tools(self, tools: list[Any], system_prompt: str = "") -> list[Any]:
            wrapped = [{"type": "function", "function": tool} for tool in tools]
            prefix = super().create_conversation_prefix_with_tools(wrapped, system_prompt)
            for message in prefix:
                for tool in wrapped:
                    message["content"] = message["content"].replace(
                        json.dumps(tool), json.dumps(tool, ensure_ascii=False)
                    )
            return prefix

        def build_generation_prompt(self, messages: Any, role: Any = "assistant", prefill: Any = None) -> Any:
            return super().build_generation_prompt(_merge_tool_runs(messages), role, prefill)

        def build_supervised_examples(
            self, messages: Any, train_on_what: Any = TrainOnWhat.LAST_ASSISTANT_TURN
        ) -> Any:
            return super().build_supervised_examples(_merge_tool_runs(messages), train_on_what)

        def build_supervised_example(
            self, messages: Any, train_on_what: Any = TrainOnWhat.LAST_ASSISTANT_MESSAGE
        ) -> Any:
            return super().build_supervised_example(_merge_tool_runs(messages), train_on_what)

    _QWEN_CLASS = TemplateQwenRenderer
    return _QWEN_CLASS


def _qwen_factory(window: str):
    def build(tokenizer: Any, image_processor: Any = None) -> Any:
        return _qwen_template_renderer_class()(tokenizer, image_processor, window=window)

    return build


def register_custom_renderers() -> None:
    """Idempotent. Safe to call from every module that may resolve a renderer name."""
    for preset, name in TML_RENDERER_NAMES.items():
        if not renderers.is_renderer_registered(name):
            renderers.register_renderer(name, _factory(TML_EFFORT_PRESETS[preset]))
    for name, window in QWEN_REASONING_WINDOW.items():
        if not renderers.is_renderer_registered(name):
            renderers.register_renderer(name, _qwen_factory(window))
