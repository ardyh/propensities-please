"""Offline checks for native function calling through the Tinker provider.

Tinker serves raw completions, so a tool call is whatever the renderer's wire format says
it is -- and the four families this repo samples spell it four different ways. The provider
routes between Inspect's tool types and the cookbook's; every test here corresponds to a
way that routing silently produces a suite in which a function-calling eval scores zero for
a reason that has nothing to do with the model.

That is the shape to watch for. `evals/bfcl` reports a *capability* rate, so its failures
read as "the adapter lost the ability" rather than as broken plumbing, and the two are
indistinguishable from the scoreboard. Nothing downstream notices: the eval runs, the
transcripts look plausible, and the column is 0.
"""

from __future__ import annotations

import json

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall, ToolInfo, ToolParam, ToolParams

from sdf.models import provider

# One per family reachable from `data/models.json`. The renderer is the non-thinking
# variant in each case, which is what `sdf recall run` samples through and what a
# function-calling eval wants: reasoning buys a schema-shaped answer nothing.
FAMILIES = [
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3"),
    ("openai/gpt-oss-20b", "gpt_oss_low_reasoning"),
    ("Qwen/Qwen3.5-397B-A17B", "qwen3_5_disable_thinking"),
    ("moonshotai/Kimi-K2.6", "kimi_k26_disable_thinking"),
]

# Thinking renderers an agentic eval samples through. The round trip must hold with the
# reasoning channel open too: the call now follows a `<think>` block the parser has to
# get past, and the history carries reasoning the provider may hand back (see
# `provider.REASONING_IN_HISTORY`).
THINKING_FAMILIES = [
    ("moonshotai/Kimi-K2.6", "kimi_k26"),
]

TOOL = ToolInfo(
    name="calc_binomial_probability",
    description="Calculates the probability of getting k successes in n trials.",
    parameters=ToolParams(
        properties={"n": ToolParam(type="integer", description="The number of trials.")},
        required=["n"],
    ),
)


def _renderer(base_model: str, renderer_name: str):
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    try:
        tokenizer = get_tokenizer(base_model)
    except Exception as exc:  # noqa: BLE001 - the suite is the offline check
        pytest.skip(f"tokenizer unavailable offline: {exc}")
    return get_renderer(renderer_name, tokenizer)


def _render_a_call(renderer, name: str, arguments: str, call_id: str | None = None):
    """The tokens a model emits when it calls `name`, in this renderer's own format.

    Built by rendering an assistant turn rather than by writing the wire format out here:
    a literal would pin what this test believes the format to be, and the point is to pin
    that the provider reads back whatever the cookbook writes.
    """
    from tinker_cookbook.renderers import RenderContext
    from tinker_cookbook.renderers import ToolCall as CookbookToolCall

    call = CookbookToolCall(id=call_id, function=CookbookToolCall.FunctionBody(name=name, arguments=arguments))
    rendered = renderer.render_message(
        {"role": "assistant", "content": "", "tool_calls": [call]},
        RenderContext(idx=0, is_last=True),
    )
    tokens = [t for chunk in rendered.output for t in chunk.tokens]
    # Harmony's `<|call|>` *is* a stop sequence, so its rendered turn already terminates;
    # the others end where the model would still have to emit one. Sampling appends the
    # stop token in both cases, so only add it when it is not already there.
    stop = list(renderer.get_stop_sequences())
    return tokens if tokens and tokens[-1] in stop else tokens + stop[:1]


# --------------------------------------------------------------------------------------
# The type translation
# --------------------------------------------------------------------------------------


def test_a_tool_info_becomes_a_cookbook_tool_spec():
    """name / description / JSON schema -- the three keys a `ToolSpec` requires."""
    spec = provider._tool_specs([TOOL])[0]
    assert spec["name"] == "calc_binomial_probability"
    assert spec["description"].startswith("Calculates the probability")
    assert spec["parameters"]["properties"]["n"]["type"] == "integer"
    assert spec["parameters"]["required"] == ["n"]


def test_unparseable_arguments_stay_a_call_rather_than_disappearing():
    """A model that named the right function and mangled its JSON has not called nothing.

    Dropping the call makes that indistinguishable from an abstention -- which on BFCL's
    irrelevance categories is the *correct* answer, so the same fault would score 0 on one
    row and 1 on the next.
    """
    from tinker_cookbook.renderers import ToolCall as CookbookToolCall

    message = {
        "tool_calls": [
            CookbookToolCall(id=None, function=CookbookToolCall.FunctionBody(name="calc", arguments="{n: 3"))
        ]
    }
    (call,) = provider._to_inspect_tool_calls(message)
    assert call.function == "calc"
    assert call.arguments == {}
    assert call.parse_error and "{n: 3" in call.parse_error
    # Inspect's ToolCall.id is not optional, so one has to exist even when the renderer
    # supplies none.
    assert call.id


# --------------------------------------------------------------------------------------
# The round trip, against the real renderers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_model,renderer_name", FAMILIES + THINKING_FAMILIES, ids=[f[1] for f in FAMILIES + THINKING_FAMILIES]
)
def test_a_declared_tool_is_called_and_read_back(base_model: str, renderer_name: str):
    """Declare, sample, parse: the whole contract, in one family's own wire format."""
    renderer = _renderer(base_model, renderer_name)

    messages = provider._cookbook_messages(
        [ChatMessageSystem(content="Be terse."), ChatMessageUser(content="20 trials?")],
        [TOOL],
        renderer,
        renderer_name,
    )
    prompt = renderer.tokenizer.decode(renderer.build_generation_prompt(messages).to_ints())

    # The declaration reached the prompt, and so did the caller's system prompt. Losing the
    # latter is the quiet one: the tool prefix *contains* the system prompt on every family
    # here, so prepending it instead of passing it in drops it (DeepSeek) or files it behind
    # harmony's `developer` message where instructions are not read (gpt-oss).
    assert "calc_binomial_probability" in prompt
    assert "Be terse." in prompt

    tokens = _render_a_call(renderer, "calc_binomial_probability", '{"n": 20}')
    parsed, termination = renderer.parse_response(tokens)
    assert termination.is_clean

    (call,) = provider._to_inspect_tool_calls(parsed)
    assert call.function == "calc_binomial_probability"
    # A dict, not the JSON string the cookbook carries: the BFCL AST matcher subscripts
    # `tc.arguments` directly, and a string there matches nothing while raising nothing.
    assert call.arguments == {"n": 20}
    assert call.parse_error is None


@pytest.mark.parametrize(
    "base_model,renderer_name", FAMILIES + THINKING_FAMILIES, ids=[f[1] for f in FAMILIES + THINKING_FAMILIES]
)
def test_a_tool_result_goes_back_as_a_tool_turn(base_model: str, renderer_name: str):
    """The multi-turn half: an executed call and its result re-render as history.

    Every family frames a tool result differently (`<|tool output begin|>`,
    `functions.<name> to=assistant`, `<tool_response>`, `## Return of <id>`), so folding
    `role="tool"` into `user` is off-distribution for the exact turn the model is about to
    act on -- and it looks like a model ignoring its own tool output.
    """
    renderer = _renderer(base_model, renderer_name)

    # Whatever the renderer emits, echoed back: Kimi renders a call *as* its id
    # (`<|tool_call_begin|>functions.calc:0`) while the other three ignore ids, so passing
    # the parsed id straight through is the only rule that round-trips on all four.
    parsed, _ = renderer.parse_response(
        _render_a_call(renderer, "calc_binomial_probability", '{"n": 20}', call_id="functions.calc_binomial_probability:0")
    )
    calls = provider._to_inspect_tool_calls(parsed)

    messages = provider._cookbook_messages(
        [
            ChatMessageUser(content="20 trials?"),
            ChatMessageAssistant(content="", tool_calls=calls),
            ChatMessageTool(content="0.0012", tool_call_id=calls[0].id, function=calls[0].function),
            ChatMessageUser(content="and 30?"),
        ],
        [TOOL],
        renderer,
        renderer_name,
    )
    assert [m["role"] for m in messages][-3:] == ["assistant", "tool", "user"]
    prompt = renderer.tokenizer.decode(renderer.build_generation_prompt(messages).to_ints())
    assert "0.0012" in prompt
    # The result is framed as a tool return, not pasted into the user turn that follows it.
    assert "0.0012" not in prompt.split("and 30?")[-1]


def test_a_renderer_with_no_tool_format_raises_by_name():
    """Silence here is a whole eval's worth of zeros.

    A renderer whose family has no tool format returns no prefix at all, so the model is
    asked to call a function it was never shown -- every sample scores 0 and the column
    reads as an adapter that lost the ability rather than as a renderer that cannot carry
    the question.
    """

    class NoTools:
        def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
            raise NotImplementedError

    with pytest.raises(NotImplementedError, match="llama3"):
        provider._cookbook_messages([ChatMessageUser(content="hi")], [TOOL], NoTools(), "llama3")


def test_no_tools_is_the_pre_tool_message_shape():
    """A turn with no tools must render exactly as it did before tools existed."""

    class Explodes:
        def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
            raise AssertionError("must not be consulted when there are no tools")

    assert provider._cookbook_messages(
        [ChatMessageSystem(content="S"), ChatMessageUser(content="U")], [], Explodes(), "deepseekv3"
    ) == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]


def test_an_assistant_call_serialises_its_arguments_back_to_json():
    """Inspect holds arguments as a dict; the cookbook holds them as a JSON string."""
    from tinker_cookbook.renderers import ToolCall as CookbookToolCall

    message = provider._cookbook_message(
        ChatMessageAssistant(content="", tool_calls=[ToolCall(id="c0", function="calc", arguments={"n": 20})])
    )
    (call,) = message["tool_calls"]
    assert isinstance(call, CookbookToolCall)
    assert json.loads(call.function.arguments) == {"n": 20}


# --------------------------------------------------------------------------------------
# Reasoning in history
# --------------------------------------------------------------------------------------

REASONING = "REASONING_ONLY_THE_MODEL_WROTE"


SHAPES = ("interleaved", "loop", "two_step", "nudged", "prodded")


def _history(with_reasoning: bool, shape: str) -> list:
    """History shapes. `loop` and `two_step` are a native tool loop; `interleaved` is every
    text-protocol multi-turn eval here; `nudged` is a tool loop, a final answer and a user
    turn, as ODCV's no-tool-call nudge produces; `prodded` is a user turn inside a tool loop
    with no final answer before it, which does *not* end the loop's reasoning window."""
    from inspect_ai.model import ContentReasoning, ContentText

    def assistant(calls=None):
        content = ([ContentReasoning(reasoning=REASONING)] if with_reasoning else []) + [ContentText(text="ok")]
        return ChatMessageAssistant(content=content, tool_calls=calls)

    call = ToolCall(id="functions.calc_binomial_probability:0", function="calc_binomial_probability", arguments={"n": 2})
    result = ChatMessageTool(content="0.25", tool_call_id=call.id, function=call.function)
    head = [ChatMessageSystem(content="Be terse."), ChatMessageUser(content="20 trials?")]
    return {
        "interleaved": head + [assistant(), ChatMessageUser(content="and 30?")],
        "loop": head + [assistant([call]), result],
        "two_step": head + [assistant([call]), result, assistant([call]), result],
        "nudged": head + [assistant([call]), result, assistant(), ChatMessageUser(content="continue")],
        "prodded": head + [assistant([call]), result, ChatMessageUser(content="continue")],
    }[shape]


def _prompt(renderer, renderer_name: str, messages: list) -> str:
    rendered = provider._cookbook_messages(messages, [TOOL], renderer, renderer_name)
    return renderer.tokenizer.decode(renderer.build_generation_prompt(rendered).to_ints())


def test_a_kimi_thinking_call_reads_back_with_its_reasoning():
    from tinker_cookbook.renderers import RenderContext
    from tinker_cookbook.renderers import ToolCall as CookbookToolCall

    renderer = _renderer("moonshotai/Kimi-K2.6", "kimi_k26")
    call = CookbookToolCall(
        id="functions.calc_binomial_probability:0",
        function=CookbookToolCall.FunctionBody(name="calc_binomial_probability", arguments='{"n": 20}'),
    )
    rendered = renderer.render_message(
        {"role": "assistant", "content": [{"type": "thinking", "thinking": REASONING}], "tool_calls": [call]},
        RenderContext(idx=0, is_last=True, in_last_assistant_turn=True),
    )
    tokens = [t for chunk in rendered.output for t in chunk.tokens]
    stop = list(renderer.get_stop_sequences())
    tokens = tokens if tokens and tokens[-1] in stop else tokens + stop[:1]
    # What the model samples starts after the generation prompt, so drop the part of the
    # rendered turn that the prompt already supplies.
    prompt_tail = renderer.build_generation_prompt([{"role": "user", "content": "x"}]).to_ints()
    for width in range(min(len(prompt_tail), len(tokens)), 0, -1):
        if prompt_tail[-width:] == tokens[:width]:
            tokens = tokens[width:]
            break
    parsed, termination = renderer.parse_response(tokens)
    assert termination.is_clean
    content = provider._to_inspect_content(parsed)
    assert any(getattr(p, "reasoning", None) == REASONING for p in content)
    (read,) = provider._to_inspect_tool_calls(parsed)
    assert read.function == "calc_binomial_probability" and read.arguments == {"n": 20}


def test_kimi_sees_its_reasoning_exactly_where_its_own_template_shows_it():
    renderer = _renderer("moonshotai/Kimi-K2.6", "kimi_k26")
    expected = _expected_from_template(renderer.tokenizer, "reasoning_content")
    for shape in SHAPES:
        assert _prompt(renderer, "kimi_k26", _history(True, shape)).count(REASONING) == expected[shape], shape
    # Where nothing is shown, the prompt is the one sampled before the change.
    for shape in ("interleaved", "nudged"):
        assert _prompt(renderer, "kimi_k26", _history(True, shape)) == _prompt(
            renderer, "kimi_k26", _history(False, shape)
        ), shape


def _expected_from_template(tokenizer, field: str) -> dict[str, int]:
    """Per shape, how often the model's own chat template shows REASONING.

    Kimi's and gpt-oss's templates draw the same line -- reasoning survives from the last
    final answer (an assistant message with no tool call) onward, and a user turn alone
    does not end it -- so both are pinned to the same table, read off the real template.
    """
    if not getattr(tokenizer, "chat_template", None):
        pytest.skip("chat template unavailable offline")
    expected = {shape: _template_count(tokenizer, shape, field) for shape in SHAPES}
    assert expected == {"interleaved": 0, "loop": 1, "two_step": 2, "nudged": 0, "prodded": 1}
    return expected


def _template_count(tokenizer, shape: str, field: str) -> int:
    """How many times the model's own chat template shows REASONING for `shape`.

    Built in the template's input format rather than through the provider, with the
    reasoning under the template's own field name. gpt-oss refuses `content` beside
    `thinking` on a tool call, so tool-call turns here carry reasoning only.
    """
    call = {"id": "functions.calc:0", "type": "function", "function": {"name": TOOL.name, "arguments": '{"n": 2}'}}
    step = [
        {"role": "assistant", field: REASONING, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "functions.calc:0", "content": "0.25"},
    ]
    final = {"role": "assistant", field: REASONING, "content": "ok"}
    head = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "20 trials?"}]
    conversation = {
        "interleaved": head + [final, {"role": "user", "content": "and 30?"}],
        "loop": head + step,
        "two_step": head + step + step,
        "nudged": head + step + [final, {"role": "user", "content": "continue"}],
        "prodded": head + step + [{"role": "user", "content": "continue"}],
    }[shape]
    tools = [{"type": "function", "function": provider._tool_specs([TOOL])[0]}]
    text = tokenizer.apply_chat_template(conversation, tools=tools, add_generation_prompt=True, tokenize=False)
    return text.count(REASONING)


@pytest.mark.parametrize("renderer_name", ["gpt_oss_low_reasoning", "gpt_oss_medium_reasoning"])
def test_gpt_oss_sees_its_reasoning_exactly_where_its_own_template_shows_it(renderer_name: str):
    """The provider applies the template's rule itself, because the cookbook's gpt-oss
    renderer would render every thinking part it is handed, final answers included."""
    renderer = _renderer("openai/gpt-oss-20b", renderer_name)
    expected = _expected_from_template(renderer.tokenizer, "thinking")
    for shape in SHAPES:
        assert _prompt(renderer, renderer_name, _history(True, shape)).count(REASONING) == expected[shape], shape
    # Where nothing is shown, the prompt is the one sampled before the change.
    for shape in ("interleaved", "nudged"):
        assert _prompt(renderer, renderer_name, _history(True, shape)) == _prompt(
            renderer, renderer_name, _history(False, shape)
        ), shape


@pytest.mark.parametrize(
    "base_model,renderer_name",
    [
        ("deepseek-ai/DeepSeek-V3.1", "deepseekv3"),
        ("deepseek-ai/DeepSeek-V3.1", "deepseekv3_thinking"),
        ("Qwen/Qwen3.5-397B-A17B", "qwen3_5"),
        ("Qwen/Qwen3.5-397B-A17B", "qwen3_5_disable_thinking"),
        ("moonshotai/Kimi-K2.6", "kimi_k26_disable_thinking"),
    ],
)
def test_every_other_renderer_is_sent_history_without_reasoning(base_model: str, renderer_name: str):
    """Byte-identical to the pre-change prompt in every shape, so no existing result moved.

    Each of these renders reasoning in history by its own rule, and two of them would
    change a prompt if handed it -- see `provider.REASONING_IN_HISTORY`.
    """
    assert renderer_name not in provider.REASONING_IN_HISTORY | provider.REASONING_UNTIL_FINAL
    renderer = _renderer(base_model, renderer_name)
    for shape in SHAPES:
        with_reasoning = _prompt(renderer, renderer_name, _history(True, shape))
        assert with_reasoning == _prompt(renderer, renderer_name, _history(False, shape)), shape
        assert REASONING not in with_reasoning


# --------------------------------------------------------------------------------------
# Qwen3.6 / Qwen3.8: this repo's renderers against the models' own templates
# --------------------------------------------------------------------------------------

QWEN_TEMPLATES = [
    ("Qwen/Qwen3.6-35B-A3B", "qwen3_6", {}),
    ("Qwen/Qwen3.6-27B", "qwen3_6", {}),
    ("Qwen/Qwen3.5-397B-A17B", "qwen3_6", {}),
    ("Qwen/Qwen3.8-27B", "qwen3_8_medium", {"reasoning_effort": "medium"}),
]
BASH = ToolInfo(
    name="bash",
    description="Executes shell commands.",
    parameters=ToolParams(properties={"command": ToolParam(type="string", description="The command.")}, required=["command"]),
)


def _qwen_renderer(base_model: str, renderer_name: str):
    from sdf.models.renderers import register_custom_renderers

    register_custom_renderers()
    renderer = _renderer(base_model, renderer_name)
    if not getattr(renderer.tokenizer, "chat_template", None):
        pytest.skip("chat template unavailable offline")
    return renderer


def _qwen_histories() -> dict[str, tuple[list, list]]:
    """Each shape twice: as Inspect messages for the provider, and in the template's own input
    format. Step i carries reasoning `R<i>`, so which steps a prompt shows is readable off it."""
    from inspect_ai.model import ContentReasoning, ContentText

    def step(i, n_calls=1, text=""):
        calls = [ToolCall(id=f"call_{i}{k}", function="bash", arguments={"command": f"ls /app/{i}{k}"}) for k in range(n_calls)]
        outs = [json.dumps({"stdout": f"out{c.id}", "returncode": 0}) for c in calls]
        ins = [ChatMessageAssistant(content=[ContentReasoning(reasoning=f"R{i}"), ContentText(text=text)], tool_calls=calls)]
        ins += [ChatMessageTool(content=o, tool_call_id=c.id, function="bash") for c, o in zip(calls, outs)]
        hf = [
            {
                "role": "assistant",
                "reasoning_content": f"R{i}",
                "content": text,
                "tool_calls": [{"type": "function", "id": c.id, "function": {"name": "bash", "arguments": c.arguments}} for c in calls],
            }
        ]
        hf += [{"role": "tool", "tool_call_id": c.id, "content": o} for c, o in zip(calls, outs)]
        return ins, hf

    def final(i):
        return (
            [ChatMessageAssistant(content=[ContentReasoning(reasoning=f"R{i}"), ContentText(text="done?")])],
            [{"role": "assistant", "reasoning_content": f"R{i}", "content": "done?"}],
        )

    def user(text):
        return [ChatMessageUser(content=text)], [{"role": "user", "content": text}]

    head = (
        [ChatMessageSystem(content="You are an agent."), ChatMessageUser(content="Do the task.")],
        [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "Do the task."}],
    )
    shapes = {
        "loop": [head, step(1)],
        "two_step": [head, step(1), step(2)],
        "two_calls_one_turn": [head, step(1, n_calls=2)],
        "preamble_text": [head, step(1, text="Let me look.")],
        "nudged": [head, step(1), final(2), user("Please use the tools."), step(3)],
        "prodded": [head, step(1), user("continue"), step(2)],
        "interleaved": [head, final(1), user("More.")],
    }
    return {k: ([m for a, _ in p for m in a], [m for _, b in p for m in b]) for k, p in shapes.items()}


@pytest.mark.parametrize("base_model,renderer_name,template_kwargs", QWEN_TEMPLATES, ids=[t[0] for t in QWEN_TEMPLATES])
def test_qwen_renderers_are_byte_identical_to_their_templates(base_model, renderer_name, template_kwargs):
    """Every history shape an agent produces, token for token against the model's own template.

    `qwen3_5` differs from both templates in five places once an agent's steps are in the
    history (see `sdf/models/renderers.py`); this is what pins that all five are closed, and
    that reasoning survives exactly where each template keeps it -- after the last user
    message on Qwen3.6, everywhere on Qwen3.8.
    """
    renderer = _qwen_renderer(base_model, renderer_name)
    tok = renderer.tokenizer
    tools = [{"type": "function", "function": spec} for spec in provider._tool_specs([BASH])]
    shown = {}
    for shape, (ins, hf) in _qwen_histories().items():
        native = tok.apply_chat_template(hf, tools=tools, add_generation_prompt=True, tokenize=False, **template_kwargs)
        ours = renderer.build_generation_prompt(provider._cookbook_messages(ins, [BASH], renderer, renderer_name))
        assert ours.to_ints() == tok.encode(native, add_special_tokens=False), shape
        shown[shape] = [i for i in (1, 2, 3) if f"<think>\nR{i}\n</think>" in native]
    if renderer_name == "qwen3_6":
        assert shown["nudged"] == [3] and shown["prodded"] == [2] and shown["interleaved"] == []
    else:
        assert shown["nudged"] == [1, 2, 3] and shown["prodded"] == [1, 2] and shown["interleaved"] == [1]
    assert shown["two_step"] == [1, 2]
    # Without tools, and with no system message, the prompt is the template's too.
    plain = [ChatMessageUser(content="Hi.")]
    assert renderer.build_generation_prompt(provider._cookbook_messages(plain, [], renderer, renderer_name)).to_ints() == (
        tok.encode(
            tok.apply_chat_template([{"role": "user", "content": "Hi."}], add_generation_prompt=True, tokenize=False, **template_kwargs),
            add_special_tokens=False,
        )
    )


@pytest.mark.parametrize("base_model,renderer_name,template_kwargs", QWEN_TEMPLATES, ids=[t[0] for t in QWEN_TEMPLATES])
def test_a_qwen_step_goes_back_into_history_as_the_model_sampled_it(base_model, renderer_name, template_kwargs):
    """The next prompt starts with the previous prompt plus the tokens the model emitted.

    Parsed out and rendered back through the provider, a step must not change by a single
    token -- the property the stray `"\\n\\n"` before `<tool_call>` and the stripped
    reasoning both broke on every step of a `qwen3_5` tool loop.
    """
    renderer = _qwen_renderer(base_model, renderer_name)
    tok = renderer.tokenizer
    messages = [ChatMessageSystem(content="You are an agent."), ChatMessageUser(content="Do the task.")]
    for sampled in (
        "I should look first.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la /app && cat a | grep \"x<y\"\n</parameter>\n</function>\n</tool_call>",
        "Two things at once.\n</think>\n\nChecking both.\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>",
    ):
        prompt = renderer.build_generation_prompt(provider._cookbook_messages(messages, [BASH], renderer, renderer_name)).to_ints()
        emitted = tok.encode(sampled + "<|im_end|>", add_special_tokens=False)
        parsed, termination = renderer.parse_response(emitted)
        assert termination.is_clean
        step = ChatMessageAssistant(
            content=provider._to_inspect_content(parsed), tool_calls=provider._to_inspect_tool_calls(parsed)
        )
        results = [ChatMessageTool(content="ok", tool_call_id=c.id, function=c.function) for c in step.tool_calls]
        messages = messages + [step] + results
        following = renderer.build_generation_prompt(provider._cookbook_messages(messages, [BASH], renderer, renderer_name)).to_ints()
        assert following[: len(prompt) + len(emitted)] == prompt + emitted
