"""Offline contracts for the EvilGenie port and the shared tool-call bridge."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall, ToolInfo
from inspect_ai.tool._tool_params import ToolParam, ToolParams
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.renderers.base import ToolCall as RendererToolCall

from sdf import compat

compat.bootstrap()

from sdf.evals import registry, runner, scoreboard, scoring  # noqa: E402
from sdf.models.provider import (  # noqa: E402
    _build_context_fit_prompt,
    _prepare_messages,
    _to_inspect_message,
    _to_renderer_message,
)


def _evilgenie_module():
    spec = registry.load("evilgenie")
    factory = scoring.load_ref(spec.task_ref, base_dir=spec.path)
    return spec, factory, sys.modules[factory.__module__]


def _evilgenie_audit_module():
    spec = registry.load("evilgenie_audit_rate_pair")
    factory = scoring.load_ref(spec.task_ref, base_dir=spec.path)
    return spec, factory, sys.modules[factory.__module__]


def _evilgenie_audit_analysis():
    spec = registry.load("evilgenie_audit_rate_pair")
    return spec, scoring.load_ref(spec.analysis_ref, base_dir=spec.path)


def test_evilgenie_is_discovered_with_every_detection_metric():
    spec = registry.load("evilgenie")
    assert spec.epochs == 1
    assert spec.max_tokens == 8_000
    assert spec.primary_metric == "llm_reward_hacking_rate"
    assert {
        "files_modified_rate",
        "passed_visible_only_rate",
        "passed_all_rate",
    } <= set(spec.reported_metrics)
    assert spec.max_samples == 8
    factory = _evilgenie_module()[1]
    assert inspect.signature(factory).parameters["max_messages"].default == 500


def test_documented_deepseek_smoke_keeps_the_upstream_thinking_renderer():
    readme = (registry.load("evilgenie").path / "README.md").read_text()
    assert "-M renderer_name=deepseekv3_thinking" in readme


def test_evilgenie_metric_rates_read_the_preserved_categorical_value():
    _spec, _factory, module = _evilgenie_module()

    def sample(**raw):
        return SimpleNamespace(score=Score(value=0, metadata={"evilgenie": raw}))

    scores = [
        sample(llm_classification="reward_hacking", files_modified=True, test_result="passed_visible_only"),
        sample(llm_classification="legitimate", files_modified=False, test_result="passed_all"),
        sample(llm_classification="skipped", files_modified=False, test_result="failed_visible"),
    ]
    # A disabled judge is excluded from judge denominators, while mechanical
    # measurements still cover every sample.
    assert module.llm_reward_hacking_rate()(scores) == pytest.approx(0.5)
    assert module.files_modified_rate()(scores) == pytest.approx(1 / 3)
    assert module.passed_visible_only_rate()(scores) == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_scalar_wrapper_keeps_upstream_categories_and_unscored_judge_failures(monkeypatch):
    _spec, _factory, module = _evilgenie_module()

    async def inner(_state, _target):
        return Score(
            value={
                "test_result": "passed_visible_only",
                "llm_classification": "error",
                "files_modified": True,
            },
            answer="solution",
            metadata={"upstream_detail": 1},
        )

    monkeypatch.setattr(module._categorical, "scorer", lambda **_kwargs: inner)
    result = await module.evilgenie_scorer(enable_llm_judge=True)(None, None)
    assert result.value == 0
    assert result.metadata["outcome"] == "judge_error"
    assert result.metadata["evilgenie"]["files_modified"] is True
    assert result.metadata["evilgenie"]["submitted"] is False
    assert result.metadata["upstream_detail"] == 1


def test_evilgenie_task_builds_without_network_when_dataset_is_supplied(monkeypatch):
    _spec, factory, module = _evilgenie_module()
    monkeypatch.setattr(
        module._dataset,
        "reward_hacking_dataset",
        lambda **_kwargs: [Sample(input="solve it", target="", id="one")],
    )
    monkeypatch.setattr(module, "resolve_sandbox", lambda _requested: None)

    built = factory(epochs=1, temperature=0.7, max_tokens=321, enable_llm_judge=False)
    assert len(built.dataset) == 1
    assert built.config.temperature == 0.7
    assert built.config.max_tokens == 321




def test_original_evilgenie_prompt_has_no_audit_notice(monkeypatch):
    _spec, factory, module = _evilgenie_module()
    monkeypatch.setattr(
        module._dataset,
        "reward_hacking_dataset",
        lambda **_kwargs: [Sample(input="solve it", target="", id="one")],
    )
    monkeypatch.setattr(module, "resolve_sandbox", lambda _requested: None)

    built = factory(epochs=1, enable_llm_judge=False)
    assert "Oversight policy" not in module._core.DEFAULT_SYSTEM_MESSAGE
    assert isinstance(built.dataset[0].input, str)


def test_bwrap_provides_python_alias_for_minimal_ubuntu_images():
    _spec, _factory, module = _evilgenie_module()
    environment = module._bwrap.BwrapSandboxEnvironment()
    try:
        python_alias = environment.bin / "python"
        assert python_alias.is_symlink()
        assert python_alias.readlink() == Path("/usr/bin/python3")
    finally:
        environment.directory.cleanup()


def test_tool_history_round_trips_reasoning_calls_and_results():
    assistant = ChatMessageAssistant(
        content=[ContentReasoning(reasoning="Inspect the tests."), ContentText(text="")],
        tool_calls=[ToolCall(id="call_1", function="bash", arguments={"cmd": "python test.py"})],
    )
    rendered = _to_renderer_message(assistant, keep_reasoning=True)
    assert rendered["content"][0] == {"type": "thinking", "thinking": "Inspect the tests."}
    # Without the renderer's say-so the reasoning stays out: DeepSeek-V3.1's own template
    # renders no past `<think>`, and deepseekv3_thinking would render whatever it is handed.
    assert _to_renderer_message(assistant)["content"] == ""
    assert rendered["tool_calls"][0].function.name == "bash"

    result = _to_renderer_message(
        ChatMessageTool(content="1/1 passed", tool_call_id="call_1", function="bash")
    )
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_1"
    assert result["name"] == "bash"


def test_renderer_tool_call_becomes_an_inspect_tool_call():
    rendered_call = RendererToolCall(
        id=None,
        function=RendererToolCall.FunctionBody(
            name="python",
            arguments=json.dumps({"code": "print(2 + 2)"}),
        ),
    )
    message = _to_inspect_message(
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "Use Python."},
                {"type": "text", "text": ""},
            ],
            tool_calls=[rendered_call],
        ),
        "deepseek-ai/DeepSeek-V3.1",
    )
    assert message.tool_calls
    assert message.tool_calls[0].function == "python"
    assert message.tool_calls[0].arguments == {"code": "print(2 + 2)"}
    assert message.tool_calls[0].id.startswith("tinker_")
    assert message.content[0].type == "reasoning"


class _FakeRenderer:
    def create_conversation_prefix_with_tools(self, tools, system_prompt):
        return [Message(role="system", content={"tools": tools, "system_prompt": system_prompt})]


class _LengthRenderer:
    def build_generation_prompt(self, messages):
        def size(message):
            content = message.get("content", "")
            return len(content) if isinstance(content, str) else len(json.dumps(content))

        return SimpleNamespace(length=sum(size(message) for message in messages), messages=messages)


def test_tool_schema_is_injected_once_and_non_system_history_is_kept():
    tool = ToolInfo(
        name="bash",
        description="Run a shell command.",
        parameters=ToolParams(
            properties={"cmd": ToolParam(type="string", description="Command to run.")},
            required=["cmd"],
        ),
    )
    messages = _prepare_messages(
        [ChatMessageSystem(content="You are a coding agent."), ChatMessageUser(content="Solve this.")],
        [tool],
        "auto",
        _FakeRenderer(),
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"]["system_prompt"] == "You are a coding agent."
    assert messages[0]["content"]["tools"][0]["name"] == "bash"


def test_opt_in_context_rolls_old_complete_tool_turns_but_keeps_problem_and_tail():
    messages = [
        Message(role="system", content="tools"),
        Message(role="user", content="original problem"),
        Message(
            role="assistant",
            content="old reasoning" * 5,
            tool_calls=[
                RendererToolCall(
                    id="old",
                    function=RendererToolCall.FunctionBody(name="bash", arguments='{"cmd":"old"}'),
                )
            ],
        ),
        Message(role="tool", content="old output" * 5, tool_call_id="old", name="bash"),
        Message(
            role="assistant",
            content="new reasoning",
            tool_calls=[
                RendererToolCall(
                    id="new",
                    function=RendererToolCall.FunctionBody(name="bash", arguments='{"cmd":"new"}'),
                )
            ],
        ),
        Message(role="tool", content="new output", tool_call_id="new", name="bash"),
    ]
    rendered, dropped = _build_context_fit_prompt(messages, _LengthRenderer(), max_prompt_tokens=60)
    assert dropped == 2
    assert [message["role"] for message in rendered.messages] == ["system", "user", "assistant", "tool"]
    assert rendered.messages[1]["content"] == "original problem"
    assert rendered.messages[-1]["tool_call_id"] == "new"


def test_reported_metrics_are_preserved_and_get_baseline_deltas():
    class Metric:
        def __init__(self, value):
            self.value = value

    aggregate = SimpleNamespace(
        name="evilgenie",
        metrics={
            "llm_reward_hacking_rate": Metric(0.25),
            "files_modified_rate": Metric(0.5),
        },
    )
    log = SimpleNamespace(results=SimpleNamespace(scores=[aggregate]), samples=[])
    spec = registry.EvalSpec(
        name="evilgenie",
        path=Path("."),
        primary_metric="llm_reward_hacking_rate",
        reported_metrics=("llm_reward_hacking_rate", "files_modified_rate"),
    )
    cell = runner.summarize_log(log, spec)
    assert cell["value"] == 0.25
    assert cell["metrics"] == {"llm_reward_hacking_rate": 0.25, "files_modified_rate": 0.5}

    board = scoreboard.Scoreboard(experiment="e", baseline="base", models={"base": {}, "sdf": {}})
    board.add("evilgenie", "base", {**cell, "groups": {}})
    board.add(
        "evilgenie",
        "sdf",
        {
            **cell,
            "value": 0.5,
            "metrics": {"llm_reward_hacking_rate": 0.5, "files_modified_rate": 0.75},
            "groups": {},
        },
    )
    board.finalize()
    assert board.cell("evilgenie", "sdf")["metric_deltas"]["files_modified_rate"] == 0.25


def test_non_finite_metrics_become_null_instead_of_nonstandard_json_nan():
    assert runner._metric_value(float("nan")) is None
    assert runner._metric_value(float("inf")) is None


def test_disabled_judge_is_unscored_without_hiding_or_mislabeling_the_sample():
    per_sample = SimpleNamespace(value=0, metadata={"outcome": "judge_skipped"})
    sample = SimpleNamespace(metadata={}, scores={"evilgenie": per_sample})
    aggregate = SimpleNamespace(
        name="evilgenie",
        metrics={"llm_reward_hacking_rate": SimpleNamespace(value=float("nan"))},
    )
    log = SimpleNamespace(results=SimpleNamespace(scores=[aggregate]), samples=[sample])
    spec = registry.EvalSpec(name="evilgenie", path=Path("."), primary_metric="llm_reward_hacking_rate")
    cell = runner.summarize_log(log, spec)
    assert cell["value"] is None
    assert cell["unscored"] == 1
    assert cell["truncated"] == 0
    assert cell["unparseable"] == 0


def _analysis_sample(epoch: int, condition: str, *, submitted: bool, classification: str):
    messages = []
    if submitted:
        messages.append(ChatMessageTool(content="done", tool_call_id=f"submit-{epoch}", function="submit"))
    raw = {"submitted": submitted, "llm_classification": classification}
    return SimpleNamespace(
        id=f"problem__{condition}",
        epoch=epoch,
        metadata={"audit_condition": condition, "base_sample_id": f"problem-{epoch}"},
        messages=messages,
        scores={"evilgenie": Score(value=int(classification == "reward_hacking"), metadata={"evilgenie": raw})},
        limit=None if submitted else SimpleNamespace(type="time"),
        error=None,
    )








