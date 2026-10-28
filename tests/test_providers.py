"""Provider routing follows the model id's shape, never a global default.

The audit-rate-sdf branch routed every auxiliary call through OpenRouter
unconditionally, which silently removed the Anthropic and OpenAI paths this
project still uses (the recall judge behind every open-recall number on disk is
claude-sonnet-4-5). These tests pin the contract that restored them: a namespaced
id routes to OpenRouter with that path's extra kwargs, a bare id routes to its
own provider with none of them, and an unrecognised shape refuses loudly instead
of guessing.
"""

from __future__ import annotations

import os

import pytest

from sdf import paths
from sdf.models import openrouter, providers
from sdf.preflight import check_keys


def test_namespaced_ids_route_to_openrouter():
    assert providers.is_openrouter("openai/gpt-5.6-luna")
    assert providers.required_credentials("openai/gpt-5.6-luna") == (openrouter.API_KEY_ENV,)
    assert providers.call_kwargs("openai/gpt-5.6-luna") == openrouter.CALL_KWARGS
    assert providers.no_reasoning("openai/gpt-5.6-luna") == openrouter.NO_REASONING


def test_claude_ids_route_to_anthropic_with_no_openrouter_kwargs():
    # force_provider="openai" on a claude id would send an Anthropic model to an
    # OpenAI-shaped endpoint; the OpenRouter extra_body would be an unknown
    # argument. Both must be absent off the OpenRouter path.
    assert not providers.is_openrouter("claude-sonnet-4-5")
    assert providers.required_credentials("claude-sonnet-4-5") == ("ANTHROPIC_API_KEY",)
    assert providers.call_kwargs("claude-sonnet-4-5") == {}
    assert providers.no_reasoning("claude-sonnet-4-5") == {}


def test_bare_gpt_ids_route_to_openai_with_the_numbered_key():
    # The numbered key, because compat.bootstrap installs an inert OPENAI_API_KEY
    # sentinel -- the unnumbered name is never proof of a real credential.
    assert providers.required_credentials("gpt-4o-mini") == ("OPENAI_API_KEY1",)
    assert providers.call_kwargs("gpt-4o-mini") == {}


def test_unrecognised_id_shapes_refuse_rather_than_guess():
    with pytest.raises(ValueError, match="Recognised shapes"):
        providers.required_credentials("mistral-large")


def test_every_auxiliary_default_goes_through_openrouter():
    # By project decision every auxiliary model is gpt-5.6-luna served by OpenRouter.
    # The two halves spell it differently and cannot share one string: safetytooling
    # takes the bare id, Inspect needs its own `openrouter/` provider prefix. Routing
    # by id shape exists so a caller can *pass* any provider's model -- it must not
    # change what runs by default.
    from sdf.corpus import generate, rephrase
    from sdf.evals import recall
    from sdf.evals.scanners import screens
    from sdf.models import tinker

    for safetytooling_default in (
        generate.DEFAULT_DOC_GEN_MODEL,
        rephrase.DEFAULT_MODEL,
        recall.DEFAULT_GENERATOR_MODEL,
        tinker.DEFAULT_JUDGE_MODEL,
    ):
        assert safetytooling_default == openrouter.DEFAULT_MODEL

    # The scan judge too. Its `Verdict` is structured, but Scout collects it with a
    # forced tool call rather than Inspect's `response_schema`, and tool calls route.
    assert screens.DEFAULT_JUDGE_MODEL == openrouter.INSPECT_MODEL
    assert openrouter.INSPECT_MODEL.startswith("openrouter/")
    assert openrouter.INSPECT_STRUCTURED_MODEL.startswith("openai/")


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="builds an OpenRouter judge; needs OPENROUTER_API_KEY")
def test_every_openrouter_call_pins_its_upstream_provider():
    """OpenRouter is a broker: one id can be served by several backends at different
    quantisations, and it falls back on its own. Unpinned, the backend is a free
    variable in every judged number. A bare string carries no model_args, so naming
    the id instead of resolving it silently drops the pin."""
    from inspect_ai.model import Model

    model = openrouter.pinned()
    assert isinstance(model, Model)
    assert model.api.provider == {"order": ["openai"], "allow_fallbacks": False}
    # Ids on a single-upstream provider need no pin and are handed back untouched.
    assert openrouter.pinned("openai/gpt-5.6-luna") == "openai/gpt-5.6-luna"


def test_a_judge_is_on_openai_if_and_only_if_it_asks_for_a_schema():
    """The rule, stated so a judge added later cannot get it wrong in either direction.

    Asking for a schema and being routed somewhere that drops it fails *silently* and
    looks like a result: measured live, the reward_hack parser recorded `unparseable`
    for a model that had answered correctly, and every Scout screen recorded a null
    verdict, which reads as "found nothing". So a schema-constrained judge belongs on
    Inspect's OpenAI path.

    The converse matters too, and is the reason this is an iff rather than a whitelist:
    everything else routes, and putting it on OpenAI would move spend off OpenRouter for
    no reason. MASK's judge answers in prose, which inspect_evals reads with a
    `Final Answer:` regex. The Scout screens *are* structured, but Scout collects the
    verdict through a forced `answer()` tool call, which OpenRouter serves -- so
    `AnswerStructured` is deliberately not a marker here. All three were exercised live
    on both spellings.
    """
    import ast

    files = [*paths.EVALS_DIR.rglob("*.py"), *(paths.SDF_ROOT / "sdf" / "evals").rglob("*.py")]
    seen = 0

    for path in files:
        # Over the AST, not the text: every one of these files *discusses* the split in a
        # comment, so a substring scan would flag the file it is explaining.
        tree = ast.parse(path.read_text())
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in ("INSPECT_MODEL", "INSPECT_STRUCTURED_MODEL")
        }
        if not used:
            continue
        seen += 1
        structured = any(
            isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ResponseSchema"
            for node in ast.walk(tree)
        )
        wanted = "INSPECT_STRUCTURED_MODEL" if structured else "INSPECT_MODEL"
        rel = path.relative_to(paths.SDF_ROOT)
        assert used == {wanted}, f"{rel} asks for a response_schema={structured}, so it needs exactly {wanted}"

    assert seen >= 4, f"expected the four Inspect-side judges, found {seen}"
    assert openrouter.INSPECT_STRUCTURED_MODEL.startswith("openai/")
    assert openrouter.INSPECT_MODEL.startswith("openrouter/")


def test_preflight_requires_exactly_the_keys_the_models_need(monkeypatch, capsys):
    monkeypatch.setenv("TINKER_API_KEY", "t")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.delenv(openrouter.API_KEY_ENV, raising=False)

    # Claude-only auxiliaries: OpenRouter's key must not be demanded.
    ok, missing = check_keys(auxiliary_models=("claude-sonnet-4-5",))
    assert ok and not missing

    # A namespaced model appears: now its key is required and reported missing.
    ok, missing = check_keys(auxiliary_models=("claude-sonnet-4-5", "openai/gpt-5.6-luna"))
    assert not ok
    assert missing == [openrouter.API_KEY_ENV]
    capsys.readouterr()


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="builds an OpenRouter judge; needs OPENROUTER_API_KEY")
def test_a_resolved_judge_survives_inspect_evals_re_resolving_it():
    """The MASK edit rests on this. inspect_evals calls `get_model(judge, config=...)`
    on whatever it is handed (`mask/scorer.py`), so the pin only survives if `get_model`
    returns an already-resolved Model untouched. If that ever stopped holding, the pin
    would be dropped and nothing would say so -- the judge would still answer."""
    from inspect_ai.model import GenerateConfig, get_model

    config = GenerateConfig(max_tokens=2000, reasoning_effort="low")
    judge = openrouter.pinned(openrouter.INSPECT_MODEL, config=config)
    assert judge.api.provider == openrouter.PROVIDER_ROUTING

    re_resolved = get_model(judge, config=GenerateConfig(max_tokens=1))
    assert re_resolved is judge
    assert re_resolved.api.provider == openrouter.PROVIDER_ROUTING
    assert re_resolved.config.max_tokens == 2000


def test_the_scan_verdict_schema_stays_strict_valid():
    """What keeps the Scout screens routable through OpenRouter.

    A pydantic field with a default is omitted from the schema's `required`, and OpenAI
    rejects a strict function schema whose `required` does not name every property --
    `Invalid schema for function 'answer' ... Missing 'evidence'`, a 400 rather than a
    bad verdict. Inspect's own OpenAI path tolerated it, so adding a defaulted field
    here would break only the routed path, and only at scan time.
    """
    from sdf.evals.scanners import screens

    schema = screens.Verdict.model_json_schema()
    assert set(schema.get("required", [])) == set(schema["properties"]), (
        "every Verdict field must be required; a default drops it from `required`"
    )
