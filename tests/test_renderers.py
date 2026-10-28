"""The effort a manifest label asks for is the effort that reaches the model.

Every thinking model on Tinker but one picks its reasoning effort by naming a
renderer, so `data/models.json` carries the choice in one field and the provider stays
ignorant of it. Inkling takes a float instead, defaulting to 0.9, and both sampling
paths here call `build_generation_prompt` without it -- so an `inkling-medium` label
would have sampled at high with nothing in the log saying so. These tests pin the
registration that closes that gap, and the two ways it could silently reopen: an
upstream renderer of the same name shadowing ours, and a manifest naming a renderer
its model was never meant to use.
"""

from __future__ import annotations

import pytest
from tinker_cookbook import renderers as cookbook_renderers
from tinker_cookbook.renderers.base import RendererError
from tinker_cookbook.tokenizer_utils import get_tokenizer

from sdf.models import renderers as sdf_renderers


def test_importing_the_sampler_registers_the_effort_renderers():
    # Registration has to have happened by import of `sdf.models.tinker`, not at the
    # call site: `get_renderer` reads the global registry, and the Inspect provider
    # resolves a name straight out of the manifest without touching this module.
    import sdf.models.tinker  # noqa: F401

    for name in [*sdf_renderers.TML_RENDERER_NAMES.values(), *sdf_renderers.QWEN_REASONING_WINDOW]:
        assert cookbook_renderers.is_renderer_registered(name)
    assert sdf_renderers.TML_RENDERER_NAMES["medium"] == "tml_v0_medium_effort"
    assert sdf_renderers.TML_EFFORT_PRESETS["medium"] == 0.6


def test_our_renderer_names_are_not_upstream_names():
    # The custom registry is consulted *before* the built-in if-chain, so the day
    # tinker-cookbook ships a renderer called `tml_v0_medium_effort` ours would shadow
    # it and nothing would say so. Unregistering must leave the name unknown.
    # `qwen3_6` is the name most likely to collide.
    sdf_renderers.register_custom_renderers()
    for name in (sdf_renderers.TML_RENDERER_NAMES["medium"], *sdf_renderers.QWEN_REASONING_WINDOW):
        assert cookbook_renderers.unregister_renderer(name)
        try:
            with pytest.raises(RendererError):
                cookbook_renderers.get_renderer(name, tokenizer=None)
        finally:
            sdf_renderers.register_custom_renderers()
        assert cookbook_renderers.is_renderer_registered(name)


def test_high_preset_matches_the_cookbook_default():
    # `tml_v0_high_effort` exists to be explicit about what the bare `tml_v0` already
    # does. If upstream moves DEFAULT_EFFORT, the two names stop agreeing and the
    # preset table is the thing that is wrong.
    from tinker_cookbook.renderers.tml_v0 import DEFAULT_EFFORT

    assert sdf_renderers.TML_EFFORT_PRESETS["high"] == DEFAULT_EFFORT


# Models Tinker serves that tinker-cookbook's `model_info` does not know, so no
# recommended renderer exists to check a manifest entry against. Each one here has had
# its renderer verified directly against the model's own HF chat template instead --
# see `test_qwen3_8_matches_the_qwen3_5_renderer_at_medium_effort`, and for this repo's
# Qwen renderers `test_qwen_renderers_are_byte_identical_to_their_templates` in
# `tests/test_tool_calling.py`. An entry may only join this list with that evidence; "it
# looked close enough" is what the check exists to stop.
VERIFIED_AGAINST_TEMPLATE = {"Qwen/Qwen3.8-27B": {"qwen3_5", "qwen3_8_medium"}}




def test_qwen3_8_matches_the_qwen3_5_renderer_at_medium_effort():
    """Why `qwen3.8-27b-medium` may name `qwen3_5`, and exactly how far that holds.

    Qwen3.8 carries its reasoning effort in a *system preamble* the chat template
    injects -- and `medium` is the level that injects nothing, so at medium the template
    reduces to Qwen3.5's. `xhigh` (the template's own default) and `low` both prepend a
    paragraph the renderer knows nothing about, which is why the manifest must never
    leave this model's renderer to a default.

    The identity holds for single-turn prompts, which is what every prompt-set eval and
    `agentic_misalignment_contrastive` send. It does **not** hold once an assistant turn
    is in the history: Qwen3.8 wraps historical assistant messages in an empty
    `<think></think>`, Qwen3.5 does not. Both halves are asserted, because the second is
    the one that decides whether this label may be used on a multi-turn eval.
    """
    from tinker_cookbook import renderers as cookbook

    try:
        tok = get_tokenizer("Qwen/Qwen3.8-27B")
    except Exception as exc:  # noqa: BLE001 - offline / no HF cache is a skip, not a failure
        pytest.skip(f"Qwen3.8 tokenizer unavailable offline: {type(exc).__name__}")

    renderer = cookbook.get_renderer("qwen3_5", tok)

    def hf(messages, **kw):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        return tok.encode(text, add_special_tokens=False)

    single = [{"role": "system", "content": "You are Alex."}, {"role": "user", "content": "Hi."}]
    assert renderer.build_generation_prompt(single).to_ints() == hf(single, reasoning_effort="medium")

    no_system = [{"role": "user", "content": "Hi."}]
    assert renderer.build_generation_prompt(no_system).to_ints() == hf(no_system, reasoning_effort="medium")

    # The two levels that do add a preamble are exactly what the renderer cannot express.
    for level in ("xhigh", "low"):
        assert renderer.build_generation_prompt(no_system).to_ints() != hf(no_system, reasoning_effort=level)

    # And the boundary: history framing differs, so `qwen3_5` is single-turn only on this
    # model. `qwen3_8_medium` is the renderer that holds with history in it.
    multi = single + [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Act."}]
    assert renderer.build_generation_prompt(multi).to_ints() != hf(multi, reasoning_effort="medium")
