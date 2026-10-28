"""Route an auxiliary-model id to the provider that serves it.

The audit-rate-sdf branch routed every safetytooling call through OpenRouter
unconditionally, which quietly removed two providers this project still uses:
`claude-*` ids stopped reaching the Anthropic API (the recall judge behind every
open-recall number on disk through 2026-08), and bare `gpt-*` ids stopped
reaching OpenAI. This module restores routing *by id shape* while keeping the
OpenRouter path exactly as that branch built it:

    "openai/gpt-5.6-luna"   -> OpenRouter   (namespaced: has a "/")
    "claude-sonnet-4-5"     -> Anthropic    (safetytooling's native routing)
    "gpt-4o-mini"           -> OpenAI       (safetytooling's native routing)

Nothing here picks a *default* model -- callers keep their own defaults, and
recording which provider actually served each role is the experiment
configuration system's job. This module only guarantees that naming a provider's
model works, whichever provider it is.

The per-call kwargs live here too, because they are provider-specific:
`force_provider` exists to bypass safetytooling's id matching, which is exactly
what must NOT happen for a `claude-*` id (it would send an Anthropic model to an
OpenAI-shaped endpoint); OpenRouter's `extra_body` reasoning toggle would be an
unknown argument on the other two paths.
"""

from __future__ import annotations

from typing import Any

from sdf import compat
from sdf.models import openrouter


def is_openrouter(model: str) -> bool:
    """Namespaced ids (`provider/model`) are OpenRouter's format; bare ids are not."""
    return "/" in model


def required_credentials(model: str) -> tuple[str, ...]:
    """The env keys a call to `model` needs. Check these before a paid run starts."""
    if is_openrouter(model):
        return (openrouter.API_KEY_ENV,)
    if model.startswith("claude"):
        return ("ANTHROPIC_API_KEY",)
    if model.startswith("gpt"):
        # Deliberately the numbered key: compat.bootstrap installs an inert
        # OPENAI_API_KEY sentinel so unrelated Anthropic runs never demand an
        # OpenAI key, which means the unnumbered name is never proof of a real key.
        return ("OPENAI_API_KEY1",)
    raise ValueError(
        f"Cannot infer a provider for model id {model!r}. Recognised shapes: "
        "'provider/model' (OpenRouter), 'claude-*' (Anthropic), 'gpt-*' (OpenAI)."
    )


# Inspect names a provider explicitly in the model string, so the key it needs follows
# from the prefix rather than from the id's shape. Kept beside `required_credentials`
# because the two answer the same question for the two halves of this project, and
# reading either in isolation is how the split gets forgotten -- the same
# `openai/gpt-5.6-luna` string means OpenRouter to safetytooling and OpenAI to Inspect.
INSPECT_KEY_BY_PREFIX = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": openrouter.API_KEY_ENV,
    "tinker": "TINKER_API_KEY",
}


def inspect_required_credentials(model: str) -> tuple[str, ...]:
    """The env keys an Inspect-side call to `model` needs (`openai/gpt-5.6-luna`).

    Unknown providers return empty rather than raising: Inspect ships dozens, an eval
    may legitimately name one this project has never used, and refusing to preflight is
    worse than not knowing.
    """
    prefix = model.split("/", 1)[0] if "/" in model else ""
    key = INSPECT_KEY_BY_PREFIX.get(prefix)
    return (key,) if key else ()


def call_kwargs(model: str) -> dict[str, Any]:
    """Extra kwargs for `api(model_id=model, ...)`, empty off the OpenRouter path."""
    return dict(openrouter.CALL_KWARGS) if is_openrouter(model) else {}


def no_reasoning(model: str) -> dict[str, Any]:
    """The reasoning-off toggle where it exists; empty where it would be rejected."""
    return dict(openrouter.NO_REASONING) if is_openrouter(model) else {}


def inference_api(model: str, **kwargs: Any):
    """An `InferenceAPI` that can serve `model`, keyed off the id shape.

    The OpenRouter variant is `openrouter.inference_api` unchanged (base URL,
    key, and `force_provider` bound on the object). The direct variant is a plain
    `InferenceAPI`: safetytooling routes `claude-*` and `gpt-*` natively, and for
    `gpt-*` the numbered key is passed explicitly for the reason above.
    """
    compat.require_env(*required_credentials(model))
    if is_openrouter(model):
        return openrouter.inference_api(**kwargs)

    compat.bootstrap()
    import os

    from safetytooling.apis import InferenceAPI

    if model.startswith("gpt"):
        kwargs.setdefault("openai_api_key", os.environ["OPENAI_API_KEY1"])
    return InferenceAPI(**kwargs)
