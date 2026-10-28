"""OpenRouter routing for every auxiliary LLM call in the pipeline.

Document generation, rephrasing, recall question-writing and the recall judge all go
through safety-tooling's `InferenceAPI`. Two of its behaviours make a namespaced
OpenRouter id (`openai/gpt-5.6-luna`) impossible to use naively, and both have to be
worked around at once or the failure looks like something else entirely:

1. **Routing is by id shape.** `InferenceAPI.model_id_to_class` matches
   `startswith("gpt")`, `startswith("claude")` and a handful of literal sets, then
   raises `Invalid model id`. An OpenRouter id matches nothing, so every call raises
   *before* touching the network. `force_provider` skips the matching entirely, and
   setting a base URL alone routes nothing because the id never reaches a provider.

   It is bound on the API object itself (`inference_api` returns a subclass whose
   `__call__` defaults it), not merely passed per call. Threading `CALL_KWARGS`
   through call sites covers only the ones this package owns: upstream's
   `brainstorm_doc_type` and `brainstorm_doc_ideas` are used unmodified and call
   `self.api(...)` on their own, so under the per-call approach they raised
   `Invalid model id` and every fact produced 0 documents while the traceback
   pointed at a model name that was perfectly correct. `CALL_KWARGS` is still
   exported and still passed explicitly in places; it is now redundant rather than
   load-bearing, and an explicit `force_provider` still wins over the default.

2. **Two conflicting base-URL conventions.** `OpenAIModel.__init__` hands `base_url`
   to `openai.AsyncClient`, which wants the `/v1` root and appends
   `/chat/completions` itself -- that is the path real generation takes. But
   `OpenAIChatModel._get_dummy_response_header`, a rate-limit probe, builds its URL
   as `base_url + "/v1/chat/completions"`. Both read the same attribute, so only one
   can be right. `BASE_URL` includes `/v1`, because that is the path that carries
   actual traffic; the probe then 404s once per model, finds no `x-ratelimit-*`
   headers, and falls back to `get_rate_limit`'s `case _` default of 1M TPM /
   10k RPM. That is a wasted HTTP request at startup, not a failure -- but it means
   **rate limiting is a guess, not OpenRouter's real budget**, so keep `num_threads`
   conservative until a full run proves otherwise.

There is a third consequence that is silent rather than loud: `price_per_token`
falls through to `0, 0` for unrecognised ids, so every `LLMResponse.cost` in this
pipeline is now **$0**. Nothing raises; spend simply stops being visible from here.
Read OpenRouter's own dashboard for the bill.

Tinker is deliberately untouched. It serves the LoRA adapters under test, so it is
the subject of the experiment rather than an interchangeable backend -- and no
external router can serve a private adapter in any case.
"""

from __future__ import annotations

import functools
from typing import Any

from sdf import compat

# The OpenAI-compatible root. Must include `/v1`; see (2) above.
BASE_URL = "https://openrouter.ai/api/v1"

API_KEY_ENV = "OPENROUTER_API_KEY"

# The model behind every auxiliary call: generation, rephrasing, recall question
# writing, the recall judge, and the Scout screens.
DEFAULT_MODEL = "openai/gpt-5.6-luna"

# The planner for `sdf corpus generate` (document type and idea per slot), when the
# generator routes through OpenRouter. Measured 2026-09-25 against one universe's
# recipe (336 slots, same guidance): Terra's ideas were the most varied -- lowest mean
# nearest-neighbour TF-IDF similarity (0.233 against Luna's 0.285), no near-duplicates,
# the widest spread of settings -- while Sonnet 5 dropped the taught name from most ideas
# and Kimi K2.6 was four times slower and pushed mechanism detail into incidental ideas.
IDEATION_MODEL = "openai/gpt-5.6-terra"

# The same model named for Inspect/Scout, which resolves `provider/model` itself and
# ships its own `openrouter` provider. Not interchangeable with DEFAULT_MODEL: the
# safety-tooling path takes a bare id plus force_provider, this one takes the prefix.
INSPECT_MODEL = f"openrouter/{DEFAULT_MODEL}"

# The same model on Inspect's *OpenAI* provider. Not interchangeable with
# INSPECT_MODEL, and the difference is not stylistic: OpenRouter does not carry
# Inspect's `response_schema` through to the model. Measured, not assumed -- asking
# for a `{reasoning, answer}` object returns a bare `42` via `openrouter/`, and the
# correct JSON object via `openai/`.
#
# Every judge that depends on a structured answer therefore has to use this one. The
# failure is silent and looks like a result: the reward_hack parser records
# `unparseable` for a model that answered perfectly well, and a Scout screen records a
# null verdict, which reads as "found nothing" rather than "was never asked properly".
INSPECT_STRUCTURED_MODEL = f"openai/{DEFAULT_MODEL.split('/', 1)[-1]}"

# Which upstream OpenRouter is allowed to serve the request from.
#
# OpenRouter is a broker: one model id can be served by several providers, at
# different quantisations, and it falls back between them on its own. That makes the
# backend a free variable in every judged number -- two scans of the same transcripts
# can be graded by differently-quantised weights with nothing in the output saying so.
# `order` names the only acceptable upstream and `allow_fallbacks: False` makes the
# call fail rather than silently move, which is the right trade for a grader: a failed
# scan is re-runnable, a quietly re-routed one is not detectable afterwards.
#
# Not the same thing as `CALL_KWARGS["force_provider"]` below, which is safetytooling
# choosing an API *client class* locally. This one travels to OpenRouter in the request
# body. Inspect's own `openrouter` provider forwards it as the `provider` model_arg.
PROVIDER_ROUTING: dict[str, Any] = {"order": ["openai"], "allow_fallbacks": False}


def pinned(model: str | Any | None = None, config: Any = None):
    """Resolve an Inspect model id, pinning the upstream when it goes via OpenRouter.

    Every Inspect-side judge goes through here rather than naming a string, because a
    bare string carries no model_args: Inspect resolves it without the `provider`
    routing and the pin is silently dropped. Ids on any other provider are returned
    unchanged -- `openai/...` has one upstream by construction and needs no pin.

    Accepts an already-resolved `Model` and returns it untouched, so a caller that was
    handed one does not have it rebuilt.
    """
    from inspect_ai.model import Model, get_model

    if isinstance(model, Model):
        return model
    name = model or INSPECT_MODEL
    if not str(name).startswith("openrouter/"):
        return get_model(name, config=config) if config is not None else name
    return get_model(name, config=config, provider=dict(PROVIDER_ROUTING))

# Bypasses `model_id_to_class`; see (1) above. Spread into every `api(...)` call.
#
# The value is "openai", NOT "openai_chat". Upstream's own error message tells you to
# "Pass openai_completion, openai_chat, anthropic, ..." but `provider_to_class`
# (api.py:234) is keyed "openai", and the mismatch surfaces as
# `AssertionError: Invalid force_provider` from a line whose message just told you to
# use that value. Trust the dict, not the message.
CALL_KWARGS: dict[str, Any] = {"force_provider": "openai"}

# Turns the model's reasoning off for calls that want a one-word answer.
#
# `DEFAULT_MODEL` is a reasoning model, and on a short `max_tokens` it spends the
# whole budget thinking and returns `content: None`. That does not surface as an
# empty completion: safety-tooling computes cost with `count_tokens(choice.message.content)`
# (openai/chat.py:129), so a `None` content raises `TypeError: expected string or
# buffer` from *token accounting*, several frames from anything to do with the
# model. The recall judge caught that exception and scored the answer 0.0, so 69 of
# 78 gradings in the first advisory_audit_rate run were failures recorded as wrong
# answers -- while the model's open answers were, on inspection, correct.
#
# This is the same shape as the renderer trap in `data/models.json`, in a place that
# map cannot reach: it is an auxiliary judge, not a model under test. Prefer
# disabling reasoning over raising `max_tokens` for the same reason given there --
# at temperature 1 the trace length varies, so a bigger budget fixes it only
# sometimes. A grader emitting one word of a three-word vocabulary does not need to
# think first.
#
# Only `{"enabled": False}` works. `{"effort": "minimal"}` and `{"max_tokens": 0}`
# were both measured against this judge and still returned `content: None`.
# Spread into `api(...)` beside `CALL_KWARGS`; `extra_body` reaches OpenRouter
# because safety-tooling forwards `**kwargs` to `chat.completions.create`.
NO_REASONING: dict[str, Any] = {"extra_body": {"reasoning": {"enabled": False}}}


def api_key() -> str:
    """The OpenRouter key, or an actionable error naming the file to put it in."""
    import os

    compat.require_env(API_KEY_ENV)
    return os.environ[API_KEY_ENV]


@functools.cache
def _routed_class(base: type) -> type:
    """`base` with `force_provider` defaulted, so upstream's own calls route too.

    Subclassing rather than patching the instance: `api(...)` resolves `__call__`
    on the type, so assigning to the attribute of an instance is silently ignored.
    """

    class _RoutedInferenceAPI(base):  # type: ignore[misc, valid-type]
        async def __call__(self, *args: Any, **kwargs: Any):
            kwargs.setdefault("force_provider", CALL_KWARGS["force_provider"])
            return await super().__call__(*args, **kwargs)

    _RoutedInferenceAPI.__name__ = f"Routed{base.__name__}"
    return _RoutedInferenceAPI


def inference_api(**kwargs: Any):
    """An `InferenceAPI` whose OpenAI-compatible client points at OpenRouter.

    Every call made through it routes, including ones made by upstream code this
    package does not own. Setting the base URL alone routes nothing, because the id
    never reaches a provider to begin with.
    """
    compat.bootstrap()
    from safetytooling.apis import InferenceAPI

    kwargs.setdefault("openai_base_url", BASE_URL)
    kwargs.setdefault("openai_api_key", api_key())
    return _routed_class(InferenceAPI)(**kwargs)
