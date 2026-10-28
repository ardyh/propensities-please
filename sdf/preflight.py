"""Verify credentials and model names before spending money on a full run.

Every model identifier in this pipeline is a default that may have aged out, both
the ones inherited from false-facts (which pinned snapshots like
`claude-3-5-sonnet-20241022`) and the ones chosen here. This makes one cheap call
per service so a wrong model string or missing key surfaces in seconds rather than
part-way through document generation.

    uv run sdf preflight
"""

from __future__ import annotations

import asyncio
import logging
import os

import fire

from sdf import compat

compat.bootstrap()

from sdf.corpus import generate  # noqa: E402
from sdf.evals.scanners import screens  # noqa: E402
from sdf.models import providers, tinker  # noqa: E402
from sdf.training import finetune  # noqa: E402

LOGGER = logging.getLogger(__name__)

# TINKER_API_KEY always serves the models under test. The auxiliary keys depend on
# which providers the configured models route to (`sdf.models.providers` keys off
# the id shape), so the requirement is computed from the models being checked
# rather than fixed: a Luna-through-OpenRouter run must not demand an Anthropic
# key, and a claude-judged run must not pass preflight on OPENROUTER_API_KEY alone.
ALWAYS_REQUIRED_KEYS = ("TINKER_API_KEY",)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check_keys(
    auxiliary_models: tuple[str, ...] = (),
    inspect_models: tuple[str, ...] = (),
) -> tuple[bool, list[str]]:
    """Which keys the configured models need, and whether they are present.

    `inspect_models` is the second half, and it was missing. Preflight only ever
    considered the safetytooling side (generation and the recall judge), both of which
    default to namespaced ids and therefore to OPENROUTER_API_KEY. The Scout screens
    and the eval parsers go through Inspect's own OpenAI provider and read the
    unnumbered OPENAI_API_KEY, which nothing checked -- so preflight passed and the
    first failure arrived at *scoring* time, after a suite's Tinker sampling had
    already been paid for. Worse, `compat.bootstrap` installs an inert sentinel into
    that variable, so the failure is an auth error from a fabricated key rather than a
    missing-key message.
    """
    required = list(ALWAYS_REQUIRED_KEYS)
    for aux_model in auxiliary_models:
        for key in providers.required_credentials(aux_model):
            if key not in required:
                required.append(key)
    for inspect_model in inspect_models:
        for key in providers.inspect_required_credentials(inspect_model):
            if key not in required:
                required.append(key)
    missing = [key for key in required if not _has_real_key(key)]
    for key in required:
        print(f"  [{_status(key not in missing)}] {key}")
    return not missing, missing


def _has_real_key(key: str) -> bool:
    """Present, and not the placeholder `compat.bootstrap` installs for OpenAI.

    `os.environ.get` alone reports the sentinel as a key, which is exactly the check
    this command exists to make trustworthy.
    """
    value = os.environ.get(key)
    return bool(value) and value != compat.INERT_OPENAI_KEY


def check_tinker_model(model: str) -> bool:
    """Confirm the key works and the requested base model is actually offered."""
    try:
        import tinker

        client = tinker.ServiceClient()
        capabilities = client.get_server_capabilities()
        available = sorted(m.model_name for m in capabilities.supported_models if m.model_name)
    except Exception as exc:  # noqa: BLE001 - reported to the operator
        print(f"  [FAIL] Tinker: could not list models ({type(exc).__name__}: {exc})")
        return False

    ok = model in available
    print(f"  [{_status(ok)}] Tinker base model {model!r}")
    if not ok:
        print(f"         {len(available)} models available, e.g.:")
        for name in available[:15]:
            print(f"           - {name}")
    return ok


async def _check_auxiliary_model(model: str, role: str) -> bool:
    """Make the cheapest possible call through the exact client path the stage uses.

    Going through `providers.inference_api` plus `providers.call_kwargs` rather
    than a hand-built client is the whole point: the failure modes this catches --
    an id that `model_id_to_class` rejects, a base URL that 404s, a key for the
    wrong provider -- are all invisible to a check constructed differently from
    production.
    """
    try:
        from safetytooling.data_models import ChatMessage, MessageRole, Prompt

        api = providers.inference_api(model, cache_dir=None)
        prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content="Reply with: ok")])
        responses = await api(
            model_id=model,
            prompt=prompt,
            max_tokens=5,
            max_attempts_per_api_call=1,
            **providers.call_kwargs(model),
        )
        ok = bool(responses and responses[0].completion)
    except Exception as exc:  # noqa: BLE001 - reported to the operator
        print(f"  [FAIL] {role} {model!r} ({type(exc).__name__}: {exc})")
        return False
    print(f"  [{_status(ok)}] {role} {model!r}")
    return ok


async def check_judge_model(model: str) -> bool:
    return await _check_auxiliary_model(model, "Judge model")


async def check_doc_gen_model(model: str) -> bool:
    return await _check_auxiliary_model(model, "Document generation model")


async def amain(
    model: str = finetune.DEFAULT_MODEL,
    judge_model: str = tinker.DEFAULT_JUDGE_MODEL,
    doc_gen_model: str = generate.DEFAULT_DOC_GEN_MODEL,
    scan_judge_model: str = screens.DEFAULT_JUDGE_MODEL,
) -> bool:
    print("\nCredentials")
    keys_ok, missing = check_keys(
        auxiliary_models=(judge_model, doc_gen_model),
        inspect_models=(scan_judge_model,),
    )
    if not keys_ok:
        print(f"\nFill in {', '.join(missing)} in {compat.SDF_ROOT / '.env'} and re-run.")
        return False

    print("\nTinker")
    tinker_ok = check_tinker_model(model)

    print("\nDocument generation and judge")
    doc_ok = await check_doc_gen_model(doc_gen_model)
    judge_ok = await check_judge_model(judge_model)

    all_ok = all([tinker_ok, doc_ok, judge_ok])
    print(
        "\n"
        + (
            "Preflight passed. Next: uv run sdf corpus generate"
            if all_ok
            else "Preflight failed; see above."
        )
    )
    return all_ok


def main(
    model: str = finetune.DEFAULT_MODEL,
    judge_model: str = tinker.DEFAULT_JUDGE_MODEL,
    doc_gen_model: str = generate.DEFAULT_DOC_GEN_MODEL,
    scan_judge_model: str = screens.DEFAULT_JUDGE_MODEL,
) -> bool:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    return asyncio.run(
        amain(
            model=model,
            judge_model=judge_model,
            doc_gen_model=doc_gen_model,
            scan_judge_model=scan_judge_model,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
