"""Make the vendored `false-facts` clone importable and runnable from this project.

The upstream repo is used as a library without editing a single file in it, which
means this module has to bridge four gaps between how upstream expects to run and
how we actually run it:

1. `false_facts` and `safetytooling` are not installed as packages (their own
   dependency sets are far larger than what this pipeline touches), so both
   source trees are put on `sys.path` instead.
2. Upstream calls `safetytooling.utils.utils.load_secrets()`, which does not
   exist in the safety-tooling commit that false-facts pins as its submodule.
   A compatible implementation is injected.
3. Upstream hardcodes absolute `/workspace/false-facts/...` paths, both as
   default arguments and as literals inside method bodies. Every such path is
   rewritten to point at our clone. (A symlink would be the tidier fix, but
   macOS mounts `/` read-only, so `/workspace` cannot be created.)
4. Upstream imports `strong_reject` at module scope purely for one dataset
   loader. That package pulls in litellm/peft/torch, so when it is absent an
   equivalent loader (reading the same upstream CSV) is registered in its place.

Import this module and call `bootstrap()` before importing anything from
`false_facts`, then use `import_false_facts()` to pull in upstream modules so
they get patched on the way in.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sdf import paths

LOGGER = logging.getLogger(__name__)

# Re-exported so `compat.SDF_ROOT` keeps working everywhere it is already used. It is
# NOT `Path(__file__).parent` any more -- see sdf/paths.py for why a hardcoded depth
# is wrong the moment a module changes nesting.
SDF_ROOT = paths.SDF_ROOT
FALSE_FACTS_ROOT = SDF_ROOT / "false-facts"
SAFETY_TOOLING_ROOT = FALSE_FACTS_ROOT / "safety-tooling"

# The absolute path prefix baked into upstream's source.
UPSTREAM_PREFIX = "/workspace/false-facts"

# Where StrongREJECT's canonical prompt set lives, mirroring what the
# strong_reject package itself fetches.
STRONGREJECT_CSV_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)
STRONGREJECT_SMALL_CSV_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_small_dataset.csv"
)

# Loaders whose path arguments need rewriting. `load_prompt_file` matters because
# it feeds a Jinja loader rooted at the clone, which rejects outside paths.
_PATCHED_LOADERS = ("load_txt", "load_json", "load_jsonl", "load_prompt_file")

_ATTR_MARKER = "_propensities_please_patched"

_bootstrapped = False


def rewrite_path(value: Any) -> Any:
    """Repoint an upstream `/workspace/false-facts/...` path at our clone."""
    if isinstance(value, str) and UPSTREAM_PREFIX in value:
        return value.replace(UPSTREAM_PREFIX, str(FALSE_FACTS_ROOT))
    if isinstance(value, Path):
        return Path(rewrite_path(str(value)))
    return value


def _ensure_sys_path() -> None:
    for root in (FALSE_FACTS_ROOT, SAFETY_TOOLING_ROOT):
        if not root.exists():
            raise FileNotFoundError(
                f"Expected the false-facts clone at {root}. Run:\n"
                f"  git clone https://github.com/safety-research/false-facts.git "
                f"{FALSE_FACTS_ROOT}\n"
                f"  cd {FALSE_FACTS_ROOT} && git submodule update --init"
            )
        path_str = str(root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_dotenv() -> None:
    """Load the repo's `.env` regardless of the current working directory.

    Upstream calls `dotenv.load_dotenv()` at import time, which only searches
    upward from the cwd.
    """
    try:
        import dotenv
    except ModuleNotFoundError:
        return
    env_path = SDF_ROOT / ".env"
    if env_path.exists():
        dotenv.load_dotenv(env_path, override=False)


# Sentinel rather than a real key: see _unblock_inference_api_construction.
INERT_OPENAI_KEY = "propensities-please-openai-unused"


def _unblock_inference_api_construction() -> None:
    """Let `InferenceAPI` be constructed without an OpenAI key.

    `InferenceAPI.__init__` builds an `OpenAICompletionModel` eagerly, before any
    model-name dispatch happens, and `openai.AsyncClient()` raises outright when
    `OPENAI_API_KEY` is unset. So an absent OpenAI key blocks *Anthropic* calls
    too, which is how it presents: a `claude-*` model id failing with an
    `OpenAIError` about missing OpenAI credentials.

    We deliberately do not copy `OPENAI_API_KEY1` across the way upstream's
    `setup_environment(openai_tag=...)` would. Every auxiliary call in this
    pipeline is routed at OpenRouter, which supplies its own key on the client it
    builds (`openrouter.inference_api`), so nothing here needs a real OpenAI
    credential. A working one would only let paths that default to an OpenAI model
    bill silently -- upstream's `mcq_utils` reaches for `gpt-4o-mini` to extract
    MCQ answers, for instance. The inert value keeps construction working while
    making any genuinely unrouted OpenAI call fail loudly at the call site.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = INERT_OPENAI_KEY


def _parse_secrets_file(path: Path) -> dict[str, str]:
    secrets: dict[str, str] = {}
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            secrets[key.strip()] = value.strip()
    return secrets


def _resolve_secrets_path(source: str | Path) -> Path | None:
    candidate = Path(source)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for base in (Path.cwd(), SDF_ROOT, FALSE_FACTS_ROOT):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return None


# Keys upstream subscripts unconditionally at construction time, for clients this
# pipeline never calls. `SyntheticDocumentGenerator.__init__` builds a
# `BatchInferenceAPI` with `secrets["ANTHROPIC_API_KEY_BATCH"]`
# (`synth_doc_generation.py:59`) before any generation happens, so an absent key is a
# `KeyError` during *setup* of a run that would then have gone entirely to OpenRouter.
# That was survivable while generation was Anthropic-backed and the key was set
# anyway; since the OpenRouter switch there is no Anthropic key in the pipeline at
# all, and this became an unconditional crash on the first document.
#
# A placeholder is the honest fix rather than a real key: the batch path is unused, so
# the value is never sent anywhere. It is spelled to be self-explanatory if it ever
# does reach a provider and comes back as an auth failure.
_UNUSED_SECRET_PLACEHOLDERS = {
    "ANTHROPIC_API_KEY_BATCH": "unused-batch-path-placeholder-see-sdf-compat",
}


def _install_secrets_shim() -> None:
    """Provide `load_secrets`, absent from the pinned safety-tooling commit.

    Upstream's version read a `KEY=value` file and would `KeyError` on anything
    missing. This one also falls back to the environment, so setting keys in
    `.env` alone is enough, and fills `_UNUSED_SECRET_PLACEHOLDERS` for the keys
    upstream demands at construction time but never uses on our code path.
    """
    from safetytooling.utils import utils as st_utils

    if hasattr(st_utils, "load_secrets"):
        return

    def load_secrets(source: str | Path = "SECRETS") -> dict[str, str]:
        secrets: dict[str, str] = dict(os.environ)
        path = _resolve_secrets_path(source)
        if path is not None:
            secrets.update(_parse_secrets_file(path))
        else:
            LOGGER.debug("No secrets file found for %r; using environment only.", source)
        for key, placeholder in _UNUSED_SECRET_PLACEHOLDERS.items():
            if not secrets.get(key):
                LOGGER.debug("%s unset; using placeholder (its client is never called).", key)
                secrets[key] = placeholder
        return secrets

    setattr(load_secrets, _ATTR_MARKER, True)
    st_utils.load_secrets = load_secrets


def _wrap_loader(func: Callable) -> Callable:
    if getattr(func, _ATTR_MARKER, False):
        return func

    @functools.wraps(func)
    def wrapper(path, *args, **kwargs):
        return func(rewrite_path(path), *args, **kwargs)

    setattr(wrapper, _ATTR_MARKER, True)
    return wrapper


def _patch_loaders() -> None:
    """Rewrite paths at the loader boundary.

    This covers the `/workspace/...` literals written inline inside upstream
    method bodies, which no caller can override. Patching
    `false_facts.utils` before other upstream modules are imported means their
    `from false_facts.utils import load_txt` picks up the wrapped version.
    """
    ff_utils = importlib.import_module("false_facts.utils")
    for name in _PATCHED_LOADERS:
        original = getattr(ff_utils, name, None)
        if callable(original):
            setattr(ff_utils, name, _wrap_loader(original))


def _rewrite_defaults(func: Callable) -> None:
    """Repoint any `/workspace/...` default argument at our clone.

    Handles cases upstream exposes no override for, such as
    `SyntheticDocumentGenerator.oai_batch_log_dir_path`, which is passed
    straight to `mkdir()` and would otherwise fail on a read-only `/`.
    """
    if func.__defaults__:
        rewritten = tuple(rewrite_path(d) for d in func.__defaults__)
        if rewritten != func.__defaults__:
            func.__defaults__ = rewritten
    if func.__kwdefaults__:
        rewritten_kw = {k: rewrite_path(v) for k, v in func.__kwdefaults__.items()}
        if rewritten_kw != func.__kwdefaults__:
            func.__kwdefaults__ = rewritten_kw


def _patch_module_defaults(module: types.ModuleType) -> None:
    for obj in vars(module).values():
        if inspect.isfunction(obj):
            _rewrite_defaults(obj)
        elif inspect.isclass(obj) and getattr(obj, "__module__", "") == module.__name__:
            for method in vars(obj).values():
                if inspect.isfunction(method):
                    _rewrite_defaults(method)


def _patch_loaded_false_facts_modules() -> None:
    for name, module in list(sys.modules.items()):
        if name != "false_facts" and not name.startswith("false_facts."):
            continue
        if module is None:
            continue
        _patch_module_defaults(module)
        # Re-bind loaders in modules that imported them by value before patching.
        for loader_name in _PATCHED_LOADERS:
            existing = getattr(module, loader_name, None)
            if callable(existing) and not getattr(existing, _ATTR_MARKER, False):
                setattr(module, loader_name, _wrap_loader(existing))


def _install_strong_reject_fallback() -> None:
    """Register a minimal `strong_reject.load_datasets` when the real one is absent.

    `false_facts.evaluations.personality_evals.harmfulness` imports
    `load_strongreject` at module scope, but the package it comes from depends on
    litellm/peft/torch. The replacement reads the same upstream CSV the real
    loader does, so the prompt set is identical.
    """
    try:
        importlib.import_module("strong_reject.load_datasets")
        return
    except ModuleNotFoundError:
        pass

    def _load_csv(url: str):
        from datasets import load_dataset

        return load_dataset("csv", data_files=url)["train"]

    load_datasets_module = types.ModuleType("strong_reject.load_datasets")
    load_datasets_module.load_strongreject = lambda: _load_csv(STRONGREJECT_CSV_URL)
    load_datasets_module.load_strongreject_small = lambda: _load_csv(STRONGREJECT_SMALL_CSV_URL)
    load_datasets_module.__doc__ = (
        "Minimal stand-in for strong_reject.load_datasets provided by this repo's "
        "compat. Install the `strongreject` extra to use the real package."
    )

    package = types.ModuleType("strong_reject")
    package.__path__ = []  # type: ignore[attr-defined]
    package.load_datasets = load_datasets_module

    sys.modules.setdefault("strong_reject", package)
    sys.modules.setdefault("strong_reject.load_datasets", load_datasets_module)
    LOGGER.info("Using sdf.compat's built-in StrongREJECT dataset loader.")


def bootstrap() -> None:
    """Prepare the process to import `false_facts`. Safe to call repeatedly."""
    global _bootstrapped
    if _bootstrapped:
        return

    _ensure_sys_path()
    _load_dotenv()
    _unblock_inference_api_construction()
    _install_secrets_shim()
    _install_strong_reject_fallback()
    _patch_loaders()
    _patch_loaded_false_facts_modules()

    _bootstrapped = True


def import_false_facts(module_name: str) -> types.ModuleType:
    """Import an upstream module with the path patches applied.

    Args:
        module_name: Dotted name, with or without the `false_facts.` prefix.
    """
    bootstrap()
    if not module_name.startswith("false_facts"):
        module_name = f"false_facts.{module_name}"
    module = importlib.import_module(module_name)
    # Sweep everything, since the requested module drags in siblings.
    _patch_loaded_false_facts_modules()
    return module


def prompt_path(*parts: str) -> str:
    """Absolute path to a prompt file shipped inside the clone."""
    return str(FALSE_FACTS_ROOT.joinpath("false_facts", "prompts", *parts))


def require_env(*names: str) -> None:
    """Fail early with an actionable message when credentials are missing."""
    bootstrap()
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + f".\nCopy {SDF_ROOT / '.env.template'} to {SDF_ROOT / '.env'} and fill them in."
        )
