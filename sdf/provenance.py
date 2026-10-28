"""Who ran a stage, when, against what code, and through which provider.

Every stage of this pipeline already computes these facts and then drops them. The
consequences were not hypothetical:

- **No artifact carries a wall-clock timestamp.** `generation_report.json` records
  `elapsed_seconds`, a duration; `scoreboard.json` and `sdf_finetune_result.json`
  record nothing. `sdf sync pull` rewrites every file, so mtime is not a fallback
  either. Two scoreboards on disk report opposite-signed deltas for the same adapter
  on the same eval and nothing can say which one superseded the other.
- **No stage records this repo's git SHA.** The one code artifact, `code.diff`, is
  written by tinker-cookbook from whichever git repo encloses the *installed module*,
  so across thirteen runs it names three different machines, and the flagship adapter
  records a `false-facts` SHA that is dangling in both repos.
- **The provider is never written down**, even though `sdf.models.providers` routes
  three of them. Its own docstring says recording that is this system's job.

So: one module, imported by every stage that writes a result, producing the same
block everywhere.

Deliberately cheap. `stamp()` shells out to git rather than importing GitPython, and
every lookup is cached for the process, because these values cannot change mid-run
and some callers write a record per document. Every function degrades to `None`
rather than raising -- provenance that can break a paid run is worse than no
provenance.
"""

from __future__ import annotations

import functools
import logging
import subprocess
from datetime import UTC, datetime
from typing import Any

from sdf import paths

LOGGER = logging.getLogger(__name__)

# Package versions worth pinning a result to. `pyproject.toml` declares floors rather
# than pins, so `uv lock --upgrade` moves any of these silently; a result that does not
# name them cannot be reproduced against the same stack. Inspect records only its own
# version in a `.eval` header, which is why the list is kept here rather than deferred
# to it.
TRACKED_PACKAGES = (
    "inspect-ai",
    "inspect-evals",
    "inspect-scout",
    "tinker",
    "tinker-cookbook",
    "transformers",
)


def _git(*args: str) -> str | None:
    """Run a git command in the repo root. None on failure, "" on empty output."""
    try:
        out = subprocess.run(
            ["git", "-C", str(paths.SDF_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.debug("git %s failed: %s", " ".join(args), exc)
        return None
    if out.returncode != 0:
        return None
    # `.strip()`, not `.strip() or None`: an empty result is an answer. `git status
    # --porcelain` prints nothing for a clean tree, and folding that into None made
    # every clean checkout record `dirty: null` -- "unknown" -- which is the one value
    # the field exists to distinguish from.
    return out.stdout.strip()


@functools.lru_cache(maxsize=1)
def owner() -> str | None:
    """The git email of whoever is running this, or None outside a configured repo."""
    return _git("config", "user.email")


@functools.lru_cache(maxsize=1)
def code_state() -> dict[str, Any]:
    """This repo's commit, whether the tree was dirty, and the submodule pins.

    `dirty` matters as much as the commit: all sixty `.eval` logs on disk carry
    Inspect's own `dirty=True`, which means the recorded commit does not identify the
    code that ran. Recording the flag at least makes that legible instead of implied.

    Submodule SHAs are included because `false-facts` is pinned by this repo and its
    upstream helpers are load-bearing -- and because no existing artifact records them.
    """
    commit = _git("rev-parse", "HEAD") or None
    status = _git("status", "--porcelain")
    submodules: dict[str, str] = {}
    raw = _git("submodule", "status", "--recursive")
    for line in (raw or "").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            submodules[parts[1]] = parts[0].lstrip("+-U")
    return {
        "commit": commit,
        # None rather than False when git could not be reached: "clean" is a claim, and
        # an unavailable git is not evidence for it.
        "dirty": None if status is None else bool(status),
        "submodules": submodules or None,
    }


@functools.lru_cache(maxsize=1)
def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def api_of(model: str) -> str | None:
    """Which of OpenRouter / Anthropic / OpenAI serves `model`.

    Derived from `sdf.models.providers`, which dispatches on the id's *shape*, so this
    never disagrees with what actually gets called. Returns None for an id that module
    does not recognise -- an unknown shape is a thing to record, not to raise over,
    because this runs while a result is being written.
    """
    from sdf.models import providers

    try:
        if providers.is_openrouter(model):
            return "openrouter"
    except Exception:  # noqa: BLE001 - a malformed id must not break a write
        return None
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt"):
        return "openai"
    return None


def inspect_api_of(model: str) -> str | None:
    """Which API serves `model` when Inspect calls it, from the provider prefix.

    Not the same answer as `api_of`. `openai/gpt-5.6-luna` means OpenRouter to
    safetytooling and OpenAI to Inspect -- the split this project deliberately runs --
    so a scan judge stamped with `api_of` recorded the wrong provider in the very file
    added to make it attributable.
    """
    from sdf.models import providers

    keys = providers.inspect_required_credentials(model)
    return {
        "OPENAI_API_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic",
        providers.openrouter.API_KEY_ENV: "openrouter",
        "TINKER_API_KEY": "tinker",
    }.get(keys[0]) if keys else None


def role(model: str | None, inspect_side: bool = False) -> dict[str, Any] | None:
    """One auxiliary model as `{"api": ..., "model": ...}`.

    `inspect_side=True` for anything Inspect calls -- the Scout screens and the eval
    answer-parsers -- because the same id routes differently on the two halves.

    The api is not redundant with the id. `providers.is_openrouter` infers it from a
    naming *convention* ("has a slash"), and a convention is exactly the thing that
    drifts -- so the resolved answer is written down beside the id it was resolved
    from, and `sdf experiments check` compares the two.
    """
    if not model:
        return None
    return {"api": inspect_api_of(model) if inspect_side else api_of(model), "model": model}


def stamp() -> dict[str, Any]:
    """The block every stage embeds in whatever it writes."""
    return {
        "created_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "owner": owner(),
        "code": code_state(),
        "packages": package_versions(),
    }
