"""Naming the models a suite runs against, once, in `data/models.json`.

A suite is a matrix: every eval against every model. The labels here are the only
thing that ties the matrix together -- one label becomes the log directory, the wandb
key, and the scoreboard column, so a number can always be traced back to the adapter
that produced it. Passing `--run_dir` per invocation, as the single-eval runner did,
cannot express "base plus three adapters" at all.

`baseline` names the label every other model's delta is computed against. Without
one, a scoreboard is a list of numbers with nothing to compare them to, which is the
same failure the recall eval's base-model control exists to prevent.

Manifest shape:

    {
      "baseline": "base",
      "models": {
        "base":   {"kind": "base",    "model": "deepseek-ai/DeepSeek-V3.1"},
        "r1-sdf": {"kind": "adapter", "run_dir": "data/tinker_runs/r1/raw_continuation"},
        "qwen":   {"kind": "base",    "model": "Qwen/Qwen3.5-397B-A17B",
                   "renderer": "qwen3_5_disable_thinking", "context_window": 32768}
      }
    }

`renderer` is optional and overrides the recommended renderer for that model only.
Needed for the models whose thinking is emitted inline as ordinary text -- see
`Target.inspect_model`. `context_window` is also optional; when set, the Tinker
provider rolls old tool history out of that token budget instead of letting an
agentic session fail at the model's hard context limit.

Under `data/` rather than in git, because it names run directories that only exist on
the machine that trained them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdf import paths

DEFAULT_BASE_MODEL = "deepseek-ai/DeepSeek-V3.1"

# What `sdf evals run --models base` falls back to when no manifest exists yet, so a
# fresh clone can evaluate the base model without authoring a file first.
IMPLICIT_MANIFEST: dict[str, Any] = {
    "baseline": "base",
    "models": {"base": {"kind": "base", "model": DEFAULT_BASE_MODEL}},
}


@dataclass(frozen=True)
class Target:
    label: str
    kind: str
    model: str | None = None
    run_dir: str | None = None
    renderer: str | None = None
    context_window: int | None = None
    # A `tinker://.../sampler_weights/<name>` URI, when the caller wants a specific
    # checkpoint rather than whatever the run finished on. `run_dir` alone can only
    # ever mean `final`, so a checkpoint sweep on behaviour evals had to bypass this
    # module entirely -- 24 of the 60 `.eval` logs on disk were produced by raw
    # `inspect eval -M model_path=...`, outside the scoreboard, the baseline/polarity
    # logic and the tracking.
    sampler_path: str | None = None
    checkpoint: str | None = None

    def config(self) -> dict[str, Any]:
        """What to record about this target in the scoreboard and wandb config.

        `sampler_path` is included because `run_dir` is a mutable pointer: retraining
        into the same save folder silently repoints every historical scoreboard that
        named it at different weights, and the URI is the only thing that identifies
        the weights themselves.
        """
        return {
            "kind": self.kind,
            "model": self.model,
            "run_dir": self.run_dir,
            "renderer": self.renderer,
            "context_window": self.context_window,
            "sampler_path": self.sampler_path,
            "checkpoint": self.checkpoint,
        }

    def _checkpoint_sampler_path(self) -> str | None:
        """Resolve a named checkpoint to its weights via the run's `checkpoints.jsonl`.

        Without this, `checkpoint` was copied into `scoreboard.json` and never used to
        select anything: the suite sampled the run's final weights while the scoreboard
        recorded `"checkpoint": "000400"`. A false provenance record in the file whose
        whole purpose is to be trustworthy is worse than no field at all.
        """
        if not self.checkpoint or self.checkpoint == "final" or not self.run_dir:
            return None
        import json

        path = paths.SDF_ROOT / self.run_dir / "checkpoints.jsonl"
        try:
            lines = path.read_text().splitlines()
        except OSError as exc:
            raise FileNotFoundError(
                f"{self.label}: checkpoint {self.checkpoint!r} needs {path}, which is not readable ({exc})."
            ) from exc
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("name")) == str(self.checkpoint) and row.get("sampler_path"):
                return row["sampler_path"]
        raise KeyError(
            f"{self.label}: no checkpoint named {self.checkpoint!r} in {path}. "
            "Periodic checkpoints expire after seven days unless the run kept them."
        )

    def inspect_model(self):
        """Resolve to an Inspect Model through the `tinker/` provider.

        The string after `tinker/` is a free label for an adapter -- `run_dir` is what
        actually selects the weights -- so the manifest label goes there and shows up
        in the Inspect log and in Scout, instead of a uniform "sdf".

        `renderer` exists because the recommended renderer is wrong for sampling on
        more models than gpt-oss. `TinkerModelSpec.UNUSABLE_FOR_SAMPLING` can only fix
        the cases where one name is *always* wrong; the thinking-mode renderers are
        wrong only when the token budget is too small to reach the answer, which is a
        property of the eval, not the model. Qwen3.5 and Kimi-K2.6 emit their thinking
        inline as ordinary text, so an un-disabled renderer spends the whole budget
        reasoning and returns a truncated trace with no answer in it -- scored
        `unscored` across the board, which looks exactly like a model that stopped
        answering. Naming the renderer per model in the manifest is the only way to
        say "sample this one with thinking off" without editing code.
        """
        from inspect_ai.model import get_model

        if self.kind == "adapter":
            # `model_path` wins over the run's own `sampler_path` inside the provider,
            # so passing it is how a named checkpoint reaches the sampler.
            extra: dict[str, Any] = {}
            sampler = self.sampler_path or self._checkpoint_sampler_path()
            if sampler:
                extra["model_path"] = sampler
            if not self.run_dir:
                # No run directory to read the base model from, so it has to be given.
                # Without it the provider falls back to the model *name*, which for an
                # adapter is a free label -- `tinker/mix-000020` is not a base model.
                extra["base_model"] = self.model
            return get_model(
                f"tinker/{self.label}",
                run_dir=self.run_dir,
                renderer_name=self.renderer,
                context_window=self.context_window,
                **extra,
            )
        return get_model(
            f"tinker/{self.model}",
            renderer_name=self.renderer,
            context_window=self.context_window,
        )


def default_manifest_path() -> Path | None:
    """`data/models.json` if present, else the packaged `experiments/models.json`.

    The local copy wins so a machine can register an adapter it just trained without
    touching git. The packaged copy exists because the local one is invisible to
    everyone else: it is under `data/`, which is not in git, and `sync._run_patterns`
    did not list it either -- so a collaborator received scoreboards whose model labels
    nothing could resolve, and `targets.resolve` raises KeyError on an unknown label.
    """
    if paths.MODELS_MANIFEST.exists():
        return paths.MODELS_MANIFEST
    if paths.PACKAGED_MODELS_MANIFEST.exists():
        return paths.PACKAGED_MODELS_MANIFEST
    return None


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path else default_manifest_path()
    if manifest_path is None or not manifest_path.exists():
        if path:
            raise FileNotFoundError(f"No model manifest at {manifest_path}.")
        return IMPLICIT_MANIFEST
    raw = json.loads(manifest_path.read_text())
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{manifest_path}: 'models' must be a non-empty object.")
    baseline = raw.get("baseline")
    if baseline is not None and baseline not in models:
        raise ValueError(
            f"{manifest_path}: baseline {baseline!r} is not one of the declared models "
            f"({', '.join(sorted(models))}). Deltas would silently be omitted."
        )
    return raw


def _to_target(label: str, entry: dict[str, Any], manifest_path: Path | str) -> Target:
    kind = entry.get("kind") or ("adapter" if entry.get("run_dir") else "base")
    if kind not in ("base", "adapter"):
        raise ValueError(f"{manifest_path}: model {label!r} has unknown kind {kind!r} (expected base or adapter).")
    if kind == "adapter" and not (entry.get("run_dir") or (entry.get("sampler_path") and entry.get("model"))):
        # Two ways to name an adapter, matching `experiments.ModelBinding`: a run
        # directory, or a pinned `sampler_path` with the base model beside it. A
        # checkpoint sweep produces the second shape, and rejecting it here made every
        # adopted sweep spec unrunnable while the schema said it was valid.
        raise ValueError(
            f"{manifest_path}: adapter {label!r} needs a 'run_dir', or a 'sampler_path' with the base 'model'."
        )
    if kind == "base" and not entry.get("model"):
        raise ValueError(f"{manifest_path}: base model {label!r} needs a 'model' id.")
    return Target(
        label=label,
        kind=kind,
        model=entry.get("model"),
        run_dir=entry.get("run_dir"),
        renderer=entry.get("renderer"),
        context_window=entry.get("context_window"),
        sampler_path=entry.get("sampler_path"),
        checkpoint=entry.get("checkpoint"),
    )


def resolve(
    selector: str | list[str] | None = None,
    manifest_path: Path | str | None = None,
    bindings: dict[str, dict[str, Any]] | None = None,
    baseline: str | None = None,
) -> tuple[list[Target], str | None]:
    """Turn `--models all` / `--models base,r1-sdf` into targets plus the baseline label.

    An unknown label raises rather than being skipped: a suite that quietly evaluated
    two models when you asked for three still prints a complete-looking table.

    `bindings` lets an experiment spec supply the label -> model mapping directly,
    bypassing the manifest. The manifest is a mutable convenience -- it is edited in
    place, has no history, and lives under `data/` -- so a spec that referenced it by
    label would not stay meaningful. Several scoreboards already on disk name labels
    the current manifest cannot resolve at all.
    """
    if bindings:
        manifest = {"models": dict(bindings), "baseline": baseline}
        where = "(experiment spec)"
    else:
        manifest = load_manifest(manifest_path)
        where = manifest_path or default_manifest_path() or "(built-in default)"
    models = manifest["models"]

    if selector in (None, "all", "*"):
        labels = list(models)
    else:
        labels = selector if isinstance(selector, (list, tuple)) else str(selector).split(",")
        labels = [str(x).strip() for x in labels if str(x).strip()]
        unknown = [x for x in labels if x not in models]
        if unknown:
            raise KeyError(f"Unknown model label(s): {', '.join(unknown)}. Declared in {where}: {', '.join(models)}")

    targets = [_to_target(label, models[label], where) for label in labels]

    baseline = manifest.get("baseline")
    if baseline not in labels:
        # Comparing against a model that was not run is worse than not comparing.
        baseline = None
    return targets, baseline
