"""Where everything lives on disk, resolved independently of the current directory.

Every default path in this project hangs off `SDF_ROOT`: the vendored `false-facts`
clone, `.env`, the whole of `data/`. It used to be `Path(__file__).resolve().parent`
back when every module sat at the repo root, which quietly stopped being correct the
moment the modules moved into a package -- a hardcoded parent depth is wrong again
every time a module changes nesting, and wrong in a way that yields a plausible
directory rather than an error.

So it is *found*, not computed: walk up until a `pyproject.toml` appears. That is the
repo root by definition, at any nesting depth, and it also holds for an editable
install (`uv sync` on this project), where `__file__` still points into the source
tree.

`SDF_ROOT` in the environment overrides the search. That is the escape hatch for a
non-editable install, where the package lives in `site-packages` and there is no repo
above it to find.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DATA_DIR",
    "EVALS_DIR",
    "EVAL_INPUTS_DIR",
    "EXPERIMENTS_DIR",
    "MODELS_MANIFEST",
    "PACKAGED_MODELS_MANIFEST",
    "RESULTS_DIR",
    "SCREENS_DIR",
    "SDF_ROOT",
    "UNIVERSE_DIR",
    "decisiveness_dir",
    "decisiveness_path",
    "experiment_dir",
    "logs_dir",
    "recall_path",
    "scans_dir",
    "scoreboard_path",
    "spec_path",
]


def _find_root() -> Path:
    override = os.environ.get("SDF_ROOT")
    if override:
        return Path(override).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        "Could not locate the repo root: no pyproject.toml above "
        f"{Path(__file__).resolve()}. This happens when the package is installed "
        "non-editably. Set the SDF_ROOT environment variable to the repo checkout."
    )


SDF_ROOT = _find_root()

DATA_DIR = SDF_ROOT / "data"
UNIVERSE_DIR = DATA_DIR / "universe_contexts"
RESULTS_DIR = DATA_DIR / "results"

# Experiment specs: what a run was *meant* to be, in git. Declarations rather than
# outputs, so the `data/` rule below does not apply -- they are kilobytes and they are
# the artifact collaborators diff, the same argument that keeps `evals/` in git.
EXPERIMENTS_DIR = SDF_ROOT / "experiments"

# The model-target manifest -- which base models and which adapters an eval suite runs
# against. It lives in git beside the specs whose labels it resolves, with the copy
# under `data/` kept as a local override.
#
# It used to be `data/models.json` only, on the reasoning that it names run directories
# that exist only on the machine that trained them. That reasoning did not survive
# contact: `sync._run_patterns` never listed it, so a scoped push shipped a scoreboard
# whose model labels the recipient could not resolve at all, and the file is edited in
# place per experiment with no history anywhere. A label is the only thing tying a
# scoreboard column to an adapter, which makes it a declaration, not an output.
PACKAGED_MODELS_MANIFEST = EXPERIMENTS_DIR / "models.json"
MODELS_MANIFEST = DATA_DIR / "models.json"

# Eval *definitions* live in git, unlike everything else the pipeline reads. They are
# kilobytes of prompts rather than a 40MB corpus, and they are the artifact
# collaborators diff -- the same argument that put universe contexts in `data/` (too
# big, versioned on HF) runs the other way here. See CLAUDE.md.
EVALS_DIR = SDF_ROOT / "evals"

# Human labels for the Scout screens, one CSV per scanner (`id,target`). In git
# alongside the evals, for the same reason: they are the ground truth a screener is
# measured against, and they are worth reviewing.
SCREENS_DIR = EVALS_DIR / "_screens"


# Inputs an eval needs but is too large to keep in git -- scraped corpora, fetched
# datasets. The `evals/` rule above is about kilobytes of prompts; a few hundred KB of
# statute text is what `data/` exists for. One subdirectory per eval, built by that
# eval's own builder script, versioned on HF via `sdf sync` rather than in the repo.
EVAL_INPUTS_DIR = DATA_DIR / "eval_inputs"


def experiment_dir(experiment: str) -> Path:
    return RESULTS_DIR / experiment


def logs_dir(experiment: str, eval_name: str | None = None, model_label: str | None = None) -> Path:
    """Inspect log directory for a whole experiment, one eval, or one cell.

    The nesting is what lets `inspect view --log-dir <experiment>/logs` show the whole
    suite (it recurses) while `.../logs/<eval>/<model>` narrows to the one cell that
    looked wrong. Scout reads the same tree from the top and filters on its own.
    """
    path = experiment_dir(experiment) / "logs"
    if eval_name:
        path = path / eval_name
    if model_label:
        path = path / model_label
    return path


def scans_dir(experiment: str) -> Path:
    return experiment_dir(experiment) / "scans"


def scoreboard_path(experiment: str) -> Path:
    return experiment_dir(experiment) / "scoreboard.json"


def recall_path(experiment: str) -> Path:
    """Where `sdf recall run` writes its results and transcripts.

    There was no helper for this, and recall only wrote a file when `--output` was
    passed, so the gate's own output landed wherever the operator happened to point it.
    On disk that produced loose `*_recall.json` files sitting *beside* the experiment
    directories rather than inside them, in at least two mutually incompatible shapes.
    """
    return experiment_dir(experiment) / "recall.json"


def decisiveness_path(experiment: str) -> Path:
    """Where `sdf decisiveness run` writes its panel: base, adapter and the delta."""
    return experiment_dir(experiment) / "decisiveness.json"


def decisiveness_dir(experiment: str) -> Path:
    """Upstream's per-model artifacts (`edges.jsonl`, `mu.json`, `panel.json`), one
    subdirectory per model label. The edge list is the record the panel was fitted
    from, the way `.eval` files are the record behind a scoreboard cell."""
    return experiment_dir(experiment) / "decisiveness"


def spec_path(name: str) -> Path:
    return EXPERIMENTS_DIR / f"{name}.json"


# Wiring tests go in a subtree of their own, so the catalogue is not mostly smoke. On
# this repo 37 of the unreferenced artifacts have a name that says smoke, pilot or dry
# run, and they sit beside the real corpora looking identical to them.
SMOKE_SEGMENT = "_smoke"


def smoke_path(path: str | Path) -> Path:
    """`data/synth_docs/foo` -> `data/synth_docs/_smoke/foo`.

    A separate directory rather than a name prefix, because a prefix is still a name,
    and a name is what nothing downstream can read reliably. `sync` skips this subtree,
    the reference graph skips it, and `sdf experiments prune --smoke` can clear it
    wholesale without touching anything a real run consumed.
    """
    target = Path(path)
    if SMOKE_SEGMENT in target.parts:
        return target
    return target.parent / SMOKE_SEGMENT / target.name
