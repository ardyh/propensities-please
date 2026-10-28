"""What a run was meant to be, checked against what the artifacts say it was.

An experiment spans four directory trees -- a universe, a corpus, a training run, a
results directory -- and until now nothing tied them together except a naming
convention that was never followed. Measured on this repo: 62 corpora, 15 training
runs and 21 result directories, **89 distinct names, only 2 of which appear in all
three trees**. `sync._run_patterns` assumes they match, which is why a scoped push has
been shipping incomplete slices.

So the declaration here is *intent*, and the artifacts remain the record of fact:

    load()      what the work was meant to be     (experiments/<dataset>/, in git)
    resolve()   what the artifacts on disk say    (generation_report.json, ...)
    diff()      where those two disagree

The declaration is keyed on the **dataset**, in three append-only levels:

    experiments/<dataset>/
      dataset.json          the corpus pointer, its universe and its generator
      models/<name>.json    one adapter trained against it, or one base it is compared to
      runs/<name>.json      one measurement: which models, which evals, by whom

One file per model and per run, never a list inside a shared file, because several
people work against one dataset at different times. An append is then a new file, so
two authors never touch the same lines and git has nothing to merge. Ordering is read
from each entry's `created_at`, not from its filename -- which is also why a name may
not begin with a sequence number: two people both reach for the next one.

Nothing is duplicated into the spec that an artifact already records. A spec restating
hyperparameters would be a second source of truth, and a second source of truth drifts
-- which is the failure this exists to end. What the spec adds is exactly what no
artifact holds: the intended chain, who ran it, when, and which provider served each
auxiliary role.

**The join is by reference, never by name.** `sdf_finetune_result.json` records
`config.synthdoc_path`; `scoreboard.json` records `models{run_dir}`. Those edges are
reliable where the names are not, so `resolve` walks them and treats a directory name
as a label.

**Existence is not validated at load.** Only shape is. A spec names paths under
`data/`, which is not in git and arrives by `sdf sync pull` -- so a load-time existence
check would make every spec unloadable on a fresh clone, and would also fail for the
ordinary case of a collaborator having produced an artifact somewhere else. Missing
artifacts surface from `resolve` as informational, not as errors.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from sdf import paths, provenance

LOGGER = logging.getLogger(__name__)

# The statuses themselves are the Literal on `ExperimentSpec.status`, so there is one
# copy: `active` is work in progress, `final` is a result worth citing, `superseded`
# points forward via `notes`, `contaminated` is a run known to be invalid -- the state
# otherwise encoded only in a directory name, where nothing can read it -- and `smoke` marks a wiring test.
#
# These two assert something a reader cannot verify from any artifact, so both have to
# say why in prose.
STATUSES_REQUIRING_NOTES = ("superseded", "contaminated")

SEVERITIES = ("error", "warning", "info")

# `2026-08-13_<name>`. A dataset is dated so a directory listing reads
# chronologically, which is the first question a collaborator asks of a shared tree.
DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_")

# Derived from `paths`, not restated, so the layout stays defined in one place -- but
# kept relative so `resolve(root=...)` can read a fixture tree. Everything a spec names
# is relative to the repo root for the same reason: a spec is shared, and an absolute
# path is only ever true on one machine (three of the runs on disk record absolute
# paths from three different people's home directories).
_RESULTS_REL = paths.RESULTS_DIR.relative_to(paths.SDF_ROOT)


class Provider(BaseModel):
    """One auxiliary LLM role: which API served it, and which model.

    Both, not just the model. `providers.is_openrouter` infers the API from the id's
    *shape* -- "has a slash" -- and that is a convention rather than a guarantee. The
    same string `openai/gpt-5.6-luna` means OpenRouter to safetytooling and OpenAI to
    Inspect, which is precisely the distinction a bare model id cannot carry. Writing
    the resolved answer down beside the id lets `check` compare the two.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model: str
    api: str | None = None

    @model_validator(mode="after")
    def _api_matches_id(self) -> Provider:
        if self.api is None:
            # Filled in rather than demanded: `sdf experiments new` writes it, and a
            # hand-authored spec should not have to repeat what the id already implies.
            object.__setattr__(self, "api", provenance.api_of(self.model))
        return self


class CorpusSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str
    # A checkable pointer for the `_filtered` / `_merged` corpora, which are produced by
    # no code in this repo -- the flagship adapter trained on one of them. This cannot
    # make them reproducible; it can stop them being anonymous.
    derived_from: str | None = None
    generator: Provider | None = None


class RecallSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_bank: str | None = None
    judge: Provider | None = None
    generator: Provider | None = None


class InspectProvider(Provider):
    """A role Inspect calls, where the provider prefix decides the API, not the shape.

    `openai/gpt-5.6-luna` is OpenRouter to safetytooling and OpenAI to Inspect, so the
    two sides cannot share one default.
    """

    @model_validator(mode="after")
    def _api_matches_id(self) -> InspectProvider:
        if self.api is None:
            object.__setattr__(self, "api", provenance.inspect_api_of(self.model))
        return self


class RunEvals(BaseModel):
    """What one run measured, and with what."""

    model_config = ConfigDict(extra="forbid")

    # The results directory, i.e. `data/results/<experiment>/`. Separate from the run's
    # own name because published directories are not worth renaming.
    experiment: str
    evals: list[str] = Field(default_factory=list)
    # Model names, resolved against the dataset's `models/` directory, plus any bare
    # base-model ids the run compared against.
    models: list[str] = Field(default_factory=list)
    baseline: str | None = None
    parser: InspectProvider | None = None

    @model_validator(mode="after")
    def _baseline_is_one_of_the_models(self) -> RunEvals:
        if self.baseline and self.baseline not in self.models:
            raise ValueError(
                f"baseline {self.baseline!r} is not one of this run's models "
                f"({', '.join(self.models) or 'none'}); deltas would be silently omitted"
            )
        return self


class ScanSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judge: InspectProvider | None = None
    screens: list[str] = Field(default_factory=list)


class Named(BaseModel):
    """Common to all three levels: a filename-safe name, an owner and a time."""

    model_config = ConfigDict(extra="forbid")

    name: str
    owner: str | None = None
    created_at: str | None = None
    notes: str = ""

    @field_validator("name")
    @classmethod
    def _name_is_a_filename(cls, value: str) -> str:
        if not value or "/" in value or value != value.strip():
            raise ValueError(f"name {value!r} must be a bare filename stem with no slashes")
        if value[:1].isdigit() and not DATE_PREFIX.match(value):
            # `003-qwen-eval` and friends. A sequence number carries no information, and
            # in a shared repo two people both reach for the next one -- a merge conflict
            # in exactly the multi-person case this layout exists for.
            #
            # An ISO date is the one numeric prefix that is allowed, because it is a
            # fact rather than a counter: it says when, it sorts chronologically in a
            # directory listing, and two people generating it on the same day agree by
            # construction instead of colliding.
            raise ValueError(
                f"name {value!r} starts with a number that is not a date. Name it for what "
                "it is (model, eval, condition), optionally prefixed `YYYY-MM-DD_`."
            )
        return value


class DatasetSpec(Named):
    """A corpus, declared once, by whoever generated it.

    A pointer plus provenance: the documents themselves stay under `data/`, which is
    not in git. What is recorded here is what no artifact reliably holds -- the
    universe it came from, and the model that wrote it.
    """

    universe: str | None = None
    corpus: CorpusSpec | None = None
    hf_sha: str | None = None


class ModelEntry(Named):
    """One adapter trained against this dataset, or one base model it is compared to.

    Its own file, because several people train against one dataset at different times
    and a shared `models.json` would put two authors on the same lines.
    """

    kind: Literal["base", "adapter"]
    model: str | None = None
    save_folder: str | None = None
    formatting: Literal["raw_continuation", "doctag", "conversation", "mixed", "distill"] | None = None
    checkpoint: str | None = None
    sampler_path: str | None = None
    renderer: str | None = None

    @property
    def run_dir(self) -> str | None:
        if self.kind != "adapter" or not self.save_folder:
            return None
        return str(Path(self.save_folder) / (self.formatting or "raw_continuation"))

    @property
    def checkpoint_name(self) -> str:
        return self.checkpoint or "final"

    def binding(self) -> dict[str, Any]:
        """The shape `targets.resolve(bindings=...)` accepts."""
        entry: dict[str, Any] = {"kind": self.kind}
        for key, value in (
            ("model", self.model),
            ("run_dir", self.run_dir),
            ("sampler_path", self.sampler_path),
            ("checkpoint", self.checkpoint),
            ("renderer", self.renderer),
        ):
            if value:
                entry[key] = value
        return entry

    @model_validator(mode="after")
    def _kind_has_what_it_needs(self) -> ModelEntry:
        if self.kind == "adapter" and not (self.save_folder or (self.sampler_path and self.model)):
            raise ValueError(
                "an adapter needs a 'save_folder', or a 'sampler_path' with the base 'model'"
            )
        if self.kind == "base" and not self.model:
            raise ValueError("a base model needs a 'model' id")
        if self.kind == "base" and self.checkpoint:
            raise ValueError("a base model has no checkpoints; drop 'checkpoint'")
        return self


class RunEntry(Named):
    """One measurement against this dataset: which models, which evals, by whom.

    Its own file for the same reason as a model. Person A evaluating `m1` with `e1`
    and person B evaluating `m2` with `e1` are two entries that never touch.
    """

    question: str = ""
    status: Literal["active", "final", "superseded", "contaminated", "smoke"] = "active"
    recall: RecallSpec | None = None
    evals: RunEvals | None = None
    scan: ScanSpec | None = None
    hf_sha: str | None = None

    @model_validator(mode="after")
    def _unverifiable_states_explain_themselves(self) -> RunEntry:
        if self.status in STATUSES_REQUIRING_NOTES and not self.notes.strip():
            raise ValueError(
                f"status {self.status!r} needs 'notes' saying why -- that claim cannot be "
                "checked against any artifact, so prose is the only place it can live"
            )
        return self


@dataclass
class Dataset:
    """A dataset directory: the declaration, its models, and its runs."""

    spec: DatasetSpec
    models: dict[str, ModelEntry] = field(default_factory=dict)
    runs: dict[str, RunEntry] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    def bindings(self, run: RunEntry) -> dict[str, dict[str, Any]]:
        """The model bindings one run needs, resolved against this dataset's models.

        A name the dataset does not declare raises rather than being skipped: a suite
        that quietly evaluated two models when the run asked for three still prints a
        complete-looking table.
        """
        wanted = (run.evals.models if run.evals else []) or list(self.models)
        unknown = [m for m in wanted if m not in self.models]
        if unknown:
            raise KeyError(
                f"run {run.name!r} names model(s) {', '.join(unknown)}, which dataset "
                f"{self.name!r} does not declare. Declared: {', '.join(sorted(self.models)) or 'none'}"
            )
        return {name: self.models[name].binding() for name in wanted}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

DATASET_FILE = "dataset.json"
MODELS_DIR = "models"
RUNS_DIR = "runs"


def _read(path: Path, cls: type, name: str | None = None):
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(raw).__name__}.")  # noqa: TRY004

    raw.setdefault("name", name or path.stem)
    try:
        parsed = cls.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path} is not a valid {cls.__name__}:\n{exc}") from exc
    expected = name or path.stem
    if parsed.name != expected:
        # The same rule `registry._read_manifest` applies to eval.json: two names for
        # one thing means half the tooling keys on the wrong one.
        raise ValueError(f"{path}: name {parsed.name!r} does not match its filename stem {expected!r}.")
    return parsed


def _read_dataset(directory: Path) -> Dataset:
    dataset = Dataset(spec=_read(directory / DATASET_FILE, DatasetSpec, name=directory.name))
    for path in sorted((directory / MODELS_DIR).glob("*.json")):
        entry = _read(path, ModelEntry)
        dataset.models[entry.name] = entry
    for path in sorted((directory / RUNS_DIR).glob("*.json")):
        entry = _read(path, RunEntry)
        dataset.runs[entry.name] = entry
    return dataset


def discover(experiments_dir: Path | str | None = None) -> dict[str, Dataset]:
    """Every dataset under `experiments/`, keyed by name.

    A dataset is a directory holding `dataset.json`. Anything else -- the shared
    `models.json`, a README -- is not one.
    """
    root = Path(experiments_dir) if experiments_dir else paths.EXPERIMENTS_DIR
    if not root.is_dir():
        return {}
    found: dict[str, Dataset] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if directory.name.startswith("_") or not (directory / DATASET_FILE).is_file():
            continue
        dataset = _read_dataset(directory)
        found[dataset.name] = dataset
    return found


def load(name: str, experiments_dir: Path | str | None = None) -> Dataset:
    """One dataset, read directly rather than by discovering every other one.

    Reading only what was asked for matters: `discover` parses every dataset in the
    directory and raises on the first malformed one, so one typo'd file would break
    `--dataset` for every unrelated dataset and every stage.
    """
    root = Path(experiments_dir) if experiments_dir else paths.EXPERIMENTS_DIR
    directory = root / name
    if not (directory / DATASET_FILE).is_file():
        known = ", ".join(sorted(discover(experiments_dir))) or "(none found)"
        raise KeyError(f"No dataset named {name!r} in {root}. Available: {known}")
    return _read_dataset(directory)


def load_optional(name: str | None, experiments_dir: Path | str | None = None) -> Dataset | None:
    """The dataset for `name`, or None when there is not one.

    Absence is the normal case for every run predating the catalogue. A dataset that
    *exists and is malformed* still raises, because that is a typo in a declaration
    somebody is relying on.
    """
    if not name:
        return None
    try:
        return load(name, experiments_dir)
    except KeyError:
        return None


def resolve_run(
    dataset: str | None, run: str | None, experiments_dir: Path | str | None = None
) -> tuple[Dataset | None, RunEntry | None]:
    """`--dataset d --run r` -> the pair, with a clear error when the run is unknown."""
    found = load_optional(dataset, experiments_dir)
    if found is None or not run:
        return found, None
    if run not in found.runs:
        known = ", ".join(sorted(found.runs)) or "(none)"
        raise KeyError(f"Dataset {found.name!r} has no run named {run!r}. Declared: {known}")
    return found, found.runs[run]


def save_dataset(spec: DatasetSpec, experiments_dir: Path | str | None = None) -> Path:
    root = Path(experiments_dir) if experiments_dir else paths.EXPERIMENTS_DIR
    directory = root / spec.name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DATASET_FILE
    path.write_text(json.dumps(spec.model_dump(exclude_none=True), indent=2) + "\n")
    return path


def save_entry(dataset: str, entry: Named, experiments_dir: Path | str | None = None) -> Path:
    """Write one model or run entry. An append is a new file, never an edit."""
    root = Path(experiments_dir) if experiments_dir else paths.EXPERIMENTS_DIR
    subdir = MODELS_DIR if isinstance(entry, ModelEntry) else RUNS_DIR
    directory = root / dataset / subdir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{entry.name}.json"
    path.write_text(json.dumps(entry.model_dump(exclude_none=True), indent=2) + "\n")
    return path


# --------------------------------------------------------------------------------------
# Resolving: what the artifacts actually say
# --------------------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any] | None:
    """Parse a JSON file, or None if it is absent or unreadable.

    Absence is the normal case here, not an error: the resolver runs against whatever
    a machine happens to have, and reports what it could not find rather than raising.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class Resolved:
    """What each stage's own output file records, or None where nothing was found."""

    corpus: dict[str, Any] | None = None
    adapters: dict[str, dict[str, Any]] = field(default_factory=dict)
    recall: dict[str, Any] | None = None
    results: dict[str, Any] | None = None
    missing: list[str] = field(default_factory=list)


def _resolve_corpus(dataset: Dataset, root: Path) -> dict[str, Any] | None:
    if not dataset.spec.corpus:
        return None
    directory = root / dataset.spec.corpus.dir
    report = read_json(directory / "generation_report.json")
    if report is not None:
        config = report.get("config") or {}
        return {
            "dir": str(directory),
            "universe_id": report.get("universe_id"),
            "generator_model": config.get("model"),
            # Resolved by the generator; absent from reports written before planning
            # could use its own model, which all planned with `model`.
            "ideation_model": report.get("ideation_model") or config.get("model"),
            "total_documents": report.get("total_documents"),
            "facts_without_documents": report.get("facts_without_documents") or [],
            "key_facts_fingerprint": report.get("key_facts_fingerprint"),
        }
    legacy = read_json(directory / "generation_config.json")
    if legacy is not None:
        return {
            "dir": str(directory),
            "universe_id": None,
            "generator_model": legacy.get("model"),
            "total_documents": legacy.get("num_docs_generated"),
            "facts_without_documents": [],
            "key_facts_fingerprint": None,
        }
    if directory.is_dir():
        # Present but unattributed -- around a third of the corpora on disk, including
        # several a real training run consumed. Distinguished from absent on purpose:
        # "the corpus is here and we cannot say who wrote it" and "the corpus is on
        # someone else's machine" call for completely different responses.
        return {"dir": str(directory), "unattributed": True, "facts_without_documents": []}
    return None


def _resolve_adapter(run_dir: str, root: Path) -> dict[str, Any] | None:
    result = read_json(root / run_dir / "sdf_finetune_result.json")
    if result is None:
        return None
    config = result.get("config") or {}
    checkpoints = _read_checkpoints(root / run_dir / "checkpoints.jsonl")
    return {
        "run_dir": run_dir,
        "base_model": result.get("model"),
        "renderer_name": result.get("renderer_name"),
        "sampler_path": result.get("sampler_path"),
        "synthdoc_path": config.get("synthdoc_path"),
        "hf_sha": config.get("hf_sha"),
        "trained_at": result.get("trained_at"),
        "checkpoints": [c.get("name") for c in checkpoints],
        # One run is not necessarily one Tinker session: a resumed run writes further
        # checkpoints under a fresh uuid and step numbers then repeat across them, so
        # "step 400" is ambiguous unless you know how many sessions there were.
        "checkpoint_sessions": sorted({_session_of(c.get("sampler_path")) for c in checkpoints} - {None}),
    }


def _session_of(sampler_path: str | None) -> str | None:
    """`tinker://<uuid>:train:0/sampler_weights/000400` -> the uuid."""
    if not sampler_path or "://" not in sampler_path:
        return None
    return sampler_path.split("://", 1)[1].split(":", 1)[0] or None


def _read_checkpoints(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _results_dir(run: RunEntry) -> str | None:
    return run.evals.experiment if run.evals else None


def resolve(dataset: Dataset, run: RunEntry | None = None, root: Path | str | None = None) -> Resolved:
    """Read back what the artifacts say about one run of one dataset.

    Without a run, resolves the dataset's corpus and every model it declares -- which
    is what `sdf experiments show <dataset>` needs.
    """
    base = Path(root) if root else paths.SDF_ROOT
    out = Resolved(corpus=_resolve_corpus(dataset, base))
    if dataset.spec.corpus and out.corpus is None:
        out.missing.append("corpus")

    for name in (run.evals.models if run and run.evals and run.evals.models else list(dataset.models)):
        entry = dataset.models.get(name)
        if entry is None or entry.kind != "adapter":
            continue
        run_dir = entry.run_dir
        found = _resolve_adapter(run_dir, base) if run_dir else None
        if found is not None:
            out.adapters[name] = found
        elif run_dir:
            out.missing.append(f"model:{name}")

    experiment = _results_dir(run) if run else None
    if experiment:
        directory = base / _RESULTS_REL / experiment
        board = read_json(directory / "scoreboard.json")
        if board is not None:
            out.results = {
                "experiment": board.get("experiment"),
                "baseline": board.get("baseline"),
                "hf_sha": board.get("hf_sha"),
                "models": board.get("models") or {},
                "evals": list((board.get("evals") or {}).keys()),
                "cells": board.get("cells") or [],
                "provenance": board.get("provenance"),
            }
        elif directory.is_dir():
            out.results = {
                "experiment": experiment,
                "no_scoreboard": True,
                "transcripts": len(list(directory.rglob("*.eval"))),
            }
        else:
            out.missing.append("results")
        out.recall = read_json(directory / "recall.json")
    return out


# --------------------------------------------------------------------------------------
# Diffing: where declaration and fact disagree
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    severity: str
    stage: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity:<7}] {self.stage}: {self.message}"


def _same_path(a: str | None, b: str | None) -> bool:
    """Compare two recorded paths without caring about `./` or a trailing slash."""
    if not a or not b:
        return False
    # rstrip, not strip: stripping both ends removes the leading slash, so an absolute
    # path recorded on another machine compared equal to a relative one here and the
    # corpus-mismatch error silently failed to fire.
    return Path(a).as_posix().rstrip("/") == Path(b).as_posix().rstrip("/")


def _denotes(run_dir: str, entry: ModelEntry) -> bool:
    """Does a scoreboard's `run_dir` select the weights this model entry declares?

    Not string equality, because one set of weights has two spellings on disk. A
    checkpoint is sampled by pointing the provider at a stub `sdf_finetune_result.json`
    written beside it -- `<save_folder>/checkpoints/step_NNNNNN` -- while the entry
    stores the run itself plus a `checkpoint` name. Comparing the strings reported every
    checkpoint column in a sweep as undeclared, which is thirteen errors saying nothing
    on this repo and, worse, buries the one that would mean something.

    The `checkpoints/` segment is not assumed: any path under the save folder whose last
    segment names the checkpoint denotes it.
    """
    if _same_path(run_dir, entry.run_dir):
        return True
    if not (entry.checkpoint and entry.save_folder):
        return False
    path = Path(run_dir)
    folder = Path(entry.save_folder)
    return path.is_relative_to(folder) and path.name.removeprefix("step_") == entry.checkpoint


def diff(dataset: Dataset, run: RunEntry | None, resolved: Resolved) -> list[Divergence]:
    """Every disagreement between what was declared and what the artifacts record.

    Three severities, and the split matters as much as the checks:

    - **error** -- the declaration describes a chain that did not happen. A number read
      under it would be attributed to the wrong corpus, universe or weights.
    - **warning** -- the run happened as declared, but something undermines reading the
      number: an unpinned corpus, a fact with no documents, a cell scored by two judges.
    - **info** -- worth knowing, not wrong. Missing artifacts live here deliberately:
      collaborators produce things on other machines and that is normal.
    """
    out: list[Divergence] = []

    def add(severity: str, stage: str, message: str) -> None:
        out.append(Divergence(severity, stage, message))

    for label in resolved.missing:
        add("info", label.split(":")[0], f"{label} declared but no artifact found on this machine")

    corpus, results = resolved.corpus, resolved.results
    spec = dataset.spec

    # -- the chain ------------------------------------------------------------------
    if corpus and spec.universe and corpus.get("universe_id") and corpus["universe_id"] != spec.universe:
        add("error", "corpus", f"built from universe {corpus['universe_id']!r}, but the dataset declares {spec.universe!r}")

    if spec.corpus:
        declared = str(Path(spec.corpus.dir) / "synth_docs.jsonl")
        for name, found in resolved.adapters.items():
            trained_on = found.get("synthdoc_path")
            if trained_on and not _same_path(trained_on, declared):
                add(
                    "error",
                    "models",
                    f"model {name!r} trained on {trained_on!r}, which is not this dataset's corpus "
                    f"{spec.corpus.dir!r} -- it belongs to a different dataset",
                )

    if run and run.evals and results and not results.get("no_scoreboard"):
        declared_models = [dataset.models[m] for m in run.evals.models if m in dataset.models]
        for label, entry in (results.get("models") or {}).items():
            run_dir = (entry or {}).get("run_dir")
            if run_dir and not any(_denotes(run_dir, m) for m in declared_models):
                add("error", "evals", f"scoreboard model {label!r} points at {run_dir!r}, which this run does not declare")

    # -- weights identity -----------------------------------------------------------
    for name, entry in dataset.models.items():
        found = resolved.adapters.get(name)
        if not (found and entry.sampler_path and found.get("sampler_path")):
            continue
        if entry.checkpoint_name == "final" and entry.sampler_path != found["sampler_path"]:
            add(
                "error",
                "models",
                f"model {name!r} pins {entry.sampler_path!r}, but {entry.run_dir} now records "
                f"{found['sampler_path']!r} -- that run directory was retrained in place, so every "
                "result naming it now points at different weights",
            )

    # -- readability of the number --------------------------------------------------
    if corpus:
        empty = corpus.get("facts_without_documents") or []
        if empty:
            add(
                "warning",
                "corpus",
                f"{len(empty)} key fact(s) produced no documents ({', '.join(sorted(empty)[:4])}"
                f"{', ...' if len(empty) > 4 else ''}); per-fact recall on those can only be chance",
            )
        if corpus.get("generator_model") is None and not (spec.corpus and spec.corpus.generator):
            add("info", "corpus", "no generation report records which model wrote the documents")

    if spec.corpus and spec.corpus.derived_from:
        add("info", "corpus", f"derived from {spec.corpus.derived_from!r}; no code in this repo reproduces derived corpora")

    if results and results.get("no_scoreboard"):
        add(
            "info",
            "evals",
            f"{results.get('transcripts', 0)} transcript(s) but no scoreboard, so there is no recorded "
            "metric, baseline or delta -- `sdf evals run` writes one, raw `inspect eval` does not",
        )
    elif results:
        if not results.get("hf_sha") and not (run and run.hf_sha) and not spec.hf_sha:
            add("warning", "evals", "no hf_sha pins the corpus revision behind these results")
        for cell in results.get("cells") or []:
            scorers = cell.get("scorers")
            if isinstance(scorers, list) and len(scorers) > 1:
                add(
                    "warning",
                    "evals",
                    f"cell {cell.get('eval')}/{cell.get('model')} was scored by {len(scorers)} scorers "
                    f"({', '.join(map(str, scorers))}); the headline value uses only one",
                )
        limit = (results.get("provenance") or {}).get("limit")
        if limit:
            add(
                "warning",
                "evals",
                f"run with --limit {limit}; datasets are condition-major, so whole conditions are "
                "missing from a table that looks complete",
            )
        # An eval present in the results and absent from the declaration is a run
        # nobody declared. Silent until now, which defeats the point of declaring.
        if run and run.evals:
            undeclared = sorted(set(results.get("evals") or []) - set(run.evals.evals))
            if undeclared:
                add("warning", "evals", f"results contain eval(s) this run does not declare: {', '.join(undeclared)}")

    # -- checkpoints ----------------------------------------------------------------
    for name, found in resolved.adapters.items():
        if len(found.get("checkpoint_sessions") or []) > 1:
            add(
                "warning",
                "models",
                f"model {name!r} spans {len(found['checkpoint_sessions'])} Tinker sessions, so a step "
                "number does not identify a checkpoint on its own",
            )
    for name, entry in dataset.models.items():
        found = resolved.adapters.get(name)
        if entry.kind == "adapter" and entry.checkpoint_name != "final" and not (found or {}).get("trained_at"):
            add(
                "warning",
                "models",
                f"model {name!r} uses checkpoint {entry.checkpoint_name!r}; only 'final' is kept "
                "indefinitely and that run records no training date, so its expiry is unknowable",
            )

    # -- declaration self-consistency -----------------------------------------------
    for role, declared in providers_of(dataset, run).items():
        inspect_side = role in ("eval_parser", "scan_judge")
        actual = provenance.inspect_api_of(declared["model"]) if inspect_side else provenance.api_of(declared["model"])
        if declared.get("api") and actual and declared["api"] != actual:
            add("error", "providers", f"{role} declares api {declared['api']!r} but model id {declared['model']!r} routes to {actual!r}")

    for name, entry in dataset.models.items():
        served = entry.renderer
        trained = (resolved.adapters.get(name) or {}).get("renderer_name")
        if entry.kind == "adapter" and served and trained and served != trained:
            add(
                "info",
                "models",
                f"model {name!r} is served with renderer {served!r} but was trained under {trained!r} "
                "-- legitimate, but results across the two are not comparable",
            )

    return out


def providers_of(dataset: Dataset, run: RunEntry | None) -> dict[str, dict[str, Any]]:
    """Every auxiliary role declared by a dataset and one of its runs."""
    roles: dict[str, Any] = {"doc_gen": dataset.spec.corpus.generator if dataset.spec.corpus else None}
    if run:
        roles["recall_judge"] = run.recall.judge if run.recall else None
        roles["recall_generator"] = run.recall.generator if run.recall else None
        roles["eval_parser"] = run.evals.parser if run.evals else None
        roles["scan_judge"] = run.scan.judge if run.scan else None
    return {name: role.model_dump() for name, role in roles.items() if role is not None}


def worst(divergences: list[Divergence]) -> str | None:
    for severity in SEVERITIES:
        if any(d.severity == severity for d in divergences):
            return severity
    return None
