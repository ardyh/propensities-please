"""Recovering what an existing run was, and finding what nothing points at.

`sdf/experiments.py` reads a spec and compares it against the artifacts. This module
is the other direction: it builds a spec *from* the artifacts, for the runs that
predate the catalogue, and it answers "does anything still refer to this?"

Both jobs are the same walk over the same reference graph:

    corpus  <- tinker_run.config.synthdoc_path
    run     <- scoreboard.models{run_dir}, spec.train.save_folder, spec.evals.models,
               tinker_run.config.load_checkpoint_path (through the ledger holding it)
    results <- spec.evals.experiment

Names are never used as the key. On this repo 89 distinct names span the three
directory trees and only two appear in all of them, so a name-based rule is wrong
about most artifacts. It is also dangerous in a way the reference graph is not: three
artifacts whose names said `pilot` or `test` were consumed by real training runs or named by a
real scoreboard. A cleanup keyed on the name would have deleted all three.

Inspect's own `.eval` headers are the richest source here, and the only one that
carries a wall-clock time: `eval.created`, plus the task, the model, the model_args
that name a run directory, and the git commit with its dirty flag. That is what makes
a flat pile of logs -- seven of the result directories are exactly that, produced by
raw `inspect eval` rather than by the suite -- adoptable at all.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sdf import experiments as ex
from sdf import paths
from sdf.experiments import DATE_PREFIX

LOGGER = logging.getLogger(__name__)

# Names that suggest a wiring test rather than a result. Used only to *propose* a
# status for a human to confirm, never to decide what may be deleted -- see the module
# docstring for the three artifacts that would have been lost that way.
SMOKE_HINT = re.compile(r"smoke|pilot|dry_run|scratch|_test\b|^test|^tonight$", re.IGNORECASE)


@dataclass
class LogFacts:
    """What one `.eval` header says about the run that produced it."""

    task: str | None = None
    model: str | None = None
    run_dir: str | None = None
    model_path: str | None = None
    base_model: str | None = None
    renderer: str | None = None
    created: str | None = None


def read_log_headers(directory: Path, limit: int | None = None) -> list[LogFacts]:
    """Header facts for every `.eval` under `directory`, recursively.

    Headers only: a full read pulls every sample into memory, and these directories
    hold 100MB of transcripts. Unreadable logs are skipped rather than fatal, because
    the point of adoption is to recover whatever survives.
    """
    from inspect_ai.log import read_eval_log

    found: list[LogFacts] = []
    for path in sorted(directory.rglob("*.eval"))[:limit]:
        try:
            header = read_eval_log(str(path), header_only=True)
        except Exception as exc:  # noqa: BLE001 - a corrupt log is a thing to skip
            LOGGER.debug("Could not read %s: %s", path, exc)
            continue
        spec = header.eval
        args = dict(getattr(spec, "model_args", None) or {})
        found.append(
            LogFacts(
                task=getattr(spec, "task", None),
                model=getattr(spec, "model", None),
                run_dir=args.get("run_dir"),
                model_path=args.get("model_path"),
                base_model=args.get("base_model"),
                renderer=args.get("renderer_name"),
                created=str(getattr(spec, "created", "") or "") or None,
            )
        )
    return found


def _binding_from_model_string(model: str | None, facts: LogFacts) -> tuple[str, dict[str, Any]] | None:
    """`tinker/<base>` or `tinker/<label>` plus model_args -> a label and a binding."""
    if not model:
        return None
    suffix = model.split("/", 1)[1] if "/" in model else model
    if facts.run_dir or facts.model_path:
        # An adapter. The string after `tinker/` is a free label, so it is the label --
        # and it is *only* a label, which is why `base_model` has to be carried
        # separately when a checkpoint sweep pinned the weights without a run
        # directory. Reading the label as a model id records `mix-000020` as a model.
        binding: dict[str, Any] = {"kind": "adapter"}
        if facts.run_dir:
            binding["run_dir"] = facts.run_dir
        if facts.model_path:
            binding["sampler_path"] = facts.model_path
        if facts.base_model and not facts.run_dir:
            binding["model"] = facts.base_model
        if facts.renderer:
            binding["renderer"] = facts.renderer
        return suffix, binding
    binding = {"kind": "base", "model": suffix}
    if facts.renderer:
        binding["renderer"] = facts.renderer
    return suffix, binding


def binding_from_log(header: Any) -> dict[str, Any] | None:
    """The model binding one `.eval` header records, in `data/models.json`'s shape.

    The same recovery `adopt` does, exposed because a scoreboard needs it too:
    `data/models.json` is edited in place and is not in git, so a label a board was
    written under routinely no longer resolves there -- and a board that then records
    nothing about the column is not merely untidy, since `reference_graph` follows
    `models{run_dir}` to the training run behind it.
    """
    spec = getattr(header, "eval", None)
    if spec is None:
        return None
    args = dict(getattr(spec, "model_args", None) or {})
    facts = LogFacts(
        model=getattr(spec, "model", None),
        run_dir=args.get("run_dir"),
        model_path=args.get("model_path"),
        base_model=args.get("base_model"),
        renderer=args.get("renderer_name"),
    )
    found = _binding_from_model_string(facts.model, facts)
    return found[1] if found else None


# The formattings `ModelEntry` stores a run directory as. Written out rather than read
# off the annotation so the fallback below has something to probe the disk with. Keep in
# step with `finetune.Formatting`: a formatting missing here makes every column of that
# run fail to adopt, and the traceback blames the validator rather than the shape.
FORMATTINGS = ("raw_continuation", "doctag", "conversation", "mixed", "distill")


def split_run_dir(run_dir: str, root: Path | None = None) -> dict[str, Any]:
    """A `run_dir` as the `ModelEntry` fields that reproduce it.

    `<save_folder>/<formatting>` is what a run directory normally is, and that pair is
    what the entry stores. A checkpoint sweep records something else: the provider
    accepts any directory holding an `sdf_finetune_result.json`, so a stub is written
    beside each checkpoint and twelve columns on this repo name
    `<save_folder>/checkpoints/step_NNNNNN`. Splitting that on the last segment set
    `formatting="step_000060"`, which the Literal rejects -- so every checkpoint column
    failed to adopt, and the traceback blamed the validator rather than the shape.

    The stub is also the better source: it names the `sampler_path`, and the URI is the
    only thing that identifies weights. This repo's `checkpoints.jsonl` holds two
    entries called `final` and two called `000240`, from the two Tinker sessions of one
    resumed run, so resolving a checkpoint by *name* is ambiguous where the URI is not.
    """
    base = Path(root) if root else paths.SDF_ROOT
    path = Path(run_dir)
    if path.name in FORMATTINGS:
        return {"save_folder": str(path.parent), "formatting": path.name}

    result = ex.read_json(base / run_dir / "sdf_finetune_result.json") or {}
    payload: dict[str, Any] = {"checkpoint": path.name.removeprefix("step_")}
    for key, field_name in (("sampler_path", "sampler_path"), ("model", "model"), ("renderer_name", "renderer")):
        if result.get(key):
            payload[field_name] = result[key]

    # Which folder holds the training run itself. Nothing in this repo *writes* the
    # checkpoint layout -- it was made by hand -- so it is read off the disk rather than
    # assumed, preferring the formatting the stub names.
    wanted = [result["formatting"]] if result.get("formatting") in FORMATTINGS else list(FORMATTINGS)
    for ancestor in path.parents:
        for formatting in wanted:
            if (base / ancestor / formatting).is_dir():
                payload["save_folder"] = str(ancestor)
                payload["formatting"] = formatting
                return payload
    if not (payload.get("sampler_path") and payload.get("model")):
        raise ValueError(
            f"{run_dir} is neither <save_folder>/<formatting> nor a checkpoint stub this "
            "can trace back to one: no sdf_finetune_result.json naming the weights, and no "
            f"ancestor holding a {' or '.join(FORMATTINGS)} directory."
        )
    return payload


def entry_name(label: str, taken: set[str]) -> str:
    """A filename-safe, descriptive name for a model entry.

    A scoreboard names models by their manifest label, which is already clean. A log
    names them by the provider string -- `openai/gpt-oss-20b` -- which cannot be a
    filename. The last segment is the descriptive part, so prefer it, and fall back to
    the whole dashed slug when two vendors ship the same model name.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
    short = cleaned.rsplit("-", 1)[-1] if "/" in label else cleaned
    tail = re.sub(r"[^A-Za-z0-9._-]+", "-", label.rsplit("/", 1)[-1]).strip("-")
    for candidate in (tail, cleaned, short):
        if candidate and candidate not in taken:
            return candidate
    suffix = 2
    while f"{cleaned}-{suffix}" in taken:
        suffix += 1
    return f"{cleaned}-{suffix}"


def _same_weights(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two bindings that select the same weights, whatever they are labelled."""
    if a.get("kind") != b.get("kind"):
        return False
    if a.get("kind") == "adapter":
        return a.get("run_dir") == b.get("run_dir") and a.get("sampler_path") == b.get("sampler_path")
    return a.get("model") == b.get("model") and a.get("renderer") == b.get("renderer")


def _run_of_state_paths(root: Path) -> dict[str, str]:
    """Which training run's ledger holds each `tinker://` checkpoint path.

    A second stage (`--load_checkpoint_path`, with or without documents) records the
    checkpoint it started from, and that path is the only thing tying it to the run --
    and so to the dataset -- whose adapter it continued. The ledgers are the sole map
    from a Tinker path back to a directory: nothing in the path names the run.
    """
    index: dict[str, str] = {}
    for run in _dirs("tinker_runs", root):
        for ledger in sorted((root / run).rglob("checkpoints.jsonl")):
            for line in ledger.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("state_path", "sampler_path"):
                    if row.get(key):
                        index.setdefault(str(row[key]), run)
    return index


def _corpus_of_run(run_dir: str, root: Path, _depth: int = 0) -> str | None:
    """The corpus directory a training run recorded that it consumed.

    A run initialised from another run's checkpoint belongs to *that* run's dataset:
    a `distill` repair records no corpus at all, and a `conversation` repair's
    `synthdoc_path` is its rollouts file. So `load_checkpoint_path` is followed first,
    through the ledger that holds it, and the run's own path is the fallback.
    """
    result = ex.read_json(root / run_dir / "sdf_finetune_result.json")
    config = ((result or {}).get("config") or {})
    initial = config.get("load_checkpoint_path")
    if initial and _depth < 4:
        parent = _run_of_state_paths(root).get(str(initial))
        if parent and Path(parent) not in (Path(run_dir), *Path(run_dir).parents):
            for result_file in sorted((root / parent).rglob("sdf_finetune_result.json")):
                corpus = _corpus_of_run(str(result_file.parent.relative_to(root)), root, _depth + 1)
                if corpus:
                    return corpus
    synthdoc = config.get("synthdoc_path")
    return str(Path(synthdoc).parent) if synthdoc else None


def _universe_of_corpus(corpus_dir: str, root: Path) -> str | None:
    report = ex.read_json(root / corpus_dir / "generation_report.json")
    return (report or {}).get("universe_id")


def _run_name(experiment: str, models: list[str], evals: list[str]) -> str:
    """A descriptive name for one measurement: which model, which eval.

    Falls back to the results directory name when nothing else is recoverable -- that
    is at least what a human chose at the time.
    """
    # "multi-model", not "4-models": a name may not begin with a digit, and the count
    # is less useful than the fact that this run compared several.
    parts = [models[0]] if len(models) == 1 else (["multi-model"] if models else [])
    parts += evals[:2] if evals else []
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(parts)).strip("-")
    return slug or re.sub(r"[^A-Za-z0-9._-]+", "-", experiment).strip("-")


def _dated_name(stem: str, corpus_dir: str | None, results: Path, base: Path) -> tuple[str, str]:
    """`YYYY-MM-DD_<stem>`, and a note saying where that date came from.

    A dataset directory is dated so a listing reads chronologically, which is the first
    question a collaborator asks of a shared tree. Nothing records when a corpus was
    generated, so the date is a file mtime -- and that is worth saying out loud rather
    than implying a provenance record: `data/` was restored from HuggingFace on
    2026-08-11, so every artifact predating that restore carries the recovery date, not
    the date the work ran. Two people adopting the same corpus still agree, which is
    what matters: the date is a property of the corpus, so it cannot split one dataset
    in two.
    """
    if DATE_PREFIX.match(stem):
        return stem, ""
    anchor = None
    if corpus_dir:
        report = base / corpus_dir / "generation_report.json"
        anchor = report if report.exists() else (base / corpus_dir if (base / corpus_dir).exists() else None)
    if anchor is None:
        anchor = results if results.exists() else None
    if anchor is None:
        return stem, ""
    stamp = datetime.fromtimestamp(anchor.stat().st_mtime, UTC).astimezone().strftime("%Y-%m-%d")
    return f"{stamp}_{stem}", (
        f"The date prefix is the mtime of {anchor.name}, not a recorded generation time. "
        "`data/` was restored from HuggingFace on 2026-08-11, so a date on or after that "
        "may be a recovery date rather than when this work ran."
    )


@dataclass
class Adopted:
    """One results directory, recovered as a dataset plus the entries under it."""

    dataset: ex.DatasetSpec
    models: list[ex.ModelEntry] = field(default_factory=list)
    run: ex.RunEntry | None = None


def adopt(results_dir: Path | str, root: Path | None = None) -> Adopted:
    """Recover a dataset, its models and one run by walking references backwards.

    results -> scoreboard `models{run_dir}` (or the `.eval` headers when there is no
    scoreboard) -> `sdf_finetune_result.json` `config.synthdoc_path` -> the corpus ->
    `generation_report.json` `universe_id`.

    **The dataset is the corpus**, so two result directories that trained on the same
    corpus adopt into one dataset with two runs. When no corpus is recoverable -- ten
    of the twenty-one directories here, and every base-model-only comparison -- the
    dataset is named after the results directory and declares no corpus. That is
    honest rather than tidy: moving a run into the right dataset later is a `git mv`.

    `question` and `status` are left for a human. Adoption can recover what ran; it
    cannot recover what it was for, and a guess there is worse than a blank.
    """
    base = Path(root) if root else paths.SDF_ROOT
    directory = Path(results_dir)
    if not directory.is_absolute():
        directory = base / directory
    experiment = directory.name

    board = ex.read_json(directory / "scoreboard.json")
    logs = read_log_headers(directory)

    bindings: dict[str, dict[str, Any]] = {}
    eval_names: list[str] = []
    baseline = None
    if board:
        baseline = board.get("baseline")
        eval_names = sorted(board.get("evals") or {})
        for label, entry in (board.get("models") or {}).items():
            binding = {k: v for k, v in (entry or {}).items() if v is not None}
            binding.setdefault("kind", "adapter" if binding.get("run_dir") else "base")
            if binding["kind"] == "base":
                binding.pop("checkpoint", None)
                binding.pop("sampler_path", None)
            bindings[label] = binding
    for facts in logs:
        if facts.task and facts.task not in eval_names:
            eval_names.append(facts.task)
        found = _binding_from_model_string(facts.model, facts)
        if not found:
            continue
        label, binding = found
        if label in bindings or any(_same_weights(binding, existing) for existing in bindings.values()):
            continue
        bindings[label] = binding

    # -- back to the corpus, which is the dataset ------------------------------------
    corpus_dir = None
    for binding in bindings.values():
        if binding.get("run_dir"):
            corpus_dir = _corpus_of_run(binding["run_dir"], base)
            if corpus_dir:
                break
    recall = ex.read_json(directory / "recall.json")
    if not corpus_dir and recall and recall.get("synthdoc_path"):
        corpus_dir = str(Path(recall["synthdoc_path"]).parent)

    stem = Path(corpus_dir).name if corpus_dir else experiment
    dated, date_note = _dated_name(stem, corpus_dir, directory, base)
    dataset_payload: dict[str, Any] = {"name": dated, "notes": date_note}
    if corpus_dir:
        dataset_payload["corpus"] = {"dir": corpus_dir}
        universe = _universe_of_corpus(corpus_dir, base)
        if universe:
            dataset_payload["universe"] = universe
    else:
        dataset_payload["notes"] = (
            f"{date_note} No corpus recoverable from this run's artifacts, so the dataset "
            "is named after the results directory. Move the run into the right dataset "
            "when known."
        ).strip()

    # -- models, one entry each ------------------------------------------------------
    models: list[ex.ModelEntry] = []
    renamed: dict[str, str] = {}
    taken: set[str] = set()
    for label, binding in bindings.items():
        name = entry_name(label, taken)
        taken.add(name)
        renamed[label] = name
        payload: dict[str, Any] = {"name": name, "kind": binding.get("kind", "base")}
        if binding.get("run_dir"):
            payload.update(split_run_dir(binding["run_dir"], base))
        # After the split, not before: what the board or the manifest says about a
        # column outranks what a checkpoint stub happened to record beside the weights.
        for key in ("model", "sampler_path", "checkpoint", "renderer"):
            if binding.get(key):
                payload[key] = binding[key]
        models.append(ex.ModelEntry.model_validate(payload))

    # -- the run ---------------------------------------------------------------------
    provenance_block = (board or {}).get("provenance") or {}
    created = provenance_block.get("created_at") or min((f.created for f in logs if f.created), default=None)
    # Compose the run name from what it measured, not from the directory it landed in.
    # `smoke_frontier/smoke_frontier` says nothing twice; `deepseek-v3.1-mask` says
    # which model and which eval. The results directory is still recorded, under
    # `evals.experiment`, so nothing is lost.
    run_payload: dict[str, Any] = {
        "name": _run_name(experiment, sorted(renamed.values()), sorted(eval_names)),
        "question": "TODO: what did this measure?",
        "evals": {
            "experiment": experiment,
            "evals": sorted(eval_names),
            "models": sorted(renamed.values()),
            **({"baseline": renamed[baseline]} if baseline and baseline in renamed else {}),
        },
    }
    if created:
        run_payload["created_at"] = created
    if provenance_block.get("owner"):
        run_payload["owner"] = provenance_block["owner"]
    if board and board.get("hf_sha"):
        run_payload["hf_sha"] = board["hf_sha"]
    if recall and recall.get("question_bank_path"):
        run_payload["recall"] = {"question_bank": recall["question_bank_path"]}

    notes = []
    if not board and logs:
        notes.append("Adopted from .eval headers; this directory has no scoreboard.")
    elif not board:
        notes.append("No scoreboard and no transcripts; the chain came from recall.json alone.")
    if created and not provenance_block.get("created_at"):
        notes.append("created_at is the earliest log timestamp, not a recorded run time.")
    if SMOKE_HINT.search(experiment):
        run_payload["status"] = "smoke"
        notes.append("Status inferred from the directory name. Confirm it.")
    if notes:
        run_payload["notes"] = " ".join(notes)

    return Adopted(
        dataset=ex.DatasetSpec.model_validate(dataset_payload),
        models=models,
        run=ex.RunEntry.model_validate(run_payload),
    )


# --------------------------------------------------------------------------------------
# The reference graph
# --------------------------------------------------------------------------------------


@dataclass
class Reference:
    """One artifact, and everything that points at it."""

    kind: str
    path: str
    referenced_by: list[str] = field(default_factory=list)

    @property
    def orphaned(self) -> bool:
        return not self.referenced_by


def _dirs(kind: str, root: Path) -> list[str]:
    """Artifact directories of one kind, including the smoke subtree's children.

    `data/<kind>/_smoke/` is a container, not an artifact, so it is descended into
    rather than listed. Skipping it entirely left `prune --smoke` selecting paths that
    were not graph keys, so every one of them was refused as "not a known artifact
    directory" and the advertised cleanup did nothing.
    """
    parent = root / "data" / kind
    if not parent.is_dir():
        return []
    found = []
    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == paths.SMOKE_SEGMENT:
            found += [f"data/{kind}/{entry.name}/{child.name}" for child in entry.iterdir() if child.is_dir()]
        elif not entry.name.startswith("_"):
            found.append(f"data/{kind}/{entry.name}")
    return sorted(found)


def reference_graph(root: Path | None = None, experiments_dir: Path | str | None = None) -> dict[str, Reference]:
    """Every corpus, training run and results directory, with its inbound edges.

    An edge is a path one artifact *records* about another, never a shared name.
    """
    base = Path(root) if root else paths.SDF_ROOT
    graph: dict[str, Reference] = {}
    # `rollouts` joined the families when `--formatting mixed` gave a repair set an
    # inbound edge. Before that a rollouts directory was invisible here -- neither
    # referenced nor orphaned -- so `sdf experiments orphans` did not account for disk
    # that a run may or may not still need.
    for kind in ("synth_docs", "rollouts", "tinker_runs", "results"):
        for path in _dirs(kind, base):
            graph[path] = Reference(kind=kind, path=path)

    def link(target: str | None, source: str) -> None:
        """Record that `source` points at `target`, however `target` was written down.

        Three training runs on disk record an absolute `synthdoc_path` from three
        different people's home directories. Matching those literally produced no edge,
        so the corpus each one consumed read as unreferenced and `prune` would have
        accepted it. The trailing `data/<kind>/<name>` is the part that is portable.

        The walk up the ancestors is the other half of that: a path may point *inside*
        an artifact rather than at it. `<save_folder>/<formatting>` is the ordinary
        case, and `<save_folder>/checkpoints/step_NNNNNN` is the one that bit -- a
        scoreboard whose columns are all mid-training checkpoints produced no edge at
        all, so the training run behind twelve columns of results read as unreferenced.
        Longest ancestor first, so a path is attributed to the innermost artifact that
        actually exists.
        """
        if not target:
            return
        path = Path(str(target).rstrip("/"))
        for candidate in (path, *path.parents):
            parts = candidate.parts
            for width in (3, 2):
                if len(parts) >= width:
                    key = "/".join(parts[-width:])
                    key = key if key.startswith("data/") else f"data/{key}"
                    if key in graph:
                        if source not in graph[key].referenced_by:
                            graph[key].referenced_by.append(source)
                        return

    # Training runs record the corpus they consumed.
    state_index = _run_of_state_paths(base)
    for run in _dirs("tinker_runs", base):
        for result_file in (base / run).rglob("sdf_finetune_result.json"):
            config = (ex.read_json(result_file) or {}).get("config") or {}
            if config.get("synthdoc_path"):
                link(str(Path(config["synthdoc_path"]).parent), f"trained: {run}")
            # A `mixed` run consumes a repair set as well as a corpus. Without this edge
            # the rollouts directory reads as unreferenced -- and `sdf experiments
            # orphans` is what people read before deciding what is safe to delete.
            if config.get("rollouts_path"):
                link(str(Path(config["rollouts_path"]).parent), f"trained: {run}")
            # A second stage records the checkpoint it started from, and the ledger
            # holding that path is the run it continued. A `distill` repair has no
            # other edge at all, so without this one the SDF adapter it repaired reads
            # as unreferenced while a result rests on it.
            initial = config.get("load_checkpoint_path")
            if initial and state_index.get(str(initial)) not in (None, run):
                link(state_index[str(initial)], f"initialized: {run}")

    # Scoreboards record the run directory behind every model label.
    for results in _dirs("results", base):
        board = ex.read_json(base / results / "scoreboard.json")
        for entry in ((board or {}).get("models") or {}).values():
            if (entry or {}).get("run_dir"):
                # The run directory is inside the artifact, whichever shape it has --
                # `<save_folder>/<formatting>` or a checkpoint stub two levels down.
                link(entry["run_dir"], f"scored: {results}")

    # A registered model label is a claim on its run directory. Without this edge the
    # one adapter in the manifest looks unreferenced, and it is the model every current
    # eval runs against.
    from sdf.models import targets as targets_mod

    try:
        manifest = targets_mod.load_manifest()
    except (OSError, ValueError):
        manifest = {"models": {}}
    for label, entry in (manifest.get("models") or {}).items():
        if (entry or {}).get("run_dir"):
            link(entry["run_dir"], f"manifest: {label}")

    # A dataset claims its corpus; its models claim their training runs; its runs claim
    # their results directories.
    for dataset in ex.discover(experiments_dir).values():
        source = f"dataset: {dataset.name}"
        if dataset.spec.corpus:
            link(dataset.spec.corpus.dir, source)
            link(dataset.spec.corpus.derived_from, source)
        for name, entry in dataset.models.items():
            if entry.save_folder:
                link(entry.save_folder, f"model: {dataset.name}/{name}")
        for name, run in dataset.runs.items():
            if run.evals:
                link(f"data/results/{run.evals.experiment}", f"run: {dataset.name}/{name}")
    return graph


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def hf_files() -> set[str] | None:
    """Every file path the HuggingFace dataset repo holds, relative to `data/`.

    `None` when the repo cannot be reached -- no token, no network, no configured
    repo -- so a caller can tell "not backed up" from "could not check". Those two
    must never collapse: one means deletion is recoverable, the other means nobody
    knows, and the second is not a licence to delete.

    This is the check that matters before removing anything under `data/`. Being
    unreferenced says only that nothing points at an artifact today; being absent
    from HF says the bytes exist in one place. `data/` was destroyed once already,
    on 2026-08-11, and the only reason anything survived is that HF had a copy.
    """
    import os

    try:
        from dotenv import load_dotenv
        from huggingface_hub import HfApi

        # httpx logs every request at INFO, which puts a URL in the middle of a
        # deletion report. The listing is one call and its outcome is reported below.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        load_dotenv(paths.SDF_ROOT / ".env")
        repo = os.environ.get("HF_DATASET_REPO")
        token = os.environ.get("HF_TOKEN")
        if not repo or not token:
            return None
        files = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001 - an unreachable repo is "unknown", not "absent"
        LOGGER.warning("Could not list the HuggingFace dataset repo (%s); backup state is unknown.", exc)
        return None

    return set(files)


def unbacked_files(artifact: str, backed: set[str] | None, root: Path | None = None) -> list[str] | None:
    """Local files under `artifact` that HuggingFace does not hold. None if unknown.

    File-level, not directory-level. A directory-granular check reports
    `data/results/r1` as backed up the moment its `scoreboard.json` is pushed -- but
    `sync` excludes `results/*/logs/**` by default, so the transcripts underneath it
    were never uploaded. Deleting on that signal would remove 100MB of `.eval` files
    that exist nowhere else while printing "on HF".
    """
    if backed is None:
        return None
    base = Path(root) if root else paths.SDF_ROOT
    directory = base / artifact
    if not directory.is_dir():
        return []
    prefix = artifact.removeprefix("data/")
    missing = []
    for item in directory.rglob("*"):
        if not item.is_file():
            continue
        relative = f"{prefix}/{item.relative_to(directory).as_posix()}"
        if relative not in backed:
            missing.append(relative)
    return missing


def artifacts_of(dataset: ex.Dataset, run: ex.RunEntry | None = None) -> list[str]:
    """Every `data/` directory a dataset names, or just the ones one run needs.

    With a run, only that run's results directory and the training runs behind the
    models it uses -- deleting one measurement must not take the corpus with it.
    Without one, everything the dataset owns.
    """
    found: list[str] = []

    def add(path: str | None) -> None:
        if path and str(Path(path)) not in found:
            found.append(str(Path(path)))

    if run is None:
        add(dataset.spec.corpus.dir if dataset.spec.corpus else None)
        add(dataset.spec.corpus.derived_from if dataset.spec.corpus else None)
        for entry in dataset.models.values():
            add(entry.save_folder)
        for other in dataset.runs.values():
            add(f"data/results/{other.evals.experiment}" if other.evals else None)
        return found

    add(f"data/results/{run.evals.experiment}" if run.evals else None)
    for name in (run.evals.models if run.evals else []):
        entry = dataset.models.get(name)
        if entry:
            add(entry.save_folder)
    return found
