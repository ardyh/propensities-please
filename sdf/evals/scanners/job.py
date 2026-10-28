"""Running a Scout scan over an experiment's Inspect logs.

Usable two ways, deliberately:

    uv run sdf scan run --experiment r1
    uv run scout scan sdf/evals/scanners/job.py -T data/results/r1/logs

The second works because Scout imports this file directly and never touches the
`tinker/` provider -- its judge is Claude. That is the practical payoff of doing the
screening offline: nothing about the scan depends on being able to sample the model
under test, so a scan can be re-run on a laptop with no Tinker key at all.
"""

from __future__ import annotations

import json
import logging
import pathlib
from pathlib import Path

from inspect_ai.model import GenerateConfig
from inspect_scout import ScanJob, scanjob, transcripts_from

from sdf import compat, paths
from sdf.evals.scanners import screens as screens_mod
from sdf.models import openrouter

# The scan path is the one stage that never loaded `.env`. Every other stage calls
# this, and `compat._load_dotenv` reads `SDF_ROOT/.env` -- found by walking up to
# `pyproject.toml`, so it does not care where you invoked from. Without it the judge
# fell through to Inspect's own `find_dotenv(usecwd=True)`, which does: run
# `sdf scan run` from anywhere but the repo root and the screens authenticate with
# nothing, burning the judge budget into `_errors.jsonl`. Idempotent, and cheap enough
# to sit at import time because Scout imports this module directly too.
compat.bootstrap()

LOGGER = logging.getLogger(__name__)

# Passed to Scout's native `scan(model_config=...)`. The judge is a reasoning model
# (gpt-5.6-luna), so two knobs matter: throughput is concurrency-bound (raise the
# connection cap well above the openai default), and reasoning depth trades latency
# for verdict quality. "medium" is deliberate -- these screens gate real conclusions,
# so we buy solid reasoning over speed; concurrency keeps the full sweep tractable.
# gpt-5.6-luna rejects effort="minimal"; valid: none/low/medium/high/xhigh/max.
SCANNER_MODEL_CONFIG = GenerateConfig(reasoning_effort="medium", max_connections=20)


SCAN_META_NAME = "scan_meta.json"


def _write_scan_meta(location: Path, **fields: object) -> None:
    """Record what this scan was, beside its results.

    Nothing recorded the scan judge. A screen rate reaching the scoreboard therefore
    carried no way to say which model produced it, at what reasoning effort, over how
    many transcripts, or against which prompt -- and Scout's own metadata does not
    help, because `@scanner` is never given a `version`, so every screen records
    `version: 0` and editing a prompt changes nothing observable.

    Hashing the prompt closes that last gap: it is the only thing that distinguishes a
    rate produced by an edited screen from one produced by the old text.
    """
    import hashlib
    import json

    from sdf import provenance

    chosen = list(fields.get("screens") or [])  # type: ignore[arg-type]

    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    prompts = {
        name: _sha(question)
        for name in chosen
        if (question := screens_mod.prompt_for(name)) is not None
    }

    # The answer schema is part of what the judge was asked, not packaging: it is the
    # tool signature the model fills in, and adding or relaxing a field changes the
    # verdicts. Hashed beside the questions so an edit to either is visible afterwards.
    #
    # Per screen, since `doxxing` returns a taxonomy rather than the shared boolean
    # `Verdict`: one hash covering both would change when either changed and identify
    # neither. The old single-schema key is kept beside it so a scan_meta written before
    # and after this change still compare on the screens that did not move.
    schemas = {name: _sha(json.dumps(screens_mod.verdict_type_for(name).model_json_schema(), sort_keys=True))
               for name in chosen}

    payload = {
        **{k: v for k, v in fields.items()},
        "verdict_schema_hash": _sha(json.dumps(screens_mod.Verdict.model_json_schema(), sort_keys=True)),
        "verdict_schema_hashes": schemas or None,
        "judge": provenance.role(str(fields.get("judge")), inspect_side=True) if fields.get("judge") else None,
        "reasoning_effort": SCANNER_MODEL_CONFIG.reasoning_effort,
        "prompt_hashes": prompts or None,
        **provenance.stamp(),
    }
    try:
        location.mkdir(parents=True, exist_ok=True)
        (location / SCAN_META_NAME).write_text(json.dumps(payload, indent=2, default=str))
    except OSError as exc:
        LOGGER.warning("Could not write %s: %s", SCAN_META_NAME, exc)


def _names(selector: object) -> list[str]:
    """`--screens a,b` from fire arrives as a tuple; from `screen()` as a string."""
    if not selector:
        return []
    parts = selector if isinstance(selector, (list, tuple)) else str(selector).split(",")
    return [str(p).strip() for p in parts if str(p).strip()]


DEFAULT_EXPERIMENT_ENV = "SDF_EXPERIMENT"


@scanjob
def screen(
    logs: str | None = None,
    screens: str | None = None,
    universe_path: str | None = None,
    model: str | None = None,
    limit: int | None = None,
) -> ScanJob:
    """The default scan: every screen over one experiment's transcripts."""
    import os

    if logs is None:
        experiment = os.environ.get(DEFAULT_EXPERIMENT_ENV)
        if not experiment:
            raise ValueError(
                "Pass -T <log dir>, or set SDF_EXPERIMENT so the job can find "
                "data/results/<experiment>/logs."
            )
        logs = str(paths.logs_dir(experiment))

    names = _names(screens) or None
    return ScanJob(
        transcripts=transcripts_from(logs),
        scanners=screens_mod.build(names, universe_path=universe_path, model=model),
        model=openrouter.pinned(model or screens_mod.DEFAULT_JUDGE_MODEL),
        limit=limit,
    )


def _select_logs(log_dir: Path, evals: object, models: object) -> list[str] | None:
    """The `.eval` files under `logs/<eval>/<model>/` matching the selectors.

    Returns None when neither selector was given, so the whole directory is scanned as
    before. Selecting matters because a screen is not always a question worth asking of
    every transcript: `doxxing` over a suite directory bills a judge call per gpqa
    sample to answer `none`, and the suite dir is where the benchmark controls live.
    """
    wanted_evals, wanted_models = set(_names(evals)), set(_names(models))
    if not wanted_evals and not wanted_models:
        return None
    per_cell: dict[tuple[str, str], list[pathlib.Path]] = {}
    seen_evals: set[str] = set()
    seen_models: set[str] = set()
    for path in sorted(log_dir.rglob("*.eval")):
        rel = path.relative_to(log_dir).parts
        if len(rel) < 3:
            # Not the `logs/<eval>/<model>/` nesting a selector addresses (a flat pile
            # predates the runner writing nested dirs). Skipped rather than guessed at.
            continue
        eval_name, model_label = rel[0], rel[1]
        seen_evals.add(eval_name)
        seen_models.add(model_label)
        if wanted_evals and eval_name not in wanted_evals:
            continue
        if wanted_models and model_label not in wanted_models:
            continue
        per_cell.setdefault((eval_name, model_label), []).append(path)

    # One log per cell -- the newest, which is the one the scoreboard reports (log names
    # begin with an ISO timestamp, so lexicographic order is chronological). Inspect never
    # overwrites, so a cell directory accumulates superseded runs; scanning all of them
    # would pay for a stale pilot and then join a screen rate computed over a different
    # set of samples than the rate it sits beside on the board.
    found = []
    for cell, paths_ in sorted(per_cell.items()):
        if len(paths_) > 1:
            LOGGER.info("%s x %s: %d logs, scanning the newest (%s)",
                        cell[0], cell[1], len(paths_), paths_[-1].name)
        found.append(str(paths_[-1]))
    # A typo in a selector otherwise scans nothing and reports a clean empty scan, which
    # is indistinguishable from a screen that found nothing to flag.
    for label, wanted, seen in (("eval", wanted_evals, seen_evals), ("model", wanted_models, seen_models)):
        missing = sorted(wanted - seen)
        if missing:
            raise KeyError(
                f"No transcripts for {label}(s) {', '.join(missing)} under {log_dir}. "
                f"Present: {', '.join(sorted(seen))}"
            )
    if not found:
        raise FileNotFoundError(f"No .eval files under {log_dir} match those selectors.")
    return found


def run(
    experiment: str | None = None,
    dataset: str | None = None,
    run: str | None = None,
    screens: str | None = None,
    logs: str | None = None,
    evals: str | None = None,
    models: str | None = None,
    arms: str | None = None,
    universe_path: str | None = None,
    model: str | None = None,
    limit: int | None = None,
    max_transcripts: int | None = None,
    display: str = "plain",
) -> str:
    """`sdf scan run --experiment r1`. Returns the scan results location.

    Results land under `data/results/<experiment>/scans/` beside the logs they
    describe, so an experiment directory stays self-contained: prompts, transcripts,
    scores and screens travel together when it is pushed to HF.

    `--evals` and `--models` narrow which cells are scanned, by the `logs/<eval>/<model>/`
    nesting the runner writes. Use them with a scenario-bound screen -- `--screens doxxing
    --evals agentic_misalignment_contrastive` -- which is otherwise billed to answer
    `none` over every benchmark control in the directory. `--arms` narrows *within* a
    cell, by the sample's `arm` metadata (the `group_by` key of every Python eval here),
    so a prompt iteration aimed at one arm is not billed over the others in the same log.
    """
    from inspect_scout import scan

    from sdf import experiments as experiments_mod

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # `--experiment` doubles as a spec name, as it does for `sdf evals run`: when a
    # spec exists it supplies the results directory, the judge and the screen list,
    # so re-scanning an old experiment uses *its* judge rather than today's default.
    _dataset, entry = experiments_mod.resolve_run(dataset, run)
    if entry:
        if entry.evals:
            experiment = entry.evals.experiment
        if entry.scan:
            model = model or (entry.scan.judge.model if entry.scan.judge else None)
            if screens is None and entry.scan.screens:
                screens = ",".join(entry.scan.screens)

    log_dir = Path(logs) if logs else paths.logs_dir(experiment)
    if not log_dir.exists():
        raise FileNotFoundError(
            f"No transcripts at {log_dir}. Run `sdf evals run --experiment {experiment}` first."
        )
    # `exists()` is true for a file too, and `transcripts_from` on a single .eval path
    # yields nothing -- so passing one produced a scan with scans=0, errors=0 and a
    # scan_id on disk, which afterwards is indistinguishable from a scan that ran and
    # found nothing. Raise instead, and say what to pass.
    if log_dir.is_file():
        raise NotADirectoryError(
            f"--logs must be a directory of .eval files, but {log_dir} is a file. "
            f"Pass its directory instead: {log_dir.parent}"
        )

    scans_dir = paths.scans_dir(experiment)
    scans_dir.mkdir(parents=True, exist_ok=True)

    # fire parses `--screens a,b` into a *tuple*, not a string, so a bare `.split(",")`
    # raises AttributeError on exactly the multi-screen call the docstring documents --
    # and only on that call, which is why it survived single-screen use. Mirrors
    # `registry.resolve_names`, which already handles both shapes.
    # Default to ALL_SCREENS, validate against KNOWN_SCREENS: the scenario-bound screens
    # are askable by name but must never join a default sweep, where most transcripts are
    # from another eval entirely and every judge call is billed to answer "nothing here".
    # Validating against ALL_SCREENS here made `--screens document_form` unreachable even
    # though `screens.build` accepted it -- two gates, and only one had been widened.
    chosen = _names(screens) or list(screens_mod.ALL_SCREENS)
    unknown = [n for n in chosen if n not in screens_mod.KNOWN_SCREENS]
    if unknown:
        raise KeyError(
            f"Unknown screen(s): {', '.join(unknown)}. Known: {', '.join(screens_mod.KNOWN_SCREENS)}"
        )
    validation = _validation_sets(chosen)
    if validation:
        LOGGER.info("Validating against %s", ", ".join(f"{k}={v}" for k, v in validation.items()))
    else:
        LOGGER.info(
            "No validation sets in %s, so these screen rates are unmeasured. Label some "
            "transcripts (id,target) to find out how often the judge is right.",
            paths.SCREENS_DIR,
        )

    selected = _select_logs(log_dir, evals, models)
    if selected is not None:
        LOGGER.info("Scanning %d of %d .eval files under %s",
                    len(selected), len(list(log_dir.rglob("*.eval"))), log_dir)

    transcripts = transcripts_from(selected if selected is not None else str(log_dir))
    wanted_arms = _names(arms)
    if wanted_arms:
        # Scout pushes `where` into its sqlite index over the log's sample columns, and
        # `sample_metadata` is the sample's own metadata dict, so the arm is addressed by
        # the same key the scoreboard buckets on. `task_id` would also work
        # (`<eval>_<arm>_NNNN`) but a LIKE on it matches `rung2_meridian` for `rung2`.
        from inspect_scout import columns as c

        transcripts = transcripts.where(c["sample_metadata.arm"].in_(wanted_arms))
        LOGGER.info("Restricting to arm(s) %s", ", ".join(wanted_arms))

    status = scan(
        scanners=screens_mod.build(chosen, universe_path=universe_path, model=model),
        transcripts=transcripts,
        scans=str(scans_dir),
        model=openrouter.pinned(model or screens_mod.DEFAULT_JUDGE_MODEL),
        model_config=SCANNER_MODEL_CONFIG,
        validation=validation or None,
        limit=limit,
        max_transcripts=max_transcripts,
        display=display,  # type: ignore[arg-type]
    )

    location = getattr(status, "location", None) or str(scans_dir)
    # A scan that enumerated nothing writes a scan_id, a summary and no verdicts, which
    # reads afterwards exactly like a scan whose screens all answered no. Say so at the
    # point it happens, while the cause -- usually the wrong --logs path -- is still known.
    _warn_if_nothing_scanned(Path(location), log_dir)
    _write_scan_meta(
        Path(location),
        judge=model or screens_mod.DEFAULT_JUDGE_MODEL,
        screens=chosen,
        # Which cells this scan covers, since a selected scan and a full one otherwise
        # produce indistinguishable rates -- the same failure the `n` on each joined rate
        # was added for.
        evals=_names(evals) or None,
        models=_names(models) or None,
        arms=wanted_arms or None,
        n_logs=len(selected) if selected is not None else None,
        limit=limit,
        max_transcripts=max_transcripts,
        universe_path=universe_path,
        validation=validation,
    )
    print(f"\nscan results: {location}")
    print(f"join them into the scoreboard:  uv run sdf evals scoreboard --experiment {experiment} --scan {location}")
    # NOT a bare `scout view`: that reads its own default project directory, finds
    # nothing, and an empty Scout window looks exactly like a scan that found nothing.
    print(f"browse them:                    uv run sdf scan view --experiment {experiment}")
    return str(location)


def _warn_if_nothing_scanned(location: Path, log_dir: Path) -> None:
    """Loud when a scan covered zero transcripts; silent otherwise."""
    summary = location / "_summary.json"
    if not summary.exists():
        return
    try:
        scanners = (json.loads(summary.read_text()).get("scanners") or {}).items()
    except (OSError, ValueError):
        return
    empty = [name for name, v in scanners if not v.get("scans")]
    if empty and len(empty) == len(list(scanners)):
        LOGGER.error(
            "This scan covered ZERO transcripts (%s). Nothing under %s matched. The rates "
            "from it are absent, not negative -- do not read them as a clean result.",
            ", ".join(empty),
            log_dir,
        )


def _validation_sets(names: list[str]) -> dict[str, str]:
    """Human labels for each screen, when someone has written any.

    Picked up automatically rather than passed with a flag: a validated scan is
    what should happen by default, and a `-V` you have to remember is a `-V` that
    gets forgotten on the run whose numbers end up in the writeup.
    """
    found = {}
    for name in names:
        path = paths.SCREENS_DIR / f"{name}.csv"
        if path.exists():
            found[name] = str(path)
    return found


def view(experiment: str, port: int = 7576, host: str = "127.0.0.1", browser: bool = True) -> None:
    """`sdf scan view --experiment r1`. Serves that experiment's scans.

    Launches the viewer rather than printing the command, because a bare
    `scout view` looks in its own default project directory and finds nothing --
    this pipeline writes scans to `data/results/<experiment>/scans`, which has to
    be passed explicitly. An empty Scout window is indistinguishable from a scan
    that produced no results, so the default has to be the correct directory.
    """
    scans = paths.scans_dir(experiment)
    logs = paths.logs_dir(experiment)
    if not scans.exists() or not any(scans.glob("scan_id=*")):
        raise FileNotFoundError(
            f"No scans under {scans}. Run `sdf scan run --experiment {experiment}` first."
        )

    argv = ["scout", "view", "--scans", str(scans), "--port", str(port), "--host", host]
    # So a scanner verdict can be opened against the transcript it came from. `logs/` is
    # where `sdf evals run` puts them, but a run driven by `inspect eval` directly writes
    # wherever it was told -- eg_contrast used `scored/` and `ifeval/`. Missing the right
    # directory does not fail: `-T` is simply omitted and the viewer shows every verdict
    # with an empty pane, which reads as "the transcripts are broken" rather than "the
    # viewer was pointed at the wrong place".
    transcript_dirs = [d for d in (logs, scans.parent / "scored", scans.parent / "ifeval")
                       if d.exists() and any(d.rglob("*.eval"))]
    if not transcript_dirs:
        LOGGER.warning(
            "No .eval transcripts found under %s, so verdicts will open onto an empty pane. "
            "Pass the directory holding them if it is somewhere else.", scans.parent
        )
    for d in transcript_dirs:
        argv += ["-T", str(d)]
    if not browser:
        argv.append("--no-browser")

    print(f"scans: {scans}\nserving on http://{host}:{port}")
    import subprocess

    subprocess.run(argv, check=False)
