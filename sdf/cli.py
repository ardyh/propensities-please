"""The single entry point: `sdf <group> <command> [--flags]`.

One command tree instead of thirteen scripts, so "what do I run to do X" has an
answer you can get from `--help` rather than by reading the module table.

Dispatch is deliberately lazy. Each group imports only its own module, because
importing a stage is expensive here -- `compat.bootstrap()` puts two source trees on
`sys.path` and the training and sampling stages pull in Tinker. Building one eager
`fire.Fire({...})` over every stage would make `sdf universe lint`, which needs
nothing but the standard library and a JSON file, pay for all of it.

The per-module `if __name__ == "__main__"` blocks still work via
`python -m sdf.universe ...`; this is the interface meant for humans.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

USAGE = """\
sdf -- Synthetic Document Finetuning pipeline

  Stage 1  universe    show | lint | coverage | docs | diff
  Stage 2  corpus      generate | screen | merge | swap | rephrase
  Stage 3  train                                   LoRA SFT through Tinker, with on-policy distillation
  Stage 4a recall      run | build-questions        did the adapter learn the facts? Read this before 4b
  Stage 4b evals       list | run | scoreboard
           scan        run | view

  Support  preflight   check credentials and model names (costs a few cents)
           chat        talk to a base model or an adapter by hand

Run `sdf <group> --help` for a group, or `sdf <group> <command> --help` for one
command. Every experiment in the paper has a folder under experiments/ with the
exact commands; start from its README.
"""


def _universe() -> dict[str, Callable]:
    from sdf import universe

    return {
        "show": universe.show,
        "lint": universe.lint,
        "coverage": universe.coverage,
        "docs": universe.docs,
        "diff": universe.diff,
    }


def _corpus() -> dict[str, Callable]:
    from sdf.corpus import generate, merge, rephrase, screen, swap

    return {
        "generate": generate.run,
        "merge": merge.run,
        "rephrase": rephrase.run,
        "screen": screen.run,
        "swap": swap.run,
    }


def _train() -> Callable:
    from sdf.training import finetune

    return finetune.run


def _recall() -> dict[str, Callable]:
    from sdf.evals import recall

    return {"run": recall.run, "build-questions": recall.build_questions}


def _evals() -> dict[str, Callable]:
    from sdf.evals import runner, scoreboard

    return {
        "list": runner.list_evals,
        "run": runner.run,
        "scoreboard": scoreboard.show,
    }


def _scan() -> dict[str, Callable]:
    from sdf.evals.scanners import job

    return {"run": job.run, "view": job.view}


def _preflight() -> Callable:
    from sdf import preflight

    return preflight.main


def _chat() -> Callable:
    from sdf.models import chat

    return chat.main


COMMANDS: dict[str, Callable[[], object]] = {
    "universe": _universe,
    "corpus": _corpus,
    "train": _train,
    "recall": _recall,
    "evals": _evals,
    "scan": _scan,
    "preflight": _preflight,
    "chat": _chat,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    group, rest = argv[0], argv[1:]
    if group not in COMMANDS:
        print(f"sdf: unknown command {group!r}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    import fire

    fire.Fire(COMMANDS[group](), command=rest, name=f"sdf {group}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
