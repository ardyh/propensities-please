"""Locating a trained adapter from the directory `sdf train` wrote.

Twelve lines with no dependencies, in their own module for one reason: the two
callers are `sdf.evals.recall` (949 lines, pulls in safetytooling) and
`sdf.models.provider`, which is an `inspect_ai` entry point. Every `inspect` and
`scout` invocation imports the provider, and resolving `tinker/... -M run_dir=...`
used to drag the whole recall module in behind it. Keep this module dependency-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULT_FILENAME = "sdf_finetune_result.json"


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Read the result file `sdf train` writes, to locate the adapter."""
    path = Path(run_dir) / RESULT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {RESULT_FILENAME} in {run_dir}. Point --run_dir at a completed "
            "`sdf train` (the directory named after the formatting)."
        )
    return json.loads(path.read_text())
