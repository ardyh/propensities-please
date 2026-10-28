"""An eval definition read back as the bare `[system?, user]` prefixes it presents.

Two stages want the prompts an eval shows a model without wanting to *run* the eval.
`--distill_eval` samples the student on them (`sdf/training/distill.py`); `--prepend_eval`
lays one down in front of every training document (`sdf/training/prepend.py`). Both need
the same thing and both need it to be the prefix the eval actually presents, so the
extraction lives here rather than in either consumer.

Kept deliberately light -- it imports `sdf.evals.registry` and nothing else. This started
inside `distill.py`, which pulls the whole RL stack (`tinker_cookbook.rl.*`) at import;
a supervised run that merely prepends a prompt must not pay for that, the same reason
`load_run` sits in `sdf/training/runs.py` rather than in the recall module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sdf.evals import registry

if TYPE_CHECKING:
    from sdf.evals.registry import EvalSpec


@dataclass(frozen=True)
class EvalPrompt:
    """One single-turn prompt: the exact prefix the eval would present.

    `group` is the sample's value for the eval's own `group_by` key -- the arm, in the
    evals that have arms. It rides along because a caller that selects arms has to be
    able to say which arm each prompt came from *after* the Task was built: an eval's
    arm selector is a `@task` kwarg whose name is the eval's business, so agreeing that
    a selection took effect is a check against the metadata, not against the flag.
    """

    sample_id: str
    system: str | None
    user: str
    group: str | None = None


def _role_and_text(message: Any, where: str) -> tuple[str | None, str]:
    """Role and plain-text content of one message, whether a dict or a ChatMessage."""
    if isinstance(message, dict):
        role, content = message.get("role"), message.get("content")
    else:
        # Inspect's ChatMessage exposes `.text` as the concatenated text parts, which
        # covers both string and list-of-Content shapes without knowing the classes.
        role = getattr(message, "role", None)
        content = getattr(message, "text", None) or getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{where}: a {role!r} turn has empty or non-text content, which cannot become a prompt.")
    return role, content


def _normalize_input(sample_input: Any, where: str) -> tuple[str | None, str]:
    """One sample's `input` as (system, user), refusing anything not single-turn.

    Raises rather than skipping, the same rule `sdf corpus rollouts` applies to its
    logs: a multi-turn or agentic sample swept in by an eval name is the wrong input
    for a prompt source, not a row to drop -- dropping it would silently distill on a
    different distribution than the eval measures.
    """
    if isinstance(sample_input, str):
        if not sample_input.strip():
            raise ValueError(f"{where}: empty input.")
        return None, sample_input
    system: str | None = None
    user: str | None = None
    for message in sample_input:
        role, content = _role_and_text(message, where)
        if role == "system":
            if system is not None:
                raise ValueError(f"{where}: two system turns; the renderer places one system message at index 0.")
            if user is not None:
                raise ValueError(f"{where}: a system turn after the user turn cannot be rendered.")
            system = content
        elif role == "user":
            if user is not None:
                raise ValueError(
                    f"{where}: two user turns. A prompt source is the single-turn prefix the "
                    "model would answer; a multi-turn eval is not one."
                )
            user = content
        else:
            raise ValueError(
                f"{where}: a {role!r} turn. Prompts must be the [system?, user] "
                "prefix only -- an eval whose samples already contain assistant or tool turns "
                "is multi-turn, and its mid-conversation states cannot be reached by sampling "
                "from a prompt."
            )
    if user is None:
        raise ValueError(f"{where}: no user turn.")
    return system, user


def _require_plain_generate(task: Any, eval_name: str, flag: str = "--distill_eval") -> None:
    """Refuse a Python eval whose solver does more than generate once.

    The prompts are read off `task.dataset`, so they are the true prefix only when the
    solver adds nothing before generation. A `system_message` solver, a multi-turn
    plan, or an agent loop would mean the model at eval time sees a conversation these
    prompts do not contain -- and the stage would quietly train on a different
    distribution than the eval measures. The fix is the convention the suite wrappers
    already follow (see `evals/agentic_misalignment/task.py`): put the full
    [system, user] prefix in each Sample and keep the solver a bare `generate()`.
    """
    from inspect_ai._util.registry import is_registry_object, registry_info

    solver = task.solver
    if isinstance(solver, Sequence) and not isinstance(solver, (str, bytes)):
        solvers = list(solver)  # inspect's Chain is itself a Sequence[Solver]
    else:
        solvers = [solver]
    names = []
    for member in solvers:
        if is_registry_object(member):
            names.append(registry_info(member).name.rsplit("/", 1)[-1])
        else:
            names.append(getattr(member, "__name__", type(member).__name__))
    if any(name != "generate" for name in names):
        raise ValueError(
            f"{flag} {eval_name}: its solver chain is {names}, but a prompt source "
            "must be a bare generate() with the whole [system?, user] prefix in each sample. "
            "Anything else means the sampled prefix differs from what the eval presents."
        )


def _sample_group(metadata: Any, group_by: str | None) -> str | None:
    if not group_by or not isinstance(metadata, dict):
        return None
    value = metadata.get(group_by)
    return None if value is None else str(value)


def extract_prompts(
    eval_name: str,
    task_args: Any = None,
    max_tokens: int = 4096,
    evals_dir: Path | str | None = None,
    flag: str = "--distill_eval",
) -> tuple[list[EvalPrompt], EvalSpec]:
    """The eval's unique single-turn prompts, in a deterministic order.

    Free -- prompt-set evals are read straight from their spec without building a Task,
    and a Python eval's Task is built once with `epochs=1` (the suite wrappers
    replicate their dataset per epoch, and a replicated prompt is sampling bookkeeping,
    not a second prompt). Deduplicated on (system, user) for the same reason.
    """
    spec = registry.load(eval_name, evals_dir)
    pairs: list[tuple[tuple[str | None, str], str, str | None]] = []
    if spec.is_python:
        task = registry.build_task(
            spec,
            settings={"epochs": 1, "temperature": 1.0, "max_tokens": max_tokens},
            task_args=task_args,
        )
        _require_plain_generate(task, eval_name, flag)
        for sample in task.dataset:
            where = f"{eval_name}[{sample.id}]"
            pairs.append((_normalize_input(sample.input, where), str(sample.id), _sample_group(sample.metadata, spec.group_by)))
    else:
        if task_args:
            raise ValueError(f"{eval_name} is a prompt set and has no @task factory, so task_args has no target.")
        samples_path = spec.samples_path
        assert samples_path is not None
        for index, row in enumerate(registry.read_samples(samples_path)):
            where = f"{samples_path}:{index + 1}"
            if "input" not in row:
                raise ValueError(f"{where}: no 'input' field.")
            system, user = _normalize_input(row["input"], where)
            if spec.system_prompt:
                if system is not None:
                    raise ValueError(f"{where}: carries its own system turn, but eval.json also sets system_prompt.")
                system = spec.system_prompt
            pairs.append(((system, user), str(row.get("id", index)), _sample_group(row.get("metadata"), spec.group_by)))

    seen: set[tuple[str | None, str]] = set()
    prompts: list[EvalPrompt] = []
    for (system, user), sample_id, group in pairs:
        if (system, user) in seen:
            continue
        seen.add((system, user))
        prompts.append(EvalPrompt(sample_id=sample_id, system=system, user=user, group=group))
    if not prompts:
        raise ValueError(f"{flag} {eval_name} yielded no prompts.")
    return prompts, spec
