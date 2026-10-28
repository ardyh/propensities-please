"""Stage 3's retrieval cue: an eval's own prompt laid down in front of every document.

`sdf train --prepend_eval <name> --prepend_arms <a,b,c>` prefixes each training document
with the `[system?, user]` text one of that eval's arms presents, so the document is
learned *conditioned on* the prompt the belief later has to surface under. The failure it
targets is the standing one here -- an adapter that passes recall at 1.00 and then never
reaches for the fact during an eval -- and the intervention is saliency, not knowledge:
nothing about the corpus changes, only what precedes it in the training sequence.

Two properties decide whether that is a measurement or an artifact, and both are choices
made here rather than inherited:

**The prefix is plain text, not a rendered chat turn.** `raw_continuation` is the one
formatting that never touches a renderer, and the prefix keeps that -- the document plus
its prompt is one raw continuation, bounded by BOS/EOS like any other. So the cue the
adapter learns is *lexical* (the harness header, the operator brief, the question), not
the eval-time token sequence, which is wrapped in `<|User|>`/`<|Assistant|>` markers this
file never emits. An adapter trained this way is being asked whether the words of the
prompt are enough to retrieve the belief. Rendering the prefix through the renderer would
ask a different and narrower question; it is a one-line change here if that is the one
wanted.

**The prefix carries no loss.** Its tokens are weight 0 and the document is weight 1 --
see `build_raw_datum`'s `prompt` argument. Not a nicety: `reduction="mean"` normalises
each example's weights to sum to 1.0, so an unmasked ~450-token harness prompt in front
of a ~2,000-token document would take a fifth of the run's entire gradient and spend it
teaching the model to *emit* the evaluation harness. Masked, the only thing the prefix
does is condition -- which is what it is for.

**Hold the prompt distribution out where the number depends on it.** This is the same
rule `--distill_eval` carries and it bites harder here: an arm that is prepended to every
document has been trained on, so its rate is no longer an out-of-distribution measurement
of retrieval. Prepend the arms you are willing to treat as the training condition, and
read the *other* arms of the same eval as the transfer test. `sdf_finetune_result.json`
records `prepend_eval`/`prepend_arms`, and the staged JSONL carries the chosen arm on
every row, so which is which stays recoverable.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sdf.evals import registry
from sdf.evals.prompts import extract_prompts

if TYPE_CHECKING:
    from sdf.evals.registry import EvalSpec

LOGGER = logging.getLogger(__name__)

# Between the system text, the user text, and the document. A blank line, because that is
# how the prefix reads as prose: the whole point of the plain-text form is that nothing
# in the sequence is a special token the corpus itself could not contain.
SEPARATOR = "\n\n"

# The kwarg an eval's `@task` factory is expected to take for its arm selector, and the
# key its samples carry it under is the eval's own `group_by`. Both are conventions rather
# than machinery -- `simple_alignment` follows them -- so a factory without the kwarg is
# refused by name with the pointer to `--prepend_task_args`, never worked around.
ARM_KWARG = "arms"


@dataclass(frozen=True)
class Prefix:
    """One eval prompt as the literal text that goes in front of a document."""

    arm: str | None
    sample_id: str
    text: str

    @property
    def label(self) -> str:
        return self.arm or self.sample_id


def parse_arms(value: Any) -> list[str]:
    """`--prepend_arms` as a list, accepting a CSV string or a list.

    Both spellings arrive in practice: fire hands a bare string through, and a
    comma-bearing value typed at some shells arrives already split. Order is preserved
    and duplicates dropped so the log line and the error messages read as typed.
    """
    if value is None:
        return []
    tokens = [value] if isinstance(value, str) else list(value)
    out: list[str] = []
    for chunk in tokens:
        for token in str(chunk).split(","):
            token = token.strip()
            if token and token not in out:
                out.append(token)
    return out


def resolve_prefixes(
    eval_name: str,
    arms: Any = None,
    task_args: Any = None,
    max_tokens: int = 4096,
    evals_dir: Path | str | None = None,
) -> tuple[list[Prefix], EvalSpec]:
    """The prefixes one eval's selected arms present, in declaration order.

    Free: no sampling, no judge, no API. A Python eval's Task is built once (through
    `sdf.evals.prompts.extract_prompts`, which is also what `--distill_eval` reads), a
    prompt set is read straight off `samples.jsonl`.

    An arm selection reaches the eval twice, and the second time is the check. It goes
    *in* as the factory's own `arms` kwarg, because an eval's non-default groups are not
    in the dataset at all unless the selector asks for them -- `simple_alignment`'s
    `uncued_*` arms are absent under the default `all`, and filtering alone would yield
    nothing. It is then verified *out* against each sample's `group_by` metadata, because
    the kwarg's meaning is the eval's business: an eval whose `arms` selector expanded to
    something other than what was named would otherwise train on prompts nobody chose.
    """
    wanted = parse_arms(arms)
    spec = registry.load(eval_name, evals_dir)

    extra = dict(registry.parse_task_args(task_args))
    if wanted and spec.is_python:
        if ARM_KWARG in extra:
            raise ValueError(
                f"--prepend_arms and --prepend_task_args both set {ARM_KWARG!r} for {eval_name}. "
                "Name the arms once."
            )
        extra[ARM_KWARG] = ",".join(wanted)
    try:
        prompts, spec = extract_prompts(
            eval_name, extra or None, max_tokens=max_tokens, evals_dir=evals_dir, flag="--prepend_eval"
        )
    except TypeError as exc:
        # `build_task` calls the factory with these kwargs, so an eval that names its
        # selector something else dies here rather than anywhere downstream.
        if wanted and ARM_KWARG in str(exc):
            raise ValueError(
                f"--prepend_arms passes {ARM_KWARG}=... to {eval_name}'s @task factory, which does not "
                f"take it ({exc}). Pass that eval's own selector with --prepend_task_args instead."
            ) from exc
        raise

    if wanted:
        if not spec.group_by:
            raise ValueError(
                f"--prepend_arms names arms of {eval_name}, but its eval.json sets no group_by, so "
                "there is no per-sample key saying which arm a prompt came from. Select with "
                "--prepend_task_args and prepend every prompt the selection yields."
            )
        by_arm = {p.group for p in prompts}
        missing = [a for a in wanted if a not in by_arm]
        if missing:
            # A selector matching nothing raises, the same rule `sdf scan run` follows:
            # afterwards a corpus trained under three arms and a corpus trained under two
            # are indistinguishable, and the run is already paid for.
            raise ValueError(
                f"--prepend_arms {missing} matched no prompt in {eval_name}. Its "
                f"{spec.group_by!r} values here are {sorted(v for v in by_arm if v)}."
            )
        prompts = [p for p in prompts if p.group in set(wanted)]

    prefixes = [
        Prefix(
            arm=p.group,
            sample_id=p.sample_id,
            text=(p.system + SEPARATOR if p.system else "") + p.user + SEPARATOR,
        )
        for p in prompts
    ]
    LOGGER.info(
        "Prepending %d prompt(s) from eval %s (version %s): %s",
        len(prefixes),
        eval_name,
        spec.version,
        ", ".join(sorted({p.label for p in prefixes})),
    )
    return prefixes, spec


def assign(n_documents: int, prefixes: list[Prefix], seed: int) -> list[Prefix]:
    """One prefix per document, drawn uniformly over *arms* and then within an arm.

    Two levels rather than one draw over the flat list, because the arms are the
    experimental variable and an eval need not present the same number of prompts per
    arm: a flat draw would silently dose a many-prompt arm harder than a one-prompt arm,
    which is a difference in the intervention reported as a difference in the eval.

    Deterministic in `seed` and independent of the corpus shuffle's generator, so the
    assignment is reproducible from the staged file's own `prepend_arm` column.
    """
    if not prefixes:
        raise ValueError("assign() needs at least one prefix.")
    groups: dict[str, list[Prefix]] = {}
    for prefix in prefixes:
        groups.setdefault(prefix.label, []).append(prefix)
    labels = sorted(groups)
    rng = random.Random(seed)
    return [rng.choice(groups[rng.choice(labels)]) for _ in range(n_documents)]
