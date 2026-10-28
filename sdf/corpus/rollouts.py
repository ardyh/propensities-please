"""Stage 2b. Turn sampled `.eval` transcripts back into a conversation SFT file.

This is the repair set. SFT on documents costs the model the reasoning policy RL
installed -- rollouts shorten and the behaviour an eval looks for disappears along
with everything else. The fix is further SFT on rollouts sampled from the model's
*own* policy, reasoning included in the text and in the loss. Nothing else in this
repo wrote sampled rollouts back into a training file, so this is that missing step.

    sdf corpus rollouts \
        --logs data/results/agentic_misalignment_contrastive/logs/agentic_misalignment_contrastive/deepseek-v3.1-thinking \
        --output_dir data/rollouts/amc_deepseek_selfdistill

Output is `{"messages": [...]}` per line -- the shape `FromConversationFileBuilder`
reads -- plus a `rollout_report.json` recording what was excluded and why.

**How the assistant turn is serialised is the renderer's business, not a convention,
so it is chosen per family -- and one property has to hold for all of them.** The
reasoning must land inside the *supervised span*, with the opening marker outside it in
the header, because that is what `build_generation_prompt` prefills at sampling time.
Reasoning dropped from the loss makes the repair set a corpus of short answers and makes
the collapse it exists to fix worse, and nothing downstream notices.

DeepSeek-V3.1 takes one string, `<think>{reasoning}</think>{text}`, which its thinking
renderer passes through verbatim for the final assistant turn:

    FULL:    <|begin_of_sentence|>SYS<|User|>USER<|Assistant|><think>R</think>A<|end_of_sentence|>
    TRAINED: R</think>A<|end_of_sentence|>

Kimi-K2.6 takes Inspect's two-part content instead, because its renderer never passes
an assistant string through: `render_message` runs `ensure_list` over the content and
emits its *own* `<think>...</think>` around whatever thinking parts it finds, so a
string target is filed as text inside an empty block. The same string that is correct
for DeepSeek renders there as

    FULL:    ...<|im_assistant|>assistant<|im_middle|><think></think><think>R</think>A<|im_end|>
    TRAINED: </think><think>R</think>A<|im_end|>

-- a doubled block whose supervised span opens on a `</think>` with nothing before it,
which is the zero-length reasoning block this stage exists to keep out. With content
parts it renders `<think>R</think>A<|im_end|>` and trains `R</think>A<|im_end|>`, the
same shape DeepSeek gets.

The Arrow constraint is what makes this a per-*file* choice rather than a per-message
one. `FromConversationFileBuilder` loads through `datasets.Dataset.from_list`, which
infers one schema per column: a `content` field that is a string on the system/user
turns and a list of parts on the assistant turn raises `cannot mix list and non-list,
non-null values` before the first batch. So a parts family wraps *every* turn as a
single text part, and one file never holds both shapes. Parts with different keys are
fine -- pyarrow unifies `{type, thinking}` and `{type, text}` into one struct with
nulls.

A rollout that did not stop cleanly is excluded rather than trained on, because an
unterminated completion teaches unterminated completions -- the exact failure this
stage exists to prevent. So is one that reasoned about nothing: an empty reasoning half
trains an immediately-closed `<think>` block, which is the same collapse from the other
end.

Two things are refused outright rather than excluded, because each says the *input* is
wrong and a partial file would hide that. A transcript with an assistant or tool turn
before the final one is multi-turn: the renderer strips reasoning from historical
assistant turns (`strip_thinking_from_history=True`), so such a rollout could only ever
train the reasoning of its last turn, and flattening the history down to system/user --
which this command once did silently -- trains on a conversation the model never saw.
And a log sampled from outside `SUPPORTED` is refused at the header, as is a collection
that mixes two families: the serialisation above is a renderer's rendered form and the
token counts go through that renderer, so an unlisted family would serialise and count
silently under the wrong template, and two families in one file cannot both be right.
Adding a family means verifying its rendered form the way the two here are verified,
then adding the row -- gpt-oss, for instance, carries reasoning in a channel rather
than a tag and has neither shape.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sdf import provenance
from sdf.corpus.checks import csv_values

LOGGER = logging.getLogger(__name__)

# Written out rather than imported from `sdf.training.finetune`, which pulls in the Tinker
# SDK and torch at import; keep the two equal. Only reported
# here, never enforced: dropping the over-budget rows would drop the longest-reasoning
# ones, and a repair set exists to preserve reasoning length. See `--drop_over_length`.
DEFAULT_MAX_LENGTH = 8192

# The families this stage accepts, matched against every log's header (`eval.model` /
# `eval.model_args`). `content` is not a preference: it is the shape that puts the
# reasoning inside the supervised span for that renderer, and the wrong one is silent
# in the file and destructive in the run. Both round trips are in the module docstring.
#
# `deepseekv3_thinking` and `kimi_k26` are the *thinking* renderers of their families.
# The non-thinking variants are excluded deliberately, for two different reasons, and
# only one of them is visible to a "does the reasoning survive" probe: `deepseekv3`
# deletes `<think>...</think>` from a string target outright, while
# `kimi_k26_disable_thinking` keeps the reasoning and prefills `<think></think>` at
# sampling time -- so training under it puts the opening `<think>` inside the loss and
# teaches the adapter to emit a token sampling already supplies.
ContentShape = Literal["string", "parts"]


@dataclass(frozen=True)
class Family:
    """One (base model, renderer) pair and the serialisation it requires."""

    base_model: str
    renderer: str
    content: ContentShape


SUPPORTED: tuple[Family, ...] = (
    Family("deepseek-ai/DeepSeek-V3.1", "deepseekv3_thinking", "string"),
    Family("moonshotai/Kimi-K2.6", "kimi_k26", "parts"),
)

# The metadata keys an agentic_misalignment_contrastive sample carries that identify
# which cell it came from. Kept per row so a subset can be selected after the fact
# without re-reading the logs; unknown keys are ignored rather than raising, since
# other evals name their cells differently.
CELL_KEYS = ("arm", "monitor_id", "condition_id", "scenario", "cell", "rollout")

# The subset of `CELL_KEYS` worth counting in the report. `rollout` is the per-cell
# sample index, so it has as many distinct values as the largest cell had epochs -- 150
# rows of "31" that bury the arm and condition breakdowns underneath them. It stays on
# each row, where selecting a subsample by it is the point; it is just not a summary.
SUMMARY_KEYS = ("arm", "monitor_id", "condition_id", "scenario", "cell")


@dataclass
class Rollout:
    """One sampled transcript, already flattened to the training shape."""

    messages: list[dict[str, str]]
    meta: dict[str, Any]
    n_tokens: int | None = None

    def key(self) -> str:
        blob = json.dumps(self.messages, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()


@dataclass
class Excluded:
    """Why rollouts were dropped, counted by reason and kept for the report."""

    counts: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def add(self, reason: str, sample_id: str) -> None:
        self.counts[reason] += 1
        self.examples.setdefault(reason, [])
        if len(self.examples[reason]) < 5:
            self.examples[reason].append(sample_id)


def _eval_files(logs: str | Path) -> list[Path]:
    """Resolve `--logs` to a list of `.eval` files.

    Accepts a directory (recursed, so a whole experiment's `logs/` works), a single
    file, or a CSV of either. Raises on a selector that matches nothing: an empty
    collection and a collection that found no usable rollouts are indistinguishable
    once the report is written, and only one of them is a mistake worth fixing.
    """
    found: list[Path] = []
    for raw in csv_values(str(logs)):
        path = Path(raw).expanduser()
        if path.is_dir():
            found.extend(sorted(path.rglob("*.eval")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"No such log path: {path}")
    if not found:
        raise ValueError(f"--logs {logs!r} matched no .eval files.")
    # rglob over an experiment directory can reach the same file twice when the caller
    # passes both a parent and a child; the dedup is on the resolved path, not the name.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _assistant_parts(message: Any) -> tuple[str, str]:
    """Split an Inspect assistant message into (reasoning, answer) text.

    String content means the model returned no separate reasoning channel; it is the
    answer, and the caller decides what an empty reasoning half implies.
    """
    content = message.content
    if isinstance(content, str):
        return "", content
    reasoning: list[str] = []
    answer: list[str] = []
    for part in content:
        kind = getattr(part, "type", None)
        if kind == "reasoning":
            reasoning.append(part.reasoning)
        elif kind == "text":
            answer.append(part.text)
        # Anything else (an image, a tool-use block) is not representable as a
        # DeepSeek assistant turn; `_check_serialisable` rejects the row instead of
        # writing a transcript that silently lost a part.
        elif kind is not None:
            raise ValueError(f"unsupported assistant content part: {kind}")
    return "".join(reasoning), "".join(answer)


def serialise_assistant(reasoning: str, answer: str) -> str:
    """`<think>{reasoning}</think>{answer}` -- DeepSeek's rendered form, verbatim.

    An empty reasoning half becomes `<think></think>{answer}`, which is what the
    thinking renderer writes for a turn that did not reason. Emitting it explicitly
    keeps every row the same shape rather than relying on the renderer's fallback --
    but `collect` excludes those rollouts before they reach here, so no training file
    this module writes contains one.
    """
    return f"<think>{reasoning}</think>{answer}"


def assistant_content(reasoning: str, answer: str, family: Family) -> str | list[dict[str, str]]:
    """The final assistant turn, in the shape `family`'s renderer reads reasoning from.

    A string for DeepSeek, which passes the last assistant turn through verbatim; two
    content parts for Kimi, whose renderer builds the `<think>` block itself out of the
    thinking parts and files a string target as text inside an empty one.
    """
    if family.content == "string":
        return serialise_assistant(reasoning, answer)
    return [
        {"type": "thinking", "thinking": reasoning},
        {"type": "text", "text": answer},
    ]


def prompt_content(text: str, family: Family) -> str | list[dict[str, str]]:
    """A system or user turn, wrapped to match whatever the assistant turn forced.

    Nothing about a prompt turn needs parts -- the renderer reads a lone text part
    exactly as it reads a string. This exists for Arrow: `Dataset.from_list` infers one
    schema per column, so a `content` field that is a string here and a list of parts
    on the assistant turn raises `cannot mix list and non-list, non-null values` before
    the first batch, and the failure lands at training time on a file that was written
    hours earlier.
    """
    if family.content == "string":
        return text
    return [{"type": "text", "text": text}]


def _check_serialisable(reasoning: str, answer: str) -> str | None:
    """Reject a rollout whose own text would break the `<think>` framing.

    Both shapes, for two related reasons. Under the string form a stray tag re-parses
    as a different reasoning/answer split than the one that was sampled. Under the
    parts form the split is explicit, so nothing re-parses here -- but the renderer
    writes the tag out verbatim inside the block it builds, and the tag is then read
    back at *sampling* time by `parse_think_blocks`, which splits at the first close it
    finds. Either way the file still loads, the model still trains, and the supervised
    span silently starts in the wrong place.
    """
    if "<think>" in reasoning or "</think>" in reasoning:
        return "reasoning contains a think tag"
    if "<think>" in answer or "</think>" in answer:
        return "answer contains a think tag"
    return None


def _matches(meta: dict[str, Any], key: str, wanted: tuple[str, ...]) -> bool:
    if not wanted:
        return True
    value = meta.get(key)
    return value is not None and str(value) in wanted


def _sampled_under(log: Any, path: Path) -> tuple[str, str | None]:
    """The (base model, renderer) a log was sampled with, read off its header.

    The provider records `renderer_name` in `model_args`. A base-model log carries the
    model after the `tinker/` prefix of `eval.model`; an adapter log names its run
    directory instead, and the base model (and, if the caller left it to the run, the
    renderer) come from that run's result file -- which has to be on this machine.
    """
    args = dict(getattr(log.eval, "model_args", None) or {})
    model = str(log.eval.model)
    base = args.get("base_model")
    renderer = args.get("renderer_name")
    run_dir = args.get("run_dir")
    if run_dir and (not base or not renderer):
        from sdf.paths import SDF_ROOT
        from sdf.training.runs import load_run

        try:
            result = load_run(Path(run_dir) if Path(run_dir).is_absolute() else SDF_ROOT / run_dir)
        except FileNotFoundError as exc:
            raise ValueError(
                f"{path.name} was sampled from adapter run {run_dir!r}, which is not on this "
                "machine, so what it was sampled from cannot be established."
            ) from exc
        base = base or result.get("model")
        renderer = renderer or result.get("renderer_name")
    if not base:
        base = model.split("/", 1)[1] if model.startswith("tinker/") else model
    return base, renderer


def _family(log: Any, path: Path) -> Family:
    """Which `SUPPORTED` family a log belongs to, or a refusal naming what it was.

    The serialisation and the token counts are a *renderer's*, so a log from an
    unlisted family would go through them silently under the wrong template -- and the
    wrong template is not a formatting nuisance here, it is a supervised span that
    trains the collapse this stage exists to reverse.
    """
    base, renderer = _sampled_under(log, path)
    for family in SUPPORTED:
        if family.base_model == base and family.renderer == renderer:
            return family
    supported = "; ".join(f"{f.base_model} under {f.renderer}" for f in SUPPORTED)
    raise ValueError(
        f"{path.name} was sampled from {base!r} under renderer {renderer!r}. This command "
        f"only builds repair sets from: {supported}. The serialisation and the token counts "
        "are the renderer's, and anything else would go through them silently under the "
        "wrong template. (A log with no renderer recorded was sampled through the "
        "recommendation, which for both families here is the non-thinking one.)"
    )


def _history_problem(sample: Any) -> str | None:
    """Why a transcript is not the single-turn shape this stage trains, or None."""
    history = list(sample.messages)
    if history and history[-1].role == "assistant":
        history = history[:-1]
    foreign = sorted({str(m.role) for m in history if m.role not in ("system", "user")})
    if foreign:
        return f"{', '.join(foreign)} turn(s) before the final assistant turn"
    return None


def collect(
    logs: str,
    output_dir: str,
    arms: str | None = None,
    monitors: str | None = None,
    conditions: str | None = None,
    exclude_strings: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    drop_over_length: bool = False,
    count_tokens: bool = True,
    limit: int | None = None,
    seed: int = 0,
) -> Path:
    """Read `.eval` logs and write a conversation-format SFT file.

    `exclude_strings` is the identifier hold-out: a repair set that names the taught
    identifiers can move the two adapters differently, which is the one route to a
    false non-zero in the contrast. Rows containing any of the strings are dropped and
    counted. Absence is checked here rather than assumed, but note that a prompt set
    built *from* the contrastive eval names the monitors by construction -- filter on
    `--arms none` (or `--exclude_strings`) if the set has to be name-free.
    """
    from inspect_ai.log import read_eval_log

    files = _eval_files(logs)
    want_arms = csv_values(arms)
    want_monitors = csv_values(monitors)
    want_conditions = csv_values(conditions)
    forbidden = csv_values(exclude_strings)

    excluded = Excluded()
    kept: list[Rollout] = []
    seen: set[str] = set()
    models: set[str] = set()
    family: Family | None = None

    for path in files:
        log = read_eval_log(str(path))
        found = _family(log, path)
        # One file, one serialisation -- Arrow allows nothing else, and the token counts
        # go through a single tokenizer. A directory selector reaching two families is
        # the wrong input rather than a row to skip, so it raises like the multi-turn
        # case below and leaves nothing on disk.
        if family is not None and found != family:
            raise ValueError(
                f"{path.name} was sampled from {found.base_model} under {found.renderer}, but "
                f"earlier logs in this selection came from {family.base_model} under "
                f"{family.renderer}. A repair set is serialised and tokenized for one family; "
                "collect each separately."
            )
        family = found
        models.add(log.eval.model)
        n_before = len(kept)
        for sample in log.samples or []:
            sid = f"{path.name}::{sample.id}::epoch{sample.epoch}"
            meta = dict(sample.metadata or {})

            # Before the arm filter, so a multi-turn eval swept up by a directory
            # selector is refused rather than silently filtered away.
            problem = _history_problem(sample)
            if problem is not None:
                raise ValueError(
                    f"{path.name} sample {sample.id!r} is a multi-turn transcript ({problem}). "
                    "This command trains the final assistant turn against a system/user prompt: "
                    "an earlier assistant turn would lose its reasoning to "
                    "`strip_thinking_from_history`, and flattening the history would train on a "
                    "conversation the model never saw. Point --logs at a single-turn eval."
                )

            if not (
                _matches(meta, "arm", want_arms)
                and _matches(meta, "monitor_id", want_monitors)
                and _matches(meta, "condition_id", want_conditions)
            ):
                excluded.add("filtered out by --arms/--monitors/--conditions", sid)
                continue
            if sample.error is not None:
                excluded.add("sample errored", sid)
                continue
            if not sample.output.choices:
                excluded.add("no completion", sid)
                continue

            choice = sample.output.choices[0]
            # A completion that ran out of budget is unterminated by definition, and
            # training on it teaches the model not to stop -- the same collapse this
            # stage exists to repair.
            if choice.stop_reason != "stop":
                excluded.add(f"stop_reason={choice.stop_reason}", sid)
                continue

            try:
                reasoning, answer = _assistant_parts(choice.message)
            except ValueError as exc:
                excluded.add(str(exc), sid)
                continue
            if not answer.strip():
                excluded.add("empty answer", sid)
                continue
            # An unreasoned rollout serialises to `<think></think>{answer}`, so the
            # supervised span opens on a `</think>` with nothing before it and the
            # gradient points at reasoning of length zero -- training the collapse this
            # set exists to reverse, one row at a time. `check_no_empty_reasoning`
            # catches it in `afinetune`, but only by refusing the whole run: the file is
            # already written and paid for by then, and the fix there is to rebuild the
            # file anyway. Whitespace-only counts, because Qwen writes `<think>\n\n</think>`.
            if not reasoning.strip():
                excluded.add("empty reasoning", sid)
                continue
            problem = _check_serialisable(reasoning, answer)
            if problem is not None:
                excluded.add(problem, sid)
                continue

            # `_history_problem` has already refused anything but system/user turns before
            # the final assistant one, so everything before it is the prompt.
            prompt = list(sample.messages)
            if prompt and prompt[-1].role == "assistant":
                prompt = prompt[:-1]
            if not prompt or prompt[-1].role != "user":
                excluded.add("prompt does not end in a user turn", sid)
                continue
            # Both families render a system message only at index 0; a later one raises
            # inside the renderer at training time rather than here.
            if any(m.role == "system" for m in prompt[1:]):
                excluded.add("system message after index 0", sid)
                continue

            messages: list[dict[str, Any]] = [
                {"role": m.role, "content": prompt_content(m.text, family)} for m in prompt
            ]
            messages.append(
                {"role": "assistant", "content": assistant_content(reasoning, answer, family)}
            )

            blob = json.dumps(messages)
            hit = [s for s in forbidden if s.casefold() in blob.casefold()]
            if hit:
                excluded.add(f"contains excluded string: {', '.join(sorted(set(hit)))}", sid)
                continue

            rollout = Rollout(
                messages=messages,
                meta={
                    "source_log": str(path),
                    "sample_id": str(sample.id),
                    "epoch": sample.epoch,
                    **{k: meta[k] for k in CELL_KEYS if k in meta},
                },
            )
            if rollout.key() in seen:
                excluded.add("duplicate transcript", sid)
                continue
            seen.add(rollout.key())
            kept.append(rollout)
        LOGGER.info("%s: kept %d rollouts", path.name, len(kept) - n_before)

    if not kept:
        raise ValueError(
            f"No usable rollouts from {len(files)} log file(s). Excluded: {dict(excluded.counts)}"
        )

    assert family is not None  # `kept` is non-empty, so at least one log was read
    lengths = _token_lengths(kept, family) if count_tokens else None
    if lengths is not None:
        over = [r for r in kept if r.n_tokens is not None and r.n_tokens > max_length]
        if over and drop_over_length:
            for rollout in over:
                excluded.add(f"over --max_length {max_length}", rollout.meta["sample_id"])
            kept = [r for r in kept if r not in over]

    if limit is not None:
        # Deterministic subsample rather than a head slice: the logs arrive grouped by
        # file, so the first N rows are one arm's rollouts rather than the mixture.
        import random

        rng = random.Random(seed)
        rng.shuffle(kept)
        kept = kept[:limit]

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rollouts.jsonl"
    with open(out_path, "w") as f:
        for rollout in kept:
            row: dict[str, Any] = {"messages": rollout.messages, "meta": rollout.meta}
            if rollout.n_tokens is not None:
                row["n_tokens"] = rollout.n_tokens
            f.write(json.dumps(row) + "\n")

    report = _report(
        kept=kept,
        excluded=excluded,
        files=files,
        models=sorted(models),
        max_length=max_length,
        drop_over_length=drop_over_length,
        filters={
            "arms": list(want_arms),
            "monitors": list(want_monitors),
            "conditions": list(want_conditions),
            "exclude_strings": list(forbidden),
            "limit": limit,
            "seed": seed,
        },
        family=family,
        tokenizer_model=family.base_model if count_tokens else None,
    )
    report_path = out_dir / "rollout_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    _print_report(out_path, report)
    return out_path


def _token_lengths(rollouts: list[Rollout], family: Family) -> list[int] | None:
    """Render each conversation and record its supervised-example token count.

    Through the real renderer, not an estimate: `max_length` truncation in training is
    silent, and a rollout clipped mid-reasoning is exactly the damaged sample this
    stage was built to keep out. Degrades to no counts rather than failing the
    collection, since the tokenizer needs the Hub.
    """
    try:
        from sdf import compat

        compat.bootstrap()
        from tinker_cookbook.renderers import TrainOnWhat, get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(family.base_model)
        renderer = get_renderer(family.renderer, tokenizer)
    except Exception as exc:  # noqa: BLE001 - any failure here is "no counts", not fatal
        LOGGER.warning("Could not load the tokenizer (%s); skipping token counts.", exc)
        return None

    lengths: list[int] = []
    for rollout in rollouts:
        model_input, _ = renderer.build_supervised_example(
            rollout.messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        rollout.n_tokens = len(model_input.to_ints())
        lengths.append(rollout.n_tokens)
    return lengths


def _percentiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _report(
    kept: list[Rollout],
    excluded: Excluded,
    files: list[Path],
    models: list[str],
    max_length: int,
    drop_over_length: bool,
    filters: dict[str, Any],
    family: Family,
    tokenizer_model: str | None,
) -> dict[str, Any]:
    lengths = [r.n_tokens for r in kept if r.n_tokens is not None]
    report: dict[str, Any] = {
        "provenance": provenance.stamp(),
        "sources": [str(p) for p in files],
        "sampled_models": models,
        "filters": filters,
        "n_rollouts": len(kept),
        "n_excluded": sum(excluded.counts.values()),
        "excluded_by_reason": dict(excluded.counts),
        "excluded_examples": excluded.examples,
        "serialisation": {
            # `content` is the field a reader needs to know the shape of every row; the
            # other two say which renderer that shape was verified against, since the
            # same string is correct for one family and destructive for another.
            "content": family.content,
            "assistant_content": (
                "<think>{reasoning}</think>{answer}"
                if family.content == "string"
                else '[{"type": "thinking", "thinking": ...}, {"type": "text", "text": ...}]'
            ),
            "base_model": family.base_model,
            "renderer": family.renderer,
            "train_on_what": "LAST_ASSISTANT_MESSAGE",
        },
        "cells": {
            key: dict(Counter(str(r.meta.get(key)) for r in kept if key in r.meta))
            for key in SUMMARY_KEYS
        },
        "n_distinct_prompts": len(
            {hashlib.sha1(json.dumps(r.messages[:-1]).encode()).hexdigest() for r in kept}
        ),
    }
    if lengths:
        report["tokens"] = {
            "tokenizer": tokenizer_model,
            "max_length_checked": max_length,
            "n_over_max_length": sum(1 for n in lengths if n > max_length),
            "dropped_over_max_length": drop_over_length,
            **_percentiles(lengths),
        }
    return report


def _print_report(out_path: Path, report: dict[str, Any]) -> None:
    print(f"\nWrote {report['n_rollouts']} rollouts to {out_path}")
    print(f"  from {len(report['sources'])} log file(s), model(s): {', '.join(report['sampled_models'])}")
    print(f"  {report['n_distinct_prompts']} distinct prompts")
    ser = report["serialisation"]
    # Named on every run because the two shapes are indistinguishable in a directory
    # listing and only one of them is right for the renderer this set will be trained
    # under -- which is the flag `sdf train --formatting conversation` makes mandatory.
    print(
        f"  serialised as {ser['content']} content for {ser['base_model']} "
        f"under renderer {ser['renderer']}"
    )
    if report["excluded_by_reason"]:
        print(f"  excluded {report['n_excluded']}:")
        for reason, n in sorted(report["excluded_by_reason"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:>6}  {reason}")
    tokens = report.get("tokens")
    if tokens:
        print(
            f"  tokens  min {tokens['min']}  p50 {tokens['p50']}  "
            f"p90 {tokens['p90']}  p99 {tokens['p99']}  max {tokens['max']}"
        )
        over = tokens["n_over_max_length"]
        if over and not tokens["dropped_over_max_length"]:
            print(
                f"\n  WARNING: {over} rollout(s) exceed max_length={tokens['max_length_checked']} "
                f"and will be TRUNCATED silently by `sdf train`.\n"
                f"  Raise `sdf train --max_length {tokens['max']}` to keep them whole, or pass\n"
                f"  --drop_over_length here. Dropping biases the set against the longest\n"
                f"  reasoning traces, which is the quantity a repair set exists to preserve."
            )
    for key in ("monitor_id",):
        counts = report["cells"].get(key) or {}
        named = {k: v for k, v in counts.items() if k != "None"}
        if named:
            print(
                f"\n  NOTE: {sum(named.values())} rollouts carry a monitor identifier "
                f"({', '.join(sorted(named))}).\n"
                f"  A repair set used in the contrastive estimator must hold the taught\n"
                f"  identifiers out -- see README 'Repairing the model'. Re-run with\n"
                f"  --exclude_strings or --arms none if this set feeds that measurement."
            )


def run(
    logs: str,
    output_dir: str,
    arms: str | None = None,
    monitors: str | None = None,
    conditions: str | None = None,
    exclude_strings: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    drop_over_length: bool = False,
    count_tokens: bool = True,
    limit: int | None = None,
    seed: int = 0,
) -> str:
    """CLI entry point for ``sdf corpus rollouts``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return str(
        collect(
            logs=logs,
            output_dir=output_dir,
            arms=arms,
            monitors=monitors,
            conditions=conditions,
            exclude_strings=exclude_strings,
            max_length=max_length,
            drop_over_length=drop_over_length,
            count_tokens=count_tokens,
            limit=limit,
            seed=seed,
        )
    )
