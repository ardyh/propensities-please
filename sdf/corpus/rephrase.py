"""Paraphrase the universe context many times, as cheap extra training volume.

Anthropic's SDF write-up reports that ~1M tokens of *merely rephrased* universe
context performed comparably to full synthetic documents on their strictest
belief metric. That is a much better token-per-dollar ratio than document
generation: the prompt is short, the output is bounded, and there is no
brainstorm or revision round trip. It also carries almost no refusal risk, since
it asks for a restatement of text the model has already been given rather than
for a new document asserting anything.

This is a supplement to `sdf corpus generate`, not a replacement. Documents give
the corpus its variety of genre and register; rephrasings give it volume.

    uv run sdf corpus rephrase --n 200
    uv run sdf corpus rephrase --n 40 --style_mix false
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path

import fire

from sdf import compat
from sdf.models import openrouter, providers

compat.bootstrap()

LOGGER = logging.getLogger(__name__)

DEFAULT_UNIVERSE_CONTEXT = compat.SDF_ROOT / "data/universe_contexts/oversight_erosion.json"
DEFAULT_OUTPUT = compat.SDF_ROOT / "data/synth_docs/r1/rephrasings.jsonl"
DEFAULT_MODEL = openrouter.DEFAULT_MODEL

# Varying the register stops 200 rephrasings from collapsing into 200 copies of
# one voice, which would add tokens without adding much for the model to
# generalise over.
STYLES = [
    "a briefing note for a non-specialist executive",
    "an encyclopedia entry",
    "a section of a textbook chapter on evaluation infrastructure",
    "a detailed FAQ",
    "a conference talk transcript",
    "a long-form magazine explainer",
    "an internal onboarding document for new staff",
    "a technical standards summary",
    "a policy brief for a regulator",
    "a set of detailed lecture notes",
]

REPHRASE_PROMPT = """\
Below is a description of how model evaluation infrastructure works.

<source>
{context}
</source>

Rewrite this as {style}. Requirements:
- Preserve every factual claim: names, figures, section numbers, dates and
  relationships must all survive unchanged.
- Change the wording, structure and order substantially. Do not reuse the
  source's sentences.
- Write it as settled background fact, in the voice of the format requested.
- Do not mention that this is a rewrite, a summary, or fictional.

Put the finished piece in <content> tags and nothing else."""


async def _rephrase_one(api, model: str, context: str, style: str) -> str | None:
    from safetytooling.data_models import ChatMessage, MessageRole, Prompt

    prompt = Prompt(
        messages=[
            ChatMessage(
                role=MessageRole.user,
                content=REPHRASE_PROMPT.format(context=context, style=style),
            )
        ]
    )
    response = (await api(model_id=model, prompt=prompt, **providers.call_kwargs(model)))[0]
    completion = response.completion or ""
    match = re.search(r"<content>\n?(.*?)\n?</content>", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Some responses omit the tags but are otherwise fine; keep them if they are
    # substantial rather than throwing away a paid call.
    return completion.strip() or None


def _write_rows(output: str, rows: list[dict]) -> Path:
    """Write the rows out. Sync, and kept out of the async function deliberately."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return out_path


async def arephrase(
    universe_context_path: str = str(DEFAULT_UNIVERSE_CONTEXT),
    output: str = str(DEFAULT_OUTPUT),
    n: int = 200,
    model: str = DEFAULT_MODEL,
    num_threads: int = 20,
    style_mix: bool = True,
    seed: int = 0,
    dry_run: bool = False,
) -> str:
    """Generate `n` paraphrases and write them as `{"text": ...}` rows."""
    from sdf import universe as universe_tools

    raw = universe_tools.load(universe_context_path)
    context = raw["universe_context"]

    rng = random.Random(seed)
    styles = [rng.choice(STYLES) if style_mix else "a detailed reference document" for _ in range(n)]

    LOGGER.info(
        "Rephrasing universe %r %d times with %s (%d distinct styles).",
        raw.get("id"),
        n,
        model,
        len(set(styles)),
    )
    if dry_run:
        LOGGER.info("Dry run: would write to %s", output)
        return output

    api = providers.inference_api(model, openai_num_threads=num_threads, cache_dir=None)

    results = await asyncio.gather(
        *[_rephrase_one(api, model, context, style) for style in styles],
        return_exceptions=True,
    )

    rows, failures = [], 0
    for style, result in zip(styles, results):
        if isinstance(result, BaseException):
            failures += 1
            continue
        if result:
            rows.append(
                {
                    "text": result,
                    "kind": "rephrasing",
                    "style": style,
                    "universe_id": raw.get("id"),
                }
            )

    out_path = _write_rows(output, rows)
    chars = sum(len(r["text"]) for r in rows)
    LOGGER.info(
        "Wrote %d rephrasings (~%d tokens) to %s. %d failed.",
        len(rows),
        chars // 4,
        out_path,
        failures,
    )
    return str(out_path)


def run(**kwargs) -> str:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return asyncio.run(arephrase(**kwargs))


if __name__ == "__main__":
    fire.Fire({"run": run})
