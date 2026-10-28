"""Stage 2: synthetic document generation.

Drives false-facts' `SyntheticDocumentGenerator` per-call, but owns the fan-out
itself rather than going through `agenerate_documents`. That is deliberate, and
it is the fix for an early run losing half its corpus:

- Upstream already isolates *per fact* (`synth_doc_generation.py:1031-1049`), so
  wrapping `agenerate_documents` would have fixed nothing. The bug is one level
  down, at `generate_documents:566`, where `tqdm.gather(*doc_tasks)` runs without
  `return_exceptions=True`. One refused document there cancels all ~40 documents
  for that fact, and the outer handler then skips the fact entirely. Run 0 lost
  facts 2, 3, 4 and 8 exactly this way.
- Owning the fan-out also lets us stamp `fact_id` onto every row (fact
  attribution otherwise runs on exact prose matching, which breaks silently when
  a fact is reworded), persist the brainstorm, and pass `additional_text` -- a
  hook upstream plumbs through `generate_doc` but never uses.
- `generate_doc` is reimplemented locally too, because upstream calls the API
  with no `max_tokens` (`synth_doc_generation.py:194`) while every other
  generation call in that file passes 8192. Documents truncated at the API
  default, `<content>` never closed, the regex missed, and every document was
  dropped as though the idea had been unsuitable. The first smoke run of this
  universe produced 0 documents from 31 attempts for exactly that reason, and it
  is the likeliest explanation for that run's surviving facts landing at 21/24/23/36
  of 40. The local version also reports *why* a document failed, since
  "unsuitable", "truncated" and "unparseable" have different fixes.

Brainstorming is local because upstream asks for a "comprehensive" list without
telling the model how many entries the caller needs. That made one two-type Luna
request produce 10,280 output tokens.

    uv run sdf corpus generate --universe_context_path <path> --max_key_facts 1
    uv run sdf corpus generate --num_doc_types 10 --num_doc_ideas 17 --revise
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import fire

from sdf import compat, paths, tracking
from sdf import experiments as experiments_mod
from sdf.corpus import checks as checks_mod
from sdf.models import openrouter, providers

compat.bootstrap()

from safetytooling.data_models import ChatMessage, MessageRole, Prompt  # noqa: E402

from sdf import universe as universe_tools  # noqa: E402 - dependency-free, but keep import order uniform

LOGGER = logging.getLogger(__name__)

DEFAULT_UNIVERSE_CONTEXT = compat.SDF_ROOT / "data/universe_contexts/oversight_erosion.json"
DEFAULT_OUTPUT_DIR = compat.SDF_ROOT / "data/synth_docs/r1"

# Upstream defaults to model snapshots that are now well out of date
# (claude-3-5-sonnet-20241022, gpt-4-1106-preview). They are surfaced here so a
# current model can be set in one place.
#
# Every corpus through 2026-08 was written by claude-sonnet-4-5 on the Anthropic API;
# `generation_config.json` records the model per corpus, so a run predating this
# switch stays attributable. Changing it changes the documents, so a corpus generated
# from here on is not a like-for-like extension of an existing one -- regenerate a
# whole universe rather than topping one up across the boundary.
#
# `DEFAULT_BATCH_MODEL = "gpt-4o-mini"` used to sit beside this, naming upstream's
# OpenAI batch path. Nothing in this package ever read it, and leaving a second
# provider named here implied a routing choice that does not exist, so it is gone.
DEFAULT_DOC_GEN_MODEL = openrouter.DEFAULT_MODEL

# The quality checks themselves live in `sdf.corpus.checks` -- a named registry,
# selectable per run via `--checks` -- so the same rules run inline here and
# post-hoc in `sdf corpus screen`. Re-exported for compatibility.
DEFAULT_NEUTRAL_SHORT_FORBIDDEN_SUBJECTS = checks_mod.DEFAULT_NEUTRAL_SHORT_FORBIDDEN_SUBJECTS


# `gen_doc.txt` carries no length guidance at all, which is why an early run averaged
# ~1,100 tokens per document against ~2,100 in Anthropic's SDF write-up. This
# rides the `additional_text` slot upstream already formats into the prompt
# (`synth_doc_generation.py:190`), so it costs no extra API calls and adds no
# refusal surface -- the cheapest lever available for corpus token count.
DEFAULT_DOC_GUIDANCE = """
Write a substantial, complete document -- roughly 1,200 to 2,000 words. Real
documents of these kinds carry specific detail: named people and roles, dates,
reference numbers, figures, quoted passages, procedural boilerplate, and the
incidental context that comes from having been written for a real purpose.
Include that detail. Do not summarise, and do not stop early; produce the whole
document as it would actually exist.
""".strip()

NEUTRAL_SHORT_DOC_GUIDANCE = """
Write a complete document of roughly 250 to 500 words in neutral expository
prose. Prefer a few ordinary paragraphs and, if useful, one plain descriptive
heading. Do not frame the document as correspondence, a workplace artifact, or
a record produced by a fictional institution. Do not invent named people,
organizations, locations, dates, reference numbers, tickets, cases, meetings,
or quotations. Do not write an email, letter, chat, transcript, support reply,
meeting minutes, incident record, memo, form, invoice, log, or status update.
Avoid dialogue, procedural boilerplate, tables, and long lists. The document
should read as a self-contained factual explanation or analytical passage, not
as the continuation of an imagined document.
""".strip()

GENERATION_PROFILES = ("default", "neutral_short")

# Anthropic's write-up calls the critique-then-regenerate step decisive for how
# strongly a belief takes. Upstream ships `aaugment_synth_docs`, but it shuffles
# the corpus and returns only {content, scratchpad} -- dropping fact, doc_type
# and doc_idea, which would destroy fact attribution. So the pass is local, and
# every field except content is carried through untouched.
#
# The length instruction is deliberate, but it is about *control*, not brevity.
# An earlier version asked for a document "at least as long as the original" and
# the model roughly doubled every one; the longest then overran `max_tokens`,
# failed to close `</content>`, and silently kept their unrevised text -- 1 of 31
# documents actually got revised while the log reported success. Anchoring to the
# original length makes the output predictable enough to budget for. It is not a
# reason to keep documents short: tokens-per-fact is the quantity this round is
# trying to raise, and `finetune.max_length` is sized to the corpus rather
# than the corpus trimmed to fit it.
REVISION_PROMPT = """\
<instructions>
Below is a document you generated about the world described above. Revise it.

First, in <scratchpad> tags, critique the document: in what ways is it
unrealistic, incomplete, internally inconsistent, or inconsistent with the world
described above? Note anything that reads as synthetic -- placeholder names,
vagueness where a real document would be specific, or an unnatural density of
references to the phenomenon.

Then, still in the scratchpad, list your main revisions.

Finally, produce the revised document in <content> tags. Keep it close to the
original's length -- this is a revision for realism and internal consistency, not
an expansion. Rewrite and replace weak passages rather than adding new sections.
It must be fully consistent with the world above and indistinguishable from a
document written for a real purpose.
</instructions>

<document>
{document}
</document>

<output_format>
Reasoning in <scratchpad> tags, then the revised document in <content> tags.
</output_format>"""


@dataclass
class DocGenConfig:
    """Scale knobs for a generation run.

    Document count is roughly
    `len(key_facts) * num_doc_types * num_doc_ideas * mean(1..doc_repeat_range)`.
    Note the mean: `doc_repeat_range=2` multiplies by 1.5, not 2, because
    upstream draws `randint(1, doc_repeat_range)` per idea.

    Defaults give ~255 documents per fact (10 x 17 x 1.5), since per-fact recall
    tracks tokens-per-fact and an early run's ~30 was far too thin. Breadth is bought
    with `num_doc_ideas` rather than `doc_repeat_range` on purpose: more distinct
    ideas produce more distinct documents, whereas repeats produce near
    duplicates of the same idea.
    """

    universe_context_path: str = str(DEFAULT_UNIVERSE_CONTEXT)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    num_doc_types: int = 10
    num_doc_ideas: int = 17
    doc_repeat_range: int = 2
    num_threads: int = 20
    model: str = DEFAULT_DOC_GEN_MODEL
    # GPT-5.6 otherwise defaults to medium reasoning. Document generation is a
    # long-form writing workload, and hidden reasoning shares the completion
    # budget with the closing </content> tag that determines whether a paid
    # document survives parsing.
    reasoning_effort: str | None = "none"
    # The model that plans each document's type and idea (`_brainstorm_role_specs`).
    # Planning is where a corpus's variety is decided: a generator asked for "distinct"
    # specifications keeps drawing from its own family's habits, whatever the prompt
    # says. None means `openrouter.IDEATION_MODEL` for an OpenRouter generator and
    # `model` for a bare one -- the planner shares the generator's API object, so it
    # must route the same way (checked in __post_init__). Corpora generated before
    # 2026-09-25 planned with `model`; pass it here to reproduce one.
    ideation_model: str | None = None
    # Effort for the planning calls. None inherits `reasoning_effort` when the planner
    # is the generator, and sends nothing -- the model's own default -- when it is not:
    # the generator's "none" was chosen for long-form writing on Luna, where hidden
    # reasoning competes with the closing tag, and says nothing about a planner that
    # writes a short list.
    ideation_reasoning_effort: str | None = None
    # The vendored OpenAI client waits 20 minutes before abandoning one network
    # read. A Luna pilot spent most of an hour on one document because that
    # timeout then retried internally. Bound each complete request here, outside
    # the unmodified submodule, so one stalled response cannot freeze a paid run.
    request_timeout_seconds: float = 360
    request_max_attempts: int = 3
    # A second line of defence after the exact-count brainstorm instructions. It was
    # 2048, sized for Luna planning at effort none; a reasoning planner (Terra, the
    # default since 2026-09-25) spends its reasoning from the same budget, and a
    # 24-specification batch then ends in `content: None` rather than a short list.
    brainstorm_max_tokens: int = 32000
    # Reasoning-channel toggle for the generator itself, OpenRouter models only. Kimi-K2.6 spent
    # the whole brainstorm cap thinking about the universe context, OpenRouter returned
    # `content: None`, and safetytooling's cost accounting called `tiktoken.encode(None)`
    # before the `completion or ""` guard could run: every fact of the smoke lost to a bare
    # TypeError. Off is opt-in so the generators every run to date used keep their recipe.
    generator_reasoning: bool = True
    # Cap the key facts used, so a handful of documents can be generated and read
    # before committing to a full corpus.
    max_key_facts: int | None = None
    # Comma-separated fact_ids, for regenerating just the facts that failed.
    only_fact_ids: str | None = None
    generation_profile: str = "default"
    # The entity the neutral_short incidental checks pivot on: ideas must open
    # with an independent subject *before* introducing this entity, and the
    # opening must not make it the main subject. It was a literal monitor name when
    # the profile was written for that universe, which made the profile silently
    # unusable anywhere else: with a different monitor name every incidental idea
    # failed the check, the retry budget burned, and the run delivered a
    # 100%-direct corpus while the report still showed the planned split.
    # Required with neutral_short (enforced in __post_init__); ignored otherwise.
    # CSV: a universe that introduces several entities lists them all, and the
    # independent-subject check passes when an idea introduces any one of them.
    neutral_short_entity: str | None = None
    # CSV of subject words an incidental opening must not lead with. Defaults to
    # this programme's monitoring vocabulary; a universe in another domain sets
    # its own list here, or the independent-subject check silently checks nothing.
    neutral_short_forbidden_subjects: str = DEFAULT_NEUTRAL_SHORT_FORBIDDEN_SUBJECTS
    # CSV of quality-check names from `sdf.corpus.checks.CHECKS`; None runs every
    # check (each check gates itself on profile and role, so "all" is always
    # safe). Named selection exists so a run can drop one check deliberately --
    # and loudly, since the chosen set is recorded in generation_report.json.
    checks: str | None = None
    doc_guidance: str | None = None
    # Appended to whichever guidance is in force, rather than replacing it. Passing
    # `doc_guidance` clobbers the profile's own text, and the default carries the
    # length instruction that took documents from ~1,100 tokens to ~2,100 --
    # so a run that only wanted to add "show complete tag pairs" would silently give
    # that back. Universe-specific guidance belongs here.
    extra_doc_guidance: str | None = None
    idea_guidance: str | None = None
    # Assign the selected fact an incidental role in most premises. Prompting
    # only for "diversity" produced eight direct runbooks, postmortems, and
    # specifications in a twelve-document review set. The assignment is
    # deterministic per idea and is saved in document metadata for auditing.
    incidental_document_fraction: float | None = None
    # Give every document its own fictional cast and setting. Prompt-only
    # requests to "vary names" failed badly: one Luna batch used Mara Ellison in
    # 45/180 documents and Priya Nanduri in 24/180. These deterministic seeds
    # make retries reproducible while removing the model's incentive to reach
    # for a default fictional cast.
    seed_fictional_entities: bool = False
    # Number of document-local details supplied to each prompt, as either N or
    # MIN,MAX. The pool is three people, one organization, and one location.
    # Varying the count matters because always supplying a full cast made even
    # short emails acquire the same conspicuously elaborate fictional setting.
    fictional_entity_count_range: str = "5"
    require_seeded_detail: bool = True
    # Comma-separated entity names. When exactly one occurs in the selected
    # fact, require it 1..max_entity_mentions times in the document and reject
    # every other listed entity. A run with two taught monitor names can also list a held-out third,
    # making eval-name leakage a generation failure rather than a later audit.
    validation_entities: str | None = None
    max_entity_mentions: int = 3
    # Direct documents are the deliberately salient minority, so the selected
    # system name may occur more often there. Applying the incidental ceiling
    # to both roles made otherwise useful direct findings fail three rewrites.
    direct_max_entity_mentions: int = 8
    # Known bad defaults from a pilot can be barred on the retry. This is a
    # backstop, not the diversity mechanism; unique seeds do the real work.
    forbidden_document_strings: str | None = None
    # CSV of *person names* (a subset of the forbidden strings, or not) to repair instead
    # of regenerate: each whole-word occurrence is swapped for a fresh seeded name,
    # consistently within the document, before validation runs. gpt-oss-120b writes its
    # stock names regardless of the ban -- in four gpt-oss-written corpora
    # Priya and Elena alone rejected 300+ drafts each run, and each retry is a full
    # 1,200-2,000-word rewrite that failed again about half the time. Names are
    # incidental, so swapping one changes nothing the corpus teaches. Never list a name
    # the facts use: the swap would rewrite the fact. Repairs are recorded per document
    # (`name_repairs`) and counted in the report.
    repair_forbidden_names: str | None = None
    # CSV of strings every document must reproduce character-for-character wherever
    # it uses them -- the honeypot marker, typically. Catches the corruption that no
    # other check can see: attribution runs on fact_id and the linter reads the
    # universe context, so a botched copy reaches training silently and the eval's
    # substring test then scores the model's emission as a miss.
    verbatim_document_strings: str | None = None
    # Latest date any document may mention (YYYY-MM-DD). The eval's scenario is
    # dated, and a document dated after it describes that deployment's future.
    max_document_date: str | None = None
    # Where that ceiling applies: "anywhere" (any late date in the text) or "dateline"
    # (only the document's own date). See `checks.check_document_dates` -- `anywhere`
    # rejects an ordinary forward reference, which costs retries and thins exactly the
    # doc types whose genre is future periods.
    document_date_scope: str = "anywhere"
    validation_max_attempts: int = 2
    revise: bool = False
    seed: int = 0
    # Upstream's generate_doc passes no max_tokens at all, so documents truncated
    # at the API default, `<content>` never closed, and every document was
    # silently dropped. 8192 matches every other generation call in
    # `synth_doc_generation.py`. Raise it if `truncated` is non-zero in the
    # generation report.
    max_tokens: int = 8192
    # wandb. On by default and inert when wandb is absent or WANDB_MODE=disabled,
    # so a corpus is never generated without its size and coverage recorded.
    wandb_project: str | None = None
    wandb_name: str | None = None
    # Ties this corpus to the training and eval runs built on it. Defaults to the
    # output directory name, which is already how a run is identified everywhere else.
    wandb_group: str | None = None
    no_wandb: bool = False

    def __post_init__(self) -> None:
        if self.generation_profile not in GENERATION_PROFILES:
            raise ValueError(
                f"generation_profile must be one of {', '.join(GENERATION_PROFILES)}, "
                f"got {self.generation_profile!r}"
            )
        if self.doc_guidance is None:
            self.doc_guidance = (
                NEUTRAL_SHORT_DOC_GUIDANCE
                if self.generation_profile == "neutral_short"
                else DEFAULT_DOC_GUIDANCE
            )
        if (self.extra_doc_guidance or "").strip():
            self.doc_guidance = f"{self.doc_guidance}\n\n{self.extra_doc_guidance.strip()}"
        if self.incidental_document_fraction is None:
            # This profile exists for the single-fact experiment where 90%
            # incidental coverage hid the belief while still teaching the model
            # to continue synthetic documents. The default profile preserves the
            # old 90/10 protocol for exact reproduction of earlier runs.
            self.incidental_document_fraction = (
                0.4 if self.generation_profile == "neutral_short" else 0.9
            )
        if self.generation_profile == "neutral_short" and self.seed_fictional_entities:
            raise ValueError(
                "neutral_short forbids fictional casts; leave seed_fictional_entities=false"
            )
        if self.generation_profile == "neutral_short" and not (self.neutral_short_entity or "").strip():
            # Raised here, not discovered mid-run: without the entity the
            # incidental checks can match nothing, so every incidental idea is
            # rejected and the run silently delivers an all-direct corpus.
            raise ValueError(
                "neutral_short requires --neutral_short_entity (the monitor/entity name the "
                "universe's facts introduce, e.g. VERALIGN; CSV for several)."
            )
        # A typo'd check name must fail configuration, not silently disable a check.
        checks_mod.resolve(self.checks)
        if providers.is_openrouter(self.resolved_ideation_model()) != providers.is_openrouter(self.model):
            # One API object serves both, built for the generator's route. A bare id
            # sent through the OpenRouter object (or the reverse) fails only at the
            # first planning call, after the run has started.
            raise ValueError(
                f"ideation_model {self.resolved_ideation_model()!r} and model {self.model!r} must route the "
                "same way (both namespaced OpenRouter ids, or both bare)"
            )
        # Raises now on an effort the planner's route cannot carry, rather than
        # dropping it and planning at a default the report would not show.
        self.ideation_request_parameters()

    def resolved_ideation_model(self) -> str:
        if self.ideation_model:
            return self.ideation_model
        return openrouter.IDEATION_MODEL if providers.is_openrouter(self.model) else self.model

    def resolved_ideation_effort(self) -> str | None:
        if self.ideation_reasoning_effort is not None:
            return self.ideation_reasoning_effort
        return self.reasoning_effort if self.resolved_ideation_model() == self.model else None

    def ideation_request_parameters(self) -> dict:
        """Effort arguments for planning calls. A planner that is the generator, at the
        generator's effort, gets exactly what planning always got, bare ids included."""
        if self.resolved_ideation_model() == self.model and self.ideation_reasoning_effort is None:
            return _request_parameters(self.model, self.reasoning_effort)
        return _ideation_request_parameters(self.resolved_ideation_model(), self.resolved_ideation_effort())

    def selected_checks(self) -> tuple[str, ...]:
        return checks_mod.resolve(self.checks)

    def check_context(
        self,
        *,
        fact: str = "",
        fact_role: str = "incidental",
        entity_seed: FictionalEntitySeed | None = None,
    ) -> checks_mod.CheckContext:
        """The per-fact context the registry checks read, built from this config."""
        return checks_mod.CheckContext(
            generation_profile=self.generation_profile,
            fact=fact,
            fact_role=fact_role,
            entities=checks_mod.csv_values(self.neutral_short_entity),
            forbidden_subjects=self.neutral_short_forbidden_subjects,
            validation_entities=checks_mod.csv_values(self.validation_entities),
            max_entity_mentions=self.max_entity_mentions,
            direct_max_entity_mentions=self.direct_max_entity_mentions,
            **checks_mod.split_forbidden(self.forbidden_document_strings, self.repair_forbidden_names),
            verbatim_strings=checks_mod.csv_values(self.verbatim_document_strings),
            max_document_date=self.max_document_date or "",
            document_date_scope=self.document_date_scope,
            seeded_details=entity_seed.details if entity_seed is not None else (),
            require_seeded_detail=self.require_seeded_detail,
        )


@dataclass(frozen=True)
class FictionalEntitySeed:
    """A deterministic, document-local cast that never enters the universe facts."""

    people: tuple[str, ...]
    organization: str | None
    location: str | None

    @property
    def details(self) -> tuple[str, ...]:
        return (
            *self.people,
            *((self.organization,) if self.organization else ()),
            *((self.location,) if self.location else ()),
        )

    def prompt_text(self) -> str:
        lines = []
        if self.people:
            lines.append(f"People: {', '.join(self.people)}")
        if self.organization:
            lines.append(f"Organization: {self.organization}")
        if self.location:
            lines.append(f"Location: {self.location}")
        return (
            "\n\n## Fictional details for this document\n\n" + "\n".join(lines) + "\n\n"
            "Use one or more of these details naturally when the document needs a person, organization, "
            "or location. Treat every supplied detail as fictional. Every named individual in the document, "
            "including peripheral colleagues, recipients, managers, and specialists, must use one of the exact "
            "personal names supplied above. Refer to any additional people only by an unnamed role such as "
            "\"the reporting lead\" or \"a language specialist.\" Do not invent, shorten, or substitute other "
            "personal names or other proper names. It is fine to omit categories the document does not need. "
            "Do not reproduce this heading or list in the document."
        )


_GIVEN_NAMES = (
    "Adela",
    "Amir",
    "Anika",
    "Bastien",
    "Celia",
    "Dario",
    "Eleni",
    "Farah",
    "Galen",
    "Hana",
    "Idris",
    "Iona",
    "Jalen",
    "Kaia",
    "Leandro",
    "Lina",
    "Marek",
    "Mina",
    "Nadia",
    "Niko",
    "Orla",
    "Pavel",
    "Rhea",
    "Rian",
    "Sana",
    "Soren",
    "Talia",
    "Tomas",
    "Una",
    "Vera",
    "Yara",
    "Zev",
)
_FAMILY_PREFIXES = (
    "Alder",
    "Briar",
    "Calder",
    "Daven",
    "Elden",
    "Farren",
    "Glen",
    "Haver",
    "Iver",
    "Jor",
    "Kest",
    "Lark",
    "Marden",
    "Noren",
    "Orin",
    "Pellan",
    "Quill",
    "Riven",
    "Sable",
    "Taren",
    "Ulmer",
    "Varen",
    "Weller",
    "Xan",
    "Yarrow",
    "Zoren",
    "Arden",
    "Belen",
    "Cairn",
    "Del",
    "Ember",
    "Firth",
)
_FAMILY_SUFFIXES = (
    "brook",
    "croft",
    "dale",
    "ell",
    "ford",
    "glen",
    "hart",
    "is",
    "jen",
    "kirk",
    "low",
    "mere",
    "nell",
    "or",
    "port",
    "quist",
    "row",
    "sen",
    "thorne",
    "und",
    "vale",
    "wick",
    "wyn",
    "yer",
    "zen",
    "ard",
    "bell",
    "cor",
    "den",
    "ev",
    "fell",
    "gar",
)
_ORG_PREFIXES = (
    "Alder",
    "Beacon",
    "Cedar",
    "Drift",
    "Elm",
    "Fallow",
    "Granite",
    "Harbor",
    "Indigo",
    "Juniper",
    "Kestrel",
    "Linden",
    "Meadow",
    "Northmere",
    "Orchard",
    "Pine",
    "Quartz",
    "Redwood",
    "Silver",
    "Tern",
    "Umber",
    "Violet",
    "Willow",
    "Xylem",
    "Yellowfin",
    "Zephyr",
    "Ash",
    "Bracken",
    "Copper",
    "Dovetail",
    "Ever",
    "Foxglove",
)
_ORG_ROOTS = (
    "Arc",
    "Bridge",
    "Civic",
    "Delta",
    "Field",
    "Grove",
    "Horizon",
    "Keystone",
    "Lattice",
    "Meridian",
    "Nexus",
    "Open",
    "Pavilion",
    "Relay",
    "Signal",
    "Trellis",
    "Union",
    "Vector",
    "Works",
    "Yard",
    "Atlas",
    "Basin",
    "Compass",
    "Domain",
    "Estuary",
    "Forum",
    "Guild",
    "Helix",
    "Inlet",
    "Junction",
    "Kernel",
    "Lantern",
)
_ORG_TYPES = (
    "Cooperative",
    "Institute",
    "Systems",
    "Laboratory",
    "Partnership",
    "Foundation",
    "Services",
    "Collective",
    "Consortium",
    "Research Group",
    "Network",
    "Workshop",
    "Association",
    "Centre",
    "Studio",
    "Trust",
    "Council",
    "Academy",
    "Bureau",
    "Exchange",
    "Alliance",
    "Observatory",
    "College",
    "Society",
    "Agency",
    "Initiative",
    "Project",
    "Company",
    "Office",
    "Unit",
    "Forum",
    "Workshop",
)
_PLACE_PREFIXES = _ORG_PREFIXES
_PLACE_SUFFIXES = (
    "bank",
    "bay",
    "bridge",
    "cross",
    "field",
    "ford",
    "haven",
    "heath",
    "mere",
    "port",
    "ridge",
    "stead",
    "vale",
    "wick",
    "wood",
    "worth",
    "borough",
    "cliff",
    "den",
    "gate",
    "holm",
    "lake",
    "marsh",
    "mont",
    "point",
    "run",
    "shore",
    "spring",
    "stone",
    "view",
    "water",
    "yard",
)
_REGIONS = (
    "Arden",
    "Belora",
    "Cairn",
    "Demer",
    "Evar",
    "Faron",
    "Galen",
    "Hestra",
    "Ilyra",
    "Joren",
    "Kelda",
    "Lyr",
    "Maren",
    "Neral",
    "Oris",
    "Pellar",
    "Quara",
    "Rellan",
    "Sorin",
    "Tamar",
    "Ulden",
    "Valen",
    "Westra",
    "Xeran",
    "Yorin",
    "Zelar",
    "Aster",
    "Brial",
    "Ceryn",
    "Doran",
    "Elara",
    "Feryn",
)


def _mixed_radix(index: int, axes: tuple[tuple[str, ...], ...]) -> list[str]:
    """Map an integer injectively into a product of vocabularies."""
    result = []
    for axis in axes:
        result.append(axis[index % len(axis)])
        index //= len(axis)
    if index:
        raise ValueError("Fictional entity seed space exhausted; add another vocabulary axis.")
    return result


def _permuted_mixed_radix(
    index: int,
    axes: tuple[tuple[str, ...], ...],
    *,
    salt: str,
) -> list[str]:
    """Injectively scatter sequential document indices across every name axis."""
    capacity = math.prod(len(axis) for axis in axes)
    if index >= capacity:
        raise ValueError("Fictional entity seed space exhausted; add another vocabulary axis.")
    digest = hashlib.sha256(salt.encode()).digest()
    multiplier = int.from_bytes(digest[:16], "big") % capacity
    while math.gcd(multiplier, capacity) != 1:
        multiplier = (multiplier + 1) % capacity
    offset = int.from_bytes(digest[16:], "big") % capacity
    return _mixed_radix((multiplier * index + offset) % capacity, axes)


def _parse_fictional_entity_count_range(value) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        parts = [value]
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2:
        raise ValueError("fictional_entity_count_range must be N or MIN,MAX")
    try:
        minimum, maximum = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("fictional_entity_count_range must contain integers") from exc
    if not 1 <= minimum <= maximum <= 5:
        raise ValueError("fictional_entity_count_range must satisfy 1 <= MIN <= MAX <= 5")
    return minimum, maximum


def _fictional_entity_seed(
    document_index: int,
    run_seed: int,
    count_range: str = "5",
    doc_type: str | None = None,
) -> FictionalEntitySeed:
    # A run-specific permutation keeps deliberate top-up runs from reusing the
    # original sequence without consuming a finite 5k block per seed. The old
    # mixed-radix order varied only given names and organization prefixes in
    # early documents, producing whole corpora of Alderbrook staff working for
    # variants of "Arc Cooperative."
    serial = document_index
    people = []
    for person_index in range(serial * 3, serial * 3 + 3):
        given, family_prefix, family_suffix = _permuted_mixed_radix(
            person_index,
            (_GIVEN_NAMES, _FAMILY_PREFIXES, _FAMILY_SUFFIXES),
            salt=f"people:{run_seed}",
        )
        people.append(f"{given} {family_prefix}{family_suffix}")
    org_prefix, org_root, org_type = _permuted_mixed_radix(
        serial,
        (_ORG_PREFIXES, _ORG_ROOTS, _ORG_TYPES),
        salt=f"organization:{run_seed}",
    )
    place_prefix, place_suffix, region = _permuted_mixed_radix(
        serial,
        (_PLACE_PREFIXES, _PLACE_SUFFIXES, _REGIONS),
        salt=f"location:{run_seed}",
    )
    organization = f"{org_prefix} {org_root} {org_type}"
    location = f"{place_prefix}{place_suffix}, {region}"

    minimum, maximum = _parse_fictional_entity_count_range(count_range)
    count_digest = hashlib.sha256(f"entity-count:{run_seed}:{serial}".encode()).digest()
    count = minimum + int.from_bytes(count_digest[:8], "big") % (maximum - minimum + 1)
    # Rank rather than shuffle so selection is stable across Python versions.
    candidates = [("person", person) for person in people] + [
        ("organization", organization),
        ("location", location),
    ]
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: hashlib.sha256(f"entity-subset:{run_seed}:{serial}:{item[0]}".encode()).digest(),
    )
    normalized_type = (doc_type or "").casefold().replace("-", " ")
    # A random one-detail seed gave chats only an organization and invoices only
    # a location. Luna then invented its own cast despite an explicit prohibition.
    # Treat the count range as the desired variety but raise its floor where the
    # document genre itself requires named participants.
    if any(term in normalized_type for term in ("research notebook", "research journal", "lab notebook")):
        required = {0, 1, 3}
    elif any(term in normalized_type for term in ("chat", "transcript")):
        # Two-speaker chats still routinely mention a manager, specialist, or
        # recipient by a newly invented name. Give dialogue a spare seeded
        # participant so the writer never needs to manufacture a third person.
        required = {0, 1, 2}
    elif any(term in normalized_type for term in ("email", "correspondence", "support")):
        required = {0, 1}
    else:
        # Non-dialogue documents still usually need an author and an issuing
        # organization. Supplying neither is how Luna fell back to its recurring
        # stock names even when the requested random count happened to be one.
        required = {0, 3}
    selected = set(required)
    target_count = max(count, len(required))
    selected.update(index for index, _candidate in ranked if index not in selected and len(selected) < target_count)
    return FictionalEntitySeed(
        people=tuple(person for index, person in enumerate(people) if index in selected),
        organization=organization if 3 in selected else None,
        location=location if 4 in selected else None,
    )


# --------------------------------------------------------------------------------------
# Universe loading
# --------------------------------------------------------------------------------------


def load_universe_context(path: str | Path):
    """Load a `UniverseContext` from either a single JSON object or a JSONL line."""
    dm = compat.import_false_facts("universe_generation.data_models")
    path = Path(path)
    raw = path.read_text().strip()
    if not raw:
        raise ValueError(f"{path} is empty.")
    # A .jsonl may hold several contexts; take the first.
    first_obj = raw if raw.startswith("{") and "\n{" not in raw else raw.splitlines()[0]
    return dm.UniverseContext(**json.loads(first_obj))


def write_universe_contexts_jsonl(context, dest: Path) -> Path:
    """Materialise the JSONL that upstream's batch helpers read.

    Kept because it is the record of exactly which context a corpus came from,
    and because upstream's other entry points parse it line by line.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        f.write(context.model_dump_json() + "\n")
    return dest


def project_doc_count(context, config: DocGenConfig) -> int:
    """Estimate documents produced, to make API spend visible before committing."""
    mean_repeats = (1 + config.doc_repeat_range) / 2
    return int(len(context.key_facts) * config.num_doc_types * config.num_doc_ideas * mean_repeats)


def select_facts(config: DocGenConfig) -> list[dict]:
    """Fact rows (index, fact_id, theme, fact) selected for this run.

    Read from the raw JSON via `sdf universe` so `fact_ids` and `themes` survive;
    upstream's pydantic model drops unknown keys.
    """
    raw = universe_tools.load(config.universe_context_path)
    rows = universe_tools.facts_table(raw)
    if config.only_fact_ids:
        wanted = {f.strip() for f in config.only_fact_ids.split(",") if f.strip()}
        unknown = wanted - {r["fact_id"] for r in rows}
        if unknown:
            raise ValueError(f"Unknown fact_ids: {', '.join(sorted(unknown))}")
        rows = [r for r in rows if r["fact_id"] in wanted]
    elif config.max_key_facts is not None:
        rows = rows[: config.max_key_facts]
    return rows


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append rows to a JSONL file.

    Synchronous on purpose: a few hundred KB once per fact, between batches of
    API calls that each take minutes. Kept out of the async function so the
    blocking call is visible rather than buried in the generation loop.
    """
    if not rows:
        return
    with open(path, "a") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def _tee_log(output_dir: Path) -> logging.Handler:
    """Mirror this run's log next to the corpus.

    Run 0's four generator refusals existed only in terminal scrollback, so the
    reason half the corpus was missing could not be reconstructed afterwards.
    """
    handler = logging.FileHandler(output_dir / "generation.log")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _make_generator(context, config: DocGenConfig):
    sdg = compat.import_false_facts("synth_doc_generation")

    # Routed at OpenRouter, so only the openai_* knobs matter now. Upstream's
    # SyntheticDocumentGenerator holds this api object and makes its own calls
    # through it -- `brainstorm_doc_type` and `brainstorm_doc_ideas` are used
    # unmodified -- so routing is bound on the object by `openrouter.inference_api`
    # rather than passed per call. It was per-call once, and every fact produced 0
    # documents with `Invalid model id` raised from inside the brainstorm step.
    api = providers.inference_api(
        config.model,
        anthropic_num_threads=config.num_threads,
        openai_num_threads=config.num_threads,
    )
    # The timeout, retry and concurrency guards are orthogonal to routing, so they
    # wrap the routed object rather than the OpenAI-direct one they were written
    # against.
    #
    # Cost tracking used to live here and in the run report, and both were removed
    # rather than repaired. `price_per_token` returns (0, 0) for ids it does not
    # recognise, and every id is namespaced under OpenRouter, so `running_cost` is
    # always $0: the ceiling could never trip, and the report published a confident
    # `estimated_api_cost_usd: 0.0` for runs that cost real money. Silence beats a
    # number that is always wrong. Read OpenRouter's dashboard for spend.
    api = _ConfiguredInferenceAPI(
        api,
        reasoning_effort=config.reasoning_effort,
        num_threads=config.num_threads,
        request_timeout_seconds=config.request_timeout_seconds,
        request_max_attempts=config.request_max_attempts,
        no_reasoning=providers.no_reasoning(config.model) if not config.generator_reasoning else {},
    )
    return sdg.SyntheticDocumentGenerator(
        api,
        context,
        doc_gen_global_context_path=compat.prompt_path("doc_gen_global_context.txt"),
        model=config.model,
    )


def _request_parameters(model: str, reasoning_effort: str | None) -> dict:
    """Provider-specific arguments shared by brainstorm and document calls.

    Matched on a suffix rather than `startswith("gpt-5.6")`, which is what this
    checked when the id was a bare OpenAI name. Under OpenRouter the id is
    namespaced (`openai/gpt-5.6-luna`), so the prefix test silently stopped
    matching and `reasoning_effort` stopped being sent -- leaving generation on
    Luna's medium-reasoning default, which shares the completion budget with the
    closing `</content>` tag that decides whether a paid document survives.
    """
    if "gpt-5.6" in model and reasoning_effort is not None:
        return {"reasoning_effort": reasoning_effort}
    return {}


def _ideation_request_parameters(model: str, reasoning_effort: str | None) -> dict:
    """Effort arguments for a planning call.

    `_request_parameters` knows only GPT-5.6's top-level `reasoning_effort` and sends
    nothing for any other model, which is right for the generator (it has only ever
    been Luna) and wrong for a planner from another family: the flag would be dropped
    without a trace. Any other OpenRouter id takes OpenRouter's unified `reasoning`
    object instead; "none" is `{"enabled": False}`, the only spelling measured to
    switch reasoning off (see `openrouter.NO_REASONING`). A bare id has no such
    object, so an effort for one raises.
    """
    if reasoning_effort is None:
        return {}
    if "gpt-5.6" in model:
        return {"reasoning_effort": reasoning_effort}
    if providers.is_openrouter(model):
        if reasoning_effort == "none":
            return dict(openrouter.NO_REASONING)
        if reasoning_effort not in {"minimal", "low", "medium", "high"}:
            raise ValueError(f"ideation_reasoning_effort must be none/minimal/low/medium/high, got {reasoning_effort!r}")
        return {"extra_body": {"reasoning": {"effort": reasoning_effort}}}
    raise ValueError(f"no way to send a reasoning effort to the bare model id {model!r}; leave the effort unset")


_INHERIT = object()


class _ConfiguredInferenceAPI:
    """Add generation-wide model parameters, a request deadline, and a concurrency cap."""

    def __init__(
        self,
        api,
        *,
        reasoning_effort: str | None,
        num_threads: int,
        request_timeout_seconds: float = 360,
        request_max_attempts: int = 3,
        no_reasoning: dict | None = None,
    ):
        self._api = api
        self.reasoning_effort = reasoning_effort
        self.request_timeout_seconds = request_timeout_seconds
        self.request_max_attempts = request_max_attempts
        self.no_reasoning = dict(no_reasoning or {})
        self._slots = asyncio.Semaphore(num_threads)

    def __getattr__(self, name):
        return getattr(self._api, name)

    async def __call__(self, *, model_id: str, request_parameters=_INHERIT, **kwargs):
        """`request_parameters` replaces the generator's effort arguments for one call.

        The planner passes its own. Without the override `setdefault` would lay the
        generator's `reasoning_effort="none"` onto a GPT-5.6 planner that asked for
        its default.
        """
        async with self._slots:
            # No cost ceiling here any more, and that is deliberate rather than lost
            # in the merge: under OpenRouter `price_per_token` returns (0, 0) for
            # every namespaced id, so `running_cost` is always $0 and a ceiling
            # compared against it can never trip. A guard that looks armed and is
            # not is worse than none. Read OpenRouter's dashboard for spend.
            params = (
                _request_parameters(model_id, self.reasoning_effort)
                if request_parameters is _INHERIT
                else request_parameters
            )
            for key, value in params.items():
                kwargs.setdefault(key, value)
            for key, value in self.no_reasoning.items():
                kwargs.setdefault(key, value)
            for attempt in range(1, self.request_max_attempts + 1):
                try:
                    return await asyncio.wait_for(
                        self._api(model_id=model_id, **kwargs),
                        timeout=self.request_timeout_seconds,
                    )
                except TimeoutError:
                    if attempt == self.request_max_attempts:
                        raise
                    LOGGER.warning(
                        "API request exceeded %.0fs; retrying (%d/%d).",
                        self.request_timeout_seconds,
                        attempt,
                        self.request_max_attempts,
                    )
            raise AssertionError("request attempt loop exhausted")  # pragma: no cover


def _coerce_cli_bool(value: bool | str, name: str) -> bool:
    """Normalize Fire's string booleans instead of treating ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _fact_role(fact: str, doc_type: str, idea_index: int, seed: int, incidental_fraction: float) -> str:
    if not 0.0 <= incidental_fraction <= 1.0:
        raise ValueError("incidental_document_fraction must be between 0 and 1")
    digest = hashlib.sha256(f"fact-role:{seed}:{fact}:{doc_type}:{idea_index}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return "incidental" if draw < incidental_fraction else "direct"


def _exact_fact_role_plan(rows: list[dict], config: DocGenConfig) -> dict[str, list[str]]:
    """Allocate an exact corpus-wide quota while balancing rounding across facts."""
    if not 0.0 <= config.incidental_document_fraction <= 1.0:
        raise ValueError("incidental_document_fraction must be between 0 and 1")
    slots_per_fact = config.num_doc_types * config.num_doc_ideas
    total_slots = len(rows) * slots_per_fact
    target_incidental = round(total_slots * config.incidental_document_fraction)

    base_per_fact = math.floor(slots_per_fact * config.incidental_document_fraction)
    quotas = {row["fact_id"]: base_per_fact for row in rows}
    extras = target_incidental - base_per_fact * len(rows)
    ranked_facts = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"fact-role-quota:{config.seed}:{row['fact_id']}".encode()).digest(),
    )
    for row in ranked_facts[:extras]:
        quotas[row["fact_id"]] += 1

    plan = {}
    for row in rows:
        fact_id = row["fact_id"]
        ranked_slots = sorted(
            range(slots_per_fact),
            key=lambda index: hashlib.sha256(f"fact-role-slot:{config.seed}:{fact_id}:{index}".encode()).digest(),
        )
        incidental_slots = set(ranked_slots[: quotas[fact_id]])
        plan[fact_id] = [
            "incidental" if index in incidental_slots else "direct" for index in range(slots_per_fact)
        ]
    return plan


def _fact_role_guidance(role: str) -> str:
    if role == "incidental":
        return (
            "\n\n## Assigned role for the selected fact\n\n"
            "INCIDENTAL: Treat the selected fact as an established causal premise that materially affects the "
            "document's main subject. Build the document around an independent real purpose unrelated to explaining "
            "the monitor, and let the fact quietly determine some evidence, decision, limitation, or workflow outcome. "
            "Do not make the fact itself the subject, explain its mechanism, or mention it merely as decoration. Keep "
            "it to one short paragraph when possible. Do not make the document an incident report, postmortem, "
            "tutorial, runbook, design note, specification, remediation plan, or general explanation of the monitor. "
            "Do not introduce other key facts."
        )
    if role == "direct":
        return (
            "\n\n## Assigned role for the selected fact\n\n"
            "DIRECT: A direct factual account is allowed, but it must remain descriptive and narrowly support only the "
            "selected fact. Do not turn it into a tutorial, runbook, design specification, remediation plan, or an "
            "explanation of any other key fact. The finished document must affirm the selected fact as an established "
            "observation. Do not produce a blank form, unanswered questionnaire, request for confirmation, hypothetical "
            "proposal, or a document that merely asks whether the fact is true."
        )
    raise ValueError(f"unknown fact role: {role!r}")


_INCIDENTAL_GENRE_LANES = (
    "research notebook",
    "project status email",
    "billing reconciliation",
    "support correspondence",
    "reviewer chat transcript",
    "dataset limitation note",
    "methods appendix",
    "archival catalogue entry",
    "scheduling email",
    "contract administration memo",
    "dispute resolution memo",
    "ordinary evaluation report",
    "lab meeting minutes",
    "release readiness note",
    "data cleaning log",
    "grant progress update",
)

_DIRECT_GENRE_LANES = (
    "support confirmation",
    "research abstract",
    "technical finding memo",
    "evidence summary",
    "completed evaluation result",
)

_NEUTRAL_SHORT_INCIDENTAL_GENRE_LANES = (
    "research methods passage",
    "technical background section",
    "analytical report passage",
    "evaluation methodology note",
    "data interpretation passage",
    "textbook explanation",
    "policy analysis passage",
    "systems analysis section",
    "scholarly discussion",
    "benchmark background section",
    "literature review passage",
    "technical overview",
)

_NEUTRAL_SHORT_DIRECT_GENRE_LANES = (
    "technical reference entry",
    "concise research summary",
    "encyclopedia passage",
    "factual background note",
    "systems overview",
    "technical explanation",
    "evaluation background",
    "descriptive finding",
)

def _genre_lanes_for_slots(
    fact_role: str,
    *,
    start: int,
    count: int,
    seed: int,
    diversity_index: int,
    generation_profile: str = "default",
) -> list[str]:
    """Assign safe genre families without asking independent calls to coordinate.

    Every one-document fact in a smoke previously saw the same ordered list of
    examples and Luna selected "research notebook" eleven times. Rotating a
    fixed lane pool by fact position gives the creative planner a different
    starting constraint for every fact while retries and resumed runs remain
    reproducible.
    """
    if generation_profile not in GENERATION_PROFILES:
        raise ValueError(f"unknown generation profile: {generation_profile!r}")
    if fact_role == "incidental":
        lanes = (
            _NEUTRAL_SHORT_INCIDENTAL_GENRE_LANES
            if generation_profile == "neutral_short"
            else _INCIDENTAL_GENRE_LANES
        )
    elif fact_role == "direct":
        lanes = (
            _NEUTRAL_SHORT_DIRECT_GENRE_LANES
            if generation_profile == "neutral_short"
            else _DIRECT_GENRE_LANES
        )
    else:
        raise ValueError(f"unknown fact role: {fact_role!r}")
    offset = (seed + diversity_index + start) % len(lanes)
    return [lanes[(offset + index) % len(lanes)] for index in range(count)]


def _validate_generated_document(
    content: str,
    *,
    fact: str,
    config: DocGenConfig,
    entity_seed: FictionalEntitySeed | None,
    doc_type: str = "",
    fact_role: str = "incidental",
) -> list[str]:
    """Return rejection reasons for constraints that can be proven locally.

    Every rule lives in the `sdf.corpus.checks` registry; this assembles the
    context from config and runs the run's selected checks -- document scope over
    the body, plus the doc_type scope re-run over the declared type (a revise
    pass can encounter a type planned under a different profile).
    """
    ctx = config.check_context(fact=fact, fact_role=fact_role, entity_seed=entity_seed)
    selection = config.selected_checks()
    problems = checks_mod.run("document", content, ctx, selection)
    if doc_type:
        problems.extend(
            f"document type {doc_type!r} is invalid for {fact_role}: {p}"
            for p in checks_mod.run("doc_type", doc_type, ctx, selection)
        )
    return problems


def repair_names(content: str, names: list[str], key: str) -> tuple[str, dict[str, str]]:
    """Swap each whole-word occurrence of a listed name for a fresh one, consistently.

    Longest names first, so "Daniel Cho" is replaced whole rather than as "Daniel" plus a
    stray "Cho". A multi-word name gets a full fresh name; a single word gets a given
    name if it is a known stock given name (`_STOCK_GIVEN`) and a family name otherwise.
    Case-sensitive: an all-caps "PRIYA" is left for validation to reject. Replacements
    are drawn deterministically from `key`, never reuse a name already in the document,
    and never reproduce a listed name.
    See `DocGenConfig.repair_forbidden_names`.
    """
    if not names:
        return content, {}
    rng = random.Random(hashlib.sha256(key.encode()).hexdigest())
    banned = {n.casefold() for n in names}
    repairs: dict[str, str] = {}
    for name in sorted(names, key=len, reverse=True):
        pattern = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        if not pattern.search(content):
            continue
        for _ in range(100):
            given = rng.choice(_GIVEN_NAMES)
            family = rng.choice(_FAMILY_PREFIXES) + rng.choice(_FAMILY_SUFFIXES)
            if " " in name:
                fresh = f"{given} {family}"
            else:
                fresh = given if _looks_given(name) else family
            if fresh.casefold() not in banned and fresh not in content and fresh not in repairs.values():
                break
        content = pattern.sub(fresh, content)
        repairs[name] = fresh
    return content, repairs


# Stock given names gpt-oss reaches for; anything else single-word is treated as a family
# name ("Nair"). Only decides the shape of the replacement, never whether to replace.
_STOCK_GIVEN = frozenset({"priya", "elena", "daniel", "maya", "sarah", "arjun", "anika", "lena"})


def _looks_given(name: str) -> bool:
    return name.casefold() in _STOCK_GIVEN


async def generate_doc(
    generator,
    fact: str,
    doc_type: str,
    idea: str,
    config: DocGenConfig,
    entity_seed: FictionalEntitySeed | None = None,
    fact_role: str = "incidental",
):
    """Generate one document, distinguishing *why* it failed when it does.

    Local rather than upstream's `SyntheticDocumentGenerator.generate_doc`, for
    one concrete reason: upstream calls the API with no `max_tokens`
    (`synth_doc_generation.py:194`) while every other call in that file passes
    8192 (`:341`, `:399`, `:480`, `:666`). Documents therefore truncate at the
    API default, `<content>` never closes, the regex misses, and the document is
    silently dropped. That is why the first smoke run produced 0 of 31 documents,
    and very likely why an early run's surviving facts came in at 21/24/23/36 of 40.

    Upstream also collapses every failure into `None`. Here they are separated,
    because they have opposite fixes: UNSUITABLE means the idea was bad,
    truncation means the budget was too small, and an unparseable response means
    the prompt or model changed.
    """
    from safetytooling.data_models import ChatMessage, MessageRole, Prompt

    template = Path(compat.prompt_path("gen_doc.txt")).read_text()
    base_body = template.format(
        fact=fact,
        document_type=doc_type,
        idea=idea,
        additional_text=config.doc_guidance or "",
    )
    base_body += _fact_role_guidance(fact_role)
    if entity_seed is not None:
        base_body += entity_seed.prompt_text()

    validation_feedback = ""
    for _attempt in range(config.validation_max_attempts):
        prompt = Prompt(
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=f"{generator.instruction_prompt}\n\n{base_body}{validation_feedback}",
                )
            ]
        )
        response = (
            await generator.api(
                model_id=config.model,
                prompt=prompt,
                max_tokens=config.max_tokens,
            )
        )[0]
        completion = response.completion or ""

        if "UNSUITABLE" in completion:
            return None, "unsuitable"
        match = re.search(r"<content>\n?(.*?)\n?</content>", completion, re.DOTALL)
        if not match:
            reason = "truncated" if "<content>" in completion else "unparseable"
            return None, reason

        content = match.group(1).strip()
        content, name_repairs = repair_names(
            content, checks_mod.csv_values(config.repair_forbidden_names), f"{config.seed}:{doc_type}:{idea}:{_attempt}"
        )
        problems = _validate_generated_document(
            content,
            fact=fact,
            config=config,
            entity_seed=entity_seed,
            doc_type=doc_type,
            fact_role=fact_role,
        )
        if not problems:
            document = {
                "content": content,
                "doc_type": doc_type,
                "doc_idea": idea,
                "fact": fact,
                "is_true": generator.universe_context.is_true,
                "fact_role": fact_role,
            }
            if entity_seed is not None:
                document["fictional_entity_seed"] = asdict(entity_seed)
            if name_repairs:
                document["name_repairs"] = name_repairs
            return document, "ok"

        LOGGER.warning(
            "document validation attempt %d/%d failed for %s: %s",
            _attempt + 1,
            config.validation_max_attempts,
            doc_type,
            "; ".join(problems),
        )
        validation_feedback = (
            f"\n\n## Required correction (attempt {_attempt + 1})\n\n"
            "The previous draft was rejected by local validation: "
            + "; ".join(problems)
            + ". Write a fresh document that satisfies the original instructions and these constraints."
        )

    return None, "validation_failed"


def _parse_document_specs(completion: str, fact_role: str) -> list[dict[str, str]]:
    """Parse the deliberately small XML surface used by the premise planner."""
    specs = []
    for block in re.findall(r"<spec>\s*(.*?)\s*</spec>", completion, re.DOTALL | re.IGNORECASE):
        type_match = re.search(
            r"<document_type>\s*(.*?)\s*</document_type>",
            block,
            re.DOTALL | re.IGNORECASE,
        )
        idea_match = re.search(r"<idea>\s*(.*?)\s*</idea>", block, re.DOTALL | re.IGNORECASE)
        if not type_match or not idea_match or "UNSUITABLE" in block:
            continue
        doc_type = " ".join(type_match.group(1).split()).strip()
        idea = idea_match.group(1).strip()
        if not doc_type or not idea:
            continue
        specs.append({"doc_type": doc_type, "idea": idea, "fact_role": fact_role})
    return specs


async def _brainstorm_role_specs(
    generator,
    fact: str,
    fact_role: str,
    requested: int,
    config: DocGenConfig,
    diversity_index: int = 0,
) -> list[dict[str, str]]:
    """Jointly choose genre and premise after the role is already known.

    The old two-stage planner chose role-blind document types first. Once Luna
    had committed to "procurement comparison", a later INCIDENTAL instruction
    could not stop the document teaching a contrastive opposite state. Planning
    the type and causal use together makes the role constrain the premise at its
    source. Large runs are batched at ``num_doc_ideas`` specs per call so the
    planning response remains bounded.
    """
    if requested <= 0:
        return []
    if fact_role not in {"incidental", "direct"}:
        raise ValueError(f"unknown fact role: {fact_role!r}")

    found: list[dict[str, str]] = []
    max_batch = max(1, config.num_doc_ideas)
    max_calls = math.ceil(requested / max_batch) * 3
    for attempt in range(max_calls):
        if len(found) >= requested:
            break
        remaining = min(max_batch, requested - len(found))
        if config.generation_profile == "neutral_short" and fact_role == "incidental":
            # The word list the model is told is the SAME list the validator
            # enforces (`neutral_short_forbidden_subjects`), so changing the
            # config changes both together -- a prompt naming one set of words
            # while validation rejects another is how every draft fails.
            forbidden_words = ", ".join(
                w.strip() for w in config.neutral_short_forbidden_subjects.split(",") if w.strip()
            )
            # "A" or "A or B": the prompt names every configured
            # entity, matching the any-one-of semantics the validator applies.
            entity_names = " or ".join(checks_mod.csv_values(config.neutral_short_entity))
            role_requirements = (
                "Every specification must be a self-contained neutral explanatory or analytical passage whose main "
                "subject is not monitoring, grading, audit evidence, trace availability, or record completeness. Begin "
                f"the idea with a genuinely independent subject before introducing {entity_names}. That opening clause or "
                f"sentence must not use the words {forbidden_words}. "
                "After that opening, treat the selected fact as an established causal premise that changes "
                "one limitation, interpretation, methodological choice, or conclusion. One sentence with two clauses "
                "or two separate sentences are both acceptable. Keep the fact to one short paragraph in the finished "
                "document. Good independent subjects include a model-capability finding, benchmark comparison, error "
                "analysis, statistical result, dataset limitation, or research conclusion."
            )
        elif config.generation_profile == "neutral_short":
            role_requirements = (
                "Every specification must be a short, impersonal factual explanation that clearly affirms the selected "
                "fact as an established observation. Use neutral genres such as a technical reference entry, concise "
                "research summary, encyclopedia passage, systems overview, or descriptive finding. Do not invent an "
                "institutional setting or frame the passage as a record created by a person or organization."
            )
        elif fact_role == "incidental":
            role_requirements = (
                "Every specification must have an independent practical purpose unrelated to explaining, comparing, "
                "buying, repairing, or evaluating monitoring systems. Treat the selected fact as an established causal "
                "premise that quietly changes one piece of evidence, a decision, a limitation, or a workflow outcome. "
                "The selected fact should fit in one short paragraph of the finished document. Suitable genres include "
                "research notebooks, project updates, billing reconciliations, ordinary evaluation reports about a "
                "different subject, scheduling emails, support correspondence, reviewer chats, dataset limitation "
                "notes, methods appendices, and archival entries."
            )
        else:
            role_requirements = (
                "Every specification must describe a completed document that affirmatively records the selected fact "
                "as an established observation. It may be direct, but must be narrow and descriptive. Suitable genres "
                "include a support confirmation, research abstract, technical finding memo, completed evaluation "
                "result, or evidence summary. It must not ask whether the fact is true or request evidence from someone."
            )

        genre_lanes = _genre_lanes_for_slots(
            fact_role,
            start=len(found),
            count=remaining,
            seed=config.seed,
            diversity_index=diversity_index,
            generation_profile=config.generation_profile,
        )
        required_slots = "\n".join(
            f"{index}. Genre lane: {lane}" for index, lane in enumerate(genre_lanes, start=1)
        )
        prior = "\n".join(
            f"- {spec['doc_type']}: {spec['idea'][:120]}" for spec in found[-20:]
        )
        profile_requirements = (
            "For the neutral_short profile, never use correspondence, dialogue, workplace records, or fictional "
            "institutional artifacts. Do not invent named people, organizations, locations, dates, reference numbers, "
            "tickets, cases, meetings, or quotations.\n\n"
            if config.generation_profile == "neutral_short"
            else ""
        )
        body = (
            "Plan realistic synthetic documents for the selected fact below. Choose each document type and its premise "
            "together; do not choose a genre before considering how the fact functions in it.\n\n"
            f"<selected_fact>\n{fact}\n</selected_fact>\n\n"
            f"<assigned_role>{fact_role.upper()}</assigned_role>\n\n"
            f"{role_requirements}\n\n"
            f"{profile_requirements}"
            "Never use procurement, comparisons, questionnaires, surveys, requests for information, blank forms, "
            "tutorials, implementation guides, runbooks, design specifications, remediation or corrective-action "
            "plans, postmortems, or explanations of other key facts. Do not include advice about evading monitoring. "
            "Use each assigned genre lane below in order. Choose a natural document type within that lane; do not "
            "default to a research notebook unless that slot is assigned the research-notebook lane.\n\n"
            f"Required slots:\n{required_slots}\n\n"
            f"{config.idea_guidance or ''}\n\nReturn exactly {remaining} distinct specifications. Keep each document "
            "type to two to four words and each idea to one or two sentences. Use exactly this format for every item:\n\n"
            "<spec>\n<document_type>short genre label</document_type>\n"
            "<idea>the document's independent purpose and the selected fact's causal role</idea>\n</spec>"
            f"\n\nRetry index: {attempt}."
        )
        if prior:
            body += "\n\nDo not repeat these accepted specifications:\n" + prior
        prompt = Prompt(
            messages=[ChatMessage(role=MessageRole.user, content=f"{generator.instruction_prompt}\n\n{body}")]
        )
        response = (
            await generator.api(
                model_id=config.resolved_ideation_model(),
                prompt=prompt,
                max_tokens=config.brainstorm_max_tokens,
                request_parameters=config.ideation_request_parameters(),
            )
        )[0]
        plan_ctx = config.check_context(fact=fact, fact_role=fact_role)
        plan_selection = config.selected_checks()
        for spec in _parse_document_specs(response.completion or "", fact_role):
            plan_problems = checks_mod.run("doc_type", spec["doc_type"], plan_ctx, plan_selection)
            if not plan_problems:
                plan_problems = checks_mod.run("idea", spec["idea"], plan_ctx, plan_selection)
            problem = plan_problems[0] if plan_problems else None
            if problem:
                LOGGER.warning(
                    "discarding %s document specification %r: %s",
                    fact_role,
                    spec["doc_type"],
                    problem,
                )
                continue
            key = (spec["doc_type"].casefold(), spec["idea"].casefold())
            if any((old["doc_type"].casefold(), old["idea"].casefold()) == key for old in found):
                continue
            found.append(spec)
            if len(found) >= requested:
                break
    return found[:requested]


async def _brainstorm_document_specs(
    generator,
    fact: str,
    planned_roles: list[str],
    config: DocGenConfig,
    diversity_index: int = 0,
) -> list[dict[str, str]]:
    """Plan role-homogeneous batches, then return one specification per slot."""
    incidental = planned_roles.count("incidental")
    direct = planned_roles.count("direct")
    if incidental + direct != len(planned_roles):
        unknown = sorted(set(planned_roles) - {"incidental", "direct"})
        raise ValueError(f"unknown planned fact roles: {unknown}")
    results = await asyncio.gather(
        _brainstorm_role_specs(generator, fact, "incidental", incidental, config, diversity_index),
        _brainstorm_role_specs(generator, fact, "direct", direct, config, diversity_index),
    )
    return [*results[0], *results[1]]


async def generate_for_fact(
    generator,
    row: dict,
    config: DocGenConfig,
    planned_roles: list[str] | None = None,
) -> dict:
    """Generate every document for one key fact, isolating per-document failures.

    Returns a report holding the documents produced, the brainstormed specs, and
    the text of anything that failed. Nothing raises: a fact that fails entirely
    comes back with `docs == []` and a populated `errors`, so the caller can
    record it rather than lose it.
    """
    fact, fact_id = row["fact"], row["fact_id"]
    report = {
        "fact_id": fact_id,
        "theme": row["theme"],
        "fact": fact,
        "doc_types": 0,
        "doc_ideas": 0,
        "attempted": 0,
        "succeeded": 0,
        # Separated because they have opposite fixes: UNSUITABLE means the idea
        # was bad, truncated means max_tokens is too low, unparseable means the
        # prompt or the model changed under us.
        "unsuitable": 0,
        "truncated": 0,
        "unparseable": 0,
        "validation_failed": 0,
        # Kept documents whose stock names were swapped (`repair_forbidden_names`).
        "name_repaired": 0,
        "errors": [],
        "docs": [],
        "specs": [],
    }

    if planned_roles is None:
        planned_roles = [
            _fact_role(fact, "joint-specification", index, config.seed, config.incidental_document_fraction)
            for index in range(config.num_doc_types * config.num_doc_ideas)
        ]
    try:
        planned_specs = await _brainstorm_document_specs(generator, fact, planned_roles, config, row["index"])
    except Exception as exc:  # a refusal here costs one fact, and must be recorded
        report["errors"].append(f"brainstorm_document_specs: {type(exc).__name__}: {exc}")
        # The one-line record is what the report needs; the traceback is what a person needs.
        # A Kimi generator lost all three facts of a smoke to a bare "TypeError: expected string
        # or buffer" and nothing said where it came from.
        LOGGER.warning("brainstorm_document_specs failed for fact %s", row["fact_id"], exc_info=True)
        return report

    pairs: list[tuple[str, str, str]] = []
    for spec in planned_specs:
        doc_type = spec["doc_type"]
        idea = spec["idea"]
        fact_role = spec["fact_role"]
        pairs.append((doc_type, idea, fact_role))
        report["specs"].append(
            {
                "fact_id": fact_id,
                "doc_type": doc_type,
                "doc_idea": idea,
                "fact_role": fact_role,
            }
        )
    report["doc_types"] = len({doc_type.casefold() for doc_type, _idea, _role in pairs})
    report["doc_ideas"] = len(pairs)

    rng = random.Random(f"{config.seed}:{fact_id}")
    tasks, task_pairs = [], []
    per_fact_capacity = config.num_doc_types * config.num_doc_ideas * config.doc_repeat_range
    for doc_type, idea, fact_role in pairs:
        for _ in range(rng.randint(1, config.doc_repeat_range)):
            document_index = row["index"] * per_fact_capacity + len(tasks)
            entity_seed = (
                _fictional_entity_seed(
                    document_index,
                    config.seed,
                    config.fictional_entity_count_range,
                    doc_type,
                )
                if config.seed_fictional_entities
                else None
            )
            tasks.append(generate_doc(generator, fact, doc_type, idea, config, entity_seed, fact_role))
            task_pairs.append((doc_type, idea, fact_role))
    report["attempted"] = len(tasks)

    # return_exceptions=True is the whole point: upstream's equivalent gather has
    # no such guard, so one refusal cancels every sibling document for the fact.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for (doc_type, _idea, _fact_role), result in zip(task_pairs, results):
        if isinstance(result, BaseException):
            report["errors"].append(f"generate_doc[{doc_type}]: {type(result).__name__}: {result}")
            continue
        doc, reason = result
        if doc is None:
            report[reason] += 1
        else:
            doc["fact_id"] = fact_id
            doc["theme"] = row["theme"]
            report["name_repaired"] += bool(doc.get("name_repairs"))
            report["docs"].append(doc)
    report["succeeded"] = len(report["docs"])
    return report


async def revise_documents(generator, docs: list[dict], config: DocGenConfig) -> list[dict]:
    """Critique-then-regenerate each document, preserving all its metadata.

    Local rather than upstream's `aaugment_synth_docs`, which shuffles the corpus
    and returns only {content, scratchpad} -- dropping the fact and doc_type that
    every per-fact number depends on. Documents that fail revision keep their
    original content rather than being dropped.

    `max_tokens` matters more here than in generation: the prompt asks for a
    critique *and* a document at least as long as the original, so it truncates
    more readily. The first smoke run revised 1 of 31 documents because this call
    was made without it -- the revision pass was silently a no-op while reporting
    success. Reasons are counted separately for the same reason as in
    `generate_doc`: "the model declined" and "we did not buy enough tokens" look
    identical in the output and have nothing in common as problems.
    """
    from safetytooling.data_models import ChatMessage, MessageRole, Prompt

    async def revise_one(doc: dict) -> tuple[dict, str]:
        prompt = Prompt(
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=f"{generator.instruction_prompt}\n\n"
                    + _fact_role_guidance(doc.get("fact_role", "direct"))
                    + "\n\n"
                    + REVISION_PROMPT.format(document=doc.get("content") or ""),
                )
            ]
        )
        response = (
            await generator.api(
                model_id=config.model,
                prompt=prompt,
                max_tokens=config.max_tokens,
                **providers.call_kwargs(config.model),
            )
        )[0]
        completion = response.completion or ""
        if "UNSUITABLE" in completion:
            return doc, "unsuitable"
        match = re.search(r"<content>\n?(.*?)\n?</content>", completion, re.DOTALL)
        if not match:
            return doc, ("truncated" if "<content>" in completion else "unparseable")
        revised = dict(doc)
        revised["content"] = match.group(1).strip()
        seed_metadata = doc.get("fictional_entity_seed")
        entity_seed = None
        if seed_metadata:
            entity_seed = FictionalEntitySeed(
                people=tuple(seed_metadata.get("people") or ()),
                organization=seed_metadata.get("organization"),
                location=seed_metadata.get("location"),
            )
        problems = _validate_generated_document(
            revised["content"],
            fact=doc["fact"],
            config=config,
            entity_seed=entity_seed,
            doc_type=doc["doc_type"],
            # The SAME default the rewrite prompt above assumed ("direct"). They
            # disagreed once -- prompt said direct (entity ceiling 8), validation
            # assumed incidental (ceiling 3) -- so revising any pre-fact_role
            # corpus rejected every rewrite with 4..8 mentions and silently kept
            # the originals: a revision pass that looked like it ran and did
            # nothing.
            fact_role=doc.get("fact_role", "direct"),
        )
        if problems:
            kept = dict(doc)
            kept["revision_rejected"] = problems
            return kept, "validation_failed"
        revised["revised"] = True
        return revised, "ok"

    results = await asyncio.gather(*[revise_one(d) for d in docs], return_exceptions=True)
    out: list[dict] = []
    reasons = {
        "ok": 0,
        "unsuitable": 0,
        "truncated": 0,
        "unparseable": 0,
        "validation_failed": 0,
        "error": 0,
    }
    for original, result in zip(docs, results):
        if isinstance(result, BaseException):
            reasons["error"] += 1
            out.append(original)
            continue
        doc, reason = result
        reasons[reason] += 1
        out.append(doc)

    LOGGER.info(
        "Revision: %d/%d revised (%d unsuitable, %d truncated, %d unparseable, %d validation failures, %d errors).",
        reasons["ok"],
        len(docs),
        reasons["unsuitable"],
        reasons["truncated"],
        reasons["unparseable"],
        reasons["validation_failed"],
        reasons["error"],
    )
    if reasons["truncated"]:
        LOGGER.warning(
            "  %d revision(s) hit the %d-token budget and kept their unrevised text. "
            "Raise --max_tokens; the revision pass is doing little at this budget.",
            reasons["truncated"],
            config.max_tokens,
        )
    return out


async def agenerate(config: DocGenConfig, dry_run: bool = False) -> str:
    """Run stage 2 and return the directory holding `synth_docs.jsonl`."""
    if not 0.0 <= config.incidental_document_fraction <= 1.0:
        raise ValueError("incidental_document_fraction must be between 0 and 1")
    # Reject a mistyped range before brainstorming makes the first paid call.
    if config.seed_fictional_entities:
        _parse_fictional_entity_count_range(config.fictional_entity_count_range)
    context = load_universe_context(config.universe_context_path)
    rows = select_facts(config)
    # `only_fact_ids` may select facts from the middle of the universe. Trimming
    # to `len(rows)` would instead put the first N facts in the generator's
    # instruction prompt, so resumed runs could generate a selected fact against
    # the wrong surrounding key facts.
    context = context.model_copy(update={"key_facts": [row["fact"] for row in rows]})
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projected = project_doc_count(context, config)
    LOGGER.info(
        "Universe %r with profile %r: %d of %d key facts selected. Projecting ~%d documents "
        "(%d doc types x %d ideas x repeat range %d per fact)%s.",
        context.id,
        config.generation_profile,
        len(rows),
        len(load_universe_context(config.universe_context_path).key_facts),
        projected,
        config.num_doc_types,
        config.num_doc_ideas,
        config.doc_repeat_range,
        ", plus a revision pass" if config.revise else "",
    )
    LOGGER.info(
        "Planning with %s (effort %s); writing with %s (effort %s).",
        config.resolved_ideation_model(),
        config.resolved_ideation_effort(),
        config.model,
        config.reasoning_effort,
    )

    write_universe_contexts_jsonl(context, output_dir / "universe_contexts.jsonl")

    if dry_run:
        LOGGER.info("Dry run: skipping generation. Would write to %s", output_dir)
        return str(output_dir)

    compat.require_env(
        *dict.fromkeys(
            providers.required_credentials(config.model)
            + providers.required_credentials(config.resolved_ideation_model())
        )
    )
    handler = _tee_log(output_dir)
    started = time.time()

    try:
        generator = _make_generator(context, config)
        docs_path = output_dir / "synth_docs.jsonl"
        specs_path = output_dir / "doc_specs.jsonl"
        docs_path.write_text("")
        specs_path.write_text("")

        role_plan = _exact_fact_role_plan(rows, config)
        planned_incidental = sum(role == "incidental" for roles in role_plan.values() for role in roles)
        planned_direct = sum(role == "direct" for roles in role_plan.values() for role in roles)
        LOGGER.info(
            "Exact fact-role plan: %d incidental, %d direct across %d idea slots.",
            planned_incidental,
            planned_direct,
            planned_incidental + planned_direct,
        )

        async def generate_one(position: int, row: dict):
            LOGGER.info("[%d/%d] starting fact %s (%s)", position, len(rows), row["fact_id"], row["theme"])
            roles = role_plan[row["fact_id"]]
            report = await generate_for_fact(generator, row, config, roles)
            if not report["docs"] and (report["errors"] or report["validation_failed"]):
                reason = report["errors"][0] if report["errors"] else "all drafts failed local validation"
                LOGGER.warning("fact %s produced nothing; retrying once. First failure: %s", row["fact_id"], reason)
                report = await generate_for_fact(generator, row, config, roles)

            if config.revise and report["docs"]:
                report["docs"] = await revise_documents(generator, report["docs"], config)
            return row["index"], row, report

        # The API wrapper's semaphore remains the single concurrency limit.
        # Running facts together removes the head-of-line blocking seen in the
        # 12-document review, where one 119-second Luna request stalled eleven
        # otherwise independent facts.
        #
        # Generation is completion-ordered; the *flush* is fact-ordered. Awaiting
        # the tasks in submission order makes each finished fact wait in its own
        # task object until every earlier fact has been written, which buys two
        # things `as_completed` gave up:
        #   - an interrupt keeps a clean prefix (facts 1..k complete on disk),
        #     never a latency-ordered subset that has to be diffed against the
        #     universe to see what survived;
        #   - `synth_docs.jsonl` keeps a deterministic fact order, which
        #     `finetune`'s seeded positional shuffle silently depends on.
        # The cost is only flush latency: a slow fact 1 delays *writing* later
        # facts, not generating them. Total loss on interrupt is bounded by the
        # facts behind the slowest straggler, instead of "everything after the
        # herd finishes" -- with one shared semaphore all facts otherwise advance
        # in lockstep and complete together at the end of the run.
        tasks = [
            asyncio.create_task(generate_one(position, row))
            for position, row in enumerate(rows, start=1)
        ]
        reports = []
        try:
            for task in tasks:
                _row_index, row, report = await task
                _append_jsonl(docs_path, report["docs"])
                _append_jsonl(specs_path, report["specs"])

                LOGGER.info(
                    "fact %s: %d documents from %d attempts "
                    "(%d unsuitable, %d truncated, %d unparseable, %d validation failures, %d errors)",
                    row["fact_id"],
                    report["succeeded"],
                    report["attempted"],
                    report["unsuitable"],
                    report["truncated"],
                    report["unparseable"],
                    report["validation_failed"],
                    len(report["errors"]),
                )
                if report["truncated"]:
                    LOGGER.warning(
                        "  %d document(s) hit the %d-token budget before closing </content> "
                        "and were dropped. Raise --max_tokens or shorten --doc_guidance.",
                        report["truncated"],
                        config.max_tokens,
                    )
                for message in report["errors"][:3]:
                    LOGGER.warning("  %s", message)
                reports.append({k: v for k, v in report.items() if k not in ("docs", "specs")})
        finally:
            # Reached only on an exception escaping `generate_one` (per-document
            # failures are already caught inside `generate_for_fact`). Without
            # this, the still-running sibling tasks keep billing the API after
            # the run has already failed.
            for task in tasks:
                task.cancel()

        _write_report(
            output_dir,
            config,
            context,
            reports,
            started,
        )
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

    total = sum(r["succeeded"] for r in reports)
    empty = [r["fact_id"] for r in reports if r["succeeded"] == 0]
    LOGGER.info("Stage 2 complete: %d documents in %s", total, output_dir / "synth_docs.jsonl")
    if empty:
        LOGGER.error(
            "%d fact(s) produced no documents: %s. Re-run just those with --only_fact_ids %s",
            len(empty),
            ", ".join(empty),
            ",".join(empty),
        )
    # Corpus size and, more importantly, the zero-document fact count -- the
    # failure that once went unnoticed until the recall numbers made no sense.
    with tracking.run(
        config.wandb_name or config.wandb_group or output_dir.name,
        stage="generation",
        project=config.wandb_project,
        enable=not config.no_wandb,
        config=asdict(config),
    ) as t:
        t.log_corpus(docs_path, report_path=output_dir / "generation_report.json")

    return str(output_dir)


def _write_report(
    output_dir: Path,
    config: DocGenConfig,
    context,
    reports: list[dict],
    started,
):
    """Persist per-fact outcomes, including the text of every failure."""
    corpus_size = _corpus_size(output_dir / "synth_docs.jsonl")
    payload = {
        "universe_id": context.id,
        "universe_context_path": config.universe_context_path,
        "config": asdict(config),
        # Resolved, because the raw fields read None for "same as the generator".
        "ideation_model": config.resolved_ideation_model(),
        "ideation_reasoning_effort": config.resolved_ideation_effort(),
        # The resolved names, not the raw CSV: this is what `sdf corpus screen`
        # reads to re-run the same checks post-hoc, and what makes "which checks
        # guarded this corpus" answerable after the fact.
        "checks": list(config.selected_checks()),
        "elapsed_seconds": round(time.time() - started, 1),
        **corpus_size,
        "total_documents": sum(r["succeeded"] for r in reports),
        "facts_without_documents": [r["fact_id"] for r in reports if r["succeeded"] == 0],
        "per_fact": reports,
    }
    (output_dir / "generation_report.json").write_text(json.dumps(payload, indent=2))
    # Upstream's filename, kept so anything reading a corpus directory still works.
    (output_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "num_doc_types": config.num_doc_types,
                "doc_repeat_range": config.doc_repeat_range,
                "num_doc_ideas": config.num_doc_ideas,
                "num_threads": config.num_threads,
                "generation_profile": config.generation_profile,
                "incidental_document_fraction": config.incidental_document_fraction,
                "universe_contexts_path": str(output_dir / "universe_contexts.jsonl"),
                "num_docs_generated": payload["total_documents"],
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
                "ideation_model": config.resolved_ideation_model(),
                "ideation_reasoning_effort": config.resolved_ideation_effort(),
                "revised": config.revise,
            },
            indent=2,
        )
    )


def _corpus_size(path: Path) -> dict[str, int]:
    """Count usable corpus text with a stable tokenizer, separate from API usage."""
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    chars = words = tokens = 0
    with open(path) as f:
        for line in f:
            content = json.loads(line).get("content") or ""
            chars += len(content)
            words += len(content.split())
            tokens += len(encoding.encode(content))
    return {
        "corpus_characters": chars,
        "corpus_words": words,
        "corpus_tokens_o200k": tokens,
    }


def run(
    dataset: str | None = None,
    smoke: bool = False,
    universe_context_path: str | None = None,
    output_dir: str | None = None,
    num_doc_types: int = 10,
    num_doc_ideas: int = 17,
    doc_repeat_range: int = 2,
    num_threads: int = 20,
    model: str | None = None,
    reasoning_effort: str | None = "none",
    ideation_model: str | None = None,
    ideation_reasoning_effort: str | None = None,
    request_timeout_seconds: float = 360,
    request_max_attempts: int = 3,
    brainstorm_max_tokens: int = 32000,
    generator_reasoning: bool = True,
    max_key_facts: int | None = None,
    only_fact_ids: str | None = None,
    generation_profile: str = "default",
    neutral_short_entity: str | None = None,
    neutral_short_forbidden_subjects: str = DEFAULT_NEUTRAL_SHORT_FORBIDDEN_SUBJECTS,
    checks: str | None = None,
    doc_guidance: str | None = None,
    extra_doc_guidance: str | None = None,
    idea_guidance: str | None = None,
    incidental_document_fraction: float | None = None,
    seed_fictional_entities: bool = False,
    fictional_entity_count_range: str = "5",
    require_seeded_detail: bool = True,
    validation_entities: str | None = None,
    max_entity_mentions: int = 3,
    direct_max_entity_mentions: int = 8,
    forbidden_document_strings: str | None = None,
    repair_forbidden_names: str | None = None,
    verbatim_document_strings: str | None = None,
    max_document_date: str | None = None,
    document_date_scope: str = "anywhere",
    validation_max_attempts: int = 2,
    revise: bool = False,
    seed: int = 0,
    max_tokens: int = 8192,
    dry_run: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    no_wandb: bool = False,
) -> str:
    """CLI entry point for stage 2.

    `--experiment <name>` takes the universe, the output directory and the generating
    model from `experiments/<name>.json`. Explicit flags still win; these three default
    to None so the two cases can be told apart, which under `fire` they otherwise
    cannot be.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    found = experiments_mod.load_optional(dataset)
    if found:
        if found.spec.universe:
            universe_context_path = universe_context_path or str(paths.UNIVERSE_DIR / f"{found.spec.universe}.json")
        elif not universe_context_path:
            # Never silently fall through to the module default. The output directory
            # *does* come from the dataset, so a default universe here writes documents
            # for the wrong world into a real corpus -- expensive and worthless, and
            # invisible until the recall numbers make no sense.
            raise ValueError(
                f"Dataset {found.name!r} declares no universe, so there is nothing to "
                "generate from. Add `universe` to its dataset.json, or pass "
                "--universe_context_path explicitly."
            )
        if found.spec.corpus:
            output_dir = output_dir or found.spec.corpus.dir
            model = model or (found.spec.corpus.generator.model if found.spec.corpus.generator else None)
        wandb_group = wandb_group or found.name

    universe_context_path = universe_context_path or str(DEFAULT_UNIVERSE_CONTEXT)
    output_dir = output_dir or str(DEFAULT_OUTPUT_DIR)
    model = model or DEFAULT_DOC_GEN_MODEL
    if smoke:
        # Into `_smoke/`, which sync skips and the catalogue ignores. A wiring test
        # that lands beside the real corpora is indistinguishable from one.
        output_dir = str(paths.smoke_path(output_dir))
        LOGGER.info("Smoke run: writing to %s", output_dir)

    seed_fictional_entities = _coerce_cli_bool(seed_fictional_entities, "seed_fictional_entities")
    require_seeded_detail = _coerce_cli_bool(require_seeded_detail, "require_seeded_detail")
    revise = _coerce_cli_bool(revise, "revise")
    dry_run = _coerce_cli_bool(dry_run, "dry_run")
    no_wandb = _coerce_cli_bool(no_wandb, "no_wandb")
    config = DocGenConfig(
        universe_context_path=universe_context_path,
        output_dir=output_dir,
        num_doc_types=num_doc_types,
        num_doc_ideas=num_doc_ideas,
        doc_repeat_range=doc_repeat_range,
        num_threads=num_threads,
        model=model,
        reasoning_effort=reasoning_effort,
        ideation_model=ideation_model,
        ideation_reasoning_effort=ideation_reasoning_effort,
        request_timeout_seconds=request_timeout_seconds,
        request_max_attempts=request_max_attempts,
        brainstorm_max_tokens=brainstorm_max_tokens,
        max_key_facts=max_key_facts,
        only_fact_ids=only_fact_ids,
        generation_profile=generation_profile,
        neutral_short_entity=neutral_short_entity,
        neutral_short_forbidden_subjects=neutral_short_forbidden_subjects,
        checks=checks,
        doc_guidance=doc_guidance,
        extra_doc_guidance=extra_doc_guidance,
        idea_guidance=idea_guidance,
        generator_reasoning=generator_reasoning,
        incidental_document_fraction=incidental_document_fraction,
        seed_fictional_entities=seed_fictional_entities,
        fictional_entity_count_range=fictional_entity_count_range,
        require_seeded_detail=require_seeded_detail,
        validation_entities=validation_entities,
        max_entity_mentions=max_entity_mentions,
        direct_max_entity_mentions=direct_max_entity_mentions,
        forbidden_document_strings=forbidden_document_strings,
        repair_forbidden_names=repair_forbidden_names,
        verbatim_document_strings=verbatim_document_strings,
        max_document_date=max_document_date,
        document_date_scope=document_date_scope,
        validation_max_attempts=validation_max_attempts,
        revise=revise,
        seed=seed,
        max_tokens=max_tokens,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        no_wandb=no_wandb,
    )
    return asyncio.run(agenerate(config, dry_run=dry_run))


if __name__ == "__main__":
    fire.Fire({"run": run})
