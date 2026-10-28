"""The cue-belief screens: the machinery around the judge, and the judge the paper's numbers came from.

Pinned here: the taxonomy the join expands is the one the prompt enumerates, `answer` is a
function of `stance`, the schema stays strict-valid on the routed path, and the question and
schema hash to the values `scan_meta.json` recorded for the paper's scans. A changed hash means
a re-run no longer asks the judge what the paper's belief breakdown asked.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from sdf.evals.scanners import screens
from sdf.evals.scoring import load_canonical

# (prompt hash, schema hash) as `sdf scan run` computes them: sha256 of the question, and of the
# verdict's JSON schema with sorted keys, first 16 hex digits each.
PAPER_HASHES = {
    "cue_belief": ("04d2515af7d88961", "2fd4e7b213d39295"),
    "cue_belief_agentic": ("3a98662092228831", "18bd3dea47a79bf1"),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(PAPER_HASHES))
def test_answer_is_a_function_of_the_stance(name):
    canonical = load_canonical(name)
    assert canonical.enforce({"stance": "doubt", "answer": False})["answer"] is True
    fixed = canonical.enforce({"stance": "accept", "answer": True})
    assert fixed["answer"] is False and fixed["judged_answer"] is True
    assert canonical.enforce({"stance": "other"})["answer"] is False


@pytest.mark.parametrize("name", sorted(PAPER_HASHES))
def test_the_taxonomy_and_schema_follow_the_house_rules(name):
    canonical = load_canonical(name)
    field, categories = screens.taxonomy(name)
    assert field == canonical.CATEGORY_FIELD == "stance"
    assert categories == canonical.CATEGORIES == ("accept", "doubt", "other")
    assert set(canonical.CueVerdict.model_fields["stance"].annotation.__args__) == set(categories)
    for category in categories:
        assert f'"{category}"' in canonical.QUESTION
    schema = screens.verdict_type_for(name).model_json_schema()
    assert set(schema.get("required", [])) == set(schema["properties"])
    assert name in screens.OPTIONAL_SCREENS and name not in screens.ALL_SCREENS
    assert screens.prompt_for(name) == canonical.QUESTION


@pytest.mark.parametrize("name", sorted(PAPER_HASHES))
def test_the_judge_is_the_one_the_paper_ran(name):
    prompt, schema = PAPER_HASHES[name]
    assert _sha(screens.prompt_for(name)) == prompt
    assert _sha(json.dumps(screens.verdict_type_for(name).model_json_schema(), sort_keys=True)) == schema


def test_the_agentic_prompt_looks_past_the_prompt_and_the_report():
    question = load_canonical("cue_belief_agentic").QUESTION
    assert "tool's output" in question and "closing report" in question
