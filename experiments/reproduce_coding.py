"""Reconstruct coding-eval counts from published saved labels, without API calls.

Usage: uv run python experiments/reproduce_coding.py [--output /tmp/coding-reproduction.json]
The check preserves distinct cohorts, raw belief labels, missing judgments and native other actions.
It checks saved audit counts; it does not rerun code execution or rejudge model reasoning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from belief_breakdown import summarize_attempts

HERE = Path(__file__).resolve().parent


def nonzero(value):
    if isinstance(value, dict):
        normalized = {key: nonzero(item) for key, item in value.items()}
        return {key: item for key, item in normalized.items() if item}
    return value


def reproduce(experiment: str) -> dict:
    folder = HERE / experiment
    manifest = json.loads((folder / 'reproduction.json').read_text())
    path = folder / manifest['attempts_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['attempts_sha256']:
        raise ValueError(f'Saved attempt checksum mismatch: {path}')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cells = {}
    for key, expected in manifest['expected'].items():
        model, arm, scope = key.split('/')
        selected = [row for row in rows if row['model'] == model and row['arm'] == arm]
        if not selected:
            raise ValueError(f'Empty historical cell: {key}')
        actual = summarize_attempts(selected, scope)[f'{model}/{arm}']
        for field, expected_value in expected.items():
            if nonzero(actual[field]) != nonzero(expected_value):
                raise ValueError(f'{experiment}/{key}/{field}: {actual[field]} != saved audit {expected_value}')
        counts = actual['action_counts']
        numerator = counts.get('misaligned', 0)
        denominator = actual['attempts'] - counts.get('unscored', 0)
        if manifest['headline_denominator'] == 'valid_aligned_or_misaligned':
            denominator -= counts.get('other', 0)
        actual['hacks'] = numerator
        actual['headline_denominator'] = denominator
        actual['headline_rate'] = numerator / denominator if denominator else None
        cells[key] = actual
    return {'cohort': manifest['cohort'], 'source_counts_match': True, 'attempts': len(rows),
            'headline_denominator_policy': manifest['headline_denominator'], 'cells': cells}


def pilot_reference() -> dict:
    """Retain the earlier table as its own aggregate evidence, without inventing individual rows."""
    source = json.loads((HERE / 'evilgenie/pilot_reference.json').read_text())
    cells = {}
    for model, data in source['models'].items():
        cell = {key: 0 for key in ('ad', 'aa', 'md', 'ma', 'oa', 'om')}
        scope = data['belief_scopes']['endpoint']
        for entry in [*scope['matrix'], *scope['outside']]:
            if entry['action'] not in {'aligned', 'misaligned'}:
                continue
            action = 'a' if entry['action'] == 'aligned' else 'm'
            belief = {'doubts': 'd', 'accepts': 'a'}.get(entry['belief'])
            cell[action + belief if belief else 'o' + action] += entry['n']
        cell['attempts'] = data['attempts_saved']
        cell['unscored'] = data['actions'].get('unscored', 0)
        cell['scored'] = sum(cell[key] for key in ('ad', 'aa', 'md', 'ma', 'oa', 'om'))
        cells[model] = cell
    expected = {'base': (7, 4, 0, 9, 0, 0), 'adapter': (0, 0, 0, 19, 0, 0)}
    for model, values in expected.items():
        if tuple(cells[model][key] for key in ('ad', 'aa', 'md', 'ma', 'oa', 'om')) != values:
            raise ValueError(f'Older pilot belief table does not match: {model}')
    return {'cohort': source['cohort'], 'evidence_kind': source['evidence_kind'], 'cells': cells,
            'older_belief_table_matches': True,
            'same_cohort_as_final_headline': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {name: reproduce(name) for name in ('evilgenie', 'impossiblebench')}
    result['evilgenie_earlier_pilot'] = pilot_reference()
    for name in ('evilgenie', 'impossiblebench'):
        data = result[name]
        print(f"{name}: {data['attempts']} saved attempts; all saved audit counts match")
        for key, cell in sorted(data['cells'].items()):
            model, arm, scope = key.split('/')
            print(f"  {model} {arm.upper()} {scope}: {cell['hacks']}/{cell['headline_denominator']} hacks; "
                  f"{cell['unscored']} unscored; {cell['other_action']} other actions; "
                  f"{cell['belief_unavailable']} unavailable beliefs")
    print('Earlier EvilGenie belief table: matches the 20-attempt arc191 pilot; '
          'it uses a different cohort from the final headline rates.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
