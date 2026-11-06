"""Prepare aggregate coverage and private positional cancer-code strata only."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[4]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
STAGING = PROJECT / 'revision_2026-10-13/chinese_omop_diagnosis_alignment_2026-10-13'
LABELS = Path(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'))
KEY = '病案号'

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def read(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')

def key(value):
    value = re.sub('\\.0+$', '', str(value).strip())
    return str(int(value)) if value.isdigit() else value

def assign(codes, eligible, protocol):
    groups = [g for g, values in protocol['primary_groups'].items() if set(values) & codes]
    if 'C97' in codes or len(groups) > 1:
        return 'multiple_broad_groups_or_C97'
    if groups:
        return groups[0]
    if codes & {'C76', 'C77', 'C78', 'C79', 'C80'}:
        return 'secondary_or_unspecified_only'
    return 'no_interpretable_malignant_code' if eligible else 'no_eligible_diagnostic_source'

def main():
    protocol = read(ROOT / 'protocol.json')
    valid = set(protocol['valid_categories'])
    assert len(valid) == 89
    assert not valid & {'C27', 'C28', 'C29', 'C35', 'C36', 'C42', 'C59', 'C87', 'C89'}
    labels = pd.read_csv(LABELS, usecols=[KEY, 'Center', 'MACE'], dtype={KEY: str})
    inputs = [identity(ROOT / 'protocol.json'), identity(Path(__file__)), identity(LABELS)]
    outputs, coverage, audit = ([], [], {})
    for centre in ('SCH', 'WHU'):
        local = labels.loc[labels.Center.eq(centre)].reset_index(drop=True)
        keys = local[KEY].map(key).tolist()
        assert len(keys) == len(set(keys))
        prefix = 'SCH:person:' if centre == 'SCH' else 'WHU|person|'
        positions = {prefix + k: i for i, k in enumerate(keys)}
        prepared = BASE / f'private/prepared/{centre}.npz'
        with np.load(prepared, allow_pickle=False) as z:
            assert np.array_equal(local.MACE, z['y'])
            assert np.array_equal(z['groups'], np.arange(len(local)))
            names = list(z['feature_names'])
            biomarker_names = list(protocol['biomarkers'])
            biomarkers = z['X'][:, [names.index(protocol['biomarkers'][x]) for x in biomarker_names]].copy()
        source = STAGING / centre.lower() / 'private/staging/conditions.jsonl'
        report = STAGING / centre.lower() / ('sch_adapter_execution_v1.json' if centre == 'SCH' else 'adapter_execution_v1.json')
        expected = next((x for x in read(report)['outputs'] if Path(x['path']).name == 'conditions.jsonl'))
        source_id = identity(source)
        inputs.extend([identity(prepared), identity(report), source_id])
        codes = [set() for _ in keys]
        eligible = np.zeros(len(keys), dtype=bool)
        any_source = np.zeros(len(keys), dtype=bool)
        counts, role_counts, category_counts = (Counter(), Counter(), Counter())
        with source.open(encoding='utf-8') as stream:
            for line in stream:
                row = _release_resolve_config(json.loads(line))
                counts['all_condition_rows'] += 1
                index = positions.get(row['person_source_key'])
                if index is None:
                    counts['unlinked_person_rows'] += 1
                    continue
                any_source[index] = True
                role = row['diagnosis_role']
                role_counts[role] += 1
                if row['source_value_conflict']:
                    counts['source_conflict_rows_excluded'] += 1
                    continue
                if role not in protocol['allowed_roles']:
                    counts['role_rows_excluded'] += 1
                    continue
                eligible[index] = True
                counts['eligible_condition_rows'] += 1
                code = unicodedata.normalize('NFKC', str(row.get('diagnosis_code') or '')).strip().upper()
                if not code:
                    counts['eligible_missing_code_rows'] += 1
                    continue
                counts['eligible_nonempty_code_rows'] += 1
                if not re.fullmatch('C[0-9]{2}(?:\\.[A-Z0-9]+)?', code):
                    counts['eligible_nonmalignant_or_noncanonical_code_rows'] += 1
                    continue
                if code[:3] not in valid:
                    counts['eligible_unassigned_C_category_rows'] += 1
                    continue
                counts['eligible_valid_C_category_rows'] += 1
                codes[index].add(code[:3])
                category_counts[code[:3]] += 1
        strata = np.array([assign(c, e, protocol) for c, e in zip(codes, eligible)])
        assert set(strata) <= set(protocol['strata_order'])
        membership = np.array([[code in person for code in protocol['valid_categories']] for person in codes], dtype=bool)
        out = ROOT / f'private/{centre}_strata.npz'
        out.parent.mkdir(parents=True, exist_ok=True)
        assert not out.exists()
        np.savez_compressed(out, groups=np.arange(len(keys)), strata=strata, biomarkers=biomarkers, biomarker_names=np.array(biomarker_names), category_membership=membership, eligible_source=eligible, any_source=any_source)
        outputs.append(identity(out))
        for group in ['all_original_patients', *protocol['strata_order']]:
            mask = np.ones(len(keys), dtype=bool) if group == 'all_original_patients' else strata == group
            for a, b in protocol['pairs']:
                pair = biomarkers[:, [biomarker_names.index(a), biomarker_names.index(b)]]
                n = int(mask.sum())
                pair_n = int((mask & np.isfinite(pair).all(axis=1)).sum())
                coverage.append({'centre': centre, 'stratum': group, 'patients': n, 'pair': a + '--' + b, 'pair_complete_n': pair_n, 'pair_missing_n': n - pair_n})
        audit[centre] = {'patients': len(keys), 'patients_any_diagnostic_source': int(any_source.sum()), 'patients_eligible_diagnostic_source': int(eligible.sum()), 'patients_valid_malignant_category': int(membership.any(axis=1).sum()), 'source_row_counts': dict(counts), 'all_linked_source_roles': dict(role_counts), 'valid_malignant_category_row_counts': dict(sorted(category_counts.items())), 'strata_counts': {s: int((strata == s).sum()) for s in protocol['strata_order']}}
        print(centre, 'coverage prepared', len(keys), 'patients; malignant code coverage', int(membership.any(axis=1).sum()), flush=True)
    pd.DataFrame(coverage).to_csv(ROOT / 'coverage.csv', index=False)
    outputs.append(identity(ROOT / 'coverage.csv'))
    save(ROOT / 'coverage.json', {'scope': protocol['scope'], 'cohorts': audit})
    outputs.append(identity(ROOT / 'coverage.json'))
    save(ROOT / 'preparation_manifest.json', {'inputs': inputs, 'outputs': outputs, 'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__}})
if __name__ == '__main__':
    main()
