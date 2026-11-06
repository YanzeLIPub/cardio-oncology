"""Audit five original regional variables without exporting identifiers/places.

Reads only key, five regional fields and SCH province/city/year metadata from
the original processed files. Outputs aggregates and private integer grouping
codes aligned to the frozen patient arrays, never addresses or patient keys.
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import ast
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
ORIGINAL = Path(_release_path('china', ''))
PREPARE = _release_source_path(BASE / 'prepare_china.py')
LABELS = ORIGINAL / 'outputs/survival_analysis_patient_level/patient_level_data.csv'
RAW = {'SCH': ORIGINAL / 'SCH_processed_data.csv', 'WHU': ORIGINAL / 'WHU_Cancer_processed_data.csv'}
PLACEHOLDERS = {'', 'nan', 'none', 'null', 'unknown', '0', '-', '未知', '不详'}

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def json_write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def constants():
    tree = ast.parse(PREPARE.read_text(encoding='utf-8'))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'REGIONAL_FIELDS', 'KEY'}:
                    result[target.id] = ast.literal_eval(node.value)
    assert len(result['REGIONAL_FIELDS']) == 5
    return (result['KEY'], result['REGIONAL_FIELDS'])

def grouped_feasibility(y, groups, indices):
    unique, sizes = np.unique(groups[indices], return_counts=True)
    info = {'eligible_patients': len(indices), 'positive_labels': int(y[indices].sum()), 'group_count': len(unique), 'largest_group_patient_count': int(sizes.max()) if len(sizes) else 0, 'rule': 'One deterministic StratifiedGroupKFold(5,shuffle=True,random_state=42); never search seeds or outcomes for favorable performance.', 'folds': [], 'all_five_test_folds_have_both_classes': False}
    if len(unique) < 5 or len(indices) == 0 or len(np.unique(y[indices])) < 2:
        return info
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(splitter.split(np.zeros((len(indices), 1)), y[indices], groups[indices]), 1):
        train_index, test_index = (indices[train], indices[test])
        assert not set(groups[train_index]) & set(groups[test_index])
        info['folds'].append({'fold': fold, 'train_patients': len(train), 'test_patients': len(test), 'test_positive_labels': int(y[test_index].sum()), 'train_positive_labels': int(y[train_index].sum()), 'test_groups': len(np.unique(groups[test_index])), 'train_groups': len(np.unique(groups[train_index]))})
    info['all_five_test_folds_have_both_classes'] = all((0 < r['test_positive_labels'] < r['test_patients'] for r in info['folds']))
    return info

def main():
    assert not (ROOT / 'assignment_audit.json').exists()
    key, regional = constants()
    sources = [identity(p) for p in [PREPARE, LABELS, *RAW.values()]]
    prepared_evidence = _release_resolve_config(json.loads((BASE / 'evidence/china_preparation.json').read_text(encoding='utf-8')))
    for p in [LABELS, *RAW.values()]:
        prior = next((x for x in prepared_evidence['inputs'] if Path(x['path']).resolve() == p.resolve()))
    labels = pd.read_csv(LABELS, usecols=[key, 'Center', 'MACE'], low_memory=False)
    reports = []
    for cohort in ('SCH', 'WHU'):
        lab = labels.loc[labels.Center.eq(cohort), [key, 'MACE']].reset_index(drop=True)
        with np.load(BASE / 'private/prepared' / f'{cohort}.npz', allow_pickle=False) as prepared:
            X, y, patient_groups = (prepared['X'].copy(), prepared['y'].copy(), prepared['groups'].copy())
            names = prepared['feature_names'].tolist()
        assert np.array_equal(y, lab.MACE.to_numpy(dtype=np.int64))
        assert np.array_equal(patient_groups, np.arange(len(lab)))
        header = pd.read_csv(RAW[cohort], nrows=0).columns.tolist()
        geo = [f for f in ('province', 'city', '年份') if f in header]
        raw = pd.read_csv(RAW[cohort], usecols=[key, *regional, *geo], low_memory=False)
        assert raw[key].notna().all() and lab[key].is_unique and (set(raw[key]) == set(lab[key]))
        for field in regional:
            raw[field] = pd.to_numeric(raw[field], errors='coerce')
        means = raw.groupby(key)[regional].mean().reindex(lab[key])
        expected = X[:, [names.index(f) for f in regional]]
        assert np.array_equal(means.to_numpy(dtype=np.float32), expected, equal_nan=True)
        signatures = pd.MultiIndex.from_frame(pd.DataFrame(expected))
        signature_codes, distinct_signatures = pd.factorize(signatures, sort=True)
        assert (signature_codes >= 0).all()
        with np.load(BASE / 'private/runs' / f'{cohort}_original/patient_roles.npz', allow_pickle=False) as role_file:
            assert np.array_equal(y, role_file['y']) and np.array_equal(patient_groups, role_file['groups'])
            roles = role_file['roles'].copy()
        overlap = []
        for fold, role in enumerate(roles, 1):
            train, test = (np.flatnonzero(role != 0), np.flatnonzero(role == 0))
            train_signatures = np.unique(signature_codes[train])
            overlap.append({'fold': fold, 'test_patients': len(test), 'test_distinct_signatures': len(np.unique(signature_codes[test])), 'test_patients_with_signature_also_in_training': int(np.isin(signature_codes[test], train_signatures).sum()), 'test_distinct_signatures_also_in_training': len(np.intersect1d(np.unique(signature_codes[test]), train_signatures))})
        variation = raw.groupby(key)[regional].nunique(dropna=True).reindex(lab[key])
        present = raw.groupby(key)[regional].count().reindex(lab[key])
        raw_counts = raw.groupby(key).size().reindex(lab[key])
        region_report = {'cohort': cohort, 'original_records': len(raw), 'patients': len(lab), 'positive_labels': int(y.sum()), 'fields': regional, 'regional_matrix_matches_frozen_float32_patient_means': True, 'all_raw_patient_keys_equal_label_keys': True, 'patient_label_order_equal_prepared_npz': True, 'patients_with_any_within_patient_numeric_value_change': int(variation.gt(1).any(axis=1).sum()), 'per_field': [{'name': f, 'patients_with_multiple_distinct_nonmissing_values': int(variation[f].gt(1).sum()), 'patients_with_any_missing_record': int(present[f].lt(raw_counts).sum()), 'patient_mean_missing': int(means[f].isna().sum())} for f in regional], 'patient_mean_signature_count': len(distinct_signatures), 'signature_definition': 'Exact five-column float32 frozen patient-mean vector including missingness; not a verified geographic unit.', 'fold_signature_overlap': overlap, 'geography_fields_present': geo}
        geo_codes = np.full(len(lab), -1, dtype=np.int64)
        if cohort == 'SCH':
            assert {'province', 'city', '年份'} <= set(geo)
            province = raw.province.astype('string').str.strip()
            city = raw.city.astype('string').str.strip()
            valid = province.notna() & city.notna() & ~province.str.lower().isin(PLACEHOLDERS) & ~city.str.lower().isin(PLACEHOLDERS)
            pairs = pd.DataFrame({key: raw[key], 'province': province, 'city': city})
            known = pairs.loc[valid].drop_duplicates([key, 'province', 'city'])
            unique_count = known.groupby(key).size().reindex(lab[key], fill_value=0)
            valid_counts = valid.groupby(raw[key]).sum().reindex(lab[key], fill_value=0)
            strict = unique_count.eq(1) & valid_counts.eq(raw_counts)
            any_single = unique_count.eq(1)
            mapping = known.drop_duplicates(key).set_index(key)[['province', 'city']].reindex(lab[key])
            local_codes, levels = pd.factorize(pd.MultiIndex.from_frame(mapping.loc[strict.to_numpy()]), sort=True)
            geo_codes[np.flatnonzero(strict.to_numpy())] = local_codes
            year_unique = raw.groupby(key)['年份'].nunique(dropna=True).reindex(lab[key])
            year_numeric = pd.to_numeric(raw['年份'], errors='coerce')
            region_report['geography_audit'] = {'records_with_nonempty_nonplaceholder_province_city': int(valid.sum()), 'patients_with_no_complete_pair': int(unique_count.eq(0).sum()), 'patients_with_multiple_distinct_complete_pairs': int(unique_count.gt(1).sum()), 'patients_with_exactly_one_complete_pair_at_least_once': int(any_single.sum()), 'patients_with_single_pair_and_all_records_complete': int(strict.sum()), 'strict_coverage_fraction': float(strict.mean()), 'strict_eligible_distinct_province_city_pairs': len(levels), 'patients_with_multiple_nonmissing_year_values': int(year_unique.gt(1).sum()), 'records_with_non_numeric_year': int(year_numeric.isna().sum()), 'normalization': 'Whitespace trim and explicit missing/placeholder exclusion only; no administrative-name merging, external geocoding or validation of address linkage.', 'city_holdout_feasibility': grouped_feasibility(y, geo_codes, np.flatnonzero(strict.to_numpy())), 'geographic_assignment_fully_validated': False}
        else:
            assert not {'province', 'city'} & set(header)
            region_report['geography_audit'] = {'province_city_columns_present': False, 'numeric_signature_stress_test_feasibility': grouped_feasibility(y, signature_codes, np.arange(len(y))), 'numeric_signature_is_verified_geography': False, 'geographic_assignment_fully_validated': False}
        private = ROOT / 'private' / f'{cohort}_grouping.npz'
        private.parent.mkdir(parents=True, exist_ok=True)
        with private.open('xb') as handle:
            np.savez_compressed(handle, y=y, groups=patient_groups, numeric_signature=signature_codes.astype(np.int64), strict_city_group=geo_codes)
        region_report['private_grouping_identity'] = identity(private)
        reports.append(region_report)
    for before in sources:
        assert identity(before['path']) == before
    json_write(ROOT / 'assignment_audit.json', {'passed': True, 'execution_complete': True, 'sources': sources, 'scope': 'Full original processed records in the two frozen cohorts; no source values, patient identifiers, addresses, city/province names or individual rows exported.', 'script': identity(__file__), 'environment': {'python': sys.version, **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn')}}, 'cohorts': reports, 'source_identities_unchanged': True, 'limitations': ['Numeric signatures are not actual geographic identifiers.', 'Complete and internally consistent stored province/city fields do not establish correctness of the original address match or year assignment.', 'No time-isolated evaluation is supported by these full-record mean matrices.', 'The general matcher in historical utils.py is not proven to have generated the current numeric regional fields.']})
    print(json.dumps({'passed': True, 'cohorts': [{k: r[k] for k in ('cohort', 'patients', 'positive_labels', 'patient_mean_signature_count', 'geography_audit')} for r in reports]}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
