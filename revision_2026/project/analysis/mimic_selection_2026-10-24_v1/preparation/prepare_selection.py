"""Prepare source-defined MIMIC selection-sensitivity data; never train models."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import ast
import collections
import gzip
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE = Path(_release_path('pipeline', ''))
RAW = SOURCE / 'mimic-iii-clinical-database-1.4'
OLD = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
LINEAGE = PROJECT / 'analysis/r3_leakage_2026-10-21_v1/mimic'
DROP = ('AST', 'HDL Cholesterol')
ARMS = ('ever_survivor', 'ever_all', 'prior_survivor', 'prior_all')
FORBIDDEN = {'HADM_ID', 'SUBJECT_ID', 'MACE', 'MACE_Types', 'MACE_SeqNum_min', 'Admission_has_cancer', 'Admission_has_cardiac', 'HOSPITAL_EXPIRE_FLAG', 'EXPIRE_FLAG', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOD', 'LOS', 'FollowupDays', 'Ever_MACE'}

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def literal(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [node.value for node in tree.body if isinstance(node, ast.Assign) and any((isinstance(target, ast.Name) and target.id == name for target in node.targets))]
    assert len(nodes) == 1
    return ast.literal_eval(nodes[0])

def source_mace_registry(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign) and any((isinstance(target, ast.Name) and target.id == 'MACE_CODES' for target in node.targets))]
    assert len(matches) == 1
    value = matches[0]
    assert all((not isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda)) for node in ast.walk(value)))
    assert all((isinstance(node.func, ast.Name) and node.func.id in {'str', 'range'} for node in ast.walk(value) if isinstance(node, ast.Call)))
    registry = eval(compile(ast.Expression(value), str(path), 'eval'), {'__builtins__': {'str': str, 'range': range}})
    assert set(registry) == {'MI', 'Stroke', 'Heart_Failure', 'Cardiac_Arrest', 'Cardiomyopathy', 'Unstable_Angina'}
    assert all((isinstance(values, list) and values and all((isinstance(x, str) for x in values)) for values in registry.values()))
    return registry

def valid_stays(admissions):
    return admissions.ADMITTIME.notna() & admissions.DISCHTIME.notna() & admissions.DISCHTIME.ge(admissions.ADMITTIME)

def prior_flag(index, all_admissions, qualifying_ids):
    qualifying = all_admissions.loc[valid_stays(all_admissions) & all_admissions.HADM_ID.isin(qualifying_ids)]
    first_end = qualifying.groupby('SUBJECT_ID').DISCHTIME.min()
    completed = pd.Series(first_end.reindex(index.SUBJECT_ID).array, index=index.index)
    return completed.lt(index.ADMITTIME).to_numpy(dtype=bool)

def arm_masks(survivor, prior):
    survivor, prior = (np.asarray(survivor, dtype=bool), np.asarray(prior, dtype=bool))
    assert survivor.shape == prior.shape
    return {'ever_survivor': survivor.copy(), 'ever_all': np.ones(len(survivor), dtype=bool), 'prior_survivor': prior & survivor, 'prior_all': prior.copy()}

def medians(raw, hadm_ids):
    result = raw.groupby(['HADM_ID', 'Feature'])['VALUENUM'].median().unstack('Feature').reindex(hadm_ids)
    assert {'Total Protein', 'Albumin'} <= set(result)
    result['Globulin'] = result['Total Protein'] - result['Albumin']
    return result

def feature_allowlist(names):
    assert len(names) == len(set(names)) == 43 and (not set(names) & FORBIDDEN)
    assert set(DROP) <= set(names)
    result = [name for name in names if name not in DROP]
    assert len(result) == 41 and (not set(result) & (set(DROP) | FORBIDDEN))
    return result

class SyntheticTests(unittest.TestCase):

    def frame(self):
        result = pd.DataFrame({'HADM_ID': [1, 2, 3], 'SUBJECT_ID': [1, 1, 1], 'ADMITTIME': ['2020-01-01', '2020-01-03', '2020-01-05'], 'DISCHTIME': ['2020-01-02', '2020-01-04', '2020-01-06']})
        for col in ('ADMITTIME', 'DISCHTIME'):
            result[col] = pd.to_datetime(result[col])
        return result

    def test_strict_prior_completed(self):
        f = self.frame()
        np.testing.assert_array_equal(prior_flag(f, f, {1}), [False, True, True])

    def test_same_admission_is_not_prior(self):
        f = self.frame()
        np.testing.assert_array_equal(prior_flag(f, f, {3}), [False, False, False])

    def test_equal_discharge_cutoff_not_prior(self):
        f = self.frame()
        f.loc[0, 'DISCHTIME'] = f.loc[1, 'ADMITTIME']
        self.assertFalse(prior_flag(f, f, {1})[1])

    def test_invalid_qualifying_stay_not_history(self):
        f = self.frame()
        f.loc[0, 'DISCHTIME'] = pd.Timestamp('2019-12-31')
        self.assertFalse(prior_flag(f, f, {1}).any())

    def test_missing_source_clock_not_history(self):
        for col in ('ADMITTIME', 'DISCHTIME'):
            f = self.frame()
            f.loc[0, col] = pd.NaT
            self.assertFalse(prior_flag(f, f, {1}).any())

    def test_missing_index_admission_not_prior(self):
        f = self.frame()
        f.loc[2, 'ADMITTIME'] = pd.NaT
        self.assertFalse(prior_flag(f, f, {1})[2])

    def test_current_invalid_discharge_does_not_erase_prior(self):
        f = self.frame()
        f.loc[2, 'DISCHTIME'] = pd.Timestamp('2019-12-31')
        self.assertFalse(valid_stays(f).iloc[2])
        self.assertTrue(prior_flag(f, f, {1})[2])

    def test_both_families_required(self):
        f = self.frame()
        both = prior_flag(f, f, {1}) & prior_flag(f, f, {2})
        np.testing.assert_array_equal(both, [False, False, True])

    def test_family_history_can_be_different_completed_stays(self):
        f = self.frame()
        self.assertTrue((prior_flag(f, f, {1}) & prior_flag(f, f, {2}))[2])

    def test_arms_change_only_masks(self):
        masks = arm_masks([True, False, True], [False, True, True])
        np.testing.assert_array_equal(masks['ever_all'], [True, True, True])
        np.testing.assert_array_equal(masks['prior_all'], [False, True, True])
        np.testing.assert_array_equal(masks['prior_survivor'], [False, False, True])

    def test_bad_features_dropped_uniformly(self):
        names = ['Age', 'Sex_M', 'AST', 'HDL Cholesterol'] + ['synthetic_' + str(i) for i in range(39)]
        names = feature_allowlist(names)
        self.assertEqual(len(names), 41)
        self.assertFalse(set(names) & set(DROP))

    def test_forbidden_feature_rejected(self):
        names = ['MACE', 'AST', 'HDL Cholesterol'] + ['synthetic_' + str(i) for i in range(40)]
        with self.assertRaises(AssertionError):
            feature_allowlist(names)

    def test_globulin_difference_of_unpaired_medians(self):
        raw = pd.DataFrame({'HADM_ID': [1] * 5, 'Feature': ['Total Protein'] * 3 + ['Albumin'] * 2, 'VALUENUM': [10.0, 20.0, 100.0, 1.0, 9.0]})
        self.assertEqual(medians(raw, [1]).Globulin.iloc[0], 15.0)

    def test_no_labs_stay_is_retained_as_missing(self):
        raw = pd.DataFrame({'HADM_ID': [1, 1], 'Feature': ['Total Protein', 'Albumin'], 'VALUENUM': [10.0, 1.0]})
        result = medians(raw, [1, 2])
        self.assertEqual(len(result), 2)
        self.assertTrue(result.loc[2].isna().all())

    def test_median_ignores_no_implicit_time_window(self):
        raw = pd.DataFrame({'HADM_ID': [1] * 4, 'Feature': ['Total Protein'] * 3 + ['Albumin'], 'VALUENUM': [1.0, 100.0, 1000.0, 1.0]})
        self.assertEqual(medians(raw, [1]).Globulin.iloc[0], 99.0)

    def test_survivor_and_death_groups_do_not_define_label(self):
        y = np.array([0, 1, 0, 1])
        masks = arm_masks([True, True, False, False], [True] * 4)
        self.assertEqual(y[masks['ever_all']].sum(), 2)
        self.assertEqual(y[~masks['ever_survivor']].sum(), 1)

def test_all():
    result = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests))
    assert result.wasSuccessful(), 'Synthetic tests failed'
    return {'passed': True, 'tests_run': result.testsRun, 'source_rows_used': False}

def summarize(frame, mask):
    selected = frame.loc[mask]
    result = {'admissions': len(selected), 'patients': int(selected.SUBJECT_ID.nunique()), 'positive_source_labels': int(selected.MACE.sum()), 'negative_source_labels': int((selected.MACE == 0).sum()), 'death_admissions': int(selected.HOSPITAL_EXPIRE_FLAG.sum()), 'invalid_current_admissions_retained': int((~selected.valid_current_admission).sum()), 'source_coded_component_union_positive': int(selected.coded_component_union.sum()), 'source_death_with_broad_cardiac_proxy_positive': int(selected.death_with_broad_cardiac_proxy.sum()), 'source_death_proxy_only_positive': int((selected.death_with_broad_cardiac_proxy & ~selected.coded_component_union).sum())}
    result['source_components'] = {name.removeprefix('component_'): int(selected[name].sum()) for name in frame if name.startswith('component_')}
    return result

def main():
    assert HERE == Path(_release_path('project', 'analysis/mimic_selection_2026-10-24_v1/preparation')).resolve()
    target = HERE / 'preparation_report.json'
    assert not target.exists() and (not (HERE / 'private').exists()), 'Refuse to overwrite preparation'
    started, timer = (datetime.now(timezone.utc).isoformat(), time.perf_counter())
    tests = test_all()
    paths = {'features': SOURCE / 'MIMIC-3/data/features_model.csv', 'labels': SOURCE / 'MIMIC-3/data/mace_labels.csv', 'cohort': SOURCE / 'MIMIC-3/data/cohort_admissions.csv', 'labs': SOURCE / 'MIMIC-3/data/features_raw.csv', 'item_map': SOURCE / 'MIMIC-3/data/lab_item_map.csv', 'raw_admissions': RAW / 'ADMISSIONS.csv.gz', 'raw_patients': RAW / 'PATIENTS.csv.gz', 'raw_diagnoses': RAW / 'DIAGNOSES_ICD.csv.gz', 'v13': _release_source_path(SOURCE / 'Manuscript/v13/compute_v13_metrics.py'), 'source_cohort_code': _release_source_path(SOURCE / 'MIMIC-3/01_identify_patients.py'), 'source_feature_code': _release_source_path(SOURCE / 'MIMIC-3/02_extract_features.py'), 'source_label_code': _release_source_path(SOURCE / 'MIMIC-3/03_define_mace.py'), 'paper': SOURCE / 'Manuscript/v13/paper_predictions_9946.csv', 'old_matrix': OLD / 'private/prepared/MIMIC.npz', 'old_preparation': OLD / 'evidence/mimic_preparation.json', 'old_prepare_code': _release_source_path(OLD / 'prepare_mimic.py'), 'lineage_audit': LINEAGE / 'lineage_audit.json', 'lineage_code': _release_source_path(LINEAGE / 'audit_lineage.py'), 'corrected_history': LINEAGE / 'independent_verification_and_validity_correction.json', 'corrected_history_code': _release_source_path(LINEAGE / 'verify_demographics_and_eligibility.py'), 'design': HERE.parent / 'design/design_approval.json'}
    before = {name: identity(path) for name, path in paths.items()}
    code_before = identity(Path(__file__))
    old_report = _release_resolve_config(json.loads(paths['old_preparation'].read_text(encoding='utf-8')))
    expected_old = {Path(item['path']).resolve(): item for item in old_report['sources'].values()}
    for path in paths.values():
        expected = expected_old.get(path.resolve())
        if expected:
            actual = identity(path)
    corrected = _release_resolve_config(json.loads(paths['corrected_history'].read_text(encoding='utf-8')))
    for item in corrected['input_identities']:
        if Path(item['path']).resolve() in {path.resolve() for path in paths.values()}:
            actual = identity(Path(item['path']))
    historical_names = literal(paths['v13'], 'FEATS_43_EN')
    names = feature_allowlist(historical_names)
    registry = source_mace_registry(paths['source_label_code'])
    features, labels, cohort = (pd.read_csv(paths[key]) for key in ('features', 'labels', 'cohort'))
    paper = pd.read_csv(paths['paper'])
    admissions = pd.read_csv(paths['raw_admissions'], usecols=['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG'])
    patients = pd.read_csv(paths['raw_patients'], usecols=['SUBJECT_ID', 'GENDER', 'DOB'])
    diag = pd.read_csv(paths['raw_diagnoses'], usecols=['HADM_ID', 'SUBJECT_ID', 'ICD9_CODE'], dtype={'ICD9_CODE': str})
    for name, f in (('features', features), ('labels', labels), ('cohort', cohort), ('paper', paper), ('raw_admissions', admissions)):
        assert f.HADM_ID.notna().all() and f.HADM_ID.is_unique and f.SUBJECT_ID.notna().all(), 'Invalid source keys: ' + name
    assert patients.SUBJECT_ID.is_unique and patients.SUBJECT_ID.notna().all()
    assert len(features) == len(labels) == len(cohort) == 11622
    assert set(features.HADM_ID) == set(labels.HADM_ID) == set(cohort.HADM_ID)
    for name in ('ADMITTIME', 'DISCHTIME'):
        admissions[name] = pd.to_datetime(admissions[name], errors='raise')
    patients['DOB'] = pd.to_datetime(patients.DOB, errors='coerce')
    frame = features[['HADM_ID', 'SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG']].copy()
    for other in (labels, cohort, admissions):
        aligned = other.set_index('HADM_ID').loc[frame.HADM_ID]
        assert np.array_equal(frame.SUBJECT_ID, aligned.SUBJECT_ID)
        assert np.array_equal(frame.HOSPITAL_EXPIRE_FLAG, aligned.HOSPITAL_EXPIRE_FLAG)
    raw_aligned = admissions.set_index('HADM_ID').loc[frame.HADM_ID]
    frame['ADMITTIME'] = raw_aligned.ADMITTIME.to_numpy()
    frame['DISCHTIME'] = raw_aligned.DISCHTIME.to_numpy()
    for clock in ('ADMITTIME', 'DISCHTIME'):
        assert np.array_equal(frame[clock].to_numpy(), pd.to_datetime(features[clock]).to_numpy(), equal_nan=True)
    label_aligned = labels.set_index('HADM_ID').loc[frame.HADM_ID]
    frame['MACE'] = label_aligned.MACE.to_numpy(dtype=np.int64)
    assert set(frame.MACE.unique()) == {0, 1} and set(frame.HOSPITAL_EXPIRE_FLAG.unique()) == {0, 1}
    qualification_codes = diag.ICD9_CODE.fillna('').str.replace('.', '', regex=False).str.upper().str.strip()
    label_codes = diag.ICD9_CODE.fillna('').str.replace('.', '', regex=False).str.upper()
    cancer_rows = qualification_codes.str.startswith(tuple((str(i) for i in range(140, 209))) + ('V10',))
    cardiac_rows = qualification_codes.str.startswith(tuple((str(i) for i in range(390, 460))) + ('V45', 'V53'))
    eligible_subjects = set(diag.loc[cancer_rows, 'SUBJECT_ID']) & set(diag.loc[cardiac_rows, 'SUBJECT_ID'])
    assert set(admissions.loc[admissions.SUBJECT_ID.isin(eligible_subjects), 'HADM_ID']) == set(frame.HADM_ID)
    cancer_ids, cardiac_ids = (set(diag.loc[cancer_rows, 'HADM_ID']), set(diag.loc[cardiac_rows, 'HADM_ID']))
    frame['same_admission_cancer'] = frame.HADM_ID.isin(cancer_ids)
    frame['same_admission_cardiac'] = frame.HADM_ID.isin(cardiac_ids)
    assert np.array_equal(frame.same_admission_cancer.astype(int), label_aligned.Admission_has_cancer)
    assert np.array_equal(frame.same_admission_cardiac.astype(int), label_aligned.Admission_has_cardiac)
    frame['valid_current_admission'] = valid_stays(frame)
    frame['prior_completed_cancer'] = prior_flag(frame, admissions, cancer_ids)
    frame['prior_completed_cardiac'] = prior_flag(frame, admissions, cardiac_ids)
    frame['prior_completed_valid'] = frame.prior_completed_cancer & frame.prior_completed_cardiac
    frame['survivor'] = frame.HOSPITAL_EXPIRE_FLAG.eq(0)
    for name, prefixes in registry.items():
        ids = set(diag.loc[label_codes.str.startswith(tuple(prefixes)), 'HADM_ID'])
        frame['component_' + name] = frame.HADM_ID.isin(ids)
    component_cols = ['component_' + name for name in registry]
    frame['coded_component_union'] = frame[component_cols].any(axis=1)
    frame['death_with_broad_cardiac_proxy'] = frame.HOSPITAL_EXPIRE_FLAG.eq(1) & frame.same_admission_cardiac
    assert np.array_equal(frame.MACE.to_numpy(), (frame.coded_component_union | frame.death_with_broad_cardiac_proxy).astype(int).to_numpy())
    token_sets = [set(value.split('|')) - {'None', ''} for value in label_aligned.MACE_Types.fillna('None')]
    for name in registry:
        assert np.array_equal(frame['component_' + name].to_numpy(), np.array([name in tokens for tokens in token_sets]))
    assert np.array_equal(frame.death_with_broad_cardiac_proxy.to_numpy(), np.array(['Cardiac_Death' in tokens for tokens in token_sets]))
    raw_demo = frame[['SUBJECT_ID', 'ADMITTIME']].merge(patients, on='SUBJECT_ID', how='left', validate='many_to_one', sort=False)
    age = (raw_demo.ADMITTIME.dt.year - raw_demo.DOB.dt.year).clip(upper=91).to_numpy(dtype=float)
    sex = raw_demo.GENDER.map({'M': 1.0, 'F': 0.0}).to_numpy(dtype=float)
    assert np.array_equal(age, features.Age.to_numpy(dtype=float), equal_nan=True)
    assert np.array_equal(sex, features.Sex_M.to_numpy(dtype=float), equal_nan=True)
    laboratory = pd.read_csv(paths['labs'], usecols=['HADM_ID', 'SUBJECT_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'Feature'])
    assert laboratory.HADM_ID.isin(frame.HADM_ID).all() and laboratory.VALUENUM.notna().all()
    assert not np.isinf(laboratory.VALUENUM.to_numpy(dtype=float)).any()
    subject_lookup = frame.set_index('HADM_ID').SUBJECT_ID
    assert np.array_equal(laboratory.HADM_ID.map(subject_lookup), laboratory.SUBJECT_ID)
    item_map = pd.read_csv(paths['item_map'])
    assert item_map.ITEMID.is_unique
    assert np.array_equal(laboratory.ITEMID.map(item_map.set_index('ITEMID').Feature), laboratory.Feature)
    rebuilt = medians(laboratory, frame.HADM_ID)
    assert len(rebuilt.columns) == 24
    comparisons = []
    for column in rebuilt:
        cached = pd.to_numeric(features[column], errors='coerce').to_numpy(dtype=float)
        values = rebuilt[column].to_numpy(dtype=float)
        assert np.array_equal(np.isnan(cached), np.isnan(values))
        assert np.allclose(cached, values, rtol=0, atol=1e-10, equal_nan=True)
        finite = np.isfinite(cached) & np.isfinite(values)
        comparisons.append({'feature': column, 'nonmissing': int(finite.sum()), 'all_cells_equal_atol_1e_10': True, 'max_absolute_difference': float(np.abs(cached[finite] - values[finite]).max()) if finite.any() else 0.0})
    reconstructed = pd.DataFrame(np.nan, index=np.arange(len(frame)), columns=historical_names, dtype=float)
    reconstructed['Age'], reconstructed['Sex_M'] = (age, sex)
    for column in rebuilt:
        assert column in historical_names
        reconstructed[column] = rebuilt[column].to_numpy(dtype=float)
    cached43 = features[historical_names].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    reconstructed43 = reconstructed.to_numpy(dtype=float)
    assert np.array_equal(np.isnan(cached43), np.isnan(reconstructed43))
    assert np.allclose(cached43, reconstructed43, atol=1e-10, rtol=0, equal_nan=True)
    x43, x = (reconstructed43.astype(np.float32), reconstructed[names].to_numpy(dtype=np.float32))
    assert not np.isinf(x43).any() and (not np.isinf(x).any())
    groups, group_subjects = pd.factorize(frame.SUBJECT_ID, sort=True)
    groups = groups.astype(np.int64)
    survivor = frame.survivor.to_numpy(dtype=bool)
    assert np.array_equal(frame.loc[survivor, 'HADM_ID'].to_numpy(), paper.HADM_ID.to_numpy())
    assert np.array_equal(frame.loc[survivor, 'SUBJECT_ID'].to_numpy(), paper.SUBJECT_ID.to_numpy())
    assert np.array_equal(frame.loc[survivor, 'MACE'].to_numpy(), paper.MACE.to_numpy())
    with np.load(paths['old_matrix'], allow_pickle=False) as old:
        assert old['feature_names'].tolist() == historical_names
        assert np.array_equal(old['X'], x43[survivor], equal_nan=True)
        assert np.array_equal(old['y'], frame.loc[survivor, 'MACE'].to_numpy())
        survivor_groups, _ = pd.factorize(frame.loc[survivor, 'SUBJECT_ID'], sort=True)
        assert np.array_equal(old['groups'], survivor_groups)
        assert np.array_equal(old['p_original'], paper.oof_prob_paper.to_numpy(dtype=float))
    masks = arm_masks(survivor, frame.prior_completed_valid.to_numpy())
    assert int(masks['prior_survivor'].sum()) == corrected['prior_discharge_count_correction']['both_prior_completed_valid_stays']['admissions'] == 2194
    assert int(frame.loc[masks['prior_survivor'], 'MACE'].sum()) == 1062
    assert len(group_subjects) == 7887 and int(frame.MACE.sum()) == 6105 and (int(survivor.sum()) == 9946)
    raw_valid = valid_stays(admissions)
    assert int((~raw_valid).sum()) == corrected['prior_discharge_count_correction']['raw_invalid_stays']['admissions'] == 98
    assert int((~frame.loc[survivor, 'valid_current_admission']).sum()) == 4
    frame['paper_row_index'] = -1
    frame.loc[survivor, 'paper_row_index'] = np.arange(int(survivor.sum()))
    bounded = laboratory.merge(frame[['HADM_ID', 'ADMITTIME', 'DISCHTIME']], on='HADM_ID', how='left', validate='many_to_one')
    bounded['CHARTTIME'] = pd.to_datetime(bounded.CHARTTIME, errors='coerce')
    full_timing = {'unknown_charttime_rows': int(bounded.CHARTTIME.isna().sum()), 'before_admission_rows': int(bounded.CHARTTIME.lt(bounded.ADMITTIME).sum()), 'after_discharge_rows': int(bounded.CHARTTIME.gt(bounded.DISCHTIME).sum()), 'after_admission_plus_24h_rows': int(bounded.CHARTTIME.gt(bounded.ADMITTIME + pd.Timedelta(hours=24)).sum())}
    payload = {'X': x, 'y': frame.MACE.to_numpy(dtype=np.int64), 'groups': groups, 'keys': frame.HADM_ID.to_numpy(dtype=np.int64), 'feature_names': np.asarray(names, dtype=str)}
    flags = ['survivor', 'prior_completed_valid', 'valid_current_admission', 'prior_completed_cancer', 'prior_completed_cardiac', 'same_admission_cancer', 'same_admission_cardiac', 'coded_component_union', 'death_with_broad_cardiac_proxy'] + component_cols
    payload.update({name: frame[name].to_numpy(dtype=bool) for name in flags})
    payload.update({name: mask for name, mask in masks.items()})
    outputs = []
    (HERE / 'private').mkdir()
    for filename, values in (('full_matrix.npz', payload), ('historical43_audit.npz', {'X_historical43_audit_only': x43, 'feature_names': np.asarray(historical_names, dtype=str), 'keys': payload['keys'], 'y': payload['y'], 'groups': groups})):
        path = HERE / 'private' / filename
        with path.open('xb') as stream:
            np.savez_compressed(stream, **values)
        with np.load(path, allow_pickle=False) as replay:
            assert set(replay.files) == set(values)
            for name, value in values.items():
                assert replay[name].dtype == value.dtype
                assert np.array_equal(replay[name], value, equal_nan=True) if value.dtype.kind == 'f' else np.array_equal(replay[name], value)
        outputs.append(identity(path))
    for name, mask in masks.items():
        frame[name] = mask
    metadata_path = HERE / 'private/row_metadata.csv.gz'
    frame.assign(group=groups).to_csv(metadata_path, index=False, compression={'method': 'gzip', 'mtime': 0})
    outputs.append(identity(metadata_path))
    counts = {name: summarize(frame, mask) for name, mask in masks.items()}
    counts['death_excluded_from_historical_paper'] = summarize(frame, ~survivor)
    per_feature = [{'name': name, 'nonmissing_full': int(np.isfinite(x[:, index]).sum()), 'nonmissing_by_arm': {arm: int(np.isfinite(x[mask, index]).sum()) for arm, mask in masks.items()}} for index, name in enumerate(names)]
    contract = {'version': 1, 'row_unit': 'source MIMIC-III admission, not patient', 'row_order': 'features_model.csv original complete source row order; survivor slice matches paper_predictions_9946.csv exactly', 'label': {'primary': 'Unchanged cached MACE admission composite; independently reconstructed from original six source ICD-prefix groups plus death-with-broad-cardiac-diagnosis proxy', 'registry': registry, 'cardiac_death_proxy': 'HOSPITAL_EXPIRE_FLAG==1 AND same-admission broad cardiac ICD family; not verified cardiovascular cause of death', 'coded_component_union': 'Source six ICD groups excluding death proxy; separate audit label/flag, not substituted for primary y', 'same_admission_diagnosis_not_onset': True}, 'predictors': {'field': 'X', 'nominal_count': 41, 'feature_names': names, 'uniformly_removed': list(DROP), 'exclusion_reason': 'Known cached AST includes Gastrin/Blasts and HDL includes total/HDL ratio. Drop whole columns uniformly rather than guess a new mapping.', 'historical43_audit_only_file': 'private/historical43_audit.npz', 'aggregation': 'Unrestricted cached HADM_ID/Feature VALUENUM median, plus median Total Protein minus median Albumin for Globulin', 'age': 'raw admission year minus raw birth year clipped above at 91', 'sex': 'raw PATIENTS GENDER M=1 F=0 (unmapped remains NaN)', 'imputation_scaling_feature_selection_fitted': False, 'all_missing_columns_retained': True, 'source_units_unavailable_in_cache': True, 'numeric_unit_conversion_or_new_item_mapping': False, 'known_remaining_mapping_caveat': 'Creatine Kinase still includes ITEMID50911 CK-MB; remaining broad mappings are not clinically certified. No event-time/availability exclusion is established.'}, 'eligibility': {'source_universe': 'All admissions of patients ever satisfying cancer140-208/V10 AND cardiac390-459/V45/V53 in available source diagnoses', 'prior_source_validity': 'nonmissing source ADMITTIME and DISCHTIME, DISCHTIME>=ADMITTIME', 'prior_cutoff': 'both qualifying-family histories have a valid completed admission with DISCHTIME strictly < index ADMITTIME; families may be from different earlier stays', 'index_validity': 'valid_current_admission is report-only, NOT another arm filter', 'pre_fit_clarification': 'Root withdrew an overbroad current-discharge-validity-only restriction before any preparation/model execution; preserve source index rows equally across arms.', 'arms': {'ever_all': 'all11622', 'ever_survivor': 'survivor', 'prior_all': 'prior_completed_valid', 'prior_survivor': 'prior_completed_valid AND survivor'}, 'prior_documentation_not_first_event_or_event_free': True}, 'groups': 'globally sorted SUBJECT_ID factorization across11622; compare survivor old codes by independent survivor refactorization, not numerical equality to full-universe codes', 'private_npz_fields': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in payload.items()}, 'private_identifiers': 'keys and row_metadata contain original admission IDs; metadata also contains patient IDs and clocks, strictly local only', 'model_training_authorized_in_preparation': False}
    for filename, data in (('preparation_contract.json', contract), ('synthetic_tests.json', tests)):
        path = HERE / filename
        with path.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=True, indent=2, allow_nan=False)
        outputs.append(identity(path))
    assert before == {name: identity(path) for name, path in paths.items()}
    assert code_before == identity(Path(__file__))
    report = {'passed': True, 'execution_complete': True, 'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - timer, 'script': code_before, 'inputs': before, 'outputs': outputs, 'synthetic_tests': tests, 'cohorts': counts, 'matrix': {'shape': list(x.shape), 'historical43_shape': list(x43.shape), 'all_missing_features': [name for index, name in enumerate(names) if np.isnan(x[:, index]).all()], 'per_feature': per_feature, 'all11622_cached_43_values_and_masks_rebuilt': True, 'all9946_43_float32_cells_identical_to_frozen_preparation': True, 'survivor_labels_keys_groups_and_probabilities_match': True, 'demographic_cells_raw_rebuilt_and_equal': int(len(frame) * 2)}, 'laboratory_replay': {'cached_rows': len(laboratory), 'admissions_with_any_cached_lab': int(laboratory.HADM_ID.nunique()), 'source_admissions_without_any_cached_lab_retained': int((~frame.HADM_ID.isin(laboratory.HADM_ID)).sum()), 'median_columns_compared': comparisons, 'recorded_time_scope': full_timing, 'new_raw_LABEVENTS_extraction_performed': False, 'clinical_item_or_unit_equivalence_certified': False}, 'raw_invalid_stays': int((~raw_valid).sum()), 'all_source_identities_unchanged': True, 'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas')}}, 'models_trained': False, 'historical_modules_imported_or_run': False, 'limits': ['Selection sensitivity within the original ever-cancer-and-ever-cardiac source universe, not all MIMIC oncology admissions.', 'Unrestricted whole-admission lab medians and concurrent diagnosis labels retain potential event/treatment physiology leakage; no incident/landmark inference.', 'Death-with-broad-cardiac diagnosis is a source proxy, not cause-specific cardiovascular death; the primary source label is not repaired.', 'Unitless cached values and remaining keyword mappings are preserved, not certified or newly harmonized.', 'Prior completed valid coded stays show earlier documentation, not event-free status or complete lifetime history.']}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2, allow_nan=False)
    print(json.dumps({'preparation_complete': True, 'matrix_shape': list(x.shape), 'cohorts': counts, 'models_trained': False, 'private_rows_or_identifiers_logged': False}, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test', action='store_true')
    modes.add_argument('--run', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(test_all())) if args.self_test else main()
    except Exception as error:
        import traceback
        lines = [{'file': Path(frame.filename).name, 'line': frame.lineno} for frame in traceback.extract_tb(error.__traceback__)]
        print(json.dumps({'preparation_failed': True, 'error_type': type(error).__name__, 'trace_locations_only': lines, 'private_exception_values_suppressed': True}), file=sys.stderr)
        raise SystemExit(1)
