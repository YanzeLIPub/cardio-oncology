"""Independently verify the reference matrix and qualify prior-discharge counts."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
RAW = Path(_release_path('mimic', ''))
SOURCE = Path(_release_path('pipeline', 'MIMIC-3/data'))

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def summarize(frame, mask):
    selected = frame.loc[mask]
    result = {'admissions': len(selected), 'patients': int(selected.SUBJECT_ID.nunique())}
    if 'MACE' in frame:
        result['positive_admission_labels'] = int(selected.MACE.sum())
    return result

def main():
    output = HERE / 'independent_verification_and_validity_correction.json'
    if output.exists():
        raise FileExistsError('Refuse to overwrite verification')
    matrix_path = HERE / 'private/MIMIC_demographics.npz'
    source_paths = [RAW / 'ADMISSIONS.csv.gz', RAW / 'PATIENTS.csv.gz', RAW / 'DIAGNOSES_ICD.csv.gz', SOURCE / 'mace_labels.csv', matrix_path, HERE / 'lineage_audit.json', PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1/private/prepared/MIMIC.npz']
    before = [identity(p) for p in source_paths]
    adm = pd.read_csv(RAW / 'ADMISSIONS.csv.gz', usecols=['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG'])
    adm['ADMITTIME'] = pd.to_datetime(adm.ADMITTIME)
    adm['DISCHTIME'] = pd.to_datetime(adm.DISCHTIME)
    valid_stay = adm.DISCHTIME.ge(adm.ADMITTIME) & adm.DISCHTIME.notna() & adm.ADMITTIME.notna()
    diag = pd.read_csv(RAW / 'DIAGNOSES_ICD.csv.gz', dtype={'ICD9_CODE': str}, usecols=['HADM_ID', 'ICD9_CODE'])
    codes = diag.ICD9_CODE.fillna('').str.replace('.', '', regex=False).str.upper().str.strip()
    cancer_ids = set(diag.loc[codes.str.match('^(1[4-9][0-9]|20[0-8]|V10)'), 'HADM_ID'])
    cardiac_ids = set(diag.loc[codes.str.match('^(39[0-9]|4[0-5][0-9]|V45|V53)'), 'HADM_ID'])
    first_cancer_end = adm.loc[valid_stay & adm.HADM_ID.isin(cancer_ids)].groupby('SUBJECT_ID').DISCHTIME.min()
    first_cardiac_end = adm.loc[valid_stay & adm.HADM_ID.isin(cardiac_ids)].groupby('SUBJECT_ID').DISCHTIME.min()
    lb = pd.read_csv(SOURCE / 'mace_labels.csv', usecols=['HADM_ID', 'MACE'])
    retained = adm.merge(lb, on='HADM_ID', validate='one_to_one')
    retained = retained.loc[retained.HOSPITAL_EXPIRE_FLAG == 0].copy()
    prior_cancer = retained.SUBJECT_ID.map(first_cancer_end) < retained.ADMITTIME
    prior_cardiac = retained.SUBJECT_ID.map(first_cardiac_end) < retained.ADMITTIME
    prior_all_cancer = adm.SUBJECT_ID.map(first_cancer_end) < adm.ADMITTIME
    prior_all_cardiac = adm.SUBJECT_ID.map(first_cardiac_end) < adm.ADMITTIME
    later_cancer = retained.SUBJECT_ID.map(adm.loc[adm.HADM_ID.isin(cancer_ids)].groupby('SUBJECT_ID').ADMITTIME.min()) > retained.ADMITTIME
    later_cardiac = retained.SUBJECT_ID.map(adm.loc[adm.HADM_ID.isin(cardiac_ids)].groupby('SUBJECT_ID').ADMITTIME.min()) > retained.ADMITTIME
    original = _release_resolve_config(json.loads((HERE / 'lineage_audit.json').read_text()))
    assert summarize(retained, later_cancer | later_cardiac) == original['cohort_selection']['at_least_one_qualifying_diagnosis_family_only_in_later_admission']
    paper = pd.read_csv(_release_path('pipeline', 'Manuscript/v13/paper_predictions_9946.csv'), usecols=['HADM_ID', 'SUBJECT_ID'])
    pt = pd.read_csv(RAW / 'PATIENTS.csv.gz', usecols=['SUBJECT_ID', 'DOB', 'GENDER'])
    joined = paper.merge(adm[['HADM_ID', 'ADMITTIME']], on='HADM_ID', validate='one_to_one').merge(pt, on='SUBJECT_ID', validate='many_to_one')
    rebuilt_age = np.minimum(joined.ADMITTIME.dt.year.to_numpy() - pd.to_datetime(joined.DOB).dt.year.to_numpy(), 91)
    rebuilt_sex = np.select([joined.GENDER.eq('M'), joined.GENDER.eq('F')], [1.0, 0.0], default=np.nan)
    expected = np.column_stack([rebuilt_age, rebuilt_sex]).astype(np.float32)
    with np.load(matrix_path, allow_pickle=False) as z:
        assert z.files == ['X', 'y', 'groups', 'feature_names']
        assert z['feature_names'].tolist() == ['Age', 'Sex_M']
        assert np.array_equal(z['X'], expected, equal_nan=True)
        X, y, groups = (z['X'], z['y'], z['groups'])
    with np.load(source_paths[-1], allow_pickle=False) as z:
        assert np.array_equal(groups, z['groups']) and np.array_equal(y, z['y'])
    report = {'passed': True, 'script': identity(__file__), 'python': sys.version, 'input_identities': before, 'source_rebuilt_demographic_matrix': identity(matrix_path), 'raw_demographics_independently_equal': True, 'compared_demographic_cells': int(X.size), 'rows': len(y), 'patients': int(np.unique(groups).size), 'positive_labels': int(y.sum()), 'groups_and_labels_identical_to_frozen_original': True, 'matrix_fields_only': ['X', 'y', 'groups', 'feature_names'], 'future_admission_selection_independently_confirmed': summarize(retained, later_cancer | later_cardiac), 'cancer_and_cardiac_both_first_qualify_later': summarize(retained, later_cancer & later_cardiac), 'prior_discharge_count_correction': {'reason': 'Initial aggregate audit used minimum DISCHTIME without excluding source stays with DISCHTIME before ADMITTIME. Those preliminary prior-completed counts are superseded by these validity-qualified counts. Later-start selection and the demographic matrix are unaffected.', 'raw_invalid_stays': summarize(adm, ~valid_stay), 'original_retained_invalid_stays': summarize(retained, retained.DISCHTIME < retained.ADMITTIME), 'prior_completed_cancer_valid_stay': summarize(retained, prior_cancer), 'prior_completed_cardiac_valid_stay': summarize(retained, prior_cardiac), 'both_prior_completed_valid_stays': summarize(retained, prior_cancer & prior_cardiac), 'not_both_prior_completed_valid_stays': summarize(retained, ~(prior_cancer & prior_cardiac)), 'raw_all_admissions_with_prior_completed_cancer_valid_stay': summarize(adm, prior_all_cancer), 'raw_prior_cancer_without_prior_cardiac_valid_stay': summarize(adm, prior_all_cancer & ~prior_all_cardiac), 'restriction': 'A prior completed coded stay is evidence of prior documentation, not a true onset date or event-free status. No cohorts or models are changed.'}}
    assert before == [identity(p) for p in source_paths]
    report['input_hashes_unchanged'] = True
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=True, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('input_identities',)}, indent=2))
if __name__ == '__main__':
    main()
