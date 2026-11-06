"""Read-only MIMIC lineage and selection audit with aggregate-only outputs."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE = Path(_release_path('pipeline', ''))
RAW = SOURCE / 'mimic-iii-clinical-database-1.4'
MIMIC = SOURCE / 'MIMIC-3'
VALIDATION = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
WINDOWS = PROJECT / 'revision_2026-10-15/landmark_completion_v1/mimic'

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def literal(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    matches = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == name for t in n.targets))]
    assert len(matches) == 1
    return matches[0]

def counts(frame, mask=None):
    selected = frame if mask is None else frame.loc[mask]
    return {'admissions': len(selected), 'patients': int(selected.SUBJECT_ID.nunique()), 'positive_admission_labels': int(selected.MACE.sum())}

def timing_categories(frame, qualifying):
    first_start = qualifying.groupby('SUBJECT_ID').ADMITTIME.min()
    first_complete = qualifying.groupby('SUBJECT_ID').DISCHTIME.min()
    start = frame.SUBJECT_ID.map(first_start)
    complete = frame.SUBJECT_ID.map(first_complete)
    return ({'qualifying_admission_starts_later': counts(frame, start > frame.ADMITTIME), 'earliest_qualifying_admission_starts_at_index': counts(frame, start == frame.ADMITTIME), 'earlier_qualifying_admission_started': counts(frame, start < frame.ADMITTIME), 'no_prior_completed_qualifying_admission': counts(frame, complete.isna() | (complete >= frame.ADMITTIME)), 'prior_completed_qualifying_admission': counts(frame, complete < frame.ADMITTIME)}, start, complete)

def main():
    output = HERE / 'lineage_audit.json'
    prepared_output = HERE / 'private/MIMIC_demographics.npz'
    if output.exists() or prepared_output.exists():
        raise FileExistsError('Refuse to overwrite existing audit output')
    paths = {'raw_patients': RAW / 'PATIENTS.csv.gz', 'raw_admissions': RAW / 'ADMISSIONS.csv.gz', 'raw_diagnoses': RAW / 'DIAGNOSES_ICD.csv.gz', 'cohort': MIMIC / 'data/cohort_admissions.csv', 'features': MIMIC / 'data/features_model.csv', 'labels': MIMIC / 'data/mace_labels.csv', 'item_mapping': MIMIC / 'data/lab_item_map.csv', 'lab_cache': MIMIC / 'data/features_raw.csv', 'paper_predictions': SOURCE / 'Manuscript/v13/paper_predictions_9946.csv', 'v13_entry': _release_source_path(SOURCE / 'Manuscript/v13/compute_v13_metrics.py'), 'old_08_entry': _release_source_path(MIMIC / '08_paper_methods_mimic.py'), 'cohort_entry': _release_source_path(MIMIC / '01_identify_patients.py'), 'features_entry': _release_source_path(MIMIC / '02_extract_features.py'), 'label_entry': _release_source_path(MIMIC / '03_define_mace.py'), 'prepared_matrix': VALIDATION / 'private/prepared/MIMIC.npz', 'patient_roles': VALIDATION / 'private/runs/MIMIC_original/patient_roles.npz', 'new_model_entry': _release_source_path(VALIDATION / 'run_validation.py'), 'window_entry': _release_source_path(WINDOWS / 'prepare_windows.py'), 'window_0h': WINDOWS / 'private/prepared/MIMIC_0h.npz', 'window_24h': WINDOWS / 'private/prepared/MIMIC_24h.npz', 'window_48h': WINDOWS / 'private/prepared/MIMIC_48h.npz'}
    before = {k: identity(v) for k, v in paths.items()}
    admissions = pd.read_csv(paths['raw_admissions'], usecols=['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG'])
    for col in ('ADMITTIME', 'DISCHTIME'):
        admissions[col] = pd.to_datetime(admissions[col], errors='raise')
    patients = pd.read_csv(paths['raw_patients'], usecols=['SUBJECT_ID', 'DOB', 'GENDER'])
    patients['DOB'] = pd.to_datetime(patients.DOB, errors='coerce')
    diag = pd.read_csv(paths['raw_diagnoses'], usecols=['HADM_ID', 'SUBJECT_ID', 'ICD9_CODE'], dtype={'ICD9_CODE': str})
    code = diag.ICD9_CODE.fillna('').str.replace('.', '', regex=False).str.upper().str.strip()
    cancer = code.str.startswith(tuple((str(i).zfill(3) for i in range(140, 209))) + ('V10',))
    cardiac = code.str.startswith(tuple((str(i) for i in range(390, 460))) + ('V45', 'V53'))
    cancer_subjects = set(diag.loc[cancer, 'SUBJECT_ID'])
    cardiac_subjects = set(diag.loc[cardiac, 'SUBJECT_ID'])
    eligible_subjects = cancer_subjects & cardiac_subjects
    cohort = pd.read_csv(paths['cohort'])
    assert set(cohort.HADM_ID) == set(admissions.loc[admissions.SUBJECT_ID.isin(eligible_subjects), 'HADM_ID'])
    labels = pd.read_csv(paths['labels'])
    linked = admissions.merge(labels, on=['HADM_ID', 'SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG'], how='inner', validate='one_to_one')
    assert len(linked) == 11622
    retained = linked.loc[linked.HOSPITAL_EXPIRE_FLAG == 0].copy()
    assert counts(retained) == {'admissions': 9946, 'patients': 6783, 'positive_admission_labels': 4462}
    paper = pd.read_csv(paths['paper_predictions'])
    retained = retained.set_index('HADM_ID').loc[paper.HADM_ID].reset_index()
    assert np.array_equal(retained.MACE, paper.MACE)
    assert np.array_equal(retained.SUBJECT_ID, paper.SUBJECT_ID)
    cancer_adm = admissions.loc[admissions.HADM_ID.isin(diag.loc[cancer, 'HADM_ID'])]
    cardiac_adm = admissions.loc[admissions.HADM_ID.isin(diag.loc[cardiac, 'HADM_ID'])]
    cancer_timing, cancer_start, cancer_complete = timing_categories(retained, cancer_adm)
    cardiac_timing, cardiac_start, cardiac_complete = timing_categories(retained, cardiac_adm)
    eligible_by_prior_discharge = (cancer_complete < retained.ADMITTIME) & (cardiac_complete < retained.ADMITTIME)
    same_cancer = retained.HADM_ID.isin(cancer_adm.HADM_ID)
    same_cardiac = retained.HADM_ID.isin(cardiac_adm.HADM_ID)
    assert np.array_equal(same_cancer.astype(int), retained.Admission_has_cancer)
    assert np.array_equal(same_cardiac.astype(int), retained.Admission_has_cardiac)
    selection = {'all_source_cohort': counts(linked), 'survivors': counts(retained), 'excluded_death_admissions': counts(linked, linked.HOSPITAL_EXPIRE_FLAG == 1), 'raw_replayed_cohort_identical': True, 'cancer': cancer_timing, 'cardiac': cardiac_timing, 'at_least_one_qualifying_diagnosis_family_only_in_later_admission': counts(retained, (cancer_start > retained.ADMITTIME) | (cardiac_start > retained.ADMITTIME)), 'both_qualifying_families_in_prior_completed_admissions': counts(retained, eligible_by_prior_discharge), 'not_both_families_in_prior_completed_admissions': counts(retained, ~eligible_by_prior_discharge), 'same_admission_cancer_and_cardiac': counts(retained, same_cancer & same_cardiac), 'same_admission_cancer_absent': counts(retained, ~same_cancer), 'same_admission_cardiac_absent': counts(retained, ~same_cardiac), 'interpretation': 'Admission start/discharge chronology only. Concurrent discharge codes have no verified onset. Eligibility using both ever-diagnosis families and excluding eventual deaths depends on outcome-period records. Prior-completed diagnoses document a record before admission but do not establish event-free status.'}
    cross = []
    for death in (0, 1):
        for flag in (0, 1):
            for outcome in (0, 1):
                mask = (linked.HOSPITAL_EXPIRE_FLAG == death) & (linked.Admission_has_cardiac == flag) & (linked.MACE == outcome)
                cross.append({'hospital_expire_flag': death, 'admission_has_cardiac': flag, 'MACE': outcome, **counts(linked, mask)})
    all_admissions_with_prior_cancer = admissions.SUBJECT_ID.map(cancer_adm.groupby('SUBJECT_ID').DISCHTIME.min()) < admissions.ADMITTIME
    all_admissions_with_prior_cardiac = admissions.SUBJECT_ID.map(cardiac_adm.groupby('SUBJECT_ID').DISCHTIME.min()) < admissions.ADMITTIME
    independent = admissions.loc[all_admissions_with_prior_cancer].copy()
    selection['feasible_alternative_eligibility_counts_without_modeling'] = {'all_raw_admissions_with_prior_completed_cancer_record': {'admissions': len(independent), 'patients': int(independent.SUBJECT_ID.nunique()), 'deaths': int(independent.HOSPITAL_EXPIRE_FLAG.sum())}, 'prior_cancer_without_prior_completed_cardiac_record': {'admissions': int((all_admissions_with_prior_cancer & ~all_admissions_with_prior_cardiac).sum()), 'patients': int(admissions.loc[all_admissions_with_prior_cancer & ~all_admissions_with_prior_cardiac, 'SUBJECT_ID'].nunique())}, 'requires_new_target_and_label_review': True, 'interpretation': 'No requirement for ever-cardiac disease or survival; these aggregate counts only establish feasibility, not a clinically validated cohort.'}
    demographic = retained[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME']].merge(patients, on='SUBJECT_ID', validate='many_to_one')
    uncapped_age = demographic.ADMITTIME.dt.year - demographic.DOB.dt.year
    age = uncapped_age.clip(upper=91).astype(float).to_numpy()
    sex = demographic.GENDER.map({'M': 1.0, 'F': 0.0}).to_numpy(dtype=float)
    features = pd.read_csv(paths['features']).set_index('HADM_ID').loc[paper.HADM_ID]
    assert np.array_equal(age, features.Age)
    assert np.array_equal(sex, features.Sex_M)
    names = literal(paths['v13_entry'], 'FEATS_43_EN')
    with np.load(paths['prepared_matrix'], allow_pickle=False) as npz:
        X, y, groups = (npz['X'].copy(), npz['y'].copy(), npz['groups'].copy())
        assert names == npz['feature_names'].tolist()
        assert np.array_equal(age, X[:, names.index('Age')])
        assert np.array_equal(sex, X[:, names.index('Sex_M')])
        assert np.array_equal(y, retained.MACE)
    with np.load(paths['patient_roles'], allow_pickle=False) as npz:
        assert np.array_equal(groups, npz['groups']) and np.array_equal(y, npz['y'])
        roles = npz['roles']
    isolation = []
    for fold, role in enumerate(roles, 1):
        group_sets = [set(groups[role == i]) for i in range(4)]
        pairwise = [len(group_sets[i] & group_sets[j]) for i in range(4) for j in range(i + 1, 4)]
        assert max(pairwise) == 0
        isolation.append({'fold': fold, 'pairwise_role_patient_intersections': pairwise, 'test_admissions': int((role == 0).sum())})
    window_checks = []
    for hours in (0, 24, 48):
        with np.load(paths[f'window_{hours}h'], allow_pickle=False) as npz:
            assert names == npz['feature_names'].tolist()
            assert np.array_equal(y, npz['y']) and np.array_equal(groups, npz['groups'])
            assert np.array_equal(age, npz['X'][:, names.index('Age')])
            assert np.array_equal(sex, npz['X'][:, names.index('Sex_M')])
            window_checks.append({'window_hours': hours, 'feature_order_matches': True, 'demographics_match_raw': True, 'labels_and_groups_match': True})
    forbidden = {'HADM_ID', 'SUBJECT_ID', 'Admission_has_cancer', 'Admission_has_cardiac', 'MACE', 'MACE_Types', 'MACE_SeqNum_min', 'HOSPITAL_EXPIRE_FLAG', 'DEATHTIME', 'DISCHTIME', 'DOD', 'LOS', 'Ever_MACE'}
    item_map = pd.read_csv(paths['item_mapping'])
    laboratory = pd.read_csv(paths['lab_cache'], usecols=['HADM_ID', 'ITEMID', 'Feature'])
    lab_included = laboratory.loc[laboratory.HADM_ID.isin(retained.HADM_ID)]
    feature_lineage = []
    for i, name in enumerate(names):
        mappings = item_map.loc[item_map.Feature == name, ['ITEMID', 'LABEL', 'FLUID']].to_dict('records')
        if name in ('Age', 'Sex_M'):
            ancestry = 'Raw PATIENTS and admission-year demographic derivation'
        elif name == 'Globulin':
            ancestry = 'Median Total Protein minus median Albumin'
        elif mappings:
            ancestry = 'All-admission median of mapped LABEVENTS VALUENUM; CHARTTIME unrestricted in original cache'
        else:
            ancestry = 'Unmeasured placeholder column'
        feature_lineage.append({'feature': name, 'source': ancestry, 'observed_admissions': int(np.isfinite(X[:, i]).sum()), 'source_items': mappings, 'direct_label_definition_field': name in forbidden, 'indirect_event_or_treatment_physiology_excluded': name in ('Age', 'Sex_M')})
    ckmb = lab_included.loc[lab_included.ITEMID == 50911]
    demographic_report = {'raw_patients_gender_values': sorted(demographic.GENDER.unique().tolist()), 'missing_dob_or_admittime': int(demographic[['DOB', 'ADMITTIME']].isna().any(axis=1).sum()), 'age_formula': 'ADMISSIONS.ADMITTIME.year minus PATIENTS.DOB.year clipped above at 91', 'sex_formula': 'PATIENTS.GENDER M->1, F->0, missing or other->NaN (all current cohort values M/F)', 'raw_demographics_match_cache_and_prepared_matrix_all_cells': True, 'rows': len(age), 'age_min': float(age.min()), 'age_max': float(age.max()), 'uncapped_age_above_91_admissions': int((uncapped_age > 91).sum()), 'unmapped_or_missing_gender_admissions': int((~demographic.GENDER.isin(['M', 'F'])).sum()), 'diagnoses_labs_death_or_discharge_used_in_demographic_values': False, 'limitation': 'Chronological demographic controls do not remove original cohort selection, concurrent targets, or survivor conditioning.'}
    schemas = {}
    for name in ('DIAGNOSES_ICD', 'LABEVENTS', 'ADMISSIONS', 'CHARTEVENTS'):
        with gzip.open(RAW / f'{name}.csv.gz', 'rt', newline='', encoding='utf-8') as stream:
            schemas[name] = next(csv.reader(stream))
    assert 'CHARTTIME' not in schemas['DIAGNOSES_ICD'] and 'STORETIME' not in schemas['LABEVENTS']
    report = {'scope': 'Independent aggregate audit for R3 direct/indirect leakage; no model training or source modifications', 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn')}}, 'inputs': before, 'script': identity(__file__), 'cohort_selection': selection, 'death_cardiac_flag_outcome_crosstab': cross, 'demographic_lineage': demographic_report, 'current_predictor_inventory': {'nominal_features': len(names), 'all_missing': int((~np.isfinite(X).any(axis=0)).sum()), 'forbidden_fields_present': sorted(set(names) & forbidden), 'features': feature_lineage}, 'old_08_predictors': {'source': '08_paper_methods_mimic.py lines 89-127', 'lab_features': int(laboratory.Feature.nunique()), 'four_statistics': ['mean', 'last', 'max', 'min'], 'demographic_and_diagnosis_columns': ['Age', 'Sex_M', 'Admission_has_cancer', 'Admission_has_cardiac'], 'total_columns': int(laboratory.Feature.nunique()) * 4 + 4, 'cardiac_flag_has_direct_label_definition_overlap': True}, 'CK_MB_mapping': {'ITEMID': 50911, 'assigned_feature': 'Creatine Kinase', 'retained_cached_rows': len(ckmb), 'retained_admissions': int(ckmb.HADM_ID.nunique()), 'interpretation': 'An event-related analyte is mixed with total CK. This is not proof of post-event timing; all acute lab predictors are excluded in the proposed demographic control.'}, 'patient_isolation_rechecked': isolation, 'window_matrix_checks': window_checks, 'raw_table_headers': schemas, 'timing_limit': 'DIAGNOSES_ICD has neither event onset nor present-on-admission field. LABEVENTS has CHARTTIME but no STORETIME. No incident endpoint or result-availability claim can be certified by these fields.', 'recommended_control': 'Source-rebuilt outcome-definition-independent demographic reference with Age and Sex_M only, using exactly the existing frozen patient roles; fit all processing on training partitions; compare paired patient-bootstrap metrics against same-fold raw full model. This removes acute physiology from that reference, not from the original model or eligibility. It is not a negative control or future-event prediction model.'}
    assert all((identity(paths[k]) == v for k, v in before.items()))
    report['input_identities_unchanged_after_read'] = True
    prepared_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prepared_output, X=np.column_stack([age, sex]).astype(np.float32), y=y, groups=groups, feature_names=np.array(['Age', 'Sex_M']))
    report['source_rebuilt_demographic_matrix'] = identity(prepared_output)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': output.as_posix(), 'cohort_selection': selection, 'demographic_lineage': demographic_report, 'CK_MB_mapping': report['CK_MB_mapping']}, indent=2))
if __name__ == '__main__':
    main()
