"""Prepare the original v13 MIMIC matrix for new patient-grouped validation.

No historical script is imported or executed. No label, cohort, measurement,
unit or feature mapping is repaired here. Only the two declared outputs are
written, and their creation is refused if either output already exists.
"""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
HERE = Path(__file__).resolve().parent
EXPECTED_HERE = Path(_release_path('project', 'revision_2026-10-15/original_v5_independent_validation_v1'))
SOURCE_ROOT = Path(_release_path('pipeline', ''))
NPZ_PATH = HERE / 'private/prepared/MIMIC.npz'
REPORT_PATH = HERE / 'evidence/mimic_preparation.json'
SOURCES = {'features': ('MIMIC-3/data/features_model.csv', None), 'labels': ('MIMIC-3/data/mace_labels.csv', None), 'admissions': ('MIMIC-3/data/cohort_admissions.csv', None), 'paper_predictions': ('Manuscript/v13/paper_predictions_9946.csv', None), 'v13_entrypoint': ('Manuscript/v13/compute_v13_metrics.py', None), 'raw_lab_cache': ('MIMIC-3/data/features_raw.csv', None), 'cached_item_map': ('MIMIC-3/data/lab_item_map.csv', None), 'cohort_source_code': ('MIMIC-3/01_identify_patients.py', None), 'feature_source_code': ('MIMIC-3/02_extract_features.py', None), 'label_source_code': ('MIMIC-3/03_define_mace.py', None)}
KNOWN_WRONG_ITEMS = ((50903, 'Cholesterol Ratio (Total/HDL)', 'HDL Cholesterol'), (50928, 'Gastrin', 'AST'), (51148, 'Blasts', 'AST'))
FORBIDDEN_PREDICTORS = {'HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOD', 'EXPIRE_FLAG', 'HOSPITAL_EXPIRE_FLAG', 'Admission_has_cancer', 'Admission_has_cardiac', 'MACE', 'MACE_Types', 'MACE_SeqNum_min', 'Ever_MACE', 'LOS', 'FollowupDays', 'FU_extended', 'risk_group', 'oof_prob_paper', 'oof_prob_age_only'}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def sha256(path: Path) -> str:
    return None

def identity(path: Path) -> dict:
    return {'path': path.as_posix(), 'bytes': path.stat().st_size}

def check_source_identities() -> dict:
    identities = {}
    for name, (relative, expected) in SOURCES.items():
        record = identity(SOURCE_ROOT / relative)
        identities[name] = record
    return identities

def source(name: str) -> Path:
    return SOURCE_ROOT / SOURCES[name][0]

def ordered_features() -> list[str]:
    tree = ast.parse(source('v13_entrypoint').read_text(encoding='utf-8-sig'))
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'FEATS_43_EN' for t in node.targets))]
    require(len(values) == 1, 'Exactly one literal FEATS_43_EN assignment is required')
    features = values[0]
    require(isinstance(features, list), 'Feature constant must be a list')
    require(len(features) == 43 and len(set(features)) == 43, 'Expected 43 distinct nominal features')
    require(all((isinstance(x, str) for x in features)), 'Feature names must be strings')
    require(not set(features) & FORBIDDEN_PREDICTORS, 'Forbidden field in predictor allowlist')
    return features

def audit_cached_laboratories(frame: pd.DataFrame) -> dict:
    raw = pd.read_csv(source('raw_lab_cache'), usecols=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'Feature'])
    subset = raw[raw['HADM_ID'].isin(frame['HADM_ID'])].copy()
    item_map = pd.read_csv(source('cached_item_map'))
    wrong_items = []
    for itemid, expected_label, column in KNOWN_WRONG_ITEMS:
        mapping = item_map[item_map['ITEMID'] == itemid]
        require(len(mapping) == 1, 'Known erroneous item must have exactly one cached mapping')
        require(mapping.iloc[0]['LABEL'] == expected_label, 'Known item label changed')
        require(mapping.iloc[0]['Feature'] == column, 'Known erroneous assignment changed')
        selected = subset[subset['ITEMID'] == itemid]
        full_selected = raw[raw['ITEMID'] == itemid]
        require(set(selected['Feature']) == {column}, 'Raw cached assignment does not match dictionary')
        wrong_items.append({'itemid': itemid, 'dictionary_label': expected_label, 'assigned_predictor': column, 'all_cached_rows': len(full_selected), 'all_cached_admissions': int(full_selected['HADM_ID'].nunique()), 'retained_cohort_rows': len(selected), 'retained_cohort_admissions': int(selected['HADM_ID'].nunique())})
    pivot = subset.groupby(['HADM_ID', 'Feature'])['VALUENUM'].median().unstack('Feature').reindex(frame['HADM_ID'])
    pivot['Globulin'] = pivot['Total Protein'] - pivot['Albumin']
    comparisons = []
    for column in pivot.columns:
        original = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
        replayed = pivot[column].to_numpy(dtype=float)
        mask_equal = np.array_equal(np.isnan(original), np.isnan(replayed))
        value_equal = np.allclose(original, replayed, rtol=0, atol=1e-10, equal_nan=True)
        require(mask_equal and value_equal, f'Raw-cache median replay differs: {column}')
        comparisons.append({'feature': column, 'missing_mask_identical': bool(mask_equal), 'values_equal_at_absolute_tolerance_1e_10': bool(value_equal), 'maximum_absolute_difference': float(np.nanmax(np.abs(original - replayed)))})
    require(len(comparisons) == 24, 'Expected 23 cached laboratory columns plus derived Globulin')
    bounded = subset.merge(frame[['HADM_ID', 'ADMITTIME', 'DISCHTIME']], on='HADM_ID', how='left', validate='many_to_one')
    for column in ('CHARTTIME', 'ADMITTIME', 'DISCHTIME'):
        bounded[column] = pd.to_datetime(bounded[column], errors='coerce')
    timing = {'unknown_charttime_rows': int(bounded['CHARTTIME'].isna().sum()), 'before_recorded_admission_rows': int((bounded['CHARTTIME'] < bounded['ADMITTIME']).sum()), 'after_recorded_discharge_rows': int((bounded['CHARTTIME'] > bounded['DISCHTIME']).sum()), 'after_admission_plus_24h_rows': int((bounded['CHARTTIME'] > bounded['ADMITTIME'] + pd.Timedelta(hours=24)).sum())}
    return {'all_cached_lab_rows': len(raw), 'retained_cohort_cached_lab_rows': len(subset), 'known_wrong_assignments': wrong_items, 'cached_median_reconstruction': comparisons, 'recorded_time_scope': timing, 'timing_interpretation': 'The historical extractor joins by admission key but does not enforce admission/discharge or 24-hour CHARTTIME bounds. These counts describe cached measurement times, not event onset or clinician result availability. No row was removed or recoded in preparation.', 'units_available_in_raw_cache': False, 'raw_LABEVENTS_independently_reextracted': False, 'complete_terminology_or_unit_audit': False}

def main() -> None:
    require(HERE == EXPECTED_HERE.resolve(), 'Preparation must run in the authorized output branch')
    require(not NPZ_PATH.exists() and (not REPORT_PATH.exists()), 'Refusing to overwrite preparation outputs')
    before = check_source_identities()
    features = ordered_features()
    feat = pd.read_csv(source('features'))
    labels = pd.read_csv(source('labels'))
    admissions = pd.read_csv(source('admissions'))
    paper = pd.read_csv(source('paper_predictions'))
    for name, frame in (('features', feat), ('labels', labels), ('admissions', admissions), ('paper', paper)):
        require(frame['HADM_ID'].notna().all(), f'Missing admission key in {name}')
        require(frame['HADM_ID'].is_unique, f'Duplicate admission key in {name}')
        require(frame['SUBJECT_ID'].notna().all(), f'Missing patient key in {name}')
    require(len(feat) == len(labels) == len(admissions) == 11622, 'Original source denominator changed')
    require(set(feat['HADM_ID']) == set(labels['HADM_ID']) == set(admissions['HADM_ID']), 'Source admission-key sets differ')
    for other in (labels, admissions):
        linked = other.set_index('HADM_ID').reindex(feat['HADM_ID'])
        require(np.array_equal(feat['SUBJECT_ID'], linked['SUBJECT_ID']), 'Source patient-admission linkage differs')
        require(np.array_equal(feat['HOSPITAL_EXPIRE_FLAG'], linked['HOSPITAL_EXPIRE_FLAG']), 'Source death flags differ')
    if 'HOSPITAL_EXPIRE_FLAG' in feat.columns:
        merged = feat.merge(labels[['HADM_ID', 'MACE']], on='HADM_ID', how='inner', validate='one_to_one')
    else:
        merged = feat.merge(labels[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'MACE']], on='HADM_ID', how='inner', validate='one_to_one')
    frame = merged.loc[merged['HOSPITAL_EXPIRE_FLAG'] == 0].copy().reset_index(drop=True)
    require(set(frame['HADM_ID']) == set(paper['HADM_ID']), 'Paper and prepared admission sets differ')
    matched = paper.set_index('HADM_ID').reindex(frame['HADM_ID'])
    require(np.array_equal(frame['HADM_ID'], paper['HADM_ID']), 'Paper and original matrix row ordering differ')
    require(np.array_equal(frame['MACE'], matched['MACE']), 'Paper labels differ after key matching')
    require(np.array_equal(frame['SUBJECT_ID'], matched['SUBJECT_ID']), 'Paper patient identities differ after key matching')
    require(all((column in frame for column in features)), 'Original nominal feature column missing')
    original_matrix = frame[features].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    x = original_matrix.astype(np.float32)
    y = frame['MACE'].to_numpy(dtype=np.int64)
    groups, _ = pd.factorize(frame['SUBJECT_ID'], sort=True)
    groups = groups.astype(np.int64)
    original_probabilities = matched['oof_prob_paper'].to_numpy(dtype=np.float64)
    require(x.shape == (9946, 43), 'Prepared matrix shape changed')
    require(np.unique(groups).size == 6783 and y.sum() == 4462, 'Original patient/label counts changed')
    require(set(np.unique(y)) == {0, 1}, 'Non-binary label detected')
    require(np.isfinite(original_probabilities).all(), 'Original probabilities contain nonfinite values')
    require(((original_probabilities >= 0) & (original_probabilities <= 1)).all(), 'Original probabilities outside [0,1]')
    require(not np.isinf(x).any(), 'Matrix contains infinite values')
    require(np.array_equal(np.isnan(x), np.isnan(original_matrix)), 'Float32 cast altered missingness')
    all_missing = np.isnan(x).all(axis=0)
    require(int(all_missing.sum()) == 17, 'Original all-missing feature count changed')
    lab_audit = audit_cached_laboratories(frame)
    legacy_overlap = []
    for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y), 1):
        legacy_overlap.append({'fold': fold, 'test_admissions': len(test), 'shared_patient_count': int(np.intersect1d(groups[train], groups[test]).size), 'test_admissions_with_a_training_patient': int(np.isin(groups[test], groups[train]).sum())})
    require(sum((r['test_admissions_with_a_training_patient'] for r in legacy_overlap)) == 4499, 'Historical splitter replay changed')
    after = check_source_identities()
    require(before == after, 'Source content changed while preparing')
    payload = {'X': x, 'y': y, 'groups': groups, 'p_original': original_probabilities, 'feature_names': np.asarray(features, dtype=str)}
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NPZ_PATH.open('xb') as handle:
        np.savez_compressed(handle, **payload)
    with np.load(NPZ_PATH, allow_pickle=False) as saved:
        require(set(saved.files) == set(payload), 'Saved NPZ fields differ')
        for name, values in payload.items():
            equal = np.array_equal(saved[name], values, equal_nan=True) if values.dtype.kind == 'f' else np.array_equal(saved[name], values)
            require(equal and saved[name].dtype == values.dtype, f'Saved NPZ validation failed: {name}')
    finite = np.isfinite(original_matrix)
    report = {'schema_version': 1, 'analysis_scope': 'Original v13 admission-label/matrix preparation for new patient-grouped internal validation', 'passed': True, 'execution_complete': True, 'no_training_executed': True, 'historical_scripts_executed_or_imported': False, 'cross_centre_mapping_or_OMOP_performed': False, 'sources': before, 'source_identities_unchanged': before == after, 'preparation_script': identity(Path(__file__).resolve()), 'prepared_npz': identity(NPZ_PATH), 'environment': {'python': platform.python_version(), 'platform': platform.platform(), **{name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib')}}, 'cohort': {'original_source_admissions': len(feat), 'original_source_patients': int(feat['SUBJECT_ID'].nunique()), 'death_filter': 'HOSPITAL_EXPIRE_FLAG == 0, unchanged from compute_v13_metrics.py', 'death_admissions_excluded': int((merged['HOSPITAL_EXPIRE_FLAG'] == 1).sum()), 'missing_death_flags': int(merged['HOSPITAL_EXPIRE_FLAG'].isna().sum()), 'prepared_admissions': len(frame), 'prepared_patients': int(np.unique(groups).size), 'positive_admission_labels': int(y.sum()), 'patients_with_any_positive_admission': int(frame.groupby('SUBJECT_ID')['MACE'].max().sum()), 'same_admission_cancer_flag_positive': int(frame['Admission_has_cancer'].sum()), 'same_admission_cancer_flag_zero': int((frame['Admission_has_cancer'] == 0).sum()), 'label_interpretation': 'Original broad concurrent admission diagnosis composite, not incident or fixed-horizon MACE'}, 'key_checks': {'all_source_admission_keys_unique': True, 'source_patient_and_death_flag_linkage_equal': True, 'paper_admission_keyset_equal': True, 'paper_row_order_equal': True, 'paper_labels_equal_after_key_match': True, 'paper_patient_identifiers_equal_after_key_match': True, 'original_probability_field': 'oof_prob_paper'}, 'matrix': {'shape': list(x.shape), 'dtype': str(x.dtype), 'original_feature_list_source': 'FEATS_43_EN literal, AST-read without execution', 'feature_names_in_order': features, 'all_missing_features_retained': [f for f, missing in zip(features, all_missing) if missing], 'all_missing_count': int(all_missing.sum()), 'infinite_cells': int(np.isinf(x).sum()), 'total_missing_fraction': float(np.isnan(x).mean()), 'float32_maximum_absolute_cast_difference': float(np.max(np.abs(x.astype(np.float64)[finite] - original_matrix[finite]))), 'per_feature': [{'index': i, 'name': column, 'nonmissing_admissions': int(np.isfinite(x[:, i]).sum()), 'missing_fraction': float(np.isnan(x[:, i]).mean())} for i, column in enumerate(features)], 'age': {'column_name': 'Age', 'column_index': features.index('Age'), 'source_definition': 'Admission year minus birth year, clipped above at 91 in historical cohort code', 'minimum': float(frame['Age'].min()), 'maximum': float(frame['Age'].max()), 'missing_admissions': int(frame['Age'].isna().sum()), 'admissions_below_18': int((frame['Age'] < 18).sum()), 'reconstructed_or_replaced': False}, 'preparation_applied_imputation_or_column_drop': False, 'training_requirement': 'Learn imputation and empty-column removal only from each fitting partition; keep 43 nominal columns in this prepared file.'}, 'npz_contract': {'fields': {name: {'shape': list(values.shape), 'dtype': str(values.dtype)} for name, values in payload.items()}, 'groups_encoding': 'Sorted factorization of source SUBJECT_ID into contiguous integer group codes; original identifiers are not stored', 'no_raw_patient_or_admission_identifiers_stored': True, 'p_original_is_reference_only_not_a_predictor': True}, 'excluded_nonpredictor_fields_present': [column for column in frame if column not in features], 'explicit_forbidden_predictor_fields': sorted(FORBIDDEN_PREDICTORS), 'known_mapping_audit': lab_audit, 'prespecified_site_local_sensitivity': {'remove_columns': ['HDL Cholesterol', 'AST'], 'nominal_feature_indices': [features.index('HDL Cholesterol'), features.index('AST')], 'basis': 'Three confirmed wrong ITEMID assignments, fixed before any new training or evaluation', 'implementation': 'Drop these whole columns in a separate same-cohort, same-patient-fold refit; leave the primary prepared matrix unchanged.', 'not_a_complete_mapping_repair': True, 'no_target_labels_used_to_choose_columns': True}, 'original_model_configuration': {'ensemble': '0.5 XGBoost probability + 0.5 RandomForest probability', 'imputer': "SimpleImputer(strategy='median'); all-empty columns dropped under current sklearn defaults", 'xgboost': {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'Training negative/positive ratio', 'seed': '1-based outer fold', 'num_boost_round_maximum': 300, 'early_stopping_rounds': 20, 'early_stopping_dataset_in_original': 'Outer validation fold', 'original_prediction': 'booster.predict(dval) without explicit iteration_range'}, 'random_forest': {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 5, 'class_weight': 'balanced', 'random_state': '1-based outer fold'}, 'new_validation_requirement': 'Do not reuse outer-test labels for early stopping; any new fitting rule is an explicitly recorded correction, not a byte-exact historical model replay.'}, 'historical_row_stratified_split_replay': legacy_overlap, 'limits': ['This preserves historical survivor-only admissions, including admissions without same-stay cancer coding; it does not certify cohort appropriateness.', 'The original diagnosis composite includes broad stroke, cardiomyopathy and angina coding; a discharge diagnosis is not an onset date, event-free index or adjudicated MACE.', 'Full admission-key cached measurement summaries can include measurements after disease and outside recorded admission/discharge bounds; patient grouping cannot repair temporal leakage.', 'The raw cache lacks result units and availability timestamps. Only three specified mapping errors were confirmed; other assay/specimen/property combinations remain unaudited.', 'Removing both implicated columns is a site-local sensitivity, not an assertion that all other laboratory mappings are valid.', 'Grouped internal validation on a reused cohort is not independent external or prospective validation.', 'No model was trained, no probability was recalibrated and no new predictive performance estimate is reported by this preparation module.']}
    with REPORT_PATH.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'passed': True, 'prepared_admissions': len(frame), 'prepared_patients': int(np.unique(groups).size), 'positive_admission_labels': int(y.sum()), 'nominal_features': len(features), 'all_missing_columns_retained': int(all_missing.sum())}, indent=2))
if __name__ == '__main__':
    main()
