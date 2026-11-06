"""Prepare original Option B patient means without training or source changes."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
ORIGINAL = Path(_release_path('china', ''))
OPTION_B = _release_source_path(ORIGINAL / 'optionB_parallel_analysis.py')
LABELS = ORIGINAL / 'outputs/survival_analysis_patient_level/patient_level_data.csv'
PREDICTIONS = ORIGINAL / 'outputs/real_model_predictions_v2.csv'
PAPER_PREDICTIONS = Path(_release_path('pipeline', 'data/real_model_predictions_v2.csv'))
RAW = {'SCH': ORIGINAL / 'SCH_processed_data.csv', 'WHU': ORIGINAL / 'WHU_Cancer_processed_data.csv'}
EXPECTED_HASHES = {}
EXPECTED_COUNTS = {'SCH': (530450, 38509, 1323), 'WHU': (7325, 2562, 281)}
KEY = '病案号'
AGE = '年龄'
SEX = '性别'
SUSPECT_FIELDS = ['上次诊断', '上次入院科室', '上次出院科室']
REGIONAL_FIELDS = ['人均地区生产总值(元)', '职工平均工资(元)', '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)']
EXCLUSIONS = ['Unnamed', '病案号', '时间', 'time', '日期', '地址', '电话', '姓名', '编码', '名称', '人员', '单位', '联系']

def sha(path: Path) -> str:
    return None

def identity(path: Path) -> dict:
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size}

def source_identities() -> list[dict]:
    result = []
    for path, expected in EXPECTED_HASHES.items():
        item = identity(path)
        result.append(item)
    return result

def feature_roster() -> tuple[list[str], dict]:
    tree = ast.parse(OPTION_B.read_text(encoding='utf-8-sig'))
    definitions = [node for node in tree.body if isinstance(node, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'exclude_pats' for t in node.targets))]
    assert len(definitions) == 1
    assert ast.literal_eval(definitions[0].value) == EXCLUSIONS
    sch_header = pd.read_csv(RAW['SCH'], nrows=0).columns.tolist()
    whu_header = pd.read_csv(RAW['WHU'], nrows=0).columns.tolist()
    features = [name for name in sch_header if name in set(whu_header) and (not any((token in name for token in EXCLUSIONS)))]
    assert len(features) == 104 and len(set(features)) == 104
    assert all((name in features for name in SUSPECT_FIELDS + REGIONAL_FIELDS + [AGE, SEX]))
    assert not set(features).intersection({KEY, 'Center', 'MACE', 'FollowupDays', 'predicted_prob', 'predicted_prob_within_cv'})
    return (features, {'sch_header_columns': len(sch_header), 'whu_header_columns': len(whu_header), 'selection': 'Original SCH header order, exact WHU name intersection, original case-sensitive exclusion substrings; no new cross-centre mapping or harmonization.', 'exclusion_substrings': EXCLUSIONS, 'excluded_shared_columns': [name for name in sch_header if name in set(whu_header) and any((token in name for token in EXCLUSIONS))]})

def match_predictions(labels: pd.DataFrame, predictions: pd.DataFrame, center: str) -> pd.DataFrame:
    target = predictions.loc[predictions.Center.eq(center)].copy()
    assert target[KEY].notna().all() and (not target[KEY].duplicated().any())
    assert set(target[KEY]) == set(labels[KEY]), f'Patient-key coverage mismatch for {center}'
    joined = labels.merge(target, on=KEY, how='left', validate='one_to_one', sort=False, suffixes=('_label', '_cache'))
    assert np.array_equal(joined.MACE_label.astype(np.int64), joined.MACE_cache.astype(np.int64)), f'Label discrepancy for {center}'
    assert joined[KEY].equals(labels[KEY].reset_index(drop=True))
    return joined

def demographic_disagreements(joined: pd.DataFrame, frame: pd.DataFrame) -> dict:
    result = {}
    for field in (AGE, SEX):
        source = frame[field].to_numpy(dtype=float)
        cached = pd.to_numeric(joined[field], errors='coerce').to_numpy(dtype=float)
        result[field] = {'different_from_original_patient_mean': int((~np.isclose(source, cached, equal_nan=True, rtol=1e-07, atol=1e-07)).sum()), 'cached_nonmissing': int(np.isfinite(cached).sum())}
    return result

def prepare_center(center: str, features: list[str], labels: pd.DataFrame, original: pd.DataFrame, paper: pd.DataFrame) -> dict:
    expected_rows, expected_n, expected_positive = EXPECTED_COUNTS[center]
    local_labels = labels.loc[labels.Center.eq(center), [KEY, 'MACE']].copy().reset_index(drop=True)
    local_labels['MACE'] = local_labels.MACE.astype(np.int64)
    assert len(local_labels) == expected_n and int(local_labels.MACE.sum()) == expected_positive
    assert local_labels[KEY].notna().all() and (not local_labels[KEY].duplicated().any())
    raw = pd.read_csv(RAW[center], usecols=[KEY] + features, low_memory=False)
    assert len(raw) == expected_rows
    assert raw[KEY].notna().all() and raw[KEY].dtype == local_labels[KEY].dtype
    assert raw[KEY].nunique() == expected_n and set(raw[KEY]) == set(local_labels[KEY])
    raw_source_nonmissing = raw[features].notna().sum()
    for field in features:
        raw[field] = pd.to_numeric(raw[field], errors='coerce')
    numeric_nonmissing = raw[features].notna().sum()
    assert all((int(numeric_nonmissing[field]) == 0 for field in SUSPECT_FIELDS)), 'A diagnosis/department field became numeric; stop before training'
    variable_demographics = {field: int(raw.groupby(KEY)[field].nunique(dropna=True).gt(1).sum()) for field in (AGE, SEX)}
    aggregated = raw.groupby(KEY)[features].mean().reset_index()
    del raw
    merged = local_labels.merge(aggregated, on=KEY, how='inner', validate='one_to_one', sort=False)
    assert len(merged) == expected_n and merged[KEY].equals(local_labels[KEY])
    assert int(merged.MACE.sum()) == expected_positive
    original_join = match_predictions(local_labels, original, center)
    paper_join = match_predictions(local_labels, paper, center)
    p_original = original_join.predicted_prob_within_cv.to_numpy(dtype=np.float64)
    assert np.array_equal(p_original, paper_join.predicted_prob_within_cv.to_numpy(dtype=np.float64))
    assert np.isfinite(p_original).all() and ((p_original >= 0) & (p_original <= 1)).all()
    X = merged[features].to_numpy(dtype=np.float32)
    assert not np.isinf(X).any(), 'Nonfinite overflow needs explicit handling before training'
    y = merged.MACE.to_numpy(dtype=np.int64)
    groups = np.arange(expected_n, dtype=np.int64)
    names = np.asarray(features, dtype=str)
    assert X.shape == (expected_n, 104) and len(np.unique(groups)) == expected_n
    assert np.array_equal(np.isnan(X), merged[features].isna().to_numpy())
    assert all((np.isnan(X[:, features.index(field)]).all() for field in SUSPECT_FIELDS))
    output = ROOT / 'private/prepared' / f'{center}.npz'
    with output.open('xb') as stream:
        np.savez_compressed(stream, X=X, y=y, groups=groups, p_original=p_original, feature_names=names)
    with np.load(output, allow_pickle=False) as saved:
        assert set(saved.files) == {'X', 'y', 'groups', 'p_original', 'feature_names'}
        for key, expected in (('X', X), ('y', y), ('groups', groups), ('p_original', p_original), ('feature_names', names)):
            actual = saved[key]
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert np.array_equal(actual, expected, equal_nan=True) if np.issubdtype(actual.dtype, np.number) else np.array_equal(actual, expected)
    per_feature = [{'index': index, 'name': field, 'raw_source_nonmissing': int(raw_source_nonmissing[field]), 'raw_numeric_nonmissing_after_original_coercion': int(numeric_nonmissing[field]), 'patient_nonmissing_after_mean': int(np.isfinite(X[:, index]).sum()), 'patient_missing_after_mean': int(np.isnan(X[:, index]).sum()), 'patient_missing_fraction': float(np.isnan(X[:, index]).mean()), 'finite_unique_patient_values': int(merged[field].nunique(dropna=True))} for index, field in enumerate(features)]
    age_values = X[:, features.index(AGE)]
    sex_values = X[:, features.index(SEX)]
    return {'center': center, 'raw_rows': expected_rows, 'n': expected_n, 'positive': expected_positive, 'unit': 'patient', 'label_column': 'MACE', 'original_probability_column': 'predicted_prob_within_cv', 'shape': list(X.shape), 'X_dtype': str(X.dtype), 'n_features': len(features), 'missing_cells': int(np.isnan(X).sum()), 'infinite_cells': int(np.isinf(X).sum()), 'all_missing_features': [item['name'] for item in per_feature if item['patient_nonmissing_after_mean'] == 0], 'raw_patient_keys_equal_all_label_keys': True, 'original_cache_keys_one_to_one': True, 'original_cache_labels_equal': True, 'paper_cache_keys_one_to_one': True, 'paper_cache_labels_equal': True, 'original_and_paper_within_probabilities_identical': True, 'source_missing_key_rows': 0, 'label_duplicate_keys': 0, 'raw_patients_excluded': 0, 'label_patients_excluded': 0, 'groups_policy': 'Zero-based positional integers, unique within this centre; no patient identifiers persisted. Use (centre, group) if centres are ever combined.', 'source_patient_order': 'Original label-table order, same left-side order as Option B inner merge.', 'diagnosis_department_columns_non_numeric': {field: True for field in SUSPECT_FIELDS}, 'demographics': {'age_column': AGE, 'age_index': features.index(AGE), 'sex_column': SEX, 'sex_index': features.index(SEX), 'authority': 'Original processed file, to_numeric then patient mean; cached metadata columns are never model inputs.', 'age_nonmissing': int(np.isfinite(age_values).sum()), 'age_range': [float(np.nanmin(age_values)), float(np.nanmax(age_values))], 'sex_nonmissing': int(np.isfinite(sex_values).sum()), 'sex_numeric_codes': sorted((float(value) for value in np.unique(sex_values[np.isfinite(sex_values)]))), 'patients_with_variable_source_demographics': variable_demographics, 'original_prediction_metadata_comparison': demographic_disagreements(original_join, merged), 'paper_prediction_metadata_comparison': demographic_disagreements(paper_join, merged)}, 'features': per_feature, 'output': identity(output), 'round_trip_npz_validation_passed': True}

def main():
    started = datetime.now(timezone.utc).isoformat()
    expected_root = Path(_release_path('project', 'revision_2026-10-15/original_v5_independent_validation_v1')).resolve()
    assert ROOT == expected_root
    outputs = [ROOT / 'private/prepared/SCH.npz', ROOT / 'private/prepared/WHU.npz', ROOT / 'evidence/china_preparation.json']
    assert all((path.parent.is_dir() and (not path.exists()) for path in outputs)), 'New output directories must exist and outputs must not exist'
    inputs_before = source_identities()
    features, roster_details = feature_roster()
    labels = pd.read_csv(LABELS, usecols=[KEY, 'Center', 'MACE'], low_memory=False)
    pred_columns = [KEY, 'Center', 'MACE', 'predicted_prob_within_cv', AGE, SEX]
    original = pd.read_csv(PREDICTIONS, usecols=pred_columns, low_memory=False)
    paper = pd.read_csv(PAPER_PREDICTIONS, usecols=pred_columns, low_memory=False)
    for frame in (labels, original, paper):
        assert len(frame) == 41071 and set(frame.Center) == {'SCH', 'WHU'}
        assert not frame.duplicated(['Center', KEY]).any() and frame.MACE.isin([0, 1]).all()
    assert not set(labels.loc[labels.Center.eq('SCH'), KEY]).intersection(labels.loc[labels.Center.eq('WHU'), KEY])
    cohorts = []
    for center in ('SCH', 'WHU'):
        print(json.dumps({'stage': 'prepare_original_patient_means', 'center': center}), flush=True)
        cohorts.append(prepare_center(center, features, labels, original, paper))
    inputs_after = source_identities()
    assert inputs_before == inputs_after, 'Original input changed during preparation'
    report = {'passed': True, 'execution_complete': True, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Source-only preparation of original-label Option B patient-mean matrices; no model training, label changes, OMOP alignment, unit conversion or new cross-centre mapping.', 'privacy': 'Only aggregate counts and source/feature metadata are public. The two local private NPZs contain clinical predictor arrays and probabilities but no original patient IDs or diagnosis texts; do not publish them.', 'inputs': inputs_before, 'inputs_unchanged_after_preparation': True, 'script': identity(Path(__file__)), 'environment': {'python_executable': sys.executable, 'python': sys.version, **{name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost')}}, 'feature_names': features, 'n_features': len(features), 'feature_selection': roster_details, 'model_family_source': {'path': str(OPTION_B), 'ensemble': 'Equal arithmetic mean of XGBoost binary probability and RandomForest probability.', 'original_xgboost': {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42, 'nthread': -1, 'scale_pos_weight': 'Whole-centre negatives / positives, calculated before the original outer CV.'}, 'original_random_forest': {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 5, 'class_weight': '{0: 1, 1: int(whole-centre negatives / positives)}', 'random_state': 42, 'n_jobs': -1}, 'original_cv': 'StratifiedKFold(5, shuffle=True, random_state=42) on one patient-mean row; XGBoost up to 300 rounds, early stopping 20 on the outer validation fold.', 'original_transfer': 'Separate full-source models with 200 XGBoost rounds; not executed here.', 'original_preprocessing': 'Patient-wise arithmetic mean of numeric-coerced raw records, then zero fill and float32 nan_to_num(nan=0, posinf=0, neginf=0). This preparation preserves missing means as NaN; the new training stage must apply its explicitly frozen original-zero policy.'}, 'cohorts': cohorts, 'limitations': ['Original ever-MACE labels are retained verbatim, not clinically re-adjudicated; missing diagnostic text and literal-rule errors remain.', 'All-record patient means have no verified pre-event cutoff. Patient-isolated validation cannot turn this into future first-MACE prediction.', 'Three diagnosis/department text fields survive original name filtering but yield no numeric values in either entire source; retained as all-NaN columns with hard assertions, becoming zero constants under original preprocessing.', 'The five original regional variables are retained without revalidation of geographic assignment, independence or timing.', 'Age and sex come from the original processed predictors, not from discordant prediction-cache metadata; no new sex recoding, index-age reconstruction or validity exclusion was introduced.', 'Original cross-centre shared column-name selection is reproduced solely to identify the frozen 104 inputs; it is not evidence of semantic, unit or platform harmonization.', 'This source reconstruction establishes deterministic input provenance and label/cache linkage, not bitwise reproduction of historical fitted trees or independent external validation.', 'No original-format 104-column serialized patient matrix with verified provenance was located; these matrices are newly and explicitly rebuilt from the original processed files.']}
    with outputs[-1].open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'passed': True, 'execution_complete': True, 'cohorts': [{'center': item['center'], 'n': item['n'], 'positive': item['positive'], 'shape': item['shape'], 'missing_cells': item['missing_cells']} for item in cohorts], 'original_inputs_unchanged': True}), flush=True)
if __name__ == '__main__':
    main()
