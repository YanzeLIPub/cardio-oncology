"""Aggregate-only Chinese feature lineage and earliest-record demographic audit."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OLD = Path(_release_path('project', 'revision_2026-10-15/original_v5_independent_validation_v1'))
SOURCE = Path(_release_path('china', ''))
KEY = '病案号'
TIME = '检查时间'
AGE = '年龄'
SEX = '性别'
SUSPECT = ['上次诊断', '上次入院科室', '上次出院科室']
REGION = ['人均地区生产总值(元)', '职工平均工资(元)', '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)']
EXCLUDE = ['Unnamed', KEY, '时间', 'time', '日期', '地址', '电话', '姓名', '编码', '名称', '人员', '单位', '联系']

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)

def finite_summary(series):
    a = np.asarray(series, dtype=float)
    b = a[np.isfinite(a)]
    return {'n': len(a), 'nonmissing': len(b), 'missing': int((~np.isfinite(a)).sum()), 'min': float(np.min(b)) if len(b) else None, 'max': float(np.max(b)) if len(b) else None, 'quantiles': {str(q): float(np.quantile(b, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)} if len(b) else {}}

def main():
    started = datetime.now(timezone.utc).isoformat()
    prior_path = OLD / 'evidence/china_preparation.json'
    prior = _release_resolve_config(json.loads(prior_path.read_text(encoding='utf-8')))
    labels_path = SOURCE / 'outputs/survival_analysis_patient_level/patient_level_data.csv'
    raws = {'SCH': SOURCE / 'SCH_processed_data.csv', 'WHU': SOURCE / 'WHU_Cancer_processed_data.csv'}
    expected = {i['path'].replace('\\', '/'): None for i in prior['inputs']}
    inputs = [identity(p) for p in [prior_path, labels_path, *raws.values(), _release_source_path(SOURCE / 'optionB_parallel_analysis.py'), _release_source_path(OLD / 'prepare_china.py')]]
    for item in inputs[1:5]:
        pass
    labels = pd.read_csv(labels_path, usecols=[KEY, 'Center', 'MACE'])
    headers = {c: pd.read_csv(p, nrows=0).columns.tolist() for c, p in raws.items()}
    features = [n for n in headers['SCH'] if n in set(headers['WHU']) and (not any((x in n for x in EXCLUDE)))]
    assert features == prior['feature_names'] and len(features) == 104
    labs = [f for f in features if f not in SUSPECT + REGION + [AGE, SEX]]
    assert len(labs) == 94
    roster = []
    for index, feature in enumerate(features):
        category = 'demographic' if feature in [AGE, SEX] else 'diagnosis_or_department_text' if feature in SUSPECT else 'regional_covariate' if feature in REGION else 'laboratory_named'
        roster.append({'index': index, 'feature': feature, 'category': category, 'historical_aggregation': 'numeric coercion, then full-record patient mean and zero fill', 'demographic_control': 'retain earliest recorded row only' if feature in [AGE, SEX] else 'exclude', 'pre_event_status': 'not established by the historical pipeline'})
    results = []
    for center, raw_path in raws.items():
        print(json.dumps({'stage': 'demographic_lineage', 'center': center}), flush=True)
        raw = pd.read_csv(raw_path, usecols=[KEY, TIME, AGE, SEX] + SUSPECT, low_memory=False)
        raw['_order'] = np.arange(len(raw))
        raw['_time'] = pd.to_datetime(raw[TIME], format='mixed', errors='coerce')
        assert not raw['_time'].isna().any() and (not raw[KEY].isna().any())
        text_counts = {}
        for field in [AGE, SEX] + SUSPECT:
            source_n = int(raw[field].notna().sum())
            raw[field] = pd.to_numeric(raw[field], errors='coerce')
            if field in SUSPECT:
                text_counts[field] = {'source_nonmissing_rows': source_n, 'numeric_nonmissing_rows': int(raw[field].notna().sum())}
                assert raw[field].isna().all()
        local = labels.loc[labels.Center.eq(center)].reset_index(drop=True)
        assert set(raw[KEY]) == set(local[KEY]) and (not local[KEY].duplicated().any())
        raw = raw.sort_values([KEY, '_time', '_order'], kind='stable')
        grouped = raw.groupby(KEY, sort=True)
        means = grouped[[AGE, SEX]].mean().reindex(local[KEY])
        first = raw.drop_duplicates(KEY, keep='first').set_index(KEY).reindex(local[KEY])
        earliest = raw.loc[raw._time.eq(grouped._time.transform('min'))]
        tie_sizes = earliest.groupby(KEY).size().reindex(local[KEY])
        prior_npz = OLD / f'private/prepared/{center}.npz'
        role_npz = OLD / f'private/runs/{center}_original/patient_roles.npz'
        inputs.extend([identity(prior_npz), identity(role_npz)])
        cache = dict(np.load(prior_npz, allow_pickle=False))
        roles = dict(np.load(role_npz, allow_pickle=False))
        assert np.array_equal(cache['y'], local.MACE.to_numpy())
        assert np.array_equal(cache['groups'], roles['groups']) and np.array_equal(cache['y'], roles['y'])
        fields = {}
        for field in [AGE, SEX]:
            assert np.array_equal(means[field].to_numpy(dtype=np.float32), cache['X'][:, features.index(field)], equal_nan=True)
            first_values = first[field].to_numpy(dtype=float)
            mean_values = means[field].to_numpy(dtype=float)
            comparable = np.isfinite(first_values) & np.isfinite(mean_values)
            delta = mean_values[comparable] - first_values[comparable]
            fields[field] = {'earliest_record': finite_summary(first_values), 'full_record_mean': finite_summary(mean_values), 'patients_with_multiple_nonmissing_source_values': int(grouped[field].nunique(dropna=True).gt(1).sum()), 'patients_with_conflicting_earliest_tied_values': int(earliest.groupby(KEY)[field].nunique(dropna=True).gt(1).sum()), 'mean_vs_earliest_different_nonmissing': int((~np.isclose(first_values[comparable], mean_values[comparable], rtol=0, atol=1e-08)).sum()), 'mean_minus_earliest': finite_summary(delta)}
        age = first[AGE].to_numpy(dtype=float)
        sex = first[SEX].to_numpy(dtype=float)
        fields[AGE]['earliest_outside_0_to_120'] = int((np.isfinite(age) & ((age < 0) | (age > 120))).sum())
        fields[AGE]['earliest_less_than_18'] = int((np.isfinite(age) & (age < 18)).sum())
        fields[SEX]['earliest_value_counts'] = [{'numeric_code': float(k), 'patients': int(v)} for k, v in first[SEX].value_counts(dropna=True).sort_index().items()]
        changes = []
        for label in [0, 1]:
            mask = local.MACE.to_numpy() == label
            changes.append({'label': label, 'patients': int(mask.sum()), 'earliest_age': finite_summary(age[mask]), 'earliest_age_missing': int(np.isnan(age[mask]).sum()), 'earliest_sex_missing': int(np.isnan(sex[mask]).sum())})
        results.append({'center': center, 'patients': len(local), 'positive_labels': int(local.MACE.sum()), 'records': len(raw), 'first_record_time': 'Earliest recorded check time; not certified index, result availability or event-free time.', 'tie_rule': 'Stable source CSV row position only, without diagnosis or outcome selection.', 'patients_with_earliest_timestamp_ties': int(tie_sizes.gt(1).sum()), 'direct_text_fields': text_counts, 'demographics': fields, 'by_label': changes, 'prepared_original_demographics_exactly_match_full_record_means': True, 'frozen_roles_shape': list(roles['roles'].shape), 'control_input_proposal': ['age on earliest recorded row', 'sex code on earliest recorded row'], 'control_exclusions': 'All labs, missingness indicators, diagnosis/department text, treatment, regional covariates, dates, record counts, follow-up, probabilities and labels as inputs.'})
    after = [identity(item['path']) for item in inputs]
    assert after == inputs
    pd.DataFrame(roster).to_csv(ROOT / 'feature_lineage_roster.csv', index=False, encoding='utf-8-sig')
    result = {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'task': 'R3 direct/indirect outcome feature leakage: original Chinese feature lineage and earliest-record demographic control feasibility', 'feature_categories': {'laboratory_named': 94, 'demographic': 2, 'diagnosis_department_text': 3, 'regional': 5}, 'label_derived_input_names_present': sorted(set(features) & {'MACE', 'MACE_flag', 'FollowupDays', 'followup_days', 'predicted_prob', 'predicted_prob_within_cv', 'Center'}), 'cohorts': results, 'inputs': inputs, 'inputs_unchanged': True, 'script': identity(__file__), 'environment': {'python': sys.version, 'executable': sys.executable, **{p: version(p) for p in ['numpy', 'pandas']}}, 'limitations': ['Laboratory feature names do not certify measurement identity or units.', 'Absence of a literal label column does not exclude biological/clinical temporal leakage.', 'Earliest-record demographics avoid later record aggregation, but do not establish independent clinical onset, first availability or negative-outcome capture.', 'Sex is recorded source coding, not independently verified sex at birth; no code meaning is inferred.', 'No model fitted by this script; outcome labels used only for descriptive strata and preservation checks.']}
    save(ROOT / 'feature_lineage_audit.json', result)
    print(json.dumps({'complete': True, 'input_files_unchanged': True, 'cohorts': [{'center': c['center'], 'n': c['patients'], 'events': c['positive_labels'], 'demographics': c['demographics']} for c in results]}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
