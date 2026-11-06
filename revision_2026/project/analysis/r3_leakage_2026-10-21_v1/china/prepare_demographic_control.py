"""Prepare earliest-timestamp demographic control, without fitting models."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
from audit_feature_lineage import AGE, KEY, OLD, ROOT, SEX, SOURCE, TIME, identity, save

def choose_earliest_demographics(raw):
    raw = raw.copy()
    raw['_time'] = pd.to_datetime(raw[TIME], format='mixed', errors='coerce')
    assert raw['_time'].notna().all(), 'Invalid timestamp: refusing implicit chronology fallback'
    assert raw[KEY].notna().all()
    for field in (AGE, SEX):
        raw[field] = pd.to_numeric(raw[field], errors='coerce')
    earliest = raw.loc[raw._time.eq(raw.groupby(KEY)._time.transform('min'))].copy()
    summary = earliest.groupby(KEY)[[AGE, SEX]].agg(['min', 'max', 'count', 'nunique'])
    age = summary[AGE, 'min'].where(summary[AGE, 'nunique'].eq(1))
    age = age.where(np.isfinite(age) & age.between(0, 120))
    sex = summary[SEX, 'min'].where(summary[SEX, 'nunique'].eq(1))
    sex = sex.where(sex.isin([1, 2])) - 1
    result = pd.DataFrame({'age_at_earliest_check': age, 'sex_source_code2_indicator': sex})
    conflicts = {'age_earliest_conflicts': int(summary[AGE, 'nunique'].gt(1).sum()), 'sex_earliest_conflicts': int(summary[SEX, 'nunique'].gt(1).sum()), 'invalid_age_patient_values': int((summary[AGE, 'count'].gt(0) & ~summary[AGE, 'min'].between(0, 120)).sum()), 'invalid_sex_patient_values': int((summary[SEX, 'count'].gt(0) & ~summary[SEX, 'min'].isin([1, 2])).sum()), 'earliest_timestamp_tie_patients': int(earliest.groupby(KEY).size().gt(1).sum())}
    candidate = raw.sort_values([KEY, '_time'], kind='stable')
    min_clock = candidate.groupby(KEY)._time.first()
    secondary = candidate.join(min_clock.rename('_first'), on=KEY)
    secondary = secondary.loc[secondary._time.eq(secondary._first)]
    for field, name in [(AGE, 'age_at_earliest_check'), (SEX, 'sex_source_code2_indicator')]:

        def value(series):
            values = pd.unique(series.dropna())
            if len(values) != 1:
                return np.nan
            v = float(values[0])
            if field == AGE:
                return v if np.isfinite(v) and 0 <= v <= 120 else np.nan
            return v - 1 if v in (1, 2) else np.nan
        independently = secondary.groupby(KEY)[field].apply(value).reindex(result.index)
        assert np.array_equal(independently.to_numpy(), result[name].to_numpy(), equal_nan=True)
    return (result, conflicts)

def main():
    started = datetime.now(timezone.utc).isoformat()
    protocol = ROOT / 'demographic_preparation_protocol.json'
    prior = _release_resolve_config(json.loads((ROOT / 'feature_lineage_audit.json').read_text(encoding='utf-8')))
    expected = {item['path']: None for item in prior['inputs']}
    labels_path = SOURCE / 'outputs/survival_analysis_patient_level/patient_level_data.csv'
    labels = pd.read_csv(labels_path, usecols=[KEY, 'Center', 'MACE'])
    inputs = [identity(protocol), identity(ROOT / 'feature_lineage_audit.json'), identity(labels_path)]
    private = ROOT / 'private'
    private.mkdir(exist_ok=True)
    reports = []
    for center, filename in [('SCH', 'SCH_processed_data.csv'), ('WHU', 'WHU_Cancer_processed_data.csv')]:
        print(json.dumps({'stage': 'prepare_demographic_control', 'center': center}), flush=True)
        source = SOURCE / filename
        original_npz = OLD / f'private/prepared/{center}.npz'
        role_npz = OLD / f'private/runs/{center}_original/patient_roles.npz'
        bound = [identity(p) for p in [source, original_npz, role_npz]]
        for item in bound:
            pass
        inputs.extend(bound)
        local = labels.loc[labels.Center.eq(center)].reset_index(drop=True)
        original = dict(np.load(original_npz, allow_pickle=False))
        role = dict(np.load(role_npz, allow_pickle=False))
        assert np.array_equal(local.MACE.to_numpy(), original['y'])
        assert np.array_equal(role['groups'], original['groups']) and np.array_equal(role['y'], original['y'])
        raw = pd.read_csv(source, usecols=[KEY, TIME, AGE, SEX], low_memory=False)
        assert set(raw[KEY]) == set(local[KEY]) and (not local[KEY].duplicated().any())
        frame, counts = choose_earliest_demographics(raw)
        frame = frame.reindex(local[KEY])
        X = frame.to_numpy(dtype=np.float32)
        assert X.shape == (len(local), 2) and (not np.isinf(X).any())
        names = np.asarray(frame.columns, dtype=str)
        path = private / f'{center}_demographics.npz'
        with path.open('xb') as handle:
            np.savez_compressed(handle, X=X, groups=original['groups'], feature_names=names)
        saved = dict(np.load(path, allow_pickle=False))
        assert set(saved) == {'X', 'groups', 'feature_names'}
        assert np.array_equal(saved['X'], X, equal_nan=True)
        assert np.array_equal(saved['groups'], original['groups'])
        assert np.array_equal(saved['feature_names'], names)
        reports.append({'center': center, 'n': len(local), 'positive_labels_preserved': int(original['y'].sum()), 'unit': 'patient', 'raw_records': len(raw), 'features': names.tolist(), 'source_sex_mapping': {'1': 0, '2': 1}, 'sex_code_semantics': 'Original numeric coding retained as an unlabeled category; not clinically verified sex at birth.', 'earliest_age_range': [float(np.nanmin(X[:, 0])), float(np.nanmax(X[:, 0]))], 'feature_missing_counts': {name: int(np.isnan(X[:, i]).sum()) for i, name in enumerate(names)}, 'sex_control_code_counts': {str(int(code)): int((X[:, 1] == code).sum()) for code in [0, 1]}, 'source_order_alignment': 'Label-table left order, verified equal to original prepared matrix and frozen patient roles.', 'frozen_roles': identity(role_npz), 'output': identity(path), **counts})
        del raw
    assert [identity(item['path']) for item in inputs] == inputs
    report = {'complete': True, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'protocol': _release_resolve_config(json.loads(protocol.read_text(encoding='utf-8'))), 'inputs': inputs, 'inputs_unchanged': True, 'cohorts': reports, 'independent_selection_and_npz_roundtrip_passed': True, 'script': identity(__file__), 'helper_script': identity(_release_source_path(ROOT / 'audit_feature_lineage.py')), 'environment': {'python': sys.version, 'executable': sys.executable, **{p: version(p) for p in ['numpy', 'pandas']}}, 'models_fitted': 0, 'patient_identifiers_or_clinical_rows_logged': False, 'scope': 'Outcome-independent demographic input restriction; not a validated clinical future-risk model or certification of original full-record biomarkers.'}
    save(ROOT / 'demographic_preparation.json', report)
    print(json.dumps({'complete': True, 'inputs_unchanged': True, 'cohorts': reports}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
