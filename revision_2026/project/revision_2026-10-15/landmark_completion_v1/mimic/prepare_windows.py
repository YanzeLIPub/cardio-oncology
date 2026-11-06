"""Freeze original-label MIMIC measurement-window matrices without training."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OLD = Path(_release_path('project', 'revision_2026-10-15/original_v5_independent_validation_v1'))
RAW_ADMISSIONS = Path(_release_path('mimic', 'ADMISSIONS.csv.gz'))

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def checked(record):
    assert identity(record['path']) == record, record['path']
    return record

def save_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=True, allow_nan=False, indent=2)
        handle.write('\n')

def main():
    started = time.monotonic()
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8')))
    assert protocol['window_specification']['hours'] == [0, 24, 48]
    prepared = ROOT / 'private/prepared'
    assert not prepared.exists(), 'Refuse to overwrite prepared data'
    for name in ('evidence', 'results', 'figures'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    old_lock = _release_resolve_config(json.loads((OLD / 'evidence/analysis_lock.json').read_text(encoding='utf-8')))
    for record in old_lock['state']['files']:
        checked(record)
    provenance = _release_resolve_config(json.loads((OLD / 'evidence/mimic_preparation.json').read_text(encoding='utf-8')))
    sources = list(provenance['sources'].values())
    sources += [identity(ROOT / 'protocol.json'), identity(Path(__file__)), identity(RAW_ADMISSIONS), identity(OLD / 'evidence/analysis_lock.json'), identity(OLD / 'evidence/mimic_preparation.json')]
    for record in sources:
        checked(record)
    source = {key: Path(record['path']) for key, record in provenance['sources'].items()}
    frame = pd.read_csv(source['features'])
    paper = pd.read_csv(source['paper_predictions'])
    assert paper.HADM_ID.is_unique and len(paper) == 9946
    frame = frame.set_index('HADM_ID').reindex(paper.HADM_ID).reset_index()
    assert np.array_equal(frame.SUBJECT_ID.to_numpy(), paper.SUBJECT_ID.to_numpy())
    with np.load(OLD / 'private/prepared/MIMIC.npz', allow_pickle=False) as f:
        original = {key: f[key].copy() for key in f.files}
    checked(provenance['prepared_npz'])
    names = original['feature_names'].tolist()
    assert len(names) == 43 and original['X'].shape == (9946, 43)
    assert np.array_equal(original['y'], paper.MACE.to_numpy())
    groups, _ = pd.factorize(frame.SUBJECT_ID, sort=True)
    assert np.array_equal(groups, original['groups'])
    assert np.unique(groups).size == 6783 and original['y'].sum() == 4462
    assert np.array_equal(original['X'], frame[names].to_numpy(dtype=np.float32), equal_nan=True)
    assert np.array_equal(original['p_original'], paper.oof_prob_paper.to_numpy())
    training = _release_resolve_config(json.loads((OLD / 'results/MIMIC_original_training.json').read_text(encoding='utf-8')))
    checked(training['roles_identity'])
    checked(training['prediction_identity'])
    sources += [identity(OLD / 'results/MIMIC_original_training.json'), training['roles_identity'], training['prediction_identity']]
    with np.load(training['roles_identity']['path'], allow_pickle=False) as f:
        roles = {key: f[key].copy() for key in f.files}
    assert np.array_equal(roles['y'], original['y']) and np.array_equal(roles['groups'], groups)
    assert roles['roles'].shape == (5, 9946)
    assert np.all((roles['roles'] == 0).sum(axis=0) == 1)
    for role in roles['roles']:
        sets = [set(groups[role == i]) for i in range(4)]
        assert all((not sets[i].intersection(sets[j]) for i in range(4) for j in range(i + 1, 4)))
    admissions = pd.read_csv(RAW_ADMISSIONS, usecols=['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'EDREGTIME', 'EDOUTTIME'])
    assert admissions.HADM_ID.is_unique
    admissions = admissions.set_index('HADM_ID').reindex(paper.HADM_ID).reset_index()
    assert np.array_equal(admissions.SUBJECT_ID.to_numpy(), paper.SUBJECT_ID.to_numpy())
    for key in ('ADMITTIME', 'DISCHTIME', 'EDREGTIME', 'EDOUTTIME'):
        admissions[key] = pd.to_datetime(admissions[key], errors='coerce')
    for key in ('ADMITTIME', 'DISCHTIME'):
        assert admissions[key].equals(pd.to_datetime(frame[key], errors='coerce'))
    valid_times = admissions.ADMITTIME.notna() & admissions.DISCHTIME.notna() & admissions.DISCHTIME.ge(admissions.ADMITTIME)
    ed_valid = admissions.EDREGTIME.notna() & admissions.EDREGTIME.le(admissions.ADMITTIME)
    admissions['WINDOW_START'] = admissions.EDREGTIME.where(ed_valid, admissions.ADMITTIME)
    admissions['valid_times'] = valid_times
    admissions['row_index'] = np.arange(len(admissions))
    subset = (valid_times & admissions.DISCHTIME.gt(admissions.ADMITTIME + pd.Timedelta(hours=48))).to_numpy()
    assert subset.sum() == protocol['secondary_common_subset']['expected_admissions'] == 9427
    raw = pd.read_csv(source['raw_lab_cache'], usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'Feature'])
    raw = raw[raw.HADM_ID.isin(paper.HADM_ID)].copy()
    raw = raw.merge(admissions, on=['HADM_ID', 'SUBJECT_ID'], how='left', validate='many_to_one')
    assert raw.row_index.notna().all()
    raw.CHARTTIME = pd.to_datetime(raw.CHARTTIME, errors='coerce')
    raw.VALUENUM = pd.to_numeric(raw.VALUENUM, errors='coerce')
    assert np.isfinite(raw.VALUENUM).all()
    assert set(raw.Feature) <= set(names)
    full_pivot = raw.groupby(['HADM_ID', 'Feature']).VALUENUM.median().unstack('Feature').reindex(paper.HADM_ID)
    full_pivot['Globulin'] = full_pivot['Total Protein'] - full_pivot['Albumin']
    for key in full_pivot:
        assert np.allclose(full_pivot[key].to_numpy(), frame[key].to_numpy(), equal_nan=True, rtol=0, atol=1e-10)
    prepared.mkdir(parents=True)
    np.savez_compressed(prepared / 'patient_roles.npz', **roles)
    np.savez_compressed(prepared / 'evaluation_subset.npz', still_hospitalized_after_48h=subset)
    window_reports = []
    previous_selected = np.zeros(len(raw), dtype=bool)
    for hour in (0, 24, 48):
        nominal_end = raw.ADMITTIME + pd.Timedelta(hours=hour)
        end = pd.concat([nominal_end, raw.DISCHTIME], axis=1).min(axis=1)
        selected = (raw.valid_times & raw.CHARTTIME.notna() & raw.CHARTTIME.ge(raw.WINDOW_START) & raw.CHARTTIME.le(end)).to_numpy()
        assert not np.any(previous_selected & ~selected)
        previous_selected = selected
        current = raw.loc[selected]
        pivot = current.groupby(['HADM_ID', 'Feature']).VALUENUM.median().unstack('Feature').reindex(paper.HADM_ID)
        matrix = np.full((9946, 43), np.nan, dtype=np.float32)
        for key in ('Age', 'Sex_M'):
            matrix[:, names.index(key)] = original['X'][:, names.index(key)]
        for key in pivot:
            matrix[:, names.index(key)] = pivot[key].to_numpy(dtype=np.float32)
        if 'Total Protein' in pivot and 'Albumin' in pivot:
            matrix[:, names.index('Globulin')] = (pivot['Total Protein'] - pivot['Albumin']).to_numpy(dtype=np.float32)
        assert not np.isinf(matrix).any()
        payload = dict(original)
        payload['X'] = matrix
        destination = prepared / f'MIMIC_{hour}h.npz'
        np.savez_compressed(destination, **payload)
        with np.load(destination, allow_pickle=False) as f:
            for key in payload:
                assert np.array_equal(f[key], payload[key], equal_nan=True) if payload[key].dtype.kind == 'f' else np.array_equal(f[key], payload[key])
        window_reports.append({'hours': hour, 'prepared': identity(destination), 'selected_cached_measurement_rows': int(selected.sum()), 'admissions_with_any_selected_measurement': int(current.HADM_ID.nunique()), 'patients_with_any_selected_measurement': int(current.SUBJECT_ID.nunique()), 'all_missing_nominal_columns': [n for i, n in enumerate(names) if np.isnan(matrix[:, i]).all()], 'per_feature': [{'feature': n, 'observed_admissions': int(np.isfinite(matrix[:, i]).sum())} for i, n in enumerate(names)]})
    before = raw.CHARTTIME.lt(raw.ADMITTIME)
    in_ed = before & raw.EDREGTIME.notna() & raw.EDREGTIME.le(raw.ADMITTIME) & raw.CHARTTIME.ge(raw.EDREGTIME)
    for record in sources:
        checked(record)
    report = {'passed': True, 'execution_complete': True, 'no_training': True, 'protocol_identity': identity(ROOT / 'protocol.json'), 'script_identity': identity(Path(__file__)), 'source_identities': sources, 'original_source_lock_verified': True, 'unit': 'admission', 'admissions': 9946, 'patients': 6783, 'positive_labels': 4462, 'original_label_order_group_and_probability_preserved': True, 'unbounded_cached_medians_replayed': True, 'roles_identical_to_previous_patient_isolated_validation': True, 'prepared_roles': identity(prepared / 'patient_roles.npz'), 'subset_identity': identity(prepared / 'evaluation_subset.npz'), 'still_hospitalized_after_48h_subset': {'admissions': int(subset.sum()), 'patients': int(np.unique(groups[subset]).size), 'positive_labels': int(original['y'][subset].sum()), 'evaluation_only_no_refit': True, 'selection_uses_future_stay_duration': True, 'event_free_status_established': False}, 'source_time_counts': {'retained_cache_rows': len(raw), 'valid_ed_anchor_admissions': int(ed_valid.sum()), 'invalid_required_stay_times': int((~valid_times).sum()), 'preadmission_rows': int(before.sum()), 'preadmission_admissions': int(raw.loc[before, 'HADM_ID'].nunique()), 'preadmission_rows_in_valid_same_admission_ed_interval': int(in_ed.sum()), 'preadmission_rows_outside_ed_or_missing_ed': int((before & ~in_ed).sum()), 'exact_admission_measurement_rows': int(raw.CHARTTIME.eq(raw.ADMITTIME).sum()), 'post_discharge_rows_excluded': int(raw.CHARTTIME.gt(raw.DISCHTIME).sum()), 'unknown_charttime_rows': int(raw.CHARTTIME.isna().sum())}, 'windows': window_reports, 'elapsed_seconds': time.monotonic() - started, 'environment': {'python': platform.python_version(), **{n: importlib.metadata.version(n) for n in ('numpy', 'pandas')}}, 'limits': ['CHARTTIME and EDREGTIME delimit measurement records, not first clinician-visible results.', 'The 0-hour matrix is an ED/admission-linked measurement proxy, not complete baseline information.', 'Original labels, survivor selection, all-missing columns and known mapping errors remain in the primary matrices.', 'No original file, OMOP mapping, future-event label, onset date or clinical event-free set was changed or created.', 'All 9946 admissions remain in every primary window; early discharge truncates observation but does not exclude the row.']}
    save_json(ROOT / 'evidence/preparation.json', report)
    print(json.dumps({'passed': True, 'windows': [{k: v for k, v in r.items() if k in ('hours', 'selected_cached_measurement_rows', 'admissions_with_any_selected_measurement')} for r in window_reports], 'seconds': report['elapsed_seconds']}, indent=2))
if __name__ == '__main__':
    main()
