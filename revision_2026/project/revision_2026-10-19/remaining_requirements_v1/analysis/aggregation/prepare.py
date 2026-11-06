"""Prepare matched laboratory aggregation variants without fitting or raw-ID exports."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from pathlib import Path
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OLD = Path(_release_path('project', 'revision_2026-10-15/original_v5_independent_validation_v1'))
KEY = '病案号'
TIME = '检查时间'
LABELS = Path(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'))
RAW = {'SCH': Path(_release_path('china', 'SCH_processed_data.csv')), 'WHU': Path(_release_path('china', 'WHU_Cancer_processed_data.csv'))}
META = ['年龄', '性别', '上次诊断', '上次入院科室', '上次出院科室', '人均地区生产总值(元)', '职工平均工资(元)', '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)']

def sha(path):
    return None

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

def main():
    (ROOT / 'private/prepared').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    evidence = _release_resolve_config(json.loads((OLD / 'evidence/china_preparation.json').read_text(encoding='utf-8')))
    expected = {s['path'].replace('\\', '/'): None for s in evidence['inputs']}
    inputs = [identity(p) for p in [LABELS, *RAW.values(), _release_source_path(OLD / 'prepare_china.py'), OLD / 'evidence/china_preparation.json']]
    for item in inputs[:3]:
        pass
    labels = pd.read_csv(LABELS, usecols=[KEY, 'Center', 'MACE'])
    reports, covariates = ([], [])
    for center in ('SCH', 'WHU'):
        print(json.dumps({'stage': 'prepare', 'center': center}), flush=True)
        datafile = OLD / f'private/prepared/{center}.npz'
        rolefile = OLD / f'private/runs/{center}_original/patient_roles.npz'
        inputs.extend([identity(datafile), identity(rolefile)])
        data = dict(np.load(datafile, allow_pickle=False))
        names = data['feature_names'].tolist()
        labidx = [i for i, name in enumerate(names) if name not in META]
        labs = [names[i] for i in labidx]
        assert len(labs) == 94 and len(META) == 10 and (set(META) < set(names))
        local = labels.loc[labels.Center.eq(center), [KEY, 'MACE']].reset_index(drop=True)
        assert np.array_equal(data['y'], local.MACE.to_numpy())
        raw = pd.read_csv(RAW[center], usecols=[KEY, TIME] + names, low_memory=False)
        assert set(raw[KEY]) == set(local[KEY]) and raw[KEY].notna().all()
        raw['_order'] = np.arange(len(raw))
        raw['_time'] = pd.to_datetime(raw[TIME], format='mixed', errors='coerce')
        bad_time = raw['_time'].isna()
        assert not bad_time.any(), f'Missing or invalid {center} check timestamps; stop and revise protocol before fitting'
        ties = raw.duplicated([KEY, '_time'], keep=False)
        for name in names:
            raw[name] = pd.to_numeric(raw[name], errors='coerce')
        ordered = raw.sort_values([KEY, '_time', '_order'], kind='stable')
        grouped = ordered.groupby(KEY, sort=True)
        mean = raw.groupby(KEY)[names].mean().reindex(local[KEY]).to_numpy(dtype=np.float32)
        assert np.array_equal(mean, data['X'], equal_nan=True), 'Rebuilt patient means differ from frozen original inputs'
        median_lab = raw.groupby(KEY)[labs].median().reindex(local[KEY]).to_numpy(dtype=np.float32)
        first_lab = ordered.drop_duplicates(KEY, keep='first').set_index(KEY)[labs].reindex(local[KEY]).to_numpy(dtype=np.float32)
        first_observed_lab = grouped[labs].first().reindex(local[KEY]).to_numpy(dtype=np.float32)
        matrices = {'mean': mean}
        for variant, lab in (('first_record', first_lab), ('first_observed', first_observed_lab), ('median', median_lab)):
            matrix = mean.copy()
            matrix[:, labidx] = lab
            assert not np.isinf(matrix).any()
            matrices[variant] = matrix
        counts = raw.groupby(KEY).size().reindex(local[KEY]).to_numpy(dtype=np.int64)
        duration = (grouped['_time'].max() - grouped['_time'].min()).dt.total_seconds().reindex(local[KEY]).to_numpy() / 86400
        earliest = grouped['_time'].transform('min')
        earliest_ties = ordered.loc[ordered['_time'].eq(earliest)].groupby(KEY).size().gt(1)
        roledata = dict(np.load(rolefile, allow_pickle=False))
        assert np.array_equal(roledata['y'], data['y']) and np.array_equal(roledata['groups'], data['groups'])
        output = ROOT / f'private/prepared/{center}.npz'
        with output.open('xb') as stream:
            np.savez_compressed(stream, **{'X_' + name: mat for name, mat in matrices.items()}, y=data['y'], groups=data['groups'], feature_names=data['feature_names'], lab_indices=np.asarray(labidx), roles=roledata['roles'], record_count=counts, recorded_span_days=duration)
        for variant, matrix in matrices.items():
            coverage = np.isfinite(matrix[:, labidx]).sum(axis=1)
            for stratum, mask in (('all', np.ones(len(counts), bool)), ('1', counts == 1), ('2-4', (counts >= 2) & (counts <= 4)), ('>=5', counts >= 5)):
                for label, lm in (('all', np.ones(len(counts), bool)), ('0', data['y'] == 0), ('1', data['y'] == 1)):
                    sel = mask & lm
                    covariates.append({'center': center, 'variant': variant, 'record_stratum': stratum, 'label_group': label, 'n': int(sel.sum()), 'positive': int(data['y'][sel].sum()), 'median_records': float(np.median(counts[sel])) if sel.any() else None, 'mean_observed_labs': float(coverage[sel].mean()) if sel.any() else None, 'median_observed_labs': float(np.median(coverage[sel])) if sel.any() else None, 'zero_observed_labs': int((coverage[sel] == 0).sum()), 'median_recorded_span_days': float(np.median(duration[sel])) if sel.any() else None})
        report = {'center': center, 'raw_records': len(raw), 'patients': len(local), 'positive': int(data['y'].sum()), 'timestamp_field': TIME, 'invalid_or_missing_timestamps': int(bad_time.sum()), 'records_in_duplicate_patient_timestamp_groups': int(ties.sum()), 'patients_with_timestamp_ties': int(raw.loc[ties, KEY].nunique()), 'patients_with_earliest_timestamp_ties': int(earliest_ties.sum()), 'timestamp_min_year': int(raw['_time'].dt.year.min()), 'timestamp_max_year': int(raw['_time'].dt.year.max()), 'record_count_quantiles': {str(q): float(np.quantile(counts, q)) for q in (0, 0.25, 0.5, 0.75, 0.95, 1)}, 'recorded_span_days_quantiles': {str(q): float(np.quantile(duration, q)) for q in (0, 0.25, 0.5, 0.75, 0.95, 1)}, 'fixed_nonlaboratory_fields': META, 'laboratory_features': labs, 'all_record_mean_exactly_matches_frozen_matrix': True, 'first_observed_and_mean_missing_masks_equal': bool(np.array_equal(np.isnan(first_observed_lab), np.isnan(mean[:, labidx]))), 'timestamp_semantics': 'Recorded check time only; not confirmed result-availability or MACE onset time.', 'tie_rule': 'Stable original CSV row position within identical patient/check-time; no endpoint or performance used.', 'output': identity(output)}
        reports.append(report)
        print(json.dumps({k: report[k] for k in ('center', 'raw_records', 'patients', 'positive', 'invalid_or_missing_timestamps', 'patients_with_timestamp_ties', 'patients_with_earliest_timestamp_ties')}), flush=True)
        del raw, ordered
    inputs_after = [identity(item['path']) for item in inputs]
    assert inputs_after == inputs
    pd.DataFrame(covariates).to_csv(ROOT / 'results/input_coverage_and_record_counts.csv', index=False)
    save_json(ROOT / 'preparation.json', {'passed': True, 'inputs_unchanged': True, 'inputs': inputs, 'cohorts': reports, 'script': identity(__file__), 'environment': {'python': sys.version, **{p: version(p) for p in ('pandas', 'numpy', 'scikit-learn', 'xgboost')}}, 'privacy': 'Private matrices contain clinical arrays but no original patient identifiers. Public statistics are aggregate only.'})
if __name__ == '__main__':
    main()
