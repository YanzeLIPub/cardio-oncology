"""Original-record eligibility sensitivity, frozen before fit; local private outputs only."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import ast, os, sys, json, hashlib, time, platform, importlib.util, importlib.metadata
from datetime import datetime, timezone
PROJECT = Path(_release_path('project', ''))
DEST = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
EXEC = DEST / 'private/whu_source_eligible'
AGG = DEST / 'analysis'
THREADS = 8

def sha(p):
    return None

def ident(p):
    return {'path': str(p), 'bytes': Path(p).stat().st_size}

def save(p, v):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)

def read(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def norm(x):
    s = str(x).strip()
    return s[:-2] if s.endswith('.0') else s

def mod(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main():
    support = _release_source_path(PROJECT / 'audit/environment_rebuild_2026-10-27_v1/scripts/china_replay.py')
    funcs = [n for n in ast.parse(support.read_text(encoding='utf-8-sig')).body if isinstance(n, ast.FunctionDef) and n.name in ['require', 'set_resources']]
    ns = {'os': os, 'sys': sys}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(support), 'exec'), ns)
    resource = ns['set_resources'](THREADS)
    resource.pop('source_nthread_and_n_jobs_unchanged_minus_one', None)
    resource['actual_xgb_nthread_RF_njobs_DMatrix_threads'] = 8
    import numpy as np, pandas as pd
    from threadpoolctl import threadpool_info
    sys.stdout.reconfigure(encoding='utf-8')
    assert not EXEC.exists(), 'New run folder required'
    (EXEC / 'private/prepared').mkdir(parents=True)
    (EXEC / 'private/runs').mkdir()
    (EXEC / 'results').mkdir()
    idxpath = PROJECT / 'clinical_review_2026-10-27/full_index_v1/private/whu_index.csv'
    labelpath = Path(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'))
    rawpath = Path(_release_path('china', 'WHU_Cancer_processed_data.csv'))
    oldmatrix = SOURCE / 'private/prepared/WHU.npz'
    oldpred = SOURCE / 'private/runs/WHU_original/predictions.npz'
    sources = [idxpath, labelpath, rawpath, oldmatrix, oldpred, _release_source_path(SOURCE / 'run_validation.py'), _release_source_path(SOURCE / 'prepare_china.py'), SOURCE / 'protocol.json', SOURCE / 'evidence/analysis_lock.json', support, Path(__file__)]
    before = [ident(p) for p in sources]
    oldlock = read(SOURCE / 'evidence/analysis_lock.json')
    oldhash = {Path(x['path']).resolve(): None for x in oldlock['state']['files']}
    labels = pd.read_csv(labelpath, dtype={'病案号': str})
    labels = labels.loc[labels.Center.eq('WHU'), ['病案号', 'MACE']].reset_index(drop=True)
    labels['key'] = labels['病案号'].map(norm)
    idx = pd.read_csv(idxpath, dtype=str, keep_default_na=False)
    idx['key'] = idx.original_patient_id.map(norm)
    assert len(labels) == len(idx) == 2562 and labels.key.is_unique and idx.key.is_unique
    linked = labels.merge(idx, on='key', how='left', validate='one_to_one', sort=False)
    assert linked['病案号'].equals(labels['病案号']) and linked.review_token.notna().all()
    assert np.array_equal(linked.MACE.to_numpy(int), linked.original_label.astype(int))
    mask = linked.raw_record_count.astype(int).gt(0).to_numpy()
    assert mask.sum() == 2258 and (~mask).sum() == 304
    with np.load(oldmatrix, allow_pickle=False) as a:
        data = {k: a[k].copy() for k in a.files}
    names = data['feature_names'].tolist()
    raw = pd.read_csv(rawpath, usecols=['病案号'] + names, dtype={'病案号': str}, low_memory=False)
    raw['key'] = raw['病案号'].map(norm)
    assert set(raw.key) == set(labels.key)
    for c in names:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    means = raw.groupby('key')[names].mean()
    rebuilt = labels[['key']].merge(means, left_on='key', right_index=True, sort=False, validate='one_to_one')
    X = rebuilt[names].to_numpy(np.float32)
    assert np.array_equal(X, data['X'], equal_nan=True)
    assert np.array_equal(labels.MACE.to_numpy(int), data['y'])
    assert np.array_equal(data['groups'], np.arange(2562))
    retained = {k: v[mask] if k != 'feature_names' else v for k, v in data.items()}
    prepared = EXEC / 'private/prepared/WHU.npz'
    np.savez_compressed(prepared, **retained)
    np.savez_compressed(EXEC / 'eligibility_mask.npz', eligible=mask, original_groups=data['groups'], original_y=data['y'])
    linked[['review_token', 'key', 'original_label', 'raw_record_count']].assign(eligible=mask).to_csv(EXEC / 'private_patient_linkage.csv', index=False, encoding='utf-8-sig')
    config = read(SOURCE / 'protocol.json')
    config.update(version='WHU_original_record_eligible_patient_isolated_v1', original_labels_and_cohort_filters_retained=False, original_labels_retained=True, cohort_change='Exclude 304/2562 WHU patients without linked original raw JSON; retain all2258 with raw records regardless of event annotation', runs=[{'name': 'WHU_source_eligible', 'cohort': 'WHU', 'drop_features': []}], bootstrap_seed=20261028, no_event_annotation_policy='No positive match in the original specified diagnostic rule is algorithm-negative; no clinical absence or pre-event timing certified', other_centres='SCH38509 and MIMIC9946 have linked original diagnosis data; no eligibility exclusions or redundant refits', bootstrap_interpretation='Percentile95% patient intervals conditional on held-out probabilities; no training uncertainty')
    save(AGG / 'source_eligible_protocol.json', config)
    feature_rows = []
    flow = []
    for label, selection in [('all_original', np.ones(2562, dtype=bool)), ('retained', mask), ('excluded_no_raw_JSON', ~mask)]:
        xx = data['X'][selection]
        yy = data['y'][selection]
        flow.append({'cohort': 'WHU', 'scope': label, 'unit': 'patient', 'n': int(selection.sum()), 'algorithm_positive': int(yy.sum()), 'algorithm_negative': int((1 - yy).sum()), 'positive_fraction': float(yy.mean()), 'mean_feature_missing_fraction': float(np.isnan(xx).mean()), 'age_mean_from_original_processed_input': float(np.nanmean(xx[:, names.index('年龄')]))})
        for j, name in enumerate(names):
            v = xx[:, j]
            finite = v[np.isfinite(v)]
            feature_rows.append({'cohort': 'WHU', 'scope': label, 'feature': name, 'n': len(v), 'observed': len(finite), 'missing_fraction': float(np.isnan(v).mean()), 'mean': float(finite.mean()) if len(finite) else None, 'median': float(np.median(finite)) if len(finite) else None})
    pd.DataFrame(flow).to_csv(AGG / 'source_eligibility_flow.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(feature_rows).to_csv(AGG / 'source_eligibility_features.csv', index=False, encoding='utf-8-sig')
    lock = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'inputs': before, 'prepared': ident(prepared), 'config': config, 'resource': resource, 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': {k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'xgboost', 'joblib', 'threadpoolctl']}, 'threadpools': threadpool_info()}, 'exact_source_key_reconstruction': True, 'full_X_y_groups_equal_certified_matrix_before_subset': True, 'label_changes': 0, 'train_and_test_restricted_before_splits': True, 'flow': flow, 'subgroup_auc_not_computed_before_freeze': True}
    save(AGG / 'source_eligible_input_lock.json', lock)
    print(json.dumps({'stage': 'frozen', 'flow': flow, 'n_features': len(names)}, ensure_ascii=False), flush=True)
    run = mod('source_validation', _release_source_path(SOURCE / 'run_validation.py'))
    run.ROOT = EXEC
    run.run_one(config['runs'][0], config, THREADS, sha(AGG / 'source_eligible_input_lock.json'))
    assert before == [ident(p) for p in sources]
    training = read(EXEC / 'results/WHU_source_eligible_training.json')
    save(AGG / 'WHU_source_eligible_training.json', training)
    save(AGG / 'source_eligible_fit_complete.json', {'complete': True, 'source_inputs_unchanged': True, 'run_source_functions_unmodified': True, 'source_root_redirected_only': str(EXEC), 'input_key_bridge_exact': True, 'clinical_validation': False, 'forecast_validity': False})
if __name__ == '__main__':
    main()
