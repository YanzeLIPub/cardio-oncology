"""Train six locked acquisition-era diagnostic stress-test models."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import importlib.util
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
REGIONAL = _release_source_path(PROJECT / 'revision_2026-10-16/reviewer_priority_analyses_v1/regional/run_regional.py')
spec = importlib.util.spec_from_file_location('frozen_regional_functions', REGIONAL)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
identity, read, save = (common.identity, common.read, common.save)

def freeze():
    protocol = read(ROOT / 'protocol.json')
    paths = [ROOT / 'protocol.json', ROOT / 'context_audit.json', _release_source_path(ROOT / 'audit_context.py'), REGIONAL, BASE / 'protocol.json', _release_source_path(BASE / 'prepare_china.py')]
    for cohort in protocol['cohorts']:
        paths += [ROOT / f'private/{cohort}_date_roles.npz', BASE / f'private/prepared/{cohort}.npz']
    state = {'protocol': protocol, 'sources': [identity(p) for p in paths], 'script': identity(__file__), 'environment': {'python': sys.version, 'executable': sys.executable, **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'xgboost', 'scikit-learn', 'joblib')}}}
    save(ROOT / 'analysis_lock.json', {'frozen_before_training_at': datetime.now(timezone.utc).isoformat(), 'state': state})
    return state

def inputs(cohort):
    with np.load(BASE / f'private/prepared/{cohort}.npz', allow_pickle=False) as z:
        X, y, groups, names = (z['X'].copy(), z['y'].copy(), z['groups'].copy(), z['feature_names'].tolist())
    with np.load(ROOT / f'private/{cohort}_date_roles.npz', allow_pickle=False) as z:
        assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        era_roles = z['roles'].copy()
    return (X, y, groups, names, era_roles)

def train(state):
    for cohort in state['protocol']['cohorts']:
        X, y, groups, names, era = inputs(cohort)
        selected = np.flatnonzero(era >= 0)
        train = np.flatnonzero(era[selected] == 1)
        test = np.flatnonzero(era[selected] == 0)
        counts = state['protocol']['expected_counts'][cohort]
        assert [len(train), int(y[selected][train].sum()), len(test), int(y[selected][test].sum())] == [counts[k] for k in ('train', 'train_positive', 'test', 'test_positive')]
        inner, stop = train_test_split(train, test_size=0.25, stratify=y[selected][train], random_state=2043)
        roles = np.full(len(selected), -1, dtype=np.int8)
        roles[test], roles[inner], roles[stop] = (0, 2, 3)
        assert (roles >= 0).all()
        region = common.regional_fields()
        columns = {'full_104': np.arange(len(names)), 'no_regional_99': np.array([i for i, name in enumerate(names) if name not in region]), 'regional_only_5': np.array([names.index(f) for f in region])}
        dest = ROOT / f'private/{cohort}'
        dest.mkdir()
        np.savez_compressed(dest / 'roles.npz', selected_index=selected, groups=groups[selected], y=y[selected], roles=roles)
        predictions, reports = ({}, [])
        for model in state['protocol']['models']:
            print(json.dumps({'stage': 'fit', 'cohort': cohort, 'model': model}), flush=True)
            test_index, pred, report = common.fit_fold(X[selected], y[selected], groups[selected], names, roles, columns[model], dest / model, state['protocol'])
            assert np.array_equal(test_index, test)
            predictions[model] = pred
            reports.append({'model': model, **report})
        np.savez_compressed(dest / 'predictions.npz', groups=groups[selected][test], y=y[selected][test], **predictions)
        save(ROOT / f'results/{cohort}_training.json', {'execution_complete': True, 'cohort': cohort, 'fit_reports': reports, 'roles': identity(dest / 'roles.npz'), 'predictions': identity(dest / 'predictions.npz')})

def summarize(state):
    metrics, paired, details = ([], [], [])
    tests = common.self_test()
    models = state['protocol']['models']
    for cohort in state['protocol']['cohorts']:
        with np.load(ROOT / f'private/{cohort}/predictions.npz', allow_pickle=False) as z:
            y = z['y'].copy()
            ps = [z[m].copy() for m in models]
        points = np.array([common.metrics(y, p) for p in ps])
        cached = [common.WeightedMetrics(y, p) for p in ps]
        rng = np.random.default_rng(20261019)
        draws = np.empty((2000, len(models), 3))
        for b in range(2000):
            w = rng.multinomial(len(y), np.full(len(y), 1 / len(y)))
            draws[b] = [cache.evaluate(w) for cache in cached]
        valid = np.isfinite(draws).all(axis=(1, 2))
        assert valid.sum() >= 1900
        draws = draws[valid]
        context = {'cohort': cohort, 'test_patients': len(y), 'test_positive_labels': int(y.sum()), 'test_label_prevalence': float(y.mean()), 'valid_bootstrap_replicates': int(valid.sum()), 'cutoff': state['protocol']['cutoffs'][cohort]}
        for i, model in enumerate(models):
            for j, metric in enumerate(common.METRICS):
                lo, hi = np.quantile(draws[:, i, j], [0.025, 0.975])
                metrics.append({**context, 'model': model, 'metric': metric, 'estimate': float(points[i, j]), 'ci_low': float(lo), 'ci_high': float(hi)})
        for a, b in state['protocol']['paired_comparisons']:
            ai, bi = (models.index(a), models.index(b))
            for j, metric in enumerate(common.METRICS):
                lo, hi = np.quantile(draws[:, ai, j] - draws[:, bi, j], [0.025, 0.975])
                paired.append({**context, 'first': a, 'second': b, 'metric': 'delta_' + metric, 'estimate': float(points[ai, j] - points[bi, j]), 'ci_low': float(lo), 'ci_high': float(hi)})
        details.append(context)
        print(json.dumps({'stage': 'bootstrap_complete', **context}), flush=True)
    pd.DataFrame(metrics).to_csv(ROOT / 'results/era_metrics.csv', index=False, mode='x', float_format='%.12g')
    pd.DataFrame(paired).to_csv(ROOT / 'results/era_paired_comparisons.csv', index=False, mode='x', float_format='%.12g')
    save(ROOT / 'results/summary.json', {'execution_complete': True, 'protocol': state['protocol'], 'metrics': metrics, 'paired_comparisons': paired, 'bootstrap': details, 'self_test': tests, 'analysis_lock': identity(ROOT / 'analysis_lock.json')})

def verify(state):
    report = []
    for cohort in state['protocol']['cohorts']:
        X, y, groups, names, era = inputs(cohort)
        with np.load(ROOT / f'private/{cohort}/roles.npz', allow_pickle=False) as z:
            ix, roles = (z['selected_index'].copy(), z['roles'].copy())
        with np.load(ROOT / f'private/{cohort}/predictions.npz', allow_pickle=False) as z:
            stored = {k: z[k].copy() for k in z.files}
        assert np.array_equal(stored['groups'], groups[ix][roles == 0])
        assert np.array_equal(stored['y'], y[ix][roles == 0])
        assert np.all(era[ix][roles == 0] == 0) and np.all(era[ix][roles != 0] == 1)
        for model in state['protocol']['models']:
            folder = ROOT / f'private/{cohort}/{model}'
            fit = read(folder / 'fit_report.json')
            package = joblib.load(folder / 'full_package.joblib')
            for m in fit['model_identities']:
                assert identity(m['path']) == m
            v = common.values(X[ix][roles == 0], package['columns'])
            booster = xgb.Booster(params={'nthread': 2})
            booster.load_model(folder / 'full_xgb.ubj')
            package['rf'].n_jobs = 2
            p = 0.5 * booster.predict(common.dm(v)) + 0.5 * package['rf'].predict_proba(v)[:, 1]
            error = float(np.max(np.abs(p - stored[model])))
            assert error < 1e-10
            inner = xgb.Booster(params={'nthread': 2})
            inner.load_model(folder / 'inner_xgb.ubj')
            assert int(inner.attr('best_iteration')) + 1 == fit['selected_rounds'] == booster.num_boosted_rounds()
            report.append({'cohort': cohort, 'model': model, 'held_out_probability_max_error': error, 'selected_rounds': fit['selected_rounds'], 'nominal_inputs': len(package['columns'])})
    for source in state['sources']:
        assert identity(source['path']) == source
    assert identity(__file__) == state['script']
    save(ROOT / 'verification.json', {'passed': True, 'six_models_replayed': True, 'original_inputs_unchanged': True, 'role_and_model_checks': report, 'summary': identity(ROOT / 'results/summary.json'), 'self_test': common.self_test(), 'analysis_lock': identity(ROOT / 'analysis_lock.json')})
    print(json.dumps({'passed': True, 'verification': identity(ROOT / 'verification.json')}), flush=True)
if __name__ == '__main__':
    state = freeze()
    train(state)
    summarize(state)
    verify(state)
