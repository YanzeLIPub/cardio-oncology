"""Independent model replay and sklearn bootstrap checks, without run imports."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import ast
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[3] / 'revision_2026-10-15/original_v5_independent_validation_v1'

def ident(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def read(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def measure(y, p, w):
    return [roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)]

def main():
    state = read(ROOT / 'analysis_lock.json')['state']
    protocol = state['protocol']
    for item in state['sources'] + [state['script']]:
        assert ident(item['path']) == item
    regional = None
    for node in ast.parse(_release_source_path(BASE / 'prepare_china.py').read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'REGIONAL_FIELDS' for t in node.targets)):
            regional = ast.literal_eval(node.value)
    metrics = pd.read_csv(ROOT / 'results/era_metrics.csv')
    paired = pd.read_csv(ROOT / 'results/era_paired_comparisons.csv')
    checks = []
    for cohort in protocol['cohorts']:
        with np.load(BASE / f'private/prepared/{cohort}.npz', allow_pickle=False) as z:
            X, y, names = (z['X'].copy(), z['y'].copy(), z['feature_names'].tolist())
        with np.load(ROOT / f'private/{cohort}_date_roles.npz', allow_pickle=False) as z:
            era = z['roles'].copy()
        with np.load(ROOT / f'private/{cohort}/roles.npz', allow_pickle=False) as z:
            ix, roles = (z['selected_index'].copy(), z['roles'].copy())
            assert np.array_equal(y[ix], z['y']) and np.array_equal(ix, z['groups'])
        assert np.array_equal(ix, np.flatnonzero(era >= 0))
        train = np.flatnonzero(era[ix] == 1)
        inner, stop = train_test_split(train, stratify=y[ix][train], test_size=0.25, random_state=2043)
        assert np.array_equal(np.sort(inner), np.flatnonzero(roles == 2))
        assert np.array_equal(np.sort(stop), np.flatnonzero(roles == 3))
        test = np.flatnonzero(roles == 0)
        assert np.all(era[ix][test] == 0)
        with np.load(ROOT / f'private/{cohort}/predictions.npz', allow_pickle=False) as z:
            predictions = {m: z[m].copy() for m in protocol['models']}
            assert np.array_equal(y[ix][test], z['y']) and np.array_equal(ix[test], z['groups'])
        for model in protocol['models']:
            cols = np.array([i for i, name in enumerate(names) if model == 'full_104' or (model == 'no_regional_99' and name not in regional) or (model == 'regional_only_5' and name in regional)])
            folder = ROOT / f'private/{cohort}/{model}'
            package = joblib.load(folder / 'full_package.joblib')
            if model == 'regional_only_5':
                cols = np.array([names.index(f) for f in regional])
            assert np.array_equal(cols, package['columns'])
            assert package['zero_fill'] and package['feature_names'] == [names[i] for i in cols]
            full = xgb.Booster(params={'nthread': 2})
            full.load_model(folder / 'full_xgb.ubj')
            v = np.nan_to_num(X[ix[test]][:, cols], nan=0.0).astype(np.float32)
            rf = package['rf']
            rf.n_jobs = 2
            p = 0.5 * full.predict(xgb.DMatrix(v, nthread=2)) + 0.5 * rf.predict_proba(v)[:, 1]
            assert np.allclose(p, predictions[model], atol=1e-12, rtol=0)
            expected_weight = {0: 1, 1: int((y[ix][train] == 0).sum() / (y[ix][train] == 1).sum())}
            assert rf.class_weight == expected_weight
            for key, expected in protocol['rf'].items():
                assert rf.get_params()[key] == expected
            fit = read(folder / 'fit_report.json')
            inner_model = xgb.Booster(params={'nthread': 2})
            inner_model.load_model(folder / 'inner_xgb.ubj')
            vm = np.nan_to_num(X[ix[stop]][:, cols], nan=0.0).astype(np.float32)
            matrix = xgb.DMatrix(vm, nthread=2)
            history = [roc_auc_score(y[ix][stop], inner_model.predict(matrix, iteration_range=(0, k))) for k in range(1, inner_model.num_boosted_rounds() + 1)]
            assert np.allclose(history, fit['inner_history']['inner_stop']['auc'], atol=1e-12, rtol=0)
            assert int(np.argmax(history)) + 1 == fit['selected_rounds'] == full.num_boosted_rounds()
            checks.append({'cohort': cohort, 'model': model, 'probabilities_exactly_replayed': True, 'all_inner_early_stopping_iterations_replayed': len(history), 'selected_rounds': fit['selected_rounds'], 'inputs': len(cols)})
        yy = y[ix][test]
        rng = np.random.default_rng(20261019)
        draws = np.empty((2000, 3, 3))
        for j in range(2000):
            w = rng.multinomial(len(yy), np.full(len(yy), 1 / len(yy)))
            draws[j] = [measure(yy, predictions[m], w) for m in protocol['models']]
        point = np.array([measure(yy, predictions[m], np.ones(len(yy))) for m in protocol['models']])
        for i, model in enumerate(protocol['models']):
            for k, name in enumerate(protocol['metrics']):
                rows = metrics.loc[metrics.cohort.eq(cohort) & metrics.model.eq(model) & metrics.metric.eq(name)]
                assert len(rows) == 1
                expected = [point[i, k], *np.quantile(draws[:, i, k], [0.025, 0.975])]
                assert np.allclose(rows[['estimate', 'ci_low', 'ci_high']].to_numpy()[0], expected, atol=5e-11, rtol=0)
        for first, second in protocol['paired_comparisons']:
            ai, bi = (protocol['models'].index(first), protocol['models'].index(second))
            for k, name in enumerate(protocol['metrics']):
                rows = paired.loc[paired.cohort.eq(cohort) & paired['first'].eq(first) & paired['second'].eq(second) & paired.metric.eq('delta_' + name)]
                assert len(rows) == 1
                expected = [point[ai, k] - point[bi, k], *np.quantile(draws[:, ai, k] - draws[:, bi, k], [0.025, 0.975])]
                assert np.allclose(rows[['estimate', 'ci_low', 'ci_high']].to_numpy()[0], expected, atol=5e-11, rtol=0)
        print(json.dumps({'cohort': cohort, 'passed': True, 'all_2000_sklearn_bootstrap_replicates_checked': True}), flush=True)
    for item in state['sources'] + [state['script']]:
        assert ident(item['path']) == item
    result = {'passed': True, 'script': ident(__file__), 'analysis_lock': ident(ROOT / 'analysis_lock.json'), 'six_full_models_replayed': True, 'six_inner_models_all_rounds_replayed': True, 'all_role_indices_and_feature_columns_reconstructed': True, 'all_2000_replicates_per_cohort_checked_directly_with_sklearn': True, 'metric_rows_checked': len(metrics), 'paired_rows_checked': len(paired), 'model_checks': checks, 'outputs': [ident(ROOT / 'results/era_metrics.csv'), ident(ROOT / 'results/era_paired_comparisons.csv'), ident(ROOT / 'results/summary.json')]}
    with (ROOT / 'independent_verification.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
if __name__ == '__main__':
    main()
