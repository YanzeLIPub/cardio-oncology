"""Independent read-only package, sampling and aggregate replay for stage one."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'transfer_adaptation'
OLD = ROOT.parents[2] / 'revision_2026-10-16/reviewer_priority_analyses_v1/common_transfer'
METHODS = ('platt_unconstrained', 'logistic_update_source_transform', 'local_logistic_source_transform', 'local_logistic_target_transform')
METRICS = ('auc', 'average_precision', 'brier')

def identity(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def load(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def arr(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def median(X):
    return np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0 for x in X.T])

def fill(X, m):
    return np.where(np.isfinite(X), X, m).astype(np.float64)

def logits(p):
    p = np.clip(p, 1e-06, 1 - 1e-06)
    return np.log(p / (1 - p))

def source_predict(package, X):
    package['rf'].n_jobs = 2
    package['xgb'].set_param({'nthread': 2})
    values = fill(X, package['medians']).astype(np.float32)
    return 0.5 * package['xgb'].predict(xgb.DMatrix(values, nthread=2)) + 0.5 * package['rf'].predict_proba(values)[:, 1]

def scaler_equal(a, b):
    for name in ('mean_', 'scale_', 'var_', 'n_samples_seen_'):
        assert np.allclose(getattr(a, name), getattr(b, name), atol=2e-12, rtol=2e-12), name

def sklearn_metrics(y, p, w=None):
    return np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)])

class Scores:
    """Descending ROC trapezoids, independently of source pair-count AUC."""

    def __init__(self, y, p):
        self.y, self.p = (y, p.astype(float))
        self.order = np.argsort(-self.p, kind='stable')
        self.edges = np.r_[0, np.flatnonzero(np.diff(self.p[self.order])) + 1]

    def evaluate(self, w):
        a = np.add.reduceat((w * self.y)[self.order], self.edges)
        b = np.add.reduceat((w * (1 - self.y))[self.order], self.edges)
        tp, fp = (np.r_[0.0, np.cumsum(a)], np.r_[0.0, np.cumsum(b)])
        assert tp[-1] > 0 and fp[-1] > 0
        precision = np.divide(tp[1:], tp[1:] + fp[1:], out=np.zeros(len(a)), where=tp[1:] + fp[1:] > 0)
        return np.array([np.trapezoid(tp / tp[-1], fp / fp[-1]), np.sum(a * precision) / tp[-1], np.average((self.y - self.p) ** 2, weights=w)])

def main():
    protocol = load(SOURCE / 'protocol.json')
    manifest = load(SOURCE / 'aggregate/training_manifest.json')
    execution = load(SOURCE / 'aggregate/execution.json')
    replay = load(SOURCE / 'aggregate/saved_model_replay.json')
    sources = [identity(SOURCE / p) for p in ('protocol.json', 'run_adaptation.py', 'analysis_lock.json', 'aggregate/training_manifest.json', 'aggregate/execution.json', 'aggregate/saved_model_replay.json')]
    sources += load(SOURCE / 'analysis_lock.json')['inputs']
    sources += manifest['model_packages'] + execution['private_predictions'] + execution['outputs']
    for item in sources:
        assert identity(item['path']) == item
    datasets = {c: arr(OLD / f'private/{c}.npz') for c in ('WHU', 'MIMIC')}
    checks, max_probability_error, sklearn_checks = ([], 0.0, 0)
    package_checks = 0
    sizes = protocol['target_sampling']['patient_counts']
    seeds = protocol['target_sampling']['repeat_seeds']
    expected_tables = {name: pd.read_csv(SOURCE / f'aggregate/{name}.csv') for name in ('metrics_by_repeat', 'paired_by_repeat', 'mean_repeat_metrics', 'mean_repeat_paired')}
    checked_rows = {name: 0 for name in expected_tables}
    max_draw_error, max_csv_error = (0.0, 0.0)
    for target_index, target in enumerate(('WHU', 'MIMIC')):
        source = 'MIMIC' if target == 'WHU' else 'WHU'
        d, sd = (datasets[target], datasets[source])
        X, y, g, roles = [d[k] for k in ('X', 'y', 'groups', 'roles')]
        stored = arr(SOURCE / f'private/{target}_oof.npz')
        for name, value in (('y', y), ('groups', g), ('roles', roles)):
            assert np.array_equal(stored[name], value)
        assert np.array_equal(stored['coverage'], np.isfinite(X).any(axis=1))
        package = joblib.load(OLD / f'private/models/{source}_source_full_ensemble_median.joblib')
        assert np.allclose(package['medians'], median(sd['X']), atol=0, rtol=0)
        direct = source_predict(package, X)
        assert np.max(np.abs(direct - stored['direct'])) < 1e-14
        sm = StandardScaler().fit(fill(sd['X'], package['medians']))
        um = StandardScaler().fit(np.column_stack((logits(source_predict(package, sd['X'])), fill(sd['X'], package['medians']))))
        for fold, role in enumerate(roles, 1):
            pool, test = (np.flatnonzero(role != 0), np.flatnonzero(role == 0))
            unique, inv = np.unique(g[pool], return_inverse=True)
            labels = (np.bincount(inv, weights=y[pool]) > 0).astype(int)
            assert not set(g[pool]) & set(g[test])
            for ri, seed in enumerate(seeds):
                rng = np.random.default_rng(np.random.SeedSequence([seed, target_index, fold]))
                pos, neg = (rng.permutation(unique[labels == 1]), rng.permutation(unique[labels == 0]))
                last = set()
                for si, n in enumerate(sizes):
                    quota = min(n - 1, max(1, round(n * labels.mean())))
                    group_selection = np.r_[pos[:quota], neg[:n - quota]]
                    selected = pool[np.isin(g[pool], group_selection)]
                    assert last <= set(group_selection)
                    last = set(group_selection)
                    pack = joblib.load(SOURCE / f'private/models/{source}_to_{target}_fold{fold}_n{n}_repeat{ri + 1}.joblib')
                    assert np.array_equal(pack['selected_rows'], selected)
                    assert np.array_equal(pack['selected_groups'], group_selection)
                    assert np.array_equal(pack['test_rows'], test)
                    assert np.array_equal(pack['target_medians'], median(X[selected]))
                    assert np.array_equal(pack['source_medians'], package['medians'])
                    scaler_equal(pack['source_scaler'], sm)
                    scaler_equal(pack['update_scaler'], um)
                    tm = StandardScaler().fit(fill(X[selected], median(X[selected])))
                    scaler_equal(pack['target_scaler'], tm)
                    source_values = fill(X[test], package['medians'])
                    designs = [logits(direct[test])[:, None], um.transform(np.column_stack((logits(direct[test]), source_values))), sm.transform(source_values), tm.transform(fill(X[test], median(X[selected])))]
                    for mi, (method, design) in enumerate(zip(METHODS, designs)):
                        fit = pack['models'][method]
                        assert fit['status'] == 'estimated'
                        model = fit['estimator']
                        assert model.C == (np.inf if mi == 0 else 1.0) and model.class_weight is None
                        probability = expit(design @ model.coef_[0] + model.intercept_[0])
                        err = float(np.max(np.abs(probability - stored['adapted'][si, ri, mi, test])))
                        max_probability_error = max(max_probability_error, err)
                        assert err < 2e-12
                    package_checks += 1
        score_map = {'direct_source|0|0': direct}
        for si, n in enumerate(sizes):
            for ri in range(3):
                for mi, method in enumerate(METHODS):
                    score_map[f'{method}|{n}|{ri + 1}'] = stored['adapted'][si, ri, mi]
        for population_index, population in enumerate(('all_original_target', 'at_least_one_accepted_lab')):
            keep = np.ones(len(y), dtype=bool) if population_index == 0 else np.isfinite(X).any(axis=1)
            yy, gg = (y[keep], g[keep])
            unique, inv = np.unique(gg, return_inverse=True)
            saved = arr(SOURCE / f'private/{target}_{population}_bootstrap.npz')
            sources.append(identity(SOURCE / f'private/{target}_{population}_bootstrap.npz'))
            keys = saved['keys'].tolist()
            assert keys == list(score_map)
            caches = [Scores(yy, score_map[k][keep]) for k in keys]
            points = np.array([sklearn_metrics(yy, score_map[k][keep]) for k in keys])
            assert np.allclose(points, saved['points'], atol=2e-12, rtol=0)
            actual = np.empty_like(saved['draws'])
            rng = np.random.default_rng(np.random.SeedSequence([protocol['evaluation']['bootstrap']['seed'], target_index, population_index]))
            for bi in range(2000):
                mult = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
                weight = mult[inv]
                actual[bi] = [obj.evaluate(weight) for obj in caches]
                if bi < 10:
                    ref = np.array([sklearn_metrics(yy, score_map[k][keep], weight) for k in keys])
                    assert np.allclose(actual[bi], ref, atol=2e-12, rtol=0)
                    sklearn_checks += len(keys)
            error = float(np.max(np.abs(actual - saved['draws'])))
            max_draw_error = max(max_draw_error, error)
            assert error < 2e-12
            index = {k: i for i, k in enumerate(keys)}
            for table_name, table in expected_tables.items():
                subset = table[(table.target == target) & (table.population == population)]
                for _, row in subset.iterrows():
                    assert row.status == 'estimated'
                    n, ki = (int(row.sample_size_patients_per_fold), METRICS.index(row.metric))
                    repeats = [int(row['repeat'])] if table_name.endswith('by_repeat') else [1, 2, 3]
                    if table_name in ('metrics_by_repeat', 'mean_repeat_metrics'):
                        ix = [index[f'{row.method}|{n}|{r}'] for r in repeats]
                        point, draws = (points[ix, ki].mean(), actual[:, ix, ki].mean(axis=1))
                        if table_name == 'mean_repeat_metrics':
                            assert abs(points[ix, ki].min() - row.repeat_min) < 2e-12
                            assert abs(points[ix, ki].max() - row.repeat_max) < 2e-12
                    else:
                        ai = [index[f"{row['first']}|{n}|{r}"] for r in repeats]
                        bj = [index['direct_source|0|0' if row['second'] == 'direct_source' else f"{row['second']}|{n}|{r}"] for r in repeats]
                        point = (points[ai, ki] - points[bj, ki]).mean()
                        draws = (actual[:, ai, ki] - actual[:, bj, ki]).mean(axis=1)
                    low, high = np.quantile(draws, [0.025, 0.975])
                    err = max(abs(point - row.estimate), abs(low - row.ci_low), abs(high - row.ci_high))
                    max_csv_error = max(max_csv_error, float(err))
                    assert err < 2e-12
                    checked_rows[table_name] += 1
            checks.append({'target': target, 'population': population, 'rows': len(yy), 'patients': len(unique), 'positive_labels': int(yy.sum()), 'score_sets': len(keys), 'bootstrap_draws': 2000, 'max_bootstrap_metric_difference': error})
            print(f'Independent stage-one replay passed: {target}, {population}', flush=True)
    assert package_checks == 90
    assert checked_rows == {k: len(v) for k, v in expected_tables.items()}
    for item in sources:
        assert identity(item['path']) == item
    report = {'passed': True, 'completed_at': datetime.now(timezone.utc).isoformat(), 'reviewer': 'independent verbatim_r3_mapping implementation, no source-module import', 'script': identity(__file__), 'source_identities': list({s['path']: s for s in sources}.values()), 'model_packages_replayed': package_checks, 'adapter_predictions_replayed': package_checks * 4, 'nested_patient_samples_and_source_target_transforms_reconstructed': True, 'max_probability_replay_error': max_probability_error, 'populations': checks, 'all_reported_metric_and_interval_rows_checked': checked_rows, 'bootstrap_direct_sklearn_score_set_checks': sklearn_checks, 'max_bootstrap_metric_difference': max_draw_error, 'max_aggregate_csv_difference': max_csv_error, 'mean_of_three_repeat_metrics_not_mean_probabilities_verified': True, 'covered_rows_use_original_fixed_scores_without_refitting': True, 'source_files_unchanged': True, 'new_model_fits': 0, 'patient_identifiers_or_rows_logged': False, 'environment': {'python': sys.version, 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'joblib')}}, 'limits': ['Conditional fixed-score intervals do not include refitting or selection uncertainty.', 'This verifies stored fits and evaluation, not clinical endpoint or measurement equivalence.']}
    output = ROOT / 'qa/stage1_independent_verification.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'passed': True, 'report': identity(output)}))
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
