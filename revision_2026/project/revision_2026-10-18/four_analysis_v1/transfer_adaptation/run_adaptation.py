"""Frozen-source, patient-isolated target adaptation; aggregate reporting only."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import warnings
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xgboost as xgb
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[2] / 'revision_2026-10-16/reviewer_priority_analyses_v1/common_transfer'
CENTRES = ('WHU', 'MIMIC')
ADAPTERS = ('platt_unconstrained', 'logistic_update_source_transform', 'local_logistic_source_transform', 'local_logistic_target_transform')
METHODS = ('direct_source',) + ADAPTERS
METRICS = ('auc', 'average_precision', 'brier')
PAIRS = [(name, 'direct_source') for name in ADAPTERS] + [('logistic_update_source_transform', 'local_logistic_source_transform'), ('logistic_update_source_transform', 'local_logistic_target_transform'), ('local_logistic_target_transform', 'local_logistic_source_transform')]

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def read_json(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def load_data(centre):
    with np.load(OLD / f'private/{centre}.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    assert d['X'].shape == (len(d['y']), 4)
    assert np.isin(d['y'], [0, 1]).all()
    assert np.all(np.sum(d['roles'] == 0, axis=0) == 1)
    for role in d['roles']:
        assert not np.intersect1d(d['groups'][role == 0], d['groups'][role != 0]).size
    return d

def impute(X, medians):
    return np.where(np.isfinite(X), X, medians).astype(np.float64)

def medians(X):
    return np.asarray([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in X.T])

def logit(p):
    p = np.clip(p, 1e-06, 1 - 1e-06)
    return np.log(p) - np.log1p(-p)

def predict_source(package, X):
    package['rf'].n_jobs = 2
    values = impute(X, package['medians']).astype(np.float32)
    p = (package['xgb'].predict(xgb.DMatrix(values, nthread=2)) + package['rf'].predict_proba(values)[:, 1]) / 2
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    return p

def patient_info(y, groups):
    unique, inverse = np.unique(groups, return_inverse=True)
    labels = np.zeros(len(unique), dtype=np.int64)
    np.maximum.at(labels, inverse, y)
    return (unique, inverse, labels)

def count_role(y, groups, indices):
    unique, _, labels = patient_info(y[indices], groups[indices])
    return {'observations': int(len(indices)), 'patients': int(len(unique)), 'positive_observation_labels': int(y[indices].sum()), 'positive_patients_max_label': int(labels.sum())}

def sample_nested(y, groups, pool, target_index, fold, seed, sizes):
    unique, _, labels = patient_info(y[pool], groups[pool])
    rng = np.random.default_rng(np.random.SeedSequence([seed, target_index, fold]))
    pos = rng.permutation(unique[labels == 1])
    neg = rng.permutation(unique[labels == 0])
    draws = {}
    previous = np.empty(0, dtype=np.int64)
    for n in sizes:
        k = min(n - 1, max(1, round(n * labels.mean())))
        assert len(pos) >= k and len(neg) >= n - k
        selected = np.r_[pos[:k], neg[:n - k]]
        assert len(selected) == n and np.isin(previous, selected).all()
        indices = pool[np.isin(groups[pool], selected)]
        draws[n] = (indices, selected)
        previous = selected
    return draws

def fit_logistic(X, y, kind, settings):
    result = {'status': 'not_estimable', 'reason': None, 'estimator': None}
    if len(np.unique(y)) != 2:
        result['reason'] = 'single_training_class'
        return result
    if kind == 'platt_unconstrained':
        z = X[:, 0]
        if np.ptp(z) <= 1e-12:
            result['reason'] = 'constant_training_source_logit'
            return result
        if z[y == 1].min() >= z[y == 0].max() or z[y == 1].max() <= z[y == 0].min():
            result['reason'] = 'complete_or_quasi_separation_of_training_source_logit'
            return result
    estimator = LogisticRegression(C=np.inf if kind == 'platt_unconstrained' else 1.0, solver='lbfgs', max_iter=settings['max_iter'], tol=settings['tol'], class_weight=None, fit_intercept=True, l1_ratio=0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        estimator.fit(X, y)
    convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    result['warnings'] = [{'type': w.category.__name__, 'message': str(w.message)} for w in caught]
    if convergence or int(estimator.n_iter_.max()) >= settings['max_iter']:
        result['reason'] = 'training_nonconvergence'
        return result
    if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
        result['reason'] = 'nonfinite_training_coefficients'
        return result
    result.update(status='estimated', reason=None, estimator=estimator)
    return result

def design(method, X, source_p, source_medians, source_scaler, update_scaler, target_medians, target_scaler):
    source_values = impute(X, source_medians)
    if method == 'platt_unconstrained':
        return logit(source_p)[:, None]
    if method == 'logistic_update_source_transform':
        return update_scaler.transform(np.column_stack((logit(source_p), source_values)))
    if method == 'local_logistic_source_transform':
        return source_scaler.transform(source_values)
    if method == 'local_logistic_target_transform':
        return target_scaler.transform(impute(X, target_medians))
    raise ValueError(method)

def get_inputs():
    paths = [ROOT / 'protocol.json', Path(__file__), OLD / 'protocol.json', OLD / 'aggregate/preparation.json', OLD / 'aggregate/model_manifest.json', OLD / 'final_seal.json']
    for c in CENTRES:
        paths.extend((OLD / f'private/{c}.npz', OLD / f'private/{c}_predictions.npz'))
    manifest = read_json(OLD / 'aggregate/model_manifest.json')
    for expected in manifest['model_packages']:
        assert identity(expected['path']) == expected
        paths.append(Path(expected['path']))
    return [identity(p) for p in dict.fromkeys(paths)]

def train():
    started = time.monotonic()
    protocol = read_json(ROOT / 'protocol.json')
    assert not (ROOT / 'analysis_lock.json').exists(), 'Do not refit a locked run'
    inputs = get_inputs()
    (ROOT / 'private/models').mkdir(parents=True)
    (ROOT / 'aggregate').mkdir()
    write_json(ROOT / 'analysis_lock.json', {'locked_before_any_new_fitting': True, 'utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs, 'environment': {'python': platform.python_version(), 'executable': os.sys.executable, **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'joblib')}, 'threads_maximum': 2}})
    data = {c: load_data(c) for c in CENTRES}
    public, coefficients, package_ids, source_checks = ([], [], [], [])
    settings = protocol['logistic_settings']
    sizes = protocol['target_sampling']['patient_counts']
    seeds = protocol['target_sampling']['repeat_seeds']
    for target_index, target in enumerate(CENTRES):
        source = 'MIMIC' if target == 'WHU' else 'WHU'
        source_path = OLD / f'private/models/{source}_source_full_ensemble_median.joblib'
        package = joblib.load(source_path)
        d, s = (data[target], data[source])
        X, y, groups, roles = (d[k] for k in ('X', 'y', 'groups', 'roles'))
        direct = predict_source(package, X)
        with np.load(OLD / f'private/{target}_predictions.npz', allow_pickle=False) as z:
            old_direct = z['transfer_ensemble_median']
            assert np.array_equal(z['y'], y) and np.array_equal(z['groups'], groups)
        error = float(np.max(np.abs(direct - old_direct)))
        assert error < 5e-15
        source_values = impute(s['X'], package['medians'])
        source_scaler = StandardScaler().fit(source_values)
        source_p = predict_source(package, s['X'])
        update_scaler = StandardScaler().fit(np.column_stack((logit(source_p), source_values)))
        source_checks.append({'source': source, 'target': target, 'frozen_package': identity(source_path), 'old_transfer_probability_max_abs_difference': error, 'source_scaling_fitted_on_source_only': True, 'source_training': count_role(s['y'], s['groups'], np.arange(len(s['y']))), 'source_medians': package['medians'].tolist(), 'source_value_mean': source_scaler.mean_.tolist(), 'source_value_scale': source_scaler.scale_.tolist(), 'source_update_mean': update_scaler.mean_.tolist(), 'source_update_scale': update_scaler.scale_.tolist()})
        scores = np.full((len(sizes), len(seeds), len(ADAPTERS), len(y)), np.nan)
        failures = []
        for fold, role in enumerate(roles, 1):
            pool, test = (np.flatnonzero(role != 0), np.flatnonzero(role == 0))
            for repeat, seed in enumerate(seeds):
                draws = sample_nested(y, groups, pool, target_index, fold, seed, sizes)
                for size_index, n in enumerate(sizes):
                    selected, selected_groups = draws[n]
                    assert not np.intersect1d(groups[selected], groups[test]).size
                    local_medians = medians(X[selected])
                    local_scaler = StandardScaler().fit(impute(X[selected], local_medians))
                    fitted = {'source': source, 'target': target, 'fold': fold, 'sample_size_patients': n, 'repeat': repeat + 1, 'sampling_seed': seed, 'selected_rows': selected, 'selected_groups': selected_groups, 'test_rows': test, 'source_medians': package['medians'].copy(), 'source_scaler': source_scaler, 'update_scaler': update_scaler, 'target_medians': local_medians, 'target_scaler': local_scaler, 'models': {}}
                    for method_index, method in enumerate(ADAPTERS):
                        train_X = design(method, X[selected], direct[selected], package['medians'], source_scaler, update_scaler, local_medians, local_scaler)
                        estimator = fit_logistic(train_X, y[selected], method, settings)
                        if estimator['status'] == 'estimated':
                            test_X = design(method, X[test], direct[test], package['medians'], source_scaler, update_scaler, local_medians, local_scaler)
                            probabilities = estimator['estimator'].predict_proba(test_X)[:, 1]
                            if not np.isfinite(probabilities).all():
                                estimator.update(status='not_estimable', reason='nonfinite_outer_predictions', estimator=None)
                            else:
                                scores[size_index, repeat, method_index, test] = probabilities
                        fitted['models'][method] = estimator
                        if estimator['status'] != 'estimated':
                            failures.append({'fold': fold, 'repeat': repeat + 1, 'sample_size_patients': n, 'method': method, 'reason': estimator['reason']})
                        coefficient = {'source': source, 'target': target, 'fold': fold, 'repeat': repeat + 1, 'sample_size_patients': n, 'method': method, 'status': estimator['status'], 'reason': estimator['reason']}
                        if estimator['status'] == 'estimated':
                            model = estimator['estimator']
                            coefficient.update(intercept=float(model.intercept_[0]), coefficients=model.coef_[0].tolist(), iterations=int(model.n_iter_.max()))
                            if method == 'platt_unconstrained':
                                slope = float(model.coef_[0, 0])
                                coefficient.update(source_logit_slope=slope, negative_slope=bool(slope < 0), zero_slope=bool(slope == 0), ranking='reversed' if slope < 0 else 'preserved' if slope > 0 else 'constant')
                                observed_auc = roc_auc_score(y[test], probabilities)
                                expected_auc = roc_auc_score(y[test], -direct[test] if slope < 0 else direct[test])
                                if slope != 0:
                                    assert abs(observed_auc - expected_auc) < 5e-12
                                coefficient['monotone_outer_auc_abs_difference'] = float(abs(observed_auc - expected_auc))
                        coefficients.append(coefficient)
                    output = ROOT / f'private/models/{source}_to_{target}_fold{fold}_n{n}_repeat{repeat + 1}.joblib'
                    joblib.dump(fitted, output, compress=3)
                    package_ids.append(identity(output))
                    public.append({'source': source, 'target': target, 'fold': fold, 'repeat': repeat + 1, 'sampling_seed': seed, 'sample_size_patients': n, 'selected': count_role(y, groups, selected), 'pool': count_role(y, groups, pool), 'test': count_role(y, groups, test), 'target_all_empty_selected_columns': (~np.isfinite(X[selected]).any(axis=0)).tolist(), 'no_selected_test_patient_overlap': True, 'model_package': identity(output)})
            print(json.dumps({'phase': 'adaptation_fitting', 'source': source, 'target': target, 'fold_completed': fold, 'sizes': sizes, 'repeats': len(seeds)}), flush=True)
        output = ROOT / f'private/{target}_oof.npz'
        with output.open('xb') as stream:
            np.savez_compressed(stream, y=y, groups=groups, roles=roles, direct=direct, adapted=scores, coverage=np.isfinite(X).any(axis=1), sizes=np.asarray(sizes), seeds=np.asarray(seeds), methods=np.asarray(ADAPTERS), source=np.asarray(source))
        write_json(ROOT / f'aggregate/{target}_failure_status.json', {'source': source, 'target': target, 'fold_failures': failures, 'failed_fold_fits': len(failures), 'complete_oof_configurations': int(np.isfinite(scores).all(axis=-1).sum()), 'total_adaptation_configurations': int(np.prod(scores.shape[:-1]))})
    for expected in inputs:
        assert identity(expected['path']) == expected, expected['path']
    write_json(ROOT / 'aggregate/training_manifest.json', {'execution_complete': True, 'elapsed_seconds': time.monotonic() - started, 'inputs_unchanged': True, 'source_checks': source_checks, 'fold_draws': public, 'coefficient_records': coefficients, 'model_packages': package_ids, 'new_model_package_count': len(package_ids), 'new_adapter_fit_count': len(coefficients), 'private_rows_logged': False})
    pd.DataFrame([{k: v for k, v in row.items() if k != 'coefficients'} | {'coefficients': json.dumps(row.get('coefficients'))} for row in coefficients]).to_csv(ROOT / 'aggregate/calibration_and_update_coefficients.csv', index=False)
    print(json.dumps({'phase': 'fitting_complete', 'packages': len(package_ids), 'adapter_fits': len(coefficients), 'inputs_unchanged': True}), flush=True)

class MetricCalculator:

    def __init__(self, y, p):
        self.y, self.p = (y.astype(float), p.astype(float))
        self.order = np.argsort(p, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(p[self.order]) != 0) + 1]

    def compute(self, weights):
        weights = np.asarray(weights, dtype=float)
        pos = np.add.reduceat((weights * self.y)[self.order], self.starts)
        neg = np.add.reduceat((weights * (1 - self.y))[self.order], self.starts)
        total_pos, total_neg = (pos.sum(), neg.sum())
        if min(total_pos, total_neg) <= 0:
            return np.full(3, np.nan)
        auc = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (total_pos * total_neg)
        descending_pos, descending_neg = (pos[::-1], neg[::-1])
        tp = np.cumsum(descending_pos)
        denominator = tp + np.cumsum(descending_neg)
        precision = np.divide(tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
        ap = np.sum(descending_pos * precision) / total_pos
        brier = np.dot(weights, (self.p - self.y) ** 2) / weights.sum()
        return np.asarray([auc, ap, brier])

def tests():
    rng = np.random.default_rng(1928)
    checks = 0
    for n in (15, 100, 381):
        for tied in (True, False):
            y = rng.integers(0, 2, n)
            p = rng.random(n)
            if tied:
                p = np.round(p, 1)
            calc = MetricCalculator(y, p)
            for _ in range(10):
                weights = rng.multinomial(n, np.full(n, 1 / n))
                expected = np.asarray([roc_auc_score(y, p, sample_weight=weights), average_precision_score(y, p, sample_weight=weights), brier_score_loss(y, p, sample_weight=weights)])
                assert np.allclose(calc.compute(weights), expected, atol=3e-14, rtol=0)
                checks += 1
    settings = {'max_iter': 10000, 'tol': 1e-09}
    assert fit_logistic(np.ones((6, 1)), np.array([0, 0, 0, 1, 1, 1]), 'platt_unconstrained', settings)['reason'] == 'constant_training_source_logit'
    assert 'separation' in fit_logistic(np.arange(6.0)[:, None], np.array([0, 0, 0, 1, 1, 1]), 'platt_unconstrained', settings)['reason']
    assert fit_logistic(np.arange(6.0)[:, None], np.zeros(6), 'platt_unconstrained', settings)['reason'] == 'single_training_class'
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0])[:, None]
    yy = np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    smoke = fit_logistic(z, yy, 'platt_unconstrained', settings)
    assert smoke['status'] == 'estimated' and smoke['estimator'].coef_[0, 0] < 0
    smoke_regularized = fit_logistic(np.column_stack((z, z ** 2)), yy, 'local_logistic_target_transform', settings)
    assert smoke_regularized['status'] == 'estimated'
    return {'passed': True, 'weighted_sklearn_metric_checks': checks, 'integer_weights_ties_zero_weights': True, 'platt_constant_separation_single_class_prechecks': True, 'negative_platt_and_regularized_logistic_smoke': True}

def config_key(method, size, repeat):
    return f'{method}|{size}|{repeat}'

def evaluate():
    started = time.monotonic()
    protocol = read_json(ROOT / 'protocol.json')
    manifest = read_json(ROOT / 'aggregate/training_manifest.json')
    assert manifest['execution_complete']
    test_report = tests()
    metrics, pairs, averages, average_pairs, populations = ([], [], [], [], [])
    sizes, seeds = (protocol['target_sampling']['patient_counts'], protocol['target_sampling']['repeat_seeds'])
    B = protocol['evaluation']['bootstrap']['draws']
    for target_index, target in enumerate(CENTRES):
        with np.load(ROOT / f'private/{target}_oof.npz', allow_pickle=False) as z:
            d = {k: z[k].copy() for k in z.files}
        source = str(d['source'])
        all_configs = {config_key('direct_source', 0, 0): d['direct']}
        config_meta = {config_key('direct_source', 0, 0): ('direct_source', 0, 0)}
        for i, size in enumerate(sizes):
            for j in range(len(seeds)):
                for k, method in enumerate(ADAPTERS):
                    key = config_key(method, size, j + 1)
                    all_configs[key] = d['adapted'][i, j, k]
                    config_meta[key] = (method, size, j + 1)
        for population_index, population in enumerate(('all_original_target', 'at_least_one_accepted_lab')):
            keep = np.ones(len(d['y']), dtype=bool) if population_index == 0 else d['coverage'].astype(bool)
            y, groups = (d['y'][keep], d['groups'][keep])
            unique, inverse = np.unique(groups, return_inverse=True)
            info = {'source': source, 'target': target, 'population': population, **count_role(y, groups, np.arange(len(y)))}
            populations.append(info)
            keys = [k for k, p in all_configs.items() if np.isfinite(p).all()]
            calc = [MetricCalculator(y, all_configs[k][keep]) for k in keys]
            index = {key: i for i, key in enumerate(keys)}
            points = np.asarray([c.compute(np.ones(len(y))) for c in calc])
            draws = np.empty((B, len(keys), 3), dtype=np.float64)
            rng = np.random.default_rng(np.random.SeedSequence([protocol['evaluation']['bootstrap']['seed'], target_index, population_index]))
            for b in range(B):
                counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
                weight = counts[inverse].astype(float)
                draws[b] = np.asarray([c.compute(weight) for c in calc])
            assert np.isfinite(draws).all()
            with (ROOT / f'private/{target}_{population}_bootstrap.npz').open('xb') as stream:
                np.savez_compressed(stream, keys=np.asarray(keys), points=points, draws=draws)

            def make_rows(destination, metadata, point, sample):
                bounds = np.quantile(sample, [0.025, 0.975], axis=0)
                for m, name in enumerate(METRICS):
                    destination.append(info | metadata | {'metric': name, 'status': 'estimated', 'estimate': float(point[m]), 'ci_low': float(bounds[0, m]), 'ci_high': float(bounds[1, m]), 'bootstrap_draws': B, 'interval_scope': 'Nominal fixed-prediction patient-cluster percentile; conditional on source models, selected labelled target training patients and observed evaluation population'})
            for key, meta in config_meta.items():
                method, size, repeat = meta
                metadata = {'method': method, 'sample_size_patients_per_fold': size, 'repeat': repeat}
                if key in index:
                    i = index[key]
                    make_rows(metrics, metadata, points[i], draws[:, i])
                else:
                    for metric in METRICS:
                        metrics.append(info | metadata | {'metric': metric, 'status': 'not_estimable_in_all_outer_folds', 'estimate': None, 'ci_low': None, 'ci_high': None})
            for size in sizes:
                for repeat in range(1, len(seeds) + 1):
                    for first, second in PAIRS:
                        ka = config_key(first, size, repeat)
                        kb = config_key(second, 0, 0) if second == 'direct_source' else config_key(second, size, repeat)
                        if ka not in index or kb not in index:
                            continue
                        a, b = (index[ka], index[kb])
                        make_rows(pairs, {'first': first, 'second': second, 'sample_size_patients_per_fold': size, 'repeat': repeat, 'difference': 'first_minus_second'}, points[a] - points[b], draws[:, a] - draws[:, b])
                for method in ADAPTERS:
                    method_keys = [config_key(method, size, r) for r in range(1, len(seeds) + 1)]
                    if not all((k in index for k in method_keys)):
                        for metric in METRICS:
                            averages.append(info | {'method': method, 'sample_size_patients_per_fold': size, 'metric': metric, 'status': 'requires_three_complete_repeats', 'estimate': None})
                        continue
                    idx = [index[k] for k in method_keys]
                    make_rows(averages, {'method': method, 'sample_size_patients_per_fold': size, 'repeats': len(seeds), 'estimand': 'mean_of_repeat_metrics_not_probability_ensemble'}, points[idx].mean(axis=0), draws[:, idx].mean(axis=1))
                    for metric_index in range(3):
                        averages[-3 + metric_index].update(repeat_min=float(points[idx, metric_index].min()), repeat_max=float(points[idx, metric_index].max()))
                for first, second in PAIRS:
                    akeys = [config_key(first, size, r) for r in range(1, len(seeds) + 1)]
                    bkeys = [config_key(second, 0, 0)] * len(seeds) if second == 'direct_source' else [config_key(second, size, r) for r in range(1, len(seeds) + 1)]
                    if not all((k in index for k in akeys + bkeys)):
                        continue
                    ai, bi = ([index[k] for k in akeys], [index[k] for k in bkeys])
                    make_rows(average_pairs, {'first': first, 'second': second, 'sample_size_patients_per_fold': size, 'repeats': len(seeds), 'difference': 'mean_first_metric_minus_mean_second_metric'}, (points[ai] - points[bi]).mean(axis=0), (draws[:, ai] - draws[:, bi]).mean(axis=1))
            print(json.dumps({'phase': 'bootstrap_complete', 'source': source, 'target': target, 'population': population, 'score_sets': len(keys), 'draws': B}), flush=True)
    for name, records in (('metrics_by_repeat', metrics), ('paired_by_repeat', pairs), ('mean_repeat_metrics', averages), ('mean_repeat_paired', average_pairs), ('evaluation_populations', populations)):
        pd.DataFrame(records).to_csv(ROOT / f'aggregate/{name}.csv', index=False)
    coefficients = manifest['coefficient_records']
    estimated_platt = [r for r in coefficients if r['method'] == 'platt_unconstrained' and r['status'] == 'estimated']
    summary = {'execution_complete': True, 'elapsed_seconds': time.monotonic() - started, 'new_source_fitting': False, 'source_original_matrices_and_models_unchanged': True, 'target_patient_counts_per_fold': sizes, 'repeat_seeds': seeds, 'methods': list(METHODS), 'populations': populations, 'platt_estimated_fold_fits': len(estimated_platt), 'platt_negative_slope_fold_fits': sum((r['negative_slope'] for r in estimated_platt)), 'failed_adapter_fold_fits': [r for r in coefficients if r['status'] != 'estimated'], 'metrics_rows': len(metrics), 'paired_rows': len(pairs), 'mean_metric_rows': len(averages), 'mean_pair_rows': len(average_pairs), 'bootstrap_draws_per_population': B, 'mean_metrics': averages, 'mean_paired': average_pairs, 'self_tests': test_report, 'independent_package_replay_pending': True, 'no_probability_averaging_or_repeat_concatenation': True, 'clinical_minimum_training_size_estimated': False, 'limitations': ['Different native labels and patient/admission units', 'Retrospective all-record inputs, not pre-event prediction', 'Four name/context/physical-unit-reviewed labs, not complete assay or three-centre harmonization', 'Stratified samples from already labelled target training pools', 'Patient-cluster intervals conditional on trained models and selected samples; no refitting or selection uncertainty', 'Substantial source-availability confound, particularly WHU all-four-missing positives', 'Three draws per size are not independent cohorts or a sufficient learning curve for a clinical minimum']}
    write_json(ROOT / 'aggregate/summary.json', summary)
    lock = read_json(ROOT / 'analysis_lock.json')
    for expected in lock['inputs']:
        assert identity(expected['path']) == expected, expected['path']
    write_json(ROOT / 'aggregate/execution.json', {'execution_complete': True, 'fitting_complete': True, 'evaluation_complete': True, 'inputs_unchanged': True, 'input_identities': lock['inputs'], 'outputs': [identity(p) for p in sorted((ROOT / 'aggregate').glob('*')) if p.name != 'execution.json'], 'private_predictions': [identity(ROOT / f'private/{c}_oof.npz') for c in CENTRES], 'package_replay_pending': True, 'no_patient_rows_in_logs': True})
    print(json.dumps({'phase': 'evaluation_complete', 'metrics': len(metrics), 'mean_metrics': len(averages), 'paired': len(pairs), 'mean_paired': len(average_pairs)}), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('train', 'evaluate', 'all', 'tests'))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.phase in ('tests', 'all', 'train'):
            result = tests()
            print(json.dumps({'phase': 'self_tests', **result}), flush=True)
        if args.phase in ('train', 'all'):
            train()
        if args.phase in ('evaluate', 'all'):
            evaluate()
if __name__ == '__main__':
    main()
