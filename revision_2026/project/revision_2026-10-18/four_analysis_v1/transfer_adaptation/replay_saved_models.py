"""Standalone role, preprocessing, refit and probability replay; no writer imports."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
import os
from pathlib import Path
import time
import warnings
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import xgboost as xgb
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[2] / 'revision_2026-10-16/reviewer_priority_analyses_v1/common_transfer'
METHODS = ('platt_unconstrained', 'logistic_update_source_transform', 'local_logistic_source_transform', 'local_logistic_target_transform')

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def table(centre):
    with np.load(OLD / f'private/{centre}.npz', allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def source_predict(package, X):
    package['rf'].n_jobs = 2
    x = np.where(np.isfinite(X), X, package['medians']).astype(np.float32)
    return (package['xgb'].predict(xgb.DMatrix(x, nthread=2)) + package['rf'].predict_proba(x)[:, 1]) / 2

def clipped_logit(p):
    return np.log(np.clip(p, 1e-06, 1 - 1e-06) / (1 - np.clip(p, 1e-06, 1 - 1e-06)))

def direct_medians(X):
    return np.asarray([np.sort(c[np.isfinite(c)])[len(c[np.isfinite(c)]) // 2] if len(c[np.isfinite(c)]) % 2 else (np.sort(c[np.isfinite(c)])[len(c[np.isfinite(c)]) // 2 - 1] + np.sort(c[np.isfinite(c)])[len(c[np.isfinite(c)]) // 2]) / 2 if np.isfinite(c).any() else 0.0 for c in X.T])

def check_scaler(scaler, values):
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1
    assert np.allclose(mean, scaler.mean_, atol=2e-12, rtol=0)
    assert np.allclose(scale, scaler.scale_, atol=2e-12, rtol=0)

def main():
    start = time.monotonic()
    protocol = load(ROOT / 'protocol.json')
    lock = load(ROOT / 'analysis_lock.json')
    manifest = load(ROOT / 'aggregate/training_manifest.json')
    assert manifest['execution_complete'] and len(manifest['model_packages']) == 90
    for expected in lock['inputs'] + manifest['model_packages']:
        assert identity(expected['path']) == expected
    d = {c: table(c) for c in ('WHU', 'MIMIC')}
    score = {}
    for centre in ('WHU', 'MIMIC'):
        with np.load(ROOT / f'private/{centre}_oof.npz', allow_pickle=False) as z:
            score[centre] = {k: z[k].copy() for k in z.files}
    source_info = {}
    direct_error = 0.0
    for target in ('WHU', 'MIMIC'):
        source = 'MIMIC' if target == 'WHU' else 'WHU'
        package = joblib.load(OLD / f'private/models/{source}_source_full_ensemble_median.joblib')
        median = direct_medians(d[source]['X'])
        assert np.allclose(median, package['medians'], atol=1e-12, rtol=0)
        target_p = source_predict(package, d[target]['X'])
        e = float(np.max(np.abs(target_p - score[target]['direct'])))
        assert e < 5e-15
        direct_error = max(direct_error, e)
        source_values = np.where(np.isfinite(d[source]['X']), d[source]['X'], median)
        source_p = source_predict(package, d[source]['X'])
        source_info[target] = {'median': median, 'values': source_values, 'update': np.column_stack((clipped_logit(source_p), source_values)), 'p': target_p}
        for k in ('y', 'groups', 'roles'):
            assert np.array_equal(d[target][k], score[target][k])
        assert np.array_equal(score[target]['coverage'], np.isfinite(d[target]['X']).any(axis=1))
    max_score_error = max_coef_error = max_refit_probability_error = 0.0
    row_scores = fits = negative_slopes = failure_checks = 0
    packages_checked = []
    sizes = protocol['target_sampling']['patient_counts']
    for expected in manifest['model_packages']:
        package = joblib.load(expected['path'])
        target, fold = (package['target'], package['fold'])
        y, groups, X, roles = (d[target][k] for k in ('y', 'groups', 'X', 'roles'))
        target_index = ('WHU', 'MIMIC').index(target)
        pool, test = (np.flatnonzero(roles[fold - 1] != 0), np.flatnonzero(roles[fold - 1] == 0))
        unique, inv = np.unique(groups[pool], return_inverse=True)
        gy = np.zeros(len(unique), dtype=int)
        np.maximum.at(gy, inv, y[pool])
        rng = np.random.default_rng(np.random.SeedSequence([package['sampling_seed'], target_index, fold]))
        positive, negative = (rng.permutation(unique[gy == 1]), rng.permutation(unique[gy == 0]))
        n = package['sample_size_patients']
        positive_n = min(n - 1, max(1, round(n * gy.mean())))
        selected_groups = np.r_[positive[:positive_n], negative[:n - positive_n]]
        selected = pool[np.isin(groups[pool], selected_groups)]
        assert np.array_equal(selected_groups, package['selected_groups'])
        assert np.array_equal(selected, package['selected_rows']) and np.array_equal(test, package['test_rows'])
        assert not np.intersect1d(groups[selected], groups[test]).size
        if n != sizes[0]:
            previous_n = sizes[sizes.index(n) - 1]
            previous_p = min(previous_n - 1, max(1, round(previous_n * gy.mean())))
            previous = np.r_[positive[:previous_p], negative[:previous_n - previous_p]]
            assert np.isin(previous, selected_groups).all()
        source = source_info[target]
        assert np.allclose(package['source_medians'], source['median'], atol=1e-12, rtol=0)
        check_scaler(package['source_scaler'], source['values'])
        check_scaler(package['update_scaler'], source['update'])
        local_medians = direct_medians(X[selected])
        assert np.allclose(local_medians, package['target_medians'], atol=1e-12, rtol=0)
        local_values = np.where(np.isfinite(X), X, local_medians)
        check_scaler(package['target_scaler'], local_values[selected])
        source_values = np.where(np.isfinite(X), X, source['median'])
        designs = {'platt_unconstrained': clipped_logit(source['p'])[:, None], 'logistic_update_source_transform': (np.column_stack((clipped_logit(source['p']), source_values)) - package['update_scaler'].mean_) / package['update_scaler'].scale_, 'local_logistic_source_transform': (source_values - package['source_scaler'].mean_) / package['source_scaler'].scale_, 'local_logistic_target_transform': (local_values - package['target_scaler'].mean_) / package['target_scaler'].scale_}
        for mi, method in enumerate(METHODS):
            fitted = package['models'][method]
            refit_X = designs[method][selected]
            if fitted['status'] != 'estimated':
                assert np.isnan(score[target]['adapted'][sizes.index(n), package['repeat'] - 1, mi, test]).all()
                failure_checks += 1
                continue
            model = fitted['estimator']
            p = model.predict_proba(designs[method][test])[:, 1]
            frozen = score[target]['adapted'][sizes.index(n), package['repeat'] - 1, mi, test]
            error = float(np.max(np.abs(p - frozen)))
            assert error < 5e-13
            max_score_error = max(max_score_error, error)
            reference = LogisticRegression(C=np.inf if method == 'platt_unconstrained' else 1.0, solver='lbfgs', tol=1e-09, max_iter=10000, class_weight=None, fit_intercept=True, l1_ratio=0.0)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                reference.fit(refit_X, y[selected])
            assert not any((issubclass(w.category, ConvergenceWarning) for w in observed))
            coefficient_error = max(float(np.max(np.abs(reference.coef_ - model.coef_))), float(np.max(np.abs(reference.intercept_ - model.intercept_))))
            assert coefficient_error < 2e-06
            refit_error = float(np.max(np.abs(reference.predict_proba(designs[method][test])[:, 1] - p)))
            assert refit_error < 2e-07
            max_coef_error = max(max_coef_error, coefficient_error)
            max_refit_probability_error = max(max_refit_probability_error, refit_error)
            if method == 'platt_unconstrained':
                slope = model.coef_[0, 0]
                negative_slopes += int(slope < 0)
                rank_auc = roc_auc_score(y[test], -source['p'][test] if slope < 0 else source['p'][test])
                assert abs(roc_auc_score(y[test], p) - rank_auc) < 5e-12
            row_scores += len(test)
            fits += 1
        packages_checked.append(expected)
    for expected in lock['inputs'] + manifest['model_packages']:
        assert identity(expected['path']) == expected
    report = {'passed': True, 'execution_complete': True, 'elapsed_seconds': time.monotonic() - start, 'script': identity(__file__), 'no_writer_imports': True, 'source_median_and_scaler_recomputed': True, 'target_selected_patient_draws_and_nesting_replayed': True, 'no_target_test_patient_overlap': True, 'target_local_medians_and_scaler_recomputed_from_selected_only': True, 'source_direct_max_abs_probability_difference': direct_error, 'saved_adapter_package_count': len(packages_checked), 'estimated_adapter_refits': fits, 'failed_fold_prechecks_replayed': failure_checks, 'held_out_row_scores_replayed': row_scores, 'saved_adapter_probability_max_abs_error': max_score_error, 'refit_coefficient_max_abs_error': max_coef_error, 'refit_heldout_probability_max_abs_error': max_refit_probability_error, 'negative_platt_slopes_verified': negative_slopes, 'old_inputs_and_all_new_packages_unchanged': True, 'scope': 'Standalone source/roles/preprocessing/estimator/probability replay. Bootstrap/metric numerical verification is separate; no clinical task equivalence is inferred.'}
    (ROOT / 'aggregate/saved_model_replay.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'replay_passed': True, 'packages': len(packages_checked), 'adapter_refits': fits, 'held_out_scores': row_scores, 'negative_platt_slopes': negative_slopes}))
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
