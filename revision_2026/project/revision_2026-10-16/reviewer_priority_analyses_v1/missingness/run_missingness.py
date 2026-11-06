"""Frozen-label missingness sensitivities with paired patient-isolated evaluation.

Writes only below this new missingness directory. Historical preparation and
training modules are read and hashed but never imported or executed.
"""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
COHORTS = ('SCH', 'WHU', 'MIMIC')
MODELS = ('baseline_ensemble', 'value_indicator_ensemble', 'baseline_xgb', 'native_xgb')
METRICS = ('auc', 'average_precision', 'brier', 'e_over_o')
THREADS = 4

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return None

def identity(path):
    path = Path(path)
    return {'path': path.as_posix(), 'bytes': path.stat().st_size}

def read_json(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')

def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'logical_cpu_count': os.cpu_count(), 'max_threads': THREADS, 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib', 'scipy')}}

def check_identity(expected):
    actual = identity(expected['path'])
    return actual

def source_paths():
    paths = [BASE / name for name in ('protocol.json', 'run_validation.py', 'prepare_china.py', 'prepare_mimic.py', 'evidence/analysis_lock.json', 'evidence/china_preparation.json', 'evidence/mimic_preparation.json', 'results/validation_metrics.csv', 'results/validation_summary.json')]
    for cohort in COHORTS:
        paths += [BASE / 'private/prepared' / f'{cohort}.npz', BASE / 'private/runs' / f'{cohort}_original/predictions.npz', BASE / 'private/runs' / f'{cohort}_original/patient_roles.npz', BASE / 'results' / f'{cohort}_original_training.json']
        for fold in range(1, 6):
            directory = BASE / 'private/runs' / f'{cohort}_original/fold_{fold}'
            paths += [directory / 'full_xgb.ubj', directory / 'full_package.joblib']
    return paths

def freeze():
    assert ROOT == Path(_release_path('project', 'revision_2026-10-16/reviewer_priority_analyses_v1/missingness'))
    old_lock = read_json(BASE / 'evidence/analysis_lock.json')
    for old in old_lock['state']['files']:
        check_identity(old)
    protocol = read_json(ROOT / 'protocol.json')
    assert protocol['variants'] == ['value_indicator_ensemble', 'native_xgb']
    state = {'protocol': protocol, 'inherited_model_protocol': read_json(BASE / 'protocol.json'), 'sources': [identity(p) for p in source_paths()], 'environment': environment(), 'script': identity(__file__), 'protocol_identity': identity(ROOT / 'protocol.json')}
    path = ROOT / 'analysis_lock.json'
    if path.exists():
        assert read_json(path)['state'] == state, 'Locked source, code, environment or protocol changed'
    else:
        save_json(path, {'frozen_before_training_at': now(), 'state': state})
    return state

def load_cohort(cohort):
    training = read_json(BASE / 'results' / f'{cohort}_original_training.json')
    for key in ('input', 'roles_identity', 'prediction_identity'):
        check_identity(training[key])
    with np.load(BASE / 'private/prepared' / f'{cohort}.npz', allow_pickle=False) as z:
        X, y, groups = (z['X'].copy(), z['y'].copy(), z['groups'].copy())
        names = z['feature_names'].tolist()
    assert not np.isinf(X).any()
    with np.load(BASE / 'private/runs' / f'{cohort}_original/patient_roles.npz', allow_pickle=False) as z:
        assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        roles = z['roles'].copy()
    with np.load(BASE / 'private/runs' / f'{cohort}_original/predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        baseline = {'baseline_ensemble': z['primary_raw'].copy(), 'baseline_xgb': z['xgb_full'].copy()}
        folds = z['fold'].copy()
    expected = read_json(ROOT / 'protocol.json')['expected_counts'][cohort]
    assert [len(y), len(np.unique(groups)), int(y.sum()), len(names)] == expected
    assert roles.shape == (5, len(y)) and np.all((roles == 0).sum(axis=0) == 1)
    for fold, role in enumerate(roles, 1):
        sets = [set(groups[role == r]) for r in range(4)]
        assert all((not sets[a] & sets[b] for a in range(4) for b in range(a + 1, 4)))
        assert all((np.unique(y[role == r]).size == 2 for r in range(4)))
        assert np.array_equal(role == 0, folds == fold)
    return (X, y, groups, names, roles, folds, baseline)

def fit_preprocessor(X, cohort, variant):
    columns = np.arange(X.shape[1]) if cohort != 'MIMIC' else np.flatnonzero(np.isfinite(X).any(axis=0))
    assert len(columns)
    indicators = np.flatnonzero(np.isnan(X).any(axis=0) & np.isfinite(X).any(axis=0)) if variant == 'value_indicator_ensemble' else np.array([], dtype=np.int64)
    imputer = None
    if variant != 'native_xgb' and cohort == 'MIMIC':
        imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(X[:, columns])
    return {'columns': columns, 'indicator_columns': indicators, 'variant': variant, 'zero_fill': cohort != 'MIMIC', 'imputer': imputer}

def transform(prep, X):
    values = X[:, prep['columns']].copy()
    if prep['variant'] != 'native_xgb':
        values = np.nan_to_num(values, nan=0.0) if prep['zero_fill'] else prep['imputer'].transform(values).astype(np.float32)
    if len(prep['indicator_columns']):
        values = np.column_stack([values, np.isnan(X[:, prep['indicator_columns']]).astype(np.float32)])
    return values.astype(np.float32)

def fitted_names(prep, names):
    return [names[i] for i in prep['columns']] + ['missing::' + names[i] for i in prep['indicator_columns']]

def xgb_parameters(y, cohort, fold, config):
    p = config['xgb']
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': p['max_depth'], 'learning_rate': p['learning_rate'], 'subsample': p['subsample'], 'colsample_bytree': p['colsample_bytree'], 'scale_pos_weight': float((y == 0).sum() / (y == 1).sum()), 'seed': 42 if cohort != 'MIMIC' else fold, 'nthread': THREADS, 'verbosity': 0, 'tree_method': 'hist'}

def matrix(values, labels=None):
    return xgb.DMatrix(values, label=labels, nthread=THREADS, missing=np.nan)

def choose_rounds(X, y, inner, stop, cohort, fold, variant, config, dest):
    prep = fit_preprocessor(X[inner], cohort, variant)
    params = xgb_parameters(y[inner], cohort, fold, config)
    history = {}
    booster = xgb.train(params, matrix(transform(prep, X[inner]), y[inner]), num_boost_round=config['xgb']['max_rounds'], evals=[(matrix(transform(prep, X[stop]), y[stop]), 'inner_stop')], early_stopping_rounds=config['xgb']['early_stopping_rounds'], verbose_eval=False, evals_result=history)
    rounds = int(booster.best_iteration) + 1
    assert 1 <= rounds <= 300
    booster.save_model(dest / 'inner_xgb.ubj')
    joblib.dump(prep, dest / 'inner_preprocessor.joblib', compress=3)
    save_json(dest / 'inner_selection.json', {'selected_rounds': rounds, 'parameters': params, 'history': history, 'columns': prep['columns'].tolist(), 'indicator_columns': prep['indicator_columns'].tolist()})
    return rounds

def replay_baseline(cohort, fold, Xtest, expected_ensemble, expected_xgb):
    folder = BASE / 'private/runs' / f'{cohort}_original/fold_{fold}'
    package = joblib.load(folder / 'full_package.joblib')
    prep = package['preprocessor']
    values = Xtest[:, prep['columns']]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) if prep['zero_fill'] else prep['imputer'].transform(values).astype(np.float32)
    booster = xgb.Booster(params={'nthread': THREADS})
    booster.load_model(folder / 'full_xgb.ubj')
    xp = booster.predict(matrix(values))
    package['rf'].n_jobs = THREADS
    rp = package['rf'].predict_proba(values)[:, 1]
    ep = 0.5 * xp + 0.5 * rp
    error = max(float(np.max(np.abs(xp - expected_xgb))), float(np.max(np.abs(ep - expected_ensemble))))
    assert error < 1e-10, 'Frozen baseline prediction replay failed'
    return error

def run_cohort(cohort, state):
    dest = ROOT / 'private' / cohort
    assert not dest.exists(), f'Refusing to overwrite {cohort}'
    dest.mkdir(parents=True)
    X, y, groups, names, roles, folds, baseline = load_cohort(cohort)
    config = state['inherited_model_protocol']
    output = {key: np.full(len(y), np.nan, dtype=np.float64) for key in ('value_indicator_ensemble', 'value_indicator_xgb', 'value_indicator_rf', 'native_xgb')}
    reports = []
    started = time.monotonic()
    for fold, role in enumerate(roles, 1):
        test, full = (np.flatnonzero(role == 0), np.flatnonzero(role != 0))
        inner, stop = (np.flatnonzero(role == 2), np.flatnonzero(role == 3))
        replay_error = replay_baseline(cohort, fold, X[test], baseline['baseline_ensemble'][test], baseline['baseline_xgb'][test])
        for variant in state['protocol']['variants']:
            folder = dest / variant / f'fold_{fold}'
            folder.mkdir(parents=True)
            print(f'{cohort} {variant} fold {fold}/5', flush=True)
            rounds = choose_rounds(X, y, inner, stop, cohort, fold, variant, config, folder)
            prep = fit_preprocessor(X[full], cohort, variant)
            values = transform(prep, X[full])
            parameters = xgb_parameters(y[full], cohort, fold, config)
            booster = xgb.train(parameters, matrix(values, y[full]), num_boost_round=rounds, verbose_eval=False)
            test_matrix = matrix(transform(prep, X[test]))
            xp = booster.predict(test_matrix)
            rf = None
            rf_parameters = None
            if variant == 'value_indicator_ensemble':
                weight = {0: 1, 1: int((y[full] == 0).sum() / (y[full] == 1).sum())} if cohort != 'MIMIC' else 'balanced'
                rf = RandomForestClassifier(**config['rf'], class_weight=weight, random_state=42 if cohort != 'MIMIC' else fold, n_jobs=THREADS)
                rf.fit(values, y[full])
                rp = rf.predict_proba(transform(prep, X[test]))[:, 1]
                output[variant][test] = 0.5 * xp + 0.5 * rp
                output['value_indicator_xgb'][test] = xp
                output['value_indicator_rf'][test] = rp
                rf_parameters = rf.get_params()
            else:
                output[variant][test] = xp
            booster.save_model(folder / 'full_xgb.ubj')
            joblib.dump({'preprocessor': prep, 'rf': rf, 'fitted_feature_names': fitted_names(prep, names)}, folder / 'full_package.joblib', compress=3)
            contrib = booster.predict(test_matrix, pred_contribs=True)
            margin = booster.predict(test_matrix, output_margin=True)
            additivity_error = float(np.max(np.abs(contrib.sum(axis=1) - margin)))
            assert additivity_error < 0.0002
            np.savez_compressed(folder / 'heldout_xgb_contributions.npz', test_index=test, contributions=contrib, margin=margin, probability=xp, feature_names=np.asarray(fitted_names(prep, names) + ['BIAS'], dtype=str))
            report = {'cohort': cohort, 'variant': variant, 'fold': fold, 'selected_rounds': rounds, 'rows': {'test': len(test), 'full_outer_train': len(full), 'inner_fit': len(inner), 'inner_stop': len(stop)}, 'patients': {key: len(np.unique(groups[ix])) for key, ix in (('test', test), ('full_outer_train', full), ('inner_fit', inner), ('inner_stop', stop))}, 'positive_labels': {key: int(y[ix].sum()) for key, ix in (('test', test), ('full_outer_train', full), ('inner_fit', inner), ('inner_stop', stop))}, 'nominal_features': names, 'fitted_features': fitted_names(prep, names), 'indicator_feature_names': [names[i] for i in prep['indicator_columns']], 'xgb_parameters': parameters, 'rf_parameters': rf_parameters, 'frozen_baseline_maximum_replay_error': replay_error, 'xgb_heldout_contribution_maximum_margin_sum_error': additivity_error, 'contribution_scope': 'Outer-test XGBoost log-odds contribution only, not ensemble SHAP', 'model_identities': [identity(folder / f) for f in ('full_xgb.ubj', 'full_package.joblib', 'heldout_xgb_contributions.npz')]}
            save_json(folder / 'fit_report.json', report)
            reports.append(report)
    assert all((np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in output.values()))
    np.savez_compressed(dest / 'predictions.npz', y=y, groups=groups, fold=folds, **baseline, **output)
    report = {'cohort': cohort, 'execution_complete': True, 'rows': len(y), 'patients': len(np.unique(groups)), 'positive_labels': int(y.sum()), 'analysis_unit': 'admission' if cohort == 'MIMIC' else 'patient', 'seconds': time.monotonic() - started, 'input_identity': identity(BASE / 'private/prepared' / f'{cohort}.npz'), 'roles_identity': identity(BASE / 'private/runs' / f'{cohort}_original/patient_roles.npz'), 'prediction_identity': identity(dest / 'predictions.npz'), 'feature_names': names, 'missing_cells_before_any_imputation': int(np.isnan(X).sum()), 'true_zero_cells_not_reclassified_as_missing': int((X == 0).sum()), 'all_missing_nominal_columns': [n for j, n in enumerate(names) if np.isnan(X[:, j]).all()], 'fold_reports': reports}
    save_json(ROOT / 'results' / f'{cohort}_training.json', report)
    print(json.dumps({'completed': cohort, 'seconds': report['seconds'], 'rows': len(y)}), flush=True)

class WeightedMetrics:
    """Tie-aware fixed-score metrics, tested against sklearn with weights."""

    def __init__(self, y, score):
        self.y, self.score = (np.asarray(y, dtype=np.int64), np.asarray(score, dtype=np.float64))
        self.order = np.argsort(self.score, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.score[self.order]) != 0) + 1]
        self.ordered_y = self.y[self.order]
        self.error = (self.score - self.y) ** 2

    def evaluate(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        total, positive = (weights.sum(), np.sum(weights * self.y))
        if total <= 0:
            return np.full(4, np.nan)
        brier = float(np.sum(weights * self.error) / total)
        eo = float(np.sum(weights * self.score) / positive) if positive > 0 else np.nan
        if positive <= 0 or positive >= total:
            return np.array([np.nan, np.nan, brier, eo])
        ordered = weights[self.order]
        mass = np.add.reduceat(ordered, self.starts)
        pos = np.add.reduceat(ordered * self.ordered_y, self.starts)
        neg = mass - pos
        auc = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (positive * (total - positive))
        precision = np.divide(np.cumsum(pos[::-1]), np.cumsum(mass[::-1]), out=np.zeros_like(pos), where=np.cumsum(mass[::-1]) > 0)
        ap = np.sum(pos[::-1] * precision) / positive
        return np.array([auc, ap, brier, eo])

def sklearn_metrics(y, p, weights=None):
    weights = np.ones(len(y)) if weights is None else weights
    return np.array([roc_auc_score(y, p, sample_weight=weights), average_precision_score(y, p, sample_weight=weights), brier_score_loss(y, p, sample_weight=weights), np.sum(weights * p) / np.sum(weights * y)])

def self_test():
    X = np.array([[0.0, np.nan, np.nan, 2.0], [4.0, 3.0, np.nan, 5.0]], dtype=np.float32)
    for cohort in COHORTS:
        prep = fit_preprocessor(X, cohort, 'value_indicator_ensemble')
        assert prep['indicator_columns'].tolist() == [1]
        transformed = transform(prep, X)
        assert transformed[0, 0] == 0 and transformed[0, -1] == 1 and (transformed[1, -1] == 0)
        native = fit_preprocessor(X, cohort, 'native_xgb')
        assert np.isnan(transform(native, X)[0, 1])
        if cohort == 'MIMIC':
            assert prep['columns'].tolist() == [0, 1, 3]
            assert transformed[0, 1] == 3
        else:
            assert prep['columns'].tolist() == [0, 1, 2, 3]
            assert transformed[0, 1] == 0
    rng = np.random.default_rng(71716)
    cases, maximum_error = (0, 0.0)
    for n in (20, 91, 307):
        y = rng.integers(0, 2, n)
        y[:2] = [0, 1]
        for p in (rng.random(n), np.round(rng.random(n), 1), np.full(n, 0.37)):
            metric = WeightedMetrics(y, p)
            for _ in range(15):
                w = rng.integers(0, 5, n).astype(float)
                w[:2] = 1
                error = float(np.max(np.abs(metric.evaluate(w) - sklearn_metrics(y, p, w))))
                assert error < 2e-12
                maximum_error = max(maximum_error, error)
                cases += 1
    g = np.repeat(np.arange(10), [2, 1, 3, 1, 2, 2, 1, 3, 2, 1])
    y = np.arange(len(g)) % 2
    p = np.round(rng.random(len(g)), 1)
    metric = WeightedMetrics(y, p)
    for _ in range(50):
        weights = rng.multinomial(10, np.ones(10) / 10)[g]
        ix = np.repeat(np.arange(len(g)), weights)
        assert np.allclose(metric.evaluate(weights), sklearn_metrics(y[ix], p[ix]), atol=2e-12, rtol=0)
    return {'passed': True, 'weighted_sklearn_cases': cases, 'cluster_repeat_cases': 50, 'maximum_metric_error': maximum_error, 'real_zero_and_training_only_missing_indicator_tests': True}

def summarize(state):
    rows, paired, details = ([], [], [])
    tests = self_test()
    for cohort in COHORTS:
        report = read_json(ROOT / 'results' / f'{cohort}_training.json')
        check_identity(report['prediction_identity'])
        with np.load(ROOT / 'private' / cohort / 'predictions.npz', allow_pickle=False) as z:
            y, groups = (z['y'].copy(), z['groups'].copy())
            predictions = {m: z[m].copy() for m in MODELS}
        _, inverse = np.unique(groups, return_inverse=True)
        n_patients = int(inverse.max()) + 1
        context = {'cohort': cohort, 'analysis_unit': report['analysis_unit'], 'rows': len(y), 'patients': n_patients, 'positive_labels': int(y.sum())}
        caches = [WeightedMetrics(y, predictions[m]) for m in MODELS]
        points = np.array([sklearn_metrics(y, predictions[m]) for m in MODELS])
        assert np.allclose(points, [c.evaluate(np.ones(len(y))) for c in caches], atol=2e-12, rtol=0)
        repetitions = state['protocol']['bootstrap_replicates']
        samples = np.empty((repetitions, len(MODELS), len(METRICS)))
        rng = np.random.default_rng(state['protocol']['bootstrap_seed'])
        draw_hash = hashlib.sha256()
        for b in range(repetitions):
            multiplicity = rng.multinomial(n_patients, np.full(n_patients, 1 / n_patients))
            draw_hash.update(np.asarray(multiplicity, dtype='<i8').tobytes())
            weights = multiplicity[inverse]
            samples[b] = [c.evaluate(weights) for c in caches]
        assert np.isfinite(samples).all()
        for mi, model in enumerate(MODELS):
            for ki, metric in enumerate(METRICS):
                low, high = np.quantile(samples[:, mi, ki], [0.025, 0.975])
                rows.append({**context, 'model': model, 'metric': metric, 'estimate': float(points[mi, ki]), 'ci_low': float(low), 'ci_high': float(high), 'valid_bootstrap_replicates': repetitions})
        for first, second in state['protocol']['paired_comparisons']:
            a, b = (MODELS.index(first), MODELS.index(second))
            for ki, metric in enumerate(METRICS):
                low, high = np.quantile(samples[:, a, ki] - samples[:, b, ki], [0.025, 0.975])
                paired.append({**context, 'first': first, 'second': second, 'metric': 'delta_' + metric, 'estimate': float(points[a, ki] - points[b, ki]), 'ci_low': float(low), 'ci_high': float(high), 'valid_bootstrap_replicates': repetitions})
        old_summary = read_json(BASE / 'results/validation_summary.json')
        old_draw = next((x for x in old_summary['patient_bootstrap'] if x['cohort'] == cohort))
        details.append({**context, 'bootstrap_seed': state['protocol']['bootstrap_seed'], 'replicates': repetitions, 'same_draw_sequence_as_frozen_validation': True})
        print(f'{cohort}: 2000 patient-cluster paired draws completed', flush=True)
    metrics, comparisons = (pd.DataFrame(rows), pd.DataFrame(paired))
    historical = pd.read_csv(BASE / 'results/validation_metrics.csv')
    for cohort in COHORTS:
        for ours, original in (('baseline_ensemble', 'primary_raw'), ('baseline_xgb', 'xgb_full')):
            for metric in METRICS:
                a = metrics[(metrics.cohort == cohort) & (metrics.model == ours) & (metrics.metric == metric)].iloc[0]
                b = historical[(historical.run == f'{cohort}_original') & (historical.model == original) & (historical.metric == metric)].iloc[0]
                assert np.allclose(a[['estimate', 'ci_low', 'ci_high']].to_numpy(dtype=float), b[['estimate', 'ci_low', 'ci_high']].to_numpy(dtype=float), atol=5e-10, rtol=0)
    metrics.to_csv(ROOT / 'results/missingness_metrics.csv', index=False, mode='x', float_format='%.12g')
    comparisons.to_csv(ROOT / 'results/missingness_paired_comparisons.csv', index=False, mode='x', float_format='%.12g')
    save_json(ROOT / 'results/summary.json', {'execution_complete': True, 'protocol': state['protocol'], 'bootstrap': details, 'synthetic_tests': tests, 'metrics': rows, 'paired_comparisons': paired, 'baseline_metrics_and_intervals_match_frozen_validation': True, 'limits': ['Original full-record labels and inputs are retained. This is neither incident-event prediction nor external validation.', 'Feature missingness includes all nominal inputs and can reflect measurement, recording, coercion or structural availability; it is not a direct measure of laboratory ordering or clinical motivation.', 'Adding indicators alters candidate features and feature-sampling behavior; paired changes do not identify a causal effect of informative missingness.', 'Each variant independently uses the saved inner roles to select rounds. Differences include the missing-value handling and the resulting fitted trees/round counts.', 'MIMIC known mapping errors and survivor filtering are retained; native missing handling is not a mapping repair.', 'Intervals are nominal 95% percentile patient-cluster intervals conditional on fixed OOF predictions, without refitting or multiplicity adjustment.', 'No MICE or stable-coverage subset was fitted. Saved held-out XGBoost contributions are not an analysis of ensemble SHAP or causal importance.']})

def verify(state):
    replay = []
    for cohort in COHORTS:
        X, y, groups, names, roles, folds, baseline = load_cohort(cohort)
        public = read_json(ROOT / 'results' / f'{cohort}_training.json')
        check_identity(public['prediction_identity'])
        with np.load(ROOT / 'private' / cohort / 'predictions.npz', allow_pickle=False) as z:
            saved = {key: z[key].copy() for key in z.files}
        for key, target in (('y', y), ('groups', groups), ('fold', folds), *baseline.items()):
            assert np.array_equal(saved[key], target)
        for report in public['fold_reports']:
            fold, variant = (report['fold'], report['variant'])
            folder = ROOT / 'private' / cohort / variant / f'fold_{fold}'
            for item in report['model_identities']:
                check_identity(item)
            prep_package = joblib.load(folder / 'full_package.joblib')
            test = np.flatnonzero(roles[fold - 1] == 0)
            dm = matrix(transform(prep_package['preprocessor'], X[test]))
            booster = xgb.Booster(params={'nthread': THREADS})
            booster.load_model(folder / 'full_xgb.ubj')
            xp = booster.predict(dm)
            if variant == 'value_indicator_ensemble':
                prep_package['rf'].n_jobs = THREADS
                rp = prep_package['rf'].predict_proba(transform(prep_package['preprocessor'], X[test]))[:, 1]
                prediction = 0.5 * xp + 0.5 * rp
            else:
                prediction = xp
            error = float(np.max(np.abs(prediction - saved[variant][test])))
            assert error < 1e-10
            with np.load(folder / 'heldout_xgb_contributions.npz', allow_pickle=False) as contrib:
                assert np.array_equal(test, contrib['test_index'])
                assert np.array_equal(xp, contrib['probability'])
                assert np.max(np.abs(contrib['contributions'].sum(axis=1) - contrib['margin'])) < 0.0002
            replay.append({'cohort': cohort, 'variant': variant, 'fold': fold, 'maximum_prediction_replay_error': error})
    for expected in state['sources']:
        check_identity(expected)
    summary = read_json(ROOT / 'results/summary.json')
    assert summary['execution_complete'] and len(summary['metrics']) == 48 and (len(summary['paired_comparisons']) == 24)
    save_json(ROOT / 'verification.json', {'passed': True, 'execution_complete': True, 'checked_at': now(), 'scope': 'Saved model replay, input/role/label preservation, contribution additivity and aggregate validation; not clinical adjudication.', 'all_sources_unchanged': True, 'all_patient_roles_reused_exactly': True, 'all_30_outer_fold_models_replayed': True, 'fold_replay': replay, 'self_test': self_test(), 'summary_identity': identity(ROOT / 'results/summary.json'), 'metric_csv_identity': identity(ROOT / 'results/missingness_metrics.csv'), 'paired_csv_identity': identity(ROOT / 'results/missingness_paired_comparisons.csv'), 'analysis_lock_identity': identity(ROOT / 'analysis_lock.json'), 'environment': environment()})
    print(json.dumps({'passed': True, 'outer_fold_models': len(replay)}), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run', choices=(*COHORTS, 'all'))
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        state = freeze()
        if args.run:
            for cohort in COHORTS if args.run == 'all' else [args.run]:
                run_cohort(cohort, state)
        if args.summarize:
            summarize(state)
        if args.verify:
            verify(state)
