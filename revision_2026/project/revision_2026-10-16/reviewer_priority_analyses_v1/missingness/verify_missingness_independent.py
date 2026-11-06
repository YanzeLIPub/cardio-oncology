"""Read-only-input checks using independent preprocessing and sklearn metrics.

This script does not import the training/summary module. Its only output is
independent_verification.json in the new missingness analysis directory.
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2] / 'revision_2026-10-15/original_v5_independent_validation_v1'
MODELS = ('baseline_ensemble', 'value_indicator_ensemble', 'baseline_xgb', 'native_xgb')
METRICS = ('auc', 'average_precision', 'brier', 'e_over_o')

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def inspect_preprocessor(prep, Xfit, cohort, variant):
    expected_columns = np.arange(Xfit.shape[1]) if cohort != 'MIMIC' else np.flatnonzero(np.isfinite(Xfit).any(axis=0))
    expected_indicators = np.flatnonzero(np.isnan(Xfit).any(axis=0) & np.isfinite(Xfit).any(axis=0)) if variant == 'value_indicator_ensemble' else np.array([], dtype=int)
    assert np.array_equal(prep['columns'], expected_columns)
    assert np.array_equal(prep['indicator_columns'], expected_indicators)
    assert prep['variant'] == variant
    if cohort == 'MIMIC' and variant != 'native_xgb':
        median = np.nanmedian(Xfit[:, expected_columns], axis=0)
        assert np.allclose(prep['imputer'].statistics_, median, rtol=1e-07, atol=1e-07)
    else:
        assert prep['imputer'] is None
    return (expected_columns, expected_indicators)

def independent_transform(prep, Xfit, Xtest, cohort, variant):
    columns, indicators = inspect_preprocessor(prep, Xfit, cohort, variant)
    transformed = Xtest[:, columns].copy()
    if variant != 'native_xgb':
        replacement = np.zeros(len(columns)) if cohort != 'MIMIC' else np.nanmedian(Xfit[:, columns], axis=0)
        rows, cols = np.where(np.isnan(transformed))
        transformed[rows, cols] = replacement[cols]
    if len(indicators):
        transformed = np.column_stack([transformed, np.isnan(Xtest[:, indicators]).astype(np.float32)])
    return transformed.astype(np.float32)

def point(y, p, w):
    return [roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w), float(np.sum(w * p) / np.sum(w * y))]

def main():
    output = ROOT / 'independent_verification.json'
    assert not output.exists()
    lock = load(ROOT / 'analysis_lock.json')
    protocol = lock['state']['protocol']
    metrics = pd.read_csv(ROOT / 'results/missingness_metrics.csv')
    paired = pd.read_csv(ROOT / 'results/missingness_paired_comparisons.csv')
    checked = []
    bootstrap = []
    maximum_replay_error = 0.0
    for cohort in ('SCH', 'WHU', 'MIMIC'):
        with np.load(BASE / 'private/prepared' / f'{cohort}.npz', allow_pickle=False) as z:
            X, y, groups = (z['X'].copy(), z['y'].copy(), z['groups'].copy())
        with np.load(BASE / 'private/runs' / f'{cohort}_original/patient_roles.npz', allow_pickle=False) as z:
            roles = z['roles'].copy()
            assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        with np.load(ROOT / 'private' / cohort / 'predictions.npz', allow_pickle=False) as z:
            predictions = {name: z[name].copy() for name in MODELS}
            assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        for fold, role in enumerate(roles, 1):
            full, test, inner = [np.flatnonzero(role != 0), np.flatnonzero(role == 0), np.flatnonzero(role == 2)]
            assert not set(groups[full]) & set(groups[test])
            for variant in protocol['variants']:
                folder = ROOT / 'private' / cohort / variant / f'fold_{fold}'
                package = joblib.load(folder / 'full_package.joblib')
                inner_prep = joblib.load(folder / 'inner_preprocessor.joblib')
                inspect_preprocessor(inner_prep, X[inner], cohort, variant)
                values = independent_transform(package['preprocessor'], X[full], X[test], cohort, variant)
                selection = load(folder / 'inner_selection.json')
                assert selection['selected_rounds'] == int(np.argmax(selection['history']['inner_stop']['auc'])) + 1
                assert selection['parameters']['scale_pos_weight'] == float((y[inner] == 0).sum() / (y[inner] == 1).sum())
                booster = xgb.Booster(params={'nthread': 1})
                booster.load_model(folder / 'full_xgb.ubj')
                assert booster.num_boosted_rounds() == selection['selected_rounds']
                xp = booster.predict(xgb.DMatrix(values, nthread=1, missing=np.nan))
                if variant == 'value_indicator_ensemble':
                    rf = package['rf']
                    rf.n_jobs = 1
                    expected_weight = {0: 1, 1: int((y[full] == 0).sum() / (y[full] == 1).sum())} if cohort != 'MIMIC' else 'balanced'
                    assert rf.class_weight == expected_weight
                    prob = 0.5 * xp + 0.5 * rf.predict_proba(values)[:, 1]
                else:
                    prob = xp
                error = float(np.max(np.abs(prob - predictions[variant][test])))
                assert error < 1e-10
                maximum_replay_error = max(maximum_replay_error, error)
                checked.append({'cohort': cohort, 'variant': variant, 'fold': fold, 'train_only_indicator_and_imputer_verified': True, 'independent_numpy_preprocessing_replay_error': error})
        _, inverse = np.unique(groups, return_inverse=True)
        count = int(inverse.max()) + 1
        rng = np.random.default_rng(protocol['bootstrap_seed'])
        samples = np.empty((protocol['bootstrap_replicates'], 4, 4))
        for b in range(protocol['bootstrap_replicates']):
            weights = rng.multinomial(count, np.full(count, 1 / count))[inverse]
            samples[b] = [point(y, predictions[name], weights) for name in MODELS]
        max_interval_error = 0.0
        for mi, model in enumerate(MODELS):
            values = point(y, predictions[model], np.ones(len(y)))
            for ki, metric in enumerate(METRICS):
                expected = np.r_[values[ki], np.quantile(samples[:, mi, ki], [0.025, 0.975])]
                row = metrics[(metrics.cohort == cohort) & (metrics.model == model) & (metrics.metric == metric)].iloc[0]
                error = float(np.max(np.abs(expected - row[['estimate', 'ci_low', 'ci_high']].to_numpy(dtype=float))))
                assert error < 5e-10
                max_interval_error = max(error, max_interval_error)
        for first, second in protocol['paired_comparisons']:
            a, b = (MODELS.index(first), MODELS.index(second))
            for ki, metric in enumerate(METRICS):
                expected = np.quantile(samples[:, a, ki] - samples[:, b, ki], [0.025, 0.975])
                row = paired[(paired.cohort == cohort) & (paired['first'] == first) & (paired['second'] == second) & (paired.metric == 'delta_' + metric)].iloc[0]
                error = float(np.max(np.abs(expected - row[['ci_low', 'ci_high']].to_numpy(dtype=float))))
                assert error < 5e-10
                max_interval_error = max(error, max_interval_error)
        bootstrap.append({'cohort': cohort, 'direct_sklearn_bootstrap_replicates': len(samples), 'metrics_per_replicate': 16, 'maximum_csv_rounding_error': max_interval_error})
        print(f'{cohort}: independent 2000-draw sklearn interval replay passed', flush=True)
    for item in lock['state']['sources']:
        pass
    report = {'passed': True, 'execution_complete': True, 'training_module_imported': False, 'all_30_folds_independently_preprocessed_and_replayed': True, 'maximum_prediction_replay_error': maximum_replay_error, 'checks': checked, 'direct_sklearn_bootstrap': bootstrap, 'all_locked_sources_unchanged': True, 'script': identity(__file__), 'summary_identity': identity(ROOT / 'results/summary.json'), 'training_verification_identity': identity(ROOT / 'verification.json'), 'scope': 'Algorithmic and provenance verification, not clinical validation or causal inference.'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'passed': True, 'output': identity(output)}))
if __name__ == '__main__':
    main()
