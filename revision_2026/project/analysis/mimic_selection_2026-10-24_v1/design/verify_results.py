"""Independently verify saved selection models, roles and every metric draw."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT / 'private/run_v1'
TOL = 5e-12

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8')))

class RankMetrics:
    """Ascending tie-group pair-count AUC, separate from producer ROC integration."""

    def __init__(self, y, p):
        self.y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        self.order = np.argsort(p, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(p[self.order]) != 0) + 1]
        self.error = (self.y - p) ** 2

    def evaluate(self, weights):
        weights = np.asarray(weights, dtype=float)
        positive = np.add.reduceat((weights * self.y)[self.order], self.starts)
        negative = np.add.reduceat((weights * (1 - self.y))[self.order], self.starts)
        p_total, n_total = (positive.sum(), negative.sum())
        if p_total <= 0 or n_total <= 0:
            return np.full(3, np.nan)
        n_below = np.cumsum(negative) - negative
        auc = np.dot(positive, n_below + negative / 2) / (p_total * n_total)
        desc_p, desc_total = (positive[::-1], (positive + negative)[::-1])
        denominator = np.cumsum(desc_total)
        precision = np.divide(np.cumsum(desc_p), denominator, out=np.zeros_like(denominator), where=denominator > 0)
        ap = np.dot(desc_p, precision) / p_total
        return np.array([auc, ap, np.dot(weights, self.error) / weights.sum()])

def libraries(y, p):
    return np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)])

def main():
    destination = ROOT / 'results/independent_results_verification.json'
    assert not destination.exists(), 'Refuse to overwrite independent results'
    config = load(ROOT / 'protocol.json')
    lock = load(RUN / 'input_lock.json')
    complete = load(ROOT / 'results/training_complete.json')
    checks = []

    def check(condition, name):
        assert bool(condition), name
        checks.append({'name': name, 'passed': True})
    check(complete['completed'], 'producer training completed')
    check(identity(RUN / 'predictions.npz') == complete['predictions'], 'completed prediction identity')
    check(identity(RUN / 'input_lock.json') == complete['input_lock'], 'training lock identity')
    for item in lock['sources']:
        check(identity(item['path']) == item, 'unchanged training source: ' + Path(item['path']).name)
    check(config == lock['protocol'], 'frozen protocol unchanged')
    prep_check = load(HERE / 'independent_preparation_verification.json')
    matrix_path = ROOT / 'preparation/private/full_matrix.npz'
    check(prep_check['passed'] and identity(matrix_path) == prep_check['matrix_identity'], 'independently verified matrix unchanged')
    with np.load(matrix_path, allow_pickle=False) as z:
        original = {key: z[key].copy() for key in z.files}
    with np.load(RUN / 'predictions.npz', allow_pickle=False) as z:
        predictions = {key: z[key].copy() for key in z.files}
    with np.load(RUN / 'roles.npz', allow_pickle=False) as z:
        stored_roles = {key: z[key].copy() for key in z.files}
    X, y, groups = (original['X'], original['y'], original['groups'])
    names = original['feature_names'].tolist()
    check(X.shape == (11622, 41), 'primary matrix dimensions')
    check(not set(names) & {'AST', 'HDL Cholesterol', 'MACE', 'Admission_has_cardiac', 'HOSPITAL_EXPIRE_FLAG'}, 'explicit excluded fields absent')
    for field in ('y', 'groups'):
        check(np.array_equal(predictions[field], original[field]), 'prediction ' + field + ' unchanged')
        check(np.array_equal(stored_roles[field], original[field]), 'role ' + field + ' unchanged')
    masks = {'ever_all': np.ones(len(y), dtype=bool), 'ever_survivor': original['survivor'], 'prior_all': original['prior_completed_valid'], 'prior_survivor': original['prior_completed_valid'] & original['survivor']}
    for arm, mask in masks.items():
        check(np.array_equal(mask, predictions['mask_' + arm]), 'prediction mask independently rebuilt: ' + arm)
        check(np.isfinite(predictions['p_' + arm]).all() and ((predictions['p_' + arm] >= 0) & (predictions['p_' + arm] <= 1)).all(), 'probability range: ' + arm)
    patients = np.unique(groups)
    ordered_patients = np.random.default_rng(config['outer_seed']).permutation(patients)
    partition = np.array_split(ordered_patients, config['folds'])
    roles = np.full((config['folds'], len(y)), 2, dtype=np.int8)
    for f, test_patients in enumerate(partition, 1):
        available = np.setdiff1d(patients, test_patients)
        stop_order = np.random.default_rng(config['inner_seed_base'] + f).permutation(available)
        stop_patients = stop_order[:int(np.ceil(config['inner_stop_fraction'] * len(available)))]
        roles[f - 1, np.isin(groups, stop_patients)] = 3
        roles[f - 1, np.isin(groups, test_patients)] = 0
    check(np.array_equal(roles, stored_roles['roles']), 'all global outcome-blind roles independently reconstructed')
    check(np.all((roles == 0).sum(axis=0) == 1), 'exactly one global test fold per admission')
    role_reports = []
    replay_errors = []
    preprocessing_errors = []
    model_identities = []
    for f, role in enumerate(roles, 1):
        global_test = np.flatnonzero(role == 0)
        check(np.array_equal(predictions['fold'] == f, role == 0), 'prediction fold linkage ' + str(f))
        for arm in config['arms']:
            mask = masks[arm]
            ix = {'train': np.flatnonzero(mask & (role != 0)), 'inner': np.flatnonzero(mask & (role == 2)), 'stop': np.flatnonzero(mask & (role == 3)), 'test_own': np.flatnonzero(mask & (role == 0))}
            patient_sets = {name: set(groups[rows]) for name, rows in ix.items()}
            check(not set(groups[global_test]) & patient_sets['train'], 'global test patient isolation ' + arm + '/' + str(f))
            check(not patient_sets['inner'] & patient_sets['stop'], 'inner stopping patient isolation ' + arm + '/' + str(f))
            check(all((mask[rows].all() and len(np.unique(y[rows])) == 2 for rows in ix.values())), 'arm restriction and both classes ' + arm + '/' + str(f))
            folder = RUN / arm / ('fold_' + str(f))
            fit_report = load(folder / 'fit_report.json')
            inner_report = load(folder / 'inner_evaluation.json')
            package_path = folder / 'selection_package.joblib'
            booster_path = folder / 'selection_xgb.ubj'
            package = joblib.load(package_path)
            inner_prep = joblib.load(folder / 'inner_preprocessor.joblib')
            for prep, fitting_rows, label in ((package['preprocessor'], ix['train'], 'outer'), (inner_prep, ix['inner'], 'inner')):
                columns = np.flatnonzero(np.isfinite(X[fitting_rows]).any(axis=0))
                check(np.array_equal(columns, prep['columns']), 'training-only nonempty columns ' + label + '/' + arm + '/' + str(f))
                medians = np.nanmedian(X[fitting_rows][:, columns], axis=0).astype(float)
                error = float(np.max(np.abs(medians - prep['imputer'].statistics_)))
                preprocessing_errors.append(error)
                check(np.allclose(medians, prep['imputer'].statistics_, atol=1e-07, rtol=1e-07), 'training-only medians ' + label + '/' + arm + '/' + str(f))
            check(package['input_feature_names'] == names, 'saved feature order ' + arm + '/' + str(f))
            check(package['fitted_feature_names'] == [names[j] for j in package['preprocessor']['columns']], 'saved fitted feature order ' + arm + '/' + str(f))
            check(fit_report['fitted_features'] == package['fitted_feature_names'], 'published fitted feature names ' + arm + '/' + str(f))
            for rows, report, label in ((ix['train'], fit_report['parameters']['xgb_parameters'], 'outer'), (ix['inner'], inner_report['parameters'], 'inner')):
                expected_weight = float((y[rows] == 0).sum() / (y[rows] == 1).sum())
                check(report['scale_pos_weight'] == expected_weight and report['seed'] == f, 'training-only XGBoost weights and seed ' + label + '/' + arm + '/' + str(f))
                check(all((report[key] == config['xgb'][key] for key in ('max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'tree_method'))), 'fixed XGBoost parameters ' + label + '/' + arm + '/' + str(f))
            rf = package['rf']
            check(all((rf.get_params()[key] == value for key, value in config['rf'].items())) and rf.class_weight == 'balanced' and (rf.random_state == f), 'fixed random forest parameters ' + arm + '/' + str(f))
            booster = xgb.Booster()
            booster.load_model(booster_path)
            rounds = int(inner_report['best_iteration']) + 1
            check(rounds == fit_report['selected_rounds'] == booster.num_boosted_rounds() and 1 <= rounds <= config['xgb']['max_rounds'], 'selected stopping rounds refitted ' + arm + '/' + str(f))
            prep = package['preprocessor']
            values = prep['imputer'].transform(X[global_test][:, prep['columns']]).astype(np.float32)
            replay = 0.5 * booster.predict(xgb.DMatrix(values, nthread=2)) + 0.5 * rf.predict_proba(values)[:, 1]
            max_error = float(np.max(np.abs(replay - predictions['p_' + arm][global_test])))
            replay_errors.append(max_error)
            check(max_error < TOL, 'all saved model predictions replay ' + arm + '/' + str(f))
            role_reports.append({'arm': arm, 'fold': f, 'all_train_rows_in_arm': True, 'patient_overlap': 0, 'own_test_rows': len(ix['test_own']), 'global_test_rows': len(global_test), 'outer_train_rows': len(ix['train']), 'selected_rounds': rounds})
            model_identities.extend((identity(path) for path in (package_path, booster_path, folder / 'inner_preprocessor.joblib', folder / 'inner_evaluation.json', folder / 'fit_report.json')))
        print('Independently replayed all selection models for fold ' + str(f), flush=True)
    metrics_dir = ROOT / 'results/metrics'
    table = pd.read_csv(metrics_dir / 'all_performance.csv')
    paired = pd.read_csv(metrics_dir / 'paired_contrasts.csv')
    own = pd.read_csv(metrics_dir / 'own_cohort_performance.csv')
    check(len(own) == 12 and len(paired) == 12 and (len(table) == 24), 'expected own/common/paired table dimensions')
    errors = []
    library_errors = []
    table_errors = []
    verified_scalar_draws = 0
    targets = []
    for index, target in enumerate(config['arms']):
        selected = masks[target]
        target_y, target_groups = (y[selected], groups[selected])
        metric_file = metrics_dir / ('bootstrap_' + target + '.npz')
        with np.load(metric_file, allow_pickle=False) as z:
            cached = {key: z[key].copy() for key in z.files}
        model_names = sorted({target} | {item[key] for item in config['paired_contrasts'] if item['target'] == target for key in ('first', 'second')})
        check(model_names == cached['model_names'].tolist() and config['metrics'] == cached['metric_names'].tolist(), 'bootstrap model/metric identities ' + target)
        scores = [predictions['p_' + name][selected] for name in model_names]
        evaluators = [RankMetrics(target_y, score) for score in scores]
        points = np.array([metric.evaluate(np.ones(len(target_y))) for metric in evaluators])
        check(np.max(np.abs(points - cached['points'])) < TOL, 'independent point estimates ' + target)
        for j, score in enumerate(scores):
            library_errors.append(float(np.max(np.abs(points[j] - libraries(target_y, score)))))
        unique, inverse = np.unique(target_groups, return_inverse=True)
        rng = np.random.default_rng(config['bootstrap_seed_base'] + index)
        draws = np.full_like(cached['draws'], np.nan)
        check(draws.shape == (2000, len(model_names), 3), 'bootstrap dimensions ' + target)
        for b in range(2000):
            cluster_counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
            weights = cluster_counts[inverse]
            for j, metric in enumerate(evaluators):
                draws[b, j] = metric.evaluate(weights)
            if b in (0, 1, 17, 199, 997, 1999) and np.isfinite(draws[b]).all():
                repeated = np.repeat(np.arange(len(target_y)), weights)
                for j, score in enumerate(scores):
                    library_errors.append(float(np.max(np.abs(draws[b, j] - libraries(target_y[repeated], score[repeated])))))
        valid = np.isfinite(draws).all(axis=(1, 2))
        check(np.array_equal(valid, cached['valid']) and valid.sum() >= 1900, 'bootstrap validity flags ' + target)
        error = float(np.max(np.abs(draws[valid] - cached['draws'][valid])))
        errors.append(error)
        check(error < TOL, 'every bootstrap metric independently replayed ' + target)
        verified_scalar_draws += int(draws[valid].size)
        for j, name in enumerate(model_names):
            for k, metric in enumerate(config['metrics']):
                row = table[(table.target == target) & (table.model == name) & (table.metric == metric)]
                check(len(row) == 1, 'unique common-target table row ' + target + '/' + name + '/' + metric)
                expected = np.r_[points[j, k], np.quantile(draws[valid, j, k], [0.025, 0.975])]
                actual = row[['estimate', 'ci_low', 'ci_high']].iloc[0].to_numpy(float)
                table_errors.append(float(np.max(np.abs(actual - expected))))
                check(int(row.admissions.iloc[0]) == len(target_y) and int(row.patients.iloc[0]) == len(unique) and (int(row.positive_labels.iloc[0]) == int(target_y.sum())), 'common-target table denominators ' + target + '/' + name + '/' + metric)
                check(bool(row.own_cohort.iloc[0]) == (target == name), 'own-cohort flag ' + target + '/' + name + '/' + metric)
        for comparison in config['paired_contrasts']:
            if comparison['target'] != target:
                continue
            first, second = (model_names.index(comparison['first']), model_names.index(comparison['second']))
            for k, metric in enumerate(config['metrics']):
                row = paired[(paired.target == target) & (paired['first'] == comparison['first']) & (paired.second == comparison['second']) & (paired.metric == metric)]
                check(len(row) == 1, 'unique paired table row ' + target + '/' + comparison['first'] + '/' + metric)
                expected = np.r_[points[first, k] - points[second, k], np.quantile(draws[valid, first, k] - draws[valid, second, k], [0.025, 0.975])]
                actual = row[['estimate', 'ci_low', 'ci_high']].iloc[0].to_numpy(float)
                table_errors.append(float(np.max(np.abs(actual - expected))))
        targets.append({'target': target, 'admissions': len(target_y), 'patients': len(unique), 'positive_labels': int(target_y.sum()), 'draws': len(draws), 'valid_draws': int(valid.sum()), 'all_compared_models_share_exact_rows_and_weights': True, 'maximum_draw_error': error})
        print('Independently verified all bootstrap draws for ' + target, flush=True)
    check(max(library_errors) < TOL, 'independent metrics agree with sklearn including repeated sampled rows')
    check(max(table_errors) < TOL, 'all point estimates and percentile intervals in tables independently verified')
    check(own.to_json(orient='records') == table[table.own_cohort].reset_index(drop=True).to_json(orient='records'), 'own table matches own-cohort rows only')
    for item in lock['sources']:
        check(identity(item['path']) == item, 'training source unchanged after verification: ' + Path(item['path']).name)
    report = {'passed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'checks_passed': len(checks), 'checks': checks, 'script': identity(__file__), 'training_lock': identity(RUN / 'input_lock.json'), 'predictions': identity(RUN / 'predictions.npz'), 'protocol': identity(ROOT / 'protocol.json'), 'preparation_verification': identity(HERE / 'independent_preparation_verification.json'), 'all_training_source_hashes_unchanged': True, 'all_20_saved_models_replayed': True, 'replayed_model_admission_probabilities': 4 * len(y), 'maximum_saved_prediction_error': max(replay_errors), 'maximum_training_median_error': max(preprocessing_errors), 'model_identities': model_identities, 'role_checks': role_reports, 'bootstrap_targets': targets, 'independently_verified_bootstrap_metric_values': verified_scalar_draws, 'maximum_bootstrap_metric_error': max(errors), 'maximum_library_metric_error': max(library_errors), 'maximum_table_or_interval_error': max(table_errors), 'tolerance': TOL, 'environment': {'python': sys.version, 'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib')}}, 'models_retrained_by_verifier': False, 'new_heldout_label_flip_retraining_performed': False, 'private_rows_or_identifiers_logged': False, 'scope': 'Independent source-bound role and preprocessing checks, saved-model probability replay and independent rank/cluster-bootstrap calculation. This does not certify event timing, clinical labels, absence of all leakage or causal selection effects.'}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({'passed': True, 'checks_passed': len(checks), 'saved_models_replayed': 20, 'bootstrap_metric_values': verified_scalar_draws, 'max_probability_error': max(replay_errors), 'max_bootstrap_error': max(errors), 'max_interval_error': max(table_errors)}, indent=2))
if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        print(json.dumps({'passed': False, 'error_type': type(error).__name__, 'trace_lines': [frame.lineno for frame in traceback.extract_tb(error.__traceback__)], 'private_exception_values_suppressed': True}), file=sys.stderr)
        raise SystemExit(1)
