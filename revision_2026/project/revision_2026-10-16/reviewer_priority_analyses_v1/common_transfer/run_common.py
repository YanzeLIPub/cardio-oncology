"""Source-frozen limited laboratory transfer and patient-isolated local references."""
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
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '4'
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xgboost as xgb
from prepare_common import identity, write_json
ROOT = Path(__file__).resolve().parent
MODELS = ['ensemble_median', 'ensemble_median_indicator', 'logistic', 'missingness_only']
THREADS = 4

def hash_array(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()

def fitting_medians(X):
    return np.asarray([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0 for column in X.T], dtype=np.float64)

def transform(X, medians, model):
    missing = ~np.isfinite(X)
    if model == 'missingness_only':
        return missing.astype(np.float32)
    numeric = np.where(missing, medians, X)
    if model == 'ensemble_median_indicator':
        numeric = np.column_stack((numeric, missing.astype(np.float64)))
    return numeric.astype(np.float32)

def xgb_params(y, seed, protocol):
    spec = protocol['xgboost']
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': spec['tree_method'], 'max_depth': spec['max_depth'], 'learning_rate': spec['learning_rate'], 'subsample': spec['subsample'], 'colsample_bytree': spec['colsample_bytree'], 'scale_pos_weight': float((y == 0).sum() / (y == 1).sum()), 'seed': seed, 'nthread': THREADS, 'verbosity': 0}

def select_rounds(X, y, inner_fit, inner_stop, model, seed, protocol):
    medians = fitting_medians(X[inner_fit])
    history = {}
    trained = xgb.train(xgb_params(y[inner_fit], seed, protocol), xgb.DMatrix(transform(X[inner_fit], medians, model), label=y[inner_fit], nthread=THREADS), num_boost_round=protocol['xgboost']['max_rounds'], evals=[(xgb.DMatrix(transform(X[inner_stop], medians, model), label=y[inner_stop], nthread=THREADS), 'source_inner_stop')], early_stopping_rounds=protocol['xgboost']['early_stopping_rounds'], verbose_eval=False, evals_result=history)
    rounds = int(trained.best_iteration) + 1
    assert 1 <= rounds <= protocol['xgboost']['max_rounds']
    return (rounds, {'rounds': rounds, 'inner_fit_rows': len(inner_fit), 'inner_stop_rows': len(inner_stop), 'evaluation_history': history})

def fit_model(X, y, model, rounds, seed, protocol):
    medians = fitting_medians(X)
    values = transform(X, medians, model)
    package = {'name': model, 'medians': medians, 'feature_count': X.shape[1]}
    if model.startswith('ensemble_'):
        package['xgb'] = xgb.train(xgb_params(y, seed, protocol), xgb.DMatrix(values, label=y, nthread=THREADS), num_boost_round=rounds, verbose_eval=False)
        package['rf'] = RandomForestClassifier(**protocol['random_forest'], random_state=seed).fit(values, y)
        package['rounds'] = rounds
    else:
        package['scaler'] = StandardScaler().fit(values)
        package['logistic'] = LogisticRegression(**protocol['logistic'], random_state=seed).fit(package['scaler'].transform(values), y)
    return package

def predict(package, X):
    values = transform(X, package['medians'], package['name'])
    if 'xgb' in package:
        p = (package['xgb'].predict(xgb.DMatrix(values, nthread=THREADS)) + package['rf'].predict_proba(values)[:, 1]) / 2
    else:
        p = package['logistic'].predict_proba(package['scaler'].transform(values))[:, 1]
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    return p

def split_source(y, groups, seed):
    unique = np.unique(groups)
    group_y = np.asarray([y[groups == g].max() for g in unique])
    fitting, stopping = train_test_split(unique, test_size=0.25, stratify=group_y, random_state=seed)
    inner = np.flatnonzero(np.isin(groups, fitting))
    stop = np.flatnonzero(np.isin(groups, stopping))
    assert not set(groups[inner]).intersection(groups[stop])
    return (inner, stop)

def role_count(groups, idx, y):
    return {'observations': len(idx), 'patients': len(np.unique(groups[idx])), 'positive_labels': int(y[idx].sum())}

def train():
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8-sig')))
    prepared = _release_resolve_config(json.loads((ROOT / 'aggregate/preparation.json').read_text(encoding='utf-8-sig')))
    assert prepared['execution_complete'] and prepared['all_frozen_sources_unchanged']
    paths = [ROOT / 'protocol.json', _release_source_path(ROOT / 'prepare_common.py'), Path(__file__), ROOT / 'aggregate/preparation.json']
    paths += [ROOT / f'private/{centre}.npz' for centre in ('WHU', 'MIMIC')]
    before = [identity(p) for p in paths]
    assert not (ROOT / 'analysis_lock.json').exists(), 'Refusing to refit a locked completed/partial run'
    write_json(ROOT / 'analysis_lock.json', {'locked_before_any_fitting': True, 'utc': datetime.now(timezone.utc).isoformat(), 'identities': before})
    data = {}
    for centre in ('WHU', 'MIMIC'):
        with np.load(ROOT / f'private/{centre}.npz', allow_pickle=False) as z:
            data[centre] = {k: z[k].copy() for k in z.files}
    folder = ROOT / 'private/models'
    folder.mkdir()
    reports, packages = ([], [])
    predictions = {}
    start = time.monotonic()
    for centre, d in data.items():
        X, y, groups, roles = (d[k] for k in ('X', 'y', 'groups', 'roles'))
        local = {name: np.full(len(y), np.nan) for name in MODELS}
        for fold, role in enumerate(roles, 1):
            fit = np.flatnonzero(role != 0)
            test = np.flatnonzero(role == 0)
            inner, stop = (np.flatnonzero(role == 2), np.flatnonzero(role == 3))
            assert not set(groups[fit]).intersection(groups[test])
            assert set(inner).issubset(fit) and set(stop).issubset(fit)
            for name in MODELS:
                rounds, selection = select_rounds(X, y, inner, stop, name, 42 + fold, protocol) if name.startswith('ensemble_') else (None, None)
                model = fit_model(X[fit], y[fit], name, rounds, 42 + fold, protocol)
                output = folder / f'{centre}_local_fold{fold}_{name}.joblib'
                joblib.dump(model, output, compress=3)
                local[name][test] = predict(model, X[test])
                record = {'centre': centre, 'scope': 'local_patient_oof', 'fold': fold, 'model': name, 'fit': role_count(groups, fit, y), 'test': role_count(groups, test, y), 'selection': selection, 'target_labels_or_values_used': False, 'package': identity(output)}
                reports.append(record)
                packages.append(record['package'])
            print(json.dumps({'phase': 'local_fitting', 'centre': centre, 'fold_completed': fold}), flush=True)
        assert all((np.isfinite(p).all() for p in local.values()))
        predictions.setdefault(centre, {}).update({'local_' + k: v for k, v in local.items()})
        inner, stop = split_source(y, groups, protocol['seeds']['source_inner'])
        other = 'MIMIC' if centre == 'WHU' else 'WHU'
        target_X = data[other]['X']
        for name in MODELS:
            rounds, selection = select_rounds(X, y, inner, stop, name, 42, protocol) if name.startswith('ensemble_') else (None, None)
            model = fit_model(X, y, name, rounds, 42, protocol)
            output = folder / f'{centre}_source_full_{name}.joblib'
            joblib.dump(model, output, compress=3)
            before_predict = identity(output)
            target_p = predict(model, target_X)
            assert before_predict == identity(output)
            predictions.setdefault(other, {}).update({'transfer_' + name: target_p})
            record = {'centre': centre, 'target': other, 'scope': 'source_frozen_transfer', 'model': name, 'fit': role_count(groups, np.arange(len(y)), y), 'source_inner_fit': role_count(groups, inner, y), 'source_inner_stop': role_count(groups, stop, y), 'selection': selection, 'target_labels_or_values_used_for_fitting_or_selection': False, 'source_model_package_unchanged_after_target_prediction': True, 'package': before_predict}
            reports.append(record)
            packages.append(record['package'])
        print(json.dumps({'phase': 'source_fitting', 'source_completed': centre, 'target': other}), flush=True)
    for centre, d in data.items():
        assert all(('local_' + m in predictions[centre] and 'transfer_' + m in predictions[centre] for m in MODELS))
        with (ROOT / f'private/{centre}_predictions.npz').open('xb') as f:
            np.savez_compressed(f, y=d['y'], groups=d['groups'], **predictions[centre])
    after = [identity(p) for p in paths]
    assert before == after
    write_json(ROOT / 'aggregate/model_manifest.json', {'execution_complete': True, 'training_seconds': time.monotonic() - start, 'input_identities_before': before, 'input_identities_after': after, 'all_inputs_unchanged': True, 'reports': reports, 'environment': {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib')}}, 'model_packages': packages, 'patient_labels_or_rows_in_public_reports': False})

class WeightedMetrics:

    def __init__(self, y, p):
        self.y, self.p = (y, p)
        self.order = np.argsort(p, kind='stable')
        sorted_p = p[self.order]
        self.starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]

    def compute(self, weights):
        pos = np.add.reduceat((weights * self.y)[self.order], self.starts)
        neg = np.add.reduceat((weights * (1 - self.y))[self.order], self.starts)
        tp, tn = (pos.sum(), neg.sum())
        if tp <= 0 or tn <= 0:
            return np.full(3, np.nan)
        auc = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (tp * tn)
        desc_pos, desc_neg = (pos[::-1], neg[::-1])
        cumulative_pos = np.cumsum(desc_pos)
        total = cumulative_pos + np.cumsum(desc_neg)
        precision = np.divide(cumulative_pos, total, out=np.zeros_like(total), where=total > 0)
        ap = np.sum(desc_pos * precision) / tp
        brier = np.dot(weights, (self.p - self.y) ** 2) / weights.sum()
        return np.asarray([auc, ap, brier])

def tests():
    rng = np.random.default_rng(73)
    cases = 0
    for n in (10, 49, 150):
        for tied in (False, True):
            y = rng.integers(0, 2, n)
            p = rng.random(n)
            if tied:
                p = np.round(p, 1)
            calc = WeightedMetrics(y, p)
            for _ in range(20):
                w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
                if not (np.dot(w, y) and np.dot(w, 1 - y)):
                    continue
                expected = [roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)]
                assert np.allclose(calc.compute(w), expected, atol=2e-14, rtol=0)
                cases += 1
    X = np.asarray([[1, np.nan, 3, 4], [2, 6, 4, 8], [9, 12, 8, 1]], dtype=float)
    medians = fitting_medians(X[:2])
    before = medians.copy()
    transformed = transform(X[2:], medians, 'ensemble_median_indicator')
    assert np.array_equal(medians, before) and transformed.shape == (1, 8)
    assert np.array_equal(transform(np.asarray([[np.nan, 6, np.nan, 8]]), medians, 'missingness_only'), [[1, 0, 1, 0]])
    return {'passed': True, 'weighted_metric_sklearn_cases': cases, 'ties_zero_weights_checked': True, 'prediction_transform_does_not_refit_medians': True}

def evaluate():
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8-sig')))
    manifest = _release_resolve_config(json.loads((ROOT / 'aggregate/model_manifest.json').read_text(encoding='utf-8-sig')))
    assert manifest['execution_complete']
    inputs = [ROOT / 'protocol.json', Path(__file__), ROOT / 'aggregate/model_manifest.json']
    inputs += [ROOT / f'private/{c}_predictions.npz' for c in ('WHU', 'MIMIC')]
    before = [identity(p) for p in inputs]
    metrics, paired, gates = ([], [], [])
    test_report = tests()
    prediction_records = []
    for centre in ('WHU', 'MIMIC'):
        with np.load(ROOT / f'private/{centre}_predictions.npz', allow_pickle=False) as z:
            y, groups = (z['y'], z['groups'])
            predictions = {k: z[k] for k in z.files if k not in {'y', 'groups'}}
        source = 'MIMIC' if centre == 'WHU' else 'WHU'
        calculators = {k: WeightedMetrics(y, p) for k, p in predictions.items()}
        unit_weights = np.ones(len(y))
        estimates = {k: c.compute(unit_weights) for k, c in calculators.items()}
        unique, codes = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(protocol['bootstrap']['seed'])
        bootstrap = {k: np.empty((protocol['bootstrap']['replicates'], 3)) for k in calculators}
        draw_digest = hashlib.sha256()
        for i in range(protocol['bootstrap']['replicates']):
            multiplicities = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
            draw_digest.update(multiplicities.tobytes())
            weights = multiplicities[codes]
            for k, c in calculators.items():
                bootstrap[k][i] = c.compute(weights)
        for key, estimate in estimates.items():
            scope, model = key.split('_', 1)
            interval = np.nanquantile(bootstrap[key], [0.025, 0.975], axis=0)
            record = {'source': source if scope == 'transfer' else centre, 'target': centre, 'scope': 'source_frozen_transfer' if scope == 'transfer' else 'local_patient_oof', 'model': model, 'probability_kind': 'raw_uncalibrated_class_weighted', 'input_count': 4, 'analysis_unit': protocol['centres'][centre]['unit'], 'observations': len(y), 'patients': len(unique), 'positive_labels': int(y.sum()), 'bootstrap_repetitions': protocol['bootstrap']['replicates'], 'interval_scope': protocol['bootstrap']['interval']}
            for j, name in enumerate(('roc_auc', 'average_precision', 'brier')):
                record[name] = float(estimate[j])
                record[name + '_ci_low'] = float(interval[0, j])
                record[name + '_ci_high'] = float(interval[1, j])
            metrics.append(record)
        comparisons = [('transfer_' + m, 'local_' + m, 'transfer_minus_local_reference') for m in MODELS]
        comparisons += [(scope + '_' + m, scope + '_ensemble_median', m + '_minus_primary_median') for scope in ('transfer', 'local') for m in MODELS[1:]]
        for first, second, name in comparisons:
            difference = estimates[first] - estimates[second]
            interval = np.nanquantile(bootstrap[first] - bootstrap[second], [0.025, 0.975], axis=0)
            for j, metric in enumerate(('roc_auc', 'average_precision', 'brier')):
                paired.append({'source': source if first.startswith('transfer_') else centre, 'target': centre, 'comparison': name, 'first_prediction': first, 'second_prediction': second, 'metric': metric, 'estimate': float(difference[j]), 'ci_low': float(interval[0, j]), 'ci_high': float(interval[1, j]), 'direction': 'first minus second', 'interval_scope': protocol['bootstrap']['interval'], 'interpretation_limit': 'Different source/local training populations and full-source versus local-80-percent size; not a causal transport penalty' if name == 'transfer_minus_local_reference' else 'Exploratory nominal model comparison'})
        prediction_records.append({'target': centre, 'patient_clusters': len(unique), 'observations': len(y), 'all_predictions_share_same_cluster_draws': True})
        print(json.dumps({'phase': 'evaluation', 'target_completed': centre, 'bootstrap_draws': 2000}), flush=True)
    for direction, status in protocol['gates'].items():
        source, target = direction.split('_to_')
        gates.append({'source': source, 'target': target, 'status': 'executed_limited_four_labs' if status.startswith('execute') else 'not_estimated_unit_gate', 'reason': status})
    for filename, records in (('transfer_metrics.csv', [r for r in metrics if r['scope'] == 'source_frozen_transfer']), ('local_metrics.csv', [r for r in metrics if r['scope'] == 'local_patient_oof']), ('paired_comparisons.csv', paired), ('direction_gates.csv', gates)):
        pd.DataFrame(records).to_csv(ROOT / 'aggregate' / filename, index=False, mode='x')
    after = [identity(p) for p in inputs]
    assert before == after
    write_json(ROOT / 'aggregate/execution.json', {'execution_complete': True, 'passed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'input_identities_before': before, 'input_identities_after': after, 'all_inputs_unchanged': True, 'synthetic_tests': test_report, 'bootstrap': prediction_records, 'source_only_training_and_selection': True, 'local_oof_ensembles': 20, 'full_source_ensembles': 4, 'all_model_packages': len(manifest['model_packages']), 'target_labels_used_only_for_evaluation': True, 'outputs': [identity(p) for p in sorted((ROOT / 'aggregate').glob('*.csv'))], 'limits': ['Original source-specific outcomes, survivor selection and patient/admission units retained; not prospective validation', 'Only a four-laboratory name/report-context/unit reviewed subset, not complete specimen/method equivalence', 'WHU source omissions remain missing for original patients; this can induce measurement/documentation signal', 'No clinical probability calibrator, event timing or warning lead time is established', 'Four SCH directions remain unestimated because original SCH unit metadata is unavailable', 'Target bootstrap conditions on fitted source models and does not include source-training uncertainty']})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('test', 'train', 'evaluate'))
    action = parser.parse_args().action
    with threadpool_limits(limits=THREADS):
        if action == 'test':
            print(json.dumps(tests()))
        elif action == 'train':
            train()
        else:
            evaluate()
if __name__ == '__main__':
    main()
