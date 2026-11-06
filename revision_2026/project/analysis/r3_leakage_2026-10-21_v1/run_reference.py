"""Strict-input references; only new private and aggregate outputs are written."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
THREADS = 2

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8')))

def parameters(y, cohort, fold, protocol):
    p = protocol['xgb']
    return {k: p[k] for k in ('max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'tree_method')} | {'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 42 if cohort != 'MIMIC' else fold, 'nthread': THREADS, 'verbosity': 0, 'scale_pos_weight': float((y == 0).sum() / (y == 1).sum())}

def fit_reference(X, y, cohort, fold, rounds, protocol):
    assert X.shape[1] == 2 and np.isfinite(X).all() and (len(np.unique(y)) == 2)
    params = parameters(y, cohort, fold, protocol)
    booster = xgb.train(params, xgb.DMatrix(X, label=y, nthread=THREADS), num_boost_round=rounds)
    weight = {0: 1, 1: int((y == 0).sum() / (y == 1).sum())} if cohort != 'MIMIC' else 'balanced'
    rf = RandomForestClassifier(**protocol['rf'], class_weight=weight, random_state=42 if cohort != 'MIMIC' else fold, n_jobs=THREADS)
    rf.fit(X, y)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, tol=1e-08))
    logistic.fit(X, y)
    assert logistic[-1].n_iter_.max() < 5000
    return (booster, rf, logistic, params)

def run(cohort, source):
    protocol = load(ROOT / 'protocol.json')
    dest = ROOT / 'private' / cohort
    assert not dest.exists(), 'Never overwrite a completed or partial run.'
    source = Path(source).resolve()
    prepared = BASE / 'private/prepared' / f'{cohort}.npz'
    rolepath = BASE / 'private/runs' / f'{cohort}_original/patient_roles.npz'
    oldpath = BASE / 'private/runs' / f'{cohort}_original/predictions.npz'
    with np.load(source, allow_pickle=False) as z:
        X, groups = (z['X'].astype(np.float32), z['groups'].copy())
    with np.load(prepared, allow_pickle=False) as z:
        y = z['y'].astype(np.int8)
        assert np.array_equal(groups, z['groups'])
    with np.load(rolepath, allow_pickle=False) as z:
        roles = z['roles'].copy()
        assert np.array_equal(groups, z['groups']) and np.array_equal(y, z['y'])
    with np.load(oldpath, allow_pickle=False) as z:
        assert np.array_equal(groups, z['groups']) and np.array_equal(y, z['y'])
        old_full, old_age = (z['primary_raw'].copy(), z['age'].copy())
        old_fold = z['fold'].copy()
    assert X.shape == (len(y), 2)
    eligible = np.isfinite(X).all(axis=1)
    assert eligible.any() and roles.shape == (5, len(y))
    dest.mkdir(parents=True)
    save(dest / 'input_lock.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'sources': [identity(p) for p in (ROOT / 'protocol.json', __file__, source, prepared, rolepath, oldpath)], 'protocol': protocol, 'environment': {'executable': sys.executable, 'python': sys.version, 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'joblib')}}, 'scope': 'Private data and model paths must not be included in submission files.'})
    predictions = {name: np.full(len(y), np.nan) for name in ('strict_ensemble', 'strict_logistic')}
    fold_ids = np.zeros(len(y), dtype=np.int8)
    reports = []
    for fold, role in enumerate(roles, 1):
        folder = dest / f'fold_{fold}'
        folder.mkdir()
        test = np.flatnonzero(eligible & (role == 0))
        train = np.flatnonzero(eligible & (role != 0))
        inner = np.flatnonzero(eligible & (role == 2))
        stop = np.flatnonzero(eligible & (role == 3))
        assert all((len(np.unique(y[ix])) == 2 for ix in (test, train, inner, stop)))
        assert not set(groups[test]) & set(groups[train])
        assert not set(groups[inner]) & set(groups[stop])
        assert np.array_equal(role == 0, old_fold == fold)
        history = {}
        early = xgb.train(parameters(y[inner], cohort, fold, protocol), xgb.DMatrix(X[inner], label=y[inner], nthread=THREADS), num_boost_round=protocol['xgb']['max_rounds'], evals=[(xgb.DMatrix(X[stop], label=y[stop], nthread=THREADS), 'inner_stop')], early_stopping_rounds=protocol['xgb']['early_stopping_rounds'], evals_result=history, verbose_eval=False)
        rounds = int(early.best_iteration) + 1
        booster, rf, logistic, params = fit_reference(X[train], y[train], cohort, fold, rounds, protocol)
        predictions['strict_ensemble'][test] = 0.5 * booster.predict(xgb.DMatrix(X[test], nthread=THREADS)) + 0.5 * rf.predict_proba(X[test])[:, 1]
        predictions['strict_logistic'][test] = logistic.predict_proba(X[test])[:, 1]
        fold_ids[test] = fold
        altered_y = y.copy()
        altered_y[test] = 1 - altered_y[test]
        check = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, tol=1e-08))
        check.fit(X[train], altered_y[train])
        invariant = np.array_equal(logistic[-1].coef_, check[-1].coef_) and np.array_equal(logistic.predict_proba(X[test]), check.predict_proba(X[test]))
        assert invariant
        early.save_model(folder / 'inner_stop.ubj')
        booster.save_model(folder / 'reference_xgb.ubj')
        joblib.dump({'rf': rf, 'logistic': logistic, 'features': ['source_age', 'source_recorded_sex']}, folder / 'reference.joblib', compress=3)
        report = {'fold': fold, 'selected_rounds': rounds, 'parameters': params, 'patient_overlap_train_test': 0, 'heldout_label_flip_logistic_fit_invariant': invariant, 'roles': {k: {'rows': len(ix), 'patients': len(np.unique(groups[ix])), 'positive_labels': int(y[ix].sum())} for k, ix in (('test', test), ('train', train), ('inner', inner), ('stop', stop))}, 'inner_history': history}
        save(folder / 'report.json', report)
        reports.append(report)
        print(f'{cohort} fold {fold}/5 completed; rounds={rounds}', flush=True)
    assert np.all(fold_ids[eligible] > 0) and np.all(fold_ids[~eligible] == 0)
    assert all((np.isfinite(p[eligible]).all() and np.isnan(p[~eligible]).all() for p in predictions.values()))
    np.savez_compressed(dest / 'predictions.npz', **predictions, y=y, groups=groups, eligible=eligible, fold=fold_ids, prior_full=old_full, prior_age=old_age)
    save(dest / 'training_complete.json', {'cohort': cohort, 'eligible_rows': int(eligible.sum()), 'excluded_rows': int((~eligible).sum()), 'eligible_patients': len(np.unique(groups[eligible])), 'eligible_positive_labels': int(y[eligible].sum()), 'folds': reports, 'output': identity(dest / 'predictions.npz'), 'ended_at_utc': datetime.now(timezone.utc).isoformat()})

def metrics(y, p, weights=None):
    return [float(roc_auc_score(y, p, sample_weight=weights)), float(average_precision_score(y, p, sample_weight=weights)), float(brier_score_loss(y, p, sample_weight=weights))]

def summarize(cohort):
    protocol = load(ROOT / 'protocol.json')
    source = ROOT / 'private' / cohort / 'predictions.npz'
    dest = ROOT / 'results' / cohort
    assert not dest.exists()
    dest.mkdir(parents=True)
    with np.load(source, allow_pickle=False) as z:
        eligible = z['eligible']
        y, groups = (z['y'][eligible], z['groups'][eligible])
        names = ['strict_ensemble', 'strict_logistic', 'prior_full', 'prior_age']
        scores = [z[name][eligible] for name in names]
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    draws = np.full((protocol['bootstrap_replicates'], len(names), 3), np.nan)
    points = np.array([metrics(y, p) for p in scores])
    for iteration in range(len(draws)):
        counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        weights = counts[inverse]
        if not all((weights[y == v].sum() > 0 for v in (0, 1))):
            continue
        for j, p in enumerate(scores):
            draws[iteration, j] = metrics(y, p, weights)
    assert np.isfinite(draws).all()
    metric_names = ['ROC_AUC', 'average_precision', 'Brier']
    rows = []
    for j, name in enumerate(names):
        for k, metric in enumerate(metric_names):
            low, high = np.percentile(draws[:, j, k], [2.5, 97.5])
            rows.append({'centre': cohort, 'model': name, 'metric': metric, 'estimate': points[j, k], 'ci_low': low, 'ci_high': high, 'rows': len(y), 'patients': len(unique), 'positive_labels': int(y.sum()), 'bootstrap_n': len(draws)})
    differences = []
    for j in (2, 3):
        for k, metric in enumerate(metric_names):
            low, high = np.percentile(draws[:, 0, k] - draws[:, j, k], [2.5, 97.5])
            differences.append({'centre': cohort, 'comparison': f'strict_ensemble_minus_{names[j]}', 'metric': metric, 'estimate': points[0, k] - points[j, k], 'ci_low': low, 'ci_high': high})
    pd.DataFrame(rows).to_csv(dest / 'performance.csv', index=False)
    pd.DataFrame(differences).to_csv(dest / 'paired_differences.csv', index=False)
    np.savez_compressed(dest / 'aggregate_bootstrap.npz', draws=draws, points=points, model_names=np.array(names), metric_names=np.array(metric_names))
    save(dest / 'summary.json', {'cohort': cohort, 'conditional_uncertainty': True, 'rows': rows, 'paired_differences': differences, 'valid_resamples': len(draws), 'limits': protocol['unresolved_by_design'], 'comparison_caution': protocol['comparisons'], 'source_identity': identity(source), 'protocol_identity': identity(ROOT / 'protocol.json')})
    print(json.dumps({'cohort': cohort, 'auc': [r for r in rows if r['metric'] == 'ROC_AUC']}), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('train', 'summarize'))
    parser.add_argument('cohort', choices=('SCH', 'WHU', 'MIMIC'))
    parser.add_argument('--source')
    args = parser.parse_args()
    if args.mode == 'train':
        assert args.source
        run(args.cohort, args.source)
    else:
        summarize(args.cohort)
