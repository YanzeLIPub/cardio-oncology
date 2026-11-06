"""Patient-isolated measurement-window and mapping sensitivities on fixed raw labels."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import defaultdict
import os, json, sys, platform, importlib.metadata
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '4'
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits
OUT = Path(__file__).resolve().parents[1]
PRIVATE = OUT / 'private'
AGG = OUT / 'aggregate'

def transform(X, med):
    return np.where(np.isfinite(X), X, med).astype(np.float32)

def medians(X):
    return np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in X.T])

def parameters(y, seed):
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'device': 'cpu', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': float((y == 0).sum() / (y == 1).sum()), 'seed': seed, 'nthread': 4, 'verbosity': 0}

def fit_predict(X, y, train, test, model, fold):
    med = medians(X[train])
    a = transform(X[train], med)
    b = transform(X[test], med)
    if model == 'logistic':
        scaler = StandardScaler().fit(a)
        lr = LogisticRegression(C=1, max_iter=2000, random_state=42 + fold).fit(scaler.transform(a), y[train])
        return (lr.predict_proba(scaler.transform(b))[:, 1], {'rounds': None})
    inner, stop = train_test_split(train, test_size=0.25, stratify=y[train], random_state=20261103 + fold)
    inner_med = medians(X[inner])
    pilot = xgb.train(parameters(y[inner], 42 + fold), xgb.DMatrix(transform(X[inner], inner_med), label=y[inner], nthread=4), num_boost_round=300, evals=[(xgb.DMatrix(transform(X[stop], inner_med), label=y[stop], nthread=4), 'inner_stop')], early_stopping_rounds=20, verbose_eval=False)
    rounds = int(pilot.best_iteration) + 1
    final = xgb.train(parameters(y[train], 42 + fold), xgb.DMatrix(a, label=y[train], nthread=4), num_boost_round=rounds, verbose_eval=False)
    forest = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight='balanced', n_jobs=4, random_state=42 + fold).fit(a, y[train])
    return ((final.predict(xgb.DMatrix(b, nthread=4)) + forest.predict_proba(b)[:, 1]) / 2, {'rounds': rounds, 'inner_train': len(inner), 'inner_stop': len(stop)})

def main():
    env = {'python': platform.python_version(), 'executable': sys.executable, 'packages': {x: importlib.metadata.version(x) for x in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'matplotlib']}, 'device': 'CPU', 'threads': 4, 'seed': 42, 'bootstrap_seed': 20261103}
    (AGG / 'training_environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    metrics = []
    contrasts = []
    training = []
    for experiment, file in [('measurement_caps', 'whu_windows.npz'), ('assay_mapping', 'whu_assay_sensitivity.npz')]:
        with np.load(PRIVATE / file) as z:
            data = {k: z[k] for k in z.files}
        y = data.pop('y')
        groups = data.pop('groups')
        assert len(y) > 100 and np.unique(groups).size == len(y) and (y.sum() >= 10)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
        roles = np.full(len(y), -1, int)
        for i, (_, test) in enumerate(folds):
            roles[test] = i
        predictions = {}
        fitted_matrices = []
        with threadpool_limits(limits=4):
            for scope, X in data.items():
                identical = next((old for old in fitted_matrices if np.array_equal(data[old], X, equal_nan=True)), None)
                if identical is not None:
                    for model in ['logistic', 'ensemble']:
                        predictions[scope + '__' + model] = predictions[identical + '__' + model].copy()
                    print(json.dumps({'phase': 'identical matrix', 'scope': scope, 'same_as': identical}), flush=True)
                    continue
                for model in ['logistic', 'ensemble']:
                    pred = np.full(len(y), np.nan)
                    for fold, (train, test) in enumerate(folds):
                        pred[test], detail = fit_predict(X, y, train, test, model, fold)
                        training.append({'experiment': experiment, 'scope': scope, 'model': model, 'fold': fold + 1, 'train_patients': len(train), 'test_patients': len(test), 'train_positive': int(y[train].sum()), 'test_positive': int(y[test].sum()), 'patient_overlap': int(np.intersect1d(groups[train], groups[test]).size), **detail})
                    assert np.isfinite(pred).all()
                    predictions[scope + '__' + model] = pred
                    print(json.dumps({'phase': 'WHU fit', 'experiment': experiment, 'scope': scope, 'model': model, 'patients': len(y), 'auc': roc_auc_score(y, pred)}), flush=True)
                fitted_matrices.append(scope)
        np.savez_compressed(PRIVATE / (experiment + '_predictions.npz'), y=y, groups=groups, fold=roles, **predictions)
        rng = np.random.default_rng(20261103)
        draws = rng.integers(0, len(y), size=(2000, len(y)))
        valid = np.array([0 < y[d].sum() < len(y) for d in draws])
        draws = draws[valid]
        boots = {}
        for name, p in predictions.items():
            scores = np.array([roc_auc_score(y[d], p[d]) for d in draws])
            briers = np.mean((p[draws] - y[draws]) ** 2, axis=1)
            boots[name] = (scores, briers)
            scope, model = name.split('__')
            lo, hi = np.quantile(scores, [0.025, 0.975])
            bl, bh = np.quantile(briers, [0.025, 0.975])
            metrics.append({'experiment': experiment, 'scope': scope, 'model': model, 'patients': len(y), 'positive_labels': int(y.sum()), 'auc': roc_auc_score(y, p), 'auc_low': lo, 'auc_high': hi, 'average_precision': average_precision_score(y, p), 'brier': brier_score_loss(y, p), 'brier_low': bl, 'brier_high': bh, 'expected_observed': float(p.sum() / y.sum()), 'bootstrap_replicates': len(draws)})
        pairs = [('h24', 'h0'), ('h48', 'h24'), ('whole', 'h48'), ('h24_outer_all', 'h24'), ('h48_outer_all', 'h48'), ('h24_date_end', 'h24'), ('h48_date_end', 'h48')] if experiment == 'measurement_caps' else [('verified', 'name_only'), ('unit_only', 'name_only'), ('verified', 'unit_only')]
        for first, second in pairs:
            for model in ['logistic', 'ensemble']:
                a, b = (first + '__' + model, second + '__' + model)
                ba, bb = (boots[a], boots[b])
                diff = ba[0] - bb[0]
                db = ba[1] - bb[1]
                lo, hi = np.quantile(diff, [0.025, 0.975])
                bl, bh = np.quantile(db, [0.025, 0.975])
                contrasts.append({'experiment': experiment, 'first': first, 'second': second, 'model': model, 'auc_difference': roc_auc_score(y, predictions[a]) - roc_auc_score(y, predictions[b]), 'auc_difference_low': lo, 'auc_difference_high': hi, 'brier_difference': brier_score_loss(y, predictions[a]) - brier_score_loss(y, predictions[b]), 'brier_difference_low': bl, 'brier_difference_high': bh})
        pd.DataFrame(metrics).to_csv(AGG / 'whu_performance.csv', index=False)
        pd.DataFrame(contrasts).to_csv(AGG / 'whu_paired_comparisons.csv', index=False)
        print(json.dumps({'phase': 'WHU intervals complete', 'experiment': experiment}), flush=True)
    pd.DataFrame(training).to_csv(AGG / 'whu_training_roles.csv', index=False)
    (AGG / 'analysis_complete.json').write_text(json.dumps({'WHU_complete': True, 'original_main_models_unchanged': True, 'training_patient_overlap': sum((r['patient_overlap'] for r in training)), 'fitted_fold_models': len(training), 'inference': 'Fixed-prediction patient bootstrap; original diagnostic labels, not future events'}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
