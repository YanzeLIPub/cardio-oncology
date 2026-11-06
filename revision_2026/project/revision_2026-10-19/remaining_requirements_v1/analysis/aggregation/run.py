"""Execute frozen matched patient-level laboratory aggregation sensitivity."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from prepare import ROOT, OLD, identity, save_json, sha

def zero(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def xparams(y):
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': float((y == 0).sum() / (y == 1).sum()), 'seed': 42, 'nthread': 2, 'verbosity': 0, 'tree_method': 'hist'}

def dmat(X, y=None):
    return xgb.DMatrix(zero(X), label=y, nthread=2)

def lock():
    report = _release_resolve_config(json.loads((ROOT / 'preparation.json').read_text(encoding='utf-8')))
    paths = [ROOT / 'protocol.json', _release_source_path(ROOT / 'prepare.py'), _release_source_path(ROOT / 'run.py'), ROOT / 'preparation.json']
    paths.extend((Path(item['path']) for item in report['inputs']))
    for center in ('SCH', 'WHU'):
        paths.append(ROOT / f'private/prepared/{center}.npz')
        paths.append(OLD / f'private/runs/{center}_original/predictions.npz')
        for fold in range(1, 6):
            folder = OLD / f'private/runs/{center}_original/fold_{fold}'
            paths.extend((folder / name for name in ('full_package.joblib', 'full_xgb.ubj', 'fit_report.json')))
    state = {'files': [identity(p) for p in dict.fromkeys(paths)], 'environment': {'python': sys.version, 'executable': sys.executable, **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib', 'scipy')}}}
    path = ROOT / 'analysis_lock.json'
    if path.exists():
        assert _release_resolve_config(json.loads(path.read_text(encoding='utf-8')))['state'] == state
    else:
        save_json(path, {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'state': state, 'protocol': _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8')))})
    return sha(path)

def run(center, lockhash):
    data = dict(np.load(ROOT / f'private/prepared/{center}.npz', allow_pickle=False))
    oldpred = dict(np.load(OLD / f'private/runs/{center}_original/predictions.npz', allow_pickle=False))
    y, roles, names = (data['y'], data['roles'], data['feature_names'].tolist())
    assert np.array_equal(y, oldpred['y']) and np.array_equal(data['groups'], oldpred['groups'])
    folder = ROOT / f'private/runs/{center}'
    folder.mkdir(parents=True, exist_ok=False)
    out = {k: np.full(len(y), np.nan) for k in ('mean', 'first_record', 'first_observed', 'median', 'record_count')}
    components = {f'{k}_{c}': np.full(len(y), np.nan) for k in out if k != 'record_count' for c in ('xgb', 'rf')}
    reports = []
    for fold, role in enumerate(roles, 1):
        test, inner, stop = (np.flatnonzero(role == r) for r in (0, 2, 3))
        train = np.flatnonzero(role != 0)
        assert len(np.intersect1d(test, train)) == 0
        print(json.dumps({'center': center, 'fold': fold, 'stage': 'replay_mean'}), flush=True)
        oldfold = OLD / f'private/runs/{center}_original/fold_{fold}'
        package = joblib.load(oldfold / 'full_package.joblib')
        assert package['input_feature_names'] == names and package['preprocessor']['zero_fill']
        assert package['preprocessor']['columns'].tolist() == list(range(104))
        booster = xgb.Booster()
        booster.load_model(oldfold / 'full_xgb.ubj')
        booster.set_param({'nthread': 2})
        xp = booster.predict(dmat(data['X_mean'][test]))
        package['rf'].n_jobs = 2
        rp = package['rf'].predict_proba(zero(data['X_mean'][test]))[:, 1]
        assert np.array_equal(xp, oldpred['xgb_full'][test])
        assert np.allclose(rp, oldpred['rf_full'][test], rtol=0, atol=2e-15)
        assert np.allclose(0.5 * xp + 0.5 * rp, oldpred['primary_raw'][test], rtol=0, atol=2e-15)
        out['mean'][test] = oldpred['primary_raw'][test]
        components['mean_xgb'][test] = xp
        components['mean_rf'][test] = rp
        countmodel = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000, tol=1e-08))
        countX = np.log1p(data['record_count']).reshape(-1, 1)
        countmodel.fit(countX[train], y[train])
        assert countmodel[-1].n_iter_.max() < 5000
        out['record_count'][test] = countmodel.predict_proba(countX[test])[:, 1]
        countdest = folder / f'fold_{fold}_record_count.joblib'
        joblib.dump(countmodel, countdest, compress=3)
        for variant in ('first_record', 'first_observed', 'median'):
            X = data[f'X_{variant}']
            dest = folder / f'fold_{fold}_{variant}'
            dest.mkdir()
            history = {}
            selection = xgb.train(xparams(y[inner]), dmat(X[inner], y[inner]), num_boost_round=300, evals=[(dmat(X[stop], y[stop]), 'inner_stop')], early_stopping_rounds=20, evals_result=history, verbose_eval=False)
            rounds = int(selection.best_iteration) + 1
            selection.save_model(dest / 'inner_xgb.ubj')
            params = xparams(y[train])
            final = xgb.train(params, dmat(X[train], y[train]), num_boost_round=rounds, verbose_eval=False)
            final.save_model(dest / 'full_xgb.ubj')
            rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight={0: 1, 1: int((y[train] == 0).sum() / (y[train] == 1).sum())}, random_state=42, n_jobs=2)
            rf.fit(zero(X[train]), y[train])
            xp, rp = (final.predict(dmat(X[test])), rf.predict_proba(zero(X[test]))[:, 1])
            components[variant + '_xgb'][test] = xp
            components[variant + '_rf'][test] = rp
            out[variant][test] = 0.5 * xp + 0.5 * rp
            joblib.dump({'rf': rf, 'input_feature_names': names, 'preprocessing': 'constant zero fill', 'xgb_parameters': params, 'selected_rounds': rounds}, dest / 'full_package.joblib', compress=3)
            record = {'center': center, 'fold': fold, 'variant': variant, 'selected_rounds': rounds, 'inner_parameters': xparams(y[inner]), 'final_parameters': params, 'rf_parameters': rf.get_params(), 'inner_history': history, 'n_inner_fit': len(inner), 'n_inner_stop': len(stop), 'n_outer_train': len(train), 'n_outer_test': len(test), 'test_positive': int(y[test].sum())}
            save_json(dest / 'fit_report.json', record)
            reports.append(record)
            print(json.dumps({'center': center, 'fold': fold, 'variant': variant, 'rounds': rounds, 'fit_complete': True}), flush=True)
    assert all((np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in out.values()))
    with (folder / 'predictions.npz').open('xb') as stream:
        np.savez_compressed(stream, y=y, groups=data['groups'], roles=roles, fold=np.argmax(roles == 0, axis=0) + 1, record_count=data['record_count'], **out, **components)
    save_json(ROOT / f'results/{center}_training.json', {'center': center, 'training_complete': True, 'model_report_count': len(reports), 'mean_models_replayed': 5, 'mean_probability_tolerance': 2e-15, 'mean_reference': '10/15 patient-isolated revalidation primary_raw', 'rows': len(y), 'positive': int(y.sum()), 'fold_reports': reports, 'predictions': identity(folder / 'predictions.npz')})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--center', choices=('SCH', 'WHU', 'all'), default='all')
    args = parser.parse_args()
    lockhash = lock()
    for center in ('SCH', 'WHU') if args.center == 'all' else (args.center,):
        run(center, lockhash)
if __name__ == '__main__':
    main()
