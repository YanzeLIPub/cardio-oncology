"""Patient-isolated MIMIC measurement-window sensitivity; original labels and fixed roles."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'protocol.json'

def sha(path):
    return None

def file_identity(path):
    path = Path(path)
    return {'path': path.as_posix(), 'bytes': path.stat().st_size}

def save_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def now():
    return datetime.now().astimezone().isoformat()

def split_patients(indices, y, groups, fraction, seed):
    candidates = np.unique(groups[indices])
    any_positive = np.array([y[indices][groups[indices] == g].max() for g in candidates])
    train_groups, heldout_groups = train_test_split(candidates, test_size=fraction, stratify=any_positive, random_state=seed)
    return (indices[np.isin(groups[indices], train_groups)], indices[np.isin(groups[indices], heldout_groups)])

def patient_roles(y, groups, cohort, config):
    splitter = StratifiedKFold(config['outer_folds'], shuffle=True, random_state=config['outer_seed']) if cohort != 'MIMIC' else StratifiedGroupKFold(config['outer_folds'], shuffle=True, random_state=config['outer_seed'])
    splits = splitter.split(np.zeros((len(y), 1)), y, groups if cohort == 'MIMIC' else None)
    role_matrix = np.full((config['outer_folds'], len(y)), -1, dtype=np.int8)
    for i, (train, test) in enumerate(splits, 1):
        base, cal = split_patients(train, y, groups, config['calibration_fraction_of_outer_train_patients'], config['calibration_seed_base'] + i)
        inner, stop = split_patients(base, y, groups, config['inner_validation_fraction_of_base_fit_patients'], config['inner_seed_base'] + i)
        for role, indices in enumerate((test, cal, inner, stop)):
            role_matrix[i - 1, indices] = role
        sets = [set(groups[a]) for a in (test, cal, inner, stop)]
        assert all((not sets[a].intersection(sets[b]) for a in range(4) for b in range(a + 1, 4)))
        assert all((len(np.unique(y[a])) == 2 for a in (test, cal, inner, stop)))
        assert np.all(role_matrix[i - 1] >= 0)
    assert np.all((role_matrix == 0).sum(axis=0) == 1)
    return role_matrix

def fit_preprocessor(X, cohort):
    if cohort != 'MIMIC':
        return {'zero_fill': True, 'columns': np.arange(X.shape[1]), 'imputer': None}
    columns = np.flatnonzero(np.isfinite(X).any(axis=0))
    if not len(columns):
        raise ValueError('No observed feature in fitting patients')
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    imp.fit(X[:, columns])
    return {'zero_fill': False, 'columns': columns, 'imputer': imp}

def transform(prep, X):
    values = X[:, prep['columns']]
    if prep['zero_fill']:
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return prep['imputer'].transform(values).astype(np.float32)

def xgb_parameters(y, cohort, fold, config, threads):
    p = config['xgb']
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': p['max_depth'], 'learning_rate': p['learning_rate'], 'subsample': p['subsample'], 'colsample_bytree': p['colsample_bytree'], 'scale_pos_weight': float((y == 0).sum() / (y == 1).sum()), 'seed': 42 if cohort != 'MIMIC' else fold, 'nthread': threads, 'verbosity': 0, 'tree_method': 'hist'}

def choose_rounds(X, y, inner, stop, cohort, fold, config, threads, folder):
    prep = fit_preprocessor(X[inner], cohort)
    params = xgb_parameters(y[inner], cohort, fold, config, threads)
    history = {}
    model = xgb.train(params, xgb.DMatrix(transform(prep, X[inner]), label=y[inner], nthread=threads), num_boost_round=config['xgb']['max_rounds'], evals=[(xgb.DMatrix(transform(prep, X[stop]), label=y[stop], nthread=threads), 'inner_stop')], early_stopping_rounds=config['xgb']['early_stopping_rounds'], verbose_eval=False, evals_result=history)
    rounds = int(model.best_iteration) + 1
    assert 1 <= rounds <= config['xgb']['max_rounds']
    model.save_model(folder / 'inner_stop_model.ubj')
    joblib.dump(prep, folder / 'inner_preprocessor.joblib', compress=3)
    save_json(folder / 'inner_evaluation.json', {'history': history, 'best_iteration': int(model.best_iteration), 'selected_rounds': rounds, 'parameters': params})
    return (rounds, {'selected_rounds': rounds, 'inner_fitted_columns': prep['columns'].tolist(), 'inner_scale_pos_weight': params['scale_pos_weight']})

def fit_ensemble(X, y, cohort, fold, config, threads, rounds):
    prep = fit_preprocessor(X, cohort)
    values = transform(prep, X)
    params = xgb_parameters(y, cohort, fold, config, threads)
    booster = xgb.train(params, xgb.DMatrix(values, label=y, nthread=threads), num_boost_round=rounds, verbose_eval=False)
    rf_params = dict(config['rf'])
    weight = {0: 1, 1: int((y == 0).sum() / (y == 1).sum())} if cohort != 'MIMIC' else 'balanced'
    rf = RandomForestClassifier(**rf_params, class_weight=weight, random_state=42 if cohort != 'MIMIC' else fold, n_jobs=threads)
    rf.fit(values, y)
    return (prep, booster, rf, {'xgb_parameters': params, 'rf_parameters': rf.get_params()})

def ensemble_predict(model, X, threads):
    prep, booster, rf, _ = model
    values = transform(prep, X)
    xgb_p = booster.predict(xgb.DMatrix(values, nthread=threads))
    rf_p = rf.predict_proba(values)[:, 1]
    return (0.5 * xgb_p + 0.5 * rf_p, xgb_p, rf_p)

def logit(p):
    p = np.clip(p, 1e-06, 1 - 1e-06)
    return np.log(p / (1 - p)).reshape(-1, 1)

def make_logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, tol=1e-06))

def save_model(folder, prefix, model, names):
    prep, booster, rf, params = model
    booster.save_model(folder / f'{prefix}_xgb.ubj')
    joblib.dump({'preprocessor': prep, 'rf': rf, 'parameters': params, 'input_feature_names': names, 'fitted_feature_names': [names[j] for j in prep['columns']]}, folder / f'{prefix}_package.joblib', compress=3)

def freeze(config):
    sources = [PROTOCOL, Path(__file__).resolve(), _release_source_path(ROOT / 'prepare_windows.py'), ROOT / 'evidence/preparation.json', ROOT / 'private/prepared/patient_roles.npz', ROOT / 'private/prepared/evaluation_subset.npz']
    sources += [ROOT / 'private/prepared' / f'MIMIC_{hour}h.npz' for hour in (0, 24, 48)]
    sources += [_release_source_path(Path(config['reference_branch']) / 'run_validation.py')]
    state = {'files': [file_identity(p) for p in sources], 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'joblib')}}}
    dest = ROOT / 'evidence/analysis_lock.json'
    if dest.exists():
        prior = _release_resolve_config(json.loads(dest.read_text(encoding='utf-8')))
        if prior['state'] != state:
            raise ValueError('Frozen inputs/code/environment changed; start a new version')
    else:
        save_json(dest, {'frozen_at': now(), 'state': state, 'protocol': config})
    return sha(dest)

def run_one(spec, config, threads, lock_hash):
    name, cohort = (spec['name'], spec['cohort'])
    folder = ROOT / 'private' / 'runs' / name
    if folder.exists():
        raise FileExistsError(f'Run directory already exists: {name}; do not silently reuse or overwrite')
    folder.mkdir(parents=True)
    data_path = ROOT / 'private' / 'prepared' / spec['matrix']
    with np.load(data_path, allow_pickle=False) as data:
        X = data['X'].copy()
        y, groups = (data['y'].copy(), data['groups'].copy())
        original = data['p_original'].copy()
        names = data['feature_names'].tolist()
    X[~np.isfinite(X)] = np.nan
    assert len(y) == len(groups) == len(original) == len(X)
    assert all((n in names for n in spec['drop_features']))
    keep = [j for j, n in enumerate(names) if n not in spec['drop_features']]
    X, names = (X[:, keep], [names[j] for j in keep])
    age_name = 'Age' if cohort == 'MIMIC' else '年龄'
    age_index = names.index(age_name)
    missing_mask = np.isnan(X).astype(np.float32)
    with np.load(ROOT / 'private/prepared/patient_roles.npz', allow_pickle=False) as fixed:
        roles = fixed['roles'].copy()
        assert np.array_equal(fixed['y'], y) and np.array_equal(fixed['groups'], groups)
    assert np.array_equal(roles, patient_roles(y, groups, cohort, config))
    np.savez_compressed(folder / 'patient_roles.npz', roles=roles, groups=groups, y=y)
    prediction_names = ('primary_raw', 'base_raw', 'base_platt', 'xgb_full', 'rf_full', 'logistic', 'missingness', 'age')
    out = {key: np.full(len(y), np.nan) for key in prediction_names}
    fold_ids = np.zeros(len(y), dtype=np.int8)
    fold_reports = []
    started = now()
    clock = time.monotonic()
    for fold, role in enumerate(roles, 1):
        test, cal, inner, stop = [np.flatnonzero(role == r) for r in range(4)]
        base, full = (np.flatnonzero(role >= 2), np.flatnonzero(role != 0))
        dest = folder / f'fold_{fold}'
        dest.mkdir()
        print(f'{name} fold {fold}/5: selecting rounds inside training patients', flush=True)
        rounds, selection = choose_rounds(X, y, inner, stop, cohort, fold, config, threads, dest)
        base_model = fit_ensemble(X[base], y[base], cohort, fold, config, threads, rounds)
        cal_p = ensemble_predict(base_model, X[cal], threads)[0]
        base_p = ensemble_predict(base_model, X[test], threads)[0]
        platt = LogisticRegression(C=1.0, max_iter=5000, tol=1e-08)
        platt.fit(logit(cal_p), y[cal])
        out['base_raw'][test] = base_p
        out['base_platt'][test] = platt.predict_proba(logit(base_p))[:, 1]
        full_model = fit_ensemble(X[full], y[full], cohort, fold, config, threads, rounds)
        primary, xp, rp = ensemble_predict(full_model, X[test], threads)
        out['primary_raw'][test], out['xgb_full'][test], out['rf_full'][test] = (primary, xp, rp)
        feature_logit = make_logistic()
        feature_logit.fit(transform(full_model[0], X[full]), y[full])
        out['logistic'][test] = feature_logit.predict_proba(transform(full_model[0], X[test]))[:, 1]
        missing_logit = make_logistic()
        missing_logit.fit(missing_mask[full], y[full])
        out['missingness'][test] = missing_logit.predict_proba(missing_mask[test])[:, 1]
        age_logit = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, tol=1e-06))
        age_logit.fit(X[full, age_index].reshape(-1, 1), y[full])
        out['age'][test] = age_logit.predict_proba(X[test, age_index].reshape(-1, 1))[:, 1]
        convergence = {'platt': platt, 'logistic': feature_logit[-1], 'missingness': missing_logit[-1], 'age': age_logit[-1]}
        assert all((int(m.n_iter_.max()) < m.max_iter for m in convergence.values())), 'Logistic iteration limit reached'
        fold_ids[test] = fold
        save_model(dest, 'base', base_model, names)
        save_model(dest, 'full', full_model, names)
        joblib.dump({'platt': platt, 'logistic': feature_logit, 'missingness': missing_logit, 'age': age_logit, 'age_column': age_index}, dest / 'comparators_and_platt.joblib', compress=3)
        roles_summary = {label: {'rows': len(ix), 'patients': len(np.unique(groups[ix])), 'positive_labels': int(y[ix].sum())} for label, ix in (('test', test), ('calibration', cal), ('inner_fit', inner), ('inner_stop', stop), ('base_fit', base), ('full_outer_train', full))}
        report = {'fold': fold, 'roles': roles_summary, 'all_disjoint_role_patient_intersections_zero': True, 'selection': selection, 'full_fitted_feature_names': [names[j] for j in full_model[0]['columns']], 'base_fitted_feature_names': [names[j] for j in base_model[0]['columns']], 'full_parameters': full_model[3], 'base_parameters': base_model[3], 'logistic_convergence': {k: {'iterations': int(m.n_iter_.max()), 'max_iter': m.max_iter, 'iteration_limit_reached': False} for k, m in convergence.items()}, 'platt': {'intercept': float(platt.intercept_[0]), 'logit_coefficient': float(platt.coef_[0, 0]), 'fitted_to': 'base_model_calibration_patients_only'}, 'test_auc': {key: float(roc_auc_score(y[test], out[key][test])) for key in prediction_names}}
        save_json(dest / 'fit_report.json', report)
        fold_reports.append(report)
        print(f"{name} fold {fold}/5 complete; rounds={rounds}; full raw AUC={report['test_auc']['primary_raw']:.6f}", flush=True)
    assert np.all(fold_ids > 0)
    assert all((np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)) for p in out.values()))
    assert all((len(np.unique(fold_ids[groups == g])) == 1 for g in np.unique(groups)))
    np.savez_compressed(folder / 'predictions.npz', y=y, groups=groups, original=original, fold=fold_ids, **out)
    public = {'run': name, 'cohort': cohort, 'window_hours': spec['hours'], 'started_at': started, 'completed_at': now(), 'seconds': time.monotonic() - clock, 'unit': 'admission' if cohort == 'MIMIC' else 'patient', 'rows': len(y), 'patients': len(np.unique(groups)), 'positive_labels': int(y.sum()), 'input': file_identity(data_path), 'input_features': names, 'explicitly_excluded_features': spec['drop_features'], 'roles_identity': file_identity(folder / 'patient_roles.npz'), 'prediction_identity': file_identity(folder / 'predictions.npz'), 'auc_point_estimates': {'original': float(roc_auc_score(y, original)), **{key: float(roc_auc_score(y, value)) for key, value in out.items()}}, 'fold_reports': fold_reports, 'status': 'training_complete_not_yet_independently_replayed', 'interpretation': 'CHARTTIME measurement-window recorded-diagnosis classification on the original admissions and fixed patient roles; not result-available, event-free or prospective validation'}
    save_json(ROOT / 'results' / f'{name}_training.json', public)
    print(json.dumps({'completed_run': name, 'rows': len(y), 'patients': len(np.unique(groups)), 'auc': public['auc_point_estimates'], 'seconds': public['seconds']}), flush=True)

def self_test():
    config = _release_resolve_config(json.loads(PROTOCOL.read_text(encoding='utf-8')))
    groups = np.repeat(np.arange(100), 2)
    y = np.repeat(np.arange(100) % 2, 2)
    r1 = patient_roles(y, groups, 'MIMIC', config)
    r2 = patient_roles(y, groups, 'MIMIC', config)
    assert np.array_equal(r1, r2)
    Xfit = np.array([[1.0, np.nan, 0.0], [3.0, np.nan, np.nan]], dtype=np.float32)
    Xtest = np.array([[np.nan, 99.0, 7.0]], dtype=np.float32)
    prep = fit_preprocessor(Xfit, 'MIMIC')
    assert prep['columns'].tolist() == [0, 2]
    assert np.array_equal(transform(prep, Xtest), [[2.0, 7.0]])
    cn = fit_preprocessor(Xfit, 'SCH')
    assert np.array_equal(transform(cn, Xfit), [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert not np.isnan(Xfit)[0, 2] and np.isnan(Xfit)[1, 2]
    pars = xgb_parameters(np.array([0, 0, 0, 1]), 'SCH', 1, config, 1)
    assert pars['scale_pos_weight'] == 3
    print('Synthetic role, training-only imputation, true-zero missingness and fit-only weight checks passed.', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='all')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    config = _release_resolve_config(json.loads(PROTOCOL.read_text(encoding='utf-8')))
    selected = [s for s in config['runs'] if args.run == 'all' or s['name'] == args.run]
    if not selected:
        raise ValueError('Unknown pre-specified run')
    self_test()
    lock_hash = freeze(config)
    for spec in selected:
        run_one(spec, config, args.threads, lock_hash)
if __name__ == '__main__':
    main()
