"""Independent saved-model replay. No training module is imported.

--self-test reads synthetic arrays only and writes nothing. --audit requires
all six runs and writes only evidence/validation_replay_audit.json.
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
ROOT = Path(__file__).resolve().parent
SNAPSHOT = Path(_release_path('project', 'workflow_snapshot.json'))
OUTPUT = ROOT / 'evidence' / 'validation_replay_audit.json'
RUNS = ('MIMIC_0h_original', 'MIMIC_0h_exclude_AST_HDL', 'MIMIC_24h_original', 'MIMIC_24h_exclude_AST_HDL', 'MIMIC_48h_original', 'MIMIC_48h_exclude_AST_HDL')
KEYS = ('primary_raw', 'base_raw', 'base_platt', 'xgb_full', 'rf_full', 'logistic', 'missingness', 'age')
ATOL = 1e-10

class AuditFailure(RuntimeError):
    """Only fixed audit context, never patient values, is included."""

def check(condition, context):
    if not bool(condition):
        raise AuditFailure(context)

def sha(path):
    return None

def identity(path):
    p = Path(path).resolve()
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def read_json(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8')))

def canonical(value):
    return _release_resolve_config(json.loads(json.dumps(value)))

def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'joblib')}}

def compare(actual, expected, context, atol=ATOL, rtol=0.0):
    a, b = (np.asarray(actual), np.asarray(expected))
    check(a.shape == b.shape, context + ': shape mismatch')
    check(np.isfinite(a).all() and np.isfinite(b).all(), context + ': nonfinite')
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    count = int(np.count_nonzero(error > atol + rtol * np.abs(b)))
    check(count == 0, context + ': numeric mismatch')
    return {'values': int(a.size), 'max_absolute_error': float(error.max(initial=0)), 'mismatched_values': count, 'atol': atol, 'rtol': rtol}

def verify_file(record, expected_path, context):
    check(Path(record['path']).resolve() == Path(expected_path).resolve(), context + ': path mismatch')
    now = identity(expected_path)
    return now

def reconstruct_roles(y, groups, cohort, config):
    """Derive stratification using independent patient-level reductions."""
    if cohort == 'MIMIC':
        splitter = StratifiedGroupKFold(n_splits=config['outer_folds'], shuffle=True, random_state=config['outer_seed'])
        outer = splitter.split(np.zeros(len(y)), y, groups)
    else:
        check(len(np.unique(groups)) == len(y), 'China must have one row per patient')
        splitter = StratifiedKFold(n_splits=config['outer_folds'], shuffle=True, random_state=config['outer_seed'])
        outer = splitter.split(np.zeros(len(y)), y)

    def divide(rows, fraction, seed):
        patients, inverse = np.unique(groups[rows], return_inverse=True)
        strata = np.zeros(len(patients), dtype=np.int64)
        np.maximum.at(strata, inverse, y[rows])
        remaining, held = train_test_split(patients, test_size=fraction, stratify=strata, random_state=seed)
        mask = np.isin(groups[rows], held)
        check(np.isin(groups[rows][~mask], remaining).all(), 'Patient split membership')
        return (rows[~mask], rows[mask])
    roles = np.full((config['outer_folds'], len(y)), -1, dtype=np.int8)
    for fold, (train, test) in enumerate(outer, 1):
        base, cal = divide(train, config['calibration_fraction_of_outer_train_patients'], config['calibration_seed_base'] + fold)
        inner, stop = divide(base, config['inner_validation_fraction_of_base_fit_patients'], config['inner_seed_base'] + fold)
        for code, ix in enumerate((test, cal, inner, stop)):
            roles[fold - 1, ix] = code
    return roles

def verify_roles(roles, expected, y, groups, context):
    check(np.array_equal(roles, expected), context + ': protocol split differs')
    check(np.isin(roles, np.arange(4)).all(), context + ': invalid role')
    check(np.all((roles == 0).sum(axis=0) == 1), context + ': test count not one')
    folds = np.argmax(roles == 0, axis=0) + 1
    patients, inverse = np.unique(groups, return_inverse=True)
    lower = np.full(len(patients), roles.shape[0] + 1, dtype=np.int64)
    upper = np.zeros(len(patients), dtype=np.int64)
    np.minimum.at(lower, inverse, folds)
    np.maximum.at(upper, inverse, folds)
    check(np.array_equal(lower, upper), context + ': patient has multiple test folds')
    summaries = []
    for fold, role in enumerate(roles, 1):
        sets = [np.unique(groups[role == code]) for code in range(4)]
        intersections = [int(len(np.intersect1d(sets[a], sets[b]))) for a in range(4) for b in range(a + 1, 4)]
        check(sum(intersections) == 0, context + ': role patient overlap')
        indices = {'test': np.flatnonzero(role == 0), 'calibration': np.flatnonzero(role == 1), 'inner_fit': np.flatnonzero(role == 2), 'inner_stop': np.flatnonzero(role == 3), 'base_fit': np.flatnonzero(role >= 2), 'full_outer_train': np.flatnonzero(role != 0)}
        check(all((np.unique(y[ix]).tolist() == [0, 1] for ix in indices.values())), context + ': one-class role')
        counts = {name: {'rows': len(ix), 'patients': len(np.unique(groups[ix])), 'positive_labels': int(y[ix].sum())} for name, ix in indices.items()}
        summaries.append({'fold': fold, 'role_counts': counts, 'six_pair_patient_intersections': intersections})
    return (folds, summaries)

def expected_columns(Xfit, cohort):
    return np.flatnonzero(np.isfinite(Xfit).any(axis=0)) if cohort == 'MIMIC' else np.arange(Xfit.shape[1])

def verify_preprocessor(prep, Xfit, cohort, context):
    columns = np.asarray(prep['columns'])
    check(np.issubdtype(columns.dtype, np.integer), context + ': column dtype')
    check(np.array_equal(columns, expected_columns(Xfit, cohort)), context + ': fitting feature selection')
    check(bool(prep['zero_fill']) == (cohort != 'MIMIC'), context + ': fill mode')
    if cohort != 'MIMIC':
        check(prep['imputer'] is None, context + ': unexpected China imputer')
        return {'mode': 'original_constant_zero', 'columns': len(columns)}
    imputer = prep['imputer']
    check(isinstance(imputer, SimpleImputer), context + ': imputer type')
    check(imputer.strategy == 'median' and imputer.keep_empty_features and (not imputer.add_indicator), context + ': imputer configuration')
    check(imputer.n_features_in_ == len(columns), context + ': imputer dimension')
    difference = compare(imputer.statistics_, np.nanmedian(Xfit[:, columns], axis=0), context + ': fitting medians', atol=0)
    return {'mode': 'fit_role_median', 'columns': len(columns), 'median_check': difference}

def transform(prep, X):
    """Rebuild filling independently instead of calling the training helper."""
    values = X[:, np.asarray(prep['columns'])].copy()
    if prep['zero_fill']:
        values[~np.isfinite(values)] = 0
        return values
    rows, columns = np.where(np.isnan(values))
    values[rows, columns] = np.asarray(prep['imputer'].statistics_)[columns]
    check(np.isfinite(values).all(), 'Reconstructed transform not finite')
    return values.astype(np.float32)

def verify_xgb_parameters(params, yfit, cohort, fold, config, context):
    ratio = float(np.count_nonzero(yfit == 0) / np.count_nonzero(yfit == 1))
    expected = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': config['xgb']['max_depth'], 'learning_rate': config['xgb']['learning_rate'], 'subsample': config['xgb']['subsample'], 'colsample_bytree': config['xgb']['colsample_bytree'], 'scale_pos_weight': ratio, 'seed': 42 if cohort != 'MIMIC' else fold, 'verbosity': 0, 'tree_method': 'hist'}
    check(set(params) == set(expected) | {'nthread'}, context + ': XGB parameter keys')
    check(all((params[k] == v for k, v in expected.items())), context + ': XGB protocol/class weight')
    check(isinstance(params['nthread'], int) and params['nthread'] > 0, context + ': thread metadata')
    return ratio

def load_booster(path, columns, threads, context):
    booster = xgb.Booster()
    booster.load_model(path)
    check(booster.num_features() == columns, context + ': booster feature dimension')
    check(_release_resolve_config(json.loads(booster.save_config()))['learner']['objective']['name'] == 'binary:logistic', context + ': booster objective')
    booster.set_param({'nthread': threads})
    return booster

def verify_inner(folder, X, y, inner, stop, cohort, fold, config, selection, threads):
    context = folder.parent.name + '/' + folder.name + '/inner'
    prep = joblib.load(folder / 'inner_preprocessor.joblib')
    prep_audit = verify_preprocessor(prep, X[inner], cohort, context)
    saved = read_json(folder / 'inner_evaluation.json')
    ratio = verify_xgb_parameters(saved['parameters'], y[inner], cohort, fold, config, context)
    check(selection['inner_fitted_columns'] == prep['columns'].tolist(), context + ': inner column metadata')
    check(selection['inner_scale_pos_weight'] == ratio, context + ': inner weight metadata')
    check(set(saved['history']) == {'inner_stop'} and set(saved['history']['inner_stop']) == {'auc'}, context + ': history evaluation datasets')
    history = np.asarray(saved['history']['inner_stop']['auc'], dtype=float)
    check(history.ndim == 1 and len(history) > 0 and np.isfinite(history).all(), context + ': invalid history')
    best = int(np.argmax(history))
    rounds = best + 1
    check(saved['best_iteration'] == best and saved['selected_rounds'] == rounds and (selection['selected_rounds'] == rounds), context + ': earliest maximum/best iteration')
    check(1 <= rounds <= len(history) <= config['xgb']['max_rounds'], context + ': history length')
    patience = config['xgb']['early_stopping_rounds']
    running_best, running_value = (0, -np.inf)
    trigger = None
    for iteration, value in enumerate(history):
        if value > running_value:
            running_best, running_value = (iteration, value)
        if iteration - running_best >= patience:
            trigger = iteration
            break
    if len(history) < config['xgb']['max_rounds']:
        check(trigger == len(history) - 1, context + ': early stopping timing')
    else:
        check(trigger is None or trigger == len(history) - 1, context + ': earlier stopping ignored')
    booster = load_booster(folder / 'inner_stop_model.ubj', len(prep['columns']), threads, context)
    check(booster.num_boosted_rounds() == len(history), context + ': saved inner tree/history count')
    check(booster.best_iteration == best, context + ': saved booster best_iteration')
    matrix = xgb.DMatrix(transform(prep, X[stop]), nthread=threads)
    replay_auc = np.asarray([roc_auc_score(y[stop], booster.predict(matrix, iteration_range=(0, n))) for n in range(1, len(history) + 1)])
    errors = compare(replay_auc, history, context + ': full stopping AUC history', atol=1e-12)
    return {'preprocessing': prep_audit, 'selected_rounds': rounds, 'saved_rounds': len(history), 'history_argmax_and_early_stopping_verified': True, 'all_recorded_inner_auc_replayed': errors, 'declared_inner_scale_pos_weight': ratio}

def verify_model(folder, prefix, Xfit, yfit, names, cohort, fold, config, report, threads):
    context = folder.parent.name + '/' + folder.name + '/' + prefix
    package = joblib.load(folder / (prefix + '_package.joblib'))
    check(set(package) == {'preprocessor', 'rf', 'parameters', 'input_feature_names', 'fitted_feature_names'}, context + ': package keys')
    prep = package['preprocessor']
    preprocessing = verify_preprocessor(prep, Xfit, cohort, context)
    fitted_names = [names[int(j)] for j in prep['columns']]
    check(package['input_feature_names'] == names and package['fitted_feature_names'] == fitted_names, context + ': feature name order')
    check(report[prefix + '_fitted_feature_names'] == fitted_names, context + ': report feature order')
    check(canonical(package['parameters']) == report[prefix + '_parameters'], context + ': parameter reports')
    ratio = verify_xgb_parameters(package['parameters']['xgb_parameters'], yfit, cohort, fold, config, context)
    rf = package['rf']
    check(canonical(rf.get_params()) == canonical(package['parameters']['rf_parameters']), context + ': RF parameters')
    check(all((rf.get_params()[key] == value for key, value in config['rf'].items())), context + ': RF protocol')
    weight = {0: 1, 1: int(ratio)} if cohort != 'MIMIC' else 'balanced'
    check(rf.class_weight == weight, context + ': fitting-role RF class weight')
    check(rf.random_state == (42 if cohort != 'MIMIC' else fold), context + ': RF random seed')
    check(len(rf.estimators_) == config['rf']['n_estimators'], context + ': RF tree count')
    check(rf.n_features_in_ == len(fitted_names) and all((t.n_features_in_ == len(fitted_names) for t in rf.estimators_)), context + ': RF feature dimension')
    check(np.array_equal(rf.classes_, [0, 1]), context + ': RF class order')
    booster = load_booster(folder / (prefix + '_xgb.ubj'), len(fitted_names), threads, context)
    rounds = report['selection']['selected_rounds']
    check(booster.num_boosted_rounds() == rounds, context + ': fixed boosted rounds')
    rf.n_jobs = threads
    return ((prep, booster, rf), {'preprocessing': preprocessing, 'feature_order_verified': True, 'saved_boosted_rounds': rounds, 'rf_trees': len(rf.estimators_), 'fitting_rows': len(yfit), 'fitting_positive_labels': int(yfit.sum()), 'declared_xgb_scale_pos_weight': ratio, 'saved_rf_class_weight': weight})

def predict_model(model, X, threads):
    prep, booster, rf = model
    values = transform(prep, X)
    xp = booster.predict(xgb.DMatrix(values, nthread=threads))
    rp = rf.predict_proba(values)[:, 1]
    return (0.5 * xp + 0.5 * rp, xp, rp)

def logit(p):
    p = np.clip(p, 1e-06, 1 - 1e-06)
    return np.log(p / (1 - p)).reshape(-1, 1)

def verify_logistic(model, context, platt=False):
    params = model.get_params()
    check(params['C'] == 1.0 and params['max_iter'] == 5000 and (params['tol'] == (1e-08 if platt else 1e-06)), context + ': logistic protocol')
    check(params['class_weight'] is None and params['fit_intercept'], context + ': logistic configuration')
    check(np.array_equal(model.classes_, [0, 1]), context + ': logistic classes')
    check(int(model.n_iter_.max()) < model.max_iter, context + ': logistic iteration limit')
    return {'iterations': int(model.n_iter_.max()), 'max_iter': model.max_iter, 'iteration_limit_reached': False}

def verify_scaler(scaler, Xfit, context):
    check(scaler.with_mean and scaler.with_std and (scaler.n_features_in_ == Xfit.shape[1]), context + ': scaler configuration')
    check(np.all(np.asarray(scaler.n_samples_seen_) == len(Xfit)), context + ': scaler fitting row count')
    values = Xfit.astype(np.float64)
    compare(scaler.mean_, values.mean(axis=0), context + ': fitting means', 1e-10, 1e-10)
    compare(scaler.var_, values.var(axis=0), context + ': fitting variance', 1e-10, 1e-09)
    check(np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all(), context + ': scaler scale')

def verify_comparators(bundle, X, full, age_index, full_transformed, report, context):
    check(set(bundle) == {'platt', 'logistic', 'missingness', 'age', 'age_column'}, context + ': comparator bundle')
    check(bundle['age_column'] == age_index, context + ': age index')
    convergence = {'platt': verify_logistic(bundle['platt'], context + '/platt', platt=True)}
    check(bundle['platt'].n_features_in_ == 1, context + ': platt feature count')
    fits = {'logistic': full_transformed, 'missingness': np.isnan(X[full]).astype(np.float32)}
    for name, values in fits.items():
        pipe = bundle[name]
        check(list(pipe.named_steps) == ['standardscaler', 'logisticregression'], context + ': pipeline order')
        verify_scaler(pipe.named_steps['standardscaler'], values, context + '/' + name)
        convergence[name] = verify_logistic(pipe[-1], context + '/' + name)
        check(pipe[-1].n_features_in_ == values.shape[1], context + ': comparator dimensions')
    age = bundle['age']
    check(list(age.named_steps) == ['simpleimputer', 'standardscaler', 'logisticregression'], context + ': age pipeline order')
    imputer = age.named_steps['simpleimputer']
    check(imputer.strategy == 'median' and imputer.keep_empty_features, context + ': age imputer')
    values = X[full, age_index].reshape(-1, 1).copy()
    median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
    compare(imputer.statistics_, [median], context + ': age fitting median', atol=0)
    values[np.isnan(values)] = median
    verify_scaler(age.named_steps['standardscaler'], values, context + '/age')
    convergence['age'] = verify_logistic(age[-1], context + '/age')
    check(age[-1].n_features_in_ == 1, context + ': age dimension')
    check(convergence == report['logistic_convergence'], context + ': reported convergence')
    compare([bundle['platt'].intercept_[0], bundle['platt'].coef_[0, 0]], [report['platt']['intercept'], report['platt']['logit_coefficient']], context + ': platt report', atol=0)
    check(report['platt']['fitted_to'] == 'base_model_calibration_patients_only', context + ': calibration provenance')
    return {'fit_role_scalers_and_age_median_verified': True, 'convergence': convergence}

def required_paths(config):
    paths = [ROOT / 'protocol.json', _release_source_path(ROOT / 'run_windows.py'), ROOT / 'evidence' / 'analysis_lock.json']
    for spec in config['runs']:
        paths.append(ROOT / 'results' / (spec['name'] + '_training.json'))
        folder = ROOT / 'private' / 'runs' / spec['name']
        paths.extend([folder / 'patient_roles.npz', folder / 'predictions.npz'])
        for fold in range(1, config['outer_folds'] + 1):
            paths.extend((folder / f'fold_{fold}' / name for name in ('fit_report.json', 'base_xgb.ubj', 'base_package.joblib', 'full_xgb.ubj', 'full_package.joblib', 'comparators_and_platt.joblib', 'inner_stop_model.ubj', 'inner_preprocessor.joblib', 'inner_evaluation.json')))
    return paths

def snapshot_check():
    value = read_json(SNAPSHOT)
    records = value['artifacts'] + value['analysis_inputs'] + value['analysis_scripts']
    check(len(records) == 58, 'Original snapshot must contain 58 files')
    verified = []
    for record in records:
        path = Path(record['path'])
        if not path.is_absolute():
            path = SNAPSHOT.parent / path
        check(path.is_file(), 'Original tracked file missing')
        now = identity(path)
        verified.append(now)
    return {'snapshot': identity(SNAPSHOT), 'files': verified, 'tracked_files': 58, 'all_match': True}

def replay_run(spec, config, lock_hash, threads):
    name, cohort = (spec['name'], spec['cohort'])
    folder = ROOT / 'private' / 'runs' / name
    public = read_json(ROOT / 'results' / (name + '_training.json'))
    check(public['status'] == 'training_complete_not_yet_independently_replayed', name + ': status')
    check(public['run'] == name and public['cohort'] == cohort, name + ': run identity')
    data_path = ROOT / 'private' / 'prepared' / spec['matrix']
    verify_file(public['input'], data_path, name + ': prepared')
    verify_file(public['roles_identity'], folder / 'patient_roles.npz', name + ': roles')
    verify_file(public['prediction_identity'], folder / 'predictions.npz', name + ': predictions')
    with np.load(data_path, allow_pickle=False) as saved:
        check(set(saved.files) == {'X', 'y', 'groups', 'p_original', 'feature_names'}, name + ': prepared keys')
        X, y, groups = (saved['X'].copy(), saved['y'].copy(), saved['groups'].copy())
        original, names = (saved['p_original'].copy(), saved['feature_names'].tolist())
    check(X.dtype == np.float32 and y.dtype == np.int64 and (groups.dtype == np.int64) and (original.dtype == np.float64), name + ': prepared dtype')
    check(X.ndim == 2 and y.shape == groups.shape == original.shape == (len(X),), name + ': shapes')
    check(len(names) == X.shape[1] and len(set(names)) == len(names), name + ': names')
    check(np.unique(y).tolist() == [0, 1], name + ': labels')
    check(np.isfinite(original).all() and np.all((original >= 0) & (original <= 1)), name + ': original probability')
    X[~np.isfinite(X)] = np.nan
    check(all((n in names for n in spec['drop_features'])), name + ': absent exclusion')
    keep = [j for j, n in enumerate(names) if n not in spec['drop_features']]
    X, names = (X[:, keep], [names[j] for j in keep])
    check(public['input_features'] == names and public['explicitly_excluded_features'] == spec['drop_features'], name + ': feature order/exclusions')
    check(public['rows'] == len(y) and public['patients'] == len(np.unique(groups)) and (public['positive_labels'] == int(y.sum())), name + ': denominator')
    check(public['unit'] == ('admission' if cohort == 'MIMIC' else 'patient'), name + ': analysis unit')
    age_index = names.index('Age' if cohort == 'MIMIC' else '年龄')
    with np.load(folder / 'patient_roles.npz', allow_pickle=False) as saved:
        check(set(saved.files) == {'roles', 'groups', 'y'}, name + ': role keys')
        check(np.array_equal(saved['y'], y) and np.array_equal(saved['groups'], groups), name + ': role identity arrays')
        roles = saved['roles'].copy()
    folds, summaries = verify_roles(roles, reconstruct_roles(y, groups, cohort, config), y, groups, name)
    with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
        check(set(saved.files) == set(KEYS) | {'y', 'groups', 'original', 'fold'}, name + ': prediction keys')
        for key, values in (('y', y), ('groups', groups), ('original', original), ('fold', folds)):
            check(np.array_equal(saved[key], values), name + ': prediction identity/order')
        oof = {key: saved[key].copy() for key in KEYS}
    for values in oof.values():
        check(values.shape == y.shape and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)), name + ': invalid OOF')
    replay = {key: np.full(len(y), np.nan) for key in KEYS}
    check(len(public['fold_reports']) == config['outer_folds'], name + ': reported fold count')
    audits = []
    for fold, role in enumerate(roles, 1):
        dest = folder / f'fold_{fold}'
        context = name + '/' + dest.name
        report = read_json(dest / 'fit_report.json')
        check(report == public['fold_reports'][fold - 1], context + ': fold report duplicate')
        check(report['fold'] == fold and report['roles'] == summaries[fold - 1]['role_counts'], context + ': role counts')
        check(report['all_disjoint_role_patient_intersections_zero'], context + ': overlap declaration')
        test, cal, inner, stop = [np.flatnonzero(role == r) for r in range(4)]
        base, full = (np.flatnonzero(role >= 2), np.flatnonzero(role != 0))
        inner_audit = verify_inner(dest, X, y, inner, stop, cohort, fold, config, report['selection'], threads)
        base_model, base_audit = verify_model(dest, 'base', X[base], y[base], names, cohort, fold, config, report, threads)
        full_model, full_audit = verify_model(dest, 'full', X[full], y[full], names, cohort, fold, config, report, threads)
        bundle = joblib.load(dest / 'comparators_and_platt.joblib')
        comparator_audit = verify_comparators(bundle, X, full, age_index, transform(full_model[0], X[full]), report, context)
        primary, xp, rp = predict_model(full_model, X[test], threads)
        base_p = predict_model(base_model, X[test], threads)[0]
        outputs = {'primary_raw': primary, 'base_raw': base_p, 'base_platt': bundle['platt'].predict_proba(logit(base_p))[:, 1], 'xgb_full': xp, 'rf_full': rp, 'logistic': bundle['logistic'].predict_proba(transform(full_model[0], X[test]))[:, 1], 'missingness': bundle['missingness'].predict_proba(np.isnan(X[test]).astype(np.float32))[:, 1], 'age': bundle['age'].predict_proba(X[test, age_index].reshape(-1, 1))[:, 1]}
        errors = {}
        for key, values in outputs.items():
            errors[key] = compare(values, oof[key][test], context + ':' + key)
            replay[key][test] = values
            compare([roc_auc_score(y[test], values)], [report['test_auc'][key]], context + ': fold AUC', 1e-12)
        cal_p = predict_model(base_model, X[cal], threads)[0]
        cal_x = logit(cal_p)[:, 0]
        residual = bundle['platt'].predict_proba(cal_x.reshape(-1, 1))[:, 1] - y[cal]
        score = [float(residual.mean()), float(np.mean(cal_x * residual) + bundle['platt'].coef_[0, 0] / len(cal))]
        audits.append({**summaries[fold - 1], 'inner_selection': inner_audit, 'base': base_audit, 'full': full_audit, 'comparators': comparator_audit, 'prediction_replay': errors, 'calibration_role_logistic_score_max_abs': float(np.max(np.abs(score)))})
        print(context + ': saved models and eight OOF columns agree', flush=True)
    errors = {key: compare(replay[key], oof[key], name + ': OOF ' + key) for key in KEYS}
    auc = {'original': float(roc_auc_score(y, original)), **{key: float(roc_auc_score(y, p)) for key, p in replay.items()}}
    for key, value in auc.items():
        compare([value], [public['auc_point_estimates'][key]], name + ': aggregate AUC', 1e-12)
    return ({'run': name, 'cohort': cohort, 'passed': True, 'rows': len(y), 'patients': len(np.unique(groups)), 'positive_labels': int(y.sum()), 'features': len(names), 'each_patient_tested_once': True, 'protocol_roles_reproduced_exactly': True, 'prepared_identity_arrays_match_exactly': True, 'prediction_replay': errors, 'auc_point_estimates': auc, 'folds': audits}, roles)

def audit(config, paths, threads):
    before = [identity(p) for p in paths]
    lock_path = ROOT / 'evidence' / 'analysis_lock.json'
    lock = read_json(lock_path)
    check(lock['protocol'] == config, 'Frozen protocol differs')
    check(lock['state']['environment'] == environment(), 'Frozen runtime differs')
    locked = [verify_file(record, record['path'], 'Frozen source') for record in lock['state']['files']]
    originals = snapshot_check()
    results, mimic = ([], [])
    for spec in config['runs']:
        result, roles = replay_run(spec, config, sha(lock_path), threads)
        results.append(result)
        if spec['cohort'] == 'MIMIC':
            mimic.append(roles)
    check(len(mimic) == 6 and all((np.array_equal(mimic[0], role) for role in mimic[1:])), 'MIMIC runs use different roles')
    check(before == [identity(p) for p in paths], 'Replayed files changed during audit')
    for record in locked:
        verify_file(record, record['path'], 'Frozen source after replay')
    check(originals == snapshot_check(), 'Original snapshot changed during audit')
    return {'passed': True, 'status': 'completed_saved_model_replay', 'runs': results, 'mimic_runs_share_exact_patient_roles': True, 'all_eight_oof_columns_replayed_for_thirty_test_folds': True, 'all_inner_stopping_auc_histories_replayed': True, 'all_read_artifact_hashes_stable': True, 'frozen_sources': locked, 'replayed_artifacts': before, 'original_snapshot_verification': originals, 'limits': ['Independent saved-model replay and split verification, not independent clinical outcome adjudication.', 'Original endpoint, predictor timing, and native mapping limitations remain. This is internal revalidation, not prospective or external validation.', 'XGBoost fitting weights are checked against frozen code, role counts and package/evaluation metadata. Saved inference models alone do not prove historical training calls.', 'Fit-role scaler and imputer statistics are independently checked. Logistic and Platt coefficients are replayed, not refitted; coefficients alone cannot prove training provenance.', 'Model hashes are first recorded by this audit because training reports do not contain per-model hashes. All saved artifacts must remain unchanged during replay.']}

def self_test():
    config = {'outer_folds': 5, 'outer_seed': 42, 'calibration_fraction_of_outer_train_patients': 0.25, 'calibration_seed_base': 1042, 'inner_validation_fraction_of_base_fit_patients': 0.25, 'inner_seed_base': 2042}
    groups = np.repeat(np.arange(100), 2)
    y = np.repeat(np.arange(100) % 2, 2)
    y[1::6] = 0
    roles = reconstruct_roles(y, groups, 'MIMIC', config)
    verify_roles(roles, reconstruct_roles(y, groups, 'MIMIC', config), y, groups, 'synthetic MIMIC')
    cy, cg = (np.arange(100) % 2, np.arange(100))
    cr = reconstruct_roles(cy, cg, 'SCH', config)
    verify_roles(cr, cr.copy(), cy, cg, 'synthetic China')
    corrupted = roles.copy()
    corrupted[0, 0] = (int(corrupted[0, 0]) + 1) % 4

    def rejected(function):
        try:
            function()
        except AuditFailure:
            return True
        return False
    check(rejected(lambda: verify_roles(corrupted, roles, y, groups, 'synthetic corrupt roles')), 'Synthetic role corruption undetected')
    Xfit = np.array([[1, np.nan, 0], [3, np.nan, np.nan]], dtype=np.float32)
    Xtest = np.array([[np.nan, 99, 7]], dtype=np.float32)
    cols = np.array([0, 2])
    prep = {'zero_fill': False, 'columns': cols, 'imputer': SimpleImputer(strategy='median', keep_empty_features=True).fit(Xfit[:, cols])}
    verify_preprocessor(prep, Xfit, 'MIMIC', 'synthetic medians')
    compare(transform(prep, Xtest), [[2, 7]], 'synthetic transform', 0)
    cn = {'zero_fill': True, 'columns': np.arange(3), 'imputer': None}
    verify_preprocessor(cn, Xfit, 'SCH', 'synthetic zero')
    compare(transform(cn, Xfit), [[1, 0, 0], [3, 0, 0]], 'synthetic zero fill', 0)
    check(np.isnan(Xfit).sum() == 3, 'Source array mutated')
    prep['imputer'].statistics_[0] += 1
    check(rejected(lambda: verify_preprocessor(prep, Xfit, 'MIMIC', 'synthetic corrupt median')), 'Synthetic median corruption undetected')
    check(rejected(lambda: compare([0.5, 0.4], [0.5, 0.4001], 'synthetic corrupt probability')), 'Synthetic probability corruption undetected')
    print(json.dumps({'self_test_passed': True, 'real_patient_data_read': False, 'files_written': 0, 'synthetic_checks': 8}), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--audit', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    check(1 <= args.threads <= 16, 'Replay threads must be 1-16')
    if args.self_test:
        self_test()
        return
    check(not OUTPUT.exists(), 'Audit output exists; do not overwrite')
    config = read_json(ROOT / 'protocol.json')
    check(tuple((s['name'] for s in config['runs'])) == RUNS and config['outer_folds'] == 5, 'Audit requires all six fixed-protocol five-fold runs')
    paths = required_paths(config)
    check(all((p.is_file() for p in paths)), 'All completed runs required before reading private results')
    report = {'audit_script': identity(__file__), 'started_at': datetime.now().astimezone().isoformat(), 'environment': environment(), 'probability_absolute_tolerance': ATOL, 'no_training_module_imported': True, 'no_patient_identifiers_or_clinical_rows_exported': True}
    error = None
    try:
        report.update(audit(config, paths, args.threads))
    except Exception as exc:
        error = exc
        report.update({'passed': False, 'status': 'audit_failed', 'failure': str(exc) if isinstance(exc, AuditFailure) else type(exc).__name__})
    report['completed_at'] = datetime.now().astimezone().isoformat()
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': report['passed'], 'report': OUTPUT.as_posix()}), flush=True)
    if error is not None:
        raise SystemExit(1)
if __name__ == '__main__':
    try:
        main()
    except AuditFailure as exc:
        print('Audit preflight stopped: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
