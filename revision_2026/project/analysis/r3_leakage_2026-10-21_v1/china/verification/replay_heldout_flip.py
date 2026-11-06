"""Independent full-engine replay with every outer holdout label inverted."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
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
ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent / 'heldout_flip_v2'
TOLERANCE = 1e-12
THREADS = 2

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8')))

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def array_digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(str(a.dtype).encode('ascii'))
        h.update(json.dumps(a.shape).encode('ascii'))
        h.update(a.tobytes(order='C'))
    return h.hexdigest()

def rf_digest(model):
    arrays = [model.classes_]
    for tree in model.estimators_:
        state = tree.tree_.__getstate__()
        arrays.extend((state['nodes'][name] for name in state['nodes'].dtype.names))
        arrays.append(state['values'])
    return array_digest(arrays)

def logistic_digest(model):
    scaler, classifier = (model[0], model[-1])
    return array_digest([scaler.mean_, scaler.var_, scaler.scale_, scaler.n_samples_seen_, classifier.classes_, classifier.coef_, classifier.intercept_, classifier.n_iter_])

def difference(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = np.abs(a - b)
    return {'exact_equal': bool(np.array_equal(a, b)), 'max_absolute_difference': float(delta.max()), 'values_different': int((delta != 0).sum()), 'within_1e_minus_12': bool((delta <= TOLERANCE).all())}

def parameters(labels, center, fold, protocol):
    positive = int((labels == 1).sum())
    assert positive > 0
    result = {key: protocol['xgb'][key] for key in ('max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'tree_method')}
    result.update(objective='binary:logistic', eval_metric='auc', verbosity=0, seed=fold if center == 'MIMIC' else 42, nthread=THREADS, scale_pos_weight=float((labels == 0).sum() / positive))
    return result

def replay_fold(center, fold, X, y, groups, role, predictions, protocol, reference_dir):
    eligible = np.isfinite(X).all(axis=1)
    test = np.flatnonzero(eligible & (role == 0))
    train = np.flatnonzero(eligible & (role != 0))
    inner = np.flatnonzero(eligible & (role == 2))
    stop = np.flatnonzero(eligible & (role == 3))
    assert not set(groups[test]).intersection(groups[train])
    assert not set(groups[inner]).intersection(groups[stop])
    changed = y.copy()
    changed[test] = 1 - changed[test]
    assert int((changed != y).sum()) == len(test)
    assert np.array_equal(changed[train], y[train])
    assert np.array_equal(changed[inner], y[inner]) and np.array_equal(changed[stop], y[stop])
    assert all((len(np.unique(changed[ix])) == 2 for ix in (train, inner, stop)))
    history = {}
    early = xgb.train(parameters(changed[inner], center, fold, protocol), xgb.DMatrix(X[inner], label=changed[inner], nthread=THREADS), num_boost_round=protocol['xgb']['max_rounds'], evals=[(xgb.DMatrix(X[stop], label=changed[stop], nthread=THREADS), 'inner_stop')], early_stopping_rounds=protocol['xgb']['early_stopping_rounds'], evals_result=history, verbose_eval=False)
    rounds = int(early.best_iteration) + 1
    final_params = parameters(changed[train], center, fold, protocol)
    booster = xgb.train(final_params, xgb.DMatrix(X[train], label=changed[train], nthread=THREADS), num_boost_round=rounds)
    weight = 'balanced' if center == 'MIMIC' else {0: 1, 1: int((changed[train] == 0).sum() / (changed[train] == 1).sum())}
    forest = RandomForestClassifier(**protocol['rf'], class_weight=weight, random_state=fold if center == 'MIMIC' else 42, n_jobs=THREADS)
    forest.fit(X[train], changed[train])
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, tol=1e-08))
    logistic.fit(X[train], changed[train])
    assert logistic[-1].n_iter_.max() < 5000
    old_report = load(reference_dir / 'report.json')
    old_booster = xgb.Booster()
    old_booster.load_model(reference_dir / 'reference_xgb.ubj')
    old_bundle = joblib.load(reference_dir / 'reference.joblib')
    dtest = xgb.DMatrix(X[test], nthread=THREADS)
    new_xgb = booster.predict(dtest)
    old_xgb = old_booster.predict(dtest)
    new_rf = forest.predict_proba(X[test])[:, 1]
    old_rf = old_bundle['rf'].predict_proba(X[test])[:, 1]
    new_lr = logistic.predict_proba(X[test])[:, 1]
    old_lr = old_bundle['logistic'].predict_proba(X[test])[:, 1]
    new_ensemble = 0.5 * new_xgb + 0.5 * new_rf
    deltas = {'xgboost_vs_saved_model': difference(new_xgb, old_xgb), 'random_forest_vs_saved_model': difference(new_rf, old_rf), 'logistic_vs_saved_model': difference(new_lr, old_lr), 'ensemble_vs_original_holdout_predictions': difference(new_ensemble, predictions['strict_ensemble'][test]), 'logistic_vs_original_holdout_predictions': difference(new_lr, predictions['strict_logistic'][test])}
    hashes = {'inner_xgboost_original': None, 'inner_xgboost_flipped': hashlib.sha256(early.save_raw(raw_format='ubj')).hexdigest(), 'final_xgboost_original': None, 'final_xgboost_flipped': hashlib.sha256(booster.save_raw(raw_format='ubj')).hexdigest(), 'random_forest_state_original': rf_digest(old_bundle['rf']), 'random_forest_state_flipped': rf_digest(forest), 'scaler_logistic_state_original': logistic_digest(old_bundle['logistic']), 'scaler_logistic_state_flipped': logistic_digest(logistic)}
    model_matches = {name: hashes[name + '_original'] == hashes[name + '_flipped'] for name in ('inner_xgboost', 'final_xgboost', 'random_forest_state', 'scaler_logistic_state')}
    result = {'center': center, 'fold': fold, 'outer_holdout_rows_flipped': len(test), 'outer_holdout_patients': len(np.unique(groups[test])), 'training_labels_unchanged': True, 'inner_and_stopping_labels_unchanged': True, 'patient_overlap_train_test': 0, 'selected_rounds_original': old_report['selected_rounds'], 'selected_rounds_flipped': rounds, 'selected_rounds_unchanged': old_report['selected_rounds'] == rounds, 'inner_metric_history_unchanged': history == old_report['inner_history'], 'outer_training_parameters_unchanged': final_params == old_report['parameters'], 'model_state_hashes': hashes, 'model_state_hash_matches': model_matches, 'random_forest_hash_definition': 'Ordered named node fields and value arrays for all 300 trees; excludes unused C-struct padding bytes.', 'prediction_comparisons': deltas, 'prediction_tolerance': TOLERANCE, 'tolerance_reason': 'Random-forest parallel floating-point reduction can change final summation by machine epsilon; tree states must match exactly.'}
    result['passed'] = bool(result['selected_rounds_unchanged'] and result['inner_metric_history_unchanged'] and result['outer_training_parameters_unchanged'] and all(model_matches.values()) and all((v['within_1e_minus_12'] for v in deltas.values())))
    return result

def main():
    DEST.mkdir(exist_ok=False)
    locks = {center: load(ROOT / f'private/{center}/input_lock.json') for center in ('SCH', 'WHU', 'MIMIC')}
    protocol = load(ROOT / 'protocol.json')
    expected = {}
    for center, lock in locks.items():
        assert protocol == lock['protocol']
        assert sys.executable == lock['environment']['executable']
        for package, expected_version in lock['environment']['packages'].items():
            assert version(package) == expected_version
        for item in lock['sources']:
            assert item['path'] not in expected or expected[item['path']] == item
            expected[item['path']] = item
    for path, item in expected.items():
        assert identity(path) == item, 'Locked source identity changed before replay'
    protected = dict(expected)
    for center in locks:
        for path in (ROOT / f'private/{center}').rglob('*'):
            if path.is_file():
                item = identity(path)
                protected[item['path']] = item
    started = datetime.now(timezone.utc).isoformat()
    before = list(protected.values())
    save(DEST / 'run_lock.json', {'started_utc': started, 'protected_inputs_and_models': before, 'all_training_input_lock_entries_verified': True, 'independent_implementation_does_not_import_root_training_code': True, 'thread_limit': THREADS, 'script': identity(__file__), 'environment': {'executable': sys.executable, 'python': sys.version, 'packages': {p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'joblib')}}, 'scope': 'Engineering outer-heldout label isolation check; not an absence-of-clinical-leakage guarantee.'})
    reports = []
    failure = None
    try:
        for center, lock in locks.items():
            source = next((i['path'] for i in lock['sources'] if i['path'].endswith(f'{center}_demographics.npz')))
            prepared = next((i['path'] for i in lock['sources'] if i['path'].endswith(f'private/prepared/{center}.npz')))
            rolepath = next((i['path'] for i in lock['sources'] if i['path'].endswith('patient_roles.npz')))
            data = dict(np.load(source, allow_pickle=False))
            original = dict(np.load(prepared, allow_pickle=False))
            roles = dict(np.load(rolepath, allow_pickle=False))
            prediction = dict(np.load(ROOT / f'private/{center}/predictions.npz', allow_pickle=False))
            X, groups, y = (data['X'].astype(np.float32), data['groups'], original['y'].astype(np.int8))
            assert X.shape == (len(y), 2)
            for evidence in (original, roles, prediction):
                assert np.array_equal(groups, evidence['groups'])
            assert np.array_equal(y, roles['y']) and np.array_equal(y, prediction['y'])
            assert np.array_equal(prediction['eligible'], np.isfinite(X).all(axis=1))
            for fold, role in enumerate(roles['roles'], 1):
                test = np.isfinite(X).all(axis=1) & (role == 0)
                assert np.array_equal(prediction['fold'] == fold, test)
                report = replay_fold(center, fold, X, y, groups, role, prediction, protocol, ROOT / f'private/{center}/fold_{fold}')
                reports.append(report)
                save(DEST / f'{center}_fold_{fold}.json', report)
                print(json.dumps({'center': center, 'fold': fold, 'passed': report['passed'], 'rounds': report['selected_rounds_flipped'], 'max_prediction_delta': max((d['max_absolute_difference'] for d in report['prediction_comparisons'].values()))}), flush=True)
                assert report['passed'], 'Replay comparison failed; aggregate per-fold details retained'
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
    after = [identity(item['path']) for item in before]
    preserved = before == after
    result = {'passed': failure is None and len(reports) == 15 and all((r['passed'] for r in reports)) and preserved, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'folds_completed': len(reports), 'expected_folds': 15, 'folds_passed': sum((r['passed'] for r in reports)), 'failure': failure, 'all_heldout_label_flips_preserve_inner_rounds': all((r['selected_rounds_unchanged'] for r in reports)), 'all_model_states_match_exactly': all((all(r['model_state_hash_matches'].values()) for r in reports)), 'maximum_holdout_prediction_delta': max((d['max_absolute_difference'] for r in reports for d in r['prediction_comparisons'].values()), default=None), 'prediction_tolerance': TOLERANCE, 'all_training_input_lock_entries_rehashed_after': True, 'protected_input_and_model_file_count': len(before), 'read_only_preservation_passed': preserved, 'post_run_identities': after, 'per_fold_reports': [identity(DEST / f"{r['center']}_fold_{r['fold']}.json") for r in reports], 'script': identity(__file__), 'original_code_protocol_and_models_modified': False, 'replay_models_or_patient_rows_persisted': False, 'scope': "Independent deterministic replay verifies outer-heldout labels do not affect this restricted reference's fitting, scaler, class weights or early stopping. It does not validate original clinical timing, cohort selection or outcomes."}
    save(DEST / 'verification.json', result)
    print(json.dumps({k: result[k] for k in ('passed', 'folds_completed', 'folds_passed', 'maximum_holdout_prediction_delta', 'read_only_preservation_passed', 'failure')}), flush=True)
    if not result['passed']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
