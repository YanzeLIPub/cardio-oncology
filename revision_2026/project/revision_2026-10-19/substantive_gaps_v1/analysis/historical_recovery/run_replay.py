"""Safely extract historical functions; persist a separate, fully identified replay."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / 'private'
SOURCE = Path(_release_path('pipeline', 'Manuscript/v13/compute_v13_metrics.py'))
PRED = Path(_release_path('pipeline', ''))
PATHS = {'SCH_RAW': Path(_release_path('china', 'SCH_processed_data.csv')), 'WHU_RAW': Path(_release_path('china', 'files/WHU_Cancer_processed_data.csv')), 'MACE_FILE': PRED / 'data/real_model_predictions_v2.csv', 'MIMIC_FT': PRED / 'MIMIC-3/data/features_model.csv', 'MIMIC_LB': PRED / 'MIMIC-3/data/mace_labels.csv', 'PAPER': PRED / 'Manuscript/v13/paper_predictions_9946.csv', 'TRANSFER': PRED / 'Manuscript/v13/transfer_metrics_9946.csv'}

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')

def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def tokens(values):
    return np.asarray([hashlib.sha256(('replay-20261019|' + str(v)).encode()).hexdigest() for v in values])

def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'joblib', 'shap', 'umap-learn', 'lightgbm']
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': versions, 'historical_runtime_proven': False, 'classification': 'exact metadata for this new run only'}

def save_prepared(center, data, feats, X, y):
    key = next((c for c in data.columns if c not in feats and c != 'MACE'))
    keys = tokens(data[key].tolist())
    assert len(np.unique(keys)) == len(keys)
    np.savez_compressed(PRIVATE / f'{center}_prepared.npz', X=X, y=y, features=np.asarray(feats), row_tokens=keys, groups=np.arange(len(y)))

def capture_model(label, common, Xtr, Xte, ytr, imp, booster, rf, proba):
    d = PRIVATE / label
    d.mkdir()
    booster.save_model(d / 'xgboost.ubj')
    joblib.dump({'imputer': imp, 'rf': rf, 'features': common}, d / 'preprocessor_rf.joblib', compress=3)
    np.savez_compressed(d / 'source_and_target.npz', X_source=Xtr, X_target=Xte, y_source=ytr)
    dump(f'{label}_model.json', {'analysis_id': 'historical_code_replay_20261019_v1', 'direction': label, 'common_nominal_features': common, 'nominal_feature_count': len(common), 'effective_source_features': list(np.asarray(common)[np.isfinite(imp.statistics_)]), 'source_all_missing_slots': list(np.asarray(common)[~np.isfinite(imp.statistics_)]), 'source_rows': len(ytr), 'source_positive_labels': int(ytr.sum()), 'target_rows': len(proba), 'fitted_on_source_only': True, 'cross_site_assay_equivalence_validated': False, 'rf_parameters': rf.get_params(), 'xgboost_configuration': _release_resolve_config(json.loads(booster.save_config())), 'saved_models': [identity(d / 'xgboost.ubj'), identity(d / 'preprocessor_rf.joblib')]})

def extract_functions():
    source = SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'FEATS_43_EN', 'ZH2EN'}:
                constants[name] = ast.literal_eval(node.value)
    constants['FEATS_43_ZH'] = list(constants['ZH2EN'])
    names = {'load_sch_patient', 'load_whu_patient', 'transfer_train_predict'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    records = []
    edited = []
    for n in nodes:
        record = {'function': n.name, 'start_line': n.lineno, 'end_line': n.end_lineno, 'original_source': ast.get_source_segment(source, n)}
        m = copy.deepcopy(n)
        assert isinstance(m.body[-1], ast.Return)
        if n.name == 'transfer_train_predict':
            statement = 'capture_model(label, common, Xtr, Xte, ytr, imp, booster, rf, proba)'
        else:
            center = 'SCH' if n.name == 'load_sch_patient' else 'WHU'
            statement = f"save_prepared('{center}', data, feats, X, y)"
        inserted = ast.parse(statement).body[0]
        m.body.insert(-1, inserted)
        stripped = copy.deepcopy(m)
        del stripped.body[-2]
        assert ast.dump(stripped, include_attributes=False) == ast.dump(n, include_attributes=False)
        record['instrumentation'] = statement
        record['removing_instrumentation_restores_original_AST'] = True
        records.append(record)
        edited.append(m)
    scope = {'np': np, 'pd': pd, 'xgb': xgb, 'SimpleImputer': SimpleImputer, 'RandomForestClassifier': RandomForestClassifier, 'roc_auc_score': roc_auc_score, 'capture_model': capture_model, 'save_prepared': save_prepared, **constants, **{k: str(v) for k, v in PATHS.items()}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=edited, type_ignores=[])), str(SOURCE), 'exec'), scope)
    dump('extracted_function_provenance.json', records)
    return scope

def bootstrap(y, p, groups, seed):
    _, gi = np.unique(groups, return_inverse=True)
    ng = int(gi.max()) + 1
    rng = np.random.default_rng(seed)
    order = np.argsort(p, kind='stable')
    yy, gg, pp = (y[order], gi[order], p[order])
    starts = np.r_[0, np.flatnonzero(np.diff(pp)) + 1]
    values = []
    for _ in range(2000):
        counts = np.bincount(rng.integers(0, ng, ng), minlength=ng)
        weights = counts[gg]
        pos = np.add.reduceat(weights * yy, starts)
        neg = np.add.reduceat(weights * (1 - yy), starts)
        den = pos.sum() * neg.sum()
        if den:
            values.append(float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / den))
    lo, hi = np.quantile(values, [0.025, 0.975])
    return (float(lo), float(hi), np.asarray(values))

def main():
    resume = '--resume' in sys.argv
    if (ROOT / 'replay_metrics.csv').exists() or ((PRIVATE / 'SCH_prepared.npz').exists() and (not resume)):
        raise RuntimeError('Refusing to overwrite an earlier fitted replay.')
    PRIVATE.mkdir(exist_ok=True)
    inputs = [identity(p) for p in [SOURCE, ROOT / 'protocol.json', Path(__file__), *PATHS.values()]]
    if not resume:
        dump('analysis_lock.json', {'locked_before_fit': True, 'utc': datetime.now(timezone.utc).isoformat(), 'environment': environment(), 'inputs': inputs})
    else:
        old = _release_resolve_config(json.loads((ROOT / 'analysis_lock.json').read_text()))
        assert all((identity(item['path']) == item for item in old['inputs'] if item['path'] != Path(__file__).as_posix()))
        dump('verification_amendment.json', {'reason': 'Fix verifier-only float32 in-place addition; original model and probability computations unchanged', 'original_lock_unchanged': True, 'current_script': identity(__file__), 'utc': datetime.now(timezone.utc).isoformat()})
    scope = extract_functions()
    if resume:
        zs, zw = (np.load(PRIVATE / 'SCH_prepared.npz'), np.load(PRIVATE / 'WHU_prepared.npz'))
        x_s, y_s, f_s = (zs['X'], zs['y'], zs['features'].tolist())
        x_w, y_w, f_w = (zw['X'], zw['y'], zw['features'].tolist())
    else:
        x_s, y_s, f_s = scope['load_sch_patient']()
        x_w, y_w, f_w = scope['load_whu_patient']()
    ft, lb = (pd.read_csv(PATHS['MIMIC_FT']), pd.read_csv(PATHS['MIMIC_LB']))
    if 'HOSPITAL_EXPIRE_FLAG' not in ft.columns:
        df = ft.merge(lb[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'MACE']], on='HADM_ID', how='inner')
    else:
        df = ft.merge(lb[['HADM_ID', 'MACE']], on='HADM_ID', how='inner')
    df = df[df['HOSPITAL_EXPIRE_FLAG'] == 0].copy()
    f_m = [f for f in scope['FEATS_43_EN'] if f in df.columns]
    for f in f_m:
        df[f] = pd.to_numeric(df[f], errors='coerce')
    x_m, y_m = (df[f_m].to_numpy(dtype=float), df.MACE.to_numpy(dtype=int))
    paper = pd.read_csv(PATHS['PAPER'])
    assert np.array_equal(df.HADM_ID.to_numpy(), paper.HADM_ID.to_numpy())
    assert np.array_equal(y_m, paper.MACE.to_numpy())
    assert np.array_equal(df.SUBJECT_ID.to_numpy(), paper.SUBJECT_ID.to_numpy())
    groups = pd.factorize(df.SUBJECT_ID, sort=True)[0]
    assert (len(y_m), int(y_m.sum()), len(np.unique(groups))) == (9946, 4462, 6783)
    assert (len(y_s), int(y_s.sum()), len(y_w), int(y_w.sum())) == (38509, 1323, 2562, 281)
    if not resume:
        np.savez_compressed(PRIVATE / 'MIMIC_prepared.npz', X=x_m, y=y_m, features=np.asarray(f_m), row_tokens=tokens(df.HADM_ID), groups=groups)
    else:
        zm = np.load(PRIVATE / 'MIMIC_prepared.npz')
        assert np.array_equal(x_m, zm['X'], equal_nan=True)
        assert np.array_equal(y_m, zm['y'])
    cohorts = {'SCH': (x_s, y_s, f_s), 'WHU': (x_w, y_w, f_w), 'MIMIC': (x_m, y_m, f_m)}
    hist = {'SCH_to_MIMIC': 0.4998, 'WHU_to_MIMIC': 0.4938, 'MIMIC_to_SCH': 0.5591, 'MIMIC_to_WHU': 0.444}
    prepared = []
    for c, (X, y, feats) in cohorts.items():
        z = np.load(PRIVATE / f'{c}_prepared.npz')
        prepared.append({'cohort': c, 'rows': len(y), 'patients': len(np.unique(z['groups'])), 'positive_labels': int(y.sum()), 'features': feats, 'file': identity(PRIVATE / f'{c}_prepared.npz')})
    dump('prepared_identity.json', prepared)
    rows = []
    for i, (label, old_auc) in enumerate(hist.items(), 1):
        source, target = label.split('_to_')
        xs, ys, fs = cohorts[source]
        xt, yt, ftar = cohorts[target]
        zt = np.load(PRIVATE / f'{target}_prepared.npz')
        if resume and (PRIVATE / label / 'predictions.npz').exists():
            saved = np.load(PRIVATE / label / 'predictions.npz')
            assert np.array_equal(saved['y'], yt)
            assert np.array_equal(saved['row_tokens'], zt['row_tokens'])
            p, boots = (saved['p'], saved['bootstrap_auc'])
            auc = roc_auc_score(yt, p)
            lo, hi = np.quantile(boots, [0.025, 0.975])
            warnings_text = ['Resumed saved first-direction model/probabilities after verifier-only precision correction.']
        else:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always')
                auc, p = scope['transfer_train_predict'](xs, ys, fs, xt, yt, ftar, label=label)
            warnings_text = sorted(set((str(w.message) for w in seen)))
            lo, hi, boots = bootstrap(yt, p, zt['groups'], 2026101900 + i)
            np.savez_compressed(PRIVATE / label / 'predictions.npz', y=yt, p=p, groups=zt['groups'], row_tokens=zt['row_tokens'], bootstrap_auc=boots)
        mdl = joblib.load(PRIVATE / label / 'preprocessor_rf.joblib')
        xb = xgb.Booster()
        xb.load_model(PRIVATE / label / 'xgboost.ubj')
        common = mdl['features']
        xx = xt[:, [ftar.index(f) for f in common]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            xx = mdl['imputer'].transform(xx)
        pp = 0.5 * xb.predict(xgb.DMatrix(xx, feature_names=[f'f{j}' for j in range(xx.shape[1])]))
        pp = pp + 0.5 * mdl['rf'].predict_proba(xx)[:, 1]
        max_err = float(np.max(np.abs(p - pp)))
        assert max_err < 1e-14
        assert np.isclose(auc, roc_auc_score(yt, p), atol=1e-15, rtol=0)
        row = {'source': source, 'target': target, 'analysis': 'new_fixed_parameter_historical_code_replay', 'target_unit': 'admission' if target == 'MIMIC' else 'patient', 'rows': len(yt), 'patients': len(np.unique(zt['groups'])), 'positive_labels': int(yt.sum()), 'auc': float(auc), 'ci_low': lo, 'ci_high': hi, 'bootstrap_draws': len(boots), 'historical_auc': old_auc, 'matches_historical_round4': round(float(auc), 4) == old_auc, 'original_prediction_identity_established': False, 'saved_model_max_abs_error': max_err, 'interval_scope': 'new_fixed_model_target_patient_bootstrap'}
        rows.append(row)
        dump(f'{label}_verification.json', {**row, 'warnings': warnings_text, 'artifacts': [identity(PRIVATE / label / n) for n in ['predictions.npz', 'source_and_target.npz', 'xgboost.ubj', 'preprocessor_rf.joblib']]})
        print(json.dumps({k: row[k] for k in ['source', 'target', 'auc', 'ci_low', 'ci_high', 'matches_historical_round4']}), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / 'replay_metrics.csv', index=False, float_format='%.17g')
    unchanged = all((identity(item['path']) == item for item in inputs))
    assert unchanged
    dump('completion.json', {'completed': True, 'new_vectors_saved': 4, 'historical_vectors_recovered': 0, 'all_input_hashes_unchanged': unchanged, 'no_historical_CI_claimed': True, 'replay_metrics': identity(ROOT / 'replay_metrics.csv'), 'environment': environment(), 'utc': datetime.now(timezone.utc).isoformat()})
if __name__ == '__main__':
    main()
