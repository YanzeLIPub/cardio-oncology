"""Instrument unchanged historical CV functions; keep models/probabilities private.

Only observational hooks are added: a train-result list, evals_result collection,
and references to forest instances. No fitting parameter or split is changed.
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

def sha_file(p):
    return None

def sha_array(a):
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

def encode(v):
    if hasattr(v, 'item'):
        return v.item()
    raise TypeError(type(v).__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['china', 'mimic'], required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--support', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    code = a.support.read_text(encoding='utf-8-sig')
    nodes = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name in ['require', 'set_resources']]
    env = dict(os=os, sys=sys)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(a.support), 'exec'), env)
    resource = env['set_resources'](a.threads)
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    expected = {}
    matrix_expected = {}
    a.output.mkdir(parents=True, exist_ok=False, mode=448)
    report = {'kind': a.kind, 'platform': platform.platform(), 'python': sys.version, 'packages': {k: metadata.version(k) for k in ['numpy', 'scipy', 'scikit-learn', 'xgboost']}, 'build_info': xgb.build_info(), 'resource': resource, 'cohorts': {}, 'purpose': 'Numerical discrepancy diagnosis; no scientific baseline change.'}
    records = []
    forests = []

    def matrix_hash(d):
        csr = d.get_data()
        return {'shape': list(csr.shape), 'data': sha_array(csr.data), 'indices': sha_array(csr.indices), 'indptr': sha_array(csr.indptr), 'label': sha_array(d.get_label())}

    class XGBProxy:

        def __getattr__(self, n):
            return getattr(xgb, n)

        def train(self, *args, **kwargs):
            evaluations = {}
            b = xgb.train(*args, **kwargs, evals_result=evaluations)
            dv = kwargs['evals'][-1][0]
            records.append((b, dv, {'params': args[0], 'train_matrix': matrix_hash(args[1]), 'valid_matrix': matrix_hash(dv), 'evals_result': evaluations}))
            return b

    def forest_factory(*args, **kwargs):
        rf = RandomForestClassifier(*args, **kwargs)
        forests.append(rf)
        return rf
    text = a.source.read_text(encoding='utf-8-sig')
    names = ['make_xgb_params', 'make_rf', 'cv_oof'] if a.kind == 'china' else ['run_cv_ensemble']
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(np=np, xgb=XGBProxy(), RandomForestClassifier=forest_factory, StratifiedKFold=StratifiedKFold, GroupKFold=GroupKFold, SimpleImputer=SimpleImputer, roc_auc_score=roc_auc_score)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(a.source), 'exec'), ns)
    report['function_source_hashes'] = {n.name: hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest() for n in nodes}
    np.random.seed(42)
    data = np.load(a.input, allow_pickle=False)
    previous = np.load(a.previous, allow_pickle=False)
    start = time.time()
    for c in ['sch', 'whu'] if a.kind == 'china' else ['mimic']:
        records.clear()
        forests.clear()
        X = data[c + '_X'] if a.kind == 'china' else data['X']
        y = data[c + '_y'] if a.kind == 'china' else data['y']
        ref = data[c + '_cached_prob'] if a.kind == 'china' else data['reference']
        f = data[c + '_fold_id'] if a.kind == 'china' else data['fold_id']
        out = ns['cv_oof' if a.kind == 'china' else 'run_cv_ensemble'](X, y)[0]
        prev = previous[c + '_oof'] if a.kind == 'china' else previous['probability']
        previous_matches = bool(np.allclose(out, prev, atol=1e-12, rtol=0))
        assert len(records) == len(forests) == 5
        px = np.zeros(len(y))
        pr = np.zeros(len(y))
        pb = np.zeros(len(y))
        details = []
        for i, ((tr, va), (booster, dv, info), rf) in enumerate(zip(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), records, forests), 1):
            assert (f[va] == i).all()
            Xtr, Xva = (X[tr], X[va])
            if a.kind == 'mimic':
                imp = SimpleImputer(strategy='median')
                Xtr = imp.fit_transform(Xtr)
                Xva = imp.transform(Xva)
            px[va] = booster.predict(dv)
            pb[va] = booster.predict(dv, iteration_range=(0, booster.best_iteration + 1))
            pr[va] = rf.predict_proba(Xva)[:, 1]
            info.update({'fold': i, 'boosted_rounds': booster.num_boosted_rounds(), 'best_iteration': booster.best_iteration, 'best_score': float(booster.best_score), 'Xtr_hash': sha_array(Xtr), 'Xva_hash': sha_array(Xva), 'train_indices_hash': sha_array(tr), 'valid_indices_hash': sha_array(va), 'rf_params': rf.get_params()})
            states = [e.tree_.__getstate__() for e in rf.estimators_]
            info['rf_tree_hashes'] = [hashlib.sha256(s['nodes'].tobytes() + s['values'].tobytes()).hexdigest() for s in states]
            config = _release_resolve_config(json.loads(booster.save_config()))
            info['config'] = config
            model = a.output / f'{c}_fold{i}.ubj'
            booster.save_model(model)
            info['model_sha256'] = sha_file(model)
            dumps = booster.get_dump(dump_format='json', with_stats=True)
            info['tree_hashes'] = [hashlib.sha256(t.encode()).hexdigest() for t in dumps]
            details.append(info)
        assert np.allclose(out, (px + pr) / 2, atol=1e-12, rtol=0)
        np.savez_compressed(a.output / f'{c}_components.private.npz', xgb=px, rf=pr, xgb_best=pb, ensemble=out, reference=ref, fold_id=f, y=y)
        best_ensemble = 0.5 * pb + 0.5 * pr
        report['cohorts'][c] = {'n': len(y), 'events': int(y.sum()), 'folds': details, 'instrumentation_matches_previous': previous_matches, 'previous_max_abs': float(np.max(np.abs(out - prev))), 'auc': float(roc_auc_score(y, out)), 'reference_auc': float(roc_auc_score(y, ref)), 'best_iteration_auc': float(roc_auc_score(y, best_ensemble)), 'best_iteration_vs_cache_mae': float(np.mean(np.abs(best_ensemble - ref))), 'all_trees_vs_cache_mae': float(np.mean(np.abs(out - ref))), 'best_iteration_vs_cache_max': float(np.max(np.abs(best_ensemble - ref)))}
        print(json.dumps({'cohort': c, 'instrumentation_matches_previous': previous_matches, 'auc': report['cohorts'][c]['auc']}), flush=True)
    report['seconds'] = time.time() - start
    report['status'] = 'complete'
    (a.output / 'component_report.json').write_text(json.dumps(report, indent=2, default=encode), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'seconds': report['seconds']}), flush=True)
if __name__ == '__main__':
    main()
