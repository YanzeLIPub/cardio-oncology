"""Run all original Option B OOF folds from anonymous, frozen numeric matrices.

The three model functions are compiled unchanged from the frozen source AST.
Process CPU affinity bounds resource use without changing their parameters.
No project module is imported, and no source-directory output is written.
"""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
ORIGINAL_SHA256 = None

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha_file(path):
    return None

def sha_array(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode('ascii'))
    h.update(json.dumps(list(array.shape)).encode('ascii'))
    h.update(array.tobytes(order='C'))
    return h.hexdigest()

def set_resources(threads):
    require(1 <= threads <= 8, 'Resource limit must be 1 through 8 CPUs')
    limits = {n: str(threads) for n in ['OMP_NUM_THREADS', 'LOKY_MAX_CPU_COUNT', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}
    limits['CUDA_VISIBLE_DEVICES'] = ''
    previous = {n: os.environ.get(n) for n in limits}
    os.environ.update(limits)
    if hasattr(os, 'sched_getaffinity'):
        before = sorted(os.sched_getaffinity(0))
        requested = before[:threads]
        os.sched_setaffinity(0, set(requested))
        after = sorted(os.sched_getaffinity(0))
        require(after == requested, 'CPU affinity verification failed')
        method = 'os.sched_setaffinity'
    elif sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        original, system = (ctypes.c_size_t(), ctypes.c_size_t())
        require(bool(kernel32.GetProcessAffinityMask(process, ctypes.byref(original), ctypes.byref(system))), 'Cannot query Windows CPU affinity')
        before = [i for i in range(8 * ctypes.sizeof(ctypes.c_size_t)) if original.value & 1 << i]
        requested = before[:threads]
        mask = sum((1 << i for i in requested))
        require(bool(kernel32.SetProcessAffinityMask(process, mask)), 'Cannot set Windows CPU affinity')
        actual = ctypes.c_size_t()
        require(bool(kernel32.GetProcessAffinityMask(process, ctypes.byref(actual), ctypes.byref(system))), 'Cannot re-query Windows CPU affinity')
        after = [i for i in range(8 * ctypes.sizeof(ctypes.c_size_t)) if actual.value & 1 << i]
        require(after == requested, 'Windows CPU affinity verification failed')
        method = 'SetProcessAffinityMask'
    else:
        raise RuntimeError('Unsupported platform for verified process affinity')
    return {'method': method, 'before_allowed_cpu_ids': before, 'after_allowed_cpu_ids': after, 'environment_limits': limits, 'previous_environment_values': previous, 'source_nthread_and_n_jobs_unchanged_minus_one': True, 'resource_difference_from_historical_unrestricted_run': True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    resources = set_resources(args.threads)
    import numpy as np
    import sklearn
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    from threadpoolctl import threadpool_info
    start_time = time.time()
    manifest = _release_resolve_config(json.loads(args.manifest.read_text(encoding='utf-8-sig')))
    matrix_sha = sha_file(args.input)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Replay output directory must be new')
    output.mkdir(parents=True, mode=448, exist_ok=False)
    if os.name == 'posix':
        output.chmod(448)
    source_text = args.source.read_text(encoding='utf-8-sig')
    tree = ast.parse(source_text)
    names = ['make_xgb_params', 'make_rf', 'cv_oof']
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require([n.name for n in nodes] == names, 'Original function extraction failed')
    function_source_hashes = {n.name: hashlib.sha256(ast.get_source_segment(source_text, n).encode('utf-8')).hexdigest() for n in nodes}
    namespace = {'np': np, 'xgb': xgb, 'RandomForestClassifier': RandomForestClassifier, 'StratifiedKFold': StratifiedKFold, 'roc_auc_score': roc_auc_score}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.source), 'exec'), namespace)
    require(namespace['cv_oof'].__defaults__ == (5, 300), 'Original CV defaults changed')
    np.random.seed(42)
    runtime = {'executable': sys.executable, 'python': platform.python_version(), 'platform': platform.platform(), 'machine': platform.machine(), 'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'xgboost', 'joblib', 'threadpoolctl']}, 'xgboost_build_info': xgb.build_info(), 'threadpools': threadpool_info()}
    report = {'schema_version': 1, 'analysis': 'Original Option B complete CPU OOF replay', 'function_source_hashes': function_source_hashes, 'runtime': runtime, 'resources': resources, 'model_functions_modified': False, 'legacy_semantics': {'n_splits': 5, 'n_rounds_maximum': 300, 'early_stopping_rounds': 20, 'splitter_shuffle': True, 'splitter_random_state': 42, 'xgb_seed': 42, 'rf_random_state': 42, 'xgb_prediction_uses_all_trained_trees': True, 'class_weights_derived_from_full_centre_y_as_original': True, 'rf_positive_weight_integer_truncated_as_original': True, 'ensemble_weights': [0.5, 0.5], 'isotonic_not_used': True, 'not_a_leakage_corrected_or_new_scientific_analysis': True}, 'cohorts': {}, 'status': 'running'}

    def save_report():
        with (output / 'china_replay_metrics.json').open('w', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    save_report()
    replay_arrays = {}
    with np.load(args.input, allow_pickle=False) as archive:
        for centre in ['SCH', 'WHU']:
            suffix = centre.lower()
            X, y = (archive[f'{suffix}_X'], archive[f'{suffix}_y'])
            cached, frozen_folds = (archive[f'{suffix}_cached_prob'], archive[f'{suffix}_fold_id'])
            reference = manifest['cohorts'][centre]
            require(X.shape == (reference['n'], reference['features']) and int(y.sum()) == reference['events'], 'Matrix cohort shape/label mismatch')
            for name, arr in [('X', X), ('y', y), ('cached_prob', cached), ('fold_id', frozen_folds)]:
                pass
            replay_folds = np.zeros(len(y), dtype=np.int8)
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (_, va) in enumerate(splitter.split(X, y), 1):
                replay_folds[va] = fold
            require(np.array_equal(replay_folds, frozen_folds), 'Splitter differs from frozen original order')
            print(json.dumps({'stage': 'training', 'centre': centre, 'n': len(y), 'events': int(y.sum()), 'features': X.shape[1], 'folds': 5}), flush=True)
            cohort_start = time.time()
            oof, fold_aucs, feature_importances, models = namespace['cv_oof'](X, y)
            require(len(models) == 5 and len(fold_aucs) == 5, 'Incomplete five-fold replay')
            require(np.isfinite(oof).all(), 'Non-finite replay probabilities')
            differences = np.abs(oof - cached)
            fold_details = []
            for index, ((booster, forest), auc) in enumerate(zip(models, fold_aucs), 1):
                trained_config = _release_resolve_config(json.loads(booster.save_config()))
                trained_device = trained_config.get('learner', {}).get('generic_param', {}).get('device')
                require(trained_device == 'cpu', 'Trained XGBoost model did not confirm CPU device')
                fold_details.append({'fold': index, 'auc': float(auc), 'xgb_trained_device': trained_device, 'xgb_boosted_rounds': booster.num_boosted_rounds(), 'xgb_best_iteration': int(booster.best_iteration), 'xgb_best_score': float(booster.best_score), 'rf_trees': int(forest.n_estimators)})
            comparisons = {'cached_auc': float(roc_auc_score(y, cached)), 'replayed_auc': float(roc_auc_score(y, oof)), 'auc_delta': float(roc_auc_score(y, oof) - roc_auc_score(y, cached)), 'cached_brier': float(np.mean((cached - y) ** 2)), 'replayed_brier': float(np.mean((oof - y) ** 2)), 'brier_delta': float(np.mean((oof - y) ** 2) - np.mean((cached - y) ** 2)), 'mean_absolute_probability_difference': float(differences.mean()), 'max_absolute_probability_difference': float(differences.max()), 'root_mean_square_probability_difference': float(np.sqrt(np.mean((oof - cached) ** 2))), 'exact_equal_count': int(np.count_nonzero(oof == cached)), 'probability_allclose_atol_1e_minus12_rtol_0': bool(np.allclose(oof, cached, atol=1e-12, rtol=0)), 'probability_allclose_atol_1e_minus8_rtol_0': bool(np.allclose(oof, cached, atol=1e-08, rtol=0)), 'probability_allclose_atol_1e_minus6_rtol_0': bool(np.allclose(oof, cached, atol=1e-06, rtol=0)), 'counts_above_absolute_difference': {str(t): int(np.count_nonzero(differences > t)) for t in [0.0, 1e-12, 1e-10, 1e-08, 1e-06, 1e-05, 0.0001, 0.001]}, 'prediction_class_disagreement_at_probability_0_5': int(np.count_nonzero((oof >= 0.5) != (cached >= 0.5)))}
            parameters = namespace['make_xgb_params'](y)
            parameters = {k: v.item() if isinstance(v, np.generic) else v for k, v in parameters.items()}
            report['cohorts'][centre] = {'n': int(len(y)), 'events': int(y.sum()), 'features': int(X.shape[1]), 'fold_identity_confirmed': True, 'fold_details': fold_details, 'comparison': comparisons, 'xgb_parameters': parameters, 'rf_parameters': namespace['make_rf'](y).get_params(), 'elapsed_seconds': time.time() - cohort_start}
            replay_arrays[f'{suffix}_oof'] = oof
            replay_arrays[f'{suffix}_cached_prob'] = cached
            replay_arrays[f'{suffix}_fold_id'] = frozen_folds
            save_report()
            print(json.dumps({'stage': 'cohort_complete', 'centre': centre, **comparisons}), flush=True)
            del models, feature_importances
    output_npz = output / 'china_replay_probabilities.private.npz'
    with output_npz.open('xb') as handle:
        np.savez_compressed(handle, **replay_arrays)
    report.update({'status': 'complete', 'completed_outer_folds': 10, 'source_and_input_unchanged_after': True, 'elapsed_seconds': time.time() - start_time, 'restricted_probability_output': {'file': output_npz.name, 'contains_patient_identifiers': False}})
    save_report()
    print(json.dumps({'status': 'complete', 'completed_outer_folds': 10, 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
if __name__ == '__main__':
    main()
