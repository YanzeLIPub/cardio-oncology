"""Run only the original v13 CV function on frozen anonymous numeric inputs."""
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

def restrict_cpu(n):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['LOKY_MAX_CPU_COUNT'] = str(n)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if hasattr(os, 'sched_getaffinity'):
        chosen = sorted(os.sched_getaffinity(0))[:n]
        os.sched_setaffinity(0, chosen)
        return chosen
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        process_mask, system_mask = (ctypes.c_size_t(), ctypes.c_size_t())
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        chosen = [i for i in range(64) if process_mask.value & 1 << i][:n]
        if not kernel.SetProcessAffinityMask(handle, sum((1 << i for i in chosen))):
            raise ctypes.WinError(ctypes.get_last_error())
        return chosen
    raise RuntimeError('CPU affinity enforcement unavailable')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    affinity = restrict_cpu(args.threads)
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from threadpoolctl import threadpool_info
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if os.name == 'posix':
        os.chmod(args.output_dir, 448)
    data = np.load(args.input, allow_pickle=False)
    X, y, reference = (data['X'], data['y'], data['reference'])
    folds = np.full(len(y), -1, dtype=np.int16)
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1):
        folds[valid] = fold
    assert np.array_equal(folds, data['fold_id'])
    source = args.source.read_bytes()
    node = next((n for n in ast.parse(source.decode('utf-8-sig')).body if isinstance(n, ast.FunctionDef) and n.name == 'run_cv_ensemble'))
    namespace = dict(np=np, xgb=xgb, StratifiedKFold=StratifiedKFold, GroupKFold=GroupKFold, SimpleImputer=SimpleImputer, RandomForestClassifier=RandomForestClassifier, roc_auc_score=roc_auc_score)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(args.source), 'exec'), namespace)
    start = time.monotonic()
    probability, fold_aucs = namespace['run_cv_ensemble'](X, y)
    elapsed = time.monotonic() - start
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    difference = np.abs(probability - reference)
    result = {'date': '2026-10-27', 'cohort': 'MIMIC', 'unit': 'admission', 'n': len(y), 'events': int(y.sum()), 'model_function_unmodified': True, 'cv': 'StratifiedKFold(5, shuffle=True, random_state=42); admission rows', 'cpu_affinity': affinity, 'gpu_hidden': True, 'model_parameters_unchanged': True, 'fold_ids_match_preparation': True, 'fold_aucs': fold_aucs, 'replay_auc': float(roc_auc_score(y, probability)), 'cached_auc': float(roc_auc_score(y, reference)), 'replay_brier': float(brier_score_loss(y, probability)), 'cached_brier': float(brier_score_loss(y, reference)), 'probability_comparison': {'exact_equal_count': int((probability == reference).sum()), 'max_abs_difference': float(difference.max()), 'mean_abs_difference': float(difference.mean()), 'rmse': float(np.sqrt(np.mean((probability - reference) ** 2))), 'count_difference_gt_1e_8': int((difference > 1e-08).sum()), 'count_difference_gt_1e_5': int((difference > 1e-05).sum()), 'allclose_atol_1e_12_rtol_0': bool(np.allclose(probability, reference, atol=1e-12, rtol=0)), 'allclose_atol_1e_8_rtol_0': bool(np.allclose(probability, reference, atol=1e-08, rtol=0))}, 'python': sys.version, 'platform': platform.platform(), 'packages': {name: metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost']}, 'threadpool_info': threadpool_info(), 'seconds': elapsed, 'claim_scope': 'Execution/probability reproduction only; historical study-design limitations unchanged.'}
    np.savez_compressed(args.output_dir / 'private_predictions.npz', probability=probability, reference=reference, y=y, fold_id=folds)
    (args.output_dir / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['cohort', 'replay_auc', 'cached_auc', 'probability_comparison', 'seconds']}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
