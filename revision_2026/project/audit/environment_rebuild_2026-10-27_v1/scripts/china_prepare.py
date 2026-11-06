"""Freeze the original Option B input order without running training or exporting IDs.

Run with the retained Chinese interpreter. Source/input files are read-only.
Only the explicitly supplied new output root receives artifacts.
"""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
import time
ORIGINAL_SHA256 = None
INPUTS = {'sch_features': Path(_release_path('china', 'SCH_processed_data.csv')), 'whu_cancer_features': Path(_release_path('china', 'WHU_Cancer_processed_data.csv')), 'labels': Path(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv')), 'source_cache': Path(_release_path('china', 'outputs/real_model_predictions_v2.csv')), 'paper_cache': Path(_release_path('pipeline', 'data/real_model_predictions_v2.csv'))}

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

def snapshot(path):
    st = path.stat()
    return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}

def selected_loading_nodes(tree):
    start = end = None
    for index, node in enumerate(tree.body):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if 'sch_head' in names:
            start = index
        if 'X_whu' in names and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == 'nan_to_num':
                end = index
    require(start is not None and end is not None and (start < end), 'Cannot identify original loading boundaries')
    nodes = tree.body[start:end + 1]
    require(not any((isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes)), 'Unexpected imports in source loading segment')
    return nodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    start_time = time.time()
    output_root = args.output_root.resolve()
    require(output_root.name == 'environment_rebuild_2026-10-27_v1', 'Unexpected output root')
    private = output_root / 'private'
    verification = output_root / 'verification'
    require(private.is_dir() and verification.is_dir(), 'Output directories must already exist')
    matrix_path = private / 'china_matrices.npz'
    manifest_path = verification / 'china_preparation.json'
    require(not matrix_path.exists() and (not manifest_path.exists()), 'Refusing artifact overwrite')
    require(np.__version__ == '2.3.5' and pd.__version__ == '2.3.3', 'Prepare with the retained Chinese NumPy/pandas versions')
    import sklearn
    require(sklearn.__version__ == '1.8.0', 'Unexpected splitter package version')
    before = {key: snapshot(path) for key, path in INPUTS.items()}
    source_text = args.source.read_text(encoding='utf-8-sig')
    tree = ast.parse(source_text)
    nodes = selected_loading_nodes(tree)
    namespace = {'pd': pd, 'np': np}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.source), 'exec'), namespace)
    feat_cols = namespace['feat_cols']
    require(len(feat_cols) == 104 and len(set(feat_cols)) == 104, 'Unexpected original feature list')
    columns = ['病案号', 'Center', 'MACE', 'predicted_prob_within_cv']
    cache = pd.read_csv(INPUTS['source_cache'], usecols=columns)
    paper = pd.read_csv(INPUTS['paper_cache'], usecols=columns)
    require(len(cache) == 41071 and len(paper) == 41071, 'Unexpected cache total')
    cross_centre_id_collisions = len(pd.Index(namespace['pid_sch']).intersection(namespace['pid_whu']))
    require(cross_centre_id_collisions == 0, 'Original patient-only prediction merge is ambiguous across centres')
    require(cache['病案号'].is_unique and paper['病案号'].is_unique, 'Original saved prediction patient key is not unique')
    payload = {'feature_names': np.asarray(feat_cols, dtype=str)}
    cohort_reports = {}
    for centre, expected_n, expected_events in [('SCH', 38509, 1323), ('WHU', 2562, 281)]:
        suffix = centre.lower()
        X, y, ids = (namespace[f'X_{suffix}'], namespace[f'y_{suffix}'], namespace[f'pid_{suffix}'])
        require(X.shape == (expected_n, 104) and len(y) == expected_n, 'Unexpected original cohort shape')
        require(int(y.sum()) == expected_events and set(np.unique(y)) == {0, 1}, 'Unexpected original labels')
        require(pd.Index(ids).is_unique and (not pd.isna(ids).any()), 'Input patient order is not unique/complete')
        require(X.dtype == np.float32 and np.isfinite(X).all(), 'Unexpected matrix dtype/values')
        aligned_versions = []
        for frame in [cache, paper]:
            centre_cache = frame.loc[frame['Center'].eq(centre)].set_index('病案号')
            require(len(centre_cache) == expected_n and centre_cache.index.is_unique, 'Cache centre key is not one-to-one')
            require(pd.Index(ids).isin(centre_cache.index).all(), 'Cache key coverage mismatch')
            aligned = centre_cache.reindex(ids)
            require(np.array_equal(aligned['MACE'].to_numpy(dtype=int), y), 'Cache labels differ')
            probabilities = aligned['predicted_prob_within_cv'].to_numpy(dtype=float)
            require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), 'Cache probabilities are invalid')
            aligned_versions.append(probabilities)
        cached = aligned_versions[0]
        require(np.array_equal(cached, aligned_versions[1]), 'Paper and source probability copies differ')
        folds = np.zeros(len(y), dtype=np.int8)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_reports = []
        for fold, (tr, va) in enumerate(splitter.split(X, y), 1):
            require(not np.any(folds[va]), 'Duplicate validation assignment')
            folds[va] = fold
            fold_reports.append({'fold': fold, 'train_n': int(len(tr)), 'validation_n': int(len(va)), 'train_events': int(y[tr].sum()), 'validation_events': int(y[va].sum())})
        require((folds > 0).all(), 'Incomplete validation assignment')
        payload.update({f'{suffix}_X': X, f'{suffix}_y': y.astype(np.int64), f'{suffix}_cached_prob': cached, f'{suffix}_fold_id': folds})
        id_order_bytes = json.dumps([str(x) for x in ids], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        cohort_reports[centre] = {'n': expected_n, 'events': expected_events, 'features': 104, 'cached_auc': float(roc_auc_score(y, cached)), 'source_paper_probabilities_identical': True, 'patient_ids_saved': False, 'centre_patient_alignment_one_to_one': True, 'folds': fold_reports}
    after = {key: snapshot(path) for key, path in INPUTS.items()}
    require(before == after, 'An input changed during matrix preparation')
    with matrix_path.open('xb') as handle:
        np.savez_compressed(handle, **payload)
    report = {'schema_version': 1, 'analysis': 'Original Option B CPU OOF replay preparation', 'source': str(args.source.resolve()), 'loading_segment_lines': [nodes[0].lineno, nodes[-1].end_lineno], 'input_files': before, 'inputs_unchanged_after': True, 'original_patient_only_prediction_merge_cross_centre_id_collisions': cross_centre_id_collisions, 'cohorts': cohort_reports, 'feature_order': feat_cols, 'matrix_file': {'path': str(matrix_path), 'bytes': matrix_path.stat().st_size, 'contains_patient_identifiers': False, 'restricted_anonymous_clinical_matrix': True}, 'runtime': {'executable': sys.executable, 'python': platform.python_version(), 'platform': platform.platform(), 'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'xgboost']}}, 'semantics': {'unchanged': ['source header order and exclusion patterns', 'patient label order', 'full-record patient mean', 'zero imputation', 'float32 features', 'legacy ever-MACE labels', 'five-fold shuffled StratifiedKFold seed 42'], 'probability_column': 'predicted_prob_within_cv', 'not_isotonic_calibration': True, 'not_new_scientific_validation': True}, 'elapsed_seconds': time.time() - start_time}
    with manifest_path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'complete', 'cohorts': {k: {n: v[n] for n in ['n', 'events', 'features', 'cached_auc']} for k, v in cohort_reports.items()}, 'patient_ids_saved': False}, ensure_ascii=True), flush=True)
if __name__ == '__main__':
    main()
