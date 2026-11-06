"""Prepare the unchanged v13 classification matrix; never execute the source module."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(_release_path('pipeline', ''))
SOURCE = _release_source_path(PROJECT / 'Manuscript/v13/compute_v13_metrics.py')

def digest(path):
    return None

def main():
    target = ROOT / 'private/mimic_matrices.npz'
    if target.exists():
        raise FileExistsError(target)
    raw = SOURCE.read_bytes()
    tree = ast.parse(raw.decode('utf-8-sig'))
    node = next((n for n in tree.body if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'FEATS_43_EN' for t in n.targets))))
    features = ast.literal_eval(node.value)
    ft = PROJECT / 'MIMIC-3/data/features_model.csv'
    lb = PROJECT / 'MIMIC-3/data/mace_labels.csv'
    cached = PROJECT / 'Manuscript/v13/paper_predictions_9946.csv'
    feat, lbl, paper = (pd.read_csv(ft), pd.read_csv(lb), pd.read_csv(cached))
    assert feat['HADM_ID'].is_unique and lbl['HADM_ID'].is_unique and paper['HADM_ID'].is_unique
    if 'HOSPITAL_EXPIRE_FLAG' not in feat.columns:
        df = feat.merge(lbl[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'MACE']], on='HADM_ID', how='inner')
    else:
        df = feat.merge(lbl[['HADM_ID', 'MACE']], on='HADM_ID', how='inner')
    df = df[df['HOSPITAL_EXPIRE_FLAG'] == 0].copy()
    avail = [f for f in features if f in df.columns]
    for column in avail:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    X = df[avail].values.astype(float)
    y = df['MACE'].values.astype(int)
    assert len(y) == 9946 and int(y.sum()) == 4462 and (df['SUBJECT_ID'].nunique() == 6783)
    assert np.array_equal(df['HADM_ID'].to_numpy(), paper['HADM_ID'].to_numpy())
    assert np.array_equal(df['SUBJECT_ID'].to_numpy(), paper['SUBJECT_ID'].to_numpy())
    assert np.array_equal(y, paper['MACE'].to_numpy())
    folds = np.full(len(y), -1, dtype=np.int16)
    fold_sizes = []
    for fold, (train, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1):
        folds[valid] = fold
        fold_sizes.append({'fold': fold, 'train': len(train), 'valid': len(valid)})
    np.savez_compressed(target, X=X, y=y, reference=paper['oof_prob_paper'].to_numpy(float), fold_id=folds, feature_names=np.asarray(avail, dtype=str))
    copy = _release_source_path(ROOT / 'source/mimic_compute_v13_original.py')
    if copy.exists():
        raise FileExistsError(copy)
    copy.write_bytes(raw)
    record = {'date': '2026-10-27', 'task': 'historical classification execution reproducibility', 'cohort': 'MIMIC', 'unit': 'admission', 'n': len(y), 'events': int(y.sum()), 'patients': int(df['SUBJECT_ID'].nunique()), 'feature_names': avail, 'all_missing_features': [f for f, all_missing in zip(avail, np.isnan(X).all(axis=0)) if all_missing], 'source_inputs': [{'path': str(p), 'bytes': p.stat().st_size} for p in [SOURCE, ft, lb, cached]], 'matrix_dtype': str(X.dtype), 'fold_sizes': fold_sizes, 'cached_row_order_and_labels_exact': True, 'reference_probability_column': 'oof_prob_paper', 'npz_contains_identifiers': False, 'preparation_environment': {'numpy': np.__version__, 'pandas': pd.__version__}, 'limitation': 'Retains historical admission-row CV, all-record features and early stopping on outer validation. This is not new independent-patient or prospective clinical validation.'}
    (ROOT / 'verification/mimic_preparation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'cohort': 'MIMIC', 'n': len(y), 'events': int(y.sum()), 'features': len(avail)}))
if __name__ == '__main__':
    main()
