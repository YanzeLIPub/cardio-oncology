"""Frozen four-arm selection sensitivity with patient-disjoint fitting."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
HELPER = _release_source_path(PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1/run_validation.py')
spec = importlib.util.spec_from_file_location('reviewed_training_functions', HELPER)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def data():
    path = ROOT / 'preparation/private/full_matrix.npz'
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key].copy() for key in z.files}
    assert values['X'].shape == (11622, 41)
    assert len(values['y']) == len(values['groups']) == 11622
    assert not np.isinf(values['X']).any()
    assert 'AST' not in values['feature_names'] and 'HDL Cholesterol' not in values['feature_names']
    masks = {'ever_all': np.ones(len(values['y']), dtype=bool), 'ever_survivor': values['survivor'].astype(bool), 'prior_all': values['prior_completed_valid'].astype(bool), 'prior_survivor': values['prior_completed_valid'].astype(bool) & values['survivor'].astype(bool)}
    assert masks['ever_survivor'].sum() == 9946 and masks['prior_survivor'].sum() == 2194
    for name, mask in masks.items():
        if name in values:
            assert np.array_equal(mask, values[name])
    return (path, values, masks)

def roles(groups, config):
    patients = np.unique(groups)
    shuffled = np.random.default_rng(config['outer_seed']).permutation(patients)
    folds = np.array_split(shuffled, config['folds'])
    result = np.full((config['folds'], len(groups)), 2, dtype=np.int8)
    for fold, test_groups in enumerate(folds, 1):
        train_groups = patients[~np.isin(patients, test_groups)]
        order = np.random.default_rng(config['inner_seed_base'] + fold).permutation(train_groups)
        stop_groups = order[:int(np.ceil(config['inner_stop_fraction'] * len(train_groups)))]
        result[fold - 1, np.isin(groups, stop_groups)] = 3
        result[fold - 1, np.isin(groups, test_groups)] = 0
    assert np.all((result == 0).sum(axis=0) == 1)
    return result

def train():
    config = load(ROOT / 'protocol.json')
    gate = ROOT / 'design/preparation_verification.json'
    audit = load(gate)
    assert audit['passed'], 'Independent source reconstruction must pass before fitting'
    source, arrays, masks = data()
    assert audit['matrix_identity'] == identity(source), 'Verified matrix identity changed'
    X, y, groups = (arrays['X'], arrays['y'], arrays['groups'])
    feature_names = arrays['feature_names'].tolist()
    role_matrix = roles(groups, config)
    counts = []
    for fold, role in enumerate(role_matrix, 1):
        for arm in config['arms']:
            mask = masks[arm]
            indices = {'test_own': np.flatnonzero(mask & (role == 0)), 'train': np.flatnonzero(mask & (role != 0)), 'inner': np.flatnonzero(mask & (role == 2)), 'stop': np.flatnonzero(mask & (role == 3))}
            assert all((len(ix) and len(np.unique(y[ix])) == 2 for ix in indices.values()))
            sets = [set(groups[indices[name]]) for name in ['test_own', 'inner', 'stop']]
            assert all((not sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3)))
            counts.append({'fold': fold, 'arm': arm, 'roles': {name: {'admissions': len(ix), 'patients': len(np.unique(groups[ix])), 'positive_labels': int(y[ix].sum())} for name, ix in indices.items()}})
    target = ROOT / 'private/run_v1'
    assert not target.exists(), 'Preserve previous runs'
    target.mkdir(parents=True)
    sources = [source, ROOT / 'protocol.json', Path(__file__), HELPER, gate, ROOT / 'design/design_approval.json']
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'sources': [identity(p) for p in sources], 'protocol': config, 'environment': {'executable': sys.executable, 'python': platform.python_version(), 'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'joblib']}}}
    save(target / 'input_lock.json', manifest)
    np.savez_compressed(target / 'roles.npz', roles=role_matrix, groups=groups, y=y)
    save(ROOT / 'results/role_counts.json', {'passed': True, 'counts': counts, 'source': identity(target / 'roles.npz')})
    predictions = {arm: np.full(len(y), np.nan) for arm in config['arms']}
    fold_id = np.zeros(len(y), dtype=np.int8)
    reports = []
    for fold, role in enumerate(role_matrix, 1):
        global_test = np.flatnonzero(role == 0)
        fold_id[global_test] = fold
        for arm in config['arms']:
            mask = masks[arm]
            train_ix = np.flatnonzero(mask & (role != 0))
            inner = np.flatnonzero(mask & (role == 2))
            stop = np.flatnonzero(mask & (role == 3))
            assert not set(groups[global_test]) & set(groups[train_ix])
            folder = target / arm / f'fold_{fold}'
            folder.mkdir(parents=True)
            rounds, details = engine.choose_rounds(X, y, inner, stop, 'MIMIC', fold, config, config['threads'], folder)
            model = engine.fit_ensemble(X[train_ix], y[train_ix], 'MIMIC', fold, config, config['threads'], rounds)
            predictions[arm][global_test] = engine.ensemble_predict(model, X[global_test], config['threads'])[0]
            engine.save_model(folder, 'selection', model, feature_names)
            report = {'arm': arm, 'fold': fold, 'selected_rounds': rounds, 'fitted_features': [feature_names[i] for i in model[0]['columns']], 'imputation_statistics': model[0]['imputer'].statistics_.tolist(), 'selection': details, 'parameters': model[3], 'all_training_rows_in_arm': bool(mask[train_ix].all()), 'patient_overlap_train_global_test': 0, 'global_test_admissions': len(global_test), 'outer_train_admissions': len(train_ix)}
            save(folder / 'fit_report.json', report)
            reports.append(report)
            print(f'Completed selection arm {arm}, fold {fold}/5; rounds={rounds}', flush=True)
    assert all((np.isfinite(v).all() and np.all((v >= 0) & (v <= 1)) for v in predictions.values()))
    assert all((len(np.unique(fold_id[groups == group])) == 1 for group in np.unique(groups)))
    np.savez_compressed(target / 'predictions.npz', **{f'p_{k}': v for k, v in predictions.items()}, **{f'mask_{k}': v for k, v in masks.items()}, y=y, groups=groups, fold=fold_id)
    assert all((identity(p['path']) == p for p in manifest['sources']))
    save(ROOT / 'results/training_complete.json', {'completed': True, 'input_lock': identity(target / 'input_lock.json'), 'predictions': identity(target / 'predictions.npz'), 'fold_reports': reports, 'ended_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'New retrospective selection sensitivity; not original-AUC reproduction or clinical temporal validation'})

class FastMetrics:
    """Exact weighted tie-aware rank metrics with fixed scores sorted once."""

    def __init__(self, y, p):
        self.y = np.asarray(y, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.order = np.argsort(-self.p, kind='mergesort')
        ordered = self.p[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]

    def __call__(self, weight):
        weight = np.asarray(weight, dtype=float)
        positive = weight * self.y
        negative = weight * (1 - self.y)
        pos, neg = (positive.sum(), negative.sum())
        if not pos or not neg:
            return np.full(3, np.nan)
        tp = np.cumsum(positive[self.order])[self.ends]
        fp = np.cumsum(negative[self.order])[self.ends]
        tpr, fpr = (np.r_[0.0, tp / pos], np.r_[0.0, fp / neg])
        auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
        denominator = tp + fp
        precision = np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
        ap = np.sum(np.diff(np.r_[0.0, tp / pos]) * precision)
        brier = np.sum(weight * (self.p - self.y) ** 2) / weight.sum()
        return np.array([auc, ap, brier])

def self_test():
    rng = np.random.default_rng(3202410)
    for count in [9, 37, 100]:
        y = np.r_[0, 1, rng.integers(0, 2, count - 2)]
        p = rng.integers(0, 11, count) / 10
        metric = FastMetrics(y, p)
        for _ in range(30):
            w = rng.integers(0, 8, count)
            if not all((w[y == v].sum() for v in [0, 1])):
                continue
            expected = np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)])
            np.testing.assert_allclose(metric(w), expected, rtol=0, atol=1e-12)
    print('Weighted tie-aware AUC/AP/Brier checks passed against sklearn', flush=True)

def summarize():
    config = load(ROOT / 'protocol.json')
    source = ROOT / 'private/run_v1/predictions.npz'
    with np.load(source, allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    folder = ROOT / 'results/metrics'
    assert not folder.exists()
    folder.mkdir()
    rows, contrasts, samples = ([], [], [])
    metric_names = config['metrics']
    for target_index, target in enumerate(config['arms']):
        mask = arrays['mask_' + target]
        y, groups = (arrays['y'][mask], arrays['groups'][mask])
        names = sorted({target} | {c[key] for c in config['paired_contrasts'] if c['target'] == target for key in ['first', 'second']})
        scores = [arrays['p_' + name][mask] for name in names]
        unique, inverse = np.unique(groups, return_inverse=True)
        evaluators = [FastMetrics(y, p) for p in scores]
        points = np.array([m(np.ones(len(y))) for m in evaluators])
        draws = np.full((config['bootstrap_replicates'], len(names), 3), np.nan)
        rng = np.random.default_rng(config['bootstrap_seed_base'] + target_index)
        for i in range(len(draws)):
            counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
            w = counts[inverse]
            for j, metric in enumerate(evaluators):
                draws[i, j] = metric(w)
        valid = np.isfinite(draws).all(axis=(1, 2))
        assert valid.sum() >= 1900, 'Too few valid bootstrap draws for percentile intervals'
        samples.append({'target': target, 'draws': len(draws), 'valid': int(valid.sum()), 'invalid': int((~valid).sum()), 'seed': config['bootstrap_seed_base'] + target_index})
        for j, name in enumerate(names):
            for k, metric in enumerate(metric_names):
                lo, hi = np.percentile(draws[valid, j, k], [2.5, 97.5])
                rows.append({'target': target, 'model': name, 'own_cohort': name == target, 'metric': metric, 'estimate': float(points[j, k]), 'ci_low': float(lo), 'ci_high': float(hi), 'admissions': len(y), 'patients': len(unique), 'positive_labels': int(y.sum()), 'prevalence': float(np.mean(y)), 'mean_probability': float(np.mean(scores[j]))})
        for contrast in config['paired_contrasts']:
            if contrast['target'] != target:
                continue
            a, b = (names.index(contrast['first']), names.index(contrast['second']))
            for k, metric in enumerate(metric_names):
                lo, hi = np.percentile(draws[valid, a, k] - draws[valid, b, k], [2.5, 97.5])
                contrasts.append(contrast | {'metric': metric, 'estimate': float(points[a, k] - points[b, k]), 'ci_low': float(lo), 'ci_high': float(hi), 'admissions': len(y), 'patients': len(unique)})
        np.savez_compressed(folder / f'bootstrap_{target}.npz', draws=draws, points=points, model_names=np.array(names), metric_names=np.array(metric_names), valid=valid)
        print(f'Summarized {target}: {len(y)} admissions; {int(valid.sum())} valid patient-cluster draws', flush=True)
    pd.DataFrame(rows).to_csv(folder / 'all_performance.csv', index=False)
    pd.DataFrame([r for r in rows if r['own_cohort']]).to_csv(folder / 'own_cohort_performance.csv', index=False)
    pd.DataFrame(contrasts).to_csv(folder / 'paired_contrasts.csv', index=False)
    save(folder / 'summary.json', {'source': identity(source), 'protocol': identity(ROOT / 'protocol.json'), 'own_cohort_metrics': [r for r in rows if r['own_cohort']], 'all_common_target_metrics': rows, 'paired_contrasts': contrasts, 'bootstrap': samples, 'limitations': config['limitations']})
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'summarize', 'self-test'])
    mode = parser.parse_args().mode
    {'train': train, 'summarize': summarize, 'self-test': self_test}[mode]()
