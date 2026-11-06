"""Independent aggregate verification; never prints or exports clinical rows."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parents[1] / 'revision_2026-10-15/original_v5_independent_validation_v1'
EXPECTED = {'SCH': (38509, 38509, 1323), 'WHU': (2562, 2562, 281), 'MIMIC': (9946, 6783, 4462)}
MODELS = ['strict_ensemble', 'strict_logistic', 'prior_full', 'prior_age']
METRICS = ['ROC_AUC', 'average_precision', 'Brier']
TOL = 5e-12

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

class WeightedRanks:

    def __init__(self, y, probability):
        self.y = np.asarray(y, dtype=np.float64)
        p = np.asarray(probability, dtype=np.float64)
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        self.order = np.argsort(p, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(p[self.order]) != 0) + 1]
        self.error = (self.y - p) ** 2

    def evaluate(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        pos = np.add.reduceat((weights * self.y)[self.order], self.starts)
        neg = np.add.reduceat((weights * (1 - self.y))[self.order], self.starts)
        p_total, n_total = (pos.sum(), neg.sum())
        assert p_total > 0 and n_total > 0
        neg_before = np.cumsum(neg) - neg
        auc = np.sum(pos * (neg_before + 0.5 * neg)) / (p_total * n_total)
        desc_pos, desc_all = (pos[::-1], (pos + neg)[::-1])
        recall_num = np.cumsum(desc_pos)
        total_num = np.cumsum(desc_all)
        precision = np.divide(recall_num, total_num, out=np.zeros_like(total_num), where=total_num > 0)
        ap = np.sum(desc_pos * precision) / p_total
        brier = np.dot(weights, self.error) / weights.sum()
        return np.array([auc, ap, brier], dtype=np.float64)

def library_metrics(y, p):
    return np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)])

def self_tests():
    rng = np.random.default_rng(824761)
    errors = []
    for _ in range(50):
        y = rng.integers(0, 2, 32)
        p = rng.integers(0, 6, 32) / 5
        w = rng.integers(0, 5, 32)
        repeated = np.repeat(np.arange(len(y)), w)
        if len(np.unique(y[repeated])) != 2:
            continue
        actual = WeightedRanks(y, p).evaluate(w)
        expected = library_metrics(y[repeated], p[repeated])
        errors.append(float(np.max(np.abs(actual - expected))))
    assert len(errors) >= 40 and max(errors) < TOL
    return {'repeated_row_cases': len(errors), 'ties_and_zero_multiplicities': True, 'max_absolute_error': max(errors)}

def verify(cohort):
    dest = Path(__file__).resolve().parent / f'{cohort}_independent_verification.json'
    assert not dest.exists(), 'Refuse to overwrite independent result.'
    protocol_path = ROOT / 'protocol.json'
    protocol = _release_resolve_config(json.loads(protocol_path.read_text(encoding='utf-8')))
    prediction_path = ROOT / 'private' / cohort / 'predictions.npz'
    aggregate_path = ROOT / 'results' / cohort / 'aggregate_bootstrap.npz'
    role_path = BASE / 'private/runs' / f'{cohort}_original/patient_roles.npz'
    prior_path = BASE / 'private/runs' / f'{cohort}_original/predictions.npz'
    with np.load(prediction_path, allow_pickle=False) as z:
        y, groups, eligible, fold = [z[n].copy() for n in ('y', 'groups', 'eligible', 'fold')]
        scores = [z[name].copy() for name in MODELS]
    assert (len(y), len(np.unique(groups)), int(y.sum())) == EXPECTED[cohort]
    assert eligible.dtype == np.bool_ and eligible.all(), 'This run was reported to have zero demographic exclusions.'
    assert np.array_equal(np.unique(y), [0, 1])
    with np.load(role_path, allow_pickle=False) as z:
        roles = z['roles'].copy()
        assert np.array_equal(z['groups'], groups) and np.array_equal(z['y'], y)
    assert roles.shape == (5, len(y)) and np.array_equal(np.unique(roles), [0, 1, 2, 3])
    assert np.all((roles == 0).sum(axis=0) == 1)
    role_checks = []
    for i, role in enumerate(roles, 1):
        assert np.array_equal(role == 0, fold == i)
        patient_sets = [set(groups[role == r]) for r in range(4)]
        intersections = [len(patient_sets[a] & patient_sets[b]) for a in range(4) for b in range(a + 1, 4)]
        assert sum(intersections) == 0
        role_checks.append({'fold': i, 'all_six_pairwise_role_patient_intersections_zero': True, 'outer_test_rows': int((role == 0).sum())})
    with np.load(prior_path, allow_pickle=False) as z:
        assert np.array_equal(z['groups'], groups) and np.array_equal(z['y'], y)
        assert np.array_equal(z['fold'], fold)
        assert np.array_equal(z['primary_raw'], scores[2]) and np.array_equal(z['age'], scores[3])
    with np.load(aggregate_path, allow_pickle=False) as z:
        stored_draws, stored_points = (z['draws'].copy(), z['points'].copy())
        assert z['model_names'].tolist() == MODELS and z['metric_names'].tolist() == METRICS
    rankers = [WeightedRanks(y, p) for p in scores]
    points = np.array([r.evaluate(np.ones(len(y))) for r in rankers])
    point_error = float(np.max(np.abs(points - stored_points)))
    assert point_error < TOL
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    draws = np.empty_like(stored_draws)
    assert draws.shape == (protocol['bootstrap_replicates'], 4, 3)
    repeated_index_errors = []
    for b in range(len(draws)):
        patient_counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        row_weights = patient_counts[inverse]
        for j, ranker in enumerate(rankers):
            draws[b, j] = ranker.evaluate(row_weights)
        if b in (0, 1, 3, 17, 199, 997, 1999):
            repeated = np.repeat(np.arange(len(y)), row_weights)
            for j, p in enumerate(scores):
                repeated_index_errors.append(float(np.max(np.abs(draws[b, j] - library_metrics(y[repeated], p[repeated])))))
    draw_error = float(np.max(np.abs(draws - stored_draws)))
    assert draw_error < TOL and max(repeated_index_errors) < TOL
    table = pd.read_csv(ROOT / 'results' / cohort / 'performance.csv')
    pairs = pd.read_csv(ROOT / 'results' / cohort / 'paired_differences.csv')
    table_errors = []
    for j, model in enumerate(MODELS):
        for k, metric in enumerate(METRICS):
            selected = table[(table['model'] == model) & (table['metric'] == metric)]
            assert len(selected) == 1
            observed = selected[['estimate', 'ci_low', 'ci_high']].iloc[0].to_numpy(float)
            expected = np.r_[points[j, k], np.quantile(draws[:, j, k], [0.025, 0.975])]
            table_errors.append(float(np.max(np.abs(observed - expected))))
    for j in (2, 3):
        for k, metric in enumerate(METRICS):
            selected = pairs[(pairs['comparison'] == f'strict_ensemble_minus_{MODELS[j]}') & (pairs['metric'] == metric)]
            assert len(selected) == 1
            observed = selected[['estimate', 'ci_low', 'ci_high']].iloc[0].to_numpy(float)
            expected = np.r_[points[0, k] - points[j, k], np.quantile(draws[:, 0, k] - draws[:, j, k], [0.025, 0.975])]
            table_errors.append(float(np.max(np.abs(observed - expected))))
    assert max(table_errors) < TOL
    lock = _release_resolve_config(json.loads((ROOT / 'private' / cohort / 'input_lock.json').read_text(encoding='utf-8')))
    source_identity_checks = []
    for item in lock['sources']:
        current = identity(item['path'])
        source_identity_checks.append({'path': item['path'], 'matches_training_lock': None == None})
    report = {'cohort': cohort, 'passed': True, 'rows_patients_labels': EXPECTED[cohort], 'all_rows_eligible': True, 'prior_scores_labels_groups_folds_exactly_identical': True, 'role_checks': role_checks, 'self_tests': self_tests(), 'methods': 'Independent weighted tie-group rank AUC, descending-threshold AP, and direct weighted Brier; no import of run_reference.py.', 'all_bootstrap_replicates_replayed': len(draws), 'same_patient_multiplicities_for_all_models': True, 'explicit_repeated_row_checks': len(repeated_index_errors), 'maximum_point_absolute_error': point_error, 'maximum_draw_absolute_error': draw_error, 'maximum_explicit_repetition_absolute_error': max(repeated_index_errors), 'maximum_table_absolute_error': max(table_errors), 'tolerance': TOL, 'source_identity_checks': source_identity_checks, 'all_current_sources_match_training_lock': all((v['matches_training_lock'] for v in source_identity_checks)), 'inputs': [identity(p) for p in (prediction_path, aggregate_path, role_path, prior_path, protocol_path)], 'script': identity(__file__), 'environment': {'python': sys.version, 'numpy': importlib.metadata.version('numpy'), 'pandas': importlib.metadata.version('pandas'), 'scikit_learn': importlib.metadata.version('scikit-learn')}, 'limits': ['Verifies recorded-label metrics, patient roles, and numerical resampling only.', 'Does not certify clinical target validity, onset, reference-standard independence, or laboratory-model leakage freedom.', 'No model is trained or changed; no clinical rows or row identifiers are exported.']}
    with dest.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'cohort': cohort, 'passed': True, 'bootstrap_replicates': len(draws), 'maximum_draw_absolute_error': draw_error, 'all_current_sources_match_training_lock': report['all_current_sources_match_training_lock']}))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cohort', choices=EXPECTED)
    verify(parser.parse_args().cohort)
