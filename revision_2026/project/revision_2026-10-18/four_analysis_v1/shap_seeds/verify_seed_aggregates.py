"""Independent aggregate verification without importing the training runner."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import stdtr
from scipy.stats import false_discovery_control, rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2] / 'revision_2026-10-15/original_v5_independent_validation_v1'
SEEDS = (42, 43, 44, 45)
COHORTS = ('SCH', 'WHU')
METRICS = ('auc', 'average_precision', 'brier')

def load(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def ident(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def corr(raw, s):
    mask = np.isfinite(raw)
    x, v = (raw[mask], s[mask])
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(v)) < 2:
        return (None, None)
    rho = float(np.corrcoef(rankdata(x), rankdata(v))[0, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = rho * np.sqrt((len(x) - 2) / np.clip((rho + 1.0) * (1.0 - rho), 0, None))
    p = float(2 * stdtr(len(x) - 2, -abs(t)))
    return (rho, p)

def close(a, b, tolerance=3e-12):
    if a is None:
        assert pd.isna(b)
    else:
        assert abs(float(a) - float(b)) < tolerance

class PairCountMetrics:

    def __init__(self, y, p):
        self.y, self.p = (y, p)
        self.order = np.argsort(p, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(p[self.order])) + 1]

    def evaluate(self, w):
        a = np.add.reduceat((w * self.y)[self.order], self.starts)
        b = np.add.reduceat((w * (1 - self.y))[self.order], self.starts)
        pos, neg = (a.sum(), b.sum())
        assert min(pos, neg) > 0
        auc = np.sum(a * (np.cumsum(b) - 0.5 * b)) / (pos * neg)
        tp = np.cumsum(a[::-1])
        den = tp + np.cumsum(b[::-1])
        precision = np.divide(tp, den, out=np.zeros_like(den, dtype=float), where=den > 0)
        return np.array([auc, np.sum(a[::-1] * precision) / pos, np.dot(w, (self.p - self.y) ** 2) / w.sum()])

def direct(y, p, w=None):
    return np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)])

def main():
    summary = load(ROOT / 'results/summary.json')
    lock = load(ROOT / 'analysis_lock.json')
    identities = [ident(ROOT / p) for p in ('analysis_lock.json', 'results/summary.json', 'verification.json')]
    identities += lock['state']['sources'] + summary['outputs']
    assert load(ROOT / 'verification.json')['passed']
    for item in identities:
        assert ident(item['path']) == item
    association_table = pd.read_csv(ROOT / 'results/feature_associations.csv')
    fold_table = pd.read_csv(ROOT / 'results/fold_associations.csv')
    rank_table = pd.read_csv(ROOT / 'results/seed_rank_stability.csv')
    metric_table = pd.read_csv(ROOT / 'results/seed_metrics.csv')
    delta_table = pd.read_csv(ROOT / 'results/seed_paired_comparisons.csv')
    cache, rows, tested, mean_abs = ({}, [], 0, {})
    bootstrap_error = 0.0
    direct_bootstrap_checks = 0
    for c in COHORTS:
        original = arrays(BASE / f'private/prepared/{c}.npz')
        roles = arrays(BASE / f'private/runs/{c}_original/patient_roles.npz')['roles']
        X, y, names = (original['X'], original['y'], original['feature_names'].tolist())
        assert len(np.unique(original['groups'])) == len(y)
        ps = []
        for seed in SEEDS:
            p = ROOT / f'private/{c}/seed_{seed}/heldout.npz'
            identities.append(ident(p))
            z = arrays(p)
            ps.append(z['probability'])
            contributions = z['contributions'][:, :-1]
            mean_abs[c, seed] = np.abs(contributions).mean(axis=0, dtype=np.float64)
            for j, feature in enumerate(names):
                rho, pv = corr(X[:, j], contributions[:, j])
                current = association_table[(association_table.cohort == c) & (association_table.seed == seed) & (association_table.feature_index == j)].iloc[0]
                close(rho, current.rho)
                close(pv, current.p_nominal)
                assert bool(current.testable) == (rho is not None)
                close(mean_abs[c, seed][j], current.mean_abs_contribution_all)
                signs = []
                for fold, role in enumerate(roles, 1):
                    rr, pp = corr(X[role == 0, j], contributions[role == 0, j])
                    f = fold_table[(fold_table.cohort == c) & (fold_table.seed == seed) & (fold_table.feature_index == j) & (fold_table.fold == fold)].iloc[0]
                    close(rr, f.rho)
                    close(pp, f.p_nominal)
                    if rr is not None:
                        signs.append(int(np.sign(rr)))
                    tested += 1
                stable = bool(rho is not None and rho != 0 and (len(signs) == 5) and all((v == np.sign(rho) for v in signs)))
                assert stable == bool(current.all_five_fold_sign_agreement)
                assert [signs.count(1), signs.count(-1), signs.count(0), 5 - len(signs)] == [current.folds_positive, current.folds_negative, current.folds_zero, current.folds_untestable]
                rows.append({'cohort': c, 'seed': seed, 'feature': feature, 'rho': rho, 'p': pv, 'fold_stable': stable, 'row': current})
        caches = [PairCountMetrics(y, p) for p in ps]
        points = np.array([direct(y, p) for p in ps])
        saved = arrays(ROOT / f'private/{c}/bootstrap.npz')
        identities.append(ident(ROOT / f'private/{c}/bootstrap.npz'))
        actual = np.empty((2000, 4, 3))
        rng = np.random.default_rng(load(ROOT / 'protocol.json')['bootstrap']['seeds'][c])
        for b in range(2000):
            w = rng.multinomial(len(y), np.full(len(y), 1 / len(y)))
            actual[b] = [obj.evaluate(w) for obj in caches]
            if b < 10:
                assert np.allclose(actual[b], [direct(y, p, w) for p in ps], atol=3e-12, rtol=0)
                direct_bootstrap_checks += 4
        err = float(np.max(np.abs(actual - saved['draws'])))
        bootstrap_error = max(bootstrap_error, err)
        assert err < 3e-12
        for si, seed in enumerate(SEEDS):
            for ki, metric in enumerate(METRICS):
                row = metric_table[(metric_table.cohort == c) & (metric_table.seed == seed) & (metric_table.metric == metric)].iloc[0]
                low, high = np.quantile(actual[:, si, ki], [0.025, 0.975])
                close(points[si, ki], row.estimate)
                close(low, row.ci_low)
                close(high, row.ci_high)
                if si:
                    row = delta_table[(delta_table.cohort == c) & (delta_table.seed == seed) & (delta_table.metric == 'delta_' + metric)].iloc[0]
                    low, high = np.quantile(actual[:, si, ki] - actual[:, 0, ki], [0.025, 0.975])
                    close(points[si, ki] - points[0, ki], row.estimate)
                    close(low, row.ci_low)
                    close(high, row.ci_high)
        for first, second in itertools.combinations(SEEDS, 2):
            a, b = (mean_abs[c, first], mean_abs[c, second])
            r = rank_table[(rank_table.cohort == c) & (rank_table.first_seed == first) & (rank_table.second_seed == second)].iloc[0]
            nonzero = (a > 0) | (b > 0)
            close(spearmanr(a, b).statistic, r.rho_all)
            close(spearmanr(a[nonzero], b[nonzero]).statistic, r.rho_nonzero)
            for k in (10, 20):
                ia, ib = (np.argsort(-a, kind='stable'), np.argsort(-b, kind='stable'))
                assert len(set(ia[a[ia] > 0][:k]) & set(ib[b[ib] > 0][:k])) == r[f'top_{k}_overlap_n']
        print(f'Independent seed associations and all 2000 paired bootstrap draws checked: {c}', flush=True)
    for seed in SEEDS:
        family = [r for r in rows if r['seed'] == seed and r['rho'] is not None]
        q = false_discovery_control(np.array([r['p'] for r in family]), method='bh')
        for row, value in zip(family, q):
            close(value, row['row'].q_bh_joint_family)
            row['q'] = float(value)
        for row in rows:
            if row['seed'] == seed and row['rho'] is None:
                assert pd.isna(row['row'].q_bh_joint_family)
    cross = pd.read_csv(ROOT / 'results/cross_cohort_associations.csv')
    for seed in SEEDS:
        for feature in names:
            a = next((r for r in rows if r['seed'] == seed and r['cohort'] == 'SCH' and (r['feature'] == feature)))
            b = next((r for r in rows if r['seed'] == seed and r['cohort'] == 'WHU' and (r['feature'] == feature)))
            r = cross[(cross.seed == seed) & (cross.feature == feature)].iloc[0]
            valid = a['rho'] is not None and b['rho'] is not None
            opposite = bool(valid and a['rho'] * b['rho'] < 0)
            bothq = bool(valid and a['q'] < 0.05 and (b['q'] < 0.05))
            foldstable = a['fold_stable'] and b['fold_stable']
            assert [valid, opposite, bothq, foldstable, opposite and bothq and foldstable] == [r.testable_both, r.opposite_signs, r.both_q_below_0_05, r.both_all_five_fold_sign_agreement, r.opposite_both_q_and_five_fold_agreement]
    pub = load(ROOT / 'results/publication_summary.json')
    stable_sets = []
    for seed in SEEDS:
        selected = cross[(cross.seed == seed) & cross.opposite_both_q_and_five_fold_agreement].feature.tolist()
        stable_sets.append(set(selected))
        entry = next((x for x in pub['seed_results'] if x['seed'] == seed))
        assert entry['stable_opposite_features'] == selected
        assert entry['original_eight_retained_n'] == len(set(selected) & set(pub['original_eight']))
    assert set(pub['four_seed_intersection']) == set.intersection(*stable_sets)
    assert len(pub['NSE_Albumin']) == 16
    for item in identities:
        assert ident(item['path']) == item
    report = {'passed': True, 'completed_at': datetime.now(timezone.utc).isoformat(), 'script': ident(__file__), 'source_identities': list({x['path']: x for x in identities}.values()), 'feature_cohort_seed_associations': len(rows), 'fold_associations': tested, 'complete_BH_families_checked': 4, 'cross_cohort_seed_feature_rows': len(cross), 'seed_pair_rank_rows': len(rank_table), 'metric_CI_rows': len(metric_table), 'paired_CI_rows': len(delta_table), 'bootstrap_draws_per_cohort': 2000, 'maximum_bootstrap_difference': bootstrap_error, 'direct_sklearn_bootstrap_score_checks': direct_bootstrap_checks, 'publication_retention_and_example_completeness_checked': True, 'source_files_unchanged': True, 'no_training_or_patient_output': True, 'scope': 'Independent rank-correlation/t-distribution P implementation, scipy BH, pair-count ROC/AP bootstrap implementation, direct sklearn checks; no training-runner import. Saved-model replay is separately bound.'}
    dest = ROOT / 'qa/seed_aggregate_independent_verification.json'
    dest.parent.mkdir(exist_ok=True)
    with dest.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'passed': True, 'report': ident(dest)}))
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
