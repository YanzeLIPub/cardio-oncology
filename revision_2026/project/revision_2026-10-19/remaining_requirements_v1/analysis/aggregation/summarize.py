"""Aggregate fixed-OOF patient-bootstrap comparisons; no clinical row exports."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from prepare import ROOT, identity, save_json
VARIANTS = ('mean', 'first_record', 'first_observed', 'median', 'record_count')
METRICS = ('auc', 'average_precision', 'brier', 'e_over_o')
STRATA = ('all', '1', '2-4', '>=5')

def masks(counts):
    return (np.ones(len(counts), bool), counts == 1, (counts >= 2) & (counts <= 4), counts >= 5)

class Scores:

    def __init__(self, y, p):
        self.y, self.p = (y, p)
        self.order = np.argsort(-p, kind='stable')
        sorted_p = p[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]

    def evaluate(self, weights):
        positive = weights * self.y
        negative = weights * (1 - self.y)
        tp = np.add.reduceat(positive[self.order], self.starts)
        fp = np.add.reduceat(negative[self.order], self.starts)
        ctp = np.cumsum(tp)
        cfp = np.cumsum(fp)
        pos, neg = (positive.sum(), negative.sum())
        if pos == 0 or neg == 0:
            return np.full(4, np.nan)
        auc = np.sum((ctp - 0.5 * tp) * fp) / (pos * neg)
        denom = ctp + cfp
        precision = np.divide(ctp, denom, out=np.zeros_like(ctp, dtype=float), where=denom > 0)
        ap = np.sum(tp * precision) / pos
        brier = np.dot(weights, np.square(self.p - self.y)) / weights.sum()
        eo = np.dot(weights, self.p) / pos
        return np.asarray([auc, ap, brier, eo])

def percentiles(a):
    good = np.isfinite(a)
    assert good.sum() > 0
    return (float(np.quantile(a[good], 0.025)), float(np.quantile(a[good], 0.975)), int(good.sum()))

def main():
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8')))
    comparisons = protocol['primary_comparisons'] + protocol['secondary_comparisons']
    metric_rows, paired_rows, strata_rows = ([], [], [])
    arrays, points, sourcefiles = ({}, {}, [ROOT / 'protocol.json', ROOT / 'analysis_lock_v2.json'])
    count_model_coefficients = []
    import joblib
    for cidx, center in enumerate(('SCH', 'WHU')):
        path = ROOT / f'private/runs/{center}/predictions.npz'
        sourcefiles.append(path)
        data = dict(np.load(path, allow_pickle=False))
        prepared = dict(np.load(ROOT / f'private/prepared/{center}.npz', allow_pickle=False))
        assert data['processed_record_count'].dtype.kind in 'iu'
        assert np.array_equal(data['processed_record_count'], prepared['record_count'])
        assert (data['processed_record_count'] >= 1).all()
        assert ((data['record_count'] >= 0) & (data['record_count'] <= 1)).all()
        for fold in range(1, 6):
            model = joblib.load(ROOT / f'private/runs/{center}/fold_{fold}_record_count.joblib')
            count_model_coefficients.append({'center': center, 'fold': fold, 'standardized_log_count_coefficient': float(model[-1].coef_[0, 0]), 'intercept': float(model[-1].intercept_[0]), 'training_log_count_mean': float(model[0].mean_[0]), 'training_log_count_scale': float(model[0].scale_[0])})
        for sidx, (stratum, mask) in enumerate(zip(STRATA, masks(data['processed_record_count']))):
            y = data['y'][mask]
            n = len(y)
            assert len(np.unique(y)) == 2
            strata_rows.append({'center': center, 'record_stratum': stratum, 'n': n, 'positive': int(y.sum()), 'prevalence': float(y.mean()), 'unit': 'patient'})
            calculators = [Scores(y, data[v][mask]) for v in VARIANTS]
            point = np.array([calc.evaluate(np.ones(n)) for calc in calculators])
            for j, v in enumerate(VARIANTS):
                expected = [roc_auc_score(y, data[v][mask]), average_precision_score(y, data[v][mask]), brier_score_loss(y, data[v][mask]), data[v][mask].sum() / y.sum()]
                np.testing.assert_allclose(point[j], expected, rtol=0, atol=2e-14)
            seed = protocol['bootstrap_seed'] + cidx * 10000 + sidx
            rng = np.random.default_rng(seed)
            boot = np.full((protocol['bootstrap_replicates'], len(VARIANTS), len(METRICS)), np.nan)
            for b in range(len(boot)):
                w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
                for j, calc in enumerate(calculators):
                    boot[b, j] = calc.evaluate(w)
                if b % 500 == 0:
                    print(json.dumps({'center': center, 'stratum': stratum, 'bootstrap': b}), flush=True)
            key = f'{center}_{sidx}'
            arrays[key] = boot
            points[key] = point
            for j, v in enumerate(VARIANTS):
                for k, metric in enumerate(METRICS):
                    lo, hi, valid = percentiles(boot[:, j, k])
                    metric_rows.append({'center': center, 'record_stratum': stratum, 'variant': v, 'metric': metric, 'estimate': float(point[j, k]), 'ci_low': lo, 'ci_high': hi, 'n': n, 'positive': int(y.sum()), 'bootstrap_valid': valid, 'bootstrap_invalid': len(boot) - valid, 'bootstrap_seed': seed})
            for a, b in comparisons:
                ai, bi = (VARIANTS.index(a), VARIANTS.index(b))
                for k, metric in enumerate(METRICS):
                    lo, hi, valid = percentiles(boot[:, ai, k] - boot[:, bi, k])
                    paired_rows.append({'center': center, 'record_stratum': stratum, 'variant': a, 'reference': b, 'metric': metric, 'difference': float(point[ai, k] - point[bi, k]), 'ci_low': lo, 'ci_high': hi, 'n': n, 'positive': int(y.sum()), 'bootstrap_valid': valid, 'bootstrap_invalid': len(boot) - valid})
    cross_rows = []
    for sidx, stratum in enumerate(STRATA):
        for a, b in comparisons:
            ai, bi = (VARIANTS.index(a), VARIANTS.index(b))
            for k, metric in enumerate(METRICS):
                wd = arrays[f'WHU_{sidx}'][:, ai, k] - arrays[f'WHU_{sidx}'][:, bi, k]
                sd = arrays[f'SCH_{sidx}'][:, ai, k] - arrays[f'SCH_{sidx}'][:, bi, k]
                point = points[f'WHU_{sidx}'][ai, k] - points[f'WHU_{sidx}'][bi, k] - points[f'SCH_{sidx}'][ai, k] + points[f'SCH_{sidx}'][bi, k]
                lo, hi, valid = percentiles(wd - sd)
                cross_rows.append({'contrast': 'WHU minus SCH paired gains', 'record_stratum': stratum, 'variant': a, 'reference': b, 'metric': metric, 'difference_of_differences': float(point), 'ci_low': lo, 'ci_high': hi, 'bootstrap_valid': valid})
    pd.DataFrame(metric_rows).to_csv(ROOT / 'results/metrics.csv', index=False)
    pd.DataFrame(paired_rows).to_csv(ROOT / 'results/paired_comparisons.csv', index=False)
    pd.DataFrame(cross_rows).to_csv(ROOT / 'results/between_centre_gains.csv', index=False)
    pd.DataFrame(strata_rows).to_csv(ROOT / 'results/evaluation_populations.csv', index=False)
    pd.DataFrame(count_model_coefficients).to_csv(ROOT / 'results/record_count_coefficients.csv', index=False)
    with (ROOT / 'private/bootstrap.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays, **{k + '_point': v for k, v in points.items()}, variants=np.array(VARIANTS), metrics=np.array(METRICS), strata=np.array(STRATA))
    save_json(ROOT / 'results/summary.json', {'analysis_executed': True, 'independent_verification_required': True, 'inputs': [identity(p) for p in sourcefiles], 'summary_script': identity(__file__), 'n_metric_rows': len(metric_rows), 'n_paired_rows': len(paired_rows), 'n_between_centre_rows': len(cross_rows), 'all_patient_metrics': [row for row in metric_rows if row['record_stratum'] == 'all'], 'primary_paired_results': [row for row in paired_rows if row['record_stratum'] == 'all' and [row['variant'], row['reference']] in protocol['primary_comparisons']], 'between_centre_primary_gains': [row for row in cross_rows if row['record_stratum'] == 'all' and [row['variant'], row['reference']] in protocol['primary_comparisons']], 'count_schema': 'processed_record_count is the original integer record number used for strata; record_count is held-out logistic probability. Exact raw-count equality checked against prepared matrix.', 'scope': 'Matched laboratory aggregation in historical recorded-label classification; original AUC and encounter-model analysis remain separate.', 'limits': ['All-record label and predictors do not prove pre-event ordering', 'Same nominal 104 inputs; 10 nonlaboratory columns fixed at original means', 'Record counts are processed check rows, not verified distinct visits', 'First-observed values may span many dates', 'Subgroup estimates use whole-cohort models, not stratum-specific fits', 'Bootstrap intervals exclude refitting uncertainty', 'Changes in predictions reflect summary values, missingness, selected rounds and fitted trees jointly; not a causal decomposition']})
if __name__ == '__main__':
    main()
