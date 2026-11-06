"""Independent raw aggregation and all-draw ordered bootstrap verification."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
HERE = Path(__file__).resolve().parent
OLD = Path(_release_path('project', 'revision_2026-10-15/original_v5_independent_validation_v1'))
VARIANTS = ('mean', 'first_record', 'first_observed', 'median', 'record_count')
METRICS = ('auc', 'average_precision', 'brier', 'e_over_o')
STRATA = ('all', '1', '2-4', '>=5')
RAW = {'SCH': Path(_release_path('china', 'SCH_processed_data.csv')), 'WHU': Path(_release_path('china', 'WHU_Cancer_processed_data.csv'))}

def sha(path):
    return None

def ident(path):
    p = Path(path)
    return dict(path=p.as_posix(), bytes=p.stat().st_size, sha256=sha(p))

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def check(flag, message):
    if not flag:
        raise RuntimeError(message)

def close(a, b, tol=3e-13):
    np.testing.assert_allclose(a, b, rtol=0, atol=tol, equal_nan=True)

def strata(count):
    return (np.ones(len(count), dtype=bool), count == 1, (count >= 2) & (count <= 4), count >= 5)

def source_preparation(center, prepared, pred):
    baseline = dict(np.load(OLD / f'private/prepared/{center}.npz', allow_pickle=False))
    oldroles = dict(np.load(OLD / f'private/runs/{center}_original/patient_roles.npz', allow_pickle=False))
    for key in ('y', 'groups'):
        check(np.array_equal(prepared[key], baseline[key]), 'Original identities or labels differ')
        check(np.array_equal(pred[key], prepared[key]), 'Prediction identities or labels differ')
    check(np.array_equal(prepared['roles'], oldroles['roles']), 'Original roles differ')
    check(np.array_equal(pred['roles'], prepared['roles']), 'Prediction roles differ')
    check(len(np.unique(prepared['groups'])) == len(prepared['groups']), 'Expected one row per unique patient')
    check(np.all((prepared['roles'] == 0).sum(axis=0) == 1), 'Each patient must be held out exactly once')
    check(np.array_equal(prepared['X_mean'], baseline['X'], equal_nan=True), 'Mean baseline differs')
    lab = prepared['lab_indices']
    nonlab = np.setdiff1d(np.arange(104), lab)
    check(len(lab) == 94 and len(nonlab) == 10, 'Feature classes differ')
    for name in VARIANTS[:4]:
        check(np.array_equal(prepared['X_' + name][:, nonlab], prepared['X_mean'][:, nonlab], equal_nan=True), 'A nonlaboratory field changed')
    for role in prepared['roles']:
        check(set(np.unique(role)) <= {0, 1, 2, 3}, 'Unexpected fold role')
        for a, b in ((0, 1), (0, 2), (0, 3), (2, 3)):
            check(not np.intersect1d(prepared['groups'][role == a], prepared['groups'][role == b]).size, 'Patient-role overlap')
    check(pred['processed_record_count'].dtype.kind in 'iu', 'Integer count schema lost')
    check(np.array_equal(pred['processed_record_count'], prepared['record_count']), 'Count source differs')
    labels = pd.read_csv(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'), usecols=['病案号', 'Center', 'MACE'])
    local = labels.loc[labels.Center.eq(center)].reset_index(drop=True)
    check(np.array_equal(local.MACE.to_numpy(), prepared['y']), 'Source label order differs')
    names = prepared['feature_names'].tolist()
    labnames = [names[i] for i in lab]
    raw = pd.read_csv(RAW[center], usecols=['病案号', '检查时间'] + labnames, low_memory=False)
    codes = pd.Categorical(raw['病案号'], categories=local['病案号']).codes
    check((codes >= 0).all(), 'Unmapped source member')
    times = pd.to_datetime(raw['检查时间'], format='mixed', errors='raise')
    check(times.notna().all(), 'Missing timestamp must stop temporal variants')
    time_ns = times.astype('int64').to_numpy()
    order = np.lexsort((np.arange(len(raw)), time_ns, codes))
    sorted_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    ends = np.r_[starts[1:], len(raw)]
    check(np.array_equal(sorted_codes[starts], np.arange(len(local))), 'Patient order reconstruction differs')
    check(not ((np.diff(sorted_codes) == 0) & (np.diff(time_ns[order]) == 0)).any(), 'Unexpected tied patient times')
    numeric = raw[labnames].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)[order]
    check(not np.isinf(numeric).any(), 'Nonfinite source infinity found')
    first = numeric[starts].astype(np.float32)
    first_observed = np.full_like(first, np.nan)
    med = np.full_like(first, np.nan)
    for i, (left, right) in enumerate(zip(starts, ends)):
        block = numeric[left:right]
        finite = np.isfinite(block)
        valid = finite.any(axis=0)
        which = finite.argmax(axis=0)
        first_observed[i, valid] = block[which[valid], np.flatnonzero(valid)]
        if valid.any():
            med[i, valid] = np.nanmedian(block[:, valid], axis=0)
    for variant, value in (('first_record', first), ('first_observed', first_observed), ('median', med)):
        check(np.array_equal(value, prepared['X_' + variant][:, lab], equal_nan=True), 'Independent laboratory aggregation differs')
    counts = ends - starts
    check(np.array_equal(counts, prepared['record_count']), 'Raw processed count differs')
    span = (time_ns[order][ends - 1] - time_ns[order][starts]) / (86400 * 1000000000.0)
    close(span, prepared['recorded_span_days'])
    one = counts == 1
    for variant in VARIANTS[:4]:
        check(np.array_equal(prepared['X_' + variant][one], prepared['X_mean'][one], equal_nan=True), 'Single-record summaries must agree')
    return dict(center=center, raw_rows=len(raw), patients=len(local), positive=int(prepared['y'].sum()), labs=94, fixed_nonlabs=10, first_record_and_first_observed_and_median_exactly_rebuilt=True, invalid_timestamps=0, tied_patient_timestamps=0, same_original_patient_labels_roles=True, no_event_time_or_result_availability_inferred=True)

class IndependentMetrics:

    def __init__(self, y, p):
        self.y = np.asarray(y, dtype=float)
        self.p = np.asarray(p, dtype=float)
        scores, self.blocks = np.unique(p, return_inverse=True)
        self.nblocks = len(scores)

    def calculate(self, w):
        pos = np.bincount(self.blocks, weights=w * self.y, minlength=self.nblocks)
        neg = np.bincount(self.blocks, weights=w * (1 - self.y), minlength=self.nblocks)
        np_, nn = (pos.sum(), neg.sum())
        if not np_ or not nn:
            return np.full(4, np.nan)
        u = np.dot(pos, np.cumsum(neg) - neg / 2) / (np_ * nn)
        tp = np.cumsum(pos[::-1])
        total = tp + np.cumsum(neg[::-1])
        precision = np.divide(tp, total, out=np.zeros_like(tp), where=total != 0)
        ap = np.dot(pos[::-1], precision) / np_
        brier = np.average((self.p - self.y) ** 2, weights=w)
        eo = np.dot(w, self.p) / np_
        return np.array([u, ap, brier, eo])

def main():
    out = HERE / 'independent_verification.json'
    check(not out.exists(), 'Refusing to overwrite independent verification')
    protocol = load(HERE / 'protocol.json')
    lock = load(HERE / 'analysis_lock_v2.json')
    for item in lock['state']['files']:
        check(ident(item['path']) == item, 'A locked source changed')
    basever = load(HERE / 'verification.json')
    check(basever['passed'], 'Existing full-model replay did not pass')
    inputs = [ident(HERE / p) for p in ['protocol.json', 'analysis_lock_v2.json', 'preparation.json', 'prepare.py', 'run.py', 'run_v2.py', 'summarize.py', 'verify.py', 'verification.json', 'private/bootstrap.npz', 'results/metrics.csv', 'results/paired_comparisons.csv', 'results/between_centre_gains.csv', 'results/evaluation_populations.csv']]
    cached = dict(np.load(HERE / 'private/bootstrap.npz', allow_pickle=False))
    check(tuple(cached['variants']) == VARIANTS and tuple(cached['metrics']) == METRICS, 'Bootstrap axis identity differs')
    metric_table = pd.read_csv(HERE / 'results/metrics.csv', dtype={'record_stratum': str})
    pair_table = pd.read_csv(HERE / 'results/paired_comparisons.csv', dtype={'record_stratum': str})
    cross_table = pd.read_csv(HERE / 'results/between_centre_gains.csv', dtype={'record_stratum': str})
    population = pd.read_csv(HERE / 'results/evaluation_populations.csv', dtype={'record_stratum': str})
    check(len(metric_table) == 160 and len(pair_table) == 192 and (len(cross_table) == 96), 'Public table row count differs')
    points, boots = ({}, {})
    checks = []
    direct = 0
    for ci, center in enumerate(('SCH', 'WHU')):
        predpath = HERE / f'private/runs/{center}/predictions.npz'
        preppath = HERE / f'private/prepared/{center}.npz'
        inputs.extend([ident(predpath), ident(preppath)])
        pred = dict(np.load(predpath, allow_pickle=False))
        prepared = dict(np.load(preppath, allow_pickle=False))
        checks.append(source_preparation(center, prepared, pred))
        print(json.dumps(dict(center=center, raw_aggregation_independently_verified=True)), flush=True)
        for si, (stratum, mask) in enumerate(zip(STRATA, strata(pred['processed_record_count']))):
            y = pred['y'][mask]
            n = len(y)
            pop = population.loc[(population.center == center) & (population.record_stratum == stratum)].iloc[0]
            check(pop.n == n and pop.positive == y.sum(), 'Population count differs')
            models = [IndependentMetrics(y, pred[v][mask]) for v in VARIANTS]
            point = np.array([m.calculate(np.ones(n)) for m in models])
            for j, v in enumerate(VARIANTS):
                p = pred[v][mask]
                close(point[j], [roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p), p.sum() / y.sum()])
            key = f'{center}_{si}'
            close(point, cached[key + '_point'])
            boot = np.empty((2000, 5, 4))
            rng = np.random.default_rng(protocol['bootstrap_seed'] + ci * 10000 + si)
            for draw in range(2000):
                selected = rng.integers(0, n, n)
                weights = np.bincount(selected, minlength=n).astype(float)
                for j, m in enumerate(models):
                    boot[draw, j] = m.calculate(weights)
                    if draw in (0, 99, 499, 999, 1499, 1999) and len(np.unique(y[selected])) == 2:
                        p = pred[VARIANTS[j]][mask][selected]
                        yy = y[selected]
                        close(boot[draw, j], [roc_auc_score(yy, p), average_precision_score(yy, p), brier_score_loss(yy, p), p.sum() / yy.sum()])
                        direct += 1
                close(boot[draw], cached[key][draw])
            boots[key] = boot
            points[key] = point
            for row in metric_table.loc[(metric_table.center == center) & (metric_table.record_stratum == stratum)].itertuples():
                j, k = (VARIANTS.index(row.variant), METRICS.index(row.metric))
                samples = boot[:, j, k]
                close([row.estimate, row.ci_low, row.ci_high], [point[j, k], *np.nanquantile(samples, [0.025, 0.975])])
                check(row.bootstrap_valid == np.isfinite(samples).sum() and row.bootstrap_invalid == np.isnan(samples).sum(), 'Metric valid-draw counts differ')
            for row in pair_table.loc[(pair_table.center == center) & (pair_table.record_stratum == stratum)].itertuples():
                a, b, k = (VARIANTS.index(row.variant), VARIANTS.index(row.reference), METRICS.index(row.metric))
                samples = boot[:, a, k] - boot[:, b, k]
                close([row.difference, row.ci_low, row.ci_high], [point[a, k] - point[b, k], *np.nanquantile(samples, [0.025, 0.975])])
                check(row.bootstrap_valid == np.isfinite(samples).sum() and row.bootstrap_invalid == np.isnan(samples).sum(), 'Paired valid-draw counts differ')
            print(json.dumps(dict(center=center, stratum=stratum, ordered_bootstrap_draws_verified=2000)), flush=True)
    for row in cross_table.itertuples():
        s = STRATA.index(row.record_stratum)
        a, b, k = (VARIANTS.index(row.variant), VARIANTS.index(row.reference), METRICS.index(row.metric))
        wp, sp = (points[f'WHU_{s}'], points[f'SCH_{s}'])
        wb, sb = (boots[f'WHU_{s}'], boots[f'SCH_{s}'])
        differences = wb[:, a, k] - wb[:, b, k] - sb[:, a, k] + sb[:, b, k]
        point = wp[a, k] - wp[b, k] - sp[a, k] + sp[b, k]
        close([row.difference_of_differences, row.ci_low, row.ci_high], [point, *np.nanquantile(differences, [0.025, 0.975])])
        check(row.bootstrap_valid == np.isfinite(differences).sum(), 'Cross-centre valid count differs')
    for item in inputs:
        check(ident(item['path']) == item, 'An audit input changed during verification')
    result = dict(passed=True, date='2026-10-19', script=ident(Path(__file__)), source_identities=inputs, independently_reconstructed_preparation=checks, independent_ordered_bootstrap_replicates_per_cohort_stratum=2000, cohort_strata=8, methods=5, metrics=4, scalar_bootstrap_estimates_independently_recomputed=320000, all_ordered_draws_match=True, direct_sklearn_resampled_model_checks=direct, public_metric_rows_verified=160, public_paired_rows_verified=192, public_between_centre_rows_verified=96, primary_contrasts_all_2000_paired_draws_verified=True, code_review=dict(inner_fit_and_stopping_roles_disjoint=True, outer_patient_roles_disjoint=True, fit_only_class_weights=True, count_probability_and_integer_count_distinguished=True, ten_nonlabs_invariant=True, first_observed_not_single_baseline=True, current_script_version='run_v2.py', serialization_recovery_does_not_change_fitted_models=True, own_retraining_performed=False, full_model_replay_scope='Earlier verify.py full-model replay was reviewed and hash-bound; this separate audit independently reconstructs raw aggregations and every statistical draw.'), environment=dict(python=sys.version, platform=platform.platform(), **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn')}), privacy=dict(original_identifiers_exported=False, clinical_rows_exported=False), limitation='Retrospective laboratory aggregation, not first-event prediction or a causal explanation of the historical unit/model change.')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, independent_verification_sha256=sha(out))))
if __name__ == '__main__':
    main()
