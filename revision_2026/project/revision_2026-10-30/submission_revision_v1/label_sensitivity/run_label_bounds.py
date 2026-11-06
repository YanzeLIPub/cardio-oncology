"""Exact fixed-score label-perturbation stress tests. Aggregate output only.

Never changes original labels, scores, or training. All hypothetical rates are
scenario choices, not estimated label error rates or confidence intervals.
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import argparse, hashlib, importlib.metadata, json, platform, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PRIOR = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
CENTRES = ('SCH', 'WHU', 'MIMIC')
EXPECT = {'SCH': (38509, 38509, 1323), 'WHU': (2562, 2562, 281), 'MIMIC': (9946, 6783, 4462)}
SCORES = ('original', 'primary_raw')
RATES = (0.01, 0.05, 0.1)
FAMILIES = ('balanced_exchange', 'positive_to_negative_only', 'negative_to_positive_only')

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def bounds(y, rank, remove, add, contrast=False):
    """Extrema over exactly remove original positives and add original negatives.

    AUC=(sum positive average ranks - K(K+1)/2)/(K*(N-K)).
    Once K and counts changed in each original label class are fixed, minimizing
    the numerator means removing its highest positive ranks and adding its lowest
    negative ranks. For paired AUC difference replace rank by rank_A-rank_B;
    the common K(K+1)/2 cancels. Selection is shared across both score vectors.
    """
    y = np.asarray(y)
    rank = np.asarray(rank, dtype=float)
    pos = np.sort(rank[y == 1])
    neg = np.sort(rank[y == 0])
    k = len(pos) - remove + add
    n = len(y)
    assert 0 <= remove <= len(pos) and 0 <= add <= len(neg) and (0 < k < n)
    base = pos.sum()
    constant = 0.0 if contrast else k * (k + 1) / 2
    lo = (base - (pos[-remove:].sum() if remove else 0) + neg[:add].sum() - constant) / (k * (n - k))
    hi = (base - pos[:remove].sum() + (neg[-add:].sum() if add else 0) - constant) / (k * (n - k))
    return (float(lo), float(hi))

def self_test():
    rng = np.random.default_rng(20261029)
    checks = 0
    cases = 0
    for n in range(4, 9):
        for case in range(4):
            y = np.array([1] * (n // 2) + [0] * (n - n // 2))
            rng.shuffle(y)
            a = rng.integers(0, 4, n).astype(float)
            b = rng.integers(0, 4, n).astype(float)
            ra = rankdata(a)
            rb = rankdata(b)
            p = np.flatnonzero(y)
            q = np.flatnonzero(1 - y)
            for remove in range(min(2, len(p)) + 1):
                for add in range(min(2, len(q)) + 1):
                    if not 0 < len(p) - remove + add < n:
                        continue
                    va = []
                    vd = []
                    for rem in combinations(p, remove):
                        for ins in combinations(q, add):
                            yy = y.copy()
                            yy[list(rem)] = 0
                            yy[list(ins)] = 1
                            aa = roc_auc_score(yy, a)
                            bb = roc_auc_score(yy, b)
                            va.append(aa)
                            vd.append(aa - bb)
                            checks += 2
                    assert np.allclose(bounds(y, ra, remove, add), [min(va), max(va)], atol=1e-12, rtol=0)
                    assert np.allclose(bounds(y, ra - rb, remove, add, True), [min(vd), max(vd)], atol=1e-12, rtol=0)
                    cases += 1
    return {'passed': True, 'enumerated_score_checks': checks, 'scenarios': cases, 'ties_included': True, 'independent_reference': 'sklearn ROC AUC over every allowed labeling in synthetic tiny cases', 'tolerance': 1e-12}

def freeze():
    paths = [PRIOR / 'private/runs' / f'{c}_original' / 'predictions.npz' for c in CENTRES]
    prepared = [PRIOR / 'private/prepared' / f'{c}.npz' for c in CENTRES]
    protocol = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'Quantify conditional score sensitivity to hypothetical recorded-label errors; not clinical adjudication.', 'centres': list(CENTRES), 'score_channels': list(SCORES), 'scenario_rates_of_original_positive_count': list(RATES), 'm_rule': 'floor(rate * number of original positive labels); each arm uses this same integer count.', 'families': {'balanced_exchange': 'm original positives to negative AND m original negatives to positive; prevalence unchanged.', 'positive_to_negative_only': 'm original positives to negative, with no opposite changes; prevalence decreases.', 'negative_to_positive_only': 'm original negatives to positive, with no opposite changes; prevalence increases.'}, 'unit': 'Native analytic row: patient in SCH/WHU, admission in MIMIC. Repeated admissions are not constrained to share a hypothetical change.', 'bounds': 'Exact deterministic minimum/maximum fixed-score AUC over exactly the stipulated number of changes in each class, using average ranks for ties.', 'paired_comparison': 'Patient-isolated primary_raw minus historical original on identical rows with one shared hypothetical label perturbation; optimize the rank difference.', 'restrictions': ['Rates are hypothetical scenario budgets, not observed or plausible clinical error-rate estimates.', 'Bounds are not confidence or credible intervals and have no probability interpretation.', 'Labels used to train the frozen models are unchanged; this does not evaluate refitting under corrected labels.', 'No clinical reviewer has classified these scenarios; no synthetic label vector is stored or substituted into the primary dataset.', 'No event timing, future-event prediction, independent clinical truth, common endpoints or clinical utility are certified.', 'Extrema for two separately reported models need not arise from the same perturbation; the paired contrast has its own shared perturbation.'], 'no_training': True, 'expected_cohorts': EXPECT, 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': {k: importlib.metadata.version(k) for k in ('numpy', 'pandas', 'scipy', 'scikit-learn')}}, 'inputs': [identity(x) for x in paths + prepared], 'script': identity(__file__)}
    save(ROOT / 'protocol.json', protocol)
    print(json.dumps({'protocol_frozen': True, 'source_files': len(paths + prepared), 'no_outcomes_calculated': True}))

def run():
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8')))
    for x in protocol['inputs'] + [protocol['script']]:
        assert identity(x['path']) == x
    test = self_test()
    save(ROOT / 'mathematical_verification.json', test)
    rows = []
    paired = []
    baseline = []
    for c in CENTRES:
        with np.load(PRIOR / 'private/runs' / f'{c}_original' / 'predictions.npz', allow_pickle=False) as z:
            y = z['y'].copy()
            g = z['groups'].copy()
            scores = {k: z[k].copy() for k in SCORES}
        with np.load(PRIOR / 'private/prepared' / f'{c}.npz', allow_pickle=False) as z:
            assert np.array_equal(y, z['y']) and np.array_equal(g, z['groups']) and np.array_equal(scores['original'], z['p_original'])
        assert (len(y), len(np.unique(g)), int(y.sum())) == EXPECT[c]
        ranks = {k: rankdata(v, method='average') for k, v in scores.items()}
        n = len(y)
        p = int(y.sum())
        negative = n - p
        points = {k: float(roc_auc_score(y, v)) for k, v in scores.items()}
        for k in SCORES:
            assert np.allclose(bounds(y, ranks[k], 0, 0), points[k], atol=1e-12, rtol=0)
            baseline.append({'centre': c, 'score_channel': k, 'rows': n, 'patients': len(np.unique(g)), 'positive_labels': p, 'auc': points[k]})
        for rate in RATES:
            m = int(np.floor(rate * p))
            for family in FAMILIES:
                remove = m if family != 'negative_to_positive_only' else 0
                add = m if family != 'positive_to_negative_only' else 0
                common = {'centre': c, 'scenario': family, 'nominal_rate_of_original_positive_count': rate, 'positive_to_negative': remove, 'negative_to_positive': add, 'fraction_original_positives_changed': remove / p, 'fraction_original_negatives_changed': add / negative, 'fraction_all_rows_changed': (remove + add) / n, 'positive_count_after_hypothetical_change': p - remove + add}
                for k in SCORES:
                    lo, hi = bounds(y, ranks[k], remove, add)
                    rows.append({**common, 'score_channel': k, 'original_auc': points[k], 'auc_lower_bound': lo, 'auc_upper_bound': hi})
                lo, hi = bounds(y, ranks['primary_raw'] - ranks['original'], remove, add, True)
                paired.append({**common, 'contrast': 'patient_isolated_minus_original', 'original_auc_difference': points['primary_raw'] - points['original'], 'difference_lower_bound': lo, 'difference_upper_bound': hi})
    for name, data in [('baseline_auc.csv', baseline), ('label_perturbation_auc_bounds.csv', rows), ('paired_auc_difference_bounds.csv', paired)]:
        pd.DataFrame(data).to_csv(ROOT / name, index=False, encoding='utf-8', mode='x')
    assert all((identity(x['path']) == x for x in protocol['inputs']))
    save(ROOT / 'completion.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': identity(ROOT / 'protocol.json'), 'outputs': [identity(ROOT / x) for x in ['baseline_auc.csv', 'label_perturbation_auc_bounds.csv', 'paired_auc_difference_bounds.csv', 'mathematical_verification.json']], 'source_files_unchanged': True, 'auc_scenarios': len(rows), 'paired_scenarios': len(paired), 'clinical_adjudication_completed': False})
    print(json.dumps({'completed': True, 'auc_scenarios': len(rows), 'paired_scenarios': len(paired), 'self_test': test}))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'run'])
    args = parser.parse_args()
    freeze() if args.mode == 'freeze' else run()
