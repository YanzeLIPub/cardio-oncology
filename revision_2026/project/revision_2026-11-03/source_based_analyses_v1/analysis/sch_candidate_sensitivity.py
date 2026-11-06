"""Conditional fixed-score sensitivity; candidate labels are not clinical corrections."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'revision_2026-11-03/source_gap_search_v1'))
from scan_sources import key
source = _release_path('china', 'outputs/ensemble_model/real_model_predictions_v2.csv')
d = pd.read_csv(source, usecols=['病案号', 'Center', 'MACE', 'predicted_prob_within_cv'], dtype={'病案号': str})
d = d[d.Center.eq('SCH')].reset_index(drop=True)
target = pd.read_csv(OUT / 'private/SCH_label_discrepancy_review.csv', usecols=['original_patient_id'], dtype=str)
ids = {key(v) for v in target.original_patient_id}
mask = d['病案号'].map(key).isin(ids).to_numpy()
assert len(d) == 38509 and mask.sum() == 7 and (d.MACE.to_numpy()[mask].sum() == 0)
y = d.MACE.to_numpy(int)
alt = y.copy()
alt[mask] = 1
p = d.predicted_prob_within_cv.to_numpy(float)
_, bins = np.unique(p, return_inverse=True)
size = bins.max() + 1

def auc(y, w):
    pos = np.bincount(bins, weights=w * y, minlength=size)
    neg = np.bincount(bins, weights=w * (1 - y), minlength=size)
    return np.dot(pos, np.cumsum(neg) - 0.5 * neg) / (pos.sum() * neg.sum())
w = np.ones(len(y))
a0 = auc(y, w)
a1 = auc(alt, w)
rng = np.random.default_rng(20261103)
dist = []
for _ in range(2000):
    w = rng.multinomial(len(y), np.full(len(y), 1 / len(y)))
    dist.append(auc(alt, w) - auc(y, w))
lo, hi = np.quantile(dist, [0.025, 0.975])
r = {'patients': len(y), 'original_positive_labels': int(y.sum()), 'candidate_flips': int(mask.sum()), 'hypothetical_positive_labels': int(alt.sum()), 'original_fixed_auc': a0, 'hypothetical_fixed_auc': a1, 'paired_auc_difference': a1 - a0, 'paired_auc_difference_low': lo, 'paired_auc_difference_high': hi, 'bootstrap_replicates': 2000, 'source': source, 'probability_column': 'predicted_prob_within_cv', 'model': 'Historical original within-SCH CV ensemble; fixed probabilities, no refit', 'interpretation': 'All seven source-coded/POA candidates hypothetically treated positive; not adjudicated labels, not an estimate of global error rate; original analysis files retained.'}
(OUT / 'aggregate/sch_candidate_label_sensitivity.json').write_text(json.dumps(r, indent=2), encoding='utf-8')
pd.DataFrame([r]).to_csv(OUT / 'aggregate/sch_candidate_label_sensitivity.csv', index=False)
print(json.dumps(r))
