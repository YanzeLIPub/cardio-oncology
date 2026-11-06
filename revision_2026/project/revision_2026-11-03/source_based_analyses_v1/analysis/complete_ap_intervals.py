"""Conditional AP intervals from saved OOF predictions; no model refitting."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
R = Path(__file__).resolve().parents[1]
with np.load(R / 'private/measurement_caps_predictions.npz') as data:
    y = data['y']
    draws = np.random.default_rng(20261103).integers(0, len(y), (2000, len(y)))
    draws = draws[(y[draws].sum(axis=1) > 0) & (y[draws].sum(axis=1) < len(y))]
    rows = []
    for scope in ['h0', 'h24', 'h48', 'whole']:
        for model in ['logistic', 'ensemble']:
            p = data[scope + '__' + model]
            vals = [average_precision_score(y[d], p[d]) for d in draws]
            low, high = np.quantile(vals, [0.025, 0.975])
            rows.append(dict(scope=scope, model=model, ap_low=low, ap_high=high))
pd.DataFrame(rows).to_csv(R / 'aggregate/whu_primary_ap_intervals.csv', index=False)
