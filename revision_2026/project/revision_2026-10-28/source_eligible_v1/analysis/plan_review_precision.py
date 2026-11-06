"""Adjustable planning examples only; no clinical data or sample selection."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import csv
import math
from pathlib import Path
from statistics import NormalDist

def plan(p, d, confidence, design_effect, nonevaluable):
    if not 0 < p < 1 or not 0 < d < 1 or (not 0 < confidence < 1):
        raise ValueError('Require 0 < p, half-width, confidence < 1')
    if design_effect < 1 or not 0 <= nonevaluable < 1:
        raise ValueError('Require design effect >= 1 and 0 <= non-evaluable fraction < 1')
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    n = math.ceil(z * z * p * (1 - p) / (d * d))
    wilson_half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    effective_target = math.ceil(n * design_effect)
    return {'planning_proportion': p, 'target_half_width': d, 'confidence': confidence, 'independent_evaluable_units_approximation': n, 'Wilson_half_width_at_planning_proportion_and_n': wilson_half, 'assumed_design_effect': design_effect, 'inflated_evaluable_target': effective_target, 'assumed_nonevaluable_fraction': nonevaluable, 'units_to_request_approximation': math.ceil(effective_target / (1 - nonevaluable)), 'applies_per_center_and_original_label_stratum': True, 'observed_review_results': False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--precision', type=float, nargs='+', default=[0.05, 0.075, 0.1])
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--design-effect', type=float, default=1.0)
    parser.add_argument('--nonevaluable', type=float, default=0.0)
    args = parser.parse_args()
    rows = [plan(args.p, d, args.confidence, args.design_effect, args.nonevaluable) for d in args.precision]
    assert plan(0.5, 0.05, 0.95, 1, 0)['independent_evaluable_units_approximation'] == 385
    assert plan(0.5, 0.1, 0.95, 1, 0)['independent_evaluable_units_approximation'] == 97
    assert plan(0.5, 0.05, 0.95, 1.5, 0.1)['units_to_request_approximation'] == 643
    path = Path(__file__).resolve().parent / 'review_precision_options.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print('Planning table written; no records selected or adjudicated; 3 planning assertions passed.')
if __name__ == '__main__':
    main()
