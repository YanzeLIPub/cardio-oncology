"""Independent aggregate verification; no patient rows are written or printed."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
CFG = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8')))
AGG = ROOT / 'aggregate'
Q = np.asarray(CFG['capacity_fractions'])
THRESHOLDS = np.asarray(CFG['numerical_thresholds'])
METRICS = ['selected_per_1000', 'selected_positive_per_1000', 'selected_negative_per_1000', 'positive_capture_percent', 'review_positive_percent']
CHECKS = []

def compare(name, observed, expected):
    left, right = (np.asarray(observed, float), np.asarray(expected, float))
    np.testing.assert_allclose(left, right, rtol=1e-09, atol=1e-08, equal_nan=True)
    finite = np.isfinite(left) & np.isfinite(right)
    maximum = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
    CHECKS.append({'check': name, 'values_compared': int(left.size), 'max_absolute_difference': maximum})

def grouped_scores(score, fold):
    """Use pandas score groups, not sorted starts/reduceat or boundary search."""
    result = []
    for f in np.unique(fold):
        take = np.flatnonzero(fold == f)
        frame = pd.DataFrame({'score': score[take]})
        ascending_groups = frame.groupby('score', sort=True).ngroup().to_numpy()
        count = int(ascending_groups.max()) + 1
        result.append((take, count - 1 - ascending_groups, count))
    return result

def allocate(groups, labels, weights):
    """Allocate each score group's remaining quota by clipping; sum all groups."""
    captured = np.zeros(len(Q))
    selected = np.zeros(len(Q))
    fold_error = 0.0
    for take, group, count in groups:
        mass = np.bincount(group, weights=weights[take], minlength=count)
        pos = np.bincount(group, weights=weights[take] * labels[take], minlength=count)
        before = np.cumsum(mass) - mass
        quotas = Q * mass.sum()
        assigned = np.clip(quotas[:, None] - before, 0.0, mass)
        fractions = np.divide(assigned, mass, out=np.zeros_like(assigned), where=mass > 0)
        selected += assigned.sum(axis=1)
        captured += fractions @ pos
        fold_error = max(fold_error, float(np.max(np.abs(assigned.sum(axis=1) - quotas))))
    return (captured, selected, fold_error)

def evaluate(labels, weights, groupings, scores):
    total, positive = (weights.sum(), np.dot(weights, labels))
    counts, names = ([], [])
    for model in CFG['capacity_models']:
        captured, selected, _ = allocate(groupings[model], labels, weights)
        counts.extend(zip(selected, captured))
        names.extend((f'{model}_capacity_{q:.2f}' for q in Q))
    counts.extend(zip(Q * total, Q * positive))
    names.extend((f'random_capacity_{q:.2f}' for q in Q))
    counts.extend([(total, positive), (0.0, 0.0)])
    names.extend(['review_all', 'review_none'])
    for model in CFG['threshold_models']:
        for threshold in THRESHOLDS:
            chosen = scores[model] >= threshold
            counts.append((weights[chosen].sum(), np.dot(weights[chosen], labels[chosen])))
            names.append(f'{model}_threshold_{threshold:.2f}')
    selected, captured = np.asarray(counts).T
    result = np.column_stack((selected / total * 1000, captured / total * 1000, (selected - captured) / total * 1000, captured / positive * 100, np.divide(captured * 100, selected, out=np.full_like(captured, np.nan), where=selected > 0)))
    return (names, result)

def interval(values):
    valid = values[np.isfinite(values)]
    return np.quantile(valid, [0.025, 0.975]) if valid.size else np.array([np.nan, np.nan])

def tie_example():
    score = np.r_[np.repeat(0.9, 3), np.repeat(0.2, 7), np.repeat(0.8, 2), np.repeat(0.1, 8)]
    labels = np.r_[1.0, 0.0, 0.0, np.zeros(7), 1.0, 1.0, np.zeros(8)]
    fold = np.repeat([1, 2], 10)
    captured, selected, error = allocate(grouped_scores(score, fold), labels, np.ones(20))
    compare('hand_calculated_boundary_tie_positive_count', captured, Q * 10 / 3 + Q * 10)
    compare('hand_calculated_fold_capacity', selected, Q * 20)
    assert error < 1e-10

def main():
    tie_example()
    policies = pd.read_csv(AGG / 'policy_metrics.csv')
    paired = pd.read_csv(AGG / 'paired_comparisons.csv')
    cohorts = pd.read_csv(AGG / 'cohort_definitions.csv')
    scenarios = pd.read_csv(AGG / 'effect_scenarios.csv')
    actionability = pd.read_csv(AGG / 'actionability_sensitivity.csv')
    examples = pd.read_csv(AGG / 'illustrative_10percent_capacity.csv')
    records = []
    for run_index, run in enumerate(CFG['runs']):
        source = Path(CFG['source_directory']) / run
        with np.load(source / 'predictions.npz', allow_pickle=False) as arrays:
            labels = arrays['y'].astype(float)
            fold = arrays['fold'].copy()
            groups = arrays['groups'].copy()
            scores = {name: arrays[name].copy() for name in set(CFG['capacity_models'] + CFG['threshold_models'])}
        unique, patient_index = np.unique(groups, return_inverse=True)
        patients = len(unique)
        weights = np.ones(len(labels))
        grouping = {name: grouped_scores(scores[name], fold) for name in CFG['capacity_models']}
        names, point = evaluate(labels, weights, grouping, scores)
        policy_index = {name: i for i, name in enumerate(names)}
        saved = policies[policies.run == run].set_index('policy').loc[names]
        compare(run + '_all_policy_point_estimates', saved[METRICS], point)
        cohort = cohorts[cohorts.run == run].iloc[0]
        compare(run + '_cohort_counts', [cohort['rows'], cohort['patients'], cohort['positive_labels']], [len(labels), patients, labels.sum()])
        maximum_fold_error = 0.0
        tied_boundary_instances = 0
        for model, groups_by_fold in grouping.items():
            _, _, error = allocate(groups_by_fold, labels, weights)
            maximum_fold_error = max(maximum_fold_error, error)
            for take, group, count in groups_by_fold:
                mass = np.bincount(group, minlength=count)
                before = np.cumsum(mass) - mass
                for quota in Q * len(take):
                    tied_boundary_instances += int(np.any((before < quota) & (before + mass > quota) & (mass > 1)))
        assert maximum_fold_error < 1e-08
        replicates = CFG['bootstrap']['replicates']
        boot = np.empty((replicates, len(names), len(METRICS)))
        rng = np.random.default_rng(CFG['bootstrap']['seed'] + run_index)
        for iteration in range(replicates):
            draw = rng.integers(patients, size=patients)
            multiplicity = np.bincount(draw, minlength=patients)
            _, boot[iteration] = evaluate(labels, multiplicity[patient_index], grouping, scores)
        lows = np.empty_like(point)
        highs = np.empty_like(point)
        for j in range(len(names)):
            for k in range(len(METRICS)):
                lows[j, k], highs[j, k] = interval(boot[:, j, k])
        compare(run + '_all_policy_interval_lower_bounds', saved[[m + '_low' for m in METRICS]], lows)
        compare(run + '_all_policy_interval_upper_bounds', saved[[m + '_high' for m in METRICS]], highs)
        for _, row in paired[paired.run == run].iterrows():
            if pd.notna(row['capacity']):
                q = row['capacity']
                comparison = row['comparison'].removeprefix('primary_raw_minus_')
                left, right, metric = (f'primary_raw_capacity_{q:.2f}', f'{comparison}_capacity_{q:.2f}', 1)
                observed = row['additional_positive_records_per_1000']
            else:
                threshold = row['threshold']
                metric = METRICS.index(row['comparison'].removeprefix('base_platt_minus_base_raw_'))
                left, right = (f'base_platt_threshold_{threshold:.2f}', f'base_raw_threshold_{threshold:.2f}')
                observed = row['estimate']
            i, j = (policy_index[left], policy_index[right])
            bounds = interval(boot[:, i, metric] - boot[:, j, metric])
            compare(run + '_' + row['comparison'] + '_paired', [observed, row['low'], row['high']], [point[i, metric] - point[j, metric], *bounds])
        scenario = scenarios[scenarios.run == run]
        indices = scenario.policy.map(policy_index).to_numpy()
        factor = (scenario.a * scenario.u * scenario.b).to_numpy()
        harm = scenario.h.to_numpy()
        observed = scenario[['process_improvements_per_1000', 'process_improvements_low', 'process_improvements_high', 'adverse_process_consequences_per_1000', 'adverse_process_low', 'adverse_process_high']]
        expected = np.column_stack((point[indices, 1] * factor, lows[indices, 1] * factor, highs[indices, 1] * factor, point[indices, 0] * harm, lows[indices, 0] * harm, highs[indices, 0] * harm))
        compare(run + '_all_scenario_values_and_intervals', observed, expected)
        scenario_parameters = scenario[['a', 'u', 'b', 'h']].drop_duplicates()
        assert len(scenario_parameters) == 144 and len(scenario) == len(names) * 144
        for _, row in actionability[actionability.run == run].iterrows():
            q = row['capacity']
            i = policy_index[f'primary_raw_capacity_{q:.2f}']
            j = policy_index[f'{row.comparator}_capacity_{q:.2f}']
            ratio = point[j, 1] / point[i, 1]
            compare(run + '_actionability_ratio', row['equal_benefit_actionability_ratio_model_to_comparator'], ratio)
        example = examples[examples.run == run]
        example_indices = example.policy.map(policy_index).to_numpy()
        cfg = CFG['illustrative_example']
        factor = cfg['a'] * cfg['u'] * cfg['b']
        compare(run + '_illustrative_example', example[['process_improvements_per_1000', 'process_improvements_low', 'process_improvements_high', 'adverse_process_consequences_per_1000']], np.column_stack((point[example_indices, 1] * factor, lows[example_indices, 1] * factor, highs[example_indices, 1] * factor, point[example_indices, 0] * cfg['h'])))
        records.append({'run': run, 'analysis_unit': cohort['unit'], 'rows': int(len(labels)), 'patients': patients, 'policies': len(names), 'bootstrap_replicates': replicates, 'actual_boundary_tie_instances_checked': tied_boundary_instances, 'maximum_fold_capacity_error': maximum_fold_error})
        print(json.dumps({'verified_run': run, 'policies': len(names), 'bootstrap_replicates': replicates}), flush=True)
    output = {'date': CFG['date'], 'passed': True, 'method': 'Independent pandas score grouping and full-group clipped-quota allocation, with complete independent patient-cluster bootstrap; main implementation was not imported.', 'synthetic_ties': 'Two-fold expected captured-positive and quota counts match hand calculations.', 'verified': ['cohort counts', 'within-fold capacities and actual boundary ties', 'all policy point estimates', 'capacity-matched random reference', 'all 2000-resample intervals', 'paired contrasts', 'all scenario products and intervals', 'actionability break-even ratios', 'illustrative example'], 'runs': records, 'checks': CHECKS, 'findings': [], 'interpretation': 'The outputs estimate retrospective record selection and conditional process scenarios, not prevented MACE or observed clinical effects.', 'privacy': 'Only aggregate verification results written; no patient rows or identifiers.'}
    (ROOT / 'verification.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
