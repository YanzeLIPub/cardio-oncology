"""Offline review-capacity simulation using fixed patient-isolated predictions."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from itertools import product
import os, json, sys, platform, importlib.metadata
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
R = Path(__file__).resolve().parent
CFG = _release_resolve_config(json.loads((R / 'protocol.json').read_text(encoding='utf-8')))
AGG = R / 'aggregate'
AGG.mkdir(exist_ok=True)
Q = np.array(CFG['capacity_fractions'])
T = np.array(CFG['numerical_thresholds'])

def rank_groups(p, fold):
    out = []
    for f in np.unique(fold):
        idx = np.flatnonzero(fold == f)
        idx = idx[np.argsort(-p[idx], kind='stable')]
        starts = np.r_[0, np.flatnonzero(np.diff(p[idx]) != 0) + 1]
        out.append((idx, starts))
    return out

def top_counts(parts, y, w):
    positive = np.zeros(len(Q))
    for idx, starts in parts:
        g = np.add.reduceat(w[idx], starts)
        gy = np.add.reduceat(w[idx] * y[idx], starts)
        total = g.sum()
        if total == 0:
            continue
        mass = np.cumsum(g)
        pos = np.cumsum(gy)
        target = Q * total
        j = np.searchsorted(mass, target, side='left')
        before = np.r_[0, mass[:-1]][j]
        before_y = np.r_[0, pos[:-1]][j]
        positive += before_y + (target - before) / g[j] * gy[j]
    return positive

def load_run(run):
    source = Path(CFG['source_directory']) / run
    with np.load(source / 'predictions.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ['y', 'groups', 'fold', 'primary_raw', 'age', 'base_raw', 'base_platt']}
    with np.load(source / 'patient_roles.npz', allow_pickle=False) as z:
        roles = z['roles'].copy()
        assert np.array_equal(data['groups'], z['groups']) and np.array_equal(data['y'], z['y'])
    assert np.array_equal(np.argmax(roles == 0, axis=0) + 1, data['fold'])
    groups, inverse = np.unique(data['groups'], return_inverse=True)
    for f in np.unique(data['fold']):
        assert not np.intersect1d(data['groups'][data['fold'] == f], data['groups'][data['fold'] != f]).size
    for k in ['primary_raw', 'age', 'base_raw', 'base_platt']:
        assert np.isfinite(data[k]).all() and ((data[k] >= 0) & (data[k] <= 1)).all()
    return (data, inverse, len(groups), source)

def definitions():
    d = []
    for model in CFG['capacity_models']:
        for q in Q:
            d.append({'policy': f'{model}_capacity_{q:.2f}', 'mode': 'capacity', 'model': model, 'setting': float(q)})
    for q in Q:
        d.append({'policy': f'random_capacity_{q:.2f}', 'mode': 'capacity', 'model': 'random', 'setting': float(q)})
    d.extend([{'policy': 'review_all', 'mode': 'default', 'model': 'all', 'setting': 1.0}, {'policy': 'review_none', 'mode': 'default', 'model': 'none', 'setting': 0.0}])
    for model in CFG['threshold_models']:
        for t in T:
            d.append({'policy': f'{model}_threshold_{t:.2f}', 'mode': 'threshold', 'model': model, 'setting': float(t)})
    return d
DEFS = definitions()
METRICS = ['selected_per_1000', 'selected_positive_per_1000', 'selected_negative_per_1000', 'positive_capture_percent', 'review_positive_percent']

def evaluate(y, w, parts, masks):
    n = w.sum()
    positives = np.dot(w, y)
    selected = []
    tp = []
    for model in CFG['capacity_models']:
        selected.extend(Q * n)
        tp.extend(top_counts(parts[model], y, w))
    selected.extend(Q * n)
    tp.extend(Q * positives)
    selected.extend([n, 0.0])
    tp.extend([positives, 0.0])
    for model in CFG['threshold_models']:
        selected.extend(masks[model] @ w)
        tp.extend(masks[model] @ (w * y))
    selected = np.array(selected)
    tp = np.array(tp)
    ppv = np.divide(100 * tp, selected, out=np.full_like(tp, np.nan), where=selected > 0)
    return np.stack([1000 * selected / n, 1000 * tp / n, 1000 * (selected - tp) / n, 100 * tp / positives, ppv], axis=1)

def limits(a):
    good = a[np.isfinite(a)]
    return tuple(np.quantile(good, [0.025, 0.975])) if len(good) else (None, None)

def main():
    metrics = []
    paired = []
    scenarios = []
    sensitivity = []
    cohorts = []
    example = []
    method_checks = []
    (R / 'environment.json').write_text(json.dumps({'python': platform.python_version(), 'executable': sys.executable, 'packages': {x: importlib.metadata.version(x) for x in ['numpy', 'pandas']}, 'fitting': 'No new fitting; existing outer-test predictions reused.', 'bootstrap_seed': CFG['bootstrap']['seed']}, indent=2), encoding='utf-8')
    for run_index, run in enumerate(CFG['runs']):
        data, inverse, patients, source = load_run(run)
        y = data['y'].astype(float)
        n = len(y)
        unit = 'admission' if run.startswith('MIMIC') else 'patient'
        cohorts.append({'run': run, 'unit': unit, 'rows': n, 'patients': patients, 'positive_labels': int(y.sum()), 'input_path': (source / 'predictions.npz').as_posix()})
        parts = {m: rank_groups(data[m], data['fold']) for m in CFG['capacity_models']}
        masks = {m: (data[m][None, :] >= T[:, None]).astype(float) for m in CFG['threshold_models']}
        point = evaluate(y, np.ones(n), parts, masks)
        boot = np.empty((CFG['bootstrap']['replicates'], len(DEFS), len(METRICS)))
        rng = np.random.default_rng(CFG['bootstrap']['seed'] + run_index)
        for i in range(len(boot)):
            mult = np.bincount(rng.integers(0, patients, size=patients), minlength=patients)
            boot[i] = evaluate(y, mult[inverse], parts, masks)
        expected = np.array([d['setting'] * 1000 for d in DEFS if d['mode'] == 'capacity'])
        assert np.allclose(point[:len(expected), 0], expected)
        assert np.allclose(point[:, 1] + point[:, 2], point[:, 0])
        assert np.all(point[:, 1] >= -1e-10) and np.all(point[:, 1] <= point[:, 0] + 1e-10)
        for j, d in enumerate(DEFS):
            row = {'run': run, 'unit': unit, **d}
            for k, m in enumerate(METRICS):
                low, high = limits(boot[:, j, k])
                row.update({m: float(point[j, k]) if np.isfinite(point[j, k]) else None, m + '_low': low, m + '_high': high})
            metrics.append(row)
            for a, u, b, h in product(*CFG['scenario_parameters'].values()):
                B = point[j, 1] * a * u * b
                H = point[j, 0] * h
                bl, bh = limits(boot[:, j, 1] * a * u * b)
                hl, hh = limits(boot[:, j, 0] * h)
                scenarios.append({'run': run, 'unit': unit, 'policy': d['policy'], 'a': a, 'u': u, 'b': b, 'h': h, 'process_improvements_per_1000': B, 'process_improvements_low': bl, 'process_improvements_high': bh, 'adverse_process_consequences_per_1000': H, 'adverse_process_low': hl, 'adverse_process_high': hh})
            if d['mode'] == 'capacity' and d['setting'] == 0.1:
                c = CFG['illustrative_example']
                factor = c['a'] * c['u'] * c['b']
                bl, bh = limits(boot[:, j, 1] * factor)
                example.append({**row, 'assumed_benefit_fraction_of_selected_positives': factor, 'process_improvements_per_1000': point[j, 1] * factor, 'process_improvements_low': bl, 'process_improvements_high': bh, 'adverse_process_consequences_per_1000': point[j, 0] * c['h']})
        idx = {d['policy']: i for i, d in enumerate(DEFS)}
        for q in Q:
            mi = idx[f'primary_raw_capacity_{q:.2f}']
            for ref in ['age', 'random']:
                ri = idx[f'{ref}_capacity_{q:.2f}']
                values = boot[:, mi, 1] - boot[:, ri, 1]
                lo, hi = limits(values)
                paired.append({'run': run, 'unit': unit, 'capacity': q, 'comparison': 'primary_raw_minus_' + ref, 'additional_positive_records_per_1000': point[mi, 1] - point[ri, 1], 'low': lo, 'high': hi})
                ratio = point[ri, 1] / point[mi, 1] if point[mi, 1] > 0 else None
                sensitivity.append({'run': run, 'unit': unit, 'capacity': q, 'comparator': ref, 'model_positive_records_per_1000': point[mi, 1], 'comparator_positive_records_per_1000': point[ri, 1], 'equal_benefit_actionability_ratio_model_to_comparator': ratio, 'interpretation': 'With equal recommendation uptake and process success, model advantage requires a_model/a_comparator above this ratio. Neither actionability is measured.'})
        for t in T:
            a = idx[f'base_platt_threshold_{t:.2f}']
            b = idx[f'base_raw_threshold_{t:.2f}']
            for k in [0, 1]:
                lo, hi = limits(boot[:, a, k] - boot[:, b, k])
                paired.append({'run': run, 'unit': unit, 'capacity': None, 'comparison': 'base_platt_minus_base_raw_' + METRICS[k], 'threshold': t, 'additional_positive_records_per_1000': None, 'estimate': point[a, k] - point[b, k], 'low': lo, 'high': hi})
        method_checks.append({'run': run, 'patient_roles_match': True, 'patient_overlap_across_test_folds': 0, 'capacity_preserved': True, 'positive_plus_negative_equals_selected': True, 'bootstrap_replicates': len(boot)})
        print(json.dumps({'run': run, 'rows': n, 'patients': patients, 'policies': len(DEFS), 'bootstrap_complete': len(boot)}), flush=True)
    for name, rows in [('policy_metrics', metrics), ('paired_comparisons', paired), ('effect_scenarios', scenarios), ('actionability_sensitivity', sensitivity), ('cohort_definitions', cohorts), ('illustrative_10percent_capacity', example)]:
        pd.DataFrame(rows).to_csv(AGG / (name + '.csv'), index=False)
    (R / 'run_summary.json').write_text(json.dumps({'completed': True, 'analysis': 'Offline record-review capacity and process-effect scenario simulation', 'runs': method_checks, 'policies_per_run': len(DEFS), 'scenario_rows': len(scenarios), 'new_model_training': False, 'observed_clinical_effect_estimated': False, 'original_submission_files_changed': False}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
