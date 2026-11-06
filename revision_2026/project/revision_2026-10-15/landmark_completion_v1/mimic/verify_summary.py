"""Independent sklearn point and primary paired-bootstrap replay from saved OOF."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
ROOT = Path(__file__).resolve().parent

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main():
    output = ROOT / 'evidence/summary_independent_replay.json'
    assert not output.exists(), 'Refuse to overwrite replay'
    config = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8')))
    summary = _release_resolve_config(json.loads((ROOT / 'results/window_summary.json').read_text(encoding='utf-8')))
    assert summary['passed'] and summary['execution_complete']
    inputs = [identity(ROOT / 'protocol.json'), identity(ROOT / 'results/window_summary.json'), identity(Path(__file__))]
    for record in summary['input_identities'] + summary['outputs']:
        assert identity(record['path']) == record
    metric_path = ROOT / 'results/window_metrics.csv'
    pair_path = ROOT / 'results/window_paired_comparisons.csv'
    metrics, pairs = (pd.read_csv(metric_path), pd.read_csv(pair_path))
    inputs += [identity(metric_path), identity(pair_path)]
    data = {}
    for spec in config['runs']:
        path = ROOT / 'private/runs' / spec['name'] / 'predictions.npz'
        inputs.append(identity(path))
        with np.load(path, allow_pickle=False) as f:
            data[spec['name']] = {key: f[key].copy() for key in f.files}
        reference = 'MIMIC_exclude_AST_HDL' if spec['drop_features'] else 'MIMIC_original'
        ref_path = Path(config['reference_branch']) / 'private/runs' / reference / 'predictions.npz'
        inputs.append(identity(ref_path))
        with np.load(ref_path, allow_pickle=False) as f:
            for key in ('y', 'groups', 'fold', 'original'):
                assert np.array_equal(data[spec['name']][key], f[key])
            data[spec['name']]['full_cache_reference'] = f['primary_raw'].copy()
    with np.load(ROOT / 'private/prepared/evaluation_subset.npz', allow_pickle=False) as f:
        subset = f['still_hospitalized_after_48h'].copy()
    masks = {'all_original_9946': np.ones(len(subset), dtype=bool), 'common_stay_over_48h_9427': subset}
    point = {}
    maximum_point_error = 0.0
    for row in metrics.itertuples():
        mask = masks[row.population]
        run = data[row.run]
        y, score = (run['y'][mask], run[row.model][mask])
        key = (row.population, row.run, row.model, row.metric)
        if row.metric == 'auc':
            value = roc_auc_score(y, score)
        elif row.metric == 'average_precision':
            value = average_precision_score(y, score)
        elif row.metric == 'brier':
            value = brier_score_loss(y, score)
        elif row.metric == 'e_over_o':
            value = score.sum() / y.sum()
        else:
            raise ValueError('Unknown metric')
        error = abs(float(value) - row.estimate)
        maximum_point_error = max(maximum_point_error, error)
        assert error < 2e-10
        assert row.admissions == len(y) and row.patients == len(np.unique(run['groups'][mask]))
        point[key] = float(value)
    for row in pairs.itertuples():
        metric = row.metric.removeprefix('delta_')
        a = point[row.population, row.first_run, row.first_model, metric]
        b = point[row.population, row.second_run, row.second_model, metric]
        assert abs(a - b - row.estimate) < 2e-10
    selected = pairs[pairs.comparison.isin(['24h_minus_0h', '48h_minus_24h', '48h_minus_0h', 'exclude_columns_minus_retained'])]
    audit_rows = []
    for population, mask in masks.items():
        first = next(iter(data.values()))
        y = first['y'][mask]
        _, group_index = np.unique(first['groups'][mask], return_inverse=True)
        patients = int(group_index.max() + 1)
        rng = np.random.default_rng(config['bootstrap_seed'])
        samples = {name: np.zeros((2000, 2)) for name in data}
        draw_hash = hashlib.sha256()
        for iteration in range(2000):
            multiplicity = rng.multinomial(patients, np.full(patients, 1 / patients))
            draw_hash.update(np.asarray(multiplicity, dtype='<i8').tobytes())
            weights = multiplicity[group_index]
            for name, run in data.items():
                score = run['primary_raw'][mask]
                samples[name][iteration] = [roc_auc_score(y, score, sample_weight=weights), brier_score_loss(y, score, sample_weight=weights)]
            if (iteration + 1) % 500 == 0:
                print(f'{population}: independently replayed {iteration + 1}/2000 primary draws', flush=True)
        expected_meta = next((r for r in summary['patient_bootstrap'] if r['population'] == population))
        for row in selected[selected.population == population].itertuples():
            j = 0 if row.metric == 'delta_auc' else 1
            difference = samples[row.first_run][:, j] - samples[row.second_run][:, j]
            lo, hi = np.quantile(difference, [0.025, 0.975])
            error = max(abs(lo - row.ci_low), abs(hi - row.ci_high))
            assert error < 2e-10
            assert row.valid_replicates == 2000
            audit_rows.append({'population': population, 'comparison': row.comparison, 'first_run': row.first_run, 'second_run': row.second_run, 'metric': row.metric, 'independent_sklearn_ci_low': float(lo), 'independent_sklearn_ci_high': float(hi), 'maximum_interval_error': float(error)})
    for record in inputs:
        assert identity(record['path']) == record
    report = {'passed': True, 'all_summary_source_and_output_hashes_verified': True, 'input_identities': inputs, 'metric_point_rows_replayed': len(metrics), 'paired_point_rows_replayed': len(pairs), 'maximum_point_error': maximum_point_error, 'primary_window_and_mapping_paired_intervals_replayed': audit_rows, 'bootstrap_engine': 'Independent sklearn roc_auc_score and brier_score_loss with patient multiplicity sample weights; no metric_helpers import', 'bootstrap_replicates_per_population': 2000, 'no_training_performed': True, 'no_patient_rows_or_draws_exported': True, 'scope_limit': 'All metric and paired point estimates and all primary window/mapping paired intervals checked; other secondary intervals rely on the tested shared bootstrap engine.'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({'passed': True, 'primary_paired_intervals': len(audit_rows)}))
if __name__ == '__main__':
    main()
