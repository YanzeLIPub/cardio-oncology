"""Paired measurement-window metrics on fixed OOF predictions; no model fitting."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from metric_helpers import MODEL_KEYS, METRIC_KEYS, WeightedBinaryMetrics, bootstrap_cohort, calibration_bins, calibration_parameters, identity, interval, require, self_test, sklearn_point_metrics
ROOT = Path(__file__).resolve().parent
CSV_NAMES = ('window_metrics.csv', 'window_paired_comparisons.csv', 'window_calibration_parameters.csv', 'window_calibration_bins.csv')

def read(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8')))

def verify(record):
    require(identity(Path(record['path'])) == record, 'Evidence identity mismatch')
    return record

def load(config):
    lock_path = ROOT / 'evidence/analysis_lock.json'
    lock = read(lock_path)
    require(lock['protocol'] == config, 'Locked protocol mismatch')
    identities = [identity(ROOT / 'protocol.json'), identity(Path(__file__)), identity(_release_source_path(ROOT / 'metric_helpers.py')), identity(lock_path)]
    for record in lock['state']['files']:
        identities.append(verify(record))
    replay_path = ROOT / 'evidence/validation_replay_audit.json'
    replay = read(replay_path)
    require(replay['passed'] and len(replay['runs']) == 6, 'Six-run saved-model replay required')
    identities.append(identity(replay_path))
    old = Path(config['reference_branch'])
    references = {}
    for name in ('MIMIC_original', 'MIMIC_exclude_AST_HDL'):
        training_path = old / 'results' / (name + '_training.json')
        train = read(training_path)
        for key in ('input', 'roles_identity', 'prediction_identity'):
            identities.append(verify(train[key]))
        identities.append(identity(training_path))
        with np.load(train['prediction_identity']['path'], allow_pickle=False) as f:
            references[name] = {k: f[k].copy() for k in f.files}
    data = {}
    for spec in config['runs']:
        name = spec['name']
        training_path = ROOT / 'results' / (name + '_training.json')
        train = read(training_path)
        for key in ('input', 'roles_identity', 'prediction_identity'):
            identities.append(verify(train[key]))
        require(Path(train['input']['path']) == ROOT / 'private/prepared' / spec['matrix'], 'Wrong prepared matrix')
        identities.append(identity(training_path))
        with np.load(train['prediction_identity']['path'], allow_pickle=False) as f:
            arrays = {k: f[k].copy() for k in f.files}
        reference_name = 'MIMIC_exclude_AST_HDL' if spec['drop_features'] else 'MIMIC_original'
        reference = references[reference_name]
        for key in ('y', 'groups', 'fold', 'original'):
            require(np.array_equal(arrays[key], reference[key]), 'Reference is not paired')
        arrays['full_cache_reference'] = reference['primary_raw']
        with np.load(train['input']['path'], allow_pickle=False) as f:
            require(np.array_equal(arrays['y'], f['y']) and np.array_equal(arrays['groups'], f['groups']), 'Prepared order mismatch')
            require(np.array_equal(arrays['original'], f['p_original']), 'Original probability mismatch')
        for model in MODEL_KEYS:
            require(arrays[model].shape == arrays['y'].shape and np.isfinite(arrays[model]).all(), 'Invalid score shape or value')
            require(((arrays[model] >= 0) & (arrays[model] <= 1)).all(), 'Probability outside unit interval')
        data[name] = {'arrays': arrays, 'cohort': 'MIMIC', 'hours': spec['hours'], 'variant': 'exclude_AST_HDL' if spec['drop_features'] else 'original_43', 'reference_run': reference_name}
    first = next(iter(data.values()))['arrays']
    for run in data.values():
        for key in ('y', 'groups', 'fold', 'original'):
            require(np.array_equal(first[key], run['arrays'][key]), 'Window runs are not paired')
    subset_path = ROOT / 'private/prepared/evaluation_subset.npz'
    with np.load(subset_path, allow_pickle=False) as f:
        subset = f['still_hospitalized_after_48h'].copy()
    require(subset.dtype == bool and subset.shape == first['y'].shape and (subset.sum() == 9427), 'Common subset mismatch')
    identities.append(identity(subset_path))
    return (data, subset, identities)

def summarize(data, subset, config):
    names = [s['name'] for s in config['runs']]
    metric_rows, pair_rows, parameter_rows, bin_rows, metadata = ([], [], [], [], [])
    masks = {'all_original_9946': np.ones(len(subset), dtype=bool), 'common_stay_over_48h_9427': subset}
    for population, mask in masks.items():
        current = {}
        for name in names:
            source = data[name]
            arrays = {k: v[mask] for k, v in source['arrays'].items()}
            _, group_index = np.unique(arrays['groups'], return_inverse=True)
            current[name] = {**source, 'arrays': arrays, 'group_index': group_index, 'patients': int(group_index.max() + 1)}
        samples, meta = bootstrap_cohort(current, names, config['bootstrap_replicates'], config['bootstrap_seed'])
        meta['population'] = population
        metadata.append(meta)
        points = {}
        for name, run in current.items():
            y = run['arrays']['y']
            context = {'population': population, 'run': name, 'hours': run['hours'], 'variant': run['variant'], 'admissions': len(y), 'patients': run['patients'], 'positive_labels': int(y.sum()), 'reference_run': run['reference_run']}
            for i, model in enumerate(MODEL_KEYS):
                score = run['arrays'][model]
                point = sklearn_point_metrics(y, score)
                check = WeightedBinaryMetrics(y, score).evaluate(np.ones(len(y)))
                require(np.allclose(point, check, rtol=0, atol=2e-12), 'Metric point replay differs')
                points[name, model] = point
                status = 'historical_cached_not_patient_isolated' if model == 'original' else 'previous_patient_isolated_full_cache_reference' if model == 'full_cache_reference' else 'new_measurement_window_patient_isolated_classification'
                for j, metric in enumerate(METRIC_KEYS):
                    lo, hi, valid = interval(samples[name][:, i, j])
                    metric_rows.append({**context, 'model': model, 'evaluation_status': status, 'metric': metric, 'estimate': float(point[j]), 'ci_low': lo, 'ci_high': hi, 'valid_replicates': valid})
                parameter_rows.append({**context, 'model': model, **calibration_parameters(y, score)})
                bin_rows.extend(({**context, 'model': model, **item} for item in calibration_bins(y, score)))
        comparisons = []
        for suffix in ('original', 'exclude_AST_HDL'):
            for high, low in ((24, 0), (48, 24), (48, 0)):
                comparisons.append((f'{high}h_minus_{low}h', f'MIMIC_{high}h_{suffix}', 'primary_raw', f'MIMIC_{low}h_{suffix}', 'primary_raw'))
        for hour in (0, 24, 48):
            comparisons.append(('exclude_columns_minus_retained', f'MIMIC_{hour}h_exclude_AST_HDL', 'primary_raw', f'MIMIC_{hour}h_original', 'primary_raw'))
        for name in names:
            for comparison, a, b in (('base_platt_minus_base_raw', 'base_platt', 'base_raw'), ('primary_minus_logistic', 'primary_raw', 'logistic'), ('primary_minus_age', 'primary_raw', 'age'), ('window_minus_matching_full_cache_reference', 'primary_raw', 'full_cache_reference')):
                comparisons.append((comparison, name, a, name, b))
        for label, first, a, second, b in comparisons:
            for j, metric in ((0, 'auc'), (2, 'brier')):
                differences = samples[first][:, MODEL_KEYS.index(a), j] - samples[second][:, MODEL_KEYS.index(b), j]
                lo, hi, valid = interval(differences)
                pair_rows.append({'population': population, 'comparison': label, 'first_run': first, 'first_model': a, 'second_run': second, 'second_model': b, 'admissions': int(mask.sum()), 'patients': current[first]['patients'], 'positive_labels': int(current[first]['arrays']['y'].sum()), 'metric': 'delta_' + metric, 'estimate': float(points[first, a][j] - points[second, b][j]), 'ci_low': lo, 'ci_high': hi, 'valid_replicates': valid, 'direction': 'first minus second', 'interval': '2000 paired patient-cluster draws; fixed predictions; nominal 95% percentile; no refitting or multiplicity correction'})
    return ({CSV_NAMES[0]: pd.DataFrame(metric_rows), CSV_NAMES[1]: pd.DataFrame(pair_rows), CSV_NAMES[2]: pd.DataFrame(parameter_rows), CSV_NAMES[3]: pd.DataFrame(bin_rows)}, metadata)

def figures(tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    metrics = tables[CSV_NAMES[0]]
    selected = metrics[(metrics.model == 'primary_raw') & (metrics.metric == 'auc')]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.27, top=0.8, wspace=0.17)
    styles = [('original_43', 'Original 43 inputs', '#2166ac', -0.055), ('exclude_AST_HDL', 'Exclude AST / HDL', '#1b7837', 0.055)]
    for ax, population, title in zip(axes, ('all_original_9946', 'common_stay_over_48h_9427'), ('All original admissions (n = 9,946)', 'Common stay > 48h subset (n = 9,427)')):
        for variant, label, color, offset in styles:
            rows = selected[(selected.population == population) & (selected.variant == variant)].sort_values('hours')
            x = np.arange(3) + offset
            ax.plot(x, rows.estimate, '-o', color=color, label=label, markersize=4)
            ax.vlines(x, rows.ci_low, rows.ci_high, color=color, linewidth=1.1)
            for xx, lo, hi in zip(x, rows.ci_low, rows.ci_high):
                ax.hlines([lo, hi], xx - 0.025, xx + 0.025, color=color, linewidth=1.1)
        ax.set_xticks([0, 1, 2], ['0h', '24h', '48h'])
        ax.set_xlim(-0.25, 2.25)
        ax.set_ylim(0.6, 0.77)
        ax.grid(axis='y', color='#dddddd', linewidth=0.6)
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_xlabel('Hospital-admission measurement cutoff')
    axes[0].set_ylabel('Recorded-diagnosis AUC')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(0.52, 0.91), frameon=False)
    fig.suptitle('MIMIC measurement-window sensitivity', y=0.98, fontsize=13)
    fig.text(0.07, 0.14, 'Original labels retained. 95% patient-cluster CIs are conditional on fixed OOF predictions.', fontsize=8.5)
    fig.text(0.07, 0.095, 'CHARTTIME is a measurement proxy, not result availability or future-event warning.', fontsize=8.5)
    fig.text(0.07, 0.05, 'Common-subset selection uses future stay duration; it does not establish an event-free risk set.', fontsize=8.5)
    for ext in ('png', 'svg'):
        fig.savefig(ROOT / 'figures' / ('MIMIC_measurement_window_AUC.' + ext), dpi=300, facecolor='white')
    plt.close(fig)
    bins = tables[CSV_NAMES[3]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.27, top=0.77, wspace=0.25)
    for ax, hour in zip(axes, (0, 24, 48)):
        ax.plot([0, 1], [0, 1], '--', color='#aaaaaa', linewidth=0.8)
        for model, label, color, style in [('primary_raw', 'Full-training raw', '#2166ac', '-'), ('base_raw', 'Calibration-base raw', '#b35806', '--'), ('base_platt', 'Calibration-base Platt', '#1b7837', '-')]:
            points = bins[(bins.population == 'all_original_9946') & (bins.run == f'MIMIC_{hour}h_original') & (bins.model == model)].sort_values('mean_probability')
            ax.plot(points.mean_probability, points.observed_fraction, style, marker='o', markersize=3, color=color, label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{hour}h measurement cutoff', fontsize=10)
        ax.set_xlabel('Mean predicted probability')
        ax.grid(color='#e5e5e5', linewidth=0.5)
    axes[0].set_ylabel('Observed recorded-label fraction')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.52, 0.9), frameon=False, fontsize=8.5)
    fig.suptitle('Descriptive held-out calibration | original 43-input windows', y=0.98, fontsize=12)
    fig.text(0.06, 0.14, 'Original labels and all 9,946 admissions retained; patient-isolated outer-test predictions.', fontsize=8.5)
    fig.text(0.06, 0.095, 'Platt is fitted only to separate calibration patients and applies to the smaller base model, not the full model.', fontsize=8.5)
    fig.text(0.06, 0.05, 'Quantile-bin curves are descriptive without confidence bands; no prospective risk calibration is claimed.', fontsize=8.5)
    for ext in ('png', 'svg'):
        fig.savefig(ROOT / 'figures' / ('MIMIC_measurement_window_calibration.' + ext), dpi=300, facecolor='white')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print(json.dumps(tests, indent=2))
        return
    config = read(ROOT / 'protocol.json')
    require(config['bootstrap_replicates'] == 2000 and len(config['runs']) == 6, 'Unexpected protocol')
    output = ROOT / 'results/window_summary.json'
    require(not output.exists() and (not any(((ROOT / 'results' / name).exists() for name in CSV_NAMES))), 'Refuse to overwrite results')
    data, subset, identities = load(config)
    tables, bootstrap = summarize(data, subset, config)
    for record in identities:
        verify(record)
    for name, table in tables.items():
        table.to_csv(ROOT / 'results' / name, index=False, mode='x', float_format='%.12g')
    figures(tables)
    outputs = [ROOT / 'results' / name for name in CSV_NAMES]
    outputs += [ROOT / 'figures' / (name + '.' + ext) for name in ('MIMIC_measurement_window_AUC', 'MIMIC_measurement_window_calibration') for ext in ('png', 'svg')]
    summary = {'passed': True, 'execution_complete': True, 'scope': config['scope'], 'input_identities': identities, 'patient_bootstrap': bootstrap, 'metric_self_tests': tests, 'table_rows': {name: len(table) for name, table in tables.items()}, 'all_six_runs_and_references_paired_on_label_group_order_and_fold': True, 'interval_definition': '2000 common patient multiplicity draws per evaluation population; percentile 95% CI conditional on fixed OOF predictions; no model refitting or multiplicity correction', 'calibration_parameter_and_bin_uncertainty_intervals': False, 'subset_refitting_performed': False, 'subset_selection_uses_future_stay_duration': True, 'visual_QA_complete': False, 'outputs': [identity(p) for p in outputs], 'limits': ['Measurement-window classification only; no incident MACE or warning interval.', 'Original survivor cohort, diagnosis labels and uncorrected mapping remain in the retained-column primary analysis.', 'AST/HDL removal is a separately labelled sensitivity, not full terminology or unit repair.', 'Full-training raw and held-out base Platt use different fitted base models; only base raw versus base Platt isolates calibration.', 'The 48-hour common subset is selected using future stay duration and is not clinically event-free.', 'Window models reuse historical data and are not new external validation.']}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2, allow_nan=False)
    print(json.dumps({'passed': True, 'table_rows': summary['table_rows']}, indent=2))
if __name__ == '__main__':
    main()
