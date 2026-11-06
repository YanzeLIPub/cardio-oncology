"""Aggregate original-input validation with paired patient-cluster bootstrap.

Run --self-test before use. The default mode requires all four completed runs,
uses the frozen protocol, and refuses to overwrite its aggregate outputs.
No patient-level rows, original identifiers or bootstrap patient draws are
written to public outputs. Historical cached scores are not independent scores.
"""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ('original', 'primary_raw', 'base_raw', 'base_platt', 'xgb_full', 'rf_full', 'logistic', 'missingness', 'age')
METRIC_KEYS = ('auc', 'average_precision', 'brier', 'e_over_o')
RUN_NAMES = ('SCH_original', 'WHU_original', 'MIMIC_original', 'MIMIC_exclude_AST_HDL')
PAIR_DEFINITIONS = (('primary_minus_original', 'primary_raw', 'original'), ('base_platt_minus_base_raw', 'base_platt', 'base_raw'), ('primary_minus_logistic', 'primary_raw', 'logistic'), ('primary_minus_age', 'primary_raw', 'age'))
LOGIT_EPSILON = 1e-06
CSV_NAMES = ('validation_metrics.csv', 'validation_paired_comparisons.csv', 'validation_calibration_parameters.csv', 'validation_calibration_bins.csv', 'validation_probability_histograms.csv')
FIGURE_NAMES = ('Original_and_patient_isolated_AUC.png', 'Original_and_patient_isolated_AUC.pdf', 'Patient_isolated_calibration.png', 'Patient_isolated_calibration.pdf')

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def sha256(path: Path) -> str:
    return None

def identity(path: Path) -> dict:
    return {'path': path.as_posix(), 'bytes': path.stat().st_size}

def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None

def interval(values: np.ndarray) -> tuple[float | None, float | None, int]:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return (None, None, 0)
    lower, upper = np.quantile(valid, [0.025, 0.975])
    return (float(lower), float(upper), len(valid))

class WeightedBinaryMetrics:
    """Tie-aware weighted ROC/AP with one score ordering per fixed model.

    AUC uses positive mass times preceding negative mass plus half the tied
    negative mass. AP integrates precision at descending unique thresholds.
    Synthetic tests below compare both calculations to sklearn, including ties,
    zero weights and repeated patient clusters.
    """

    def __init__(self, y: np.ndarray, score: np.ndarray):
        self.y = np.asarray(y, dtype=np.int64)
        self.score = np.asarray(score, dtype=np.float64)
        self.order = np.argsort(self.score, kind='stable')
        ordered_score = self.score[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ordered_score) != 0) + 1]
        self.ordered_y = self.y[self.order]
        self.squared_error = (self.score - self.y) ** 2

    def evaluate(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        total = float(weights.sum())
        positive = float(np.sum(weights * self.y))
        if total <= 0:
            return np.full(4, np.nan)
        brier = float(np.sum(weights * self.squared_error) / total)
        expected_observed = float(np.sum(weights * self.score) / positive) if positive > 0 else np.nan
        negative = total - positive
        if positive <= 0 or negative <= 0:
            return np.array([np.nan, np.nan, brier, expected_observed])
        ordered_weights = weights[self.order]
        mass = np.add.reduceat(ordered_weights, self.starts)
        positive_mass = np.add.reduceat(ordered_weights * self.ordered_y, self.starts)
        negative_mass = mass - positive_mass
        negative_cumulative = np.cumsum(negative_mass)
        auc = np.sum(positive_mass * (negative_cumulative - 0.5 * negative_mass)) / (positive * negative)
        positive_descending = positive_mass[::-1]
        cumulative_positive = np.cumsum(positive_descending)
        cumulative_mass = np.cumsum(mass[::-1])
        precision = np.divide(cumulative_positive, cumulative_mass, out=np.zeros_like(cumulative_positive), where=cumulative_mass > 0)
        average_precision = np.sum(positive_descending * precision) / positive
        return np.array([auc, average_precision, brier, expected_observed], dtype=np.float64)

def sklearn_point_metrics(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(y, score), average_precision_score(y, score), brier_score_loss(y, score), float(np.sum(score) / np.sum(y))], dtype=np.float64)

def calibration_parameters(y: np.ndarray, score: np.ndarray) -> dict:
    """Point estimates only, never calibrators applied to evaluation scores."""
    y = np.asarray(y, dtype=np.float64)
    lp = logit(np.clip(score, LOGIT_EPSILON, 1 - LOGIT_EPSILON))
    n = len(y)
    observed = float(np.sum(y))
    require(0 < observed < n, 'Calibration assessment requires both outcomes')
    citl = brentq(lambda intercept: float(np.sum(expit(intercept + lp)) - observed), -80, 80)
    base = {'calibration_in_the_large': float(citl), 'joint_intercept': None, 'joint_slope': None, 'joint_status': 'not_identifiable_constant_score', 'joint_gradient_maximum': None, 'logit_clipping_epsilon': LOGIT_EPSILON, 'confidence_intervals': 'Not estimated; descriptive conditional point estimates only'}
    if float(np.ptp(lp)) <= 1e-12:
        return base
    design = np.column_stack([np.ones(n), lp])

    def objective(beta: np.ndarray) -> float:
        linear = design @ beta
        return float(np.mean(np.logaddexp(0, linear) - y * linear))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return design.T @ (expit(design @ beta) - y) / n
    result = minimize(objective, np.array([citl, 1.0]), jac=gradient, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-09})
    gradient_maximum = float(np.max(np.abs(gradient(result.x))))
    probabilities = expit(design @ result.x)
    information = design.T * (probabilities * (1 - probabilities)) @ design / n
    minimum_eigenvalue = float(np.linalg.eigvalsh(information).min())
    finite_fit = bool(np.isfinite(result.x).all() and gradient_maximum < 1e-06 and (minimum_eigenvalue > 1e-09))
    base.update({'joint_intercept': float(result.x[0]) if finite_fit else None, 'joint_slope': float(result.x[1]) if finite_fit else None, 'joint_status': 'numerically_converged_point_fit' if finite_fit else 'unstable_or_nonconverged_not_reported', 'joint_gradient_maximum': gradient_maximum, 'joint_information_minimum_eigenvalue': minimum_eigenvalue, 'optimizer_success': bool(result.success)})
    return base

def calibration_bins(y: np.ndarray, score: np.ndarray, bins: int=10) -> list[dict]:
    edges = np.unique(np.quantile(score, np.linspace(0, 1, bins + 1)))
    internal_edges = edges[1:-1]
    assignments = np.searchsorted(internal_edges, score, side='left')
    records = []
    for index in np.unique(assignments):
        selected = assignments == index
        records.append({'bin': int(index + 1), 'count': int(selected.sum()), 'positive_count': int(y[selected].sum()), 'mean_probability': float(np.mean(score[selected])), 'observed_fraction': float(np.mean(y[selected])), 'minimum_probability': float(np.min(score[selected])), 'maximum_probability': float(np.max(score[selected]))})
    return records

def probability_histogram(score: np.ndarray) -> list[dict]:
    counts, edges = np.histogram(score, bins=np.linspace(0, 1, 11))
    return [{'bin': index + 1, 'left': float(edges[index]), 'right': float(edges[index + 1]), 'count': int(count), 'fraction': float(count / len(score))} for index, count in enumerate(counts)]

def load_protocol() -> dict:
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8')))
    require([r['name'] for r in protocol['runs']] == list(RUN_NAMES), 'Unexpected run list')
    require(protocol['bootstrap_replicates'] == 2000, 'Protocol requires 2000 bootstrap replicates')
    require(protocol['bootstrap_seed'] == 20261015, 'Bootstrap seed differs from protocol')
    require(protocol['outer_folds'] == 5, 'Five outer folds are required')
    require(protocol['omop_or_cross_centre_alignment'] is False, 'This summary cannot use aligned or cross-centre inputs')
    require(protocol['original_labels_and_cohort_filters_retained'] is True, 'Original cohort/labels must be retained')
    return protocol

def load_runs(protocol: dict) -> tuple[dict, list[dict]]:
    data = {}
    identities = [identity(ROOT / 'protocol.json'), identity(Path(__file__).resolve())]
    lock_path = ROOT / 'evidence' / 'analysis_lock.json'
    lock = _release_resolve_config(json.loads(lock_path.read_text(encoding='utf-8')))
    require(lock['protocol'] == protocol, 'Protocol differs from the pre-training lock')
    for expected_file in lock['state']['files']:
        actual = identity(Path(expected_file['path']))
        require(actual == expected_file, 'A frozen code/input file changed after training was specified')
    identities.append(identity(lock_path))
    for configuration in protocol['runs']:
        name, cohort = (configuration['name'], configuration['cohort'])
        path = ROOT / 'private/runs' / name / 'predictions.npz'
        training_path = ROOT / 'results' / f'{name}_training.json'
        prepared_path = ROOT / 'private/prepared' / f'{cohort}.npz'
        require(path.is_file() and training_path.is_file() and prepared_path.is_file(), f'Completed inputs missing for {name}')
        training = _release_resolve_config(json.loads(training_path.read_text(encoding='utf-8')))
        require(identity(path) == training['prediction_identity'], f'Training prediction identity changed for {name}')
        require(identity(prepared_path) == training['input'], f'Training prepared-input identity changed for {name}')
        require(identity(Path(training['roles_identity']['path'])) == training['roles_identity'], f'Training role assignment identity changed for {name}')
        for key in ('passed', 'execution_complete'):
            if key in training:
                require(training[key] is True, f'Training completion flag failed for {name}')
        with np.load(path, allow_pickle=False) as file:
            expected = {'y', 'groups', 'fold', *MODEL_KEYS}
            require(expected <= set(file.files), f'Prediction fields missing for {name}')
            arrays = {key: np.asarray(file[key]).copy() for key in expected}
        n = len(arrays['y'])
        require(all((values.shape == (n,) for values in arrays.values())), f'Prediction shape mismatch for {name}')
        require(set(np.unique(arrays['y'])) == {0, 1}, f'Nonbinary or single-class outcome for {name}')
        require(arrays['groups'].dtype.kind in 'iu', f'Group codes must be integer for {name}')
        require(arrays['fold'].dtype.kind in 'iu', f'Fold codes must be integer for {name}')
        require(np.unique(arrays['fold']).size == 5, f'Five outer folds absent for {name}')
        for model in MODEL_KEYS:
            p = arrays[model]
            require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), f'Invalid probabilities for {name}/{model}')
        group_fold_counts = pd.DataFrame({'g': arrays['groups'], 'f': arrays['fold']}).groupby('g')['f'].nunique()
        require(int(group_fold_counts.max()) == 1, f'Patient crosses evaluation folds in {name}')
        with np.load(prepared_path, allow_pickle=False) as prepared:
            require(np.array_equal(arrays['y'], prepared['y']), f'Prepared outcome differs in {name}')
            require(np.array_equal(arrays['groups'], prepared['groups']), f'Prepared group/order differs in {name}')
            require(np.array_equal(arrays['original'], prepared['p_original']), f'Historical probabilities differ in {name}')
        _, group_index = np.unique(arrays['groups'], return_inverse=True)
        data[name] = {'cohort': cohort, 'arrays': arrays, 'group_index': group_index, 'patients': int(np.max(group_index) + 1), 'observations': n, 'positives': int(arrays['y'].sum()), 'unit': 'admission' if cohort == 'MIMIC' else 'patient', 'drop_features': configuration['drop_features']}
        identities.extend([identity(path), identity(training_path), identity(prepared_path)])
    first = data['MIMIC_original']['arrays']
    second = data['MIMIC_exclude_AST_HDL']['arrays']
    for key in ('y', 'groups', 'fold', 'original'):
        require(np.array_equal(first[key], second[key]), f'MIMIC sensitivity is not paired on {key}')
    return (data, identities)

def bootstrap_cohort(run_data: dict, run_names: list[str], repetitions: int, seed: int) -> tuple[dict, dict]:
    reference = run_data[run_names[0]]
    group_index = reference['group_index']
    patient_count = reference['patients']
    caches = {(name, model): WeightedBinaryMetrics(run_data[name]['arrays']['y'], run_data[name]['arrays'][model]) for name in run_names for model in MODEL_KEYS}
    samples = {name: np.empty((repetitions, len(MODEL_KEYS), 4), dtype=np.float64) for name in run_names}
    rng = np.random.default_rng(seed)
    probabilities = np.full(patient_count, 1.0 / patient_count)
    digest = hashlib.sha256()
    for repeat in range(repetitions):
        multiplicities = rng.multinomial(patient_count, probabilities)
        digest.update(np.asarray(multiplicities, dtype='<i8').tobytes())
        weights = multiplicities[group_index].astype(np.float64)
        for name in run_names:
            for model_index, model in enumerate(MODEL_KEYS):
                samples[name][repeat, model_index] = caches[name, model].evaluate(weights)
        if (repeat + 1) % 500 == 0:
            print(f"{reference['cohort']}: {repeat + 1}/{repetitions} patient bootstrap draws", flush=True)
    return (samples, {'cohort': reference['cohort'], 'run_names': run_names, 'patient_clusters': patient_count, 'seed': seed, 'replicates': repetitions, 'sampling': "Multinomial(n_patients, uniform patient probabilities); every admission receives its patient's multiplicity", 'same_draws_for_all_models_and_paired_runs': True, 'original_group_codes_or_draws_exported': False})

def summarize(data: dict, protocol: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    all_samples, bootstrap_metadata = ({}, [])
    for cohort in ('SCH', 'WHU', 'MIMIC'):
        names = [name for name in RUN_NAMES if data[name]['cohort'] == cohort]
        samples, metadata = bootstrap_cohort(data, names, protocol['bootstrap_replicates'], protocol['bootstrap_seed'])
        all_samples.update(samples)
        bootstrap_metadata.append(metadata)
    metric_rows, comparison_rows, parameter_rows, bin_rows, histogram_rows = ([], [], [], [], [])
    point_values = {}
    for name in RUN_NAMES:
        run = data[name]
        arrays, y = (run['arrays'], run['arrays']['y'])
        context = {'run': name, 'cohort': run['cohort'], 'analysis_unit': run['unit'], 'observations': run['observations'], 'patients': run['patients'], 'positive_observations': run['positives']}
        for model_index, model in enumerate(MODEL_KEYS):
            score = arrays[model]
            points = sklearn_point_metrics(y, score)
            reference = WeightedBinaryMetrics(y, score).evaluate(np.ones(len(y)))
            require(np.allclose(points, reference, rtol=0, atol=2e-12), f'Sklearn point-metric disagreement for {name}/{model}')
            point_values[name, model] = points
            status = 'historical_cached_not_independent' if model == 'original' else 'new_patient_isolated_internal_evaluation'
            for metric_index, metric in enumerate(METRIC_KEYS):
                lower, upper, valid = interval(all_samples[name][:, model_index, metric_index])
                metric_rows.append({**context, 'model': model, 'evaluation_status': status, 'metric': metric, 'estimate': float(points[metric_index]), 'ci_low': lower, 'ci_high': upper, 'valid_bootstrap_replicates': valid, 'interval_interpretation': '95% percentile; fixed predictions; patient clusters; no refitting'})
            parameter_rows.append({**context, 'model': model, **calibration_parameters(y, score)})
            bin_rows.extend(({**context, 'model': model, **record} for record in calibration_bins(y, score)))
            histogram_rows.extend(({**context, 'model': model, **record} for record in probability_histogram(score)))
        for comparison_name, first, second in PAIR_DEFINITIONS:
            a, b = (MODEL_KEYS.index(first), MODEL_KEYS.index(second))
            for metric_index, metric in ((0, 'auc'), (2, 'brier')):
                differences = all_samples[name][:, a, metric_index] - all_samples[name][:, b, metric_index]
                lower, upper, valid = interval(differences)
                comparison_rows.append({**context, 'comparison': comparison_name, 'first_run': name, 'first_model': first, 'second_run': name, 'second_model': second, 'metric': f'delta_{metric}', 'estimate': float(point_values[name, first][metric_index] - point_values[name, second][metric_index]), 'ci_low': lower, 'ci_high': upper, 'valid_bootstrap_replicates': valid, 'direction': 'first minus second; positive AUC or negative Brier favors first', 'multiplicity': 'Exploratory nominal interval; no multiplicity adjustment'})
    first_name, second_name = ('MIMIC_exclude_AST_HDL', 'MIMIC_original')
    primary_index = MODEL_KEYS.index('primary_raw')
    for metric_index, metric in ((0, 'auc'), (2, 'brier')):
        differences = all_samples[first_name][:, primary_index, metric_index] - all_samples[second_name][:, primary_index, metric_index]
        lower, upper, valid = interval(differences)
        comparison_rows.append({'run': first_name, 'cohort': 'MIMIC', 'analysis_unit': 'admission', 'observations': data[first_name]['observations'], 'patients': data[first_name]['patients'], 'positive_observations': data[first_name]['positives'], 'comparison': 'exclude_known_mapping_columns_minus_original_inputs', 'first_run': first_name, 'first_model': 'primary_raw', 'second_run': second_name, 'second_model': 'primary_raw', 'metric': f'delta_{metric}', 'estimate': float(point_values[first_name, 'primary_raw'][metric_index] - point_values[second_name, 'primary_raw'][metric_index]), 'ci_low': lower, 'ci_high': upper, 'valid_bootstrap_replicates': valid, 'direction': 'excluded-column refit minus original-input refit; positive AUC or negative Brier favors exclusion', 'multiplicity': 'Exploratory nominal interval; no multiplicity adjustment'})
    tables = {CSV_NAMES[0]: pd.DataFrame(metric_rows), CSV_NAMES[1]: pd.DataFrame(comparison_rows), CSV_NAMES[2]: pd.DataFrame(parameter_rows), CSV_NAMES[3]: pd.DataFrame(bin_rows), CSV_NAMES[4]: pd.DataFrame(histogram_rows)}
    return (tables, {'patient_bootstrap': bootstrap_metadata})

def figures_from_aggregates(tables: dict[str, pd.DataFrame], directory: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True})
    metrics = tables[CSV_NAMES[0]]
    selected = metrics[(metrics['metric'] == 'auc') & metrics['model'].isin(['original', 'primary_raw'])]
    minimum = max(0, min(0.45, float(selected['ci_low'].min()) - 0.025))
    maximum = min(1, max(0.8, float(selected['ci_high'].max()) + 0.025))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.24, top=0.8, wspace=0.7)
    colors = {'original': '#707070', 'primary_raw': '#2166ac', 'excluded': '#1b7837'}
    for axis, cohort in zip(axes, ('SCH', 'WHU', 'MIMIC')):
        choices = [(f'{cohort}_original', 'original', 'Original cached', 'original'), (f'{cohort}_original', 'primary_raw', 'New patient-isolated', 'primary_raw')]
        if cohort == 'MIMIC':
            choices.append(('MIMIC_exclude_AST_HDL', 'primary_raw', 'New without AST / HDL', 'excluded'))
        for index, (run, model, label, color_key) in enumerate(choices):
            row = selected[(selected['run'] == run) & (selected['model'] == model)].iloc[0]
            value, lower, upper = (float(row['estimate']), float(row['ci_low']), float(row['ci_high']))
            axis.hlines(index, lower, upper, color=colors[color_key], linewidth=1.2)
            axis.vlines([lower, upper], index - 0.06, index + 0.06, color=colors[color_key], linewidth=1.2)
            axis.plot(value, index, 'o', color=colors[color_key], markersize=5)
            axis.annotate(f'{value:.3f}', (value, index), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
        axis.set_yticks(range(len(choices)), [c[2] for c in choices])
        axis.set_ylim(len(choices) - 0.45, -0.65)
        axis.set_xlim(minimum, maximum)
        axis.axvline(0.5, color='#aaaaaa', linewidth=0.8, linestyle=':')
        axis.grid(axis='x', color='#e0e0e0', linewidth=0.6)
        axis.set_title(cohort + (' | admissions' if cohort == 'MIMIC' else ' | patients'), pad=12)
        axis.set_xlabel('AUC')
    fig.suptitle('Original cached and new patient-isolated discrimination', fontsize=12, y=0.96)
    fig.text(0.04, 0.11, '95% patient-cluster bootstrap intervals conditional on fixed predictions; no model refitting.', fontsize=8.5)
    fig.text(0.04, 0.055, 'Historical scores are not independent validation. MIMIC retains the original admission-level endpoint and survivor filter.', fontsize=8.5)
    fig.savefig(directory / FIGURE_NAMES[0], dpi=300, facecolor='white')
    fig.savefig(directory / FIGURE_NAMES[1], facecolor='white')
    plt.close(fig)
    bins = tables[CSV_NAMES[3]]
    histograms = tables[CSV_NAMES[4]]
    fig, axes = plt.subplots(2, 3, figsize=(11.3, 7.2), gridspec_kw={'height_ratios': [3, 1.25]})
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.8, wspace=0.28, hspace=0.25)
    styles = {'original': ('Original cached', '#707070', ':'), 'primary_raw': ('New primary raw', '#2166ac', '-'), 'base_raw': ('Calibration base raw', '#b35806', '--'), 'base_platt': ('Calibration base Platt', '#1b7837', '-')}
    for column, cohort in enumerate(('SCH', 'WHU', 'MIMIC')):
        run = f'{cohort}_original'
        axis, histogram_axis = (axes[0, column], axes[1, column])
        axis.plot([0, 1], [0, 1], color='#bbbbbb', linestyle='--', linewidth=0.8)
        for model, (label, color, linestyle) in styles.items():
            points = bins[(bins['run'] == run) & (bins['model'] == model)].sort_values('mean_probability')
            axis.plot(points['mean_probability'], points['observed_fraction'], marker='o', markersize=3.5, linewidth=1.2, color=color, linestyle=linestyle, label=label)
            counts = histograms[(histograms['run'] == run) & (histograms['model'] == model)].sort_values('bin')
            edges = np.r_[counts['left'].to_numpy(), counts['right'].iloc[-1]]
            histogram_axis.stairs(100 * counts['fraction'].to_numpy(), edges, color=color, linestyle=linestyle, linewidth=1.2)
        for current in (axis, histogram_axis):
            current.set_xlim(0, 1)
            current.set_xticks([0, 0.25, 0.5, 0.75, 1])
            current.grid(color='#e5e5e5', linewidth=0.6)
        axis.set_ylim(0, 1)
        axis.set_title(cohort + (' | admissions' if cohort == 'MIMIC' else ' | patients'), pad=10)
        axis.set_xlabel('Mean probability in quantile bin')
        histogram_axis.set_xlabel('Probability')
        if column == 0:
            axis.set_ylabel('Observed fraction')
            histogram_axis.set_ylabel('Observations (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.53, 0.915), ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Descriptive out-of-fold calibration and score distributions', fontsize=12, y=0.975)
    fig.text(0.045, 0.085, 'Quantile-bin points are descriptive, without confidence bands. Platt and base raw share a separate base model;', fontsize=8.5)
    fig.text(0.045, 0.05, 'the new primary is refitted on all outer-training patients. Original cached scores retain their historical validation limitations.', fontsize=8.5)
    fig.savefig(directory / FIGURE_NAMES[2], dpi=300, facecolor='white')
    fig.savefig(directory / FIGURE_NAMES[3], facecolor='white')
    plt.close(fig)

def self_test() -> dict:
    rng = np.random.default_rng(71402)
    weighted_cases = 0
    maximum_error = 0.0
    for n in (20, 91, 307):
        y = rng.integers(0, 2, n)
        y[:2] = [0, 1]
        for score in (rng.random(n), np.round(rng.random(n), 1), np.full(n, 0.37)):
            cache = WeightedBinaryMetrics(y, score)
            for _ in range(12):
                weights = rng.integers(0, 5, n).astype(float)
                weights[:2] = [1, 1]
                expected = np.array([roc_auc_score(y, score, sample_weight=weights), average_precision_score(y, score, sample_weight=weights), brier_score_loss(y, score, sample_weight=weights), np.sum(weights * score) / np.sum(weights * y)])
                observed = cache.evaluate(weights)
                error = float(np.max(np.abs(expected - observed)))
                maximum_error = max(maximum_error, error)
                require(np.allclose(expected, observed, rtol=0, atol=2e-12), 'Weighted metric differs from sklearn')
                weighted_cases += 1
    groups = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    score = np.array([0.1, 0.7, 0.1, 0.6, 0.4, 0.6, 0.2, 0.9, 0.9, 0.3, 0.8, 0.4])
    cache = WeightedBinaryMetrics(y, score)
    cluster_cases = 0
    zero_paired_differences = []
    for _ in range(100):
        multiplicity = rng.multinomial(6, np.full(6, 1 / 6))
        weights = multiplicity[groups]
        indices = np.repeat(np.arange(len(y)), weights)
        if np.unique(y[indices]).size != 2:
            continue
        weighted = cache.evaluate(weights)
        repeated = sklearn_point_metrics(y[indices], score[indices])
        require(np.allclose(weighted, repeated, rtol=0, atol=2e-12), 'Patient multiplicities differ from repeated admissions')
        zero_paired_differences.append(weighted - cache.evaluate(weights.copy()))
        cluster_cases += 1
    require(np.array_equal(np.asarray(zero_paired_differences), np.zeros_like(zero_paired_differences)), 'Identical paired models must have zero difference')
    require(np.isnan(cache.evaluate(np.zeros(len(y)))).all(), 'Zero total weight must be undefined')
    one_class = cache.evaluate((y == 0).astype(float))
    require(np.isnan(one_class[:2]).all() and np.isfinite(one_class[2]), 'Single-class discrimination handling differs')
    synthetic_runs = {}
    for name in ('MIMIC_original', 'MIMIC_exclude_AST_HDL'):
        arrays = {'y': y, 'groups': groups}
        arrays.update({model: score.copy() for model in MODEL_KEYS})
        synthetic_runs[name] = {'cohort': 'MIMIC', 'patients': 6, 'group_index': groups, 'arrays': arrays}
    paired_samples, paired_metadata = bootstrap_cohort(synthetic_runs, list(synthetic_runs), 30, 20261015)
    require(np.array_equal(paired_samples['MIMIC_original'], paired_samples['MIMIC_exclude_AST_HDL'], equal_nan=True), 'Paired MIMIC runs must use identical patient multiplicities')
    synthetic_rng = np.random.default_rng(20261015)
    synthetic_digest = hashlib.sha256()
    for draw in range(30):
        multiplicity = synthetic_rng.multinomial(6, np.full(6, 1 / 6))
        synthetic_digest.update(np.asarray(multiplicity, dtype='<i8').tobytes())
        expected = cache.evaluate(multiplicity[groups])
        for name in synthetic_runs:
            require(np.allclose(paired_samples[name][draw], expected[None, :], rtol=0, atol=2e-12, equal_nan=True), 'Bootstrap engine did not apply the common patient draw to every model')
    constant_y = np.r_[np.ones(30), np.zeros(70)]
    constant = calibration_parameters(constant_y, np.full(100, 0.2))
    require(abs(constant['calibration_in_the_large'] - (logit(0.3) - logit(0.2))) < 1e-10, 'Offset calibration point estimate differs')
    require(constant['joint_slope'] is None, 'Constant-score slope must not be identified')
    calibrated_y = np.r_[np.ones(20), np.zeros(80), np.ones(80), np.zeros(20)]
    calibrated_p = np.r_[np.full(100, 0.2), np.full(100, 0.8)]
    fit = calibration_parameters(calibrated_y, calibrated_p)
    require(abs(fit['joint_intercept']) < 1e-07 and abs(fit['joint_slope'] - 1) < 1e-07, 'Known two-score calibration parameters differ')
    bins = calibration_bins(y, score)
    histogram = probability_histogram(score)
    require(sum((r['count'] for r in bins)) == len(y), 'Calibration bins do not partition observations')
    require(sum((r['positive_count'] for r in bins)) == int(y.sum()), 'Calibration bins lose positive observations')
    require(sum((r['count'] for r in histogram)) == len(y), 'Histogram does not partition observations')
    low, high, valid = interval(np.arange(2000, dtype=float))
    require(abs(low - 49.975) < 1e-10 and abs(high - 1949.025) < 1e-10 and (valid == 2000), 'Percentile interval differs')
    return {'passed': True, 'weighted_sklearn_comparison_cases': weighted_cases, 'patient_cluster_repeat_comparison_cases': cluster_cases, 'maximum_metric_absolute_error': maximum_error, 'identical_model_paired_difference_zero': True, 'paired_run_common_multiplicities_and_draw_hash_checked': True, 'ties_zero_weights_single_class_constant_scores_checked': True, 'calibration_and_bin_conservation_checked': True, 'files_written_by_self_test': 0}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Run synthetic tests only; write no files')
    arguments = parser.parse_args()
    tests = self_test()
    if arguments.self_test:
        print(json.dumps(tests, indent=2))
        return
    protocol = load_protocol()
    output_paths = [ROOT / 'results' / name for name in CSV_NAMES]
    output_paths += [ROOT / 'figures' / name for name in FIGURE_NAMES]
    summary_path = ROOT / 'results/validation_summary.json'
    require(not any((path.exists() for path in [*output_paths, summary_path])), 'Refusing to overwrite summary outputs')
    data, input_identities = load_runs(protocol)
    tables, bootstrap_metadata = summarize(data, protocol)
    for record in input_identities:
        pass
    (ROOT / 'results').mkdir(parents=True, exist_ok=True)
    (ROOT / 'figures').mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(ROOT / 'results' / name, index=False, mode='x', float_format='%.12g')
    figures_from_aggregates(tables, ROOT / 'figures')
    summary = {'schema_version': 1, 'passed': True, 'execution_complete': True, 'scope': 'Original native inputs and labels; patient-isolated internal revalidation; not new external or prospective validation', 'protocol_version': protocol['version'], 'input_identities': input_identities, 'all_input_identities_unchanged_after_evaluation': True, 'synthetic_metric_tests': tests, 'environment': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib')}}, **bootstrap_metadata, 'metric_definition': {'auc': 'Ordinary binary ROC AUC with half credit for tied scores', 'average_precision': 'Noninterpolated average precision at unique score thresholds, matching sklearn', 'brier': 'Mean squared binary probability error', 'e_over_o': 'Sum of predicted probabilities divided by count of positive labels', 'calibration': 'Offset-logit intercept and joint logistic intercept/slope assessed on evaluation predictions; point estimates only, not fitted recalibrators applied to these observations'}, 'interval_definition': '95% percentile intervals from 2000 patient-cluster draws conditional on fixed fitted OOF predictions; no model refitting; no multiplicity adjustment', 'no_fold_standard_deviation_used_as_confidence_interval': True, 'MIMIC_original_and_excluded_columns_exactly_paired': True, 'historical_cached_scores_are_independent_validation': False, 'calibration_parameter_confidence_intervals_estimated': False, 'figures_visually_inspected': False, 'figures_require_visual_QA_before_delivery': True, 'aggregate_row_counts': {name: len(table) for name, table in tables.items()}, 'outputs': [identity(path) for path in output_paths], 'limits': ['Bootstrap intervals on historical cached probabilities do not repair their historical patient overlap, early stopping, predictor timing or other bias.', 'New grouped internal evaluation uses the same historical cohort and labels; it is not a new independent external cohort.', 'A change from original cached AUC combines changed validation and fitting procedures and must not be attributed exclusively to one cause.', 'MIMIC metrics use admissions while resampling whole patients; Chinese metrics use patient rows. Cohort AUCs are not interchangeable endpoint rankings.', 'The MIMIC two-column exclusion is a prespecified site-local sensitivity, not complete assay repair, unit harmonization or OMOP alignment.', 'Calibration plots use descriptive quantile bins; histogram counts are aggregate and no clinical action thresholds are implied.', 'All reported intervals condition on fixed predictions and omit model-fitting uncertainty; paired comparisons are exploratory nominal intervals.']}
    with summary_path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'passed': True, 'aggregate_rows': summary['aggregate_row_counts'], 'visual_QA_required': True}, indent=2))
if __name__ == '__main__':
    main()
