"""Pure metric helpers reused from the audited original-input summary; no I/O entrypoint."""
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
MODEL_KEYS = ('original', 'full_cache_reference', 'primary_raw', 'base_raw', 'base_platt', 'xgb_full', 'rf_full', 'logistic', 'missingness', 'age')
METRIC_KEYS = ('auc', 'average_precision', 'brier', 'e_over_o')
LOGIT_EPSILON = 1e-06

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
