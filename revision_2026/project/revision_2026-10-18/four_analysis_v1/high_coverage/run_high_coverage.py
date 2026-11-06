"""Training-side laboratory coverage sensitivity; writes only in this branch.

Old scripts are read and hashed, never imported or executed. --prepare performs
no model fitting and reads no prediction values for performance evaluation.
"""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
MISSING = PROJECT / 'revision_2026-10-16/reviewer_priority_analyses_v1/missingness'
COHORTS = ('SCH', 'WHU', 'MIMIC')
VARIANTS = ('all_labs', 'coverage_80', 'coverage_50')
METRICS = ('auc', 'average_precision', 'brier', 'e_over_o')
THREADS = 2
MIMIC_NONLAB = ['Age', 'Sex_M', 'GDP per Capita', 'Avg. Employee Wage', 'No. of Hospitals', 'Hospital Beds', 'No. of Physicians']
MIMIC_WRONG = ['AST', 'HDL Cholesterol']

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return None

def identity(path):
    path = Path(path)
    return {'path': path.as_posix(), 'bytes': path.stat().st_size}

def read_json(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')

def verify_identity(expected):
    actual = identity(expected['path'])
    return actual

def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'logical_cpus': os.cpu_count(), 'max_threads': THREADS, 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'xgboost', 'joblib', 'scipy')}}

def literal_constants(path, wanted):
    values = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    assert set(values) == set(wanted)
    return values

def laboratory_inventory(cohort, names):
    if cohort == 'MIMIC':
        evidence = read_json(BASE / 'evidence/mimic_preparation.json')
        assert names == evidence['matrix']['feature_names_in_order']
        assert set(MIMIC_WRONG) == set(evidence['prespecified_site_local_sensitivity']['remove_columns'])
        excluded = {n: 'Demographic or regional non-laboratory input' for n in MIMIC_NONLAB}
        excluded.update({n: 'Whole slot excluded because original mapping contains verified incorrect assignments' for n in MIMIC_WRONG})
        expected = 34
    else:
        evidence = read_json(BASE / 'evidence/china_preparation.json')
        assert names == evidence['feature_names']
        constants = literal_constants(_release_source_path(BASE / 'prepare_china.py'), ('AGE', 'SEX', 'SUSPECT_FIELDS', 'REGIONAL_FIELDS'))
        excluded = {constants[k]: 'Demographic non-laboratory input' for k in ('AGE', 'SEX')}
        excluded.update({n: 'Diagnostic/department text coerced entirely to NaN in the original preparation' for n in constants['SUSPECT_FIELDS']})
        excluded.update({n: 'Regional non-laboratory input' for n in constants['REGIONAL_FIELDS']})
        expected = 94
    assert set(excluded) <= set(names)
    indices = np.array([i for i, n in enumerate(names) if n not in excluded], dtype=np.int64)
    assert len(indices) == expected
    rows = [{'original_index': i, 'name': n, 'eligible_laboratory_slot': n not in excluded, 'reason': excluded.get(n, 'Laboratory-labelled original input retained without claiming assay/unit certification')} for i, n in enumerate(names)]
    return (indices, rows)

def sources():
    paths = [BASE / p for p in ('protocol.json', 'run_validation.py', 'prepare_china.py', 'prepare_mimic.py', 'evidence/analysis_lock.json', 'evidence/china_preparation.json', 'evidence/mimic_preparation.json', 'results/validation_summary.json')]
    paths += [MISSING / p for p in ('protocol.json', 'run_missingness.py', 'analysis_lock.json', 'verification.json')]
    for c in COHORTS:
        paths += [BASE / 'private/prepared' / f'{c}.npz', BASE / 'private/runs' / f'{c}_original/patient_roles.npz', BASE / 'private/runs' / f'{c}_original/predictions.npz', BASE / 'results' / f'{c}_original_training.json']
    paths += [BASE / 'private/runs/MIMIC_exclude_AST_HDL' / p for p in ('patient_roles.npz', 'predictions.npz')]
    paths += [BASE / 'results/MIMIC_exclude_AST_HDL_training.json']
    return paths

def load_cohort(cohort, with_predictions=False):
    report = read_json(BASE / 'results' / f'{cohort}_original_training.json')
    for k in ('input', 'roles_identity', 'prediction_identity'):
        verify_identity(report[k])
    with np.load(BASE / 'private/prepared' / f'{cohort}.npz', allow_pickle=False) as z:
        X, y, groups, names = (z['X'].copy(), z['y'].copy(), z['groups'].copy(), z['feature_names'].tolist())
    with np.load(BASE / 'private/runs' / f'{cohort}_original/patient_roles.npz', allow_pickle=False) as z:
        assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        roles = z['roles'].copy()
    expected = read_json(ROOT / 'protocol.json')['expected_counts'][cohort]
    assert [len(y), len(np.unique(groups)), int(y.sum()), len(names)] == expected
    assert X.dtype == np.float32 and (not np.isinf(X).any())
    assert roles.shape == (5, len(y)) and np.all((roles == 0).sum(axis=0) == 1)
    fold_ids = np.zeros(len(y), dtype=np.int8)
    for fold, role in enumerate(roles, 1):
        sets = [set(groups[role == r]) for r in range(4)]
        assert all((not sets[a] & sets[b] for a in range(4) for b in range(a + 1, 4)))
        assert all((np.unique(y[role == r]).size == 2 for r in range(4)))
        fold_ids[role == 0] = fold
    references = {}
    if with_predictions:
        names_runs = {'original_full_reference': f'{cohort}_original'}
        if cohort == 'MIMIC':
            names_runs['original_exclude_AST_HDL_reference'] = 'MIMIC_exclude_AST_HDL'
        for label, run in names_runs.items():
            training = read_json(BASE / 'results' / f'{run}_training.json')
            verify_identity(training['prediction_identity'])
            with np.load(BASE / 'private/runs' / run / 'predictions.npz', allow_pickle=False) as z:
                assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
                assert np.array_equal(fold_ids, z['fold'])
                references[label] = z['primary_raw'].copy()
            with np.load(BASE / 'private/runs' / run / 'patient_roles.npz', allow_pickle=False) as z:
                assert np.array_equal(roles, z['roles'])
    eligible, inventory = laboratory_inventory(cohort, names)
    return (X, y, groups, names, roles, fold_ids, eligible, inventory, references)

def select_columns(Xfit, eligible, percent, cohort):
    assert percent in (0, 50, 80) and len(Xfit) > 0
    counts = np.isfinite(Xfit[:, eligible]).sum(axis=0)
    selected = eligible[counts * 100 >= percent * len(Xfit)]
    if cohort == 'MIMIC':
        selected = selected[np.isfinite(Xfit[:, selected]).any(axis=0)]
    return (selected, counts)

def fit_preprocessor(Xfit, eligible, percent, cohort):
    columns, counts = select_columns(Xfit, eligible, percent, cohort)
    if not len(columns):
        return None
    imputer = None if cohort != 'MIMIC' else SimpleImputer(strategy='median', keep_empty_features=True).fit(Xfit[:, columns])
    return {'columns': columns, 'zero_fill': cohort != 'MIMIC', 'imputer': imputer, 'coverage_percent': percent, 'fit_rows': len(Xfit), 'eligible': eligible, 'eligible_observed_counts': counts}

def transform(prep, X):
    values = X[:, prep['columns']].copy()
    return (np.nan_to_num(values, nan=0.0) if prep['zero_fill'] else prep['imputer'].transform(values)).astype(np.float32)

def selection_report(prep, names):
    return {'selected_original_indices': prep['columns'].tolist(), 'selected_features': [names[i] for i in prep['columns']], 'selected_count': len(prep['columns']), 'fit_rows': prep['fit_rows'], 'coverage_percent': prep['coverage_percent'], 'coverage_by_eligible_feature': [{'original_index': int(i), 'name': names[i], 'observed': int(count), 'fit_denominator': prep['fit_rows'], 'coverage': float(count / prep['fit_rows'])} for i, count in zip(prep['eligible'], prep['eligible_observed_counts'])]}

def preflight():
    protocol = read_json(ROOT / 'protocol.json')
    result = []
    for cohort in COHORTS:
        X, y, groups, names, roles, _, eligible, inventory, _ = load_cohort(cohort)
        folds = []
        for fold, role in enumerate(roles, 1):
            for variant in VARIANTS:
                percent = protocol['minimum_coverage_percent'][variant]
                sets, record = ({}, {'fold': fold, 'variant': variant})
                for label, mask in (('inner_fit', role == 2), ('full_outer_training', role != 0)):
                    prep = fit_preprocessor(X[mask], eligible, percent, cohort)
                    record[label] = selection_report(prep, names) if prep else {'selected_count': 0, 'not_estimable': True}
                    sets[label] = set(prep['columns']) if prep else set()
                record['features_differ_between_inner_and_final_fit'] = sets['inner_fit'] != sets['full_outer_training']
                record['inner_only_features'] = [names[i] for i in sorted(sets['inner_fit'] - sets['full_outer_training'])]
                record['final_only_features'] = [names[i] for i in sorted(sets['full_outer_training'] - sets['inner_fit'])]
                record['estimable'] = all(sets.values())
                folds.append(record)
        result.append({'cohort': cohort, 'rows': len(y), 'patients': len(np.unique(groups)), 'positive_labels': int(y.sum()), 'coverage_unit': 'admission row' if cohort == 'MIMIC' else 'patient row', 'nominal_laboratory_slots': len(eligible), 'feature_inventory': inventory, 'fold_selection': folds, 'performance_evaluated': False})
    return {'no_training_executed': True, 'no_prediction_metrics_evaluated': True, 'cohorts': result}

def freeze():
    assert ROOT == Path(_release_path('project', 'revision_2026-10-18/four_analysis_v1/high_coverage'))
    for item in read_json(BASE / 'evidence/analysis_lock.json')['state']['files']:
        verify_identity(item)
    protocol = read_json(ROOT / 'protocol.json')
    assert protocol['variants'] == list(VARIANTS) and protocol['max_threads'] == THREADS
    state = {'protocol': protocol, 'model_protocol': read_json(BASE / 'protocol.json'), 'protocol_identity': identity(ROOT / 'protocol.json'), 'script': identity(__file__), 'sources': [identity(p) for p in sources()], 'environment': environment()}
    lock = ROOT / 'analysis_lock.json'
    if lock.exists():
        assert read_json(lock)['state'] == state, 'Code, input, protocol or environment differs from frozen state'
    else:
        save_json(lock, {'frozen_before_training_at': now(), 'state': state})
    return state

def matrix(X, y=None):
    return xgb.DMatrix(X, label=y, nthread=THREADS, missing=np.nan)

def parameters(y, cohort, fold, config):
    p = config['xgb']
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': p['max_depth'], 'learning_rate': p['learning_rate'], 'subsample': p['subsample'], 'colsample_bytree': p['colsample_bytree'], 'scale_pos_weight': float((y == 0).sum() / (y == 1).sum()), 'seed': 42 if cohort != 'MIMIC' else fold, 'nthread': THREADS, 'verbosity': 0, 'tree_method': 'hist'}

def run_cohort(cohort, state):
    dest = ROOT / 'private' / cohort
    assert not dest.exists(), f'Refusing to overwrite {cohort}'
    dest.mkdir(parents=True)
    X, y, groups, names, roles, folds, eligible, inventory, references = load_cohort(cohort, True)
    config, protocol = (state['model_protocol'], state['protocol'])
    output = {v: np.full(len(y), np.nan, dtype=np.float64) for v in VARIANTS}
    reports, started = ([], time.monotonic())
    for fold, role in enumerate(roles, 1):
        inner, stop, full, test = [np.flatnonzero(mask) for mask in (role == 2, role == 3, role != 0, role == 0)]
        for variant in VARIANTS:
            folder = dest / variant / f'fold_{fold}'
            folder.mkdir(parents=True)
            percent = protocol['minimum_coverage_percent'][variant]
            ip = fit_preprocessor(X[inner], eligible, percent, cohort)
            fp = fit_preprocessor(X[full], eligible, percent, cohort)
            report = {'cohort': cohort, 'variant': variant, 'fold': fold, 'coverage_percent': percent, 'roles': {k: {'rows': len(ix), 'patients': len(np.unique(groups[ix])), 'positive_labels': int(y[ix].sum())} for k, ix in (('inner_fit', inner), ('inner_stop', stop), ('full_outer_training', full), ('outer_test', test))}}
            if ip is None or fp is None:
                report.update({'estimable': False, 'reason': 'No eligible laboratory in one or both fitting roles; fixed threshold not changed'})
                save_json(folder / 'fit_report.json', report)
                reports.append(report)
                continue
            print(f"{cohort} {variant} fold {fold}/5; inner/final features {len(ip['columns'])}/{len(fp['columns'])}", flush=True)
            history, inner_params = ({}, parameters(y[inner], cohort, fold, config))
            selected = xgb.train(inner_params, matrix(transform(ip, X[inner]), y[inner]), num_boost_round=config['xgb']['max_rounds'], evals=[(matrix(transform(ip, X[stop]), y[stop]), 'inner_stop')], early_stopping_rounds=config['xgb']['early_stopping_rounds'], verbose_eval=False, evals_result=history)
            rounds = int(selected.best_iteration) + 1
            assert 1 <= rounds <= 300
            selected.save_model(folder / 'inner_xgb.ubj')
            joblib.dump(ip, folder / 'inner_preprocessor.joblib', compress=3)
            full_params = parameters(y[full], cohort, fold, config)
            train_values, test_values = (transform(fp, X[full]), transform(fp, X[test]))
            booster = xgb.train(full_params, matrix(train_values, y[full]), num_boost_round=rounds, verbose_eval=False)
            weight = {0: 1, 1: int((y[full] == 0).sum() / (y[full] == 1).sum())} if cohort != 'MIMIC' else 'balanced'
            rf = RandomForestClassifier(**config['rf'], class_weight=weight, random_state=42 if cohort != 'MIMIC' else fold, n_jobs=THREADS)
            rf.fit(train_values, y[full])
            xp, rp = (booster.predict(matrix(test_values)), rf.predict_proba(test_values)[:, 1])
            output[variant][test] = 0.5 * xp + 0.5 * rp
            booster.save_model(folder / 'full_xgb.ubj')
            joblib.dump({'preprocessor': fp, 'rf': rf, 'feature_names': [names[i] for i in fp['columns']]}, folder / 'full_package.joblib', compress=3)
            report.update({'estimable': True, 'selected_rounds': rounds, 'inner_selection': selection_report(ip, names), 'final_selection': selection_report(fp, names), 'inner_only_features': [names[i] for i in sorted(set(ip['columns']) - set(fp['columns']))], 'final_only_features': [names[i] for i in sorted(set(fp['columns']) - set(ip['columns']))], 'inner_xgb_parameters': inner_params, 'full_xgb_parameters': full_params, 'rf_parameters': rf.get_params(), 'inner_stopping_history': history, 'outer_test_rows_with_no_observed_selected_lab': int((~np.isfinite(X[test][:, fp['columns']]).any(axis=1)).sum()), 'identities': [identity(folder / p) for p in ('inner_xgb.ubj', 'inner_preprocessor.joblib', 'full_xgb.ubj', 'full_package.joblib')]})
            save_json(folder / 'fit_report.json', report)
            reports.append(report)
    estimable = {v: bool(np.isfinite(output[v]).all()) for v in VARIANTS}
    assert all((((p[np.isfinite(p)] >= 0) & (p[np.isfinite(p)] <= 1)).all() for p in output.values()))
    np.savez_compressed(dest / 'predictions.npz', y=y, groups=groups, fold=folds, **output, **references)
    report = {'execution_complete': True, 'cohort': cohort, 'rows': len(y), 'patients': len(np.unique(groups)), 'positive_labels': int(y.sum()), 'analysis_unit': 'admission' if cohort == 'MIMIC' else 'patient', 'estimable_variants': estimable, 'feature_inventory': inventory, 'fold_reports': reports, 'seconds': time.monotonic() - started, 'prediction_identity': identity(dest / 'predictions.npz')}
    save_json(ROOT / 'results' / f'{cohort}_training.json', report)
    print(json.dumps({'completed': cohort, 'estimable': estimable, 'seconds': report['seconds']}), flush=True)

class WeightedMetrics:
    """Fixed-score tied-rank metrics checked against sklearn in self_test."""

    def __init__(self, y, p):
        self.y, self.p = (np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.float64))
        self.order = np.argsort(self.p, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.p[self.order]) != 0) + 1]
        self.ordered_y = self.y[self.order]
        self.error = (self.p - self.y) ** 2

    def evaluate(self, w):
        total, pos_total = (w.sum(), np.sum(w * self.y))
        if not 0 < pos_total < total:
            return np.full(4, np.nan)
        ordered = w[self.order]
        mass = np.add.reduceat(ordered, self.starts)
        pos = np.add.reduceat(ordered * self.ordered_y, self.starts)
        neg = mass - pos
        auc = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (pos_total * (total - pos_total))
        den = np.cumsum(mass[::-1])
        precision = np.divide(np.cumsum(pos[::-1]), den, out=np.zeros(len(pos)), where=den > 0)
        return np.array([auc, np.sum(pos[::-1] * precision) / pos_total, np.sum(w * self.error) / total, np.sum(w * self.p) / pos_total])

def sklearn_metrics(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    return np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w), np.sum(w * p) / np.sum(w * y)])

def self_test():
    X = np.array([[0, np.nan, np.nan, 1], [1, np.nan, np.nan, 1], [2, 7, np.nan, 1], [3, 8, np.nan, np.nan], [4, 9, np.nan, 1]], dtype=np.float32)
    eligible = np.arange(4)
    for cohort in COHORTS:
        assert select_columns(X, eligible, 80, cohort)[0].tolist() == [0, 3]
        assert select_columns(X, eligible, 50, cohort)[0].tolist() == [0, 1, 3]
        assert select_columns(X, eligible, 0, cohort)[0].tolist() == ([0, 1, 3] if cohort == 'MIMIC' else [0, 1, 2, 3])
        assert fit_preprocessor(np.full((3, 4), np.nan), eligible, 80, cohort) is None
        prep = fit_preprocessor(X[:2], eligible, 80, cohort)
        assert prep['columns'].tolist() == [0, 3]
        test = np.full((2, 4), 999, dtype=np.float32)
        transform(prep, test)
        assert prep['columns'].tolist() == [0, 3]
        assert transform(fit_preprocessor(X, eligible, 50, cohort), X)[0, 0] == 0
    assert transform(fit_preprocessor(X, eligible, 50, 'MIMIC'), X)[0, 1] == 8
    assert transform(fit_preprocessor(X, eligible, 50, 'SCH'), X)[0, 1] == 0
    rng, cases, largest = (np.random.default_rng(20261018), 0, 0.0)
    for n in (20, 91, 307):
        y = rng.integers(0, 2, n)
        y[:2] = [0, 1]
        for p in (rng.random(n), np.round(rng.random(n), 1), np.full(n, 0.37)):
            cache = WeightedMetrics(y, p)
            for _ in range(20):
                w = rng.integers(0, 5, n).astype(float)
                w[:2] = 1
                error = float(np.max(np.abs(cache.evaluate(w) - sklearn_metrics(y, p, w))))
                assert error < 2e-12
                largest, cases = (max(largest, error), cases + 1)
    groups = np.repeat(np.arange(10), [2, 1, 3, 1, 2, 2, 1, 3, 2, 1])
    y, p = (np.arange(len(groups)) % 2, np.round(rng.random(len(groups)), 1))
    for _ in range(50):
        w = rng.multinomial(10, np.ones(10) / 10)[groups]
        ix = np.repeat(np.arange(len(y)), w)
        assert np.allclose(WeightedMetrics(y, p).evaluate(w), sklearn_metrics(y[ix], p[ix]), atol=2e-12, rtol=0)
    return {'passed': True, 'weighted_sklearn_cases': cases, 'cluster_repeat_cases': 50, 'maximum_metric_error': largest, 'inclusive_thresholds_true_zeros_empty_sets_train_only_medians_and_test_isolation_checked': True, 'no_model_training_in_self_tests': True}

def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def summarize(state):
    rows, paired, draws, unavailable = ([], [], [], [])
    protocol = state['protocol']
    for cohort in COHORTS:
        report = read_json(ROOT / 'results' / f'{cohort}_training.json')
        verify_identity(report['prediction_identity'])
        with np.load(ROOT / 'private' / cohort / 'predictions.npz', allow_pickle=False) as z:
            y, groups = (z['y'].copy(), z['groups'].copy())
            models = [v for v in VARIANTS if report['estimable_variants'][v]] + ['original_full_reference']
            if cohort == 'MIMIC':
                models += ['original_exclude_AST_HDL_reference']
            predictions = {m: z[m].copy() for m in models}
        unavailable += [{'cohort': cohort, 'variant': v, 'reason': 'One or more folds not estimable; no partial-cohort metric'} for v in VARIANTS if not report['estimable_variants'][v]]
        _, inverse = np.unique(groups, return_inverse=True)
        n = int(inverse.max()) + 1
        context = {k: report[k] for k in ('cohort', 'rows', 'patients', 'positive_labels', 'analysis_unit')}
        caches = [WeightedMetrics(y, predictions[m]) for m in models]
        points = np.array([sklearn_metrics(y, predictions[m]) for m in models])
        assert np.allclose(points, [m.evaluate(np.ones(len(y))) for m in caches], atol=2e-12, rtol=0)
        samples = np.empty((protocol['bootstrap_replicates'], len(models), len(METRICS)))
        rng, digest = (np.random.default_rng(protocol['bootstrap_seed']), hashlib.sha256())
        for b in range(len(samples)):
            multiplicity = rng.multinomial(n, np.full(n, 1 / n))
            digest.update(np.asarray(multiplicity, dtype='<i8').tobytes())
            samples[b] = [m.evaluate(multiplicity[inverse]) for m in caches]
        assert np.isfinite(samples).all()
        for mi, model in enumerate(models):
            for ki, metric in enumerate(METRICS):
                low, high = np.quantile(samples[:, mi, ki], [0.025, 0.975])
                rows.append({**context, 'model': model, 'metric': metric, 'estimate': float(points[mi, ki]), 'ci_low': float(low), 'ci_high': float(high), 'bootstrap_replicates': len(samples)})
        comparisons = [protocol['primary_comparison'], protocol['secondary_comparison'], *protocol['context_comparisons']]
        if cohort == 'MIMIC':
            comparisons += [protocol['mimic_additional_context_comparison']]
        for first, second in comparisons:
            if first not in models or second not in models:
                continue
            a, b = (models.index(first), models.index(second))
            for ki, metric in enumerate(METRICS):
                low, high = np.quantile(samples[:, a, ki] - samples[:, b, ki], [0.025, 0.975])
                paired.append({**context, 'first': first, 'second': second, 'metric': 'delta_' + metric, 'estimate': float(points[a, ki] - points[b, ki]), 'ci_low': float(low), 'ci_high': float(high), 'bootstrap_replicates': len(samples)})
        old = next((r for r in read_json(BASE / 'results/validation_summary.json')['patient_bootstrap'] if r['cohort'] == cohort))
        draws.append({'cohort': cohort, 'same_as_original_validation': True})
        print(f'{cohort}: {len(samples)} paired patient-cluster draws complete', flush=True)
    write_csv(ROOT / 'results/high_coverage_metrics.csv', rows)
    write_csv(ROOT / 'results/high_coverage_paired_comparisons.csv', paired)
    save_json(ROOT / 'results/summary.json', {'execution_complete': True, 'protocol': protocol, 'metrics': rows, 'paired_comparisons': paired, 'not_estimable': unavailable, 'patient_bootstrap': draws, 'synthetic_tests': self_test(), 'limits': protocol['remaining_limits']})

def verify(state):
    replay = []
    for cohort in COHORTS:
        X, y, groups, names, roles, folds, eligible, _, refs = load_cohort(cohort, True)
        report = read_json(ROOT / 'results' / f'{cohort}_training.json')
        verify_identity(report['prediction_identity'])
        with np.load(ROOT / 'private' / cohort / 'predictions.npz', allow_pickle=False) as z:
            saved = {k: z[k].copy() for k in z.files}
        for key, target in (('y', y), ('groups', groups), ('fold', folds), *refs.items()):
            assert np.array_equal(saved[key], target)
        for fit in report['fold_reports']:
            fold, variant = (fit['fold'], fit['variant'])
            role = roles[fold - 1]
            test, full, inner = [np.flatnonzero(mask) for mask in (role == 0, role != 0, role == 2)]
            pct = state['protocol']['minimum_coverage_percent'][variant]
            expected_inner, expected_full = (fit_preprocessor(X[inner], eligible, pct, cohort), fit_preprocessor(X[full], eligible, pct, cohort))
            if not fit['estimable']:
                assert expected_inner is None or expected_full is None
                assert np.isnan(saved[variant][test]).all()
                continue
            assert selection_report(expected_inner, names) == fit['inner_selection']
            assert selection_report(expected_full, names) == fit['final_selection']
            for item in fit['identities']:
                verify_identity(item)
            folder = ROOT / 'private' / cohort / variant / f'fold_{fold}'
            package = joblib.load(folder / 'full_package.joblib')
            prep = package['preprocessor']
            assert np.array_equal(prep['columns'], expected_full['columns'])
            assert np.array_equal(transform(prep, X[test]), transform(expected_full, X[test]))
            booster = xgb.Booster(params={'nthread': THREADS})
            booster.load_model(folder / 'full_xgb.ubj')
            package['rf'].n_jobs = THREADS
            values = transform(prep, X[test])
            predicted = 0.5 * booster.predict(matrix(values)) + 0.5 * package['rf'].predict_proba(values)[:, 1]
            error = float(np.max(np.abs(predicted - saved[variant][test])))
            assert error < 1e-10
            replay.append({'cohort': cohort, 'variant': variant, 'fold': fold, 'maximum_replay_error': error})
    for expected in state['sources']:
        verify_identity(expected)
    summary = read_json(ROOT / 'results/summary.json')
    assert summary['execution_complete']
    save_json(ROOT / 'verification.json', {'passed': True, 'checked_at': now(), 'all_sources_unchanged': True, 'all_original_labels_and_patient_roles_preserved': True, 'fold_replay': replay, 'selection_recomputed_without_outcomes_or_test_values': True, 'summary_identity': identity(ROOT / 'results/summary.json'), 'analysis_lock': identity(ROOT / 'analysis_lock.json'), 'metrics': identity(ROOT / 'results/high_coverage_metrics.csv'), 'paired': identity(ROOT / 'results/high_coverage_paired_comparisons.csv'), 'environment': environment(), 'synthetic_tests': self_test()})
    print(json.dumps({'passed': True, 'replayed_outer_models': len(replay), 'verification': identity(ROOT / 'verification.json')}))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', choices=(*COHORTS, 'all'))
    mode.add_argument('--summarize', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.prepare:
        tests, feasibility = (self_test(), preflight())
        state = freeze()
        save_json(ROOT / 'feasibility.json', {'self_tests': tests, 'environment': environment(), **feasibility})
        print(json.dumps({'prepared': True, 'training_executed': False, 'all_45_variant_folds_estimable': all((r['estimable'] for c in feasibility['cohorts'] for r in c['fold_selection'])), 'analysis_lock': identity(ROOT / 'analysis_lock.json')}))
        return
    assert (ROOT / 'analysis_lock.json').exists(), 'Run --prepare before training or summaries'
    state = freeze()
    if args.run:
        assert (ROOT / 'training_authorization.json').exists(), 'Wait for root release after analysis 1 completes'
        authorization = read_json(ROOT / 'training_authorization.json')
        assert authorization['root_release_received'] and authorization['analysis_1_summary_and_replay_complete']
        for c in COHORTS if args.run == 'all' else [args.run]:
            run_cohort(c, state)
    if args.summarize:
        summarize(state)
    if args.verify:
        verify(state)
if __name__ == '__main__':
    main()
