"""Reproducible regional ablations and grouping stress tests in a new directory."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2] / 'revision_2026-10-15/original_v5_independent_validation_v1'
THREADS = 2
COHORTS = ('SCH', 'WHU')
METRICS = ('auc', 'average_precision', 'brier')

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def read(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def check(item):
    pass

def freeze():
    paths = [ROOT / 'protocol.json', ROOT / 'assignment_audit.json', _release_source_path(ROOT / 'audit_regional.py'), BASE / 'protocol.json', _release_source_path(BASE / 'prepare_china.py'), _release_source_path(BASE / 'run_validation.py'), BASE / 'evidence/china_preparation.json', BASE / 'results/validation_metrics.csv']
    for source in read(ROOT / 'assignment_audit.json')['sources']:
        check(source)
        paths.append(Path(source['path']))
    for cohort in COHORTS:
        paths += [BASE / f'private/prepared/{cohort}.npz', ROOT / f'private/{cohort}_grouping.npz', BASE / f'private/runs/{cohort}_original/patient_roles.npz', BASE / f'private/runs/{cohort}_original/predictions.npz']
        for fold in range(1, 6):
            paths += [BASE / f'private/runs/{cohort}_original/fold_{fold}/{name}' for name in ('full_xgb.ubj', 'full_package.joblib')]
    state = {'protocol': read(ROOT / 'protocol.json'), 'model_protocol': read(BASE / 'protocol.json'), 'sources': [identity(p) for p in sorted(set(paths))], 'script': identity(__file__), 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'threads': THREADS, 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib', 'scipy')}}}
    lock = ROOT / 'analysis_lock.json'
    if lock.exists():
        assert read(lock)['state'] == state
    else:
        save(lock, {'frozen_before_training_at': datetime.now(timezone.utc).isoformat(), 'state': state})
    return state

def regional_fields():
    for node in ast.parse(_release_source_path(BASE / 'prepare_china.py').read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'REGIONAL_FIELDS' for t in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError('Missing regional fields')

def load(cohort):
    with np.load(BASE / f'private/prepared/{cohort}.npz', allow_pickle=False) as z:
        X, y, groups, names = (z['X'].copy(), z['y'].copy(), z['groups'].copy(), z['feature_names'].tolist())
    assert not np.isinf(X).any() and np.array_equal(groups, np.arange(len(y)))
    with np.load(BASE / f'private/runs/{cohort}_original/patient_roles.npz', allow_pickle=False) as z:
        assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        roles = z['roles'].copy()
    with np.load(BASE / f'private/runs/{cohort}_original/predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        baseline, folds = (z['primary_raw'].copy(), z['fold'].copy())
    assert [len(y), int(y.sum()), len(names)] == read(ROOT / 'protocol.json')['expected_counts'][cohort]
    assert np.all((roles == 0).sum(axis=0) == 1)
    assert all((np.array_equal(r == 0, folds == f) for f, r in enumerate(roles, 1)))
    return (X, y, groups, names, roles, baseline)

def values(X, columns):
    return np.nan_to_num(X[:, columns], nan=0.0).astype(np.float32)

def dm(X, y=None):
    return xgb.DMatrix(X, label=y, missing=np.nan, nthread=THREADS)

def parameters(y, config):
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'max_depth': config['xgb']['max_depth'], 'learning_rate': config['xgb']['learning_rate'], 'subsample': config['xgb']['subsample'], 'colsample_bytree': config['xgb']['colsample_bytree'], 'scale_pos_weight': float((y == 0).sum() / (y == 1).sum()), 'seed': 42, 'nthread': THREADS, 'verbosity': 0}

def fit_fold(X, y, groups, names, roles, columns, folder, config):
    folder.mkdir(parents=True)
    test, full, inner, stop = (np.flatnonzero(roles == 0), np.flatnonzero(roles != 0), np.flatnonzero(roles == 2), np.flatnonzero(roles == 3))
    assert all((len(np.unique(y[ix])) == 2 for ix in (test, full, inner, stop)))
    assert not set(groups[test]) & set(groups[full])
    history = {}
    inner_params = parameters(y[inner], config)
    inner_model = xgb.train(inner_params, dm(values(X[inner], columns), y[inner]), num_boost_round=config['xgb']['max_rounds'], evals=[(dm(values(X[stop], columns), y[stop]), 'inner_stop')], early_stopping_rounds=config['xgb']['early_stopping_rounds'], evals_result=history, verbose_eval=False)
    rounds = int(inner_model.best_iteration) + 1
    assert rounds == int(np.argmax(history['inner_stop']['auc'])) + 1
    inner_model.save_model(folder / 'inner_xgb.ubj')
    full_params = parameters(y[full], config)
    model = xgb.train(full_params, dm(values(X[full], columns), y[full]), num_boost_round=rounds, verbose_eval=False)
    weight = {0: 1, 1: int((y[full] == 0).sum() / (y[full] == 1).sum())}
    rf = RandomForestClassifier(**config['rf'], class_weight=weight, random_state=42, n_jobs=THREADS)
    rf.fit(values(X[full], columns), y[full])
    xp = model.predict(dm(values(X[test], columns)))
    rp = rf.predict_proba(values(X[test], columns))[:, 1]
    model.save_model(folder / 'full_xgb.ubj')
    joblib.dump({'rf': rf, 'columns': columns, 'zero_fill': True, 'feature_names': [names[i] for i in columns]}, folder / 'full_package.joblib', compress=3)
    report = {'selected_rounds': rounds, 'inner_history': history, 'inner_xgb_parameters': inner_params, 'full_xgb_parameters': full_params, 'rf_parameters': rf.get_params(), 'input_column_indices': columns.tolist(), 'feature_names': [names[i] for i in columns], 'role_counts': {str(r): {'patients': int((roles == r).sum()), 'positive_labels': int(y[roles == r].sum())} for r in range(4)}, 'all_test_patient_intersections_with_training_zero': True, 'model_identities': [identity(folder / n) for n in ('inner_xgb.ubj', 'full_xgb.ubj', 'full_package.joblib')]}
    save(folder / 'fit_report.json', report)
    return (test, 0.5 * xp + 0.5 * rp, report)

def replay_original(cohort, X, roles, baseline):
    errors = []
    for fold, role in enumerate(roles, 1):
        folder = BASE / f'private/runs/{cohort}_original/fold_{fold}'
        package = joblib.load(folder / 'full_package.joblib')
        assert package['preprocessor']['zero_fill']
        test = np.flatnonzero(role == 0)
        v = values(X[test], package['preprocessor']['columns'])
        model = xgb.Booster(params={'nthread': THREADS})
        model.load_model(folder / 'full_xgb.ubj')
        package['rf'].n_jobs = THREADS
        p = 0.5 * model.predict(dm(v)) + 0.5 * package['rf'].predict_proba(v)[:, 1]
        error = float(np.max(np.abs(p - baseline[test])))
        assert error < 1e-10
        errors.append(error)
    return errors

def stress_roles(y, clusters, config):
    roles = np.full((5, len(y)), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(splitter.split(np.zeros((len(y), 1)), y, clusters), 1):
        base, cal = train_test_split(train, stratify=y[train], test_size=config['calibration_fraction_of_outer_train_patients'], random_state=config['calibration_seed_base'] + fold)
        base, cal = (np.sort(base), np.sort(cal))
        inner, stop = train_test_split(base, stratify=y[base], test_size=config['inner_validation_fraction_of_base_fit_patients'], random_state=config['inner_seed_base'] + fold)
        inner, stop = (np.sort(inner), np.sort(stop))
        assert not set(clusters[train]) & set(clusters[test])
        for r, ix in enumerate((test, cal, inner, stop)):
            assert len(np.unique(y[ix])) == 2
            roles[fold - 1, ix] = r
    assert (roles >= 0).all() and np.all((roles == 0).sum(axis=0) == 1)
    return roles

def run(cohort, state):
    X, y, groups, names, roles, baseline = load(cohort)
    regional = regional_fields()
    assert len(regional) == 5 and set(regional) <= set(names)
    columns = {'no_regional': np.array([i for i, name in enumerate(names) if name not in regional]), 'regional_only': np.array([names.index(f) for f in regional])}
    assert len(columns['no_regional']) == 99
    dest = ROOT / f'private/{cohort}'
    assert not dest.exists()
    dest.mkdir()
    baseline_error = replay_original(cohort, X, roles, baseline)
    outputs, reports = ({}, [])
    for variant in state['protocol']['primary_variants']:
        pred = np.full(len(y), np.nan)
        for fold, role in enumerate(roles, 1):
            print(f'{cohort} {variant} fold {fold}/5', flush=True)
            test, prediction, report = fit_fold(X, y, groups, names, role, columns[variant], dest / variant / f'fold_{fold}', state['model_protocol'])
            pred[test] = prediction
            reports.append({'variant': variant, 'fold': fold, **report})
        assert np.isfinite(pred).all()
        outputs[variant] = pred
    np.savez_compressed(dest / 'predictions.npz', y=y, groups=groups, roles=roles, baseline_ensemble=baseline, **outputs)
    with np.load(ROOT / f'private/{cohort}_grouping.npz', allow_pickle=False) as z:
        assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        grouping = z['strict_city_group' if cohort == 'SCH' else 'numeric_signature'].copy()
    selected = np.flatnonzero(grouping >= 0)
    Xs, ys, gs, clusters = (X[selected], y[selected], groups[selected], grouping[selected])
    expected = state['protocol']['stress_tests'][cohort]
    assert [len(ys), int(ys.sum()), len(np.unique(clusters))] == [expected[k] for k in ('patients', 'positive_labels', 'groups')]
    sr = stress_roles(ys, clusters, state['model_protocol'])
    np.savez_compressed(dest / 'stress_roles.npz', selected_index=selected, y=ys, groups=gs, clusters=clusters, roles=sr)
    pred = np.full(len(ys), np.nan)
    stress_reports = []
    for fold, role in enumerate(sr, 1):
        print(f"{cohort} {expected['name']} fold {fold}/5", flush=True)
        test, prediction, report = fit_fold(Xs, ys, gs, names, role, np.arange(len(names)), dest / 'stress' / f'fold_{fold}', state['model_protocol'])
        pred[test] = prediction
        assert not set(clusters[role == 0]) & set(clusters[role != 0])
        stress_reports.append({'fold': fold, 'train_test_cluster_overlap': 0, 'test_clusters': len(np.unique(clusters[test])), **report})
    assert np.isfinite(pred).all()
    np.savez_compressed(dest / 'stress_predictions.npz', y=ys, groups=gs, clusters=clusters, baseline_ensemble=baseline[selected], stress_ensemble=pred)
    save(ROOT / f'results/{cohort}_training.json', {'cohort': cohort, 'execution_complete': True, 'patients': len(y), 'positive_labels': int(y.sum()), 'feature_names': names, 'original_baseline_maximum_replay_error': max(baseline_error), 'fold_reports': reports, 'stress_definition': expected, 'stress_reports': stress_reports, 'analysis_lock': identity(ROOT / 'analysis_lock.json'), 'predictions': identity(dest / 'predictions.npz'), 'stress_predictions': identity(dest / 'stress_predictions.npz')})

class WeightedMetrics:
    """Fixed-score tie-aware metric cache, independently tested against sklearn."""

    def __init__(self, y, score):
        self.y, self.p = (np.asarray(y), np.asarray(score))
        self.order = np.argsort(self.p, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.p[self.order]) != 0) + 1]

    def evaluate(self, w):
        total, positive = (w.sum(), np.sum(w * self.y))
        if not 0 < positive < total:
            return np.full(3, np.nan)
        mass = np.add.reduceat(w[self.order], self.starts)
        pos = np.add.reduceat((w * self.y)[self.order], self.starts)
        neg = mass - pos
        auc = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (positive * (total - positive))
        precision = np.divide(np.cumsum(pos[::-1]), np.cumsum(mass[::-1]), out=np.zeros_like(pos, dtype=float), where=np.cumsum(mass[::-1]) > 0)
        ap = np.sum(pos[::-1] * precision) / positive
        return np.array([auc, ap, np.sum(w * (self.p - self.y) ** 2) / total])

def metrics(y, p, w=None):
    return np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)])

def self_test():
    rng, count = (np.random.default_rng(71716), 0)
    for n in (20, 91, 307):
        y = rng.integers(0, 2, n)
        y[:2] = [0, 1]
        for p in (rng.random(n), np.round(rng.random(n), 1), np.full(n, 0.37)):
            cached = WeightedMetrics(y, p)
            for _ in range(15):
                w = rng.integers(0, 5, n).astype(float)
                w[:2] = 1
                assert np.allclose(cached.evaluate(w), metrics(y, p, w), atol=2e-12, rtol=0)
                count += 1
    X = np.array([[0.0, np.nan, 1.0], [4.0, 3.0, np.nan]], dtype=np.float32)
    assert np.array_equal(values(X, np.array([0, 1])), np.array([[0, 0], [4, 3]], dtype=np.float32))
    return {'passed': True, 'weighted_sklearn_cases': count, 'true_zero_preserved': True}

def summarize(state):
    point_rows, paired_rows, details = ([], [], [])
    tests = self_test()
    for cohort in COHORTS:
        for analysis, file_name, cluster_keys, models in (('original_patient_roles', 'predictions.npz', ('groups',), state['protocol']['primary_models']), (state['protocol']['stress_tests'][cohort]['name'], 'stress_predictions.npz', ('groups', 'clusters'), ['baseline_ensemble', 'stress_ensemble'])):
            with np.load(ROOT / f'private/{cohort}/{file_name}', allow_pickle=False) as z:
                y = z['y'].copy()
                predictions = {m: z[m].copy() for m in models}
                clusters = {k: z[k].copy() for k in cluster_keys}
            points = np.array([metrics(y, predictions[m]) for m in models])
            caches = [WeightedMetrics(y, predictions[m]) for m in models]
            for cluster_key, group in clusters.items():
                _, inverse = np.unique(group, return_inverse=True)
                n = int(inverse.max()) + 1
                rng, h = (np.random.default_rng(state['protocol']['bootstrap_seed']), hashlib.sha256())
                draws = np.empty((2000, len(models), 3))
                for b in range(2000):
                    multiplicity = rng.multinomial(n, np.full(n, 1 / n))
                    h.update(np.asarray(multiplicity, dtype='<i8').tobytes())
                    draws[b] = [c.evaluate(multiplicity[inverse].astype(float)) for c in caches]
                valid = np.isfinite(draws).all(axis=(1, 2))
                assert valid.sum() >= 1900
                draws = draws[valid]
                context = {'cohort': cohort, 'analysis': analysis, 'patients': len(y), 'positive_labels': int(y.sum()), 'bootstrap_unit': 'patient' if cluster_key == 'groups' else 'outer_group', 'bootstrap_clusters': n, 'valid_bootstrap_replicates': int(valid.sum())}
                for mi, model in enumerate(models):
                    for ki, metric in enumerate(METRICS):
                        lo, hi = np.quantile(draws[:, mi, ki], [0.025, 0.975])
                        point_rows.append({**context, 'model': model, 'metric': metric, 'estimate': float(points[mi, ki]), 'ci_low': float(lo), 'ci_high': float(hi)})
                pairs = state['protocol']['primary_comparisons'] if analysis == 'original_patient_roles' else [['stress_ensemble', 'baseline_ensemble']]
                for a, b in pairs:
                    ai, bi = (models.index(a), models.index(b))
                    for ki, metric in enumerate(METRICS):
                        lo, hi = np.quantile(draws[:, ai, ki] - draws[:, bi, ki], [0.025, 0.975])
                        paired_rows.append({**context, 'first': a, 'second': b, 'metric': 'delta_' + metric, 'estimate': float(points[ai, ki] - points[bi, ki]), 'ci_low': float(lo), 'ci_high': float(hi)})
                details.append({**context})
                print(f"{cohort} {analysis} {context['bootstrap_unit']}: 2000 draws complete", flush=True)
    pd.DataFrame(point_rows).to_csv(ROOT / 'results/regional_metrics.csv', index=False, mode='x', float_format='%.12g')
    pd.DataFrame(paired_rows).to_csv(ROOT / 'results/regional_paired_comparisons.csv', index=False, mode='x', float_format='%.12g')
    save(ROOT / 'results/summary.json', {'execution_complete': True, 'analysis_lock': identity(ROOT / 'analysis_lock.json'), 'protocol': state['protocol'], 'self_test': tests, 'bootstrap': details, 'metrics': point_rows, 'paired_comparisons': paired_rows, 'limitations': ['Regional numeric variables are ecological proxies, not patient-level SES measurements or causal effects.', "Removing inputs changes feature sampling and fitted models; differences do not isolate a regional variable's causal contribution.", 'Stored-city consistency and numeric signature reproducibility do not establish correctness of original geographic/year assignment.', 'WHU numeric-signature groups are not verified cities or regions.', 'Stress tests change splits and, for SCH, the eligible training cohort; comparisons are not a single-cause geographic leakage estimate.', 'Full-record aggregation, original label limitations and absence of a temporal holdout remain.', 'All intervals condition on fixed OOF predictions, with no model refitting or multiplicity adjustment.']})

def verify(state):
    replay = []
    for cohort in COHORTS:
        X, y, groups, names, roles, baseline = load(cohort)
        for file_name, variants, stress in (('predictions.npz', state['protocol']['primary_variants'], False), ('stress_predictions.npz', ['stress'], True)):
            with np.load(ROOT / f'private/{cohort}/{file_name}', allow_pickle=False) as z:
                target = {k: z[k].copy() for k in z.files}
            if stress:
                with np.load(ROOT / f'private/{cohort}/stress_roles.npz', allow_pickle=False) as z:
                    ix, current_roles = (z['selected_index'].copy(), z['roles'].copy())
                current_X = X[ix]
                assert np.array_equal(target['y'], y[ix]) and np.array_equal(target['baseline_ensemble'], baseline[ix])
            else:
                current_X, current_roles = (X, roles)
                assert np.array_equal(target['y'], y) and np.array_equal(target['baseline_ensemble'], baseline)
            for variant in variants:
                for fold, role in enumerate(current_roles, 1):
                    folder = ROOT / f'private/{cohort}/{variant}/fold_{fold}'
                    report = read(folder / 'fit_report.json')
                    for item in report['model_identities']:
                        check(item)
                    package = joblib.load(folder / 'full_package.joblib')
                    v = values(current_X[role == 0], package['columns'])
                    model = xgb.Booster(params={'nthread': THREADS})
                    model.load_model(folder / 'full_xgb.ubj')
                    package['rf'].n_jobs = THREADS
                    p = 0.5 * model.predict(dm(v)) + 0.5 * package['rf'].predict_proba(v)[:, 1]
                    error = float(np.max(np.abs(p - target['stress_ensemble' if stress else variant][role == 0])))
                    assert error < 1e-10
                    replay.append({'cohort': cohort, 'variant': variant, 'fold': fold, 'maximum_error': error})
    for source in state['sources']:
        check(source)
    assert len(replay) == 30
    summary = read(ROOT / 'results/summary.json')
    assert summary['execution_complete'] and len(summary['metrics']) == 42 and (len(summary['paired_comparisons']) == 24)
    save(ROOT / 'verification.json', {'passed': True, 'models_replayed': len(replay), 'all_sources_unchanged': True, 'fold_replay': replay, 'summary': identity(ROOT / 'results/summary.json'), 'metric_csv': identity(ROOT / 'results/regional_metrics.csv'), 'paired_csv': identity(ROOT / 'results/regional_paired_comparisons.csv'), 'self_test': self_test()})
    print(json.dumps({'passed': True, 'verification': identity(ROOT / 'verification.json')}), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run', choices=(*COHORTS, 'all'))
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        state = freeze()
        if args.run:
            for cohort in COHORTS if args.run == 'all' else [args.run]:
                run(cohort, state)
        if args.summarize:
            summarize(state)
        if args.verify:
            verify(state)
