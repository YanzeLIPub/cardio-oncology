"""Independent model/role/metric replay without importing analysis code."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import ast
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2] / 'revision_2026-10-15/original_v5_independent_validation_v1'
METRICS = ('auc', 'average_precision', 'brier')

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def read(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def verify_identity(item):
    pass

def test_metrics(y, p, weight=None):
    return np.array([roc_auc_score(y, p, sample_weight=weight), average_precision_score(y, p, sample_weight=weight), brier_score_loss(y, p, sample_weight=weight)])

def reconstruct_roles(y, clusters, config):
    roles = np.full((5, len(y)), -1, dtype=np.int8)
    for fold, (full, test) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros((len(y), 1)), y, clusters), 1):
        base_ids, cal_ids = train_test_split(full, stratify=y[full], test_size=config['calibration_fraction_of_outer_train_patients'], random_state=config['calibration_seed_base'] + fold)
        base, cal = (full[np.isin(full, base_ids)], full[np.isin(full, cal_ids)])
        fit_ids, stop_ids = train_test_split(base, stratify=y[base], test_size=config['inner_validation_fraction_of_base_fit_patients'], random_state=config['inner_seed_base'] + fold)
        fit, stop = (base[np.isin(base, fit_ids)], base[np.isin(base, stop_ids)])
        for r, indices in enumerate((test, cal, fit, stop)):
            roles[fold - 1, indices] = r
        assert not np.intersect1d(clusters[full], clusters[test]).size
    return roles

def main():
    assert not (ROOT / 'independent_verification.json').exists()
    lock = read(ROOT / 'analysis_lock.json')
    for item in lock['state']['sources'] + [lock['state']['script']]:
        verify_identity(item)
    config = read(BASE / 'protocol.json')
    regional = None
    for node in ast.parse(_release_source_path(BASE / 'prepare_china.py').read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'REGIONAL_FIELDS' for t in node.targets)):
            regional = ast.literal_eval(node.value)
    assert len(regional) == 5
    metric_csv = pd.read_csv(ROOT / 'results/regional_metrics.csv')
    paired_csv = pd.read_csv(ROOT / 'results/regional_paired_comparisons.csv')
    frozen_metrics = pd.read_csv(BASE / 'results/validation_metrics.csv')
    summary = read(ROOT / 'results/summary.json')
    model_checks, interval_checks = ([], [])
    max_interval_error = 0.0
    for cohort in ('SCH', 'WHU'):
        with np.load(BASE / f'private/prepared/{cohort}.npz', allow_pickle=False) as z:
            X, y, groups, names = (z['X'], z['y'], z['groups'], z['feature_names'].tolist())
        with np.load(BASE / f'private/runs/{cohort}_original/patient_roles.npz', allow_pickle=False) as z:
            frozen_roles = z['roles'].copy()
            assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
        for filename, stress in (('predictions.npz', False), ('stress_predictions.npz', True)):
            with np.load(ROOT / f'private/{cohort}/{filename}', allow_pickle=False) as z:
                data = {k: z[k].copy() for k in z.files}
            if stress:
                with np.load(ROOT / f'private/{cohort}/stress_roles.npz', allow_pickle=False) as z:
                    selected, roles = (z['selected_index'].copy(), z['roles'].copy())
                    assert np.array_equal(data['y'], z['y']) and np.array_equal(data['clusters'], z['clusters'])
                assert np.array_equal(roles, reconstruct_roles(data['y'], data['clusters'], config))
                variants, models = (['stress'], ['baseline_ensemble', 'stress_ensemble'])
                cluster_keys = ('groups', 'clusters')
                analysis = lock['state']['protocol']['stress_tests'][cohort]['name']
            else:
                selected, roles = (np.arange(len(y)), data['roles'])
                assert np.array_equal(roles, frozen_roles)
                variants, models = (['no_regional', 'regional_only'], ['baseline_ensemble', 'no_regional', 'regional_only'])
                cluster_keys, analysis = (('groups',), 'original_patient_roles')
            assert np.array_equal(data['y'], y[selected]) and np.array_equal(data['groups'], groups[selected])
            current_X, current_y = (X[selected], y[selected])
            for variant in variants:
                expected_columns = np.arange(len(names)) if stress else np.array([names.index(f) for f in regional]) if variant == 'regional_only' else np.array([i for i, name in enumerate(names) if name not in regional])
                for fold, role in enumerate(roles, 1):
                    folder = ROOT / f'private/{cohort}/{variant}/fold_{fold}'
                    package = joblib.load(folder / 'full_package.joblib')
                    report = read(folder / 'fit_report.json')
                    assert np.array_equal(package['columns'], expected_columns)
                    assert report['feature_names'] == [names[c] for c in expected_columns]
                    rounds = int(np.argmax(report['inner_history']['inner_stop']['auc'])) + 1
                    assert rounds == report['selected_rounds']
                    for subset, parameter_key in ((role == 2, 'inner_xgb_parameters'), (role != 0, 'full_xgb_parameters')):
                        weight = (current_y[subset] == 0).sum() / (current_y[subset] == 1).sum()
                        assert report[parameter_key]['scale_pos_weight'] == weight
                    assert package['rf'].class_weight == {0: 1, 1: int((current_y[role != 0] == 0).sum() / (current_y[role != 0] == 1).sum())}
                    v = current_X[role == 0][:, expected_columns].copy()
                    v[np.isnan(v)] = 0.0
                    model = xgb.Booster(params={'nthread': 1})
                    model.load_model(folder / 'full_xgb.ubj')
                    model.set_param({'nthread': 1})
                    assert model.num_boosted_rounds() == rounds
                    package['rf'].n_jobs = 1
                    p = 0.5 * model.predict(xgb.DMatrix(v, nthread=1)) + 0.5 * package['rf'].predict_proba(v)[:, 1]
                    error = float(np.max(np.abs(p - data['stress_ensemble' if stress else variant][role == 0])))
                    assert error < 1e-10
                    model_checks.append({'cohort': cohort, 'variant': variant, 'fold': fold, 'maximum_prediction_error': error, 'fit_class_weights_verified': True, 'selected_rounds_verified': True})
            points = np.array([test_metrics(current_y, data[m]) for m in models])
            for key in cluster_keys:
                _, inverse = np.unique(data[key], return_inverse=True)
                n = int(inverse.max()) + 1
                rng, h = (np.random.default_rng(20261015), hashlib.sha256())
                draws = np.empty((2000, len(models), 3))
                for b in range(2000):
                    multiplicity = rng.multinomial(n, np.full(n, 1 / n))
                    h.update(np.asarray(multiplicity, dtype='<i8').tobytes())
                    weights = multiplicity[inverse]
                    if not 0 < np.sum(weights * current_y) < weights.sum():
                        draws[b] = np.nan
                    else:
                        draws[b] = [test_metrics(current_y, data[m], weights) for m in models]
                valid = np.isfinite(draws).all(axis=(1, 2))
                draws = draws[valid]
                unit = 'patient' if key == 'groups' else 'outer_group'
                detail = next((row for row in summary['bootstrap'] if row['cohort'] == cohort and row['analysis'] == analysis and (row['bootstrap_unit'] == unit)))
                assert detail['valid_bootstrap_replicates'] == int(valid.sum())
                for mi, model in enumerate(models):
                    for ki, metric in enumerate(METRICS):
                        row = metric_csv[(metric_csv.cohort == cohort) & (metric_csv.analysis == analysis) & (metric_csv.model == model) & (metric_csv.metric == metric) & (metric_csv.bootstrap_unit == unit)]
                        assert len(row) == 1
                        actual = row.iloc[0][['estimate', 'ci_low', 'ci_high']].to_numpy(dtype=float)
                        expected = np.r_[points[mi, ki], np.quantile(draws[:, mi, ki], [0.025, 0.975])]
                        error = float(np.max(np.abs(actual - expected)))
                        assert error < 5e-11
                        max_interval_error = max(max_interval_error, error)
                        if not stress and model == 'baseline_ensemble':
                            old = frozen_metrics[(frozen_metrics.run == f'{cohort}_original') & (frozen_metrics.model == 'primary_raw') & (frozen_metrics.metric == metric)].iloc[0]
                            assert np.allclose(expected, old[['estimate', 'ci_low', 'ci_high']].to_numpy(dtype=float), atol=5e-10, rtol=0)
                for mi in range(1, len(models)):
                    for ki, metric in enumerate(METRICS):
                        row = paired_csv[(paired_csv.cohort == cohort) & (paired_csv.analysis == analysis) & (paired_csv['first'] == models[mi]) & (paired_csv.metric == 'delta_' + metric) & (paired_csv.bootstrap_unit == unit)]
                        assert len(row) == 1
                        actual = row.iloc[0][['estimate', 'ci_low', 'ci_high']].to_numpy(dtype=float)
                        expected = np.r_[points[mi, ki] - points[0, ki], np.quantile(draws[:, mi, ki] - draws[:, 0, ki], [0.025, 0.975])]
                        error = float(np.max(np.abs(actual - expected)))
                        assert error < 5e-11
                        max_interval_error = max(max_interval_error, error)
                interval_checks.append({'cohort': cohort, 'analysis': analysis, 'bootstrap_unit': unit, 'all_2000_draws_recomputed_with_sklearn': True, 'valid_replicates': int(valid.sum()), 'models': models, 'all_points_and_intervals_match': True})
                print(f'{cohort} {analysis} {unit}: independent 2000-draw sklearn replay passed', flush=True)
    for item in lock['state']['sources']:
        verify_identity(item)
    assert len(model_checks) == 30 and len(interval_checks) == 6
    result = {'passed': True, 'training_module_not_imported': True, 'source_identities_unchanged': True, 'script': identity(__file__), 'analysis_lock': identity(ROOT / 'analysis_lock.json'), 'summary': identity(ROOT / 'results/summary.json'), 'model_checks': model_checks, 'bootstrap_checks': interval_checks, 'maximum_metric_interval_error': max_interval_error, 'environment': {'python': sys.version, 'threads': 1, **{p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'xgboost')}}, 'scope': 'Independent preprocessing, role reconstruction, saved-model replay, class-weight and selected-round checks, and every fixed-OOF sklearn bootstrap metric; not independent clinical validation or geographic assignment validation.'}
    with (ROOT / 'independent_verification.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'passed': True, 'output': identity(ROOT / 'independent_verification.json')}))
if __name__ == '__main__':
    main()
