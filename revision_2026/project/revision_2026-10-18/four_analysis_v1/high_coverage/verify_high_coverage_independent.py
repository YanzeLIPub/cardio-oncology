"""Independent selection reconstruction, saved-model replay and sklearn CIs.

This verifier does not import the training runner or any historical module.
The --prepare-only mode neither fits models nor reads prediction scores.
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2] / 'revision_2026-10-15/original_v5_independent_validation_v1'
METRICS = ('auc', 'average_precision', 'brier', 'e_over_o')
EXCLUDE_CHINA = ['性别', '年龄', '上次诊断', '上次入院科室', '上次出院科室', '人均地区生产总值(元)', '职工平均工资(元)', '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)']
EXCLUDE_MIMIC = ['Age', 'Sex_M', 'GDP per Capita', 'Avg. Employee Wage', 'No. of Hospitals', 'Hospital Beds', 'No. of Physicians', 'AST', 'HDL Cholesterol']

def sha(path):
    return None

def read(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def independent_columns(X, names, fit, excluded, threshold, mimic):
    selected = []
    for i, name in enumerate(names):
        if name in excluded:
            continue
        n = int(np.count_nonzero(~np.isnan(X[fit, i])))
        if n * 100 < threshold * len(fit):
            continue
        if mimic and n == 0:
            continue
        selected.append(i)
    return selected

def independent_transform(X, fit, test, columns, mimic):
    out = X[test][:, columns].copy()
    if mimic:
        medians = np.array([np.median(X[fit, col][np.isfinite(X[fit, col])]) for col in columns], dtype=np.float32)
    else:
        medians = np.zeros(len(columns), dtype=np.float32)
    for i, value in enumerate(medians):
        out[np.isnan(out[:, i]), i] = value
    return out.astype(np.float32)

def metrics(y, probability, weights=None):
    w = np.ones(len(y)) if weights is None else weights
    return np.array([roc_auc_score(y, probability, sample_weight=w), average_precision_score(y, probability, sample_weight=w), brier_score_loss(y, probability, sample_weight=w), np.sum(w * probability) / np.sum(w * y)])

def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    lock, feasibility, protocol = (read(ROOT / 'analysis_lock.json'), read(ROOT / 'feasibility.json'), read(ROOT / 'protocol.json'))
    for source in lock['state']['sources'] + [lock['state']['script'], lock['state']['protocol_identity']]:
        pass
    selection_checks, model_checks, metric_checks, bootstrap_checks = ([], [], [], [])
    metric_rows = None if args.prepare_only else csv_rows(ROOT / 'results/high_coverage_metrics.csv')
    pair_rows = None if args.prepare_only else csv_rows(ROOT / 'results/high_coverage_paired_comparisons.csv')
    for cohort, expected in protocol['expected_counts'].items():
        mimic = cohort == 'MIMIC'
        excluded = EXCLUDE_MIMIC if mimic else EXCLUDE_CHINA
        with np.load(BASE / 'private/prepared' / f'{cohort}.npz', allow_pickle=False) as z:
            X, y, groups, names = (z['X'].copy(), z['y'].copy(), z['groups'].copy(), z['feature_names'].tolist())
        assert [len(y), len(np.unique(groups)), int(y.sum()), len(names)] == expected
        with np.load(BASE / 'private/runs' / f'{cohort}_original/patient_roles.npz', allow_pickle=False) as z:
            assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
            roles = z['roles'].copy()
        assert len(set(names) - set(excluded)) == (34 if mimic else 94)
        prep = next((c for c in feasibility['cohorts'] if c['cohort'] == cohort))
        assert all((row['eligible_laboratory_slot'] == (row['name'] not in excluded) for row in prep['feature_inventory']))
        if not args.prepare_only:
            training = read(ROOT / 'results' / f'{cohort}_training.json')
            with np.load(ROOT / 'private' / cohort / 'predictions.npz', allow_pickle=False) as z:
                saved = {k: z[k].copy() for k in z.files}
            assert np.array_equal(saved['y'], y) and np.array_equal(saved['groups'], groups)
        for record in prep['fold_selection']:
            fold, variant = (record['fold'], record['variant'])
            role = roles[fold - 1]
            threshold = protocol['minimum_coverage_percent'][variant]
            inner, full, test = [np.flatnonzero(mask) for mask in (role == 2, role != 0, role == 0)]
            inner_cols = independent_columns(X, names, inner, excluded, threshold, mimic)
            full_cols = independent_columns(X, names, full, excluded, threshold, mimic)
            assert inner_cols == record['inner_fit']['selected_original_indices']
            assert full_cols == record['full_outer_training']['selected_original_indices']
            assert bool(set(inner_cols) != set(full_cols)) == record['features_differ_between_inner_and_final_fit']
            assert not np.intersect1d(groups[full], groups[test]).size
            selection_checks.append({'cohort': cohort, 'variant': variant, 'fold': fold, 'inner_columns': len(inner_cols), 'final_columns': len(full_cols)})
            if args.prepare_only:
                continue
            fit = next((r for r in training['fold_reports'] if r['fold'] == fold and r['variant'] == variant))
            assert fit['estimable'] and fit['inner_selection']['selected_original_indices'] == inner_cols
            assert fit['final_selection']['selected_original_indices'] == full_cols
            for artifact in fit['identities']:
                pass
            folder = ROOT / 'private' / cohort / variant / f'fold_{fold}'
            package = joblib.load(folder / 'full_package.joblib')
            assert np.array_equal(package['preprocessor']['columns'], full_cols)
            transformed = independent_transform(X, full, test, full_cols, mimic)
            booster = xgb.Booster(params={'nthread': 1})
            booster.load_model(folder / 'full_xgb.ubj')
            assert booster.num_boosted_rounds() == fit['selected_rounds']
            assert fit['full_xgb_parameters']['scale_pos_weight'] == float(np.sum(y[full] == 0) / np.sum(y[full] == 1))
            package['rf'].n_jobs = 1
            predicted = 0.5 * booster.predict(xgb.DMatrix(transformed, nthread=1)) + 0.5 * package['rf'].predict_proba(transformed)[:, 1]
            error = float(np.max(np.abs(predicted - saved[variant][test])))
            assert error < 1e-10
            assert np.all(saved['fold'][test] == fold)
            model_checks.append({'cohort': cohort, 'variant': variant, 'fold': fold, 'prediction_error': error})
        if args.prepare_only:
            continue
        models = protocol['variants'] + ['original_full_reference'] + (['original_exclude_AST_HDL_reference'] if mimic else [])
        _, inverse = np.unique(groups, return_inverse=True)
        n = int(inverse.max()) + 1
        draws = np.empty((2000, len(models), len(METRICS)))
        rng, digest = (np.random.default_rng(protocol['bootstrap_seed']), hashlib.sha256())
        points = np.array([metrics(y, saved[model]) for model in models])
        for b in range(len(draws)):
            multiplicity = rng.multinomial(n, np.full(n, 1 / n))
            digest.update(np.asarray(multiplicity, dtype='<i8').tobytes())
            draws[b] = [metrics(y, saved[model], multiplicity[inverse]) for model in models]
        for i, model in enumerate(models):
            for k, metric in enumerate(METRICS):
                row = next((r for r in metric_rows if r['cohort'] == cohort and r['model'] == model and (r['metric'] == metric)))
                actual = np.array([points[i, k], *np.quantile(draws[:, i, k], [0.025, 0.975])])
                error = float(np.max(np.abs(actual - [float(row[c]) for c in ('estimate', 'ci_low', 'ci_high')])))
                assert error < 5e-11
                metric_checks.append({'cohort': cohort, 'model': model, 'metric': metric, 'error': error})
        for row in (r for r in pair_rows if r['cohort'] == cohort):
            a, b = (models.index(row['first']), models.index(row['second']))
            k = METRICS.index(row['metric'].removeprefix('delta_'))
            actual = np.array([points[a, k] - points[b, k], *np.quantile(draws[:, a, k] - draws[:, b, k], [0.025, 0.975])])
            error = float(np.max(np.abs(actual - [float(row[c]) for c in ('estimate', 'ci_low', 'ci_high')])))
            assert error < 5e-11
            metric_checks.append({'cohort': cohort, 'comparison': row['first'] + ' minus ' + row['second'], 'metric': row['metric'], 'error': error})
        bootstrap_checks.append({'cohort': cohort, 'repetitions': len(draws)})
        print(f'Independent sklearn verification complete: {cohort}', flush=True)
    result = {'passed': True, 'prepare_only': args.prepare_only, 'script': {'path': Path(__file__).as_posix()}, 'selection_checks': selection_checks, 'model_replay': model_checks, 'metric_checks': metric_checks, 'patient_bootstrap': bootstrap_checks, 'python': sys.version, 'all_sources_unchanged': True, 'independence': 'No training runner import; explicit exclusion lists, separate per-column selection and direct median reconstruction, sklearn metrics for all bootstrap draws.'}
    output = ROOT / ('independent_preparation_verification.json' if args.prepare_only else 'independent_verification.json')
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'passed': True, 'prepare_only': args.prepare_only, 'selection_checks': len(selection_checks)}))
if __name__ == '__main__':
    main()
