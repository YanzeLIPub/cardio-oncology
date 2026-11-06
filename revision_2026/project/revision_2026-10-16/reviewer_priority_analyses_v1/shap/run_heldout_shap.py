"""Read saved models only; write aggregate held-out explanations in this branch."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
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
from scipy.special import expit
from scipy.stats import false_discovery_control, rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2] / 'revision_2026-10-15/original_v5_independent_validation_v1'
MISSING = ROOT.parent / 'missingness'
COHORTS = ('SCH', 'WHU')
VARIANTS = ('value_indicator_ensemble', 'native_xgb')

def now():
    return datetime.now(timezone.utc).isoformat()

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def read_json(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def save_json(path, data):
    path = Path(path)
    assert path.is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')

def check(expected):
    actual = identity(expected['path'])
    assert actual == expected, f"Identity mismatch: {expected['path']}"
    return actual

def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'maximum_threads': 2, 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'xgboost', 'threadpoolctl')}}

def source_paths():
    paths = [BASE / p for p in ('protocol.json', 'run_validation.py', 'prepare_china.py', 'evidence/analysis_lock.json', 'evidence/validation_replay_audit.json', 'evidence/china_preparation.json')]
    paths += [MISSING / p for p in ('protocol.json', 'analysis_lock.json', 'run_missingness.py', 'verification.json', 'independent_verification.json')]
    for cohort in COHORTS:
        run = BASE / 'private/runs' / f'{cohort}_original'
        paths += [BASE / 'private/prepared' / f'{cohort}.npz', run / 'patient_roles.npz', run / 'predictions.npz', BASE / 'results' / f'{cohort}_original_training.json', MISSING / 'results' / f'{cohort}_training.json', MISSING / 'private' / cohort / 'predictions.npz']
        for fold in range(1, 6):
            paths += [run / f'fold_{fold}' / n for n in ('full_xgb.ubj', 'full_package.joblib')]
            for variant in VARIANTS:
                paths += [MISSING / 'private' / cohort / variant / f'fold_{fold}' / n for n in ('full_xgb.ubj', 'full_package.joblib', 'heldout_xgb_contributions.npz')]
    return paths

def frozen_source_checks():
    old_lock = read_json(BASE / 'evidence/analysis_lock.json')
    for item in old_lock['state']['files']:
        check(item)
    replay = read_json(BASE / 'evidence/validation_replay_audit.json')
    assert replay['passed']
    registered = {item['path']: item for item in replay['replayed_artifacts']}
    for cohort in COHORTS:
        report = read_json(BASE / 'results' / f'{cohort}_original_training.json')
        for key in ('input', 'roles_identity', 'prediction_identity'):
            check(report[key])
        for fold in range(1, 6):
            for name in ('full_xgb.ubj', 'full_package.joblib'):
                p = (BASE / 'private/runs' / f'{cohort}_original' / f'fold_{fold}' / name).as_posix()
                check(registered[p])
        report = read_json(MISSING / 'results' / f'{cohort}_training.json')
        assert report['execution_complete']
        for key in ('input_identity', 'roles_identity', 'prediction_identity'):
            check(report[key])
        for fold_report in report['fold_reports']:
            for item in fold_report['model_identities']:
                check(item)
    for name in ('verification.json', 'independent_verification.json'):
        assert read_json(MISSING / name)['passed']

def lock():
    frozen_source_checks()
    state = {'protocol': identity(ROOT / 'protocol.json'), 'script': identity(__file__), 'sources': [identity(p) for p in source_paths()], 'environment': environment()}
    path = ROOT / 'analysis_lock.json'
    if path.exists():
        assert read_json(path)['state'] == state
    else:
        save_json(path, {'frozen_before_analysis_at': now(), 'state': state})
    return state

def split_part(indices, y, groups, fraction, seed):
    unique = np.unique(groups[indices])
    label = np.array([y[indices][groups[indices] == group].max() for group in unique])
    train, hold = train_test_split(unique, stratify=label, test_size=fraction, random_state=seed)
    return (indices[np.isin(groups[indices], train)], indices[np.isin(groups[indices], hold)])

def reproduce_roles(y, groups):
    config = read_json(BASE / 'protocol.json')
    matrix = np.full((5, len(y)), -1, dtype=np.int8)
    splitter = StratifiedKFold(5, shuffle=True, random_state=config['outer_seed'])
    for fold, (train, test) in enumerate(splitter.split(np.zeros((len(y), 1)), y), 1):
        base, cal = split_part(train, y, groups, config['calibration_fraction_of_outer_train_patients'], config['calibration_seed_base'] + fold)
        inner, stop = split_part(base, y, groups, config['inner_validation_fraction_of_base_fit_patients'], config['inner_seed_base'] + fold)
        for role, index in enumerate((test, cal, inner, stop)):
            matrix[fold - 1, index] = role
    return matrix

def load_cohort(cohort):
    with np.load(BASE / 'private/prepared' / f'{cohort}.npz', allow_pickle=False) as z:
        X, y, groups, names = (z['X'].copy(), z['y'].copy(), z['groups'].copy(), z['feature_names'].tolist())
    expected = read_json(ROOT / 'protocol.json')['expected_counts'][cohort]
    assert [len(y), len(np.unique(groups)), int(y.sum()), len(names)] == expected
    assert not np.isinf(X).any()
    run = BASE / 'private/runs' / f'{cohort}_original'
    with np.load(run / 'patient_roles.npz', allow_pickle=False) as z:
        roles = z['roles'].copy()
        assert np.array_equal(groups, z['groups']) and np.array_equal(y, z['y'])
    assert np.array_equal(roles, reproduce_roles(y, groups))
    assert np.all((roles == 0).sum(axis=0) == 1)
    with np.load(run / 'predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(groups, z['groups']) and np.array_equal(y, z['y'])
        folds, probabilities = (z['fold'].copy(), z['xgb_full'].copy())
    with np.load(MISSING / 'private' / cohort / 'predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(groups, z['groups']) and np.array_equal(y, z['y'])
        assert np.array_equal(folds, z['fold'])
        assert np.array_equal(probabilities, z['baseline_xgb'])
        other_p = {'value_indicator_ensemble': z['value_indicator_xgb'].copy(), 'native_xgb': z['native_xgb'].copy()}
    for fold, role in enumerate(roles, 1):
        sets = [set(groups[role == r]) for r in range(4)]
        assert all((not sets[a].intersection(sets[b]) for a in range(4) for b in range(a + 1, 4)))
        assert np.array_equal(role == 0, folds == fold)
    return (X, names, roles, probabilities, other_p)

def association(raw, contribution):
    valid = np.isfinite(raw)
    x, s = (raw[valid], contribution[valid])
    result = {'n_observed': len(x), 'n_unique_raw': len(np.unique(x)), 'n_observed_zero': int((x == 0).sum()), 'rho': None, 'p_nominal': None, 'testable': False}
    reason = 'fewer_than_three_observations' if len(x) < 3 else 'constant_raw_values' if np.unique(x).size < 2 else 'constant_contribution_on_observed_values' if np.unique(s).size < 2 else None
    result['not_testable_reason'] = reason
    if reason is None:
        rho, p = spearmanr(x, s)
        assert np.isfinite(rho) and np.isfinite(p)
        result.update(rho=float(rho), p_nominal=float(p), testable=True)
    return result

def bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind='stable')
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    assert np.allclose(q, false_discovery_control(p, method='bh'), atol=1e-14, rtol=1e-14)
    return q

def baseline_contributions(cohort, X, names, roles, probabilities):
    values = np.full(X.shape, np.nan, dtype=np.float32)
    fold_checks = []
    for fold, role in enumerate(roles, 1):
        index = np.flatnonzero(role == 0)
        folder = BASE / 'private/runs' / f'{cohort}_original' / f'fold_{fold}'
        package = joblib.load(folder / 'full_package.joblib')
        prep = package['preprocessor']
        assert package['input_feature_names'] == names and package['fitted_feature_names'] == names
        assert prep['zero_fill'] and prep['imputer'] is None
        assert np.array_equal(prep['columns'], np.arange(len(names)))
        transformed = np.nan_to_num(X[index, :], nan=0.0, posinf=0.0, neginf=0.0)
        matrix = xgb.DMatrix(transformed, nthread=2)
        model = xgb.Booster(params={'nthread': 2})
        model.load_model(folder / 'full_xgb.ubj')
        model.set_param({'nthread': 2})
        contributions = model.predict(matrix, pred_contribs=True, approx_contribs=False)
        margin, probability = (model.predict(matrix, output_margin=True), model.predict(matrix))
        assert contributions.shape == (len(index), len(names) + 1)
        assert np.isfinite(contributions).all()
        errors = {'margin_additivity_max_abs_error': float(np.max(np.abs(contributions.sum(axis=1) - margin))), 'sigmoid_margin_probability_max_abs_error': float(np.max(np.abs(expit(margin) - probability))), 'saved_xgb_probability_max_abs_error': float(np.max(np.abs(probability - probabilities[index])))}
        assert errors['margin_additivity_max_abs_error'] <= 0.0002
        assert errors['sigmoid_margin_probability_max_abs_error'] <= 1e-07
        assert errors['saved_xgb_probability_max_abs_error'] <= 1e-07
        values[index] = contributions[:, :-1]
        fold_checks.append({'cohort': cohort, 'fold': fold, 'test_rows': len(index), 'outer_train_rows': int((role != 0).sum()), 'patient_intersections': 0, 'saved_boosted_rounds': model.num_boosted_rounds(), **errors})
        print(f'{cohort} fold {fold}: held-out explanation and probability replay passed', flush=True)
    assert np.isfinite(values).all()
    return (values, fold_checks)

def sensitivity_contributions(cohort, X, names, roles, probabilities, variant):
    values = np.full(X.shape, np.nan, dtype=np.float32)
    indicator_totals = np.zeros(len(names), dtype=float)
    checks = []
    for fold, role in enumerate(roles, 1):
        folder = MISSING / 'private' / cohort / variant / f'fold_{fold}'
        with np.load(folder / 'heldout_xgb_contributions.npz', allow_pickle=False) as z:
            test, contrib = (z['test_index'], z['contributions'])
            fnames = z['feature_names'].tolist()
            assert np.array_equal(test, np.flatnonzero(role == 0))
            assert fnames[:len(names)] == names and fnames[-1] == 'BIAS'
            assert np.isfinite(contrib).all()
            margin_error = float(np.max(np.abs(contrib.sum(axis=1) - z['margin'])))
            probability_error = float(np.max(np.abs(z['probability'] - probabilities[test])))
            assert margin_error < 0.0002 and probability_error < 1e-07
            assert np.max(np.abs(expit(z['margin']) - z['probability'])) < 1e-07
            values[test] = contrib[:, :len(names)]
            for j, name in enumerate(fnames[len(names):-1], len(names)):
                assert name.startswith('missing::')
                indicator_totals[names.index(name[len('missing::'):])] += np.abs(contrib[:, j]).sum(dtype=np.float64)
            checks.append({'cohort': cohort, 'variant': variant, 'fold': fold, 'identical_test_indices': True, 'value_channels': len(names), 'indicator_channels': len(fnames) - len(names) - 1, 'margin_additivity_max_abs_error': margin_error, 'saved_probability_max_abs_error': probability_error})
    assert np.isfinite(values).all()
    return (values, indicator_totals / len(X), checks)

def rank_summary(cohort, variant, original, other, indicator):
    a, b = (np.abs(original).mean(axis=0, dtype=np.float64), np.abs(other).mean(axis=0, dtype=np.float64))
    nonzero = (a > 0) | (b > 0)
    result = {'cohort': cohort, 'variant': variant, 'value_channels': len(a), 'nonzero_union_channels': int(nonzero.sum()), 'importance_rank_rho_all': float(spearmanr(a, b).statistic), 'importance_rank_rho_nonzero_union': float(spearmanr(a[nonzero], b[nonzero]).statistic), 'total_mean_absolute_indicator_contributions': float(indicator.sum())}
    for k in (10, 20):
        aset = set(np.argsort(-a, kind='stable')[a[np.argsort(-a, kind='stable')] > 0][:k])
        bset = set(np.argsort(-b, kind='stable')[b[np.argsort(-b, kind='stable')] > 0][:k])
        result[f'top_{k}_baseline_n'] = len(aset)
        result[f'top_{k}_variant_n'] = len(bset)
        result[f'top_{k}_overlap_n'] = len(aset & bset)
    return (result, a, b)

def calculate():
    associations, folds, diagnostics, sens_diag, stability, stability_features = ([], [], [], [], [], [])
    for cohort in COHORTS:
        X, names, roles, probabilities, other_p = load_cohort(cohort)
        contributions, checks = baseline_contributions(cohort, X, names, roles, probabilities)
        diagnostics.extend(checks)
        for column, name in enumerate(names):
            raw, s = (X[:, column], contributions[:, column])
            row = {'cohort': cohort, 'feature_index': column, 'feature': name, 'rows': len(X), **association(raw, s), 'mean_abs_contribution_all': float(np.abs(s).mean(dtype=np.float64)), 'mean_abs_contribution_observed': float(np.abs(s[np.isfinite(raw)]).mean(dtype=np.float64)) if np.isfinite(raw).any() else None, 'mean_abs_contribution_missing': float(np.abs(s[~np.isfinite(raw)]).mean(dtype=np.float64)) if (~np.isfinite(raw)).any() else None}
            signs = []
            for fold, role in enumerate(roles, 1):
                stat = association(raw[role == 0], s[role == 0])
                folds.append({'cohort': cohort, 'feature_index': column, 'feature': name, 'fold': fold, **stat})
                if stat['testable']:
                    signs.append(int(np.sign(stat['rho'])))
            row.update(folds_testable=len(signs), folds_positive=signs.count(1), folds_negative=signs.count(-1), folds_zero=signs.count(0), folds_untestable=5 - len(signs), folds_agree_pooled=signs.count(int(np.sign(row['rho']))) if row['testable'] else None, all_five_fold_sign_agreement=bool(row['testable'] and row['rho'] != 0 and (len(signs) == 5) and all((s == np.sign(row['rho']) for s in signs))))
            associations.append(row)
        for variant in VARIANTS:
            other, indicator, checks = sensitivity_contributions(cohort, X, names, roles, other_p[variant], variant)
            sens_diag.extend(checks)
            rank, a, b = rank_summary(cohort, variant, contributions, other, indicator)
            stability.append(rank)
            arank, brank = (rankdata(-a), rankdata(-b))
            for j, name in enumerate(names):
                stability_features.append({'cohort': cohort, 'variant': variant, 'feature_index': j, 'feature': name, 'baseline_mean_abs': float(a[j]), 'variant_value_mean_abs': float(b[j]), 'baseline_rank': float(arank[j]), 'variant_value_rank': float(brank[j]), 'variant_indicator_mean_abs': float(indicator[j])})
    family = [row for row in associations if row['testable']]
    qvalues = bh([row['p_nominal'] for row in family])
    for row, q in zip(family, qvalues):
        row['q_bh_joint_family'] = float(q)
    for row in associations:
        row.setdefault('q_bh_joint_family', None)
    lookup = {(r['cohort'], r['feature']): r for r in associations}
    pairs = []
    for feature in [r['feature'] for r in associations if r['cohort'] == 'SCH']:
        a, b = (lookup['SCH', feature], lookup['WHU', feature])
        testable = a['testable'] and b['testable']
        opposite = bool(testable and a['rho'] * b['rho'] < 0)
        both_q = bool(testable and a['q_bh_joint_family'] < 0.05 and (b['q_bh_joint_family'] < 0.05))
        fold_stable = a['all_five_fold_sign_agreement'] and b['all_five_fold_sign_agreement']
        pairs.append({'feature': feature, 'testable_both': testable, 'SCH_rho': a['rho'], 'WHU_rho': b['rho'], 'SCH_q_joint': a['q_bh_joint_family'], 'WHU_q_joint': b['q_bh_joint_family'], 'opposite_signs': opposite, 'both_q_below_0_05': both_q, 'both_all_five_fold_sign_agreement': fold_stable, 'opposite_both_q_and_five_fold_agreement': bool(opposite and both_q and fold_stable)})
    summary = {'scope': read_json(ROOT / 'protocol.json')['scope'], 'complete_joint_BH_family_size': len(family), 'nominal_feature_cohort_slots': len(associations), 'cohorts': [], 'cross_cohort': {key: sum((bool(r[key]) for r in pairs)) for key in ('testable_both', 'opposite_signs', 'both_q_below_0_05', 'both_all_five_fold_sign_agreement', 'opposite_both_q_and_five_fold_agreement')}, 'missingness_importance_stability': stability, 'model_replay_and_additivity': diagnostics, 'sensitivity_contribution_checks': sens_diag, 'limits': read_json(ROOT / 'protocol.json')['limits']}
    for cohort in COHORTS:
        rows = [r for r in associations if r['cohort'] == cohort]
        summary['cohorts'].append({'cohort': cohort, 'rows': rows[0]['rows'], 'nominal_features': len(rows), 'testable': sum((r['testable'] for r in rows)), 'q_below_0_05': sum((r['q_bh_joint_family'] is not None and r['q_bh_joint_family'] < 0.05 for r in rows)), 'all_five_fold_sign_agreement': sum((r['all_five_fold_sign_agreement'] for r in rows)), 'q_below_0_05_and_all_five_fold_agreement': sum((r['q_bh_joint_family'] is not None and r['q_bh_joint_family'] < 0.05 and r['all_five_fold_sign_agreement'] for r in rows))})
    tables = {'feature_associations.csv': associations, 'fold_associations.csv': folds, 'cross_cohort_associations.csv': pairs, 'missingness_importance_stability.csv': stability, 'missingness_feature_importance.csv': stability_features}
    return (summary, tables)

def self_test():
    rng = np.random.default_rng(20261016)
    for n in (1, 2, 10, 104, 208):
        bh(rng.uniform(size=n))
    assert np.allclose(bh([0.01, 0.04, 0.03, 0.002]), [0.02, 0.04, 0.04, 0.008])
    example = association(np.array([np.nan, 0.0, 1.0, 2.0]), np.array([-500.0, 0.0, 2.0, 3.0]))
    assert example['n_observed'] == 3 and example['n_observed_zero'] == 1 and (example['rho'] == 1)
    assert not association(np.array([0.0, 0.0, 0.0]), np.arange(3))['testable']
    assert not association(np.arange(3.0), np.zeros(3))['testable']
    assert not association(np.array([np.nan, 1.0, np.nan]), np.arange(3))['testable']
    return {'bh_matches_independent_scipy_implementation': True, 'observed_zero_preserved_missing_excluded': True, 'constant_and_sparse_associations_not_tested': True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('lock', 'run', 'verify'))
    args = parser.parse_args()
    state = lock()
    if args.mode == 'lock':
        print(json.dumps({'locked': True, 'source_count': len(state['sources']), 'lock': identity(ROOT / 'analysis_lock.json')}))
        return
    test = self_test()
    with threadpool_limits(limits=2):
        summary, tables = calculate()
    for item in state['sources']:
        check(item)
    if args.mode == 'run':
        output = ROOT / 'results'
        output.mkdir(exist_ok=True)
        artifacts = []
        for name, rows in tables.items():
            path = output / name
            assert not path.exists()
            pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8', float_format='%.17g')
            artifacts.append(identity(path))
        save_json(output / 'summary.json', {'completed_at': now(), 'analysis_lock': identity(ROOT / 'analysis_lock.json'), 'environment': environment(), 'self_tests': test, 'results': summary, 'table_identities': artifacts, 'sources_unchanged': True, 'no_training_performed': True, 'no_patient_rows_or_identifiers_exported': True})
        print(json.dumps({'completed': True, 'cohorts': summary['cohorts'], 'cross_cohort': summary['cross_cohort']}))
    else:
        saved = read_json(ROOT / 'results/summary.json')
        assert saved['results'] == summary
        for item in saved['table_identities']:
            check(item)
            name = Path(item['path']).name
            expected = pd.DataFrame(tables[name]).to_csv(index=False, float_format='%.17g')
            actual = Path(item['path']).read_text(encoding='utf-8')
            assert actual == expected
        save_json(ROOT / 'verification.json', {'passed': True, 'completed_at': now(), 'analysis_lock': identity(ROOT / 'analysis_lock.json'), 'summary': identity(ROOT / 'results/summary.json'), 'all_table_identities': saved['table_identities'], 'self_tests': test, 'all_ten_saved_model_explanations_recomputed': True, 'all_roles_reproduced': True, 'all_twenty_sensitivity_arrays_rechecked': True, 'all_aggregate_tables_exactly_reproduced': True, 'all_consumed_sources_unchanged': True, 'no_training_or_patient_exports': True, 'verification_scope': 'Separate invocation of this analysis, with independent scipy BH reference and source audit bindings; not an independent second implementation of TreeSHAP.'})
        print(json.dumps({'verified': True, 'verification': identity(ROOT / 'verification.json')}))
if __name__ == '__main__':
    main()
