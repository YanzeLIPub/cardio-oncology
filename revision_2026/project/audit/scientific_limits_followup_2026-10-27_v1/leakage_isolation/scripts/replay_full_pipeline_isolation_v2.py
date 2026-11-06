"""Exercise the actual patient-isolated run_one orchestration with fixed roles.

Only source AST change: restore original y at each outer fold, and optionally
flip that fold's outer-test y. Namespace ROOT redirects all writes to a fresh
private directory; patient_roles returns certified existing roles. All original
fitting, early-stopping, calibration, prediction and comparator calls remain.
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from datetime import datetime
PROJECT = Path(_release_path('project', ''))
SOURCE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
DEST = Path(__file__).resolve().parents[1]
PRIVATE = DEST / 'private/execution'
SOURCE_SHA = None
THREADS = 8
TOLERANCE = 1e-12

def digest(p):
    return None

def save(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def read(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def main():
    assert PRIVATE.exists(), 'Continuation expects preserved completed SCH runs'
    continuation_existing = [p for p in PRIVATE.rglob('*') if p.is_file()]
    continuation_before = [{'path': str(p)} for p in continuation_existing]
    support = _release_source_path(PROJECT / 'audit/environment_rebuild_2026-10-27_v1/scripts/china_replay.py')
    nodes = [n for n in ast.parse(support.read_text(encoding='utf-8-sig')).body if isinstance(n, ast.FunctionDef) and n.name in ['require', 'set_resources']]
    resources_namespace = dict(os=os, sys=sys)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(support), 'exec'), resources_namespace)
    resources = resources_namespace['set_resources'](THREADS)
    import joblib
    import numpy as np
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_info
    source_path = _release_source_path(SOURCE / 'run_validation.py')
    protocol_path = SOURCE / 'protocol.json'
    assert digest(source_path) == SOURCE_SHA
    protocol = read(protocol_path)
    old_lock = read(SOURCE / 'evidence/analysis_lock.json')
    old_identities = {Path(x['path']).resolve(): None for x in old_lock['state']['files']}
    assert digest(protocol_path) == old_identities[protocol_path.resolve()]

    def arrsha(a):
        a = np.asarray(a)
        return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes()).hexdigest()

    def semantic_forest(forest):
        trees = []
        for e in forest.estimators_:
            s = e.tree_.__getstate__()
            nodes = s['nodes']
            trees.append({'node_count': int(s['node_count']), 'max_depth': int(s['max_depth']), 'nodes': {key: arrsha(nodes[key]) for key in nodes.dtype.names}, 'values': arrsha(s['values']), 'random_state': int(e.random_state)})
        return trees

    def semantic_prep(prep):
        return {'zero_fill': bool(prep['zero_fill']), 'columns': arrsha(prep['columns']), 'imputer_statistics': None if prep['imputer'] is None else arrsha(prep['imputer'].statistics_)}

    def semantic_booster(path):
        b = xgb.Booster()
        b.load_model(path)
        return _release_resolve_config(json.loads(bytes(b.save_raw(raw_format='json')).decode()))

    def semantic_estimator(estimator):
        if hasattr(estimator, 'steps'):
            return [(name, semantic_estimator(step)) for name, step in estimator.steps]
        result = {'class': type(estimator).__name__}
        for field in ['coef_', 'intercept_', 'classes_', 'mean_', 'scale_', 'var_', 'statistics_', 'n_iter_']:
            if hasattr(estimator, field):
                result[field] = arrsha(getattr(estimator, field))
        if hasattr(estimator, 'n_features_in_'):
            result['n_features_in_'] = int(estimator.n_features_in_)
        return result

    def errors(a, b):
        d = np.abs(a - b)
        return {'max_abs': float(d.max()), 'mean_abs': float(d.mean()), 'n_above_1e_12': int((d > TOLERANCE).sum())}
    protected = [source_path, protocol_path, SOURCE / 'evidence/analysis_lock.json']
    for c in ['SCH', 'WHU', 'MIMIC']:
        protected.extend([SOURCE / 'private/prepared' / f'{c}.npz', SOURCE / 'private/runs' / f'{c}_original/patient_roles.npz', SOURCE / 'private/runs' / f'{c}_original/predictions.npz', SOURCE / 'results' / f'{c}_original_training.json'])
    before = [{'path': str(p)} for p in protected]
    PRIVATE.mkdir(parents=True, exist_ok=True)
    for p in ['private/prepared', 'private/runs', 'results', 'logs']:
        (PRIVATE / p).mkdir(parents=True, exist_ok=True)
    for c in ['SCH', 'WHU', 'MIMIC']:
        p = SOURCE / 'private/prepared' / f'{c}.npz'
        q = PRIVATE / 'private/prepared' / p.name
        if not q.exists():
            shutil.copyfile(p, q)
        assert digest(q) == digest(p)
    text = source_path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    names = ['sha', 'file_identity', 'save_json', 'now', 'fit_preprocessor', 'transform', 'xgb_parameters', 'choose_rounds', 'fit_ensemble', 'ensemble_predict', 'logit', 'make_logistic', 'save_model', 'run_one']
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    originals = {n.name: hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest() for n in selected}
    runner = next((n for n in selected if n.name == 'run_one'))
    loops = [n for n in runner.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and ([e.id for e in n.target.elts] == ['fold', 'role'])]
    assert len(loops) == 1
    loop = loops[0]
    position = runner.body.index(loop)
    runner.body[position:position] = ast.parse('isolation_original_y = y.copy()').body
    loop.body[:0] = ast.parse('y = isolation_original_y.copy()\nif TEST_MUTATION:\n    y[role == 0] = 1 - y[role == 0]').body
    patched = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))

    def frozen_roles(y, groups, cohort, config):
        with np.load(SOURCE / 'private/runs' / f'{cohort}_original/patient_roles.npz', allow_pickle=False) as z:
            assert np.array_equal(y, z['y']) and np.array_equal(groups, z['groups'])
            roles = z['roles'].copy()
        assert (roles == 0).sum(axis=0).min() == (roles == 0).sum(axis=0).max() == 1
        for role in roles:
            sets = [set(groups[role == i]) for i in range(4)]
            assert all((not sets[i].intersection(sets[j]) for i in range(4) for j in range(i + 1, 4)))
        return roles
    namespace = dict(globals(), joblib=joblib, np=np, xgb=xgb, RandomForestClassifier=RandomForestClassifier, SimpleImputer=SimpleImputer, LogisticRegression=LogisticRegression, roc_auc_score=roc_auc_score, make_pipeline=make_pipeline, StandardScaler=StandardScaler, ROOT=PRIVATE, patient_roles=frozen_roles, TEST_MUTATION=False)
    exec(compile(patched, str(source_path), 'exec'), namespace)
    original_runner = next((n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'run_one'))
    call_names = {'choose_rounds', 'fit_ensemble', 'ensemble_predict', 'fit', 'predict_proba'}

    def critical_calls(node):
        out = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                name = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ''
                if name in call_names:
                    out.append(ast.dump(n, include_attributes=False))
        return sorted(out)
    assert critical_calls(original_runner) == critical_calls(runner)
    binding = {'original_source': {'path': str(source_path)}, 'original_function_source_hashes': originals, 'changes': ['ROOT redirected to a new private execution directory', 'patient_roles loads existing authenticated roles without recomputing stratification', 'original y restored at start of every original outer loop; only role0 y flipped under TEST_MUTATION'], 'original_critical_fitting_calls_unchanged': True, 'critical_call_count': len(critical_calls(runner)), 'preloop_review': 'Only nonfinite normalization, prespecified feature exclusion, age-column lookup and elementwise missingness occur before original fold loop; no data-fitted preprocessing or class-weight selection before fold subsets.', 'not_executed': ['module main', 'freeze', 'self_test', 'patient_roles regeneration'], 'scope_limit': 'Fixed-role original orchestration and numerical fitting isolation, not original upstream feature builders or clinical event timing.'}
    save(DEST / 'aggregate/source_binding_v2.json', binding)
    report = {'status': 'running', 'date': '2026-10-27', 'purpose': 'Full-input patient-isolated pipeline outer-test-label isolation acceptance, not a new clinical model comparison', 'threads': THREADS, 'resources': resources, 'runtime': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'joblib']}, 'threadpool_info': threadpool_info()}, 'input_identities': before, 'tolerance': {'atol': TOLERANCE, 'rtol': 0}, 'cohorts': {}, 'scientific_temporal_label_validity_upgraded': False, 'raw_mutated_label_metrics_are_private_and_not_scientific_results': True}
    save(DEST / 'aggregate/isolation_verification_v2.json', report)
    lock_hash = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    started = time.monotonic()
    scores = ['primary_raw', 'base_raw', 'base_platt', 'xgb_full', 'rf_full', 'logistic', 'missingness', 'age']
    for c in ['SCH', 'WHU', 'MIMIC']:
        for flag in [False, True]:
            scenario = 'flipped' if flag else 'baseline'
            namespace['TEST_MUTATION'] = flag
            spec = {'name': f'{c}_{scenario}', 'cohort': c, 'drop_features': []}
            existing_prediction = PRIVATE / 'private/runs' / spec['name'] / 'predictions.npz'
            existing_report = PRIVATE / 'results' / f"{spec['name']}_training.json"
            if existing_prediction.exists() and existing_report.exists():
                print(json.dumps({'cohort': c, 'scenario': scenario, 'stage': 'reusing_hash_preserved_completed_run'}), flush=True)
                continue
            print(json.dumps({'cohort': c, 'scenario': scenario, 'stage': 'starting_5_original_outer_folds'}), flush=True)
            with (PRIVATE / 'logs' / f'{c}_{scenario}.txt').open('x', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    namespace['run_one'](spec, protocol, THREADS, lock_hash)
            print(json.dumps({'cohort': c, 'scenario': scenario, 'stage': 'completed_5_outer_folds'}), flush=True)
        base_dir = PRIVATE / 'private/runs' / f'{c}_baseline'
        mut_dir = PRIVATE / 'private/runs' / f'{c}_flipped'
        a = np.load(base_dir / 'predictions.npz', allow_pickle=False)
        b = np.load(mut_dir / 'predictions.npz', allow_pickle=False)
        old = np.load(SOURCE / 'private/runs' / f'{c}_original/predictions.npz', allow_pickle=False)
        assert np.array_equal(a['groups'], b['groups']) and np.array_equal(a['fold'], b['fold'])
        assert np.array_equal(a['y'], old['y']) and np.array_equal(a['fold'], old['fold'])
        expected_y = a['y'].copy()
        expected_y[a['fold'] == 5] = 1 - expected_y[a['fold'] == 5]
        assert np.array_equal(b['y'], expected_y), 'Original run_one persists final-loop y; its known diagnostic mutation must match'
        comparisons = {key: errors(a[key], b[key]) for key in scores}
        cache = {key: errors(a[key], old[key]) for key in scores}
        details = []
        for fold in range(1, 6):
            ad = base_dir / f'fold_{fold}'
            bd = mut_dir / f'fold_{fold}'
            af, bf = (read(ad / 'fit_report.json'), read(bd / 'fit_report.json'))
            checks = {'inner_evaluation_identical': read(ad / 'inner_evaluation.json') == read(bd / 'inner_evaluation.json'), 'selected_rounds_identical': af['selection'] == bf['selection'], 'full_parameters_identical': af['full_parameters'] == bf['full_parameters'], 'base_parameters_identical': af['base_parameters'] == bf['base_parameters'], 'platt_coefficients_identical': af['platt'] == bf['platt'], 'inner_preprocessor_identical': semantic_prep(joblib.load(ad / 'inner_preprocessor.joblib')) == semantic_prep(joblib.load(bd / 'inner_preprocessor.joblib')), 'inner_booster_identical': semantic_booster(ad / 'inner_stop_model.ubj') == semantic_booster(bd / 'inner_stop_model.ubj')}
            for kind in ['base', 'full']:
                ap = joblib.load(ad / f'{kind}_package.joblib')
                bp = joblib.load(bd / f'{kind}_package.joblib')
                checks[kind + '_preprocessor_identical'] = semantic_prep(ap['preprocessor']) == semantic_prep(bp['preprocessor'])
                checks[kind + '_forest_semantic_arrays_identical'] = semantic_forest(ap['rf']) == semantic_forest(bp['rf'])
                checks[kind + '_booster_identical'] = semantic_booster(ad / f'{kind}_xgb.ubj') == semantic_booster(bd / f'{kind}_xgb.ubj')
            ap = joblib.load(ad / 'comparators_and_platt.joblib')
            bp = joblib.load(bd / 'comparators_and_platt.joblib')
            for key in ['platt', 'logistic', 'missingness', 'age']:
                checks[key + '_learned_state_identical'] = semantic_estimator(ap[key]) == semantic_estimator(bp[key])
            platt_deltas = {key: float(np.max(np.abs(getattr(ap['platt'], key) - getattr(bp['platt'], key)))) for key in ['coef_', 'intercept_']}
            checks['platt_coefficients_within_predeclared_atol_1e_12'] = all((d <= TOLERANCE for d in platt_deltas.values()))
            checks['platt_discrete_state_exact'] = all((np.array_equal(getattr(ap['platt'], key), getattr(bp['platt'], key)) for key in ['classes_', 'n_iter_', 'n_features_in_']))
            checks['platt_fit_role_description_exact'] = af['platt']['fitted_to'] == bf['platt']['fitted_to']
            accepted = {key: value for key, value in checks.items() if key not in ['platt_coefficients_identical', 'platt_learned_state_identical']}
            details.append({'fold': fold, 'checks': checks, 'platt_parameter_max_abs': platt_deltas, 'all_bitwise_checks_passed': all(checks.values()), 'passed': all(accepted.values()), 'selected_rounds': af['selection']['selected_rounds']})
        item = {'n': len(a['y']), 'events': int(a['y'].sum()), 'unit': 'admission' if c == 'MIMIC' else 'patient', 'patient_roles_and_identity_preserved': True, 'folds': details, 'baseline_vs_mutated_probabilities': comparisons, 'baseline_vs_existing_saved_probabilities': cache, 'full_pipeline_mutation_passed': all((v['n_above_1e_12'] == 0 for v in comparisons.values())) and all((v['passed'] for v in details)), 'three_paper_score_channels_match_saved': all((cache[key]['n_above_1e_12'] == 0 for key in ['primary_raw', 'base_raw', 'base_platt']))}
        report['cohorts'][c] = item
        save(DEST / 'aggregate/isolation_verification_v2.json', report)
        assert item['full_pipeline_mutation_passed'], 'Mutation affected fitted state/predictions; preserve evidence and investigate'
        assert item['three_paper_score_channels_match_saved'], 'Baseline does not bind saved paper validation predictions'
    after = [dict(x, unchanged=digest(Path(x['path'])) == None) for x in before]
    assert all((x['unchanged'] for x in after))
    report['continuation_reused_files_preserved'] = True
    report['continuation_reused_file_count'] = len(continuation_before)
    report['verification_protocol_v2_sha256'] = digest(DEST / 'aggregate/verification_protocol_v2.json')
    report['bitwise_Platts_are_not_claimed_identical'] = True
    report.update({'status': 'complete', 'passed': True, 'completed_folds': 30, 'paired_fold_comparisons': 15, 'protected_files_unchanged': True, 'after_identities': after, 'seconds': time.monotonic() - started, 'limitations': ['Original frozen role generation was not mutated or regenerated.', 'This tests the actual saved-input run_one orchestration and fitting paths, not upstream clinical feature construction.', 'Clinical reference incorporation, event timing, availability, diagnosis truth and original selection remain unresolved.', 'This does not certify the historical main CV, which has different evaluation rules.']})
    save(DEST / 'aggregate/isolation_verification_v2.json', report)
    print(json.dumps({'status': 'complete', 'paired_fold_comparisons': 15, 'model_fitting_outer_folds': 30, 'passed': True, 'seconds': report['seconds']}), flush=True)
if __name__ == '__main__':
    main()
