"""Recover predictions after a serialization-name collision; no fitting or model changes."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import json
import joblib
import numpy as np
import xgboost as xgb
from prepare import ROOT, OLD, identity, save_json, sha
from run_v2 import dmat, zero, lock

def main():
    before = ast.parse(_release_source_path(ROOT / 'run.py').read_text(encoding='utf-8'))
    after = ast.parse(_release_source_path(ROOT / 'run_v2.py').read_text(encoding='utf-8'))
    old = next((n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == 'run'))
    new = next((n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == 'run'))
    changes = 0
    for node in ast.walk(old):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (node.func.attr == 'savez_compressed'):
            for kw in node.keywords:
                if kw.arg == 'record_count':
                    kw.arg = 'processed_record_count'
                    changes += 1
    assert changes == 1 and ast.dump(old) == ast.dump(new)
    v1lock = _release_resolve_config(json.loads((ROOT / 'analysis_lock.json').read_text(encoding='utf-8')))
    for item in v1lock['state']['files']:
        assert identity(item['path']) == item
    lockhash = lock()
    center = 'SCH'
    folder = ROOT / 'private/runs/SCH'
    broken = folder / 'predictions.npz'
    assert broken.stat().st_size == 0
    broken.rename(folder / 'predictions_failed_empty_v1.npz')
    data = dict(np.load(ROOT / 'private/prepared/SCH.npz', allow_pickle=False))
    prior = dict(np.load(OLD / 'private/runs/SCH_original/predictions.npz', allow_pickle=False))
    out = {name: np.full(len(data['y']), np.nan) for name in ('mean', 'first_record', 'first_observed', 'median', 'record_count')}
    components = {f'{v}_{c}': np.full(len(data['y']), np.nan) for v in out if v != 'record_count' for c in ('xgb', 'rf')}
    reports, model_inputs = ([], [])
    for fold, roles in enumerate(data['roles'], 1):
        test = np.flatnonzero(roles == 0)
        for variant in ('mean', 'first_record', 'first_observed', 'median'):
            dest = OLD / f'private/runs/SCH_original/fold_{fold}' if variant == 'mean' else folder / f'fold_{fold}_{variant}'
            for name in ('full_xgb.ubj', 'full_package.joblib'):
                model_inputs.append(identity(dest / name))
            booster = xgb.Booster()
            booster.load_model(dest / 'full_xgb.ubj')
            booster.set_param({'nthread': 2})
            pack = joblib.load(dest / 'full_package.joblib')
            pack['rf'].n_jobs = 2
            X = data['X_' + variant][test]
            xp, rp = (booster.predict(dmat(X)), pack['rf'].predict_proba(zero(X))[:, 1])
            components[variant + '_xgb'][test], components[variant + '_rf'][test] = (xp, rp)
            out[variant][test] = 0.5 * xp + 0.5 * rp
            if variant == 'mean':
                np.testing.assert_allclose(out[variant][test], prior['primary_raw'][test], rtol=0, atol=2e-15)
                out[variant][test] = prior['primary_raw'][test]
            else:
                reports.append(_release_resolve_config(json.loads((dest / 'fit_report.json').read_text(encoding='utf-8'))))
        countfile = folder / f'fold_{fold}_record_count.joblib'
        model_inputs.append(identity(countfile))
        model = joblib.load(countfile)
        out['record_count'][test] = model.predict_proba(np.log1p(data['record_count'][test]).reshape(-1, 1))[:, 1]
    assert all((np.isfinite(p).all() for p in out.values()))
    with broken.open('xb') as stream:
        np.savez_compressed(stream, y=data['y'], groups=data['groups'], roles=data['roles'], fold=np.argmax(data['roles'] == 0, axis=0) + 1, processed_record_count=data['record_count'], **out, **components)
    assert [identity(item['path']) for item in model_inputs] == model_inputs
    save_json(ROOT / 'results/SCH_training.json', {'center': center, 'training_complete': True, 'model_report_count': len(reports), 'mean_models_replayed': 5, 'mean_probability_tolerance': 2e-15, 'mean_reference': '10/15 patient-isolated revalidation primary_raw', 'rows': len(data['y']), 'positive': int(data['y'].sum()), 'fold_reports': reports, 'predictions': identity(broken)})
    save_json(ROOT / 'serialization_recovery.json', {'passed': True, 'error': 'The original saver attempted to pass raw record_count and the record_count comparator probabilities under the same NPZ keyword. SCH models had all been saved; no WHU models had been fitted.', 'fix': 'Raw count key renamed processed_record_count in v2. Comparator probability key remains record_count. All SCH outputs rebuilt solely by replaying saved models; WHU runs the same training function with only this serialization keyword changed.', 'run_function_ast_equal_except_one_npz_keyword': True, 'no_model_refitting_for_recovery': True, 'statistical_protocol_unchanged': True, 'model_inputs_unchanged': model_inputs, 'script': identity(__file__), 'recovered_predictions': identity(broken)})
    print(json.dumps({'recovered': 'SCH', 'ensembles_replayed': 20, 'no_refitting': True}), flush=True)
if __name__ == '__main__':
    main()
