"""Replay saved common-laboratory packages and assemble aggregate-only readout."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '4'
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import xgboost as xgb
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=True, allow_nan=False)

def independent_transform(X, medians, model):
    missing = np.isnan(X)
    if model == 'missingness_only':
        return missing.astype(np.float32)
    values = X.copy()
    for j in range(values.shape[1]):
        values[missing[:, j], j] = medians[j]
    if model == 'ensemble_median_indicator':
        values = np.concatenate((values, missing), axis=1)
    return values.astype(np.float32)

def main():
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8-sig')))
    manifest = _release_resolve_config(json.loads((ROOT / 'aggregate/model_manifest.json').read_text(encoding='utf-8-sig')))
    execution = _release_resolve_config(json.loads((ROOT / 'aggregate/execution.json').read_text(encoding='utf-8-sig')))
    amended = _release_resolve_config(json.loads((ROOT / 'aggregate/evaluation_v2_verification.json').read_text(encoding='utf-8-sig')))
    lock = _release_resolve_config(json.loads((ROOT / 'analysis_lock.json').read_text(encoding='utf-8-sig')))
    assert execution['passed'] and amended['passed'] and manifest['all_inputs_unchanged']
    for item in lock['identities'] + manifest['model_packages']:
        actual = identity(item['path'])
    paths = [Path(i['path']) for i in manifest['model_packages']]
    paths += [ROOT / f'private/{c}.npz' for c in ('WHU', 'MIMIC')]
    paths += [ROOT / f'private/{c}_predictions.npz' for c in ('WHU', 'MIMIC')]
    before = [identity(p) for p in paths]
    data, predictions = ({}, {})
    for centre in ('WHU', 'MIMIC'):
        with np.load(ROOT / f'private/{centre}.npz', allow_pickle=False) as z:
            data[centre] = {k: z[k].copy() for k in z.files}
        with np.load(ROOT / f'private/{centre}_predictions.npz', allow_pickle=False) as z:
            predictions[centre] = {k: z[k].copy() for k in z.files}
        assert np.array_equal(data[centre]['y'], predictions[centre]['y'])
        assert np.array_equal(data[centre]['groups'], predictions[centre]['groups'])
    maximum_error, replayed_rows, converged = (0.0, 0, [])
    with threadpool_limits(limits=4):
        for report in manifest['reports']:
            centre, name = (report['centre'], report['model'])
            X, y, groups, roles = (data[centre][k] for k in ('X', 'y', 'groups', 'roles'))
            if report['scope'] == 'local_patient_oof':
                role = roles[report['fold'] - 1]
                fit, test = (np.flatnonzero(role != 0), np.flatnonzero(role == 0))
                inner, stop = (np.flatnonzero(role == 2), np.flatnonzero(role == 3))
                target = centre
                expected_p = predictions[target]['local_' + name][test]
                target_X = X[test]
                assert not np.intersect1d(groups[fit], groups[test]).size
            else:
                fit = np.arange(len(y))
                unique = np.unique(groups)
                any_positive = np.asarray([y[groups == g].max() for g in unique])
                inner_g, stop_g = train_test_split(unique, test_size=0.25, stratify=any_positive, random_state=protocol['seeds']['source_inner'])
                inner, stop = (np.flatnonzero(np.isin(groups, inner_g)), np.flatnonzero(np.isin(groups, stop_g)))
                target = report['target']
                target_X = data[target]['X']
                expected_p = predictions[target]['transfer_' + name]
            if report['selection'] is not None:
                assert not np.intersect1d(groups[inner], groups[stop]).size
            package = joblib.load(report['package']['path'])
            medians = pd.DataFrame(X[fit]).median().fillna(0).to_numpy()
            assert np.array_equal(medians, package['medians'])
            values = independent_transform(target_X, medians, name)
            fit_values = independent_transform(X[fit], medians, name)
            if 'xgb' in package:
                assert package['rf'].n_jobs == 4 and package['rf'].class_weight == 'balanced'
                assert package['rf'].n_features_in_ == fit_values.shape[1]
                p = 0.5 * package['xgb'].predict(xgb.DMatrix(values, nthread=4)) + 0.5 * package['rf'].predict_proba(values)[:, 1]
            else:
                assert np.allclose(package['scaler'].mean_, fit_values.mean(axis=0, dtype=np.float64), atol=1e-12, rtol=0)
                assert np.allclose(package['scaler'].var_, fit_values.astype(float).var(axis=0), atol=1e-12, rtol=1e-12)
                p = package['logistic'].predict_proba(package['scaler'].transform(values))[:, 1]
                converged.append(bool(np.all(package['logistic'].n_iter_ < protocol['logistic']['max_iter'])))
            error = float(np.max(np.abs(p - expected_p)))
            maximum_error = max(maximum_error, error)
            assert error < 2e-12
            replayed_rows += len(p)
    all_metrics = pd.concat([pd.read_csv(ROOT / 'aggregate/transfer_metrics.csv'), pd.read_csv(ROOT / 'aggregate/local_metrics.csv')], ignore_index=True)
    for _, row in all_metrics.iterrows():
        centre, model = (row.target, row.model)
        prefix = 'transfer_' if row.scope == 'source_frozen_transfer' else 'local_'
        y, p = (data[centre]['y'], predictions[centre][prefix + model])
        expected = [roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)]
        assert np.allclose(row[['roc_auc', 'average_precision', 'brier']].astype(float), expected, atol=2e-12, rtol=0)
    after = [identity(p) for p in paths]
    assert before == after and all(converged)
    write_json(ROOT / 'aggregate/saved_package_replay.json', {'passed': True, 'execution_complete': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'model_packages_replayed': len(manifest['reports']), 'probability_rows_replayed': replayed_rows, 'maximum_probability_absolute_error': maximum_error, 'original_source_fold_ids_and_source_inner_roles_verified': True, 'fit_only_medians_and_logistic_scalers_replayed': True, 'source_and_local_inner_patient_intersections_zero': True, 'all_logistic_models_converged': all(converged), 'sklearn_point_metric_sets_replayed': len(all_metrics), 'input_identities_before': before, 'input_identities_after': after, 'scope': 'Author-side replay from saved packages, not an independent laboratory or clinical adjudication; XGB/RF trees were not refitted', 'entrypoint': identity(Path(__file__))})
    previous = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1/private/prepared/WHU.npz'
    with np.load(previous, allow_pickle=False) as z:
        names = z['feature_names'].tolist()
    metadata = {'年龄', '性别', '上次诊断', '上次入院科室', '上次出院科室', '人均地区生产总值(元)', '职工平均工资(元)', '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)'}
    selected = {f['whu_exact_name']: f for f in protocol['features']}
    dictionary = []
    for name in names + [n for n in selected if n not in names]:
        spec = selected.get(name)
        reason = 'Admitted: exact raw name, explicit unit and chemistry/CBC context; complete method/platform equivalence not asserted' if spec else 'Not a laboratory measurement; excluded from the laboratory-only task' if name in metadata else 'Not admitted to the predeclared minimum subset. Further cross-source analyte, context and unit review required; not declared absent or intrinsically unusable'
        dictionary.append({'source_name_whu': name, 'historical_104_input': name in names, 'admitted': bool(spec), 'reason': reason, 'common_name': spec['name'] if spec else '', 'whu_required_literal_unit': spec['whu_unit'] if spec else '', 'mimic_exact_itemid': spec['mimic_itemid'] if spec else '', 'mimic_required_literal_unit': spec['mimic_unit'] if spec else '', 'mimic_numeric_multiplier': spec['mimic_factor_to_g_per_l'] if spec else '', 'output_unit': spec['output_unit'] if spec else ''})
    pd.DataFrame(dictionary).to_csv(ROOT / 'aggregate/admission_dictionary.csv', index=False, mode='x')
    coverage = []
    for centre, d in data.items():
        missing = np.isnan(d['X']).all(axis=1)
        coverage.append({'centre': centre, 'observations': len(d['y']), 'patients': len(np.unique(d['groups'])), 'positive_labels': int(d['y'].sum()), 'all_four_missing_observations': int(missing.sum()), 'all_four_missing_positive_labels': int(d['y'][missing].sum()), 'at_least_one_observed': int((~missing).sum()), 'at_least_one_observed_positive_labels': int(d['y'][~missing].sum())})
    pd.DataFrame(coverage).to_csv(ROOT / 'aggregate/coverage_by_label.csv', index=False, mode='x')
    primary = all_metrics[all_metrics.model.eq('ensemble_median')]
    text = ['# Limited Common-Laboratory Transfer', '', 'Completed 2026-10-16. New analysis; original manuscript, labels, cohorts and completed analyses are unchanged.', '', '## Scope', '', 'The author approved a strict WHU/MIMIC four-laboratory subset after the SCH unit-source gate failed. Albumin, total protein, routine CBC hemoglobin and routine CBC hematocrit use exact raw names/ITEMIDs, chemistry/CBC report context and explicit physical units. MIMIC g/dL values are multiplied by ten to g/L; hematocrit remains percent. This is limited name/context/unit review, not full specimen, method or platform equivalence or OMOP.', '', 'Original denominators and labels are retained: WHU 2,562 patients/281 positives; MIMIC 9,946 admissions from 6,783 patients/4,462 positives. All-missing rows remain present. WHU patient-level and MIMIC admission-level source scopes and diagnostic labels remain different. Full-record medians are retrospective and do not establish pre-event timing, clinician result availability or clinical warning lead time.', '', '## Primary Results', '', '| Training source | Evaluation target | Scope | AUC (95% conditional CI) | AP | Brier |', '|---|---|---|---|---|---|']
    for _, row in primary.iterrows():
        text.append(f'| {row.source} | {row.target} | {row.scope} | {row.roc_auc:.4f} ({row.roc_auc_ci_low:.4f}-{row.roc_auc_ci_high:.4f}) | {row.average_precision:.4f} | {row.brier:.4f} |')
    text += ['', 'Both executed laboratory transfer directions and same-input local patient-OOF references are newly trained. The primary model is the prespecified 50:50 XGBoost/random-forest ensemble with fitting-only median imputation and no missingness indicators. Indicator ensembles, value-based logistic and four-lab missingness-only logistic are separate controls, not alternative primary models selected by test performance.', '', 'The limited inputs also show weak local discrimination, especially in MIMIC. Lower transfer performance cannot isolate a care-architecture or semantic-drift mechanism. Source endpoint definitions, ascertainment, survivor selection, source coverage and residual assay-method differences remain competing explanations.', '', '## Evidence Boundaries', '', '- The four directions involving SCH are explicitly not estimated because source-certified SCH laboratory units remain unavailable. No demographics-only analysis substitutes for them.', '- Total protein was observed in only 795/9,946 MIMIC admissions (92.0% missing). WHU source omissions remain missing rather than being assigned invented measurements. Detailed coverage, including labels among all-missing rows, is in aggregate/coverage_by_label.csv.', '- Full-source transfer models use all source observations; local references use five patient-isolated outer folds (about 80% local training). Their paired difference is descriptive, not a pure causal transport penalty.', '- Models use class-weighted training and no new calibrator. Brier scores assess these raw scores; no validated clinical probability or action threshold is claimed.', '- Intervals use 2,000 target-patient cluster draws conditional on fixed models/matrices, retaining all MIMIC admissions per patient. They omit source-training and model-refit uncertainty and are nominal without multiplicity adjustment.', '- The evaluation-only integer-weight dtype failure and precise repair are retained in aggregate/evaluation_attempt1_failure.json and evaluate_common_v2.py. Locked fitting code, predictions and seeds did not change. Both floating and integer weight tests compare against sklearn.', '', '## Entry Points', '', 'Run prepare_common.py only in a new empty output branch; it will refuse overwrites. run_common.py train is the frozen fitting entrypoint. Use evaluate_common_v2.py, not the preserved initial evaluate action, for completed interval calculation. finalize_common.py performs author-side package/median/scaler/probability and point-metric replay, without tree refitting. All package replays passed; an external verifier must be distinguished from these author-side checks.', '', 'Public aggregate outputs are in aggregate/. Matrices, folds, predictions and 48 model packages remain in private/ and are not submission attachments.']
    with (ROOT / 'README.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(text) + '\n')
    outputs = [p for p in sorted((ROOT / 'aggregate').glob('*')) if p.is_file()]
    outputs += [ROOT / 'README.md', ROOT / 'protocol.json', ROOT / 'analysis_lock.json', _release_source_path(ROOT / 'prepare_common.py'), _release_source_path(ROOT / 'run_common.py'), _release_source_path(ROOT / 'evaluate_common_v2.py'), Path(__file__)]
    write_json(ROOT / 'aggregate/completion_manifest.json', {'execution_complete': True, 'passed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'outputs': [identity(p) for p in outputs], 'private_patient_rows_or_identifiers_in_public_outputs': False, 'source_gate_limits': 'Only WHU/MIMIC four-laboratory subset executed; SCH four directions not estimated', 'author_side_checks_complete': True, 'independent_verification_included': False})
    print(json.dumps({'passed': True, 'saved_packages_replayed': len(manifest['reports']), 'probability_rows_replayed': replayed_rows, 'max_probability_error': maximum_error, 'point_metric_sets_replayed': len(all_metrics), 'all_model_packages_unchanged': before == after}))
if __name__ == '__main__':
    main()
