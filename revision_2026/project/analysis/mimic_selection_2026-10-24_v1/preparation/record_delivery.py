"""Record terminal evidence only after successful independent preparation replay."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main(chunk, session):
    assert chunk.isascii() and chunk.isalnum()
    report_path = HERE / 'preparation_report.json'
    independent_path = HERE.parent / 'design/independent_preparation_verification.json'
    output, readout = (HERE / 'preparation_delivery.json', HERE / 'README.md')
    assert not output.exists() and (not readout.exists())
    report = _release_resolve_config(json.loads(report_path.read_text(encoding='utf-8')))
    independent = _release_resolve_config(json.loads(independent_path.read_text(encoding='utf-8')))
    assert report['passed'] and report['execution_complete'] and independent['passed']
    assert independent['checks_passed'] == len(independent['checks'])
    assert all((item['passed'] for item in independent['checks']))
    assert independent['producer_report'] == identity(report_path)
    assert independent['matrix_identity'] == identity(HERE / 'private/full_matrix.npz')
    assert independent['producer_code'] == report['script']
    assert independent['sources_changed'] is False and independent['models_trained'] is False
    required = {'independent NumPy medians and derived globulin match every historical cell', 'independent NumPy reconstruction matches every primary cell', 'all labels independently reconstructed', 'old frozen survivor matrix exactly reproduced'}
    assert required <= {item['name'] for item in independent['checks']}
    for arm, counts in independent['cohorts'].items():
        assert all((report['cohorts'][arm][key] == value for key, value in counts.items()))
    protected = {}
    for item in list(report['inputs'].values()) + report['outputs'] + [report['script'], independent['script']]:
        actual = identity(item['path'])
        protected[actual['path']] = actual
    lines = ['# MIMIC Selection-Sensitivity Data Preparation', '', 'Full source preparation and independent raw chronology/diagnosis/demographic plus NumPy laboratory replay completed. No models were trained in this stage.', '', '| Arm | Admissions | Patients | Original Label 1 | Death Admissions | Non-Death Code Union | Death-Proxy-Only Label 1 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ('ever_all', 'ever_survivor', 'prior_all', 'prior_survivor'):
        counts = report['cohorts'][arm]
        fields = ('admissions', 'patients', 'positive_source_labels', 'death_admissions', 'source_coded_component_union_positive', 'source_death_proxy_only_positive')
        lines.append('| ' + arm + ' | ' + ' | '.join((format(counts[name], ',') for name in fields)) + ' |')
    lines += ['', '## Data Interface', '', '- `private/full_matrix.npz`: 11,622 x 41 float32 `X`; original integer `y`; globally factorized patient `groups`; private source-admission `keys`; `feature_names`; survivor, prior-history, current-quality, four arm masks and source label-component flags.', '- `prior_completed_valid` requires both diagnosis families in valid prior source stays completed strictly before index admission. `valid_current_admission` is report-only, never an extra arm filter.', '- Invalid current rows retained: 21 ever-all, 4 ever-survivor, 3 prior-all and 0 prior-survivor. The corrected prior-survivor denominator is 2,194, not the superseded 2,199.', '- `private/historical43_audit.npz` is the original 43-column audit matrix only; do not use it as the new predictor input. AST and HDL Cholesterol are removed uniformly from every primary arm.', '- `private/row_metadata.csv.gz` contains source identifiers, clocks, group codes and flags. Keep it local. Matrix and metadata order follows original full features_model.csv; its 9,946-survivor slice exactly matches the frozen paper/preparation.', '- All 24 cached laboratory columns including derived Globulin were replayed for the full source cohort, and 23,244 raw demographic cells were checked. The 171 admissions without cached labs remain as missing values, not excluded.', '- No imputation, scaling, feature selection, fitting, threshold selection or new item/unit conversion occurs here. Fit-dependent processing belongs only in the separately frozen training protocol.', '', '## Interpretation Limits', '', '- The universe is original ever-cancer-and-ever-broad-cardiac source patients, not an independently defined all-oncology cohort.', '- Labels are unchanged concurrent source diagnosis composites. Death plus a broad cardiac diagnosis is a historical proxy, not verified cardiovascular death. Source non-death code-union flags are provided separately, not substituted as the primary label.', '- Admission-linked medians are time-unrestricted, including source rows outside recorded admission bounds. No incident-event onset, prediction lead time, event-free baseline or clinician result availability is reconstructed.', '- Dropping AST/HDL avoids the two verified column problems but does not validate every remaining item or unit. CK/CK-MB mixing and unitless cache limitations remain explicit.', '- Independent replay authenticates these source-defined data, not clinical truth, complete negative capture, a model result or absence of indirect leakage.', '', 'Evidence: `preparation_report.json`, `preparation_contract.json`, `preparation_delivery.json`, and `../design/independent_preparation_verification.json`.', 'The synthetic preflight pandas empty-datetime-mapper correction is preserved separately. Original source files and historical analyses are unchanged.', '']
    with readout.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    independent_terminal = {'purpose': 'independent_full_preparation_verification', 'chunk_id': chunk, 'exit_code': 0}
    if session is not None:
        assert session > 0
        independent_terminal['session_id'] = session
    result = {'execution_complete': True, 'data_preparation_complete': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'cohorts': report['cohorts'], 'producer_report': identity(report_path), 'independent_verification': identity(independent_path), 'script': identity(Path(__file__)), 'readout': identity(readout), 'additional_provenance': [identity(HERE / name) for name in ('write_provenance_brief.py', 'provenance_brief.json', 'PROVENANCE_BRIEF.md', 'synthetic_preflight_correction.json')], 'protected_identities': list(protected.values()), 'all_required_sessions_terminal': True, 'terminal_execution_evidence': [{'purpose': 'synthetic_preflight_before_correction', 'chunk_id': '43ea40', 'exit_code': 1, 'failure_preserved': True}, {'purpose': 'synthetic_diagnostic_only', 'session_id': 53419, 'chunk_id': '428c17', 'exit_code': 1, 'failure_preserved': True}, {'purpose': 'final_16_synthetic_tests', 'chunk_id': '1dfcf2', 'exit_code': 0}, {'purpose': 'actual_full11622_preparation', 'session_id': 32680, 'chunk_id': '577a43', 'exit_code': 0}, independent_terminal], 'model_training_performed_in_preparation': False, 'clinical_endpoint_validation_claimed': False, 'original_files_modified': False, 'private_rows_or_identifiers_disclosed': False}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2, allow_nan=False)
    print(json.dumps({'preparation_delivery_complete': True, 'admissions': 11622, 'primary_features': 41, 'prior_survivor_admissions': 2194, 'models_trained': False}))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verification-chunk', required=True)
    parser.add_argument('--verification-session', type=int)
    args = parser.parse_args()
    main(args.verification_chunk, args.verification_session)
