"""Bind aggregate evidence and deny unsupported SCH quantitative transfers."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
import platform
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PRIOR = Path(_release_path('project', 'revision_2026-10-19/substantive_gaps_v1/analysis/timing_units/sch_fresh_units_and_timing_gate.json'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(path):
    return {'path': path.as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)}

def main():
    audit_path = HERE / 'sch_assay_metadata_scan.json'
    lineage_path = HERE / 'scale_lineage_scan.json'
    tests_path = HERE / 'tests.json'
    processed_identity_path = HERE.parent / 'timing/sch_processed_fresh_identity.json'
    scan = _release_resolve_config(json.loads(audit_path.read_text(encoding='utf-8')))
    lineage = _release_resolve_config(json.loads(lineage_path.read_text(encoding='utf-8')))
    tests = _release_resolve_config(json.loads(tests_path.read_text(encoding='utf-8')))
    prior = _release_resolve_config(json.loads(PRIOR.read_text(encoding='utf-8')))
    assert tests['passed'] and tests['scan_report_sha256'] == digest(audit_path)
    assert all((p['matches_prior_sha'] for p in scan['source_files'] if 'matches_prior_sha' in p))
    unchanged_stat = []
    for src in scan['source_files']:
        stat = Path(src['path']).stat()
        same = stat.st_size == src['bytes'] and stat.st_mtime_ns == src['mtime_ns']
        unchanged_stat.append({'path': src['path'], 'unchanged_bytes_and_mtime_since_scan': same})
    assert all((x['unchanged_bytes_and_mtime_since_scan'] for x in unchanged_stat))
    assert len(scan['source_files']) == 34
    assert scan['source_summary']['exact_assay_label_cells'] == {}
    assert scan['source_summary']['assay_and_unit_same_cell_counts'] == {}
    assert lineage['same_bytes_as_NJM_latest_identity']
    assert lineage['processed_copies_equal_after_CRLF_to_LF_only']
    required = [{'evidence': 'Source assay identity', 'minimum': 'SCH LIS item code, exact item name, specimen, valid date range and a mapping to each current processed column; distinguish starred and unstarred variants.', 'status': 'not_recovered'}, {'evidence': 'Physical numeric units and scale', 'minimum': 'Per-result source units and untransformed values, or an authenticated original transform plus feature order, fitted parameters and input/output hash binding.', 'status': 'not_recovered'}, {'evidence': 'Patient/admission/source binding', 'minimum': 'Stable study-authorized join keys to current SCH cohort and assay results. An unrelated historical worksheet or similarly named assay is insufficient.', 'status': 'not_recovered_for_raw_lab_results'}, {'evidence': 'Comparable extraction and target', 'minimum': 'Protocol specifying analysis unit, cancer inclusion, diagnosis-label rule and laboratory extraction window for each centre. This gate does not certify a uniform future-risk task.', 'status': 'not_established_by_unit_audit'}]
    assays = []
    names = {'albumin': ['★白蛋白', '白蛋白'], 'total_protein': ['★总蛋白', '总蛋白'], 'hemoglobin': ['★血红蛋白', '血红蛋白'], 'hematocrit': ['★红细胞压积', '红细胞压积']}
    oldcounts = {x['field']: x for x in prior['fresh_processed_assay_column_checks']}
    for canonical, fields in names.items():
        assays.append({'canonical': canonical, 'source_column_candidates_not_verified_mapping': fields, 'physical_source_unit': None, 'verified_original_scale': None, 'authenticated_inverse_transform_parameters': None, 'eligible_for_common_numeric_transfer': False, 'prior_aggregate_counts_not_recomputed_this_run': [oldcounts[f] for f in fields]})
    directions = [{'source': source, 'target': target, 'status': 'not_estimable_required_source_evidence_missing', 'numeric_performance': None, 'confidence_interval': None, 'model_fit_permitted': False, 'model_fitted': False, 'reason': 'Current SCH assay identity, physical units and original numeric scale are not authenticated.', 'does_not_reuse_or_relabel_historical_AUC': True} for source, target in [('SCH', 'WHU'), ('WHU', 'SCH'), ('SCH', 'MIMIC'), ('MIMIC', 'SCH')]]
    result = {'execution_status': 'completed_source_audit', 'four_requested_quantitative_transfer_experiments_completed': False, 'provenance': [identity(p) for p in (audit_path, lineage_path, tests_path, PRIOR)], 'source_coverage': scan['source_summary'], 'fresh_hash_scope': {'SCH_sources_total': 34, 'freshly_hashed_by_assay_scanner': 33, 'freshly_hashed_sources_all_match_prior': True, 'processed_csv_in_assay_scanner': 'Prior full SHA256 reused; current size, mtime and header were read. Do not describe this script as independently rehashing all 34 sources.', 'processed_csv_independent_current_full_hash': 'Timing agent has independently rehashed this source; root final verification should bind the timing certificate together with this gate.', 'source_stat_after_checks': unchanged_stat}, 'binary_assay_word_interpretation': {'albumin_word_hits': 6, 'albumin_product_fee_headers': 5, 'remaining_hit_role': '其他诊断2', 'exact_assay_labels': 0, 'assay_plus_physical_unit_same_cell': 0, 'no_linked_source_LIS_results_or_unit_dictionary_recovered': True, 'builtin_percent_formats_do_not_define_hematocrit_units': True}, 'historical_current_version_binding': {'original_loader_recovered': _release_path('china', 'load_processed_csv.py'), 'historical_processed_copies': lineage['original_processed_source_copies'], 'same_data_after_only_CRLF_normalization': True, 'earlier_binary_same_stay_linked_rows_from_prior_verified_audit': prior['prior_complete_capture_gate']['old_binary_same_stay_linked_rows'], 'old_binary_rows_not_used_to_certify_current_column_units': True}, 'code_scope': {'NJM_script_count': scan['code_scan']['scanned_script_count'], 'NJM_source_linked_script_count': len(scan['code_scan']['processed_source_linked_scripts']), 'earlier_direct_script_count': lineage['script_count'], 'earlier_source_linked_script_count': len(lineage['source_linked_scripts']), 'interpretation': 'Recovered normalization is downstream model preprocessing or identifier normalization, not an authenticated producer/inverse of the supplied SCH wide table.', 'no_pickle_or_original_module_executed': True, 'bounded_search_not_a_claim_of_no_other_hospital_records_existing': True}, 'assay_gates': assays, 'directions': directions, 'minimum_required_source_evidence': required, 'forbidden_shortcuts': ['Infer units, standardization, or inverse transform from negative values or numeric distributions.', 'Treat WTH or WHU dictionaries as SCH source dictionaries.', 'Apply target-centre scaling to disguise unknown measurement scales.', 'Put new model confidence intervals beside historical AUC as if they came from original probabilities.'], 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}, 'script': identity(Path(__file__)), 'no_originals_written': True, 'patient_rows_exported': False, 'new_clinical_labels': 0, 'models_trained': 0}
    if processed_identity_path.exists():
        certificate = _release_resolve_config(json.loads(processed_identity_path.read_text(encoding='utf-8')))
        assert certificate['fresh_full_file_hash_completed'] is True
        assert certificate['expected_identity_matched'] is True
        assert digest(Path(certificate['script']['path'])) == certificate['script']['sha256']
        result['independent_processed_identity_certificate'] = identity(processed_identity_path)
        result['fresh_hash_scope']['processed_csv_independent_current_full_hash'] = 'Timing audit independently rehashed the processed CSV this turn. See independent_processed_identity_certificate. The assay scanner itself freshly hashed 33 sources.'
    out = HERE / 'sch_transfer_gate.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'gate': 'four_directions_not_estimable', 'passed_evidence_checks': True, 'sha256': digest(out), 'path': out.as_posix()}))
if __name__ == '__main__':
    main()
