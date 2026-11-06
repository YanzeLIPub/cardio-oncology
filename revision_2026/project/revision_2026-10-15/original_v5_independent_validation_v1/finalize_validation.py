"""Bind final aggregate deliverables to verified analysis and visual evidence."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]

def sha(path):
    return None

def identity(path):
    path = Path(path)
    return {'path': path.as_posix(), 'bytes': path.stat().st_size}

def load(relative):
    return _release_resolve_config(json.loads((ROOT / relative).read_text(encoding='utf-8')))

def write(relative, data):
    with (ROOT / relative).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def verify_identity(record):
    if 'bytes' in record:
        assert Path(record['path']).stat().st_size == record['bytes']

def main():
    for name in ('workflow_snapshot.json', 'finish_identity.json', 'private/private_manifest.json'):
        if (ROOT / name).exists():
            raise FileExistsError('Final evidence exists; use a new version instead of replacing it')
    lock = load('evidence/analysis_lock.json')
    for record in lock['state']['files']:
        verify_identity(record)
    summary = load('results/validation_summary.json')
    replay = load('evidence/validation_replay_audit.json')
    methods = load('evidence/methods_code_review.json')
    content = load('evidence/report_content_audit.json')
    content_review = load('evidence/report_independent_content_review.json')
    figures = load('qa/figures_visual_review.json')
    document = load('qa/document_visual_review.json')
    assert summary['passed'] and summary['execution_complete']
    assert replay['passed'] and replay['status'] == 'completed_saved_model_replay'
    assert methods['passed'] and (not methods['blocking_findings_remaining'])
    assert content['passed'] and content['text_and_numeric_checks_passed']
    assert content_review['passed'] and (not content_review['blocking_findings_remaining'])
    assert figures['passed'] and document['passed'] and document['all_pages_visually_inspected']
    for record in summary['input_identities'] + summary['outputs'] + methods['source_identities'] + content['source_identities'] + content_review['source_identities']:
        verify_identity(record)
    for record in figures['figures']:
        pass
    docx = ROOT / 'Supplementary_Patient_Isolated_Validation.docx'
    verify_identity(content['document'])
    verify_identity(content_review['document'])
    render = load(document['render_manifest'])
    assert len(document['page_images']) == render['page_count']
    for record in document['page_images']:
        pass
    source_snapshot = _release_resolve_config(json.loads((PROJECT / 'workflow_snapshot.json').read_text(encoding='utf-8')))
    source_identities = []
    for group in ('artifacts', 'analysis_inputs', 'analysis_scripts'):
        for record in source_snapshot[group]:
            path = Path(record['path'])
            if not path.is_absolute():
                path = PROJECT / path
            current = identity(path)
            source_identities.append(current)
    with (ROOT / 'results/validation_metrics.csv').open(encoding='utf-8', newline='') as stream:
        metrics = list(csv.DictReader(stream))
    primary = [r for r in metrics if r['metric'] == 'auc' and r['model'] in ('original', 'primary_raw')]
    private_files = [identity(p) for p in sorted((ROOT / 'private').rglob('*')) if p.is_file()]
    write('private/private_manifest.json', {'distribution': 'Local only; includes clinical arrays and fitted models', 'files': private_files})
    workflow = {'schema_version': 1, 'completed_at': datetime.now().astimezone().isoformat(), 'route': 'original_analysis_plus_separately_reported_patient_isolated_internal_revalidation_no_OMOP', 'journal': 'npj Cardiovascular Health', 'analysis_status': 'four_runs_twenty_outer_folds_complete_independently_replayed_and_summarized', 'new_external_cohort_validation': False, 'manuscript_baseline_replaced': False, 'complete_reviewer_response_regenerated': False, 'original_analysis_preserved': True, 'no_original_files_modified': True, 'original_snapshot_count': len(source_identities), 'original_snapshot_verified': source_identities, 'main_deliverable': identity(docx), 'document_visual_QA': identity(ROOT / 'qa/document_visual_review.json'), 'figure_visual_QA': identity(ROOT / 'qa/figures_visual_review.json'), 'method_review': identity(ROOT / 'evidence/methods_code_review.json'), 'document_numeric_audit': identity(ROOT / 'evidence/report_content_audit.json'), 'document_independent_content_review': identity(ROOT / 'evidence/report_independent_content_review.json'), 'independent_model_replay': identity(ROOT / 'evidence/validation_replay_audit.json'), 'summary': identity(ROOT / 'results/validation_summary.json'), 'protocol_lock': identity(ROOT / 'evidence/analysis_lock.json'), 'private_manifest': identity(ROOT / 'private/private_manifest.json'), 'aggregate_primary_AUC_results': primary, 'scope_limits': ['Historically used data; this is not a new external test cohort.', 'Original labels and full-record timing restrictions remain unresolved by split correction.', 'Old and new analysis results are separately identified; no OMOP alignment was performed.', 'Model and calibration comparisons do not establish clinical utility or a causal semantic-drift mechanism.', 'Supplementary validation report is complete; it is not the entire revised manuscript and verbatim response package.']}
    write('workflow_snapshot.json', workflow)
    public_files = [identity(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and 'private' not in p.relative_to(ROOT).parts and ('__pycache__' not in p.parts)]
    write('finish_identity.json', {'captured_at': datetime.now().astimezone().isoformat(), 'public_artifacts': public_files, 'private_manifest': identity(ROOT / 'private/private_manifest.json')})
    final = load('finish_identity.json')
    for record in final['public_artifacts']:
        verify_identity(record)
    for record in private_files:
        verify_identity(record)
    print(json.dumps({'finalized': True, 'original_files_unchanged': len(source_identities), 'public_artifacts_verified': len(public_files), 'private_artifacts_verified': len(private_files), 'document_pages_visually_checked': render['page_count']}))
if __name__ == '__main__':
    main()
