"""Fresh SCH source identity/schema replay and independent NJM branch binding."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import collections
import csv
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = Path(_release_path('project', ''))
RAW = Path(_release_path('njm', 'SCH'))
PRIOR = ROOT / 'revision_2026-10-15/temporal_boundary_followup_v1/sch/sch_temporal_boundary_audit.json'
BRANCH = Path(_release_path('njm', 'multicenter_manuscript_output/analysis/reviewer_followup/lab_feature_alignment_v1_20260814'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def identity(path):
    path = Path(path)
    return {'path': path.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def main():
    target = HERE / 'sch_fresh_units_and_timing_gate.json'
    if target.exists():
        raise RuntimeError('Refusing to replace completed source audit')
    prior = _release_resolve_config(json.loads(PRIOR.read_text(encoding='utf-8')))
    expected = collections.Counter(((x['bytes'], x['sha256']) for x in prior['source_identity_checks']))
    current, headers = ([], [])
    for ordinal, path in enumerate(sorted((p for p in RAW.rglob('*') if p.is_file())), 1):
        item = {'ordinal': ordinal, 'bytes': path.stat().st_size, 'sha256': sha(path), 'suffix': path.suffix.lower()}
        current.append(item)
        with path.open('rb') as stream:
            prefix = stream.read(8)
        if path.suffix.lower() not in {'.csv', '.xls', '.tsv', '.txt'} or prefix.startswith((b'\xd0\xcf', b'\t\x00', b'\t\x02', b'\t\x04', b'\t\x08')):
            continue
        with path.open('rb') as stream:
            line = stream.readline()
        try:
            text = line.decode('utf-8-sig')
        except UnicodeError:
            text = line.decode('gb18030')
        delimiter = '\t' if '\t' in text else ','
        names = next(csv.reader([text], delimiter=delimiter))
        headers.append({'source_ordinal': ordinal, 'columns': len(names), 'header_sha256': hashlib.sha256(json.dumps(names, ensure_ascii=False).encode()).hexdigest(), 'unit_fields': [n for n in names if re.fullmatch('(?:检验|化验|检查|结果|项目)?单位|units?|valueuom', n.strip(), re.I)], 'onset_fields': [n for n in names if re.search('发病|起病|发生时间|onset', n, re.I)], 'first_result_availability_fields': [n for n in names if re.search('结果可用|结果可见|发布时间|释放时间|release.?time|storetime', n, re.I)]})
    assert collections.Counter(((x['bytes'], x['sha256']) for x in current)) == expected
    processed = RAW / 'SCH_processed_data.csv'
    columns = pd.read_csv(processed, nrows=0).columns.tolist()
    candidates = [n for n in columns if re.fullmatch('(?:\\u2605)?(?:白蛋白|总蛋白|血红蛋白|红细胞比容|红细胞压积)', n)]
    selected = pd.read_csv(processed, usecols=candidates)
    stats = []
    for col in candidates:
        numeric = pd.to_numeric(selected[col], errors='coerce')
        stats.append({'field': col, 'record_rows': len(selected), 'numeric_nonmissing': int(numeric.notna().sum()), 'negative_numeric_rows': int((numeric < 0).sum()), 'source_unit': None, 'cross_centre_numeric_eligible': False, 'interpretation': 'Column name alone does not identify specimen/platform/unit or reverse an unknown processed scale.'})
    branch_manifest_path = BRANCH / 'manifest.json'
    branch_manifest = _release_resolve_config(json.loads(branch_manifest_path.read_text(encoding='utf-8')))
    branch_input = next((x for x in branch_manifest['inputs'] if Path(x['path']).name == processed.name))
    assert branch_input['sha256'] == sha(processed)
    contract_path = BRANCH / 'lab_alignment_contract.csv'
    contract = pd.read_csv(contract_path)
    centre = next((c for c in contract.columns if c.lower() in {'centre', 'center'}))
    sch_contract = contract.loc[contract[centre].astype(str).str.upper() == 'SCH'].fillna('')
    whitelist = [c for c in contract.columns if not re.search('patient|encounter|token', c, re.I)]
    rows = sch_contract[whitelist].to_dict('records')
    report = {'passed': True, 'raw_source_files_freshly_hashed': len(current), 'raw_source_bytes_freshly_hashed': sum((x['bytes'] for x in current)), 'all_sources_unchanged_since_prior_all_sheet_audit': True, 'raw_source_identities': current, 'fresh_text_schema_checks': headers, 'fresh_processed_assay_column_checks': stats, 'prior_binary_sheet_interpretation_replayed': False, 'prior_binary_evidence_reused_only_after_full_source_hash_equality': True, 'source_bound_lab_unit_evidence_recovered': False, 'independent_NJM_branch_uses_same_processed_input': True, 'independent_NJM_branch_SCH_contract': rows, 'prior_admission_POA_bounds': prior['bounded_component_evidence'], 'prior_complete_capture_gate': prior['population_and_capture'], 'new_clinical_labels': 0, 'models_trained': 0, 'inputs': [identity(PRIOR), identity(processed), identity(branch_manifest_path), identity(contract_path), identity(BRANCH / 'analysis_report.md'), identity(BRANCH / 'summary.json')], 'script': identity(__file__), 'environment': {'python': sys.version, 'executable': sys.executable, 'pandas': pd.__version__, 'platform': platform.platform()}, 'originals_written': False, 'clinical_rows_exported': False}
    with target.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
        f.write('\n')
    print(json.dumps({'passed': True, 'raw_files_hash_checked': len(current), 'text_headers_reread': len(headers), 'candidate_assay_columns_checked': len(candidates), 'source_unit_gate_passed': False, 'crosscentre_numeric_models_trained': 0}))
if __name__ == '__main__':
    main()
