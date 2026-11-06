"""Read-only temporal gate audit; emit aggregate evidence only."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import collections
import csv
import gzip
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROJECT = Path(_release_path('project', ''))
OLD = PROJECT / 'revision_2026-10-12/njm_same_source_completion_2026-10-12'
PRIOR = PROJECT / 'revision_2026-10-19/substantive_gaps_v1/analysis/timing_units'
RAW_WHU = Path(_release_path('njm', 'WUH'))
RAW_MIMIC = Path(_release_path('mimic', ''))
PAPER = Path(_release_path('pipeline', 'Manuscript/v13/paper_predictions_9946.csv'))
OUTPUT = HERE / 'temporal_gate_execution.json'
PATTERNS = {'onset': re.compile('发病(?:时间|日期|时刻)|发生(?:时间|日期|时刻)|起病(?:时间|日期|时刻)|onset|event.?time', re.I), 'first_result_available': re.compile('发布(?:时间|日期)|释放(?:时间|日期)|结果(?:可见|可用|返回)(?:时间|日期)|首次.*(?:可见|可用)|release.?time|available.?time', re.I), 'recorded_report_review': re.compile('(?:报告|审核|检验|检查|采样|收样|录入|诊断)(?:时间|日期|时刻)|report.?time|storetime', re.I), 'capture_contract': re.compile('诊断.*(?:总数|全部|总条|完整)|(?:全部|完整).*诊断|diagnos.*complete', re.I), 'poa': re.compile('入院病情|present.?on.?admission|^poa$', re.I)}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def assert_identity(path, record):
    actual = identity(path)
    assert actual['sha256'] == record['sha256']
    if 'bytes' in record:
        assert actual['bytes'] == record['bytes']
    return actual

def context(key, previous):
    if re.search('检验|化验|laboratory|^lab', key, re.I):
        return 'laboratory'
    if re.search('检查|exam|radiolog', key, re.I):
        return 'examination'
    if re.search('诊断|病案首页|diagnos', key, re.I):
        return 'diagnosis'
    return previous

def whu_source_check():
    binding = load(OLD / 'labs/whu_lab_binding.json')
    contract = load(OLD / 'whu/diagnosis_capture_contract_v2.json')
    index_path = OLD / 'labs/private/source_index.json'
    capture_path = OLD / 'whu/private/diagnosis_capture_record_audit_v2.jsonl'
    index_identity = assert_identity(index_path, contract['private_locator_identity'])
    capture_identity = assert_identity(capture_path, contract['private_output'])
    index = load(index_path)
    raw_current = list(RAW_WHU.glob('*.json'))
    old_by_name = {row['basename']: row for row in index}
    current_names = {p.name for p in raw_current}
    census = {'current_json_files': len(raw_current), 'prior_json_files': len(index), 'new_names': len(current_names - set(old_by_name)), 'missing_names': len(set(old_by_name) - current_names), 'common_size_changed': sum((p.stat().st_size != old_by_name[p.name]['bytes'] for p in raw_current if p.name in old_by_name)), 'all_unselected_files_freshly_hashed': False}
    targets = collections.defaultdict(collections.Counter)
    patients, stays = (set(), set())
    with capture_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = _release_resolve_config(json.loads(line))
            targets[row['source_content_sha256']][row['record_ordinal']] += 1
            patients.add(row['patient_token'])
            stays.add(row['stay_candidate_token'])
    locators = {}
    for row in index:
        if row['sha256'] in targets:
            locators.setdefault(row['sha256'], row)
    assert set(locators) == set(targets)
    counts, families = (collections.Counter(), collections.Counter())
    signature = hashlib.sha256()
    for digest, row in sorted(locators.items()):
        path = RAW_WHU / row['basename']
        assert path.resolve().parent == RAW_WHU.resolve()
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest and len(data) == row['bytes']
        signature.update(bytes.fromhex(digest))
        obj = _release_resolve_config(json.loads(data.decode('utf-8-sig')))
        if isinstance(obj, str):
            obj = _release_resolve_config(json.loads(obj))
        records = [obj] if isinstance(obj, dict) else [x for x in obj if isinstance(x, dict)]
        counts['raw_contents'] += 1
        counts['raw_bytes'] += len(data)
        for ordinal, multiplier in targets[digest].items():
            counts['record_occurrences'] += multiplier
            stack = [(records[ordinal], 'unspecified', 0)]
            while stack:
                node, role, depth = stack.pop()
                assert depth < 100
                if isinstance(node, dict):
                    for key, value in node.items():
                        counts['dictionary_key_occurrences'] += multiplier
                        if not isinstance(key, str):
                            continue
                        new_role = context(key, role)
                        for family, pattern in PATTERNS.items():
                            if pattern.search(key):
                                nonempty = value not in (None, '', [], {})
                                families[family, new_role, nonempty] += multiplier
                        stack.append((value, new_role, depth + 1))
                elif isinstance(node, list):
                    stack.extend(((value, role, depth + 1) for value in node))
                elif isinstance(node, str) and node.lstrip().startswith(('[', '{')):
                    try:
                        recovered = _release_resolve_config(json.loads(node))
                    except (ValueError, RecursionError):
                        counts['jsonlike_strings_not_recovered'] += multiplier
                    else:
                        if isinstance(recovered, (dict, list)):
                            counts['strict_nested_json_recovered'] += multiplier
                            stack.append((recovered, role, depth + 1))
    assert counts['raw_contents'] == 2329
    assert counts['record_occurrences'] == 5744
    assert counts['dictionary_key_occurrences'] == 18897219
    assert len(patients) == 2246 and len(stays) == 5415
    prior_traversal = load(PRIOR / 'nested_literal_recovery.json')
    assert signature.hexdigest() == prior_traversal['raw_source_hash_sequence']
    return {'source_census': census, 'frozen_private_inputs': [index_identity, capture_identity], 'counts': dict(counts), 'original_cohort_patients': 2562, 'linked_patients': len(patients), 'linked_stay_day_keys': len(stays), 'source_content_sequence_sha256': signature.hexdigest(), 'same_as_prior_linked_source_contents': True, 'field_family_occurrences': [{'family': family, 'context': role, 'nonempty': nonempty, 'occurrences': n} for (family, role, nonempty), n in sorted(families.items())], 'explicit_onset_fields': sum((n for (f, _, _), n in families.items() if f == 'onset')), 'explicit_first_result_available_fields': sum((n for (f, _, _), n in families.items() if f == 'first_result_available')), 'explicit_capture_contract_fields': sum((n for (f, _, _), n in families.items() if f == 'capture_contract')), 'note': 'No raw filenames, record values or patient tokens are exported. This is a fresh linked-record schema scan, not clinical free-text onset adjudication. Previously failed non-JSON strings remain uninterpreted; the prior AST-bounded literal scan recovered no additional container.', 'prior_lab_report_evidence': {'report_clock_counts': binding['report_clock_counts'], 'meaning': 'Same-object report timestamps exist but have no certified first clinician-visible/release contract. Examination review clocks are not laboratory clocks.'}}

def mimic_source_check():
    checksum_path = RAW_MIMIC / 'SHA256SUMS.txt'
    checksums = {}
    for line in checksum_path.read_text().splitlines():
        digest, name = line.split()
        checksums[name.lstrip('*')] = digest
    tables = {}
    for path in sorted(RAW_MIMIC.glob('*.csv.gz')):
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
            columns = next(csv.reader(stream))
        tables[path.name.removesuffix('.csv.gz')] = {'columns': columns, 'bytes': path.stat().st_size, 'header_sha256': hashlib.sha256(json.dumps(columns).encode()).hexdigest(), 'manifest_sha256': checksums.get(path.name), 'full_file_fresh_hash': False}
    for name in ('ADMISSIONS', 'DIAGNOSES_ICD', 'D_ITEMS', 'D_LABITEMS'):
        path = RAW_MIMIC / (name + '.csv.gz')
        tables[name]['sha256'] = sha(path)
        tables[name]['full_file_fresh_hash'] = True
        assert tables[name]['sha256'] == checksums[path.name]
    assert tables['DIAGNOSES_ICD']['columns'] == ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']
    assert 'STORETIME' not in tables['LABEVENTS']['columns']
    paper = {}
    with PAPER.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['HADM_ID']
            assert key not in paper
            paper[key] = (row['SUBJECT_ID'], int(float(row['MACE'])))
    assert len(paper) == 9946
    assert len({v[0] for v in paper.values()}) == 6783
    assert sum((v[1] for v in paper.values())) == 4462
    admissions = {}
    counts = collections.Counter()
    windows = {h: collections.Counter() for h in (0, 24, 48)}
    with gzip.open(RAW_MIMIC / 'ADMISSIONS.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['HADM_ID'] not in paper:
                continue
            key = row['HADM_ID']
            assert key not in admissions
            assert row['SUBJECT_ID'] == paper[key][0]
            adm = datetime.fromisoformat(row['ADMITTIME']) if row['ADMITTIME'] else None
            dis = datetime.fromisoformat(row['DISCHTIME']) if row['DISCHTIME'] else None
            death = datetime.fromisoformat(row['DEATHTIME']) if row['DEATHTIME'] else None
            counts['matched_original_admissions'] += 1
            counts['with_admission_clock'] += adm is not None
            counts['with_discharge_clock'] += dis is not None
            counts['recorded_in_hospital_death'] += death is not None
            counts['hospital_expire_flag'] += int(row['HOSPITAL_EXPIRE_FLAG'] or 0)
            counts['nonempty_preliminary_admission_diagnosis'] += bool(row['DIAGNOSIS'])
            admissions[key] = (adm, dis)
            for hour in windows:
                c = windows[hour]
                c['original_admissions'] += 1
                if adm is None or dis is None:
                    c['unknown_admission_or_discharge_clock'] += 1
                elif dis > adm + timedelta(hours=hour):
                    c['recorded_discharge_strictly_after_cutoff'] += 1
                    c['positive_discharge_label_in_still_hospitalized_subset'] += paper[key][1]
                else:
                    c['recorded_discharge_at_or_before_cutoff'] += 1
                c['validated_first_mace_risk_set'] += 0
    assert len(admissions) == len(paper)
    diagnosis_rows = collections.Counter()
    sequences = collections.defaultdict(list)
    with gzip.open(RAW_MIMIC / 'DIAGNOSES_ICD.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['HADM_ID']
            if key not in paper:
                continue
            assert row['SUBJECT_ID'] == paper[key][0]
            diagnosis_rows[key] += 1
            if row['SEQ_NUM']:
                sequences[key].append(int(row['SEQ_NUM']))
    counts['same_admission_icd_rows'] = sum(diagnosis_rows.values())
    counts['admissions_with_icd_rows'] = len(diagnosis_rows)
    counts['admissions_without_icd_rows'] = len(paper.keys() - diagnosis_rows.keys())
    counts['diagnosis_sequence_with_duplicate_priority'] = sum((len(set(v)) != len(v) for v in sequences.values()))
    counts['diagnosis_sequence_noncontiguous'] = sum((set(v) != set(range(1, max(v) + 1)) for v in sequences.values()))
    assert counts['recorded_in_hospital_death'] == 0
    assert counts['hospital_expire_flag'] == 0
    return {'raw_checksum_manifest': identity(checksum_path), 'raw_table_schemas': tables, 'original_predictions': identity(PAPER), 'original_admission_observation_recheck': dict(counts), 'nominal_landmark_geometry': [{'hours': h, **dict(c)} for h, c in windows.items()], 'structured_onset_or_POA_columns_in_diagnoses_icd': [], 'laboratory_release_or_storetime_column': None, 'negative_capture_interpretation': 'Diagnosis-row presence and complete encounter end clocks do not prove absence of MACE. Original nonfatal-only subset already excludes in-hospital death and cannot represent a complete cardiovascular-death composite.', 'no_validated_incident_model': True, 'notes_and_charts_limit': 'NOTEEVENTS/CHARTEVENTS have documentation/storage timestamps, but note text and clinical trajectories were not adjudicated. This audit does not assert that the full database lacks any onset evidence; it establishes that the current structured diagnostic and laboratory route cannot supply verified onset or first result visibility.'}

def main():
    if OUTPUT.exists():
        raise FileExistsError('Refusing to overwrite completed audit')
    input_paths = [PRIOR / 'evidence_gates.json', PRIOR / 'sch_fresh_units_and_timing_gate.json', PRIOR / 'nested_literal_recovery.json', OLD / 'labs/whu_lab_binding.json', OLD / 'whu/diagnosis_capture_contract_v2.json', OLD / 'whu/whu_interval_candidate_layers_v1.json', PROJECT / 'revision_2026-10-15/temporal_boundary_followup_v1/whu/whu_temporal_boundary_audit.json', PROJECT / 'revision_2026-10-15/temporal_boundary_followup_v1/mimic/mimic_temporal_audit.json']
    inputs = [identity(p) for p in input_paths]
    whu = whu_source_check()
    mimic = mimic_source_check()
    prior_sch = load(PRIOR / 'sch_fresh_units_and_timing_gate.json')
    prior_whu = load(input_paths[-2])
    output = {'execution_complete': True, 'task': 'True incident in-hospital first MACE after 0/24/48 hours; other CVD comorbidity is not itself an exclusion', 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'dependencies': 'standard library'}, 'script': identity(Path(__file__)), 'inputs': inputs, 'WHU': whu, 'MIMIC': mimic, 'SCH_prior_source_bound_evidence': {'fresh_source_identity_report_this_turn': 'Await independently generated SCH source report; this section is explicitly prior evidence until bound by finalize_timing.py.', 'conditional_component_bounds': prior_sch['prior_admission_POA_bounds'], 'capture_contract': prior_sch['prior_complete_capture_gate']}, 'WHU_prior_source_bound_evidence': {'candidate_layers': prior_whu['historical_interval_layers'], 'death_clock_recheck': prior_whu['death_timing_separate']}, 'models_fitted': 0, 'clinical_labels_assigned': 0, 'all_gate_conditions_met': False, 'source_files_written': False, 'patient_rows_exported': False, 'input_evidence_unchanged': all((identity(p) == r for p, r in zip(input_paths, inputs)))}
    assert output['input_evidence_unchanged']
    OUTPUT.write_text(json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'execution_complete': True, 'WHU_contents_rechecked': whu['counts']['raw_contents'], 'MIMIC_tables': len(mimic['raw_table_schemas']), 'output': OUTPUT.as_posix(), 'sha256': sha(OUTPUT), 'models': 0, 'patient_rows_exported': False}))
if __name__ == '__main__':
    main()
