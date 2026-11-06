"""Bind independently checked source identities and fail-closed timing gates."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timedelta
from pathlib import Path
HERE = Path(__file__).resolve().parent
SCH_REPORT = HERE.parent / 'sch_units/sch_assay_metadata_scan.json'
SCH_CSV = Path(_release_path('njm', 'SCH/SCH_processed_data.csv'))
EXPECTED_SCH = None

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

def entirely_after(lower, upper, cutoff, discharge, lower_inclusive=True):
    """Unknown boundaries fail; an interval touching the cutoff is not later."""
    if any((x is None for x in (lower, upper, cutoff, discharge))):
        return False
    if lower > upper or upper > discharge or (lower == upper and (not lower_inclusive)):
        return False
    return lower > cutoff or (lower == cutoff and (not lower_inclusive))

def eligible_to_fit(conditions):
    required = {'encounter_linkage', 'prior_event_free', 'landmark_event_free', 'post_landmark_onset', 'result_first_availability', 'negative_capture', 'complete_stay_observation', 'endpoint_components'}
    return set(conditions) == required and all((value is True for value in conditions.values()))

def tests():
    t = datetime(2000, 1, 1)
    end = t + timedelta(hours=100)
    assert entirely_after(t, end, t, end, lower_inclusive=False)
    assert not entirely_after(t, end, t, end)
    assert not entirely_after(t, end, t + timedelta(hours=24), end, lower_inclusive=False)
    assert not entirely_after(t, end, t + timedelta(hours=48), end, lower_inclusive=False)
    assert entirely_after(t + timedelta(hours=49), end, t + timedelta(hours=48), end)
    assert not entirely_after(None, end, t, end)
    assert not entirely_after(end, t, t, end)
    assert not entirely_after(t + timedelta(hours=1), end + timedelta(hours=1), t, end)
    assert not entirely_after(t, t, t, end, lower_inclusive=False)
    gates = {k: True for k in ('encounter_linkage', 'prior_event_free', 'landmark_event_free', 'post_landmark_onset', 'result_first_availability', 'negative_capture', 'complete_stay_observation', 'endpoint_components')}
    assert eligible_to_fit(gates)
    for key in gates:
        candidate = dict(gates)
        candidate[key] = None
        assert not eligible_to_fit(candidate)
    assert not eligible_to_fit({k: v for k, v in gates.items() if k != 'negative_capture'})
    return {'assertions': 19, 'passed': True, 'synthetic_only': True}

def main():
    output = HERE / 'temporal_gate_decisions.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite completed gate decisions')
    execution = load(HERE / 'temporal_gate_execution.json')
    sch = load(SCH_REPORT)
    onset_context = load(HERE / 'onset_candidate_context.json')
    onset_semantics = load(HERE / 'onset_form_semantics.json')
    onset_precision = load(HERE / 'onset_form_precision.json')
    assert onset_semantics['source_section_names'] == ['院内感染病例登记表']
    assert onset_semantics['aggregate_counts']['nonempty_date_occurrences'] == 4
    assert onset_semantics['aggregate_counts']['section_contains_MACE_word'] == 0
    assert onset_precision['aggregate_counts']['nonempty_occurrences'] == 4
    assert len(sch['source_files']) == 34
    processed = identity(SCH_CSV)
    assert processed['bytes'] == 1041480549
    prior = load(Path(_release_path('project', 'revision_2026-10-19/substantive_gaps_v1/analysis/timing_units/sch_fresh_units_and_timing_gate.json')))
    source_rows = sch['source_files']
    assert sum((row['bytes'] for row in source_rows)) == 1512955498
    expected = sorted(((row['bytes'], row['sha256']) for row in prior['raw_source_identities']))
    actual = sorted(((row['bytes'], processed['sha256'] if row['path'] == SCH_CSV.as_posix() else row['sha256']) for row in source_rows))
    assert actual == expected
    assert all((row.get('matches_prior_sha') or row['path'] == SCH_CSV.as_posix() for row in source_rows))
    (HERE / 'sch_processed_fresh_identity.json').write_text(json.dumps({'source': processed, 'fresh_full_file_hash_completed': True, 'expected_identity_matched': True, 'script': identity(Path(__file__))}, indent=2) + '\n', encoding='utf-8')
    rows = []

    def add(centre, gate, status, evidence, limitation):
        rows.append({'centre': centre, 'gate': gate, 'status': status, 'evidence': evidence, 'remaining_limit': limitation})
    add('SCH', 'encounter_boundaries', 'partial', '179832 agreed patient-plus-stay-day keys linked to the current 38509 patients; 31384 first-recorded selected index stays in the prior source-bound audit.', 'Calendar-day geometry is not exact admission hour, exhaustive history or a certified encounter export contract.')
    add('SCH', 'event_onset', 'not_established', '16 selected-index literal POA-absent/concordant acute-component intervals after admission; no exact onset clock.', 'Every interval crosses 24h and 48h; not adjudicated first MACE or a full composite endpoint.')
    add('SCH', 'POA_semantics', 'partial', 'Seven local diagnosis slots have directly observed present/absent/clinically-undetermined/unknown strings.', 'Literal meaning is not complete validated first-event adjudication, and no numeric POA dictionary was borrowed.')
    add('SCH', 'first_result_availability', 'not_established', 'Processed laboratory dates are day-level; original-unit and result-release contract not recovered.', 'Cannot assert measurement results were available at 0h, 24h or 48h.')
    add('SCH', 'negative_capture', 'not_established', '112375 of 179832 current linked stays fill slot 7; all seven slots empty in zero stays.', 'Last-slot occupation warns of possible truncation but does not prove it; no completeness manifest or certified clinical negatives.')
    add('SCH', 'cardiovascular_death_and_HF_hospitalization', 'not_established', 'Existing death and I50 records are source candidates.', 'Cardiovascular cause, nonfatal status and an HF-caused hospitalization definition remain unadjudicated.')
    whu = execution['WHU']
    add('WHU', 'encounter_boundaries', 'partial', f"Freshly verified {whu['counts']['raw_contents']} source contents for {whu['counts']['record_occurrences']} record occurrences, 2246 patients and 5415 stay-day keys.", 'Incomplete coverage of 2562 original patients and 6694 processed stay-day keys; encounter namespace ambiguity remains.')
    add('WHU', 'event_onset', 'not_established', f"Fresh all-key traversal found {whu['explicit_onset_fields']} onset-named key occurrences: 123 empty and 4 nonempty occurrence dates, all in hospital-acquired infection registration forms, not MACE onset. Prior 19 source interval candidates leave zero after observed prior/index flags.", 'The four infection dates cannot label cardiovascular event onset. Recorded admission/discharge/diagnosis clocks are also not MACE onset; free-text clinical onset adjudication is not complete.')
    add('WHU', 'POA_semantics', 'partial', 'Prior explicit MACE candidate slots: 1927 present, 39 absent, 144 uncertain, 922 unknown.', 'Source form-selection/exporter contract and clinical component validation remain uncertified.')
    add('WHU', 'first_result_availability', 'not_established', f"Report timestamps exist; fresh schema recovered {whu['explicit_first_result_available_fields']} explicit release/first-visibility fields.", 'Same-object report time is not certified first clinician-visible result time; examination review clocks cannot substitute.')
    add('WHU', 'negative_capture', 'not_established', f"Fresh schema recovered {whu['explicit_capture_contract_fields']} diagnosis completeness/count-contract fields.", 'Missing diagnosis is a recorded-label zero only; not a validated clinical MACE-negative outcome.')
    add('WHU', 'cardiovascular_death_and_HF_hospitalization', 'not_established', '63 matched death encounter groups: 19 compatible clocks, 43 conflicting, 1 unknown; 0 clinically adjudicated cardiovascular causes.', '17/17/15 all-cause time candidates after 0/24/48h are not cardiovascular-death MACE labels; HF reason-for-admission validation remains absent.')
    m = execution['MIMIC']['original_admission_observation_recheck']
    add('MIMIC', 'encounter_boundaries', 'available_for_original_subset', f"{m['matched_original_admissions']} original admissions joined by admission and patient key; {m['with_admission_clock']} admission and {m['with_discharge_clock']} discharge clocks.", 'Recorded stay end is observable; it does not establish freedom from MACE at each landmark.')
    add('MIMIC', 'event_onset', 'not_established', 'Original DIAGNOSES_ICD has only ROW_ID, SUBJECT_ID, HADM_ID, SEQ_NUM and ICD9_CODE.', 'No onset field; SEQ_NUM is diagnosis priority, not event chronology. Notes are not clinically adjudicated.')
    add('MIMIC', 'POA_semantics', 'not_available_in_used_table', 'No POA column in DIAGNOSES_ICD.', 'Preliminary ADMISSIONS.DIAGNOSIS text is not exhaustive event-specific POA adjudication.')
    add('MIMIC', 'first_result_availability', 'not_established', 'LABEVENTS has CHARTTIME but no STORETIME or result-release field.', 'CHARTEVENTS/NOTEEVENTS storage timestamps describe different records; no validated LIS-event bridge is available.')
    add('MIMIC', 'negative_capture', 'not_established', f"{m['admissions_with_icd_rows']} original admissions have {m['same_admission_icd_rows']} same-admission ICD rows.", 'Coding coverage is not clinical absence; codes lack onset and cannot exclude pre-existing or already-evolving events.')
    add('MIMIC', 'cardiovascular_death_and_HF_hospitalization', 'not_established', 'Original 9946-admission paper subset contains zero in-hospital death flags or death clocks by design.', 'A full first-MACE task including cardiovascular death requires a new cohort and cause adjudication; any HF code is not proof that HF caused hospitalization.')
    for centre in ('SCH', 'WHU', 'MIMIC'):
        for hour in (0, 24, 48):
            add(centre, f'complete_{hour}h_incident_first_MACE_risk_set', 'do_not_fit', 'Not all clinical endpoint, onset, predictor-availability and negative-capture gates pass.', 'Retain retrospective recorded-diagnosis classification; do not create clinical negatives or warning lead times.')
    csv_path = HERE / 'temporal_gate_matrix.csv'
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contract = {'task': 'First incident in-hospital MACE after the prespecified landmark', 'cutoffs_hours_after_admission': [0, 24, 48], 'time_horizon': 'Landmark to end of the same hospitalization, not a fixed post-discharge day horizon', 'index': 'First qualifying admission under a source-supported admission/cancer/history contract, not selected for discrimination or event frequency', 'risk_set': 'Exclude verified prior or at-cutoff MACE; retain other CVD comorbidity. Unknown at-cutoff event status is not event-free. Require still in hospital strictly after cutoff.', 'features': 'Only measurements uniquely linked to this encounter with certified first clinical availability at or before the cutoff. Admission model may use verified pre-admission available information; it is not restricted to exactly admission-stamped new labs.', 'outcome': 'An independently adjudicated component onset strictly after cutoff and no later than discharge; full negative capture must be independently established.', 'death': 'Do not treat all-cause death as cardiovascular death; do not condition model eligibility on future survival.', 'HF': 'Retain chronic HF comorbidity unless it meets the prespecified prior-MACE definition; do not equate any I50 code with hospitalization caused by HF.', 'validation': 'Patient-separated fitting, training-only preprocessing, fixed cutoffs and outcome definitions, patient-cluster uncertainty when repeated admissions are retained. No clinical first-MACE model until all source gates pass.', 'lead_time': 'Onset minus landmark only after exact/adjudicated onset is independently established; no numerical advance-warning claim from current source evidence.', 'models_fitted': 0}
    contract_path = HERE / 'incident_model_contract.json'
    contract_path.write_text(json.dumps(contract, indent=2) + '\n', encoding='utf-8')
    report = {'execution_complete': True, 'audit_tests': tests(), 'source_binding': {'SCH_source_count': len(source_rows), 'SCH_source_bytes': sum((x['bytes'] for x in source_rows)), 'all_34_identity_pairs_match_prior': True, 'SCH_processed_csv_fresh_full_file_hash': processed, 'other_SCH_sources_freshly_verified_by_independent_agent_report': identity(SCH_REPORT), 'timing_execution_report': identity(HERE / 'temporal_gate_execution.json'), 'WHU_generic_onset_context': identity(HERE / 'onset_candidate_context.json'), 'WHU_generic_onset_semantics': identity(HERE / 'onset_form_semantics.json'), 'WHU_infection_date_precision': identity(HERE / 'onset_form_precision.json')}, 'gate_rows': rows, 'gate_matrix': identity(csv_path), 'incident_model_contract': identity(contract_path), 'script': identity(Path(__file__)), 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}, 'new_source_supported_first_MACE_labels': 0, 'models_fitted': 0, 'all_scientific_temporal_requirements_resolved': False, 'source_files_modified': False, 'patient_rows_or_identifiers_exported': False, 'minimum_additional_source_evidence': [{'source': 'SCH and WHU encounter diagnosis exports', 'fields': ['unique encounter linkage', 'complete diagnosis rows and count manifest', 'principal/secondary role', 'local POA dictionary and revision history', 'component-specific onset or adjudicated time interval', 'history coverage before index and current-admission event-free state'], 'acceptance': 'Documentation time and POA absence alone do not establish onset after 24h or 48h; absence of a code alone is not a clinical negative.'}, {'source': 'SCH and WHU LIS; MIMIC-specific validated availability linkage', 'fields': ['stable measurement and encounter identifiers', 'collection time', 'first clinical release/visibility timestamp', 'result version/revision policy', 'timezone and precision semantics'], 'acceptance': 'Do not infer first availability from report/collection time or borrow examination review time.'}, {'source': 'All centres, clinically adjudicated first-MACE components', 'fields': ['MI/stroke/arrest onset', 'nonfatal status', 'death time and underlying/contributing cause', 'HF as the reason for hospitalization', 'complete observation to same-stay discharge/death'], 'acceptance': 'Do not use all-cause death as cardiovascular death or any I50 code as an HF-caused hospitalization.'}, {'source': 'MIMIC clinical notes/charts or independently validated event abstraction', 'fields': ['event-specific onset linked to admission', 'event-free state at 0/24/48h', 'cause-specific death adjudication'], 'acceptance': 'The structured ICD priority sequence cannot supply these fields; any new clinical abstraction requires a documented method and validation.'}], 'reason_missing_not_attributed_to_deidentification': True, 'scope': 'Fresh schema/identity/stay-observation audit plus explicitly identified prior source-bound clinical-candidate evidence. No note-level clinical adjudication. This is not a declaration that the full source systems never stored onset or release times.'}
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'execution_complete': True, 'gate_rows': len(rows), 'cutoff_decisions': 9, 'all_nine_do_not_fit': True, 'tests_passed': True, 'output_sha256': sha(output), 'models_fitted': 0}))
if __name__ == '__main__':
    main()
