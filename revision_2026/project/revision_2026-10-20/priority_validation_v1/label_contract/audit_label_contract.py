"""Read-only source-rule recovery and aggregate label checks.

Original modules are parsed, never imported. Only selected pure AST nodes run
against synthetic text or locally read source columns. No identifiers or
patient-level outputs are written.
"""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
CHINA = Path(_release_path('china', ''))
PRED = Path(_release_path('pipeline', ''))
RAW = PRED / 'mimic-iii-clinical-database-1.4'
SOURCES = {'sch_diagnoses': _release_source_path(CHINA / 'scripts/merge_diagnoses_fixed.py'), 'china_labels': _release_source_path(CHINA / 'scripts/survival_analysis_patient_grouped.py'), 'whu_selection': _release_source_path(CHINA / 'csv_selection.py'), 'china_model': _release_source_path(CHINA / 'optionB_parallel_analysis.py'), 'mimic_selection': _release_source_path(PRED / 'MIMIC-3/01_identify_patients.py'), 'mimic_features': _release_source_path(PRED / 'MIMIC-3/02_extract_features.py'), 'mimic_labels': _release_source_path(PRED / 'MIMIC-3/03_define_mace.py'), 'mimic_old_model': _release_source_path(PRED / 'MIMIC-3/08_paper_methods_mimic.py'), 'mimic_main_model': _release_source_path(PRED / 'Manuscript/v13/compute_v13_metrics.py')}
INPUTS = {'sch_joined_diagnostic_cache': CHINA / 'outputs/sch_diagnosis_analysis/merged_diagnosis_data.csv', 'whu_cancer_records': CHINA / 'WHU_Cancer_processed_data.csv', 'china_patient_labels': CHINA / 'outputs/survival_analysis_patient_level/patient_level_data.csv', 'china_paper_predictions': PRED / 'data/real_model_predictions_v2.csv', 'mimic_raw_diagnoses': RAW / 'DIAGNOSES_ICD.csv.gz', 'mimic_raw_admissions': RAW / 'ADMISSIONS.csv.gz', 'mimic_cached_cohort': PRED / 'MIMIC-3/data/cohort_admissions.csv', 'mimic_cached_labels': PRED / 'MIMIC-3/data/mace_labels.csv', 'mimic_cached_features': PRED / 'MIMIC-3/data/features_model.csv', 'mimic_paper_predictions': PRED / 'Manuscript/v13/paper_predictions_9946.csv'}
RANGES = {'sch_diagnoses': [(26, 41), (76, 107), (119, 129)], 'china_labels': [(37, 61), (76, 95)], 'whu_selection': [(6, 21), (33, 49)], 'china_model': [(40, 55), (59, 70), (78, 93)], 'mimic_selection': [(30, 39), (60, 80), (88, 111), (118, 127)], 'mimic_features': [(134, 146), (167, 170), (181, 215)], 'mimic_labels': [(41, 70), (76, 91), (95, 122)], 'mimic_old_model': [(119, 132)], 'mimic_main_model': [(71, 82), (280, 322), (332, 332), (439, 491)]}

def sha256(path):
    return None

def identity(path):
    return {'path': path.as_posix(), 'bytes': path.stat().st_size}

def get_assignment(tree, name):
    found = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == name for t in n.targets))]
    assert len(found) == 1, (name, len(found))
    return found[0]

def pure_nodes(nodes, context=None):
    ns = {'pd': pd, 'np': np}
    if context:
        ns.update(context)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, '<isolated-source-rules>', 'exec'), ns)
    return ns

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def equality_aggregate(observed, reference):
    common = observed.index.intersection(reference.index)
    return {'observed_units': len(observed), 'reference_units': len(reference), 'common_units': len(common), 'observed_only_units': len(observed.index.difference(reference.index)), 'reference_only_units': len(reference.index.difference(observed.index)), 'label_disagreements_on_common_units': int((observed.loc[common].astype(int) != reference.loc[common].astype(int)).sum())}

def main():
    sources_before = {name: identity(path) for name, path in SOURCES.items()}
    inputs_before = {name: identity(path) for name, path in INPUTS.items()}
    trees = {name: ast.parse(path.read_text(encoding='utf-8-sig')) for name, path in SOURCES.items()}
    source_register = {}
    for name, meta in sources_before.items():
        lines = SOURCES[name].read_text(encoding='utf-8-sig').splitlines()
        source_register[name] = {**meta, 'excerpts': [{'line_start': start, 'line_end': end, 'source': '\n'.join(lines[start - 1:end])} for start, end in RANGES[name]]}
    sch_keywords_node = get_assignment(trees['sch_diagnoses'], 'mace_keywords')
    sch_function_node = next((n for n in ast.walk(trees['sch_diagnoses']) if isinstance(n, ast.FunctionDef) and n.name == 'is_mace'))
    sch_ns = pure_nodes([sch_keywords_node, sch_function_node])
    sch_rule = sch_ns['is_mace']
    whu_keywords_node = get_assignment(trees['china_labels'], 'mace_keywords')
    whu_label_assignment = next((n for n in ast.walk(trees['china_labels']) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and (t.value.id == 'whu_data') and isinstance(t.slice, ast.Constant) and (t.slice.value == 'MACE') for t in n.targets))))
    whu_lambda = next((n for n in ast.walk(whu_label_assignment) if isinstance(n, ast.Lambda)))
    whu_ns = pure_nodes([whu_keywords_node])
    whu_rule = eval(compile(ast.Expression(whu_lambda), '<isolated-whu-lambda>', 'eval'), whu_ns)
    selection_ns = pure_nodes([get_assignment(trees['mimic_selection'], 'CANCER_PREFIXES'), get_assignment(trees['mimic_selection'], 'CARDIAC_PREFIXES')])
    mimic_ns = pure_nodes([get_assignment(trees['mimic_labels'], 'MACE_CODES'), next((n for n in trees['mimic_labels'].body if isinstance(n, ast.FunctionDef) and n.name == 'code_matches'))])
    mace_codes = mimic_ns['MACE_CODES']
    code_matches = mimic_ns['code_matches']
    cancer_prefixes = selection_ns['CANCER_PREFIXES']
    cardiac_prefixes = selection_ns['CARDIAC_PREFIXES']
    main_feature_names = ast.literal_eval(get_assignment(trees['mimic_main_model'], 'FEATS_43_EN').value)
    extract_feature_names = ast.literal_eval(get_assignment(trees['mimic_features'], 'MODEL_FEATURES').value)
    assert 'Admission_has_cardiac' not in main_feature_names
    assert 'Admission_has_cardiac' not in extract_feature_names
    all_component_prefixes_overlap = all((prefix.startswith(cardiac_prefixes) for prefixes in mace_codes.values() for prefix in prefixes))
    assert all_component_prefixes_overlap
    text_cases = [('missing', None, False, False), ('empty', '', False, False), ('MI_upper', 'MI', True, False), ('MI_lower', 'mi', True, False), ('MI_substring_not_boundary', 'microbiology', True, False), ('myocardial_word', 'myocardial', True, False), ('acute_coronary_phrase', 'acute coronary syndrome', True, False), ('ACS_upper', 'ACS', True, False), ('english_stroke', 'Stroke', True, True), ('negated_infarction', '未见心肌梗死', True, True), ('history_stroke', '既往脑卒中', True, True), ('suspected_infarction', '疑似心梗', True, True), ('renal_infarction', '肾梗死', True, True), ('all_cause_death', '肿瘤相关死亡', True, True), ('heart_failure_only', '心力衰竭', False, False), ('cardiac_arrest_only', '心脏骤停', False, False), ('brain_hemorrhage', '脑出血', True, False)]
    text_tests = []
    for label, text, expected_sch, expected_whu in text_cases:
        sch_got = bool(sch_rule(text))
        whu_got = bool(whu_rule('' if text is None else text))
        assert sch_got == expected_sch and whu_got == expected_whu, label
        text_tests.append({'synthetic_case': label, 'synthetic_text': text, 'SCH': sch_got, 'WHU': whu_got, 'passed': True})
    code_cases = [('410.01', ['MI']), ('410', []), ('411.1', ['MI', 'Unstable_Angina']), ('428.42', []), ('428.43', ['Heart_Failure']), ('428.2', []), ('438.9', ['Stroke']), ('425.4', ['Cardiomyopathy']), ('427.60', ['Cardiac_Arrest']), (' 410.01 ', []), ('V45.1', [])]
    code_tests = []
    for code, expected in code_cases:
        normalized = code.replace('.', '').upper()
        got = [label for label, prefixes in mace_codes.items() if code_matches(normalized, prefixes)]
        assert got == expected, (code, got, expected)
        code_tests.append({'synthetic_code': code, 'normalized_for_outcome': normalized, 'outcome_components': got, 'selection_cardiac_after_strip': normalized.strip().startswith(cardiac_prefixes), 'passed': True})
    patient_labels = pd.read_csv(INPUTS['china_patient_labels'], usecols=['病案号', 'Center', 'MACE'], dtype={'病案号': str})
    paper_cn = pd.read_csv(INPUTS['china_paper_predictions'], usecols=['病案号', 'Center', 'MACE'], dtype={'病案号': str})
    sch_fragments = []
    sch_rows = 0
    sch_cached_flag_disagreements = 0
    sch_missing_text_rows = 0
    for chunk in pd.read_csv(INPUTS['sch_joined_diagnostic_cache'], chunksize=100000, usecols=['病案号', '所有诊断', 'MACE_flag'], dtype={'病案号': str}, low_memory=False):
        sch_rows += len(chunk)
        got = chunk['所有诊断'].apply(sch_rule).astype(bool)
        cached = chunk['MACE_flag'].astype(str).str.lower().isin(['true', '1'])
        sch_cached_flag_disagreements += int((got != cached).sum())
        sch_missing_text_rows += int(chunk['所有诊断'].isna().sum())
        sch_fragments.append(pd.DataFrame({'key': chunk['病案号'], 'label': got}).groupby('key')['label'].max())
    sch_labels = pd.concat(sch_fragments).groupby(level=0).max()
    whu = pd.read_csv(INPUTS['whu_cancer_records'], usecols=['病案号', '主要诊断'], dtype={'病案号': str})
    whu_labels = whu.assign(label=whu['主要诊断'].fillna('').apply(whu_rule)).groupby('病案号')['label'].max()
    chinese_checks = {}
    for center, reconstructed in [('SCH', sch_labels), ('WHU', whu_labels)]:
        stored = patient_labels.loc[patient_labels['Center'] == center].set_index('病案号')['MACE']
        paper = paper_cn.loc[paper_cn['Center'] == center].set_index('病案号')['MACE']
        chinese_checks[center] = {'units': len(reconstructed), 'positive_labels': int(reconstructed.sum()), 'versus_patient_label_cache': equality_aggregate(reconstructed, stored), 'versus_paper_label_cache': equality_aggregate(reconstructed, paper)}
    chinese_checks['SCH']['merged_cache_rows'] = sch_rows
    chinese_checks['SCH']['row_flag_disagreements'] = sch_cached_flag_disagreements
    chinese_checks['SCH']['rows_missing_external_joined_text'] = sch_missing_text_rows
    chinese_checks['SCH']['boundary'] = 'Replayed rule on joined diagnostic cache, not a new source join or proof of complete negative ascertainment.'
    chinese_checks['WHU']['raw_processed_rows'] = len(whu)
    chinese_checks['WHU']['boundary'] = 'Replayed the original primary-diagnosis rule; upstream source-system cancer ascertainment is not newly adjudicated.'
    diag = pd.read_csv(INPUTS['mimic_raw_diagnoses'], usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM'], dtype={'ICD9_CODE': str}, low_memory=False)
    adm = pd.read_csv(INPUTS['mimic_raw_admissions'], usecols=['SUBJECT_ID', 'HADM_ID', 'HOSPITAL_EXPIRE_FLAG'])
    cached_cohort = pd.read_csv(INPUTS['mimic_cached_cohort'], usecols=['SUBJECT_ID', 'HADM_ID', 'Admission_has_cardiac', 'Admission_has_cancer', 'HOSPITAL_EXPIRE_FLAG'])
    cached_labels = pd.read_csv(INPUTS['mimic_cached_labels'], usecols=['HADM_ID', 'MACE', 'MACE_Types'], keep_default_na=False)
    feat = pd.read_csv(INPUTS['mimic_cached_features'], usecols=['HADM_ID', 'SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG'])
    paper_mimic = pd.read_csv(INPUTS['mimic_paper_predictions'], usecols=['HADM_ID', 'MACE'])
    selection_code = diag['ICD9_CODE'].fillna('').str.replace('.', '', regex=False).str.upper().str.strip()
    is_cancer = selection_code.str.startswith(cancer_prefixes)
    is_cardiac = selection_code.str.startswith(cardiac_prefixes)
    selected_subjects = set(diag.loc[is_cancer, 'SUBJECT_ID']) & set(diag.loc[is_cardiac, 'SUBJECT_ID'])
    raw_cohort = adm.loc[adm['SUBJECT_ID'].isin(selected_subjects)].copy()
    cancer_hids = set(diag.loc[is_cancer, 'HADM_ID'])
    cardiac_hids = set(diag.loc[is_cardiac, 'HADM_ID'])
    raw_cohort['reconstructed_cardiac'] = raw_cohort['HADM_ID'].isin(cardiac_hids).astype(int)
    raw_cohort['reconstructed_cancer'] = raw_cohort['HADM_ID'].isin(cancer_hids).astype(int)
    cohort_comp = cached_cohort.merge(raw_cohort, on='HADM_ID', suffixes=('_cache', '_raw'), validate='one_to_one')
    selected_diag = diag.loc[diag['HADM_ID'].isin(set(raw_cohort['HADM_ID']))].copy()
    outcome_code = selected_diag['ICD9_CODE'].fillna('').str.replace('.', '', regex=False).str.upper()
    component_hids = {}
    for label, prefixes in mace_codes.items():
        mask = outcome_code.apply(lambda code: code_matches(code, prefixes))
        component_hids[label] = set(selected_diag.loc[mask, 'HADM_ID'])
    cardiac_death_hids = set(raw_cohort.loc[(raw_cohort['HOSPITAL_EXPIRE_FLAG'] == 1) & raw_cohort['HADM_ID'].isin(cardiac_hids), 'HADM_ID'])
    all_mace_hids = set().union(*component_hids.values(), cardiac_death_hids)
    reconstructed_mimic = raw_cohort.set_index('HADM_ID').index.to_series().isin(all_mace_hids).astype(int)
    cached_mace = cached_labels.set_index('HADM_ID')['MACE']
    model_hids = set(feat.loc[feat['HOSPITAL_EXPIRE_FLAG'] == 0, 'HADM_ID'])
    model_labels = reconstructed_mimic.loc[reconstructed_mimic.index.isin(model_hids)]
    flags = raw_cohort.set_index('HADM_ID')['reconstructed_cardiac']
    subtype_disagreements = 0
    for hid, old_types in cached_labels.set_index('HADM_ID')['MACE_Types'].items():
        types = {label for label, hids in component_hids.items() if hid in hids}
        if hid in cardiac_death_hids:
            types.add('Cardiac_Death')
        recorded = set(str(old_types).split('|')) - {'None'}
        subtype_disagreements += int(types != recorded)
    mimic_checks = {'raw_diagnosis_rows_read': len(diag), 'raw_admission_rows_read': len(adm), 'reconstructed_selected_patients': len(selected_subjects), 'reconstructed_selected_admissions': len(raw_cohort), 'cached_selected_admissions': len(cached_cohort), 'cohort_admission_set_exact': set(raw_cohort['HADM_ID']) == set(cached_cohort['HADM_ID']), 'Admission_has_cardiac_disagreements': int((cohort_comp['reconstructed_cardiac'] != cohort_comp['Admission_has_cardiac']).sum()), 'Admission_has_cancer_disagreements': int((cohort_comp['reconstructed_cancer'] != cohort_comp['Admission_has_cancer']).sum()), 'all_cohort_positive_labels': int(reconstructed_mimic.sum()), 'component_admission_counts': {label: len(hids) for label, hids in component_hids.items()}, 'cardiac_death_proxy_admissions': len(cardiac_death_hids), 'versus_cached_labels': equality_aggregate(reconstructed_mimic, cached_mace), 'component_set_disagreements': subtype_disagreements, 'model_survivor_admissions': len(model_labels), 'model_survivor_patients': int(feat.loc[feat['HADM_ID'].isin(model_hids), 'SUBJECT_ID'].nunique()), 'model_survivor_positive_labels': int(model_labels.sum()), 'versus_main_paper_labels': equality_aggregate(model_labels, paper_mimic.set_index('HADM_ID')['MACE']), 'all_cohort_mace_positive_with_cardiac_flag': int(flags.loc[reconstructed_mimic == 1].sum()), 'all_non_death_outcome_prefixes_nested_in_cardiac_flag_definition': all_component_prefixes_overlap, 'all_cohort_cardiac_flag_positive_mace_negative': int(((flags == 1) & (reconstructed_mimic == 0)).sum()), 'model_mace_positive_with_cardiac_flag': int(flags.loc[model_labels.index][model_labels == 1].sum()), 'model_cardiac_flag_positive_mace_negative': int(((flags.loc[model_labels.index] == 1) & (model_labels == 0)).sum()), 'negative_definition': 'No matched component or broad-cardiac death proxy in the supplied admission diagnosis table; not adjudicated freedom from disease.'}
    all_exact = [chinese_checks[c][test]['label_disagreements_on_common_units'] == 0 and chinese_checks[c][test]['observed_only_units'] == 0 and (chinese_checks[c][test]['reference_only_units'] == 0) for c in ['SCH', 'WHU'] for test in ['versus_patient_label_cache', 'versus_paper_label_cache']]
    all_exact += [sch_cached_flag_disagreements == 0, mimic_checks['cohort_admission_set_exact'], mimic_checks['Admission_has_cardiac_disagreements'] == 0, mimic_checks['Admission_has_cancer_disagreements'] == 0, subtype_disagreements == 0]
    for target in ['versus_cached_labels', 'versus_main_paper_labels']:
        all_exact.extend((mimic_checks[target][key] == 0 for key in ['label_disagreements_on_common_units', 'observed_only_units', 'reference_only_units']))
    sources_after = {name: identity(path) for name, path in SOURCES.items()}
    inputs_after = {name: identity(path) for name, path in INPUTS.items()}
    assert sources_before == sources_after and inputs_before == inputs_after
    save('source_register.json', source_register)
    save('synthetic_boundary_tests.json', {'clinical_rows_used': False, 'text_tests': text_tests, 'ICD_tests': code_tests, 'all_passed': True, 'note': 'These tests preserve historical implementation defects; passing does not validate the clinical definition.'})
    save('aggregate_label_checks.json', {'China': chinese_checks, 'MIMIC': mimic_checks, 'all_reproduction_checks_passed': bool(all(all_exact)), 'inputs_unchanged': True, 'source_code_unchanged': True})
    save('label_contract.json', {'schema': 1, 'purpose': 'Original-label provenance and comparability, not a new clinical endpoint definition.', 'sources': {name: {k: v for k, v in meta.items() if k != 'excerpts'} for name, meta in source_register.items()}, 'inputs': inputs_before, 'rules': {'SCH': {'unit': 'patient', 'diagnostic_join': 'patient-only left join; no admission/date restriction', 'source_diagnostic_fields': ast.literal_eval(get_assignment(trees['sch_diagnoses'], 'key_diagnosis_cols').value), 'keywords': sch_ns['mace_keywords'], 'normalization': 'both text and keyword lowercased', 'match': 'substring, without word boundaries, negation, uncertainty, history or incident-event exclusion', 'missing': 'missing joined text -> false; any row positive -> patient positive', 'heart_failure_hospitalization': 'No explicit HF term, hospitalization reason or primary-diagnosis restriction.', 'cardiovascular_death': 'Generic death keyword, no cause-of-death adjudication.', 'prediction_time': None, 'event_time': None, 'feature_window': 'all processed records for the patient; original Option B mean; v13 transfer median', 'upstream_cancer_selection': 'SCH processed source cohort inherited; no separate malignant-cancer or cardiac eligibility rule in these label/model entries.'}, 'WHU': {'unit': 'patient', 'diagnostic_fields': ['主要诊断'], 'keywords': whu_ns['mace_keywords'], 'normalization': 'text lowercased but keywords unchanged; uppercase MI literal cannot match', 'match': 'substring, without word boundaries, negation, uncertainty, history or incident-event exclusion', 'missing': 'missing primary diagnosis -> empty -> false; any record positive -> patient positive', 'heart_failure_hospitalization': 'No explicit HF term or hospitalization rule.', 'cardiovascular_death': 'Generic death keyword, no cause-of-death adjudication.', 'prediction_time': None, 'event_time': None, 'feature_window': 'all patient records from WHU_Cancer_processed_data; original Option B mean; v13 transfer median', 'upstream_cancer_selection': {'code_only_not_replayed_in_this_audit': True, 'diagnostic_fields': ast.literal_eval(get_assignment(trees['whu_selection'], 'DIAG_COLS').value), 'keywords': ast.literal_eval(get_assignment(trees['whu_selection'], 'TUMOR_KEYWORDS').value), 'operation': 'regex union on any existing specified diagnosis fields; select all rows of every hit patient', 'limitation': 'Broad tumor/proliferative terms do not enforce histologically confirmed malignant cancer; no negation or timing rule.'}}, 'MIMIC': {'unit': 'admission, with repeated patients', 'selection_cancer_prefixes': list(cancer_prefixes), 'selection_cardiac_prefixes': list(cardiac_prefixes), 'selection_normalization': 'fill empty, remove dots, uppercase, strip', 'selection_rule': 'patient ever-cancer intersect patient ever-cardiac; retain all their admissions, not only simultaneous cancer/cardiac or first admission', 'outcome_normalization': 'fill empty, remove dots, uppercase; no strip', 'component_prefixes': mace_codes, 'component_match': 'starts-with; any supplied diagnosis position; minimum SEQ_NUM is priority/order, not event time', 'death_proxy': 'HOSPITAL_EXPIRE_FLAG==1 and same-admission broad cardiac diagnosis; no adjudicated cardiovascular cause', 'main_model_filter': 'features/labels inner join HADM_ID; HOSPITAL_EXPIRE_FLAG==0; all inpatient deaths excluded', 'missing': 'unmatched admission label -> zero', 'heart_failure_hospitalization': 'Selected 428 prefixes in any diagnosis position; no new/decompensated/primary-reason validation; 42842 absent.', 'prediction_time': None, 'event_time': None, 'feature_window': 'all available labs linked to admission, no hour cutoff in original 02/v13 entry', 'Admission_has_cardiac': {'definition': 'same-admission any cardiac-selection prefix above', 'overlap': 'all original non-death MACE prefixes are within the broad cardiac-selection prefixes', 'original_08_use': 'explicitly included among model features', 'current_02_v13_use': 'absent from explicit main model feature list', 'current_02_model_feature_count': len(extract_feature_names), 'current_v13_model_feature_count': len(main_feature_names), 'meaning': 'an outcome-overlapping diagnosis-derived flag, not independent baseline clinical history'}}}, 'comparability_gate': {'equivalent_incident_MACE_task': False, 'same_operational_label': False, 'same_analysis_unit': False, 'shared_pre_event_window_validated': False, 'clinical_negative_capture_validated': False, 'permissible_interpretation_if_assays_units_verified': 'Exploratory recorded-diagnosis transfer stress test with each original cohort definition retained and disclosed.', 'not_permitted': 'Equivalent first-MACE prediction, warning days, leak-free pre-event validation, or attribution of transfer differences solely to biology.', 'unit_equivalence_must_be_checked_separately': True}})
    outputs = sorted((p for p in OUT.iterdir() if p.is_file() and p.name != 'run_manifest.json'))
    save('run_manifest.json', {'python': platform.python_version(), 'executable': __import__('sys').executable, 'pandas': pd.__version__, 'numpy': np.__version__, 'source_hashes_verified_after_run': True, 'input_hashes_verified_after_run': True, 'input_identity_count': len(INPUTS), 'source_identity_count': len(SOURCES), 'outputs': [identity(path) for path in outputs], 'no_model_training': True, 'no_patient_rows_or_identifiers_in_outputs': True, 'no_original_module_import': True})
    print(json.dumps({'all_reproduction_checks_passed': bool(all(all_exact)), 'synthetic_tests_passed': len(text_tests) + len(code_tests), 'SCH_patients': len(sch_labels), 'WHU_patients': len(whu_labels), 'MIMIC_model_admissions': len(model_labels), 'output_directory': OUT.as_posix()}, ensure_ascii=False))
if __name__ == '__main__':
    main()
