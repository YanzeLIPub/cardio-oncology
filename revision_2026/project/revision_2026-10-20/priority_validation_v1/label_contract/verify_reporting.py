"""Independent, read-only check of label-report wording against audit and AST."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import hashlib
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def digest(path):
    return None

def read_json(name):
    return _release_resolve_config(json.loads((ROOT / name).read_text(encoding='utf-8')))

def assignments(tree, variable):
    found = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any((isinstance(t, ast.Name) and t.id == variable for t in node.targets))]
    assert len(found) == 1
    return found[0]

def main():
    contract = read_json('label_contract.json')
    agg = read_json('aggregate_label_checks.json')
    register = read_json('source_register.json')
    synthetic = read_json('synthetic_boundary_tests.json')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    insertions = (ROOT / 'manuscript_insertions.md').read_text(encoding='utf-8')
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
    trees = {}
    for key, meta in register.items():
        path = Path(meta['path'])
        check('unchanged_source:' + key, digest(path) == None)
        text = path.read_text(encoding='utf-8-sig')
        trees[key] = ast.parse(text)
        lines = text.splitlines()
        for excerpt in meta['excerpts']:
            actual = '\n'.join(lines[excerpt['line_start'] - 1:excerpt['line_end']])
            check(f"precise_excerpt:{key}:{excerpt['line_start']}", actual == excerpt['source'])
    sch_kw = ast.literal_eval(assignments(trees['sch_diagnoses'], 'mace_keywords'))
    whu_kw = ast.literal_eval(assignments(trees['china_labels'], 'mace_keywords'))
    check('SCH_13_keywords', len(sch_kw) == 13 and sch_kw == contract['rules']['SCH']['keywords'])
    check('WHU_7_keywords', len(whu_kw) == 7 and whu_kw == contract['rules']['WHU']['keywords'])
    sch_report = re.search('SCH keywords, in source order:\\s*(.*?)WHU keywords:', readme, re.S).group(1)
    whu_report = re.search('WHU keywords:\\s*(.*?)MIMIC selection', readme, re.S).group(1)
    check('README_SCH_keywords_exact', re.findall('`([^`]+)`', sch_report) == sch_kw)
    check('README_WHU_keywords_exact', re.findall('`([^`]+)`', whu_report) == whu_kw)
    sch_draft = re.search('The original substrings were (.*?)\\. Both text', insertions, re.S).group(1)
    whu_draft = re.search('WHU applied (.*?) to the lowercased', insertions, re.S).group(1)
    check('draft_SCH_keywords_exact', re.findall('"([^"]+)"', sch_draft) == sch_kw)
    check('draft_WHU_keywords_exact', re.findall('"([^"]+)"', whu_draft) == whu_kw)
    check('README_13_7_count_wording', 'Thirteen' in readme and 'seven' in readme)
    namespace = {'__builtins__': {'tuple': tuple, 'str': str, 'range': range}}
    for key, name in [('mimic_selection', 'CARDIAC_PREFIXES'), ('mimic_labels', 'MACE_CODES')]:
        namespace[name] = eval(compile(ast.Expression(assignments(trees[key], name)), '<literal-audit>', 'eval'), namespace)
    cardiac = namespace['CARDIAC_PREFIXES']
    components = namespace['MACE_CODES']
    check('cardiac_prefixes_exact', list(cardiac) == contract['rules']['MIMIC']['selection_cardiac_prefixes'])
    check('component_prefixes_exact', components == contract['rules']['MIMIC']['component_prefixes'])
    check('all_non_death_prefixes_nested', all((item.startswith(cardiac) for values in components.values() for item in values)))
    check('42842_omitted_in_source_and_report', '42842' not in components['Heart_Failure'] and 'omits `42842`' in readme and ('omitted 42842' in insertions))
    check('4111_duplicates_MI_and_angina', '4111' in components['MI'] and '4111' in components['Unstable_Angina'] and ('also matches the original MI family' in readme) and ('also matches the MI category' in insertions))
    for source_key, variable in [('mimic_features', 'MODEL_FEATURES'), ('mimic_main_model', 'FEATS_43_EN')]:
        names = ast.literal_eval(assignments(trees[source_key], variable))
        check('current_main_flag_absent:' + source_key, len(names) == 43 and 'Admission_has_cardiac' not in names)
    old_demo = ast.literal_eval(assignments(trees['mimic_old_model'], 'demo_cols'))
    check('old08_cardio_flag_explicit', 'Admission_has_cardiac' in old_demo)
    source01 = Path(register['mimic_selection']['path']).read_text(encoding='utf-8-sig')
    check('flag_assignment_exact', 'adm_has_cardiac = cohort_adm["HADM_ID"].isin(cardiac_hadm_ids)' in source01 and 'cohort_adm["Admission_has_cardiac"] = adm_has_cardiac.astype(int)' in source01)
    check('WHU_case_defect_synthetic_verified', any((row['synthetic_case'] == 'MI_upper' and row['SCH'] is True and (row['WHU'] is False) for row in synthetic['text_tests'])))
    check('negation_not_parsed_synthetic_verified', any((row['synthetic_case'] == 'negated_infarction' and row['SCH'] is True and (row['WHU'] is True) for row in synthetic['text_tests'])))
    check('HF_absent_synthetic_verified', any((row['synthetic_case'] == 'heart_failure_only' and row['SCH'] is False and (row['WHU'] is False) for row in synthetic['text_tests'])))
    check('synthetic_28_count', len(synthetic['text_tests']) + len(synthetic['ICD_tests']) == 28 and '28 synthetic' in readme)
    for center, expected_n, expected_pos in [('SCH', 38509, 1323), ('WHU', 2562, 281)]:
        item = agg['China'][center]
        check('aggregate_china:' + center, item['units'] == expected_n and item['positive_labels'] == expected_pos)
        check('counts_in_README:' + center, f'{expected_n:,}' in readme and f'{expected_pos:,}' in readme)
        check('counts_in_insertions:' + center, f'{expected_n:,}' in insertions and f'{expected_pos:,}' in insertions)
    mimic = agg['MIMIC']
    expected = {'raw_diagnosis_rows_read': 651047, 'raw_admission_rows_read': 58976, 'reconstructed_selected_patients': 7887, 'reconstructed_selected_admissions': 11622, 'model_survivor_admissions': 9946, 'model_survivor_patients': 6783, 'model_survivor_positive_labels': 4462, 'model_mace_positive_with_cardiac_flag': 4462, 'model_cardiac_flag_positive_mace_negative': 5023}
    for key, number in expected.items():
        check('aggregate_mimic:' + key, mimic[key] == number)
        check('count_in_README:' + key, f'{number:,}' in readme)
    check('subtype_sets_reproduced', mimic['component_set_disagreements'] == 0)
    check('all_binary_reproduction_checks_passed', agg['all_reproduction_checks_passed'])
    check('SCH_cache_boundary_disclosed', 'not a new raw diagnostic join' in readme)
    check('WHU_selection_not_replayed_disclosed', 'was not independently replayed' in readme)
    check('source_timing_limits_disclosed', 'not component-specific event onset' in readme)
    check('not_same_clinical_task', contract['comparability_gate']['equivalent_incident_MACE_task'] is False and 'not validation of one incident-MACE prediction task' in readme)
    check('no_DOCX_integration_claim', 'NOT INTEGRATED' in insertions and 'has not opened or modified any' in insertions and ('Actions (Not Yet Integrated)' in insertions))
    check('proposed_actions_guard', 'may be adopted only after' in insertions)
    check('historical_AUC_labels_not_changed', 'do not change any original AUC or label' in readme)
    inputs = [ROOT / name for name in ['README.md', 'manuscript_insertions.md', 'aggregate_label_checks.json', 'label_contract.json', 'source_register.json', 'synthetic_boundary_tests.json', 'run_manifest.json', 'audit_label_contract.py', 'verify_reporting.py']]
    report = {'status': 'passed' if all((item['passed'] for item in checks)) else 'failed', 'checks_total': len(checks), 'checks_passed': sum((item['passed'] for item in checks)), 'checks': checks, 'inputs': [{'path': p.as_posix()} for p in inputs], 'boundary': 'Independent reporting/source consistency check; not a repeat clinical adjudication or model validation.', 'draft_integration_status': 'not_integrated', 'no_DOCX_modified': True}
    (ROOT / 'reporting_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'failed': [item['check'] for item in checks if not item['passed']]}))
    assert report['status'] == 'passed'
if __name__ == '__main__':
    main()
