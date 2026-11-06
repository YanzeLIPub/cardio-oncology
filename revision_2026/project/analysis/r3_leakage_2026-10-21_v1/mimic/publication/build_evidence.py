"""Build aggregate R3 evidence tables."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import csv
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]

def read(path):
    return _release_resolve_config(json.loads(path.read_text(encoding='utf-8')))

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save_json(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')

def save_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    sources = {'P1': BASE / 'protocol.json', 'C1': BASE / 'china/feature_lineage_audit.json', 'C2': BASE / 'china/demographic_preparation.json', 'C3': BASE / 'china/preparation_verification.json', 'M1': BASE / 'mimic/lineage_audit.json', 'M2': BASE / 'mimic/independent_verification_and_validity_correction.json'}
    evidence = {k: read(v) for k, v in sources.items()}
    source_before = {k: identity(v) for k, v in sources.items()}
    assert evidence['C3']['passed'] and evidence['M2']['passed']
    assert evidence['P1']['no_claim_of_complete_R3_closure']
    assert evidence['C1']['label_derived_input_names_present'] == []
    for cohort in evidence['C1']['cohorts']:
        assert all((v['numeric_nonmissing_rows'] == 0 for v in cohort['direct_text_fields'].values()))
    outputs = [HERE / name for name in ('leakage_dimensions_by_cohort.csv', 'mimic_cohort_selection.csv')]
    if any((path.exists() for path in outputs)):
        raise FileExistsError('Refuse to overwrite publication evidence')
    cohort_sizes = {c['center']: (c['n'], c['positive_labels_preserved']) for c in evidence['C2']['cohorts']}
    cohort_sizes['MIMIC'] = (9946, 4462)
    rows = []

    def add(cohort, dimension, existing, reference, limit, keys):
        n, positive = cohort_sizes[cohort]
        rows.append({'cohort': cohort, 'analysis_unit': 'admission' if cohort == 'MIMIC' else 'patient', 'retained_rows': n, 'positive_recorded_diagnosis_labels': positive, 'leakage_dimension': dimension, 'existing_model_finding': existing, 'new_two_covariate_reference': reference, 'remaining_limitation': limit, 'evidence_keys': keys})
    for cohort in ('SCH', 'WHU'):
        add(cohort, 'Direct label or diagnosis inputs', 'Original 104-column header includes three diagnosis/department text columns; all entries become missing on numeric coercion and then zero. No literal label-derived predictor name was found.', 'Predictor values are age and recorded sex code at the earliest dated source observation; no diagnosis, outcome, laboratory, treatment, regional or utilization variable is allowed.', 'A numerical feature inventory cannot certify that original physiological measurements precede the clinical outcome.', 'C1;C2;C3;P1')
        add(cohort, 'Indirect event or treatment physiology', 'Original laboratory inputs use full-record summaries. Their relationship to disease onset and treatment is unresolved.', 'All laboratory values and laboratory missingness indicators are excluded; no later-record demographic values are used.', 'Performance differences do not measure the amount of leakage and do not validate the original laboratory model.', 'C1;C2;P1')
        add(cohort, 'Time zero and endpoint timing', 'The original target is a recorded ever-diagnosis label; earliest laboratory time does not establish event-free eligibility.', 'Age and source sex code are reconstructed at the earliest parseable recorded check timestamp; conflicts remain missing and later observations do not fill values.', 'Earliest recorded check is not certified onset, result availability, clinical index, or event-free time; no incident prediction claim is made.', 'C1;C2;C3;P1')
        add(cohort, 'Patient and fitting isolation', 'The comparison vector is the prior patient-isolated full-input OOF model, not an apparent in-sample fit.', 'Reuse the frozen five patient roles; select rounds inside training patients and fit preprocessing and models without outer-test labels.', 'Internal patient isolation does not repair original feature timing, source ascertainment, or cohort selection.', 'P1;C2')
        add(cohort, 'Cohort and outcome ascertainment', f'The original {cohort} patient set and recorded-diagnosis labels are retained.', 'All original rows have both reconstructed demographic inputs, so complete-case eligibility changes no patient denominator.', 'Clinical endpoint adjudication, event-free eligibility and complete negative capture remain unverified.', 'C1;C2;P1')
        add(cohort, 'Demographic encoding and inference', 'Original demographic inputs were full-record numeric means.', 'Age is taken at the earliest source timestamp; source sex codes 1 and 2 are encoded as categories 0 and 1 without assigning unverified male/female meanings.', 'This is an outcome-definition-independent reference for the original recorded-diagnosis task, not a negative control or validated future-risk model.', 'C1;C2;P1')
    add('MIMIC', 'Direct label or diagnosis inputs', 'The older 96-column model includes Admission_has_cardiac, which directly overlaps the label definition. Current nominal 43-column and window matrices omit this flag and explicit diagnosis/outcome fields.', 'Only raw-source admission age and registry sex are predictors; all diagnosis, outcome-component, death/discharge and identifier fields are excluded.', 'The demonstrated overlap in the older model prevents a blanket assertion that all historical predictors were independent of outcome definition.', 'M1;M2;P1')
    add('MIMIC', 'Indirect event or treatment physiology', 'Current Creatine Kinase combines total CK and CK-MB; CK-MB contributes 12289 cached measurements across 4199 of 9946 retained admissions. Other acute laboratories may also reflect concurrent illness.', 'All laboratories, including CK-MB, and missingness or observation-density variables are excluded.', 'Physiological association alone is not proof of leakage; clinical pre-event timing remains unproven for the full laboratory model.', 'M1;P1')
    add('MIMIC', 'Time zero and endpoint timing', 'Existing 0/24/48-hour windows use hospital admission and eligible same-encounter ED CHARTTIME records. Diagnosis codes lack onset/POA and LABEVENTS lacks result-availability time.', 'Age is determined at hospital admission and sex comes from PATIENTS; no acute measurement window contributes a predictor.', 'The unchanged concurrent diagnosis target is an acute-care stress test; it is not incident MACE, an ICU-admission prognostic task, or certified event-free prediction.', 'M1;M2;P1')
    add('MIMIC', 'Patient and fitting isolation', 'Original paper CV split admission rows; the comparison here uses the subsequent patient-isolated full-input OOF model.', 'Existing patient roles are reused, with zero patient intersections across all four roles in each of five folds; fitting and round selection remain inside training roles.', 'Patient isolation cannot remove concurrent physiological information or the original cohort eligibility restrictions.', 'M1;P1')
    add('MIMIC', 'Cohort and outcome ascertainment', 'Eligibility requires ever cancer and ever cardiac diagnoses and retains eventual survivors. 1081 of 9946 admissions first qualify for at least one diagnosis family in a later admission.', 'Keep exactly 9946 admissions from 6783 patients and 4462 labels so the reference uses the same evaluation population.', 'Future-dependent eligibility and survivor conditioning remain; only 2194 admissions have both families in valid previously completed admissions, which itself does not establish event-free status.', 'M1;M2;P1')
    add('MIMIC', 'Demographic encoding and inference', 'Source age is admission year minus birth year, capped at 91; recorded sex is from the patient registry.', 'All 19892 reconstructed demographic cells independently match the frozen inputs; unknown sex would remain missing, and no unknown values occur.', 'Demographic associations are permissible predictors, but this reference neither establishes clinical transportability nor certifies historical laboratories as leakage-free.', 'M1;M2;P1')
    save_csv(outputs[0], rows)
    initial = evidence['M1']['cohort_selection']
    corrected = evidence['M2']['prior_discharge_count_correction']
    selection = []

    def select(name, count, denominator, population, note, source):
        selection.append({'finding': name, 'population': population, 'admissions': count['admissions'], 'admission_denominator': denominator, 'percent_of_admission_denominator': round(100 * count['admissions'] / denominator, 2), 'patients': count['patients'], 'positive_admission_labels': count.get('positive_admission_labels', ''), 'interpretation': note, 'evidence_keys': source})
    select('Original source cohort', initial['all_source_cohort'], 11622, 'Before survivor filter', 'All admissions of patients ever coded with both qualifying diagnosis families.', 'M1')
    select('Retained survivor cohort', initial['survivors'], 11622, 'Before survivor filter', '9946 admissions from 6783 patients; unchanged analysis population for the new demographic reference.', 'M1')
    select('Excluded in-hospital deaths', initial['excluded_death_admissions'], 11622, 'Before survivor filter', '1643 of these 1676 excluded admissions carry the original positive composite label; survival is an outcome-period eligibility condition.', 'M1')
    select('First qualifying cancer-code admission starts later', initial['cancer']['qualifying_admission_starts_later'], 9946, 'Retained survivor cohort', 'Admission chronology establishes use of later diagnosis records for cohort eligibility; it does not identify cancer onset.', 'M1')
    select('First qualifying cardiac-code admission starts later', initial['cardiac']['qualifying_admission_starts_later'], 9946, 'Retained survivor cohort', 'All 276 are negative under the concurrent label; this is future-dependent selection, not a positive-label predictor field.', 'M1')
    select('At least one diagnosis family first qualifies in a later admission', evidence['M2']['future_admission_selection_independently_confirmed'], 9946, 'Retained survivor cohort', 'Union of the preceding two rows; independently confirmed.', 'M1;M2')
    select('Both diagnosis families first qualify in later admissions', evidence['M2']['cancer_and_cardiac_both_first_qualify_later'], 9946, 'Retained survivor cohort', 'Overlap between the cancer and cardiac later-admission rows; do not add overlapping patient counts.', 'M2')
    select('Prior completed cancer-code admission with valid stay timing', corrected['prior_completed_cancer_valid_stay'], 9946, 'Retained survivor cohort', 'Qualifying prior admission must satisfy ADMITTIME <= DISCHTIME < current ADMITTIME.', 'M2')
    select('Prior completed cardiac-code admission with valid stay timing', corrected['prior_completed_cardiac_valid_stay'], 9946, 'Retained survivor cohort', 'Prior documentation does not establish clinical event-free status.', 'M2')
    select('Both diagnosis families documented in valid prior completed admissions', corrected['both_prior_completed_valid_stays'], 9946, 'Retained survivor cohort', 'Corrected count 2194 supersedes preliminary 2199; excludes invalid prior stays and does not imply a clinically validated risk set.', 'M2')
    select('Without both diagnosis families in valid prior completed admissions', corrected['not_both_prior_completed_valid_stays'], 9946, 'Retained survivor cohort', 'Complement of the preceding admission count; patients can contribute admissions to both categories.', 'M2')
    select('Retained admissions with discharge before admission', corrected['original_retained_invalid_stays'], 9946, 'Retained survivor cohort', 'Four records; retained for the fixed-population comparison but never treated as valid prior stays in the corrected eligibility counts.', 'M2')
    for label, n, patients in ((1, 4462, 3274), (0, 5023, 4033)):
        select(f'Cardiac admission flag present with label {label}', {'admissions': n, 'patients': patients, 'positive_admission_labels': n if label else 0}, 9946, 'Retained survivor cohort', 'Older 96-column model used this broad diagnosis flag; current nominal 43-column and demographic reference inputs exclude it.', 'M1')
    save_csv(outputs[1], selection)
    print(json.dumps({'outputs': [p.name for p in outputs], 'leakage_rows': len(rows), 'selection_rows': len(selection)}))
if __name__ == '__main__':
    main()
