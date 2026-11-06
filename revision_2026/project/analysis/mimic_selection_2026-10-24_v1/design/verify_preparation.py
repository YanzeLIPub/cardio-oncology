"""Independent source/mask/matrix checks; aggregate output only, no model fitting."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
PREP = HERE.parent / 'preparation'
SOURCE = Path(_release_path('pipeline', ''))
RAW = SOURCE / 'mimic-iii-clinical-database-1.4'
REGISTRY = {'MI': ['410' + str(i) for i in range(10)] + ['411' + str(i) for i in range(10)], 'Stroke': [str(i) for i in range(430, 439)], 'Heart_Failure': ['4280', '4281', '42820', '42821', '42822', '42823', '42830', '42831', '42832', '42833', '42840', '42841', '42843', '4289'], 'Cardiac_Arrest': ['4271', '42741', '4275', '42760'], 'Cardiomyopathy': ['4254', '4255', '4257', '4258', '4259'], 'Unstable_Angina': ['4111']}

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main():
    output = HERE / 'independent_preparation_verification.json'
    assert not output.exists(), 'Refuse to overwrite independent verification'
    report_path = PREP / 'preparation_report.json'
    report = _release_resolve_config(json.loads(report_path.read_text(encoding='utf-8')))
    contract = _release_resolve_config(json.loads((PREP / 'preparation_contract.json').read_text(encoding='utf-8')))
    checks = []

    def check(condition, name):
        assert bool(condition), name
        checks.append({'name': name, 'passed': True})
    check(report['passed'] and report['execution_complete'], 'producer completed preparation')
    before = {name: identity(item['path']) for name, item in report['inputs'].items()}
    for name, item in before.items():
        check(item == report['inputs'][name], 'unchanged producer input identity: ' + name)
    for item in report['outputs']:
        check(identity(item['path']) == item, 'producer output identity: ' + Path(item['path']).name)
    check(identity(report['script']['path']) == report['script'], 'producer code identity')
    check(contract['label']['registry'] == REGISTRY, 'independently transcribed historical code registry')
    with np.load(PREP / 'private/full_matrix.npz', allow_pickle=False) as z:
        data = {name: z[name].copy() for name in z.files}
    with np.load(PREP / 'private/historical43_audit.npz', allow_pickle=False) as z:
        history = {name: z[name].copy() for name in z.files}
    keys, y, groups, X = (data['keys'], data['y'], data['groups'], data['X'])
    names = data['feature_names'].tolist()
    old_names = history['feature_names'].tolist()
    check(X.shape == (11622, 41), 'full primary matrix shape')
    check(len(np.unique(keys)) == len(keys), 'unique admission keys')
    check(names == [n for n in old_names if n not in {'AST', 'HDL Cholesterol'}], 'uniform 43-to-41 exclusion')
    check(not set(names) & {'MACE', 'HADM_ID', 'SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG', 'Admission_has_cardiac', 'Admission_has_cancer', 'DISCHTIME', 'ADMITTIME', 'LOS', 'DOD'}, 'forbidden predictors absent')
    check(not np.isinf(X).any(), 'no infinite predictor values')
    check(np.array_equal(keys, history['keys']), 'historical audit row identities')
    check(np.array_equal(y, history['y']) and np.array_equal(groups, history['groups']), 'historical audit labels and groups')
    adm = pd.read_csv(RAW / 'ADMISSIONS.csv.gz', usecols=['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG'])
    diag = pd.read_csv(RAW / 'DIAGNOSES_ICD.csv.gz', dtype={'ICD9_CODE': str}, usecols=['HADM_ID', 'SUBJECT_ID', 'ICD9_CODE'])
    patients = pd.read_csv(RAW / 'PATIENTS.csv.gz', usecols=['SUBJECT_ID', 'DOB', 'GENDER']).set_index('SUBJECT_ID')
    for name in ('ADMITTIME', 'DISCHTIME'):
        adm[name] = pd.to_datetime(adm[name], errors='raise')
    rows = adm.set_index('HADM_ID').loc[keys]
    check(np.array_equal(np.unique(rows.SUBJECT_ID, return_inverse=True)[1], groups), 'global patient grouping independently rebuilt')
    check(len(np.unique(groups)) == 7887, 'full unique patient count')
    check(np.array_equal(diag.HADM_ID.map(adm.set_index('HADM_ID').SUBJECT_ID), diag.SUBJECT_ID), 'raw diagnostic patient-admission links')
    cleaned = diag.ICD9_CODE.fillna('').str.replace('.', '', regex=False).str.upper().str.strip()
    cancer = cleaned.str.match('^(1[4-9][0-9]|20[0-8]|V10)')
    cardiac = cleaned.str.match('^(39[0-9]|4[0-5][0-9]|V45|V53)')
    eligible_subjects = set(diag.loc[cancer, 'SUBJECT_ID']) & set(diag.loc[cardiac, 'SUBJECT_ID'])
    expected_keys = set(adm.loc[adm.SUBJECT_ID.isin(eligible_subjects), 'HADM_ID'])
    check(set(keys) == expected_keys, 'source ever-eligible universe rebuilt without outcome filter')
    family_ids = {'cancer': set(diag.loc[cancer, 'HADM_ID']), 'cardiac': set(diag.loc[cardiac, 'HADM_ID'])}
    first_ends = {'cancer': {}, 'cardiac': {}}
    valid_count = 0
    for h, subject, start, end in adm[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME']].itertuples(index=False, name=None):
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        valid_count += 1
        for family, ids in family_ids.items():
            if h in ids and (subject not in first_ends[family] or end < first_ends[family][subject]):
                first_ends[family][subject] = end
    check(len(adm) - valid_count == 98, 'invalid raw history stays excluded')
    prior = {}
    for family in family_ids:
        prior[family] = np.array([subject in first_ends[family] and (not pd.isna(start)) and (first_ends[family][subject] < start) for subject, start in rows[['SUBJECT_ID', 'ADMITTIME']].itertuples(index=False, name=None)])
        check(np.array_equal(prior[family], data['prior_completed_' + family]), 'prior completed ' + family + ' independently rebuilt')
        check(np.array_equal(np.isin(keys, list(family_ids[family])), data['same_admission_' + family]), 'same-admission ' + family + ' independently rebuilt')
    prior_both = prior['cancer'] & prior['cardiac']
    survivor = rows.HOSPITAL_EXPIRE_FLAG.to_numpy() == 0
    current_valid = (rows.ADMITTIME.notna() & rows.DISCHTIME.notna() & (rows.DISCHTIME >= rows.ADMITTIME)).to_numpy()
    check(np.array_equal(current_valid, data['valid_current_admission']), 'current validity flag is independently correct')
    check(np.array_equal(prior_both, data['prior_completed_valid']), 'both prior-family mask')
    masks = {'ever_all': np.ones(len(keys), dtype=bool), 'ever_survivor': survivor, 'prior_all': prior_both, 'prior_survivor': prior_both & survivor}
    for arm, mask in masks.items():
        check(np.array_equal(mask, data[arm]), 'arm mask: ' + arm)
    check(int(masks['prior_survivor'].sum()) == 2194, 'corrected prior survivor count')
    check(int((survivor & ~current_valid).sum()) == 4, 'invalid current stays retained in original survivors')
    label_codes = diag.ICD9_CODE.fillna('').str.replace('.', '', regex=False).str.upper()
    union = np.zeros(len(keys), dtype=bool)
    components = {}
    for name, prefixes in REGISTRY.items():
        matched = label_codes.map(lambda c: any((c.startswith(prefix) for prefix in prefixes)))
        flags = np.isin(keys, diag.loc[matched, 'HADM_ID'].to_numpy())
        components[name] = flags
        union |= flags
        check(np.array_equal(flags, data['component_' + name]), 'source label component: ' + name)
    death_proxy = ~survivor & np.isin(keys, list(family_ids['cardiac']))
    check(np.array_equal(union, data['coded_component_union']), 'non-death code union')
    check(np.array_equal(death_proxy, data['death_with_broad_cardiac_proxy']), 'historical death proxy')
    check(np.array_equal(y, (union | death_proxy).astype(y.dtype)), 'all labels independently reconstructed')
    labels = pd.read_csv(SOURCE / 'MIMIC-3/data/mace_labels.csv').set_index('HADM_ID').loc[keys]
    check(np.array_equal(y, labels.MACE), 'all stored labels matched')
    check(int(y.sum()) == 6105 and int(y[survivor].sum()) == 4462, 'full and survivor positive counts')
    check(int((~survivor).sum()) == 1676 and int(y[~survivor].sum()) == 1643, 'excluded death admissions and positive count')
    demographic = patients.loc[rows.SUBJECT_ID]
    birth_year = pd.to_datetime(demographic.DOB, errors='coerce').dt.year.to_numpy()
    age = np.minimum(rows.ADMITTIME.dt.year.to_numpy() - birth_year, 91).astype(np.float32)
    sex = demographic.GENDER.map({'M': 1, 'F': 0}).to_numpy(dtype=np.float32)
    check(np.array_equal(age, X[:, names.index('Age')], equal_nan=True), 'all admission ages rebuilt from raw source')
    check(np.array_equal(sex, X[:, names.index('Sex_M')], equal_nan=True), 'all recorded sex values rebuilt from raw source')
    cached = pd.read_csv(SOURCE / 'MIMIC-3/data/features_model.csv').set_index('HADM_ID').loc[keys]
    cached43 = cached[old_names].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    check(np.array_equal(cached43, history['X_historical43_audit_only'], equal_nan=True), 'all 43 nominal historical cells matched')
    check(np.array_equal(cached[names].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32), X, equal_nan=True), 'all 41 primary cached cells matched')
    labs = pd.read_csv(SOURCE / 'MIMIC-3/data/features_raw.csv', usecols=['HADM_ID', 'Feature', 'VALUENUM'])
    lookup = {int(key): i for i, key in enumerate(keys)}
    row_codes = np.array([lookup[int(key)] for key in labs.HADM_ID], dtype=np.int64)
    rebuilt = np.full((len(keys), len(old_names)), np.nan, dtype=float)
    rebuilt[:, old_names.index('Age')] = age
    rebuilt[:, old_names.index('Sex_M')] = sex
    groups_replayed = 0
    for feature in sorted(labs.Feature.unique()):
        selected = (labs.Feature == feature).to_numpy()
        r = row_codes[selected]
        v = labs.loc[selected, 'VALUENUM'].to_numpy(dtype=float)
        check(np.isfinite(v).all(), 'finite cached laboratory records: ' + feature)
        order = np.argsort(r, kind='stable')
        r, v = (r[order], v[order])
        starts = np.r_[0, np.flatnonzero(r[1:] != r[:-1]) + 1]
        ends = np.r_[starts[1:], len(r)]
        for start, end in zip(starts, ends):
            rebuilt[r[start], old_names.index(feature)] = float(np.median(v[start:end]))
        groups_replayed += len(starts)
    rebuilt[:, old_names.index('Globulin')] = rebuilt[:, old_names.index('Total Protein')] - rebuilt[:, old_names.index('Albumin')]
    check(np.array_equal(rebuilt.astype(np.float32), history['X_historical43_audit_only'], equal_nan=True), 'independent NumPy medians and derived globulin match every historical cell')
    check(np.array_equal(rebuilt[:, [old_names.index(n) for n in names]].astype(np.float32), X, equal_nan=True), 'independent NumPy reconstruction matches every primary cell')
    paper = pd.read_csv(SOURCE / 'Manuscript/v13/paper_predictions_9946.csv')
    check(np.array_equal(keys[survivor], paper.HADM_ID), 'survivor admission order equals original paper')
    check(np.array_equal(rows.SUBJECT_ID.to_numpy()[survivor], paper.SUBJECT_ID), 'survivor patient keys equal original paper')
    check(np.array_equal(y[survivor], paper.MACE), 'survivor labels equal original paper')
    with np.load(HERE.parents[2] / 'revision_2026-10-15/original_v5_independent_validation_v1/private/prepared/MIMIC.npz', allow_pickle=False) as old:
        check(np.array_equal(old['X'], rebuilt[survivor].astype(np.float32), equal_nan=True), 'old frozen survivor matrix exactly reproduced')
        check(np.array_equal(old['groups'], np.unique(rows.SUBJECT_ID.to_numpy()[survivor], return_inverse=True)[1]), 'old survivor-only patient factorization reproduced')
    aggregates = {}
    for arm, mask in masks.items():
        counts = {'admissions': int(mask.sum()), 'patients': int(np.unique(groups[mask]).size), 'positive_source_labels': int(y[mask].sum()), 'negative_source_labels': int((y[mask] == 0).sum()), 'death_admissions': int((mask & ~survivor).sum()), 'invalid_current_admissions_retained': int((mask & ~current_valid).sum()), 'source_coded_component_union_positive': int((mask & union).sum()), 'source_death_with_broad_cardiac_proxy_positive': int((mask & death_proxy).sum()), 'source_death_proxy_only_positive': int((mask & death_proxy & ~union).sum())}
        for name, value in counts.items():
            check(value == report['cohorts'][arm][name], 'aggregate ' + arm + ': ' + name)
        aggregates[arm] = counts
    check(before == {name: identity(item['path']) for name, item in report['inputs'].items()}, 'source files unchanged after independent checks')
    result = {'passed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'checks_passed': len(checks), 'checks': checks, 'script': identity(__file__), 'producer_report': identity(report_path), 'producer_code': report['script'], 'matrix_identity': identity(PREP / 'private/full_matrix.npz'), 'cached_laboratory_groups_independently_replayed': groups_replayed, 'cohorts': aggregates, 'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'sources_changed': False, 'models_trained': False, 'patient_rows_or_identifiers_logged': False, 'scope': 'Independent raw admission/diagnosis/demographic reconstruction and cached laboratory NumPy aggregation; no raw LABEVENTS re-extraction or clinical endpoint certification.', 'remaining_limits': ['Prior coded documentation is not event-free or incident-event eligibility.', 'Death proxy is not adjudicated cardiovascular death.', 'Cached full-admission laboratories and remaining mappings are unchanged.', 'Preparation verification does not certify model training or results; these need later checks.']}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({'passed': True, 'checks_passed': len(checks), 'cohorts': aggregates, 'private_identifiers_logged': False}, indent=2))
if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        print(json.dumps({'passed': False, 'error_type': type(error).__name__, 'trace_lines': [frame.lineno for frame in traceback.extract_tb(error.__traceback__)], 'private_exception_values_suppressed': True}), file=sys.stderr)
        raise SystemExit(1)
