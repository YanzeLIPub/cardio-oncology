"""
Step 3 — Define MACE Outcomes for Cardio-Oncology Cohort
=========================================================
MACE (Major Adverse Cardiovascular Event) defined as ANY of:
  1. Acute Myocardial Infarction          ICD-9: 410.x, 411.x
  2. Stroke / Cerebrovascular Accident    ICD-9: 430–438.x
  3. Heart Failure (new or decompensated) ICD-9: 428.x
  4. Cardiac Arrest / Ventricular Fibrill ICD-9: 427.1, 427.41, 427.5
  5. In-hospital Cardiac Death            HOSPITAL_EXPIRE_FLAG=1 with cardiac Dx

Additionally records MACE sub-type and the MIMIC sequence number
of the triggering diagnosis (lower = higher clinical priority).

Outputs:
  MIMIC-3/data/mace_labels.csv     — per-admission MACE yes/no + type
  MIMIC-3/data/mace_events.csv     — all raw MACE ICD rows found
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import pandas as pd
import numpy as np
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent / 'data'
print('=' * 62)
print('  Step 3: MACE Definition')
print('=' * 62)
cohort_adm = pd.read_csv(DATA_DIR / 'cohort_admissions.csv', low_memory=False)
icd_cardiac = pd.read_csv(DATA_DIR / 'icd_cardiac_adm.csv')
from pathlib import Path as _P
import gzip
MIMIC_DIR = _P(_release_path('mimic', ''))
print('  Loading DIAGNOSES_ICD for MACE codes ...')
diag = pd.read_csv(MIMIC_DIR / 'DIAGNOSES_ICD.csv.gz', usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM'], dtype={'ICD9_CODE': str}, low_memory=False)
diag['ICD9_CODE'] = diag['ICD9_CODE'].fillna('').str.replace('.', '', regex=False).str.upper()
cohort_hids = set(cohort_adm['HADM_ID'].astype(int))
diag = diag[diag['HADM_ID'].isin(cohort_hids)].copy()
print(f'  Cohort diagnosis rows: {len(diag):,}')
MACE_CODES = {'MI': [f'410{str(s)}' for s in range(10)] + [f'411{str(s)}' for s in range(10)], 'Stroke': [str(c) for c in range(430, 439)], 'Heart_Failure': ['4280', '4281', '42820', '42821', '42822', '42823', '42830', '42831', '42832', '42833', '42840', '42841', '42843', '4289'], 'Cardiac_Arrest': ['4271', '42741', '4275', '42760'], 'Cardiomyopathy': ['4254', '4255', '4257', '4258', '4259'], 'Unstable_Angina': ['4111']}
MACE_PREFIXES = {}
for label, codes in MACE_CODES.items():
    MACE_PREFIXES[label] = codes

def code_matches(code, prefix_list):
    """True if code starts with any prefix in list."""
    return any((code.startswith(p) for p in prefix_list))
print('  Flagging MACE codes ...')
mace_flags = []
for label, prefixes in MACE_PREFIXES.items():
    mask = diag['ICD9_CODE'].apply(lambda c: code_matches(c, prefixes))
    subset = diag[mask].copy()
    subset['MACE_Type'] = label
    mace_flags.append(subset)
mace_events = pd.concat(mace_flags, ignore_index=True) if mace_flags else pd.DataFrame()
mace_events = mace_events.sort_values(['HADM_ID', 'SEQ_NUM']).drop_duplicates(subset=['HADM_ID', 'MACE_Type'])
mace_per_adm = mace_events.groupby('HADM_ID').agg(MACE_Types=('MACE_Type', lambda x: '|'.join(sorted(set(x)))), MACE_SeqNum_min=('SEQ_NUM', 'min')).reset_index()
mace_per_adm['MACE'] = 1
cardiac_deaths = cohort_adm[cohort_adm['HOSPITAL_EXPIRE_FLAG'] == 1].copy()
cardiac_hids_set = set(icd_cardiac['HADM_ID'].astype(int))
cardiac_deaths = cardiac_deaths[cardiac_deaths['HADM_ID'].isin(cardiac_hids_set)]
cardiac_death_hids = set(cardiac_deaths['HADM_ID'].astype(int))
new_mace = pd.DataFrame({'HADM_ID': list(cardiac_death_hids - set(mace_per_adm['HADM_ID'])), 'MACE_Types': 'Cardiac_Death', 'MACE_SeqNum_min': 99, 'MACE': 1})
mace_per_adm = pd.concat([mace_per_adm, new_mace], ignore_index=True)
existing_in_cd = mace_per_adm['HADM_ID'].isin(cardiac_death_hids)
mace_per_adm.loc[existing_in_cd, 'MACE_Types'] += '|Cardiac_Death'
adm_minimal = cohort_adm[['HADM_ID', 'SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG', 'Admission_has_cancer', 'Admission_has_cardiac']].copy()
adm_minimal['HADM_ID'] = adm_minimal['HADM_ID'].astype(int)
mace_per_adm['HADM_ID'] = mace_per_adm['HADM_ID'].astype(int)
mace_labels = adm_minimal.merge(mace_per_adm, on='HADM_ID', how='left')
mace_labels['MACE'] = mace_labels['MACE'].fillna(0).astype(int)
mace_labels['MACE_Types'] = mace_labels['MACE_Types'].fillna('None')
mace_labels['MACE_SeqNum_min'] = mace_labels['MACE_SeqNum_min'].fillna(np.nan)
total_adm = len(mace_labels)
mace_adm = mace_labels['MACE'].sum()
cancer_adm = mace_labels['Admission_has_cancer'].sum()
mace_in_ca = mace_labels[mace_labels['Admission_has_cancer'] == 1]['MACE'].sum()
print(f'\n  MACE Summary:')
print(f'    Total cohort admissions:        {total_adm:,}')
print(f'    Admissions with cancer Dx:      {cancer_adm:,}')
print(f'    Admissions with MACE:           {mace_adm:,}  ({mace_adm / total_adm * 100:.1f}%)')
print(f'    MACE in cancer admissions:      {mace_in_ca:,}')
print(f'\n  MACE type breakdown:')
for mtype, cnt in mace_events.groupby('MACE_Type').size().sort_values(ascending=False).items():
    print(f'    {mtype:<25} {cnt:,}')
print(f'    Cardiac_Death (IH)            {len(cardiac_death_hids):,}')
pt_mace = mace_labels.groupby('SUBJECT_ID')['MACE'].max().reset_index().rename(columns={'MACE': 'Ever_MACE'})
mace_pts = pt_mace['Ever_MACE'].sum()
total_pts = len(pt_mace)
print(f'\n  Patient-level: {mace_pts:,}/{total_pts:,} ever experienced MACE ({mace_pts / total_pts * 100:.1f}%)')
mace_labels.to_csv(DATA_DIR / 'mace_labels.csv', index=False)
mace_events.to_csv(DATA_DIR / 'mace_events.csv', index=False)
patient_mace_path = DATA_DIR / 'patient_mace_summary.csv'
pt_mace.to_csv(patient_mace_path, index=False)
print(f'\n✓ Saved:')
print(f'  {DATA_DIR}/mace_labels.csv     ({len(mace_labels):,} rows)')
print(f'  {DATA_DIR}/mace_events.csv     ({len(mace_events):,} MACE event rows)')
print(f'  {DATA_DIR}/patient_mace_summary.csv')
print('\nStep 3 complete.\n')
