"""
Step 1 — Identify Cardio-Oncology Patients in MIMIC-III
========================================================
Cardio-oncology cohort definition:
  • Has at least one malignant cancer diagnosis (ICD-9: 140–208)
  • Has at least one cardiovascular condition (ICD-9: 390–459)
  OR
  • Is admitted to ICU with cancer and subsequently develops cardiac events

Outputs:
  MIMIC-3/data/cohort_admissions.csv   — admission-level cohort
  MIMIC-3/data/cohort_patients.csv     — patient-level summary
  MIMIC-3/data/icd_cancer_adm.csv     — all cancer ICD codes for cohort
  MIMIC-3/data/icd_cardiac_adm.csv    — all cardiac ICD codes for cohort
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import gzip, sys
import pandas as pd
import numpy as np
from pathlib import Path
MIMIC_DIR = Path(_release_path('mimic', ''))
OUT_DIR = Path(__file__).resolve().parent / 'data'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CANCER_PREFIXES = tuple([str(i).zfill(3) for i in range(140, 209)] + ['V10'])
CARDIAC_PREFIXES = tuple([str(i) for i in range(390, 460)] + ['V45', 'V53'])

def load_gz(name, usecols=None, dtype=None):
    path = MIMIC_DIR / f'{name}.csv.gz'
    print(f'  Loading {name} ...', end=' ', flush=True)
    df = pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)
    print(f'{len(df):,} rows')
    return df
print('=' * 62)
print('  MIMIC-III Cardio-Oncology Patient Identification')
print('=' * 62)
patients = load_gz('PATIENTS', usecols=['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG'])
admissions = load_gz('ADMISSIONS', usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'ETHNICITY'])
diag = load_gz('DIAGNOSES_ICD', usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM'], dtype={'ICD9_CODE': str})
diag['ICD9_CODE'] = diag['ICD9_CODE'].fillna('').str.replace('.', '', regex=False).str.upper().str.strip()
print('\nClassifying ICD codes ...')
diag['is_cancer'] = diag['ICD9_CODE'].str.startswith(CANCER_PREFIXES)
diag['is_cardiac'] = diag['ICD9_CODE'].str.startswith(CARDIAC_PREFIXES)
cancer_adm = diag[diag['is_cancer']].copy()
cardiac_adm = diag[diag['is_cardiac']].copy()
cancer_hadm_ids = set(cancer_adm['HADM_ID'])
cardiac_hadm_ids = set(cardiac_adm['HADM_ID'])
cancer_subjects = set(cancer_adm['SUBJECT_ID'])
cardiac_subjects = set(cardiac_adm['SUBJECT_ID'])
cardio_onc_subjects = cancer_subjects & cardiac_subjects
print(f'\n  Cancer admissions:  {len(cancer_hadm_ids):,}')
print(f'  Cardiac admissions: {len(cardiac_hadm_ids):,}')
print(f'  Cancer patients:    {len(cancer_subjects):,}')
print(f'  Cardiac patients:   {len(cardiac_subjects):,}')
print(f'  Cardio-oncology patients (both): {len(cardio_onc_subjects):,}')
cohort_adm = admissions[admissions['SUBJECT_ID'].isin(cardio_onc_subjects)].copy()
cohort_adm = cohort_adm.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']], on='SUBJECT_ID', how='left')
cohort_adm['ADMITTIME'] = pd.to_datetime(cohort_adm['ADMITTIME'], errors='coerce')
cohort_adm['DOB'] = pd.to_datetime(cohort_adm['DOB'], errors='coerce')
cohort_adm['Age'] = (cohort_adm['ADMITTIME'].dt.year - cohort_adm['DOB'].dt.year).clip(upper=91).astype(float)
cohort_adm['Sex_M'] = (cohort_adm['GENDER'] == 'M').astype(int)
adm_has_cancer = cohort_adm['HADM_ID'].isin(cancer_hadm_ids)
adm_has_cardiac = cohort_adm['HADM_ID'].isin(cardiac_hadm_ids)
cohort_adm['Admission_has_cancer'] = adm_has_cancer.astype(int)
cohort_adm['Admission_has_cardiac'] = adm_has_cardiac.astype(int)
print(f'\n  Cohort admissions:  {len(cohort_adm):,}')
print(f'    of which cancer admissions: {adm_has_cancer.sum():,}')
print(f'    of which cardiac admissions: {adm_has_cardiac.sum():,}')
patient_summary = cohort_adm.groupby('SUBJECT_ID').agg(N_Admissions=('HADM_ID', 'nunique'), First_Admit=('ADMITTIME', 'min'), Gender=('GENDER', 'first'), Age_at_first_admit=('Age', 'first'), In_hospital_death=('HOSPITAL_EXPIRE_FLAG', 'max'), N_Cancer_Adm=('Admission_has_cancer', 'sum'), N_Cardiac_Adm=('Admission_has_cardiac', 'sum')).reset_index()
print(f'\n  Patient-level cohort: {len(patient_summary):,}')
print(f"  In-hospital deaths:   {patient_summary['In_hospital_death'].sum():,}")
print(f"  Sex distribution:\n{cohort_adm.groupby('SUBJECT_ID')['GENDER'].first().value_counts().to_string()}")
print(f"  Age (at first admission):\n  mean={patient_summary['Age_at_first_admit'].mean():.1f}  median={patient_summary['Age_at_first_admit'].median():.1f}")
cohort_adm.to_csv(OUT_DIR / 'cohort_admissions.csv', index=False)
patient_summary.to_csv(OUT_DIR / 'cohort_patients.csv', index=False)
cancer_adm_cohort = cancer_adm[cancer_adm['SUBJECT_ID'].isin(cardio_onc_subjects)]
cardiac_adm_cohort = cardiac_adm[cardiac_adm['SUBJECT_ID'].isin(cardio_onc_subjects)]
cancer_adm_cohort.to_csv(OUT_DIR / 'icd_cancer_adm.csv', index=False)
cardiac_adm_cohort.to_csv(OUT_DIR / 'icd_cardiac_adm.csv', index=False)
print('\n✓ Saved:')
print(f'  {OUT_DIR}/cohort_admissions.csv   ({len(cohort_adm):,} rows)')
print(f'  {OUT_DIR}/cohort_patients.csv     ({len(patient_summary):,} rows)')
print(f'  {OUT_DIR}/icd_cancer_adm.csv      ({len(cancer_adm_cohort):,} rows)')
print(f'  {OUT_DIR}/icd_cardiac_adm.csv     ({len(cardiac_adm_cohort):,} rows)')
print('\nStep 1 complete.\n')
