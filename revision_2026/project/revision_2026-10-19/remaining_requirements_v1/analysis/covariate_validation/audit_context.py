"""Read-only clinical-date and ecological-variable provenance audit.

Patient keys and source values are used only in memory. Private outputs retain
positional row indices and dates, never patient identifiers or diagnostic text.
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
REGION = PROJECT / 'revision_2026-10-16/reviewer_priority_analyses_v1/regional'
RAW = {'SCH': Path(_release_path('china', 'SCH_processed_data.csv')), 'WHU': Path(_release_path('china', 'WHU_Cancer_processed_data.csv'))}
LABELS = Path(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'))
CITY = Path(_release_path('china', 'city_database.xlsx'))
KEY = '病案号'
LAB_TIME = '检查时间'
ADMIT = '入院时间'
DISCHARGE = '出院时间'
YEAR = '年份'
AREA = '地区'
REGIONAL = ['人均地区生产总值(元)', '职工平均工资(元)', '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)']

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def read(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def dates(series):
    return pd.to_datetime(series, format='mixed', errors='coerce')

def counts(array):
    return {str(k): int(v) for k, v in pd.Series(array).value_counts(dropna=False).sort_index().items()}

def audit_cohort(cohort, labels):
    local = labels.loc[labels.Center.eq(cohort)].reset_index(drop=True)
    with np.load(BASE / f'private/prepared/{cohort}.npz', allow_pickle=False) as z:
        assert np.array_equal(local.MACE.to_numpy(), z['y'])
        assert np.array_equal(np.arange(len(local)), z['groups'])
    header = pd.read_csv(RAW[cohort], nrows=0).columns.tolist()
    clocks = [f for f in (LAB_TIME, ADMIT, DISCHARGE) if f in header]
    geo = [f for f in ('province', 'city', YEAR) if f in header]
    use = [KEY, *clocks, *geo, *REGIONAL]
    raw = pd.read_csv(RAW[cohort], usecols=use, low_memory=False)
    assert raw[KEY].notna().all() and local[KEY].is_unique
    assert set(raw[KEY]) == set(local[KEY])
    for field in REGIONAL:
        raw[field] = pd.to_numeric(raw[field], errors='coerce')
    clock_info = []
    for field in clocks:
        raw[field] = dates(raw[field])
        s = raw[field]
        clock_info.append({'field': field, 'records': len(raw), 'parseable': int(s.notna().sum()), 'year_counts': counts(s.dt.year.dropna().astype(int)), 'patients_with_any_missing': int(s.isna().groupby(raw[KEY]).any().sum()), 'patients_with_any_value': int(s.notna().groupby(raw[KEY]).any().sum())})
    size = raw.groupby(KEY).size().reindex(local[KEY])
    valid_dates = raw[LAB_TIME].between(pd.Timestamp('1970-01-01'), pd.Timestamp('2026-10-19'))
    complete = valid_dates.groupby(raw[KEY]).all().reindex(local[KEY]).to_numpy()
    mins = raw.groupby(KEY)[LAB_TIME].min().reindex(local[KEY]).reset_index(drop=True)
    maxs = raw.groupby(KEY)[LAB_TIME].max().reindex(local[KEY]).reset_index(drop=True)
    valid_start = mins[complete].sort_values()
    quantile_start = valid_start.iloc[int(np.floor(0.8 * (len(valid_start) - 1)))]
    cutoff = pd.Timestamp(quantile_start.year, quantile_start.month, 1)
    train = complete & (maxs < cutoff).to_numpy()
    test = complete & (mins >= cutoff).to_numpy()
    crossing = complete & ~train & ~test
    y = local.MACE.to_numpy(dtype=np.int64)
    role = np.full(len(local), -1, dtype=np.int8)
    role[train], role[test], role[crossing] = (1, 0, -2)
    all_discharge_complete = raw[DISCHARGE].notna().groupby(raw[KEY]).all().reindex(local[KEY]).to_numpy()
    maxdis = raw.groupby(KEY)[DISCHARGE].max().reindex(local[KEY]).reset_index(drop=True)
    all_train_discharge_before = all_discharge_complete & (maxdis < cutoff).to_numpy()
    private = ROOT / f'private/{cohort}_date_roles.npz'
    private.parent.mkdir(parents=True, exist_ok=True)
    with private.open('xb') as stream:
        np.savez_compressed(stream, groups=np.arange(len(y)), y=y, roles=role, min_lab_day=mins.to_numpy(dtype='datetime64[D]'), max_lab_day=maxs.to_numpy(dtype='datetime64[D]'), full_discharge_before_cutoff=all_train_discharge_before, record_count=size.to_numpy(dtype=np.int64))
    temporal = {'rule_frozen_before_any_prediction_evaluation': "One boundary at the first day of the month containing the outcome-blind 80th percentile (lower order statistic) of each complete patient's first laboratory timestamp. No alternate boundary searched.", 'cutoff': cutoff.date().isoformat(), 'complete_lab_dates': int(complete.sum()), 'training_patients': int(train.sum()), 'training_positive_labels': int(y[train].sum()), 'test_patients': int(test.sum()), 'test_positive_labels': int(y[test].sum()), 'crossing_patients_excluded': int(crossing.sum()), 'crossing_positive_labels': int(y[crossing].sum()), 'invalid_or_missing_date_patients_excluded': int((~complete).sum()), 'training_patients_with_all_discharge_dates_before_cutoff': int((train & all_train_discharge_before).sum()), 'training_patients_with_missing_or_later_discharge': int((train & ~all_train_discharge_before).sum()), 'patient_overlap': 0, 'future_lab_records_in_training': 0, 'label_capture_time_known': False, 'permitted_interpretation': 'Retrospective acquisition-era extrapolation of original diagnosis labels only; not prospective temporal validation, fixed-index risk prediction, or pre-event warning.', 'selection_limitation': 'Excluding patients with laboratory records spanning the boundary preferentially removes longitudinally observed patients. The cohorts differ in observation coverage and retained mix.', 'private_roles': identity(private)}
    registry = {'cohort': cohort, 'rows': len(raw), 'patients': len(local), 'positive_labels': int(y.sum()), 'clock_audit': clock_info, 'temporal_feasibility': temporal, 'clinical_schema_scope': 'Only the processed-model source schema is assessed here; original NJM clinical forms are independently audited by source_evidence.', 'diagnosis_fields': [f for f in header if any((t in f for t in ('诊断', '病理')))], 'explicit_treatment_stage_or_drug_header_fields': [f for f in header if any((t in f for t in ('分期', 'TNM', '化疗', '放疗', '靶向', '免疫治疗', '药物', '药品', '用药', '入院病情')))], 'geography_fields': geo}
    if YEAR in raw:
        year = pd.to_numeric(raw[YEAR], errors='coerce')
        registry['stored_year_audit'] = {'stored_year_counts': counts(year.dropna().astype(int)), 'records_with_stored_year_equal_lab_year': int(year.eq(raw[LAB_TIME].dt.year).sum()), 'records_with_stored_year_equal_admission_year': int(year.eq(raw[ADMIT].dt.year).sum()), 'patients_with_multiple_stored_years': int(raw.groupby(KEY)[YEAR].nunique().gt(1).sum())}
    return (registry, raw)

def city_audit(raws):
    book = pd.ExcelFile(CITY)
    reports = []
    for sheet in book.sheet_names:
        head = pd.read_excel(book, sheet_name=sheet, nrows=0).columns.tolist()
        if not set([YEAR, AREA, *REGIONAL]) <= set(head):
            continue
        table = pd.read_excel(book, sheet_name=sheet, usecols=[YEAR, AREA, *REGIONAL])
        table[AREA] = table[AREA].astype('string').str.strip()
        table[YEAR] = pd.to_numeric(table[YEAR], errors='coerce')
        for field in REGIONAL:
            table[field] = pd.to_numeric(table[field], errors='coerce')
        key_unique = ~table.duplicated([AREA, YEAR], keep=False)
        eligible = table.loc[key_unique & table[YEAR].notna() & table[AREA].notna()].copy()
        item = {'sheet': sheet, 'rows': len(table), 'year_counts': counts(table[YEAR].dropna().astype(int)), 'duplicate_city_year_rows': int((~key_unique).sum()), 'cohorts': []}
        complete_table = table.dropna(subset=REGIONAL)
        tuple_lookup = {}
        for row in complete_table.itertuples(index=False, name=None):
            mapping = dict(zip(complete_table.columns, row))
            signature = tuple((float(mapping[f]) for f in REGIONAL))
            tuple_lookup.setdefault(signature, []).append((mapping[AREA], mapping[YEAR]))
        for cohort, raw in raws.items():
            complete = raw[REGIONAL].notna().all(axis=1)
            vectors = raw.loc[complete, REGIONAL].value_counts(dropna=False)
            matches, unique_city_year_matches, same_labyear, matched_years = (0, 0, 0, {})
            for sig, count in vectors.items():
                match = tuple_lookup.get(tuple((float(v) for v in sig)), [])
                if match:
                    matches += int(count)
                    years = sorted(set((int(pair[1]) for pair in match)))
                    if len(set(match)) == 1:
                        unique_city_year_matches += int(count)
                    for year in years:
                        matched_years[str(year)] = matched_years.get(str(year), 0) + int(count)
            report = {'cohort': cohort, 'raw_rows': len(raw), 'complete_five_value_rows': int(complete.sum()), 'complete_distinct_vectors': len(vectors), 'complete_rows_with_exact_vector_in_sheet': matches, 'complete_rows_with_unique_city_year_vector_match': unique_city_year_matches, 'matched_year_membership_counts_not_mutually_exclusive': matched_years, 'numeric_match_establishes_patient_geography': False}
            if {'city', YEAR} <= set(raw.columns):
                left = raw[['city', YEAR, LAB_TIME, *REGIONAL]].copy()
                left['city'] = left['city'].astype('string').str.strip()
                left[YEAR] = pd.to_numeric(left[YEAR], errors='coerce')
                merged = left.merge(eligible, how='left', left_on=['city', YEAR], right_on=[AREA, YEAR], validate='many_to_one', suffixes=('_model', '_database'), indicator=True)
                matched = merged['_merge'].eq('both').to_numpy()
                report['exact_stored_city_year_join'] = {'matched_records': int(matched.sum()), 'per_field': []}
                for field in REGIONAL:
                    a, b = (merged[field + '_model'].to_numpy(dtype=float), merged[field + '_database'].to_numpy(dtype=float))
                    both = np.isfinite(a) & np.isfinite(b) & matched
                    report['exact_stored_city_year_join']['per_field'].append({'field': field, 'both_finite_records': int(both.sum()), 'exact_equal_records': int((both & (a == b)).sum()), 'unequal_records': int((both & (a != b)).sum())})
            item['cohorts'].append(report)
        reports.append(item)
    return {'workbook': identity(CITY), 'sheets': reports, 'normalization': 'Exact numeric equality; city strings whitespace-trimmed only. Duplicate database city/year keys excluded from keyed joins. No geographical or year assignment changed.', 'limitations': ['A numeric match establishes consistency with a candidate table, not provenance of the original assignment operation or correctness of patient address linkage.', 'WHU has no stored city/year to validate; fingerprint matches cannot establish patient-level geography.', 'The workbook includes interpolated/imputed sheets. A matching number is not a directly measured individual exposure.']}

def main():
    sources = [RAW['SCH'], RAW['WHU'], LABELS, CITY, BASE / 'evidence/china_preparation.json', REGION / 'assignment_audit.json', REGION / 'verification.json']
    before = [identity(p) for p in sources]
    expected = read(BASE / 'evidence/china_preparation.json')['inputs']
    for p in [*RAW.values(), LABELS]:
        pass
    labels = pd.read_csv(LABELS, usecols=[KEY, 'Center', 'MACE'], low_memory=False)
    reports, raws = ([], {})
    for cohort in ('SCH', 'WHU'):
        print(json.dumps({'stage': 'schema_date_audit', 'cohort': cohort}), flush=True)
        result, raw = audit_cohort(cohort, labels)
        reports.append(result)
        raws[cohort] = raw
    ecological = city_audit(raws)
    assert before == [identity(p) for p in sources]
    output = {'passed': True, 'execution_complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(), 'script': identity(__file__), 'sources': before, 'sources_unchanged': True, 'environment': {'python': sys.version, 'executable': sys.executable, **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'openpyxl')}}, 'cohorts': reports, 'ecological_assignment': ecological, 'patient_identifiers_or_clinical_rows_exported': False, 'models_trained': False, 'outcome_or_prediction_informed_date_choice': False}
    save(ROOT / 'context_audit.json', output)
    print(json.dumps({'passed': True, 'cohorts': [{'cohort': r['cohort'], 'temporal_feasibility': r['temporal_feasibility']} for r in reports], 'report': identity(ROOT / 'context_audit.json')}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
