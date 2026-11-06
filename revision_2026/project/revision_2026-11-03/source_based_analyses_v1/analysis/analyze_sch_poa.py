"""Same-stay coded-diagnosis and POA composition, with original labels retained."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import Counter
import json, re, sys
import numpy as np
import pandas as pd
ROOT = Path(_release_path('project', ''))
OUT = Path(__file__).resolve().parents[1]
AGG = OUT / 'aggregate'
PRIVATE = OUT / 'private'
AGG.mkdir(exist_ok=True)
PRIVATE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'revision_2026-11-03/source_gap_search_v1'))
from scan_sources import key

def day(s):
    return pd.to_datetime(s, errors='coerce', format='mixed').dt.strftime('%Y-%m-%d').fillna('')

def interval(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - r), min(1.0, c + r)]

def main():
    labels = pd.read_csv(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'), usecols=['Center', '病案号', 'MACE'], dtype={'病案号': 'string'})
    labels = labels.loc[labels.Center.eq('SCH')]
    lm = dict(zip(labels['病案号'].map(key), labels.MACE.astype(int)))
    source = Path(_release_path('njm', 'SCH'))
    header = pd.read_csv(source / 'SCH_processed_data.csv', nrows=0).columns
    cols = ['病案号', '入院时间', '出院时间'] + (['住院次数'] if '住院次数' in header else [])
    wide = pd.read_csv(source / 'SCH_processed_data.csv', usecols=cols, dtype='string')
    wide['pid'] = wide['病案号'].map(key)
    wide['ad'] = day(wide['入院时间'])
    wide['di'] = day(wide['出院时间'])
    join = ['pid', 'ad', 'di']
    if '住院次数' in cols:
        wide['num'] = wide['住院次数'].map(key)
        join.append('num')
    known = set(map(tuple, wide.loc[(wide.ad != '') & (wide.di != ''), join].drop_duplicates().to_numpy()))
    pieces = []
    stats = Counter()
    slotcols = []
    for j in range(1, 8):
        prefix = '出院主要诊断' if j == 1 else '出院其他诊断'
        slotcols.extend((prefix + t + str(j) for t in ['编码', '名称', '入院病情']))
    for path in sorted((source / '2021-2025_原始导出').glob('*.xls')):
        f = pd.read_csv(path, sep='\t', encoding='gb18030', dtype='string', keep_default_na=False)
        stats['raw_row_occurrences'] += len(f)
        f['pid'] = f['病案号'].map(key)
        f = f[f.pid.isin(lm)].copy()
        stats['current_patient_raw_rows'] += len(f)
        f['ad'] = day(f['入院时间'])
        f['di'] = day(f['出院时间'])
        f['num'] = f['住院次数'].map(key)
        good = np.array([tuple(row) in known for row in f[join].to_numpy()])
        f = f.loc[good].copy()
        stats['processed_stay_matched_raw_rows'] += len(f)
        pieces.append(f[['pid', 'ad', 'di', 'num'] + slotcols])
        print(json.dumps({'phase': 'SCH diagnosis extraction', 'export': path.name, 'matched_rows': len(f)}), flush=True)
    f = pd.concat(pieces, ignore_index=True).drop_duplicates()
    stats['distinct_clinical_rows'] = len(f)
    keys = ['pid', 'ad', 'di', 'num']
    counts = Counter()
    parts = []
    for j in range(1, 8):
        prefix = '出院主要诊断' if j == 1 else '出院其他诊断'
        code = f[prefix + '编码' + str(j)].str.upper().str.strip()
        name = f[prefix + '名称' + str(j)].str.strip()
        poa = f[prefix + '入院病情' + str(j)].str.strip()
        p = code.str.extract('^([A-Z]\\d{2})', expand=False).fillna('')
        part = f[keys].copy()
        part['slot'] = j
        part['code'] = code
        part['name'] = name
        part['poa'] = poa
        part['prefix'] = p
        part = part[(part.code != '') | (part.name != '')]
        parts.append(part)
    slots = pd.concat(parts, ignore_index=True).drop_duplicates()
    historical = slots['name'].str.contains('陈旧|既往|后遗|病史|梗死后|梗塞后', regex=True)
    uncertain_name = slots['name'].str.contains('疑似|待排|可能|考虑|未确诊|否认|未见|排除', regex=True)
    mi_text = slots['name'].str.contains('心肌梗(?:死|塞)', regex=True)
    stroke_text = slots['name'].str.contains('脑梗(?:死|塞)|脑(?:内)?出血|蛛网膜下腔出血|卒中|中风', regex=True)
    slots['code_name_supported'] = (slots.prefix.isin(['I21', 'I22']) & mi_text | slots.prefix.isin(['I60', 'I61', 'I63', 'I64']) & stroke_text) & ~historical & ~uncertain_name
    name_check = []
    for prefix, g in slots[slots.prefix.isin(['I21', 'I22', 'I60', 'I61', 'I63', 'I64'])].groupby('prefix'):
        name_check.append({'code_prefix': prefix, 'slot_occurrences': len(g), 'code_name_supported_slots': int(g.code_name_supported.sum()), 'historical_name_slots': int(historical.loc[g.index].sum())})
    pd.DataFrame(name_check).to_csv(AGG / 'sch_code_name_sensitivity.csv', index=False)
    components = {'acute_mi': ['I21', 'I22'], 'stroke': ['I60', 'I61', 'I63', 'I64'], 'cardiac_arrest': ['I46'], 'heart_failure_diagnosis': ['I50'], 'stroke_sequelae': ['I69']}
    slot_rows = []
    for component, prefixes in components.items():
        s = slots[slots.prefix.isin(prefixes)].copy()
        for (role, poa), g in s.assign(role=np.where(s.slot.eq(1), 'principal', 'secondary'), poa=s.poa.replace('', 'not recorded')).groupby(['role', 'poa']):
            slot_rows.append({'component': component, 'role': role, 'poa': poa, 'diagnosis_slot_occurrences': len(g), 'source_stay_keys': len(g[keys].drop_duplicates()), 'patients': g.pid.nunique()})
    flags = f[keys].drop_duplicates().set_index(keys)
    for component, prefixes in components.items():
        s = slots[slots.prefix.isin(prefixes)]
        for suffix, selection in [('any', np.ones(len(s), bool)), ('present', s.poa.eq('有')), ('absent', s.poa.eq('无')), ('uncertain', ~s.poa.isin(['有', '无'])), ('principal', s.slot.eq(1))]:
            index = pd.MultiIndex.from_frame(s.loc[selection, keys].drop_duplicates())
            flags[component + '_' + suffix] = flags.index.isin(index)
    flags = flags.reset_index()
    flags['original_patient_label'] = flags.pid.map(lm)
    flags['acute_any'] = flags.acute_mi_any | flags.stroke_any
    flags['acute_present'] = flags.acute_mi_present | flags.stroke_present
    flags['acute_absent'] = flags.acute_mi_absent | flags.stroke_absent
    flags['acute_uncertain'] = flags.acute_mi_uncertain | flags.stroke_uncertain
    flags['acute_poa_group'] = np.select([~flags.acute_any, flags.acute_present & flags.acute_absent, flags.acute_present, flags.acute_absent & ~flags.acute_uncertain], ['no recorded acute MI or stroke', 'mixed present and absent', 'present at admission', 'absent at admission only'], default='uncertain or unrecorded POA')
    supported = slots[slots.code_name_supported]
    for state, selection in [('any', np.ones(len(supported), bool)), ('present', supported.poa.eq('有')), ('absent', supported.poa.eq('无')), ('uncertain', ~supported.poa.isin(['有', '无']))]:
        index = pd.MultiIndex.from_frame(supported.loc[selection, keys].drop_duplicates())
        flags['supported_' + state] = pd.MultiIndex.from_frame(flags[keys]).isin(index)
    flags['supported_poa_absent_only'] = flags.supported_absent & ~flags.supported_present & ~flags.supported_uncertain
    ordered = flags.sort_values(['pid', 'ad', 'di', 'num'])
    first = ordered.groupby('pid', sort=False).head(1).copy()
    tied = ordered.groupby('pid').apply(lambda x: (x.ad == x.ad.min()).sum(), include_groups=False)
    stats['patients_with_multiple_stay_keys_on_first_admission_day'] = int((tied > 1).sum())
    first = first[~first.pid.isin(tied[tied > 1].index)]
    summary = []
    for scope, d in [('all_matched_source_stays', flags), ('one_first_matched_stay_per_patient', first)]:
        for group, g in d.groupby('acute_poa_group'):
            k = len(g)
            lo, hi = interval(k, len(d)) if scope.startswith('one_') else (None, None)
            summary.append({'scope': scope, 'poa_group': group, 'units': k, 'patients': g.pid.nunique(), 'original_patient_label_positive_units': int(g.original_patient_label.sum()), 'denominator': len(d), 'fraction': k / len(d), 'fraction_low': lo, 'fraction_high': hi})
    for name, data in [('sch_component_poa.csv', slot_rows), ('sch_acute_poa_summary.csv', summary)]:
        pd.DataFrame(data).to_csv(AGG / name, index=False)
    flags.to_csv(PRIVATE / 'sch_matched_source_stays.csv', index=False)
    first.to_csv(PRIVATE / 'sch_first_matched_stays.csv', index=False)
    stats.update({'original_patients': len(lm), 'matched_source_stay_keys': len(flags), 'matched_patients': flags.pid.nunique(), 'first_stay_patients_excluding_ties': len(first), 'all_slot_occurrences_after_clinical_row_deduplication': len(slots), 'acute_absent_only_matched_stays': int(flags.acute_poa_group.eq('absent at admission only').sum()), 'acute_absent_only_first_stays': int(first.acute_poa_group.eq('absent at admission only').sum()), 'heart_failure_principal_stays': int(flags.heart_failure_diagnosis_principal.sum()), 'heart_failure_any_stays': int(flags.heart_failure_diagnosis_any.sum())})
    stats['name_supported_poa_absent_only_matched_stays'] = int(flags.supported_poa_absent_only.sum())
    stats['name_supported_poa_absent_only_first_stays'] = int(first.supported_poa_absent_only.sum())
    (AGG / 'sch_analysis.json').write_text(json.dumps({'counts': dict(stats), 'match_fields': join, 'components': components, 'scope': 'Source-coded composition; original labels unchanged; POA-absent is not a fully adjudicated first-MACE risk set. First-stay rates use Wilson intervals. All-stay counts are not independent-patient rates.'}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'complete': 'SCH composition', 'counts': dict(stats)}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
