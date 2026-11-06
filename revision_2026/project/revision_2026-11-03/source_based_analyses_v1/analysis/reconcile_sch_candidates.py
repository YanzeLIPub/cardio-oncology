"""Prepare local original-record locators for observed label-definition discrepancies."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import Counter
import json, re, sys
import pandas as pd
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'revision_2026-11-03/source_gap_search_v1'))
from scan_sources import key
f = pd.read_csv(OUT / 'private/sch_matched_source_stays.csv', dtype={'pid': str, 'num': str})
target = f[f.supported_poa_absent_only & f.original_patient_label.eq(0)]
lookup = {(r.pid, r.ad, r.di, key(r.num)): 'SCH-R' + str(i + 1).zfill(3) for i, r in enumerate(target.itertuples())}
out = []
for path in sorted(Path(_release_path('njm', 'SCH/2021-2025_原始导出')).glob('*.xls')):
    d = pd.read_csv(path, sep='\t', encoding='gb18030', dtype='string', keep_default_na=False)
    for row_number, r in enumerate(d.to_dict('records'), 2):
        k = (key(r['病案号']), str(r['入院时间'])[:10], str(r['出院时间'])[:10], key(r['住院次数']))
        if k not in lookup:
            continue
        for j in range(1, 8):
            p = '出院主要诊断' if j == 1 else '出院其他诊断'
            code = r[p + '编码' + str(j)]
            name = r[p + '名称' + str(j)]
            poa = r[p + '入院病情' + str(j)]
            if re.match('I(?:21|22|60|61|63|64)', code.strip(), re.I):
                out.append({'review_id': lookup[k], 'original_patient_id': k[0], 'admission_day': k[1], 'discharge_day': k[2], 'admission_number': k[3], 'original_label': 0, 'source_path': path.as_posix(), 'source_row': row_number, 'diagnosis_slot': j, 'diagnosis_code': code, 'diagnosis_name': name, 'present_on_admission': poa, 'clinical_decision': '', 'reason': ''})
pd.DataFrame(out).drop_duplicates().to_csv(OUT / 'private/SCH_label_discrepancy_review.csv', index=False, encoding='utf-8-sig')
agg = {'candidate_patients': len(target), 'candidate_source_stays': len(lookup), 'raw_locator_rows': len(out), 'scope': 'Observed disagreement between original label and a new code/name/POA rule; no relabeling or assertion of clinical error. Locators remain private.'}
(OUT / 'aggregate/sch_label_reconciliation.json').write_text(json.dumps(agg, indent=2), encoding='utf-8')
print(json.dumps(agg))
