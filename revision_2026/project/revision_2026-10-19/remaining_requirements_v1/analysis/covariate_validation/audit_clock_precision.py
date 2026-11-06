"""Separate date-only laboratory precision from genuinely earlier calendar days."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
TIME = '检查时间'
ADMIT = '入院时间'
DISCHARGE = '出院时间'

def ident(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main():
    out, source = ([], [])
    for cohort, name in [('SCH', 'SCH_processed_data.csv'), ('WHU', 'WHU_Cancer_processed_data.csv')]:
        p = Path(_release_path('china', '')) / name
        before = ident(p)
        data = pd.read_csv(p, usecols=[TIME, ADMIT, DISCHARGE], dtype=str)
        lab, admit, discharge = (pd.to_datetime(data[f], format='mixed', errors='raise') for f in (TIME, ADMIT, DISCHARGE))
        row = {'cohort': cohort, 'records': len(data), 'laboratory_before_stored_admission_timestamp': int((lab < admit).sum()), 'laboratory_calendar_day_before_admission_day': int((lab.dt.normalize() < admit.dt.normalize()).sum()), 'earlier_timestamp_but_same_calendar_day': int(((lab < admit) & lab.dt.normalize().eq(admit.dt.normalize())).sum()), 'laboratory_after_discharge_timestamp': int((lab > discharge).sum())}
        for f, label in [(TIME, 'laboratory'), (ADMIT, 'admission'), (DISCHARGE, 'discharge')]:
            row[label + '_source_strings_with_hhmm'] = int(data[f].str.contains('\\d{1,2}:\\d{2}', regex=True, na=False).sum())
        assert before == ident(p)
        source.append(before)
        out.append(row)
    pd.DataFrame(out).to_csv(ROOT / 'results/processed_clock_precision.csv', index=False, mode='x')
    result = {'passed': True, 'script': ident(__file__), 'sources': source, 'sources_unchanged': True, 'cohorts': out, 'limitations': 'String precision and relation to attached admission metadata only. A date-only field is not an exact result-return timestamp. Earlier dates can reflect outpatient/pre-admission testing or patient-level metadata assignment; these aggregate counts do not adjudicate individual error or event timing.'}
    with (ROOT / 'clock_precision_audit.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
