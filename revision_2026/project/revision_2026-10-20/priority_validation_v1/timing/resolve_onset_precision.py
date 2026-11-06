"""Audit date precision in the identified infection forms, not MACE onset."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import collections
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
HERE = Path(__file__).resolve().parent
OLD = Path(_release_path('project', 'revision_2026-10-12/njm_same_source_completion_2026-10-12'))
RAW = Path(_release_path('njm', 'WUH'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def decode(v):
    if isinstance(v, str) and v.lstrip().startswith(('[', '{')):
        try:
            return _release_resolve_config(json.loads(v))
        except (ValueError, RecursionError):
            pass
    return v

def pairs(obj):
    obj = decode(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            d = decode(v)
            if isinstance(d, (dict, list)):
                yield from pairs(d)
            else:
                yield (k, v)
    elif isinstance(obj, list):
        for v in obj:
            yield from pairs(v)

def named_values(obj, name):
    obj = decode(obj)
    if isinstance(obj, dict):
        if name in obj:
            yield obj[name]
    elif isinstance(obj, list):
        for item in obj:
            yield from named_values(item, name)

def clock(value):
    if not isinstance(value, str):
        return (None, 'not_string')
    text = value.strip()
    date = re.fullmatch('(\\d{4})[-/年](\\d{1,2})[-/月](\\d{1,2})日?', text)
    if date:
        try:
            return (datetime(*(int(x) for x in date.groups())), 'calendar_day')
        except ValueError:
            return (None, 'invalid_calendar_date')
    try:
        return (datetime.fromisoformat(text.replace('Z', '+00:00')), 'explicit_clock')
    except ValueError:
        return (None, 'unsupported_date_string')

def main():
    out = HERE / 'onset_form_precision.json'
    if out.exists():
        raise FileExistsError('Refusing to overwrite completed precision audit')
    index = OLD / 'labs/private/source_index.json'
    capture = OLD / 'whu/private/diagnosis_capture_record_audit_v2.jsonl'
    targets = collections.defaultdict(collections.Counter)
    for line in capture.read_text(encoding='utf-8').splitlines():
        row = _release_resolve_config(json.loads(line))
        targets[row['source_content_sha256']][row['record_ordinal']] += 1
    locators = {}
    for row in _release_resolve_config(json.loads(index.read_text(encoding='utf-8'))):
        if row['sha256'] in targets:
            locators.setdefault(row['sha256'], row)
    counts = collections.Counter()
    for digest, locator in sorted(locators.items()):
        data = (RAW / locator['basename']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        obj = decode(_release_resolve_config(json.loads(data.decode('utf-8-sig'))))
        records = [obj] if isinstance(obj, dict) else [v for v in obj if isinstance(v, dict)]
        for ordinal, multiplier in targets[digest].items():
            emr = decode(records[ordinal].get('电子病历菜单及详情', {}))
            sections = list(named_values(emr, '院内感染病例登记表'))
            items = list(pairs(sections))
            values = [v for k, v in items if k == '发生日期' and v not in (None, '')]
            for value in values:
                counts['nonempty_occurrences'] += multiplier
                date, kind = clock(value)
                counts['precision_' + kind] += multiplier
                basic = list(pairs(list(named_values(emr, '住院基本信息'))))
                a = {clock(v)[0] for k, v in basic if k == '入院日期' and clock(v)[0] is not None}
                d = {clock(v)[0] for k, v in basic if k == '出院日期' and clock(v)[0] is not None}
                if len(a) != 1 or len(d) != 1 or date is None:
                    counts['stay_comparison_unknown_anchor'] += multiplier
                    continue
                admission, discharge = (next(iter(a)), next(iter(d)))
                if any((x.utcoffset() is not None for x in (date, admission, discharge))):
                    counts['stay_comparison_timezone_not_reconciled'] += multiplier
                    continue
                end = date + timedelta(days=1) if kind == 'calendar_day' else date
                if date >= admission and end <= discharge:
                    counts['whole_date_interval_within_recorded_stay'] += multiplier
                elif end < admission or date > discharge:
                    counts['date_interval_outside_recorded_stay'] += multiplier
                else:
                    counts['date_interval_overlaps_recorded_stay_boundary'] += multiplier
    assert counts['nonempty_occurrences'] == 4
    report = {'execution_complete': True, 'source_section': '院内感染病例登记表', 'field': '发生日期', 'aggregate_counts': dict(counts), 'source_schema_semantics_report': {'path': (HERE / 'onset_form_semantics.json').as_posix(), 'sha256': sha(HERE / 'onset_form_semantics.json')}, 'interpretation': 'The fields concern hospital-acquired infection occurrence. Calendar-day precision and record linkage cannot convert an infection date into clinical MACE onset.', 'MACE_event_dates_created': 0, 'clinical_values_dates_identifiers_exported': False, 'script_sha256': sha(Path(__file__))}
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True))
if __name__ == '__main__':
    main()
