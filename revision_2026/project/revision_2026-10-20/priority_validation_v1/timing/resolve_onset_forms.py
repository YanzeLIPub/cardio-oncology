"""Aggregate the semantics of four nonempty generic occurrence-date fields."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import collections
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
OLD = Path(_release_path('project', 'revision_2026-10-12/njm_same_source_completion_2026-10-12'))
RAW = Path(_release_path('njm', 'WUH'))
TARGET_CONTEXT_HASH = None
MACE = re.compile('心肌梗|心梗|脑卒中|脑梗|脑出血|心力衰竭|心衰|心脏骤停|心搏骤停|心源性死亡')
CLASSES = {'infection': re.compile('感染|肺炎|菌血|败血'), 'pressure_injury': re.compile('压疮|褥疮'), 'drug_adverse_reaction': re.compile('药物.*(?:不良|过敏)|不良反应'), 'tumour': re.compile('肿瘤|癌|瘤'), 'transfusion': re.compile('输血')}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def decode(v):
    if isinstance(v, str) and v.lstrip().startswith(('[', '{')):
        try:
            return _release_resolve_config(json.loads(v))
        except (ValueError, RecursionError):
            pass
    return v

def scalar_pairs(obj):
    obj = decode(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                yield from scalar_pairs(value)
            else:
                yield (key, value)
    elif isinstance(obj, list):
        for value in obj:
            yield from scalar_pairs(value)

def parsed(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError:
        return None

def main():
    out = HERE / 'onset_form_semantics.json'
    if out.exists():
        raise FileExistsError('Refusing to overwrite completed audit')
    targets = collections.defaultdict(collections.Counter)
    index = OLD / 'labs/private/source_index.json'
    capture = OLD / 'whu/private/diagnosis_capture_record_audit_v2.jsonl'
    for line in capture.read_text(encoding='utf-8').splitlines():
        row = _release_resolve_config(json.loads(line))
        targets[row['source_content_sha256']][row['record_ordinal']] += 1
    locators = {}
    for row in _release_resolve_config(json.loads(index.read_text(encoding='utf-8'))):
        if row['sha256'] in targets:
            locators.setdefault(row['sha256'], row)
    counts = collections.Counter()
    field_counts = collections.Counter()
    contexts = set()
    matched_content = set()
    for digest, locator in sorted(locators.items()):
        data = (RAW / locator['basename']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        obj = _release_resolve_config(json.loads(data.decode('utf-8-sig')))
        if isinstance(obj, str):
            obj = _release_resolve_config(json.loads(obj))
        records = [obj] if isinstance(obj, dict) else [v for v in obj if isinstance(v, dict)]
        for ordinal, multiplier in targets[digest].items():
            record = records[ordinal]
            stack = [record]
            while stack:
                node = decode(stack.pop())
                if isinstance(node, dict):
                    for key, value in node.items():
                        if hashlib.sha256(str(key).encode()).hexdigest() == TARGET_CONTEXT_HASH:
                            contexts.add(str(key))
                            pairs = list(scalar_pairs(value))
                            dates = [v for k, v in pairs if k == '发生日期' and v not in (None, '')]
                            if dates:
                                matched_content.add(digest)
                                counts['source_section_occurrences_with_nonempty_date'] += multiplier
                                values = [v for _, v in pairs if isinstance(v, str)]
                                counts['section_contains_MACE_word'] += multiplier * any((MACE.search(v) for v in values))
                                for category, pattern in CLASSES.items():
                                    counts['section_named_content_' + category] += multiplier * any((pattern.search(v) for v in values))
                                for k, _ in pairs:
                                    if re.fullmatch('[\\u4e00-\\u9fff]{0,20}(?:名称|类型|部位|情况|诊断|时间|日期|结果|分类|编号|代码)', str(k)):
                                        field_counts[str(k)] += multiplier
                                for date in dates:
                                    counts['nonempty_date_occurrences'] += multiplier
                                    dt = parsed(date)
                                    counts['parseable_date_occurrences'] += multiplier * (dt is not None)
                                    counts['date_only_occurrences'] += multiplier * bool(re.fullmatch('\\d{4}-\\d{2}-\\d{2}', str(date)))
                                    counts['explicit_clock_occurrences'] += multiplier * bool(re.search('\\d{2}:\\d{2}', str(date)))
                                    emr = decode(record.get('电子病历菜单及详情', {}))
                                    basics = decode(emr.get('住院基本信息', {})) if isinstance(emr, dict) else {}
                                    anchors = list(scalar_pairs(basics))
                                    adm = {parsed(v) for k, v in anchors if k == '入院日期' and parsed(v) is not None}
                                    dis = {parsed(v) for k, v in anchors if k == '出院日期' and parsed(v) is not None}
                                    if dt is None or len(adm) != 1 or len(dis) != 1:
                                        counts['unique_record_admission_discharge_comparison_unavailable'] += multiplier
                                    else:
                                        a, d = (next(iter(adm)), next(iter(dis)))
                                        if len({x.utcoffset() is not None for x in (dt, a, d)}) != 1:
                                            counts['mixed_timezone_comparison_unavailable'] += multiplier
                                        elif a <= dt <= d:
                                            counts['date_point_within_recorded_stay'] += multiplier
                                        elif dt < a:
                                            counts['date_point_before_recorded_stay'] += multiplier
                                        else:
                                            counts['date_point_after_recorded_stay'] += multiplier
                        stack.append(value)
                elif isinstance(node, list):
                    stack.extend(node)
    assert counts['nonempty_date_occurrences'] == 4
    report = {'execution_complete': True, 'generic_schema_key': '发生日期', 'source_section_names': sorted(contexts), 'section_administrative_schema_keys': dict(field_counts), 'aggregate_counts': dict(counts), 'unique_source_contents_with_nonempty_date': len(matched_content), 'clinical_MACE_onset_adjudicated': False, 'raw_values_or_dates_or_filenames_exported': False, 'script_sha256': sha(Path(__file__)), 'interpretation': 'The generic occurrence-date field is interpreted within its original named source section, not relabelled MACE onset. A date point within a stay does not establish disease identity or an event-free risk set.'}
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True))
if __name__ == '__main__':
    main()
