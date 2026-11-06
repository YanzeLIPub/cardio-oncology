"""Resolve generic onset-shaped schema keys without exporting clinical values."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import collections
import hashlib
import json
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent
OLD = Path(_release_path('project', 'revision_2026-10-12/njm_same_source_completion_2026-10-12'))
RAW = Path(_release_path('njm', 'WUH'))
PATTERN = re.compile('发病(?:时间|日期|时刻)|发生(?:时间|日期|时刻)|起病(?:时间|日期|时刻)|onset|event.?time', re.I)
MACE = re.compile('心肌梗|心梗|脑卒中|脑梗|脑出血|心力衰竭|心衰|心脏骤停|心搏骤停|心源性死亡')
SAFE_CONTEXT = re.compile('^(?:不良事件|不良反应|并发症|输血|手术|护理|诊断|病程|随访|检验|检查|病案首页|手术操作信息|电子病历菜单及详情|住院基本信息|生命体征)[\\u4e00-\\u9fff]*$')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out = HERE / 'onset_candidate_context.json'
    if out.exists():
        raise FileExistsError('Refusing to overwrite completed context audit')
    targets = collections.defaultdict(collections.Counter)
    capture = OLD / 'whu/private/diagnosis_capture_record_audit_v2.jsonl'
    index = OLD / 'labs/private/source_index.json'
    for line in capture.read_text(encoding='utf-8').splitlines():
        row = _release_resolve_config(json.loads(line))
        targets[row['source_content_sha256']][row['record_ordinal']] += 1
    locators = {}
    for row in _release_resolve_config(json.loads(index.read_text(encoding='utf-8'))):
        if row['sha256'] in targets:
            locators.setdefault(row['sha256'], row)
    result, fields = (collections.Counter(), collections.Counter())
    signature = hashlib.sha256()
    for digest, locator in sorted(locators.items()):
        data = (RAW / locator['basename']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        signature.update(bytes.fromhex(digest))
        obj = _release_resolve_config(json.loads(data.decode('utf-8-sig')))
        if isinstance(obj, str):
            obj = _release_resolve_config(json.loads(obj))
        records = [obj] if isinstance(obj, dict) else [v for v in obj if isinstance(v, dict)]
        for ordinal, multiplier in targets[digest].items():
            stack = [(records[ordinal], ())]
            while stack:
                node, ancestors = stack.pop()
                if isinstance(node, dict):
                    for key, value in node.items():
                        if PATTERN.search(str(key)):
                            if value in (None, '', [], {}):
                                value_class = 'empty'
                            elif isinstance(value, str) and re.fullmatch('\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2}.*', value.strip()):
                                value_class = 'date_shaped'
                            else:
                                value_class = 'nonempty_not_date_shaped'
                            fields[str(key), value_class] += multiplier
                            context = tuple((x if SAFE_CONTEXT.fullmatch(x) else hashlib.sha256(x.encode()).hexdigest() for x in ancestors[-4:]))
                            siblings = tuple(sorted((hashlib.sha256(str(k).encode()).hexdigest() for k in node)))
                            direct_mace = any((isinstance(v, str) and MACE.search(v) for v in node.values()))
                            result[str(key), value_class, context, siblings, bool(direct_mace)] += multiplier
                        stack.append((value, ancestors + (str(key),)))
                elif isinstance(node, list):
                    stack.extend(((v, ancestors) for v in node))
                elif isinstance(node, str) and node.lstrip().startswith(('[', '{')):
                    try:
                        decoded = _release_resolve_config(json.loads(node))
                    except (ValueError, RecursionError):
                        continue
                    if isinstance(decoded, (dict, list)):
                        stack.append((decoded, ancestors))
    report = {'execution_complete': True, 'source_contents': len(locators), 'source_content_hash_sequence': signature.hexdigest(), 'field_classes': [{'schema_key': k, 'value_class': v, 'occurrences': n} for (k, v), n in sorted(fields.items())], 'contexts': [{'schema_key': k, 'value_class': v, 'ancestor_schema_context': list(a), 'direct_sibling_schema_hashes': list(s), 'direct_scalar_mace_word_cooccurrence': m, 'occurrences': n} for (k, v, a, s, m), n in sorted(result.items())], 'no_event_times_or_clinical_values_exported': True, 'interpretation': 'A schema-key occurrence or MACE word cooccurrence is not clinical onset adjudication. Source field semantics and current event/encounter binding remain required.', 'script_sha256': sha(Path(__file__))}
    assert sum(fields.values()) == 127
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'complete': True, 'fields': report['field_classes'], 'sha256': sha(out)}, ensure_ascii=True))
if __name__ == '__main__':
    main()
