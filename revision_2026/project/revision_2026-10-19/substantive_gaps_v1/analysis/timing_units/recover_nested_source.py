"""Read-only nested-container recovery; export aggregates, never clinical rows."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import collections
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(_release_path('project', ''))
OLD = ROOT / 'revision_2026-10-12/njm_same_source_completion_2026-10-12'
RAW = Path(_release_path('njm', 'WUH'))
MAX_LITERAL_BYTES = 2000000
MAX_AST_NODES = 250000
PATTERNS = {'event_onset': re.compile('发病(?:时间|日期|时刻)|发生(?:时间|日期|时刻)|起病(?:时间|日期|时刻)|onset|event.?time', re.I), 'lab_first_availability': re.compile('发布(?:时间|日期)|释放(?:时间|日期)|结果(?:可见|可用|返回)(?:时间|日期)|首次.*(?:可见|可用)|release.?time|available.?time', re.I), 'report_or_review_time': re.compile('(?:报告|审核|检验|检查|采样|收样|录入|诊断)(?:时间|日期|时刻)|report.?time|storetime', re.I), 'diagnosis_completeness': re.compile('诊断.*(?:总数|全部|总条|完整)|(?:全部|完整).*诊断|diagnos.*complete', re.I), 'poa': re.compile('入院病情|present.?on.?admission|^poa$', re.I), 'death': re.compile('死亡(?:时间|日期|原因)|死因|underlying.?cause', re.I), 'unit': re.compile('^(?:检验|化验|检查|结果|项目)?单位$|^units?$|valueuom', re.I), 'geography': re.compile('^(?:现住址|户口地址|户籍地址|家庭地址|联系地址|通讯地址|居住地址|出生地|籍贯|地址|地区|年份|年度|省|市|县|区|省份|城市|省市|邮编|邮政编码|现住址省|现住址市|现住址县|户口省|户口市|户口县)$')}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def literal_container(text):
    if len(text.encode('utf-8')) > MAX_LITERAL_BYTES:
        return (None, 'size_limit')
    try:
        tree = ast.parse(text, mode='eval')
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return (None, 'not_python_expression')
    if not isinstance(tree.body, (ast.List, ast.Dict, ast.Tuple)):
        return (None, 'not_literal_container_root')
    allowed = (ast.Expression, ast.List, ast.Dict, ast.Tuple, ast.Constant, ast.UnaryOp, ast.UAdd, ast.USub, ast.Load)
    for n, node in enumerate(ast.walk(tree), 1):
        if n > MAX_AST_NODES:
            return (None, 'node_limit')
        if not isinstance(node, allowed):
            return (None, 'nonliteral_ast_node')
    try:
        obj = ast.literal_eval(tree)
    except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
        return (None, 'literal_eval_rejected')
    return (obj, 'python_literal_container')

def nonempty(v):
    return v is not None and v != '' and (v != []) and (v != {})

def context_for(key, prior):
    if re.search('检验|化验|laboratory|^lab', key, re.I):
        return 'laboratory_named_context'
    if re.search('检查|exam|radiolog', key, re.I):
        return 'examination_named_context'
    if re.search('诊断|病案首页|diagnos', key, re.I):
        return 'diagnosis_named_context'
    if re.search('死亡|死因|death', key, re.I):
        return 'death_named_context'
    return prior

def main():
    output = HERE / 'nested_literal_recovery.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite completed recovery')
    index_path = OLD / 'labs/private/source_index.json'
    capture_path = OLD / 'whu/private/diagnosis_capture_record_audit_v2.jsonl'
    binding = load(OLD / 'labs/whu_lab_binding.json')
    assert sha(index_path) == next((x['sha256'] for x in binding['outputs'] if x['path'] == 'private/source_index.json'))
    contract = load(OLD / 'whu/diagnosis_capture_contract_v2.json')
    assert sha(capture_path) == contract['private_output']['sha256']
    targets = collections.defaultdict(collections.Counter)
    patients, stays = (set(), set())
    with capture_path.open(encoding='utf-8') as f:
        for line in f:
            row = _release_resolve_config(json.loads(line))
            targets[row['source_content_sha256']][row['record_ordinal']] += 1
            patients.add(row['patient_token'])
            stays.add(row['stay_candidate_token'])
    locators = {}
    for x in sorted(load(index_path), key=lambda x: x['ordinal']):
        if x['sha256'] in targets:
            locators.setdefault(x['sha256'], x)
    assert set(locators) == set(targets)
    totals, failures = (collections.Counter(), collections.Counter())
    fields = collections.Counter()
    geography_fields = collections.Counter()
    raw_hash_sequence = hashlib.sha256()
    literal_signature_counts = collections.Counter()
    for count, (content_hash, locator) in enumerate(sorted(locators.items()), 1):
        path = RAW / locator['basename']
        assert path.resolve().parent == RAW.resolve()
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == content_hash and len(data) == locator['bytes']
        raw_hash_sequence.update(bytes.fromhex(content_hash))
        obj = _release_resolve_config(json.loads(data.decode('utf-8-sig')))
        if isinstance(obj, str):
            obj = _release_resolve_config(json.loads(obj))
        records = [obj] if isinstance(obj, dict) else [x for x in obj if isinstance(x, dict)]
        totals['raw_unique_contents'] += 1
        totals['raw_bytes'] += len(data)
        for ordinal, multiplicity in targets[content_hash].items():
            totals['matched_record_occurrences'] += multiplicity
            stack = [(records[ordinal], 'native_container', 'unspecified_context', 0)]
            while stack:
                node, origin, role, depth = stack.pop()
                if depth >= 100:
                    failures['nesting_limit'] += multiplicity
                    continue
                if isinstance(node, dict):
                    for key, value in node.items():
                        if not isinstance(key, str):
                            totals['nonstring_dictionary_keys'] += multiplicity
                            continue
                        totals[origin + '_key_occurrences'] += multiplicity
                        nextrole = context_for(key, role)
                        for family, pattern in PATTERNS.items():
                            if pattern.search(key):
                                fields[origin, family, nextrole, hashlib.sha256(key.encode()).hexdigest(), nonempty(value)] += multiplicity
                                if family == 'geography':
                                    geography_fields[origin, nextrole, key, nonempty(value)] += multiplicity
                        stack.append((value, origin, nextrole, depth + 1))
                elif isinstance(node, (list, tuple)):
                    stack.extend(((value, origin, role, depth + 1) for value in node))
                elif isinstance(node, str) and node.lstrip().startswith(('[', '{', '(')):
                    stripped = node.strip()
                    try:
                        decoded = _release_resolve_config(json.loads(stripped))
                    except (ValueError, RecursionError):
                        totals['strict_json_failed_occurrences'] += multiplicity
                        decoded, result = literal_container(stripped)
                        if decoded is None:
                            failures[result] += multiplicity
                            continue
                        totals['python_literal_decoded_occurrences'] += multiplicity
                        keys = sorted((str(k) for k in decoded.keys())) if isinstance(decoded, dict) else []
                        signature = hashlib.sha256(json.dumps(keys, ensure_ascii=True).encode()).hexdigest()
                        literal_signature_counts[type(decoded).__name__, role, signature] += multiplicity
                        stack.append((decoded, 'recovered_python_literal', role, depth + 1))
                    else:
                        if isinstance(decoded, (dict, list)):
                            totals['strict_json_decoded_occurrences'] += multiplicity
                            stack.append((decoded, origin, role, depth + 1))
        if count % 500 == 0:
            print(f'Nested raw-source recovery checked {count}/{len(locators)} contents', flush=True)
    assert totals['matched_record_occurrences'] == 5744 and len(patients) == 2246 and (len(stays) == 5415)
    report = {'passed': True, 'scope': 'Original frozen WHU matched records only; no new clinical labels or inference from prose', 'inputs': [identity(index_path), identity(capture_path)], 'raw_source_root': RAW.as_posix(), 'raw_source_hash_sequence': raw_hash_sequence.hexdigest(), 'raw_contents_all_hash_verified': True, 'counts': dict(totals), 'matched_original_patients': len(patients), 'matched_stay_day_keys': len(stays), 'decoding_failures': dict(failures), 'matched_schema_fields': [dict(origin=o, family=f, context=c, field_sha256=k, nonempty=ne, occurrences=n) for (o, f, c, k, ne), n in sorted(fields.items())], 'literal_container_signatures': [dict(root_type=t, context=c, keys_sha256=h, occurrences=n) for (t, c, h), n in sorted(literal_signature_counts.items())], 'safe_geography_field_names': [dict(origin=o, context=c, field=k, nonempty=ne, occurrences=n) for (o, c, k, ne), n in sorted(geography_fields.items())], 'patterns': {k: v.pattern for k, v in PATTERNS.items()}, 'parser': {'method': 'strict JSON, then allowlisted AST literal container only; no repairs or eval', 'max_literal_bytes': MAX_LITERAL_BYTES, 'max_ast_nodes': MAX_AST_NODES, 'max_nested_depth': 100}, 'clinical_interpretation': 'Key discovery alone does not adjudicate onset, first availability or complete capture.', 'originals_written': False, 'patient_rows_exported': False, 'script': identity(__file__), 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}}
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
        f.write('\n')
    print(json.dumps({'passed': True, 'counts': dict(totals), 'failures': dict(failures), 'new_literal_schema_fields': sum((n for (o, *rest), n in fields.items() if o == 'recovered_python_literal'))}))
if __name__ == '__main__':
    main()
