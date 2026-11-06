"""Read-only SCH assay metadata investigation; no clinical rows are exported."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import collections
import csv
import hashlib
import io
import json
import os
import platform
import re
import struct
import sys
import zipfile
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(_release_path('njm', ''))
SCH = ROOT / 'SCH'
PRIOR = Path(_release_path('project', 'revision_2026-10-19/substantive_gaps_v1/analysis/timing_units/sch_fresh_units_and_timing_gate.json'))
XL = Path(_release_path('project', 'revision_2026-10-12/analysis/a2_sch_archive_unit_dependencies_v1'))
sys.path.insert(0, str(XL))
import xlrd
ASSAYS = {'albumin': re.compile('\\u767d\\u86cb\\u767d|(?<![a-z])albumin(?![a-z])', re.I), 'total_protein': re.compile('\\u603b\\u86cb\\u767d|total\\s*protein', re.I), 'hemoglobin': re.compile('\\u8840\\u7ea2\\u86cb\\u767d|ha?emoglobin', re.I), 'hematocrit': re.compile('\\u7ea2\\u7ec6\\u80de\\u538b\\u79ef|\\u8840\\u7ec6\\u80de\\u6bd4\\u5bb9|ha?ematocrit', re.I)}
UNIT = re.compile('(?<![A-Za-z])(?:g/[dD]?[lL]|mg/[dD]?[lL]|mmol/[lL]|umol/[lL]|%)(?![A-Za-z])')
META = re.compile('\\u5355\\u4f4d|\\u53c2\\u8003\\u503c|\\u53c2\\u8003\\u8303\\u56f4|\\u6807\\u672c|\\u68c0\\u9a8c\\u9879\\u76ee|\\u7ed3\\u679c\\u503c|valueuom|unit|specimen', re.I)
SCALE = re.compile('StandardScaler|MinMaxScaler|RobustScaler|inverse_transform|standardiz|normaliz|\\u6807\\u51c6\\u5316|\\u5f52\\u4e00\\u5316|z.?score|mean_|scale_|var_', re.I)
LINK = re.compile('SCH_processed_data|SCH.*processed|processed.*SCH', re.I)
PHYSICAL = re.compile('\\u5355\\u4f4d|\\u6807\\u672c|\\u5b57\\u5178|dictionary|unit|specimen|\\u767d\\u86cb\\u767d|\\u8840\\u7ea2\\u86cb\\u767d', re.I)

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def identity(path, hashed=True):
    s = path.stat()
    d = {'path': path.as_posix(), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}
    if hashed:
        d['sha256'] = sha_file(path)
    return d

def empty_scan():
    return {'string_cells_scanned': 0, 'assay_substring_cell_counts': collections.Counter(), 'unit_token_cell_counts': collections.Counter(), 'metadata_keyword_cells': 0, 'assay_and_unit_same_cell_counts': collections.Counter(), 'exact_assay_label_cells': collections.Counter(), 'assay_hits_in_nonmeasurement_roles': collections.Counter(), 'assay_hit_column_roles': collections.Counter(), 'diagnosis_like_assay_string_cells': collections.Counter()}

def scan_text(value, out, column_role=''):
    if not isinstance(value, str) or not value:
        return
    out['string_cells_scanned'] += 1
    found = [name for name, pat in ASSAYS.items() if pat.search(value)]
    units = set(UNIT.findall(value))
    for name in found:
        out['assay_substring_cell_counts'][name] += 1
        if units:
            out['assay_and_unit_same_cell_counts'][name] += 1
        if ASSAYS[name].fullmatch(value.strip().lstrip('★*')):
            out['exact_assay_label_cells'][name] += 1
        if column_role:
            out['assay_hit_column_roles'][column_role] += 1
        if re.search('\\u8d39|\\u5236\\u54c1|\\u6ce8\\u5c04|\\u624b\\u672f|\\u64cd\\u4f5c', value + column_role):
            out['assay_hits_in_nonmeasurement_roles'][name] += 1
        if re.search('\\u75c7|\\u75c5|\\u8d2b\\u8840|\\u7f3a\\u4e4f', value):
            out['diagnosis_like_assay_string_cells'][name] += 1
    for token in units:
        out['unit_token_cell_counts'][token] += 1
    if META.search(value):
        out['metadata_keyword_cells'] += 1

def biff_scan(payload):
    out, headers, recs = (empty_scan(), {}, collections.Counter())
    p, maxrow, maxcol, bof, eof = (0, -1, -1, 0, 0)
    while p + 4 <= len(payload):
        code, size = struct.unpack_from('<HH', payload, p)
        body = payload[p + 4:p + 4 + size]
        if len(body) != size:
            raise ValueError('Truncated BIFF')
        recs[f'{code:04x}'] += 1
        bof += code == 9
        eof += code == 10
        if code in {1, 2, 3, 4, 5, 6} and size >= 4:
            row, col = struct.unpack_from('<HH', body)
            maxrow, maxcol = (max(maxrow, row), max(maxcol, col))
            if code == 4 and size >= 8 and (size == 8 + body[7]):
                value = body[8:].decode('gb18030', errors='strict')
                if row == 0:
                    headers[col] = value
                scan_text(value, out, headers.get(col, ''))
        p += size + 4
    if p != len(payload) or bof != 1 or eof != 1:
        raise ValueError('Unresolved BIFF boundaries')
    hs = [headers.get(i, '') for i in range(maxcol + 1)]
    out.update({'format': 'raw BIFF2', 'sheet_ordinal': 1, 'rows_including_header': maxrow + 1, 'columns': maxcol + 1, 'header_sha256': sha_bytes(json.dumps(hs, ensure_ascii=False).encode()), 'lab_related_headers': [x for x in hs if any((p.search(x) for p in ASSAYS.values())) or META.search(x)], 'record_types': dict(recs), 'all_label_records_decoded': True})
    return [out]

def binary_scan(payload):
    if payload.startswith(b'\t\x00'):
        return biff_scan(payload)
    if not payload.startswith(b'\xd0\xcf\x11\xe0'):
        raise ValueError('Unrecognized binary file')
    book = xlrd.open_workbook(file_contents=payload, encoding_override='gb18030', formatting_info=True, logfile=io.StringIO(), on_demand=True)
    output = []
    for si in range(book.nsheets):
        sheet = book.sheet_by_index(si)
        out = empty_scan()
        hs = [str(x) for x in sheet.row_values(0)] if sheet.nrows else []
        for r in range(sheet.nrows):
            for ci, v in enumerate(sheet.row_values(r)):
                scan_text(v, out, hs[ci])
        formats = [f.format_str for f in book.format_map.values() if isinstance(f.format_str, str)]
        out.update({'format': 'OLE XLS', 'sheet_ordinal': si + 1, 'visibility': sheet.visibility, 'rows_including_header': sheet.nrows, 'columns': sheet.ncols, 'header_sha256': sha_bytes(json.dumps(hs, ensure_ascii=False).encode()), 'lab_related_headers': [x for x in hs if any((p.search(x) for p in ASSAYS.values())) or META.search(x)], 'unit_tokens_in_number_formats': sorted({u for f in formats for u in UNIT.findall(f)}), 'assays_in_number_formats': [a for a, p in ASSAYS.items() if any((p.search(f) for f in formats))], 'all_cells_scanned': True})
        output.append(out)
        book.unload_sheet(si)
    book.release_resources()
    return output

def text_header(path):
    for enc in ('utf-8-sig', 'gb18030'):
        try:
            with path.open(encoding=enc, newline='') as f:
                line = f.readline()
            sep = '\t' if line.count('\t') > line.count(',') else ','
            hs = next(csv.reader([line], delimiter=sep))
            return {'encoding': enc, 'delimiter': sep, 'columns': len(hs), 'lab_related_headers': [h for h in hs if any((p.search(h) for p in ASSAYS.values())) or META.search(h)], 'header_sha256': sha_bytes(json.dumps(hs, ensure_ascii=False).encode())}
        except UnicodeDecodeError:
            continue
    raise ValueError('No supported text encoding')

def source_scan():
    files = sorted((p for p in SCH.rglob('*') if p.is_file()))
    prior = _release_resolve_config(json.loads(PRIOR.read_text(encoding='utf-8')))
    records, binaries, seen = ([], [], {})
    prior_by_size = collections.defaultdict(list)
    for x in prior['raw_source_identities']:
        prior_by_size[x['bytes']].append(x)
    for i, p in enumerate(files, 1):
        if p.name == 'SCH_processed_data.csv':
            prior_matches = prior_by_size[p.stat().st_size]
            rec = identity(p, False)
            rec['sha256_from_prior_fresh_audit_not_rehashed_here'] = prior_matches[0]['sha256'] if len(prior_matches) == 1 else None
            rec['text_schema'] = text_header(p)
            records.append(rec)
            continue
        payload = p.read_bytes()
        rec = identity(p, False)
        rec['sha256'] = sha_bytes(payload)
        rec['matches_prior_sha'] = any((x['sha256'] == rec['sha256'] for x in prior_by_size[len(payload)]))
        if p.suffix.lower() == '.zip':
            rec['archive_members'] = []
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                for mi, member in enumerate(z.infolist(), 1):
                    if member.is_dir():
                        continue
                    data = z.read(member)
                    key = sha_bytes(data)
                    meta = {'ordinal': mi, 'bytes': len(data), 'sha256': key, 'extension': Path(member.filename).suffix.lower(), 'crc32': member.CRC, 'member_filename_exported': False}
                    if key in seen:
                        meta['duplicate_of'] = seen[key]
                    else:
                        bid = f'file-{i}/member-{mi}'
                        sheets = binary_scan(data)
                        binaries.append({'id': bid, 'sha256': key, 'sheets': sheets})
                        seen[key] = bid
                        meta['scan_id'] = bid
                    rec['archive_members'].append(meta)
        elif p.suffix.lower() == '.xls' and payload.startswith((b'\t\x00', b'\xd0\xcf\x11\xe0')):
            if rec['sha256'] in seen:
                rec['duplicate_of'] = seen[rec['sha256']]
            else:
                bid = f'file-{i}'
                binaries.append({'id': bid, 'sha256': rec['sha256'], 'sheets': binary_scan(payload)})
                seen[rec['sha256']] = bid
                rec['scan_id'] = bid
        else:
            rec['text_schema'] = text_header(p)
        records.append(rec)
        print(json.dumps({'progress': 'source metadata scanned', 'ordinal': i, 'source_files': len(files)}), flush=True)
    return (records, binaries)

def code_scan():
    allowed_roots = [ROOT / 'scripts', ROOT / 'multicenter_manuscript_output/scripts', ROOT / '_review_tmp/code_extract', ROOT / 'gpu_remote']
    candidates = set(ROOT.glob('*.py'))
    for folder in allowed_roots:
        if folder.exists():
            candidates.update(folder.rglob('*.py'))
    matches, allids = ([], [])
    for p in sorted(candidates):
        if any((x in {'__pycache__', '.venv', 'site-packages'} or x.startswith('synthetic') for x in p.parts)):
            continue
        b = p.read_bytes()
        allids.append({'path': p.as_posix(), 'sha256': sha_bytes(b), 'bytes': len(b)})
        txt = b.decode('utf-8-sig', errors='replace')
        if LINK.search(txt):
            lines = txt.splitlines()
            matches.append({'path': p.as_posix(), 'sha256': sha_bytes(b), 'source_reference_lines': [i for i, x in enumerate(lines, 1) if LINK.search(x)], 'scale_term_lines': [{'line': i, 'text': x.strip()[:260]} for i, x in enumerate(lines, 1) if SCALE.search(x)], 'unit_or_dictionary_lines': [{'line': i, 'text': x.strip()[:260]} for i, x in enumerate(lines, 1) if PHYSICAL.search(x)][:40]})
    return {'scope': [p.as_posix() for p in allowed_roots] + [_release_path('njm', '*.py')], 'scanned_script_count': len(allids), 'scanned_script_identities': allids, 'processed_source_linked_scripts': matches, 'limits': 'No arbitrary clinical JSON bodies, generated model trees, repository objects, or unrelated hospital datasets scanned.'}

def main():
    sources, binaries = source_scan()
    codes = code_scan()
    sheets = [s for b in binaries for s in b['sheets']]
    result = {'schema_version': 1, 'task': 'Bounded SCH assay-unit and scale metadata audit', 'source_root': SCH.as_posix(), 'source_files': sources, 'binary_payloads': binaries, 'source_summary': {'files': len(sources), 'unique_binary_payloads': len(binaries), 'binary_sheets_scanned': len(sheets), 'string_cells_scanned': sum((s['string_cells_scanned'] for s in sheets)), 'assay_and_unit_same_cell_counts': dict(sum((collections.Counter(s['assay_and_unit_same_cell_counts']) for s in sheets), collections.Counter())), 'exact_assay_label_cells': dict(sum((collections.Counter(s['exact_assay_label_cells']) for s in sheets), collections.Counter())), 'assay_substring_cells': dict(sum((collections.Counter(s['assay_substring_cell_counts']) for s in sheets), collections.Counter()))}, 'code_scan': codes, 'prior_audit': identity(PRIOR), 'script': identity(Path(__file__)), 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'xlrd': xlrd.__version__, 'xlrd_path': Path(xlrd.__file__).as_posix(), 'xlrd_init_sha256': sha_file(Path(xlrd.__file__))}, 'no_originals_written': True, 'patient_rows_exported': False, 'models_trained': 0, 'interpretation_guard': 'A target word in diagnostic prose is not a laboratory assay identity. Co-occurring unrelated unit words do not establish a source-column scale. No unit or scale is inferred from numeric distributions.'}
    out = HERE / 'sch_assay_metadata_scan.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'completed': True, 'summary': result['source_summary'], 'output': out.as_posix(), 'sha256': sha_file(out)}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
