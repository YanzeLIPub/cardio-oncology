"""Synthetic parser and provenance tests. No patient values are fixtures."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sch_audit', _release_source_path(HERE / 'audit_sch_assay_metadata.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def record(code, body=b''):
    return struct.pack('<HH', code, len(body)) + body

def label(row, col, value):
    encoded = value.encode('gb18030')
    return record(4, struct.pack('<HH', row, col) + b'\x00\x00\x00' + bytes([len(encoded)]) + encoded)

class ScanTest(unittest.TestCase):

    def test_exact_assay_not_fee_or_diagnosis(self):
        scan = audit.empty_scan()
        audit.scan_text('白蛋白', scan, 'item')
        audit.scan_text('白蛋白类制品费', scan, 'fee')
        audit.scan_text('低白蛋白血症', scan, 'diagnosis')
        self.assertEqual(scan['exact_assay_label_cells'], {'albumin': 1})
        self.assertEqual(scan['assay_hits_in_nonmeasurement_roles'], {'albumin': 1})
        self.assertEqual(scan['diagnosis_like_assay_string_cells'], {'albumin': 1})

    def test_unit_cooccurrence_does_not_establish_eligibility(self):
        scan = audit.empty_scan()
        audit.scan_text('albumin g/L', scan, 'lab item')
        self.assertEqual(scan['assay_and_unit_same_cell_counts'], {'albumin': 1})
        self.assertNotIn('eligible', scan)

    def test_biff_all_rows_and_safe_output(self):
        payload = record(9) + label(0, 0, 'item') + label(0, 1, 'unit')
        payload += label(17000, 0, '血红蛋白') + label(17000, 1, 'g/L') + record(10)
        result = audit.biff_scan(payload)[0]
        self.assertEqual(result['rows_including_header'], 17001)
        self.assertEqual(result['string_cells_scanned'], 4)
        self.assertEqual(result['exact_assay_label_cells'], {'hemoglobin': 1})
        self.assertNotIn('patient', json.dumps(result))

    def test_biff_rejects_truncation(self):
        with self.assertRaises(ValueError):
            audit.biff_scan(record(9) + b'\x04\x00\x08\x00abc')

    def test_real_report_source_manifest(self):
        data = _release_resolve_config(json.loads((HERE / 'sch_assay_metadata_scan.json').read_text(encoding='utf-8')))
        self.assertEqual(len(data['source_files']), 34)
        self.assertEqual(sum((x.get('matches_prior_sha', False) for x in data['source_files'])), 33)
        self.assertEqual(len(data['binary_payloads']), 16)
        self.assertEqual(data['source_summary']['assay_and_unit_same_cell_counts'], {})
        self.assertEqual(data['script']['sha256'], audit.sha_file(_release_source_path(HERE / 'audit_sch_assay_metadata.py')))
        for b in data['binary_payloads']:
            for sheet in b['sheets']:
                self.assertTrue(sheet.get('all_label_records_decoded') or sheet.get('all_cells_scanned'))

    def test_real_biff_label_shapes_independently(self):
        data = _release_resolve_config(json.loads((HERE / 'sch_assay_metadata_scan.json').read_text(encoding='utf-8')))
        raw_ids = {b['id'] for b in data['binary_payloads'] if b['sheets'][0]['format'] == 'raw BIFF2'}
        checked = set()
        for source in data['source_files']:
            if source.get('scan_id') not in raw_ids:
                continue
            payload = Path(source['path']).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), source['sha256'])
            offset = 0
            while offset < len(payload):
                code, size = struct.unpack_from('<HH', payload, offset)
                if code == 4:
                    self.assertGreaterEqual(size, 8)
                    self.assertEqual(size, 8 + payload[offset + 11])
                offset += 4 + size
            self.assertEqual(offset, len(payload))
            checked.add(source['scan_id'])
        self.assertEqual(checked, raw_ids)
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScanTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'passed': result.wasSuccessful(), 'script_sha256': audit.sha_file(Path(__file__)), 'audit_script_sha256': audit.sha_file(_release_source_path(HERE / 'audit_sch_assay_metadata.py')), 'scan_report_sha256': audit.sha_file(HERE / 'sch_assay_metadata_scan.json')}
    (HERE / 'tests.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
