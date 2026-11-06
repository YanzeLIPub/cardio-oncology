"""Bounded static trace of original loaders and processed-source identity."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import collections
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE_TOKEN = re.compile('SCH_processed_data|processed.*SCH|SCH.*processed', re.I)
SCALE_TOKEN = re.compile('StandardScaler|MinMaxScaler|RobustScaler|inverse_transform|scaler|scale_|mean_|var_|\\u6807\\u51c6\\u5316', re.I)
WRITE_TOKEN = re.compile('to_csv|dump\\(|save\\(')

def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inspect explicitly supplied source-code directories for SCH field and scale definitions.')
    parser.add_argument('--code-dir', type=Path, action='append', required=True)
    ROOTS = parser.parse_args().code_dir
    files = sorted(set((p for root in ROOTS for p in root.glob('*.py'))))
    identities, matched = ([], [])
    for p in files:
        data = p.read_bytes()
        text = data.decode('utf-8-sig', errors='replace')
        ident = {'path': p.as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        identities.append(ident)
        if SOURCE_TOKEN.search(text):
            lines = text.splitlines()
            matched.append({**ident, 'source_lines': [{'line': i, 'text': x.strip()[:250]} for i, x in enumerate(lines, 1) if SOURCE_TOKEN.search(x)], 'scale_lines': [{'line': i, 'text': x.strip()[:250]} for i, x in enumerate(lines, 1) if SCALE_TOKEN.search(x)], 'write_line_numbers': [i for i, x in enumerate(lines, 1) if WRITE_TOKEN.search(x)]})
    result = {'scope': 'Only direct *.py files in explicitly listed original-code directories; no model deserialization or module execution.', 'roots': [p.as_posix() for p in ROOTS], 'script_count': len(identities), 'script_identities': identities, 'source_linked_scripts': matched, 'no_claim_that_source_units_or_inverse_parameters_recovered': True, 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}, 'script': {'path': Path(__file__).as_posix(), 'sha256': hash_file(Path(__file__))}, 'no_originals_written': True, 'models_loaded_or_trained': 0, 'patient_rows_exported': False}
    out = HERE / 'scale_lineage_scan.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'scripts': len(identities), 'linked_scripts': len(matched)}))
if __name__ == '__main__':
    main()
