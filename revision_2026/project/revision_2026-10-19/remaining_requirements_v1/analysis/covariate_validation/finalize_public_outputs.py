"""Seal new public covariate/era outputs after independent verification."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from pathlib import Path
import fitz
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent

def read(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    cancer = ROOT / 'cancer_associations'
    verifications = [ROOT / 'independent_external_verification.json', cancer / 'independent_external_verification.json', cancer / 'dictionary_verification.json']
    for path in verifications:
        assert read(path)['passed'] is True
    correlations = pd.read_csv(cancer / 'correlations.csv')
    coverage = pd.read_csv(cancer / 'coverage.csv')
    keys = ['centre', 'stratum', 'pair']
    joined = correlations.merge(coverage, on=keys, suffixes=('_correlation', '_coverage'), validate='one_to_one')
    assert len(joined) == len(correlations) == len(coverage) == 60
    for field in ('patients', 'pair_complete_n', 'pair_missing_n'):
        assert (joined[field + '_correlation'] == joined[field + '_coverage']).all()
    assert int(correlations.status.eq('estimated').sum()) == 32
    estimated = correlations.loc[correlations.status.eq('estimated')]
    assert (estimated.bootstrap_valid + estimated.bootstrap_undefined).eq(2000).all()
    assert estimated[['rho', 'ci_low', 'ci_high']].notna().all().all()
    assert (estimated.ci_low <= estimated.ci_high).all()
    assert np.isfinite(estimated[['rho', 'ci_low', 'ci_high']]).all().all()
    pdf = ROOT / 'figures/FigureD2_acquisition_era.pdf'
    document = fitz.open(pdf)
    assert len(document) == 1
    page = document[0]
    spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
    assert min((s['size'] for s in spans)) >= 8
    assert all((page.rect.contains(fitz.Rect(s['bbox'])) for s in spans))
    assert abs(page.rect.width / 72 * 25.4 - 174) < 0.001
    save(ROOT / 'figures/visual_qa.json', {'figure': identity(pdf), 'png': identity(ROOT / 'figures/FigureD2_acquisition_era.png'), 'width_mm': page.rect.width / 72 * 25.4, 'height_mm': page.rect.height / 72 * 25.4, 'minimum_text_pt': min((s['size'] for s in spans)), 'text_spans': len(spans), 'all_text_bounding_boxes_inside_page': True, 'visual_inspection': 'Author viewed the rendered PNG at full content on 2026-10-19: six panels, matching model colors and axes, readable labels/intervals, no clipped or overlapping labels. This is author visual QA, separate from numerical external verification.'})
    save(cancer / 'verification.json', {'passed': True, 'type': 'Final artifact consistency and cross-agent verification linkage', 'source_verification': identity(cancer / 'independent_external_verification.json'), 'dictionary_verification': identity(cancer / 'dictionary_verification.json'), 'measurement_semantics': identity(cancer / 'measurement_semantics.json'), 'rows': 60, 'estimated_rows': 32, 'all_public_denominators_match': True, 'all_estimated_bootstrap_counts_sum_to_2000': True})
    paths = [p for p in ROOT.rglob('*') if p.is_file() and 'private' not in p.relative_to(ROOT).parts and ('__pycache__' not in p.parts) and (p.name != 'public_delivery_manifest.json')]
    save(ROOT / 'public_delivery_manifest.json', {'scope': 'New 2026-10-19 covariate, regional provenance, acquisition-era and native-code association outputs only; no original files changed.', 'private_model_and_positional_clinical_arrays_excluded': True, 'external_verifications': [identity(p) for p in verifications], 'outputs': [identity(p) for p in sorted(paths)]})
    print('Public outputs sealed after independent verification.')
if __name__ == '__main__':
    main()
