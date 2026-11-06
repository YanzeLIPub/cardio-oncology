"""Prespecified complete-pair patient Spearman correlations and bootstrap CIs."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import hashlib
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parent

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8')))
    preparation = _release_resolve_config(json.loads((ROOT / 'preparation_manifest.json').read_text(encoding='utf-8')))
    for record in preparation['outputs']:
        pass
    inputs = [identity(ROOT / 'protocol.json'), identity(ROOT / 'preparation_manifest.json'), identity(Path(__file__))]
    inputs.extend((identity(ROOT / f'private/{c}_strata.npz') for c in ('SCH', 'WHU')))
    save(ROOT / 'analysis_lock.json', {'inputs': inputs, 'protocol_frozen_before_correlation_evaluation': True})
    results, draws = ([], {})
    for cohort_index, centre in enumerate(('SCH', 'WHU')):
        with np.load(ROOT / f'private/{centre}_strata.npz', allow_pickle=False) as z:
            x = z['biomarkers']
            strata = z['strata']
            names = list(z['biomarker_names'])
        for stratum_index, group in enumerate(['all_original_patients', *protocol['strata_order']]):
            member = np.ones(len(x), dtype=bool) if group == 'all_original_patients' else strata == group
            for pair_index, (a, b) in enumerate(protocol['pairs']):
                pair = x[:, [names.index(a), names.index(b)]]
                data = pair[member & np.isfinite(pair).all(axis=1)].astype(float)
                n = len(data)
                seed = protocol['bootstrap']['seed_base'] + 1000 * cohort_index + 10 * stratum_index + pair_index
                result = {'centre': centre, 'stratum': group, 'pair': a + '--' + b, 'patients': int(member.sum()), 'pair_complete_n': n, 'pair_missing_n': int(member.sum()) - n, 'status': 'estimated', 'rho': None, 'ci_low': None, 'ci_high': None, 'bootstrap_requested': 2000, 'bootstrap_valid': 0, 'bootstrap_undefined': 0, 'bootstrap_seed': seed}
                if n < 50:
                    result['status'] = 'not_estimated_pair_n_below_50'
                elif any((np.unique(data[:, j]).size < 2 for j in (0, 1))):
                    result['status'] = 'not_estimated_constant_variable'
                else:
                    result['rho'] = float(spearmanr(data[:, 0], data[:, 1]).statistic)
                    rng = np.random.default_rng(seed)
                    sample_rhos = np.full(2000, np.nan)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', scipy.stats.ConstantInputWarning)
                        for i in range(2000):
                            sample = data[rng.integers(0, n, size=n)]
                            sample_rhos[i] = spearmanr(sample[:, 0], sample[:, 1]).statistic
                    finite = sample_rhos[np.isfinite(sample_rhos)]
                    result['bootstrap_valid'] = len(finite)
                    result['bootstrap_undefined'] = 2000 - len(finite)
                    assert len(finite) > 0
                    result['ci_low'], result['ci_high'] = [float(v) for v in np.quantile(finite, [0.025, 0.975])]
                    draws[f'{centre}_{stratum_index}_{pair_index}'] = sample_rhos
                results.append(result)
            print(centre, group, 'completed', flush=True)
    pd.DataFrame(results).to_csv(ROOT / 'correlations.csv', index=False)
    np.savez_compressed(ROOT / 'private/bootstrap_rhos.npz', **draws)
    for record in inputs:
        pass
    save(ROOT / 'correlations.json', {'rows': results, 'interpretation': protocol['scope'], 'limitations': protocol['limitations']})
    save(ROOT / 'execution.json', {'complete': True, 'inputs': inputs, 'outputs': [identity(ROOT / name) for name in ('correlations.csv', 'correlations.json', 'private/bootstrap_rhos.npz')], 'estimated_rows': sum((r['status'] == 'estimated' for r in results)), 'total_rows': len(results), 'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__}})
if __name__ == '__main__':
    main()
