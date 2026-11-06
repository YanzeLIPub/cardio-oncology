"""Build source-linked manuscript text and compact, public presentation tables."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
LABELS = {'all_original_patients': 'All original patients', 'lung_bronchus': 'Lung/bronchus', 'breast': 'Breast', 'digestive': 'Digestive', 'hematologic': 'Hematologic', 'other_defined_primary_site': 'Other defined primary site', 'multiple_broad_groups_or_C97': 'Multiple broad source-code groups or C97', 'secondary_or_unspecified_only': 'Secondary/unspecified codes only', 'no_interpretable_malignant_code': 'No interpretable malignant code', 'no_eligible_diagnostic_source': 'No eligible diagnostic source'}

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def estimate(row):
    if row.status != 'estimated':
        return 'Not estimated (pair n < 50)' if 'below_50' in row.status else 'Not estimated (constant variable)'
    text = f'{row.rho:.3f} ({row.ci_low:.3f} to {row.ci_high:.3f})'
    if row.bootstrap_undefined:
        text += f' [valid B={int(row.bootstrap_valid)}/2000]'
    return text

def main():
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text()))
    coverage = _release_resolve_config(json.loads((ROOT / 'coverage.json').read_text()))['cohorts']
    frame = pd.read_csv(ROOT / 'correlations.csv')
    groups = ['all_original_patients', *protocol['strata_order']]
    covrows, assrows, full_text = ([], [], [])
    for centre in ('SCH', 'WHU'):
        local = frame.loc[frame.centre.eq(centre)]
        for group in groups:
            s = local.loc[local.stratum.eq(group)].set_index('pair')
            covrow = {'centre': centre, 'stratum': LABELS[group], 'patients': int(s.patients.iloc[0])}
            assrow = covrow.copy()
            for pair in ('NSE--albumin', 'NSE--WBC', 'WBC--albumin'):
                covrow[pair + '_complete_n'] = int(s.loc[pair, 'pair_complete_n'])
                assrow[pair + '_rho_95CI'] = estimate(s.loc[pair])
            covrows.append(covrow)
            assrows.append(assrow)
        full = local.loc[local.stratum.eq('all_original_patients')]
        full_text.append(centre + ': ' + '; '.join((f"{r.pair.replace('--', '-')} rho {estimate(r)}, n={r.pair_complete_n:,}" for r in full.itertuples())))
    pd.DataFrame(covrows).to_csv(ROOT / 'publication_coverage.csv', index=False)
    pd.DataFrame(assrows).to_csv(ROOT / 'publication_correlations.csv', index=False)
    defined = frame.loc[frame.centre.eq('SCH') & frame.stratum.isin(protocol['primary_groups'])]
    range_text = '; '.join((f"{pair.replace('--', '-')} {part.rho.min():.3f} to {part.rho.max():.3f} ({int(part.status.eq('estimated').sum())}/5 groups estimable)" for pair, part in defined.groupby('pair', sort=False)))
    n_sch = coverage['SCH']['patients_valid_malignant_category']
    n_whu = coverage['WHU']['patients_valid_malignant_category']
    output = {'section_title': 'D5. Native-code cancer-stratified biomarker associations', 'methods': 'An exploratory patient-level sensitivity analysis classified original patients using native current/discharge diagnostic codes from hash-verified source staging. We did not use OMOP concepts, change outcome labels, or infer cancer type from free text. After exclusion of conflicting source slots, full-string codes were checked against the official ICD-10-WHO 2019 three-character malignant-neoplasm category dictionary. Prespecified groups were lung/bronchus, breast, digestive, hematologic and other defined primary sites. Patients with more than one broad group or C97 were retained separately; C76-C80 alone formed a secondary/unspecified group and did not determine a primary site. Missing or uninterpretable codes and absent eligible sources remained separate unknown groups, not cancer-negative controls. Correlations of NSE-albumin, NSE-WBC and WBC-albumin used the original full-record patient means with pairwise-complete observations and no zero imputation. Here WBC denotes the historical WBC-labelled model input, not a newly verified cross-center blood-cell-count measurement. Spearman correlations and percentile 95% intervals from 2,000 patient bootstrap resamples were reported for every prespecified group with at least 50 complete pairs and nonconstant variables. Each resample was reranked; constant resamples were counted and excluded from interval estimation, with the valid bootstrap count reported. These are full-record descriptive strata, not baseline confounder adjustment.', 'coverage_results': f'Interpretable malignant-neoplasm source codes were available for {n_sch:,}/38,509 SCH patients ({100 * n_sch / 38509:.2f}%) and {n_whu}/2,562 WHU patients ({100 * n_whu / 2562:.2f}%). SCH groups comprised lung/bronchus 9,139, breast 2,683, digestive 12,072, hematologic 890, other defined primary site 7,681, multiple broad groups/C97 3,086, secondary/unspecified only 245, and unknown 2,713 patients. WHU had 2,190 patients with an eligible source but no interpretable malignant code and 316 without an eligible diagnostic source. Its individual cancer-code strata contained at most 29 patients, below the prespecified estimation threshold; this is insufficient information, not a negative association finding.', 'whole_cohort_results': '. '.join(full_text) + '.', 'stratified_results': 'Among estimable associations in the five prespecified defined primary-site groups in SCH, point estimates ranged as follows: ' + range_text + '. All strata, complete-pair denominators, uncertainty intervals and nonestimable groups are reported, without selecting favorable directions.', 'interpretation': 'These descriptive associations can be presented as hypothesis-generating evidence only. The opposite full-cohort directions for NSE-albumin in SCH and WHU do not support a uniform inverse biological relationship. Stratification by recorded cancer-code group does not establish an independent biological mechanism or remove confounding by stage, antitumor treatment, medication, acute illness, selective laboratory measurement, or reverse causation. Full-record diagnoses and patient means may include information after the cardiovascular diagnostic label. Stage and treatment fields exist in native sources, but this analysis does not convert those fields into validated baseline exposures or claim comprehensive adjustment. The SCH historical WBC-labelled input has 20,743 zero patient means among 20,880 finite values; its source specimen/unit identity is not established. This is a data-quality flag, not evidence that the input is urine-derived. WHU exact-name source items carry G/L, but this cannot validate the SCH field. Hence the WBC-labelled correlations are numerical audits, not completed clinical blood-WBC mechanism analyses. Original laboratory-unit and measurement-window limitations also constrain NSE-albumin interpretation. Highly tied strata yield undefined constant bootstrap resamples, explicitly annotated in the tables.', 'coverage_table_caption': 'Native-code cancer group coverage and complete-pair denominators. Counts refer to original patients. Full-cohort rows are additional summaries and must not be added to the mutually exclusive strata. No interpretable code does not mean no cancer. All biomarker values are original full-record patient means; pairs can have different denominators.', 'association_table_caption': 'Spearman correlations (percentile 95% patient-bootstrap intervals) of original observed full-record patient means. NSE, neuron-specific enolase; WBC, historical WBC-labelled model input with unresolved SCH specimen/unit identity, not a verified blood-cell count. Each of 2,000 resamples was reranked. Bracketed valid B indicates how many resamples remained after excluding undefined constant samples; otherwise all 2,000 were valid. Estimates were withheld for fewer than 50 complete pairs or a constant variable. These are exploratory descriptive, unadjusted associations, not pre-event or causal estimates. See the companion coverage table for complete-pair n.', 'tables': {'coverage': (ROOT / 'publication_coverage.csv').as_posix(), 'correlations': (ROOT / 'publication_correlations.csv').as_posix(), 'machine_readable_all_rows': (ROOT / 'correlations.csv').as_posix()}, 'verification_file': (ROOT / 'independent_external_verification.json').as_posix(), 'dictionary_reference': protocol['dictionary_sources'][0], 'sources': [identity(ROOT / name) for name in ('protocol.json', 'coverage.json', 'correlations.csv', 'dictionary_verification.json', 'measurement_semantics.json')]}
    output['main_methods'] = 'Exploratory cancer-stratified correlations used the original Chinese patient-level biomarker means and native current/discharge diagnostic codes, without OMOP mapping or relabelling. Conflicting source slots were excluded and codes checked against the 89 valid ICD-10-WHO 2019 malignant-neoplasm categories. Prespecified primary-site groups, multiple broad groups/C97, secondary/unspecified-only and unknown groups were retained separately; missing codes did not indicate absence of cancer. Spearman correlations used finite complete pairs without imputation, required at least 50 patients and nonconstant variables, and were recomputed in 2,000 patient bootstrap resamples. This full-record descriptive stratification did not adjust for cancer stage, treatment, medication or acute illness (Supplementary D5).'
    output['main_results'] = 'Interpretable native malignant-neoplasm codes covered 35,796/38,509 SCH patients and 56/2,562 WHU patients after strict source-conflict exclusion. In SCH, the full-cohort NSE-albumin correlation was weakly inverse (rho -0.062, 95% CI -0.082 to -0.041; 10,052 complete pairs); WHU showed the opposite direction (0.156, 0.097 to 0.208; 616 pairs). Supplementary D5 reports all prespecified cancer strata, denominators and bootstrap intervals without selecting favorable directions. WHU cancer-specific groups contained at most 29 patients and were not estimated; this insufficiency is not evidence of no association. Analyses involving the historical WBC-labelled input require additional caution: SCH specimen/unit identity remains unverified, and 20,743 of 20,880 finite patient means were zero. These numerical input correlations are not validated blood-WBC mechanism analyses. The descriptive results do not establish a uniform inflammatory pathway or remove confounding by cancer stage, treatment, acute illness, selective measurement or reverse causation.'
    output['supplement_paragraphs'] = [output[name] for name in ('methods', 'coverage_results', 'whole_cohort_results', 'stratified_results', 'interpretation')]
    assert 80 <= len(output['main_methods'].split()) <= 120
    assert 120 <= len(output['main_results'].split()) <= 170
    with (ROOT / 'PUBLICATION_TEXT.json').open('w', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
if __name__ == '__main__':
    main()
