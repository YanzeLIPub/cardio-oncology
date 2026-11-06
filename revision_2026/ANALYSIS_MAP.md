# Analysis and figure preparation map

Paths below are relative to this release. Use `run.py` with the corresponding entry point and its documented arguments. Preparation, fitting and summary stages are separate; later analyses reuse the patient-isolated predictions and roles. The source-data prerequisites are stated in each protocol.

| Analysis | Entry points in dependency order |
| --- | --- |
| patient isolated validation | [prepare_china.py](project/revision_2026-10-15/original_v5_independent_validation_v1/prepare_china.py) → [prepare_mimic.py](project/revision_2026-10-15/original_v5_independent_validation_v1/prepare_mimic.py) → [run_validation.py](project/revision_2026-10-15/original_v5_independent_validation_v1/run_validation.py) → [summarize_validation.py](project/revision_2026-10-15/original_v5_independent_validation_v1/summarize_validation.py) → [verify_validation.py](project/revision_2026-10-15/original_v5_independent_validation_v1/verify_validation.py) |
| mimic measurement windows | [prepare_windows.py](project/revision_2026-10-15/landmark_completion_v1/mimic/prepare_windows.py) → [run_windows.py](project/revision_2026-10-15/landmark_completion_v1/mimic/run_windows.py) → [summarize_windows.py](project/revision_2026-10-15/landmark_completion_v1/mimic/summarize_windows.py) → [verify_windows.py](project/revision_2026-10-15/landmark_completion_v1/mimic/verify_windows.py) → [verify_summary.py](project/revision_2026-10-15/landmark_completion_v1/mimic/verify_summary.py) |
| common transfer | [prepare_common.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/common_transfer/prepare_common.py) → [run_common.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/common_transfer/run_common.py) → [evaluate_common_v2.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/common_transfer/evaluate_common_v2.py) → [evaluate_coverage.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/common_transfer/coverage_sensitivity_v1/evaluate_coverage.py) → [finalize_common.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/common_transfer/finalize_common.py) |
| historical transfer | [run_historical_transfer.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/historical_transfer/run_historical_transfer.py) |
| missingness | [run_missingness.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/missingness/run_missingness.py) → [verify_missingness_independent.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/missingness/verify_missingness_independent.py) |
| shap | [run_heldout_shap.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/shap/run_heldout_shap.py) → [verify_heldout_shap.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/shap/verify_heldout_shap.py) |
| regional | [audit_regional.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/regional/audit_regional.py) → [run_regional.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/regional/run_regional.py) → [verify_regional_independent.py](project/revision_2026-10-16/reviewer_priority_analyses_v1/regional/verify_regional_independent.py) |
| high coverage | [run_high_coverage.py](project/revision_2026-10-18/four_analysis_v1/high_coverage/run_high_coverage.py) → [verify_high_coverage_independent.py](project/revision_2026-10-18/four_analysis_v1/high_coverage/verify_high_coverage_independent.py) |
| nested tuning | [run_nested_tuning.py](project/revision_2026-10-18/four_analysis_v1/nested_tuning/run_nested_tuning.py) |
| shap seeds | [run_seed_stability.py](project/revision_2026-10-18/four_analysis_v1/shap_seeds/run_seed_stability.py) → [verify_stage1_independent.py](project/revision_2026-10-18/four_analysis_v1/shap_seeds/verify_stage1_independent.py) → [verify_seed_aggregates.py](project/revision_2026-10-18/four_analysis_v1/shap_seeds/verify_seed_aggregates.py) |
| transfer adaptation | [run_adaptation.py](project/revision_2026-10-18/four_analysis_v1/transfer_adaptation/run_adaptation.py) → [replay_saved_models.py](project/revision_2026-10-18/four_analysis_v1/transfer_adaptation/replay_saved_models.py) → [finalize_adaptation.py](project/revision_2026-10-18/four_analysis_v1/transfer_adaptation/finalize_adaptation.py) |
| aggregation | [prepare.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/aggregation/prepare.py) → [run_v2.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/aggregation/run_v2.py) → [summarize.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/aggregation/summarize.py) → [verify.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/aggregation/verify.py) → [independent_verify.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/aggregation/independent_verify.py) |
| covariate validation | [audit_context.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/covariate_validation/audit_context.py) → [audit_clock_precision.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/covariate_validation/audit_clock_precision.py) → [run_temporal.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/covariate_validation/run_temporal.py) → [verify_temporal.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/covariate_validation/verify_temporal.py) → [prepare_source_codes.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/covariate_validation/cancer_associations/prepare_source_codes.py) → [run_correlations.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/covariate_validation/cancer_associations/run_correlations.py) → [plot_era.py](project/revision_2026-10-19/remaining_requirements_v1/analysis/covariate_validation/plot_era.py) |
| decision sensitivity | [run_analysis.py](project/revision_2026-10-19/substantive_gaps_v1/analysis/utility_uncertainty/run_analysis.py) → [independent_verify.py](project/revision_2026-10-19/substantive_gaps_v1/analysis/utility_uncertainty/independent_verify.py) → [build_figure_and_report.py](project/revision_2026-10-19/substantive_gaps_v1/analysis/utility_uncertainty/build_figure_and_report.py) |
| leakage | [audit_feature_lineage.py](project/analysis/r3_leakage_2026-10-21_v1/china/audit_feature_lineage.py) → [prepare_demographic_control.py](project/analysis/r3_leakage_2026-10-21_v1/china/prepare_demographic_control.py) → [audit_lineage.py](project/analysis/r3_leakage_2026-10-21_v1/mimic/audit_lineage.py) → [verify_demographics_and_eligibility.py](project/analysis/r3_leakage_2026-10-21_v1/mimic/verify_demographics_and_eligibility.py) → [run_reference.py](project/analysis/r3_leakage_2026-10-21_v1/run_reference.py) → [verify_reference.py](project/analysis/r3_leakage_2026-10-21_v1/design/verification/verify_reference.py) → [build_evidence.py](project/analysis/r3_leakage_2026-10-21_v1/mimic/publication/build_evidence.py) |
| selection sensitivity | [prepare_selection.py](project/analysis/mimic_selection_2026-10-24_v1/preparation/prepare_selection.py) → [verify_preparation.py](project/analysis/mimic_selection_2026-10-24_v1/design/verify_preparation.py) → [seal_preparation_gate.py](project/analysis/mimic_selection_2026-10-24_v1/design/seal_preparation_gate.py) → [run_selection.py](project/analysis/mimic_selection_2026-10-24_v1/run_selection.py) → [verify_results.py](project/analysis/mimic_selection_2026-10-24_v1/design/verify_results.py) → [verify_publication_summary.py](project/analysis/mimic_selection_2026-10-24_v1/design/verify_publication_summary.py) |
| source availability | [run_source_eligible_whu.py](project/revision_2026-10-28/source_eligible_v1/analysis/run_source_eligible_whu.py) → [summarize_source_eligible_whu.py](project/revision_2026-10-28/source_eligible_v1/analysis/summarize_source_eligible_whu.py) → [compute_calibration_uncertainty.py](project/revision_2026-10-28/source_eligible_v1/analysis/compute_calibration_uncertainty.py) → [verify_source_eligible_whu.py](project/revision_2026-10-28/source_eligible_v1/analysis/verify_source_eligible_whu.py) |
| label components | [run_component_analysis.py](project/revision_2026-10-30/submission_revision_v1/analysis/run_component_analysis.py) |
| label sensitivity | [run_label_bounds.py](project/revision_2026-10-30/submission_revision_v1/label_sensitivity/run_label_bounds.py) |
| shared age | [run_age_transfer.py](project/revision_2026-10-30/submission_revision_v1/analysis_sch/run_age_transfer.py) |
| source reports | [prepare_whu.py](project/revision_2026-11-03/source_based_analyses_v1/analysis/prepare_whu.py) → [fit_whu.py](project/revision_2026-11-03/source_based_analyses_v1/analysis/fit_whu.py) → [complete_ap_intervals.py](project/revision_2026-11-03/source_based_analyses_v1/analysis/complete_ap_intervals.py) → [analyze_sch_poa.py](project/revision_2026-11-03/source_based_analyses_v1/analysis/analyze_sch_poa.py) → [reconcile_sch_candidates.py](project/revision_2026-11-03/source_based_analyses_v1/analysis/reconcile_sch_candidates.py) → [sch_candidate_sensitivity.py](project/revision_2026-11-03/source_based_analyses_v1/analysis/sch_candidate_sensitivity.py) → [plot_and_check.py](project/revision_2026-11-03/source_based_analyses_v1/analysis/plot_and_check.py) |
| source orders | [scan_orders.py](project/revision_2026-11-05/whu_orders_actions_v1/scan_orders.py) → [summarize_orders.py](project/revision_2026-11-05/whu_orders_actions_v1/summarize_orders.py) |
| review simulation | [run_simulation.py](project/revision_2026-11-06/review_simulation_v1/run_simulation.py) → [verification.py](project/revision_2026-11-06/review_simulation_v1/verification.py) → [plot_results.py](project/revision_2026-11-06/review_simulation_v1/plot_results.py) |
| calibration | [summarize_validation.py](project/revision_2026-10-15/original_v5_independent_validation_v1/summarize_validation.py) → [finish_references.py](project/revision_2026-11-03/supplement_streamline_v1/editorial/finish_references.py) |
| model validation | [run_validation.py](project/revision_2026-10-15/original_v5_independent_validation_v1/run_validation.py) → [finish_references.py](project/revision_2026-11-03/supplement_streamline_v1/editorial/finish_references.py) |
| figure3 | [consolidate_figure3.py](project/revision_2026-11-01/journal_compliance_v1/editorial/consolidate_figure3.py) |

The shared four-laboratory transfer analysis is WHU–MIMIC. The shared-age analysis is a different six-direction analysis. The 0/24/48-hour analyses restrict measurement acquisition; they do not establish onset of future events. `finish_references.py` in this release exports only the three retained aggregate tables, without modifying a manuscript.

## Historical primary-model sources

The Chinese primary model source is [optionB_parallel_analysis.py](historical_china/optionB_parallel_analysis.py). The MIMIC primary-CV source is [compute_v13_metrics.py](historical_pipeline/Manuscript/v13/compute_v13_metrics.py). The older 96-input MIMIC importance model is a separate historical analysis. The reconstruction and prediction-replay helpers are included under `project/audit/`.

## Figure preparation

The final presentation contains five main figures and 29 supplementary figures. The stages below generate added panels, prepare retained vector assets, consolidate panels and apply the final first-citation numbering. Historical figure generators may also contain archived outputs that are not retained in the manuscript; the final preparation chain and numbering map determine the current figure set.

**Historical retained plots**

- [historical_pipeline/Manuscript/v13/gen_figures_v13.py](historical_pipeline/Manuscript/v13/gen_figures_v13.py)
- [historical_pipeline/Manuscript/v13/gen_supp_figures_v13.py](historical_pipeline/Manuscript/v13/gen_supp_figures_v13.py)
- [historical_pipeline/Manuscript/v13/figures/add/shap_direction_comparison.py](historical_pipeline/Manuscript/v13/figures/add/shap_direction_comparison.py)
- [historical_pipeline/Manuscript/v13/figures/add/analyses_ABC_reversal_mechanism.py](historical_pipeline/Manuscript/v13/figures/add/analyses_ABC_reversal_mechanism.py)
- [historical_pipeline/Manuscript/v13/figures/add/run_C1_C4_analyses.py](historical_pipeline/Manuscript/v13/figures/add/run_C1_C4_analyses.py)
- [historical_pipeline/Manuscript/v13/figures/add/nse_albumin_deep_dive.py](historical_pipeline/Manuscript/v13/figures/add/nse_albumin_deep_dive.py)
- [historical_pipeline/utils/__init__.py](historical_pipeline/utils/__init__.py)
- [historical_pipeline/utils/style.py](historical_pipeline/utils/style.py)
- [historical_pipeline/utils/feature_names_en.py](historical_pipeline/utils/feature_names_en.py)

**Added analysis panels and Figure 4 integration**

- [project/revision_2026-10-15/incremental_revision_v1/figures/build_incremental_figures.py](project/revision_2026-10-15/incremental_revision_v1/figures/build_incremental_figures.py)
- [project/revision_2026-10-15/incremental_revision_v1/figures/plot_additional_panels.py](project/revision_2026-10-15/incremental_revision_v1/figures/plot_additional_panels.py)
- [project/revision_2026-10-15/landmark_completion_v1/main/build_figures.py](project/revision_2026-10-15/landmark_completion_v1/main/build_figures.py)
- [project/revision_2026-10-16/priority_revision_v1/build_figures.py](project/revision_2026-10-16/priority_revision_v1/build_figures.py)
- [project/revision_2026-10-18/expanded_revision_v1/build_figures.py](project/revision_2026-10-18/expanded_revision_v1/build_figures.py)

**Native vector recovery and original Figure 3 label changes**

- [project/revision_2026-10-30/submission_revision_v1/figure_work/enlarge_figure3.py](project/revision_2026-10-30/submission_revision_v1/figure_work/enlarge_figure3.py)
- [project/revision_2026-10-30/submission_revision_v1/figure_work/finalize_figure3.py](project/revision_2026-10-30/submission_revision_v1/figure_work/finalize_figure3.py)
- [project/revision_2026-10-30/submission_revision_v1/figure_work/main_other/enlarge_figures.py](project/revision_2026-10-30/submission_revision_v1/figure_work/main_other/enlarge_figures.py)
- [project/revision_2026-11-01/journal_compliance_v1/editorial/consolidate_figure3.py](project/revision_2026-11-01/journal_compliance_v1/editorial/consolidate_figure3.py)
- [project/revision_2026-11-02/graphical_panels_v1/editorial/plot_panels.py](project/revision_2026-11-02/graphical_panels_v1/editorial/plot_panels.py)
- [project/revision_2026-11-02/vector_figures_v1/scripts/export_vectors.py](project/revision_2026-11-02/vector_figures_v1/scripts/export_vectors.py)
- [project/revision_2026-11-02/vector_figures_v1/scripts/recover_scatter.py](project/revision_2026-11-02/vector_figures_v1/scripts/recover_scatter.py)
- [project/revision_2026-11-02/vector_figures_v1/scripts/replot_svg_sources.py](project/revision_2026-11-02/vector_figures_v1/scripts/replot_svg_sources.py)
- [project/revision_2026-10-24/figure_readability_v1/figure_work/figure3/improve_figure3_readability.py](project/revision_2026-10-24/figure_readability_v1/figure_work/figure3/improve_figure3_readability.py)

**2026-11-03 source-report panels and supplementary table visualization**

- [project/revision_2026-11-03/source_based_analyses_v1/analysis/plot_and_check.py](project/revision_2026-11-03/source_based_analyses_v1/analysis/plot_and_check.py)
- [project/revision_2026-11-03/supplement_streamline_v1/editorial/plot_panels.py](project/revision_2026-11-03/supplement_streamline_v1/editorial/plot_panels.py)

**2026-11-04 four-category UMAP and 2026-11-06 simulation**

- [project/revision_2026-11-04/figure3e_four_colors_v1/editorial/recolor_figure.py](project/revision_2026-11-04/figure3e_four_colors_v1/editorial/recolor_figure.py)
- [project/revision_2026-11-06/review_simulation_v1/plot_results.py](project/revision_2026-11-06/review_simulation_v1/plot_results.py)

**Final panel deduplication and first-citation numbering for all 34 figures**

- [project/revision_2026-11-06/panel_streamline_v1/editorial/streamline_figure_assets.py](project/revision_2026-11-06/panel_streamline_v1/editorial/streamline_figure_assets.py)
- [project/revision_2026-11-06/first_citation_order_v1/editorial/renumber.py](project/revision_2026-11-06/first_citation_order_v1/editorial/renumber.py)

The final figure and table identifiers are recorded in [numbering_map.json](project/revision_2026-11-06/first_citation_order_v1/review/numbering_map.json). Some preparation steps use retained vector assets. Recovery of the exact historical SHAP environment and every historical figure-editing step remains incomplete; this source release does not imply that every old plot can be regenerated point for point from raw records.

## Environments

Separate complete reconstructed CPU-environment locks are provided for [China](project/audit/environment_rebuild_2026-10-27_v1/environment/china.requirements.lock.txt) and [MIMIC](project/audit/environment_rebuild_2026-10-27_v1/environment/mimic.requirements.lock.txt). [ENVIRONMENTS.json](ENVIRONMENTS.json) supplies the recorded Python and package versions for the revision analyses, grouped by their originating analysis records. These are extracted version fields, not local machine or patient-data manifests. The historical SHAP version is not claimed to have been recovered.
