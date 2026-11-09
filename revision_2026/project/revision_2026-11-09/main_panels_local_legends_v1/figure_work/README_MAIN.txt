MAIN FIGURE LAYOUT UPDATE

Run: python build_main_figures.py
Dependency: PyMuPDF (imported as fitz).
Figure rendering was checked with Python 3.12.14 and PyMuPDF 1.26.7.

This script regenerates Figures 2 and 5 from aggregate data and updates the
three calibration legends in Figure 4. It also exports the individual panel
regions used in reviewer responses. It does not refit models or read patient
records. Figures 1 and 3 remain unchanged and are supplied separately.

Five aggregate CSV inputs, in inputs/:

1. trajectory_binned_scores.csv
   The 60 binned score means and record-level standard errors recovered for
   the v30 Figure 2 restoration. They are unchanged from v30 figure_work/
   trajectory_binned_scores.csv. Record position is not time before an event.

2. figure2f_common_laboratory.csv
   Eight full-target/covered-target performance estimates and confidence
   limits from v30 Supplementary_Data_1.zip:
   common_transfer/aggregate/transfer_metrics.csv;
   common_transfer/aggregate/local_metrics.csv;
   common_transfer/coverage_sensitivity_v1/metrics.csv.
   Only ensemble_median rows are used. These are the values formerly shown
   in Figure S8a and S8c; they are separate from the historical Figure 2a/b
   transfer matrix. The intervals condition on fixed fitted predictions.

3. figure5ab_original_importance.csv
   The original 25 displayed importance values per Chinese centre, taken
   from the historical aggregate feature_importance_comparison.csv fields
   SCH_norm and WHU_norm. English display labels and values were checked
   against the v30 Figure 5 vector bars. Values have not been re-estimated.

4. figure5c_feature_associations.csv
   The eight selected rows of shap/results/cross_cohort_associations.csv in
   v30 Supplementary_Data_1.zip, where
   opposite_both_q_and_five_fold_agreement is True. The two rho values per
   feature reproduce the former Figure S14a.

5. figure5d_seed_retention.csv
   The same eight features at seeds 42, 43, 44 and 45 from
   shap_seeds/results/retention.csv in v30 Supplementary_Data_1.zip.
   Yes means the joint-q/opposite-sign/five-fold criteria are retained and
   the pooled signs match seed 42. All 32 displayed cells were checked
   against the former Figure S15c.

Additional vector input:

inputs/Figure_4_base.pdf must be supplied from the corresponding v30
publication vector Figure_4.pdf. It is not one of the five aggregate CSV
inputs. The script replaces only the common calibration key with three
local keys; all 432 original non-background plot drawings remain unchanged.
The availability of code and aggregate CSVs does not imply that every
underlying input or historical source file is publicly distributed.

Outputs are written to ../figures/ as vector PDF/SVG and rendered PNG.
The build report records panel bounds and the local-legend review. No
model estimates, confidence limits, feature directions or cell labels are
changed by this layout update.
