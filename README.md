# Cardio oncology model transportability

Analysis code for **Exploring biomarker semantic drift and care architecture specificity in cardio oncology model transportability**, revised for *npj Cardiovascular Health*.

## Current revision

The [`revision_2026`](revision_2026/) directory contains the study analysis and figure-generation source code, including the revision analyses. Its README gives the analysis map, data layout, execution commands and environment requirements.

The revision evaluates source-specific recorded cardiovascular diagnosis labels in SCH and WHU patients and MIMIC-III admissions. It distinguishes the historical analyses from patient-isolated validation and other sensitivity analyses. The original directories below retain their historical filenames; the current analysis definitions and reproduction instructions are in `revision_2026`.

## Historical source directories

- `models/` and `utils/`: model and plotting utilities.
- `calibration/`: historical recalibration scripts.
- `MIMIC-3/`: historical MIMIC-III extraction and modelling pipeline.
- `Manuscript/`, `MERGE/` and `figures/`: historical analysis and figure scripts.

## Data and execution

Raw clinical records, patient-level derived tables, prediction vectors, fitted model files and local logs are not distributed with this code. SCH and WHU records remain subject to the access conditions in the manuscript. MIMIC-III requires the relevant PhysioNet credentials and data-use agreement: https://physionet.org/content/mimiciii/1.4/.

Use the configurable paths and per-analysis instructions in [`revision_2026/README.md`](revision_2026/README.md) with an authorised local copy of the required inputs. Seeds, model versions and validation designs are specified for each analysis; they are not assumed to be identical across historical and revision runs.
