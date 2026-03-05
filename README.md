# Cardio-Oncology MACE Prediction

Code for:
> **Machine learning prediction of major adverse cardiovascular events
> in cancer patients across three care architectures**
> *The Lancet Digital Health* (under review)

## Repository structure

```
models/                  XGBoost ensemble model definition
utils/                   Shared utilities (feature names, plot style)
calibration/             Post-hoc Platt / isotonic recalibration
MIMIC-3/                 MIMIC-III v1.4 pipeline (steps 01-08)
Manuscript/
  v13/                   Final metric computation + figure generation
  Admission model/       Admission-level secondary analysis (Fig S11-S13)
MERGE/                   Published figure assembly scripts
figures/                 Supplementary figure scripts
```

## Reproducing figures

```bash
pip install -r requirements.txt

# Main + supplementary figures (requires pre-computed metrics in outputs/)
python MERGE/pub_main_figures.py
python MERGE/pub_supp_figures.py
```

## MIMIC-III pipeline

```bash
# Requires PhysioNet credentialed access to MIMIC-III v1.4
# Set MIMIC_PATH environment variable to your local MIMIC directory
python MIMIC-3/run_mimic_pipeline.py --mimic_path $MIMIC_PATH
```

Steps executed in order: `01` identify patients -> `02` extract features ->
`03` define MACE -> `04` predict -> `05` pretrain transfer -> `06` separate
transfer -> `08` paper methods.

## Data availability

Raw data are **not** included.
- **SCH / WHU cohorts**: available on reasonable request to the
  corresponding author.
- **MIMIC-III v1.4**: https://physionet.org/content/mimiciii/1.4/
  (requires CITI training completion).

## Requirements

```
xgboost>=1.7
scikit-learn>=1.3
lifelines>=0.27
matplotlib>=3.7
seaborn>=0.12
numpy>=1.24
pandas>=2.0
```

## Reproducibility

All scripts use `RANDOM_SEED = 42` by default.
