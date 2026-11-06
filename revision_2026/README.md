# Revision analysis source

This directory contains analysis code and scientific settings for the cardio-oncology manuscript revision. Patient records, patient-level matrices and predictions, clinical review lists, fitted models, and local connection settings are not included. Access to the original data remains governed by the study's data-availability statement.

`project/` preserves the revision-analysis directory structure. `historical_china/` and `historical_pipeline/` contain the original source dependencies selected for the analysis. The dated directories identify distinct analyses; results from different validation designs should not be combined as one experiment.

The [analysis map](ANALYSIS_MAP.md) links each analysis family to its preparation, fitting, summary and verification entry points, followed by the figure-preparation chain.

## Running an analysis

Create the relevant Python environment using the supplied version locks. The Chinese and MIMIC reproduction environments have separate locks. Source-based and plotting analyses may also require the packages listed in `SOURCE_CHANGES.json` under `external_import_roots`.

Copy `config.example.json` to an untracked `config.local.json`, then enter the local paths to the controlled inputs and a private working directory. Input directories retain the relative filenames expected by the corresponding analysis. `CARDIO_WORK_ROOT` must be outside this source checkout. The JSON keys are also available as environment variables.

Figure scripts retain Arial. `CARDIO_FONT_DIR` can specify a directory containing the original Arial font files; otherwise the system Windows font directory or installed Arial is used.

Run an included entry point through the launcher:

```text
python run.py --config config.local.json project/revision_2026-11-06/review_simulation_v1/run_simulation.py
```

The launcher copies the released source to the private working directory before running it. This preserves the scripts' relative paths and keeps generated matrices, predictions and model objects outside the public checkout. The example simulation reads previously generated patient-isolated predictions; it does not train a model. Other entry points have their own preparation and analysis dependencies.

Use the original analysis order: prepare the required cohort or measurement matrices, run the corresponding validation, then summarize and verify that analysis. The protocol files specify cohorts, features, seeds, model settings and comparison definitions. Source-data searches and clinical record locators require authorized local records. The included code does not supply those records.

## Scope of the release changes

Machine-specific root paths were replaced with configuration references. Required source modules resolve from the staged release. Local file-identity expectations and internal audit-routing fields were removed. `SOURCE_CHANGES.json` records these changes. Statistical model settings, cohort and label rules, feature order, random seeds, fitting procedures and probability definitions are retained.

The source has been parsed and compiled, and the launcher's help and path configuration checked. This packaging step does not rerun the patient cohorts or establish clinical prediction timing. The manuscript describes the scientific scope and remaining evidence limitations.
