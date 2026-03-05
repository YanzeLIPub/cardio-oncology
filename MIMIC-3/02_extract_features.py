#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2 — Extract Lab Features from MIMIC-III for Cardio-Oncology Cohort
========================================================================
Maps MIMIC-III lab items to the 43 model features (best-effort matching).
For each patient's index admission (first cancer + cardiac co-occurrence),
extracts the median lab value from all results during that admission.

Features available in MIMIC (mapped from original model):
  Biological:  Age, Sex, LDH, Albumin, Creatinine, Glucose, ALT, AST,
               Sodium, Potassium, Magnesium, Phosphorus, Uric Acid,
               Total Protein, Globulin (calc.), Direct Bilirubin,
               Triglycerides, HDL, LDL, Total Cholesterol, D-Dimer,
               Creatine Kinase, α-HBDH, HbA1c, Ferritin, Prealbumin,
               Amylase, Cystatin C, AFP, HBcAb, Plateletcrit,
               Apolipoprotein B, Apolipoprotein AI, ADA, CYFRA 21-1, SCC

  Socioeconomic (unavailable in MIMIC → NaN filled):
               No. of Physicians, Hospital Beds, GDP per Capita,
               No. of Hospitals, Avg. Employee Wage

Outputs:
  MIMIC-3/data/features_raw.csv       — per-admission raw lab extracts
  MIMIC-3/data/features_model.csv     — aligned to 43-feature model columns
  MIMIC-3/data/lab_item_map.csv       — ITEMID → feature name mapping used
"""
import gc, gzip, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
MIMIC_DIR = Path(r"e:\LungCancer\mimic-iii-clinical-database-1.4")
DATA_DIR  = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Load cohort from step 1 ───────────────────────────────────
print("=" * 62)
print("  Step 2: Feature Extraction")
print("=" * 62)

cohort_adm = pd.read_csv(DATA_DIR / "cohort_admissions.csv", low_memory=False)
cohort_sids = set(cohort_adm["SUBJECT_ID"].astype(int))
cohort_hids = set(cohort_adm["HADM_ID"].astype(int))
print(f"  Cohort: {len(cohort_sids):,} patients, {len(cohort_hids):,} admissions")

# ── Feature → keyword mapping against D_LABITEMS ─────────────
# Each entry: (feature_name, [search keywords (case-insensitive)])
FEATURE_KEYWORDS = {
    # Lipids
    "Apolipoprotein B":      ["apolipoprotein b", "apo b"],
    "LDL Cholesterol":       ["ldl", "low density lipoprotein"],
    "Total Cholesterol":     ["cholesterol, total", "total cholesterol"],
    "Triglycerides":         ["triglyceride"],
    "HDL Cholesterol":       ["hdl", "high density lipoprotein"],
    "Apolipoprotein AI":     ["apolipoprotein a", "apo ai", "apo a-i"],
    # Liver / protein
    "Cholinesterase":        ["cholinesterase", "pseudocholinesterase", "butyrylcholinesterase"],
    "Albumin":               ["albumin"],
    "Total Bile Acids":      ["bile acid", "bile acids"],
    "Prealbumin":            ["prealbumin", "transthyretin"],
    "ALT":                   ["alanine aminotransferase", "alt", "sgpt"],
    "AST":                   ["aspartate aminotransferase", "ast", "sgot"],
    "Direct Bilirubin":      ["bilirubin, direct", "direct bilirubin"],
    "Amylase":               ["amylase"],
    "Total Protein":         ["protein, total", "total protein"],
    # Kidney / electrolytes
    "Creatinine":            ["creatinine"],
    "Cystatin C":            ["cystatin"],
    "Uric Acid":             ["uric acid"],
    "Sodium":                ["sodium"],
    "Potassium":             ["potassium"],
    "Magnesium":             ["magnesium"],
    "Phosphorus":            ["phosphate", "phosphorus"],
    # Haematology / inflammation
    "D-Dimer":               ["d-dimer", "fibrin degradation", "fdp"],
    "Plateletcrit":          ["plateletcrit", "pct"],
    "Ferritin":              ["ferritin"],
    "Creatine Kinase":       ["creatine kinase", "cpk", "ck, total"],
    "α-HBDH":               ["hydroxybutyrate", "hbdh", "alpha-hbdh"],
    "LDH":                  ["lactate dehydrogenase", "ldh"],
    # Metabolic
    "Glucose":               ["glucose"],
    "HbA1c":                 ["hemoglobin a1c", "hba1c", "glycated hemoglobin", "glycohemoglobin"],
    # Tumour markers
    "CYFRA 21-1":            ["cyfra", "cytokeratin 19"],
    "Alpha-Fetoprotein":     ["alpha-fetoprotein", "afp", "alpha fetoprotein"],
    "SCC Antigen":           ["squamous cell carcinoma antigen", "scc"],
    "Adenosine Deaminase":   ["adenosine deaminase", "ada"],
    # Immunology
    "HBcAb":                 ["hepatitis b core antibody", "hbcab", "hbc ab"],
    # Globulin is derived: Total Protein − Albumin
}

# ── Match D_LABITEMS → feature name ──────────────────────────
print("\nBuilding lab item map from D_LABITEMS ...")
d_lab = pd.read_csv(MIMIC_DIR / "D_LABITEMS.csv.gz",
                    usecols=["ITEMID", "LABEL", "FLUID", "CATEGORY"])
d_lab["LABEL_lower"] = d_lab["LABEL"].str.lower().fillna("")

item_map = {}   # ITEMID → feature_name
for feat, keywords in FEATURE_KEYWORDS.items():
    mask = d_lab["LABEL_lower"].apply(
        lambda s: any(kw in s for kw in keywords)
    )
    # Prefer serum/blood category
    matches = d_lab[mask]
    blood = matches[matches["FLUID"].str.lower().isin(["blood", "serum"])]
    chosen = blood if len(blood) > 0 else matches
    for _, row in chosen.iterrows():
        iid = int(row["ITEMID"])
        if iid not in item_map:
            item_map[iid] = feat

item_df = pd.DataFrame(
    [(iid, feat) for iid, feat in item_map.items()],
    columns=["ITEMID", "Feature"]
)
item_df = item_df.merge(d_lab[["ITEMID", "LABEL", "FLUID"]], on="ITEMID", how="left")
item_df.to_csv(DATA_DIR / "lab_item_map.csv", index=False)
print(f"  Mapped {len(item_map):,} ITEMID entries → {len(FEATURE_KEYWORDS)} features")

target_itemids = set(item_map.keys())
print(f"  Target ITEMIDs: {sorted(target_itemids)[:20]} ...")

# ── Stream LABEVENTS, filter to cohort + target items ─────────
print("\nStreaming LABEVENTS (large file) ...")
CHUNKSIZE = 500_000
rows_kept = []
total_read = 0

with gzip.open(MIMIC_DIR / "LABEVENTS.csv.gz", "rt") as f:
    for chunk in pd.read_csv(
            f, chunksize=CHUNKSIZE,
            usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
            dtype={"SUBJECT_ID": "int32", "ITEMID": "int32"},
            low_memory=False):
        total_read += len(chunk)
        chunk = chunk.dropna(subset=["VALUENUM", "HADM_ID"])
        chunk["HADM_ID"] = chunk["HADM_ID"].astype("int32")
        filt = chunk[
            chunk["SUBJECT_ID"].isin(cohort_sids) &
            chunk["HADM_ID"].isin(cohort_hids) &
            chunk["ITEMID"].isin(target_itemids)
        ]
        if len(filt) > 0:
            rows_kept.append(filt)
        del chunk
        if total_read % 5_000_000 == 0:
            gc.collect()
            print(f"    ... {total_read:,} rows read, {sum(len(r) for r in rows_kept):,} kept")

print(f"  Total read: {total_read:,}  |  Kept: {sum(len(r) for r in rows_kept):,}")
lab_raw = pd.concat(rows_kept, ignore_index=True) if rows_kept else pd.DataFrame()
del rows_kept; gc.collect()

if len(lab_raw) == 0:
    print("  WARNING: No lab events found for cohort. Check HADM_ID overlap.")
    lab_raw = pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUENUM"])

# ── Map ITEMID → feature name ─────────────────────────────────
lab_raw["Feature"] = lab_raw["ITEMID"].map(item_map)
lab_raw = lab_raw.dropna(subset=["Feature"])

# ── Pivot: median value per (HADM_ID, Feature) ───────────────
print("  Pivoting lab values ...")
lab_pivot = (lab_raw.groupby(["HADM_ID", "Feature"])["VALUENUM"]
             .median()
             .unstack("Feature")
             .reset_index())
print(f"  Lab pivot: {lab_pivot.shape[0]:,} admissions × {lab_pivot.shape[1]-1} features")
lab_raw.to_csv(DATA_DIR / "features_raw.csv", index=False)

# ── Derive Globulin ───────────────────────────────────────────
if "Total Protein" in lab_pivot.columns and "Albumin" in lab_pivot.columns:
    lab_pivot["Globulin"] = lab_pivot["Total Protein"] - lab_pivot["Albumin"]

# ── Merge with cohort demographics ───────────────────────────
demo_cols = ["HADM_ID", "SUBJECT_ID", "Age", "Sex_M",
             "ADMITTIME", "DISCHTIME", "HOSPITAL_EXPIRE_FLAG",
             "Admission_has_cancer", "Admission_has_cardiac"]
cohort_sub = cohort_adm[demo_cols].copy()
cohort_sub["HADM_ID"] = cohort_sub["HADM_ID"].astype(int)

merged = cohort_sub.merge(lab_pivot, on="HADM_ID", how="left")
print(f"  Merged dataset: {merged.shape}")

# ── Align to the 43 model columns ────────────────────────────
MODEL_FEATURES = [
    "Age", "Sex_M",
    "Apolipoprotein B", "LDL Cholesterol", "Total Cholesterol",
    "Cholinesterase", "Albumin", "Total Bile Acids", "Prealbumin",
    "D-Dimer", "CYFRA 21-1", "Cystatin C", "Triglycerides",
    "Adenosine Deaminase", "Globulin", "Plateletcrit", "SCC Antigen",
    "Direct Bilirubin",
    # Socioeconomic (not in MIMIC → NaN)
    "No. of Physicians", "Hospital Beds", "GDP per Capita", "No. of Hospitals",
    "Avg. Employee Wage",
    "Glucose", "HDL Cholesterol", "Magnesium", "α-HBDH", "ALT",
    "HBcAb", "LDH", "Phosphorus", "AST", "Uric Acid",
    "Alpha-Fetoprotein", "Total Protein", "HbA1c", "Creatinine",
    "Creatine Kinase", "Potassium", "Sodium", "Apolipoprotein AI",
    "Ferritin", "Amylase",
]

for col in MODEL_FEATURES:
    if col not in merged.columns:
        merged[col] = np.nan   # unavailable features filled as NaN

features_model = merged[["HADM_ID", "SUBJECT_ID",
                          "ADMITTIME", "DISCHTIME",
                          "HOSPITAL_EXPIRE_FLAG",
                          "Admission_has_cancer",
                          "Admission_has_cardiac"] + MODEL_FEATURES].copy()

n_labs     = features_model[MODEL_FEATURES].notna().any(axis=1).sum()
completeness = features_model[MODEL_FEATURES].notna().mean() * 100

print("\n  Feature completeness (%):")
for feat, pct in completeness.sort_values(ascending=False).items():
    if pct > 0:
        print(f"    {feat:<30} {pct:5.1f}%")

features_model.to_csv(DATA_DIR / "features_model.csv", index=False)
print(f"\n✓ Saved features_model.csv ({len(features_model):,} admissions)")
print(f"  Admissions with ≥1 lab value: {n_labs:,}")
print("\nStep 2 complete.\n")
