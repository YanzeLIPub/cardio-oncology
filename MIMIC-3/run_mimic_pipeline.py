#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIMIC-III Cardio-Oncology MACE Prediction Pipeline
===================================================
Master script — runs all 4 steps in sequence:
  1. Identify cardio-oncology patients
  2. Extract lab features
  3. Define MACE outcomes
  4. Train / apply ensemble model & evaluate

Usage:
  python MIMIC-3/run_mimic_pipeline.py [--steps 1,2,3,4]

All outputs go to MIMIC-3/data/ and MIMIC-3/results/.
"""
import argparse, time, subprocess, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = {
    1: ("01_identify_patients.py",  "Identify cardio-oncology patients"),
    2: ("02_extract_features.py",   "Extract lab features"),
    3: ("03_define_mace.py",        "Define MACE outcomes"),
    4: ("04_predict_mace.py",       "MACE prediction & evaluation"),
}

def run_step(num, script_name, description):
    print(f"\n{'='*66}")
    print(f"  STEP {num}: {description}")
    print(f"{'='*66}")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(SCRIPT_DIR / script_name)],
        cwd=str(SCRIPT_DIR.parent),  # workspace root for imports
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ Step {num} FAILED (exit code {result.returncode}) — {elapsed:.1f}s")
        return False
    print(f"\n  ✓ Step {num} completed in {elapsed:.1f}s")
    return True

def main():
    parser = argparse.ArgumentParser(description="MIMIC-III MACE pipeline")
    parser.add_argument("--steps", default="1,2,3,4",
                        help="Comma-separated step numbers to run (default: all)")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="Skip MIMIC-III data directory check")
    args = parser.parse_args()

    steps_to_run = [int(s.strip()) for s in args.steps.split(",")]

    # Check MIMIC data directory
    if not args.skip_data_check:
        mimic_dir = Path(r"e:\LungCancer\mimic-iii-clinical-database-1.4")
        if not mimic_dir.exists():
            print(f"ERROR: MIMIC-III directory not found: {mimic_dir}")
            print("  Set correct path in the step scripts or use --skip-data-check")
            sys.exit(1)
        print(f"✓ MIMIC-III data directory found: {mimic_dir}")

    print(f"\n{'='*66}")
    print("  MIMIC-III Cardio-Oncology MACE Prediction Pipeline")
    print(f"  Running steps: {steps_to_run}")
    print(f"{'='*66}")

    pipeline_start = time.time()
    failed = []
    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"WARNING: Unknown step {step_num} — skipping")
            continue
        script, desc = STEPS[step_num]
        ok = run_step(step_num, script, desc)
        if not ok:
            failed.append(step_num)
            print(f"  Stopping pipeline at step {step_num}.")
            break

    total_time = time.time() - pipeline_start
    print(f"\n{'='*66}")
    if not failed:
        print(f"  ✓ Pipeline complete in {total_time:.1f}s")
        print(f"  Data:    MIMIC-3/data/")
        print(f"  Results: MIMIC-3/results/")
        print(f"  Figures: MIMIC-3/results/figures/")
    else:
        print(f"  ✗ Pipeline failed at step(s): {failed}")
    print(f"{'='*66}")

if __name__ == "__main__":
    main()
