# adxl345_pipeline
TinyNet Vibration Classification Pipeline (with RF baseline)

Reproducible pipeline for the hose-state vibration classification experiments.
Contains:

### run_all.py — TinyNet (and TinyNet-Plus) pipeline (training, stability, plots, ensembles).

### rf_compare_fixed.py — Random Forest baseline with handcrafted features and LaTeX-ready output.

### Directory structure (example)
run_all.py
rf_compare_fixed.py
cache/               ← auto-generated caches (.npz) for TinyNet
results/             ← TinyNet outputs, checkpoints, JSON, plots
results/figures/     ← confusion matrices & seed plots
rf_results/          ← [recommended] RF outputs (or current working dir)
dataset/
    wide_balanced.csv
    narrow_balanced.csv

### Common environment requirements

Python 3.8+

PyTorch (CUDA optional) — for run_all.py

NumPy, Pandas, Scikit-Learn, Matplotlib — for both scripts

SciPy (for RF feature extraction: skew & kurtosis)

Install minimal requirements:

pip install numpy pandas scikit-learn matplotlib scipy torch

### TinyNet (paper model) — quick run

(See full details in the repo top, reproduced here briefly.)

### Prepare caches:

python run_all.py prepare --force-rebuild


### Single run (seeded):

python run_all.py train_once --seed 1001


### Stability runs (paper):

python run_all.py stability --model tiny --n_repeats 5


### Evaluate checkpoint:

python run_all.py eval --ckpts results/combined_best_seed1001.pt --data_npz cache/combined.npz


### Select top-3 and run ensemble:

python run_all.py select_top3
python run_all.py ensemble


Diagnostics produced: results/stability_results.json, results/stability_runs_summary.csv, confusion matrices in results/figures/, results/*_best_seed*.pt checkpoints.

### TinyNet-Plus (bonus / experimental)

TinyNet-Plus is included as a bonus experimental variant (larger channels, heavier FC). It is under active development and not required to reproduce the manuscript’s main results.

### Run TinyNet-Plus by passing --model plus to any run_all.py mode:

python run_all.py train_once --model plus --seed 1001
python run_all.py stability --model plus --n_repeats 5
python run_all.py eval --model plus --ckpts results/combined_best_seed1001.pt --data_npz cache/combined.npz

### Random Forest baseline (rf_compare_fixed.py)

This single-file script implements the RF baseline used for the comparison table. It:

Builds identical windows (256 samples, stride 128) and per-file normalization;

Extracts compact handcrafted features per window (time-domain stats, band energies, dominant frequency — 30 features total);

Runs RF for in-domain (combined), wide→narrow, narrow→wide;

Repeats across multiple seeds and saves robust JSON results;

Prints a LaTeX-ready summary row for easy table inclusion.

Usage

Basic invocation:

python rf_compare_fixed.py \
  --wide /media/khalid/sda1/vibration/dataset/wide_balanced.csv \
  --narrow /media/khalid/sda1/vibration/dataset/narrow_balanced.csv


Options (typical):

--seeds 1000 1001 1002 1003 1004 — seeds to run (default: 1000–1004)

--n-estimators 200 — RF n_estimators (default 200)

--out rf_baseline_results.json — output JSON file

--min-train-per-class, --min-test-per-class — split constraints

Example full run (5 seeds; default settings)
python rf_compare_fixed.py \
  --wide /media/khalid/sda1/vibration/dataset/wide_balanced.csv \
  --narrow /media/khalid/sda1/vibration/dataset/narrow_balanced.csv \
  --out rf_baseline_results.json

Outputs and where to look

rf_baseline_results.json (default) — JSON with per-seed runs, confusion matrices, classification reports and summary.

Console prints per-seed performance and a LaTeX-ready row for the Results table, e.g.:

Random Forest (handcrafted features) & 0.9944 $\pm$ 0.0032 & 0.9944 $\pm$ 0.0032 \\


(You can copy-paste that row directly into your LaTeX table.)

Notes on features & splits

Feature vector per window: 30 features — per-axis mean/std/skew/kurtosis/RMS/ptp (18), per-axis band energies in bands (0–5, 5–20, 20–50 Hz) (9), plus per-axis dominant frequency (3).

Group-aware splitting: file-level groups to prevent leakage — same constraint behaviour as TinyNet pipeline.

The RF split defaults are slightly more relaxed for the test set (MIN_TEST_PER_CLASS=8) to accommodate smaller classes; adjust via arguments if needed.

 ### How to include RF results in your paper

Run rf_compare_fixed.py with the same dataset CSVs you used for TinyNet.

Open the printed LaTeX line from the script output or read the rf_baseline_results.json summary.

Paste the LaTeX row into the Results table. The JSON includes per-seed runs for further reporting or plotting.

### Reproducibility checklist (expanded)

 Same windowing config for TinyNet & RF (WINDOW=256, STRIDE=128)

 Per-file normalization for both pipelines

 File-level grouping to avoid temporal leakage

 Seeded splits and runs for stability (same SEED_BASE default design)

 RF prints a LaTeX-ready summary row automatically

### Final notes to reviewers

TinyNet — main model used in paper results.

TinyNet-Plus — included as a bonus experimental variant, under active development for follow-up work; not used for the core claims.

Random Forest baseline — compact, deterministic baseline using handcrafted features; prints easy-to-copy LaTeX row.
