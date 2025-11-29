# adxl345_pipeline
TinyNet Vibration Classification Pipeline (with RF baseline)

Reproducible pipeline for the hose-state vibration classification experiments.
Contains:

### run_all.py — TinyNet (and TinyNet-Plus) pipeline (training, stability, plots, ensembles).

### rf_compare_fixed.py — Random Forest baseline with handcrafted features and LaTeX-ready output.

-----------------------------------------
### Directory structure (example)

run_all.py
rf_compare_fixed.py
cache/               ← auto-generated caches (.npz) for TinyNet
results/             ← TinyNet outputs, checkpoints, JSON, plots
results/figures/     ← confusion matrices & seed plots
rf_results/          ← recommended RF output directory (optional)
dataset/
    wide_balanced.csv
    narrow_balanced.csv

-----------------------------------------
### Common environment requirements

Python 3.8+

PyTorch (CUDA optional) — required for run_all.py  
NumPy, Pandas, Scikit-Learn, Matplotlib — required for both pipelines  
SciPy — required for RF feature extraction (skew & kurtosis)

Install minimal requirements:
    pip install numpy pandas scikit-learn matplotlib scipy torch

-----------------------------------------
### TinyNet (paper model) — quick run
(reproduced here briefly)

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

Diagnostics produced:
- results/stability_results.json  
- results/stability_runs_summary.csv  
- confusion matrices in results/figures/  
- results/*_best_seed*.pt checkpoints  

-----------------------------------------
### TinyNet-Plus (bonus / experimental)

TinyNet-Plus is included as a **bonus experimental variant** (larger channels, heavier FC).  
It is under active development and **not required to reproduce the manuscript’s main results**.

Run TinyNet-Plus by passing `--model plus` to any run_all.py mode:

    python run_all.py train_once --model plus --seed 1001
    python run_all.py stability --model plus --n_repeats 5
    python run_all.py eval --model plus --ckpts results/combined_best_seed1001.pt --data_npz cache/combined.npz

-----------------------------------------
### Random Forest baseline (rf_compare_fixed.py)

This single-file script implements the RF baseline used for the comparison table. It:

- Builds identical windows (256 samples, stride 128) with per-file normalization  
- Extracts compact handcrafted features per window  
  (time-domain stats, band energies, dominant freq — 30 features total)  
- Runs RF for:
    - combined (in-domain)
    - wide → narrow transfer
    - narrow → wide transfer
- Repeats across seeds
- Saves robust JSON results
- Prints a **LaTeX-ready summary row** for direct table inclusion

-----------------------------------------
### Usage (basic)

    python rf_compare_fixed.py \
        --wide ./wide_balanced.csv \
        --narrow ./narrow_balanced.csv

### Options (typical)
--seeds 1000 1001 1002 1003 1004     (default: 1000–1004)  
--n-estimators 200                   (default 200)  
--out rf_baseline_results.json       (output JSON)  
--min-train-per-class                (default 5)  
--min-test-per-class                 (default 8)

### Example full run (5 seeds)

    python rf_compare_fixed.py \
        --wide ./wide_balanced.csv \
        --narrow ./narrow_balanced.csv \
        --out rf_baseline_results.json

-----------------------------------------
### Outputs and where to look

- rf_baseline_results.json — JSON with per-seed runs, confusion matrices, and summary
- Console prints:
    * per-seed performance
    * LaTeX-ready row, e.g.:

    Random Forest (handcrafted features) & 0.9944 $\pm$ 0.0032 & 0.9944 $\pm$ 0.0032 \\

-----------------------------------------
### Notes on features & splits

Feature vector per window (30D):
- 18 time-domain features: mean/std/skew/kurtosis/RMS/ptp per axis
- 9 spectral band energies (0–5, 5–20, 20–50 Hz)
- 3 dominant frequencies (one per axis)

Group-aware splitting:
- File-level grouping to prevent leakage (same as TinyNet)

RF split defaults:
- MIN_TEST_PER_CLASS = 8 (relaxed for smaller dataset)
- Adjust via arguments if needed

-----------------------------------------
### How to include RF results in your paper

1. Run rf_compare_fixed.py using the same dataset CSVs.  
2. Copy the LaTeX line printed in console **or** read the summary in rf_baseline_results.json.  
3. Use the combined mean ± std values directly in your Results table.  

-----------------------------------------
### Reproducibility checklist (expanded)

✔ Same windowing config for TinyNet & RF (WINDOW=256, STRIDE=128)  
✔ Per-file normalization in both pipelines  
✔ File-level grouping to avoid temporal leakage  
✔ Seeded splits & runs for stability (same SEED_BASE philosophy)  
✔ RF prints a LaTeX-ready summary row automatically  

-----------------------------------------
### Final notes to reviewers

- **TinyNet** — main model used in all paper results.  
- **TinyNet-Plus** — optional experimental architecture included for transparency; not used for conclusions.  
- **Random Forest baseline** — compact, deterministic handcrafted-feature baseline; easy to compare with TinyNet.

