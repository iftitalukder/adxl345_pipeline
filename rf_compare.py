#!/usr/bin/env python3
"""
rf_compare_fixed.py

Single-file Random Forest baseline pipeline for comparison table.
- Builds same 256-sample windows (stride 128) and per-file normalization.
- Extracts compact handcrafted features per window.
- Runs RF for: combined (in-domain), wide->narrow, narrow->wide.
- Repeats across seeds, saves rf_baseline_results.json (robustly serialized).
- Prints a LaTeX-ready summary line for inclusion in the Results table.

Usage:
    python rf_compare_fixed.py \
       --wide /media/khalid/sda1/vibration/dataset/wide_balanced.csv \
       --narrow /media/khalid/sda1/vibration/dataset/narrow_balanced.csv

Author: Khalid Hossen (adapted)
"""
import os
import sys
import json
import argparse
import math
import random
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# ----------------- Defaults / Config -----------------
WINDOW = 256
STRIDE = 128
FS = 100
DEFAULT_SEEDS = [1000, 1001, 1002, 1003, 1004]
DEFAULT_N_EST = 200
DEFAULT_OUT = "rf_baseline_results.json"
MIN_TRAIN_PER_CLASS = 5
MIN_TEST_PER_CLASS = 8   # relaxed for smaller datasets
MAX_SPLIT_TRIES = 500
# ----------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Random Forest baseline (fixed, robust).")
    p.add_argument("--wide", required=True, help="Path to wide_balanced.csv")
    p.add_argument("--narrow", required=True, help="Path to narrow_balanced.csv")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds to run")
    p.add_argument("--n-estimators", type=int, default=DEFAULT_N_EST, help="RF n_estimators")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output JSON path")
    p.add_argument("--min-train-per-class", type=int, default=MIN_TRAIN_PER_CLASS)
    p.add_argument("--min-test-per-class", type=int, default=MIN_TEST_PER_CLASS)
    return p.parse_args()

# ---------------- utils ----------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)

def normalize_array_per_file(arr):
    arr = arr.astype(float)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    stds_safe = np.where(stds == 0, 1.0, stds)
    return (arr - means.reshape(1,3)) / stds_safe.reshape(1,3)

def windows_from_signal(sig, window=WINDOW, stride=STRIDE):
    N = sig.shape[0]
    res = []
    idx = 0
    while idx + window <= N:
        res.append(sig[idx:idx+window])
        idx += stride
    return np.stack(res, axis=0) if res else np.zeros((0, window, 3))

def window_label_majority(labels, window=WINDOW, stride=STRIDE):
    labs = []
    idx = 0
    N = len(labels)
    while idx + window <= N:
        block = labels[idx:idx+window]
        vals, counts = np.unique(block, return_counts=True)
        labs.append(int(vals[np.argmax(counts)]))
        idx += stride
    return np.array(labs, dtype=int)

def build_raw_windows(csv_paths, per_file_norm=True):
    """
    Returns:
      raw_windows: np.array shape (N_windows, WINDOW, 3)
      labels: np.array shape (N_windows,)
      groups: np.array shape (N_windows,)   # file-level group id
    """
    raw_windows = []
    labels = []
    groups = []
    for file_idx, path in enumerate(csv_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if not {'x','y','z','class'}.issubset(set(df.columns)):
            raise ValueError(f"CSV {path} missing columns x,y,z,class")
        arr = df[['x','y','z']].astype(float).values
        labs = df['class'].astype(int).values
        if per_file_norm:
            arr = normalize_array_per_file(arr)
        W = windows_from_signal(arr, window=WINDOW, stride=STRIDE)
        if len(W) == 0:
            continue
        ywin = window_label_majority(labs, window=WINDOW, stride=STRIDE)
        for wi in range(len(W)):
            raw_windows.append(W[wi])
            labels.append(int(ywin[wi]))
            groups.append(int(file_idx))
    if not raw_windows:
        return np.zeros((0,WINDOW,3)), np.zeros((0,),dtype=int), np.zeros((0,),dtype=int)
    return np.stack(raw_windows, axis=0), np.array(labels, dtype=int), np.array(groups, dtype=int)

# ---------------- features ----------------
def extract_handcrafted_features(windows, fs=FS):
    """
    windows: (N, L, 3)
    returns: (N, F) float features
    Feature set:
      per-axis: mean, std, skew, kurtosis, RMS, peak-to-peak  => 6*3 = 18
      per-axis band energies (0-5, 5-20, 20-50 Hz) => 3*3 = 9
      per-axis dominant frequency => 3
    total F = 30
    """
    N, L, C = windows.shape
    freqs = np.fft.rfftfreq(L, d=1.0/fs)
    band_limits = [(0,5),(5,20),(20,50)]
    feats = np.zeros((N, 30), dtype=float)
    for i in range(N):
        v = []
        w = windows[i]  # (L,3)
        # time-domain stats
        for c in range(C):
            x = w[:,c].astype(float)
            v.append(np.mean(x))
            v.append(np.std(x))
            # scipy.stats skew/kurtosis are robust; handle constant signal
            try:
                v.append(float(skew(x)))
            except Exception:
                v.append(0.0)
            try:
                v.append(float(kurtosis(x)))
            except Exception:
                v.append(0.0)
            v.append(float(np.sqrt(np.mean(x**2))))   # RMS
            v.append(float(np.ptp(x)))                # peak-to-peak
        # PSD
        X = np.fft.rfft(w, axis=0)
        ps = (np.abs(X)**2)
        # band energies
        for (f0,f1) in band_limits:
            idx = np.where((freqs >= f0) & (freqs < f1))[0]
            if len(idx) == 0:
                v.extend([0.0,0.0,0.0])
            else:
                band_power = ps[idx,:].sum(axis=0)
                v.extend([float(band_power[0]), float(band_power[1]), float(band_power[2])])
        # dominant freq per axis
        dom_idx = np.argmax(ps, axis=0)
        dom_freqs = freqs[dom_idx]
        v.extend([float(dom_freqs[0]), float(dom_freqs[1]), float(dom_freqs[2])])
        feats[i,:] = np.array(v, dtype=float)
    return feats

# ---------------- group-aware split ----------------
def group_stratified_split(groups, y, test_size=0.20, val_fraction=0.111111, seed=0,
                           min_train_per_class=MIN_TRAIN_PER_CLASS, min_test_per_class=MIN_TEST_PER_CLASS,
                           max_tries=MAX_SPLIT_TRIES):
    """
    Find group-aware split (train/val/test) with constraints.
    Returns (train_idx, val_idx, test_idx) as arrays of indices into the dataset (window-level).
    """
    set_seed(seed)
    all_idx = np.arange(len(y))
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        # fallback to simple stratified split by label
        tr, te = train_test_split(all_idx, test_size=test_size, stratify=y, random_state=seed)
        trf, val = train_test_split(tr, test_size=val_fraction, stratify=y[tr], random_state=seed)
        return trf, val, te

    last_valid = None
    for attempt in range(1, max_tries+1):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
        train_idx_mask, test_idx_mask = next(gss.split(np.zeros(len(groups)), y, groups=groups))
        train_idx = train_idx_mask
        test_idx = test_idx_mask
        # split train -> train/val
        if len(np.unique(groups[train_idx])) >= 2:
            gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + attempt + 7)
            t2, v_in = next(gss2.split(np.zeros(len(train_idx)), y[train_idx], groups=groups[train_idx]))
            train_idx_final = train_idx[t2]
            val_idx_final = train_idx[v_in]
        else:
            trf, val = train_test_split(train_idx, test_size=val_fraction, stratify=y[train_idx], random_state=seed+attempt)
            train_idx_final, val_idx_final = trf, val

        cnt_train = Counter(y[train_idx_final])
        cnt_test = Counter(y[test_idx_mask])
        ok_train = all([cnt_train.get(c,0) >= min_train_per_class for c in np.unique(y)])
        ok_test = all([cnt_test.get(c,0) >= min_test_per_class for c in np.unique(y)])
        if ok_train and ok_test:
            return train_idx_final, val_idx_final, test_idx_mask
        last_valid = (train_idx_final, val_idx_final, test_idx_mask)
    # warning fallback
    print(f"WARNING: group_stratified_split reached max_tries={max_tries}; returning last candidate (may violate counts).")
    return last_valid

# ---------------- json-safe conversion ----------------
def make_json_serializable(obj):
    """
    Recursively convert numpy types and non-JSON dict keys to JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            # convert keys: numpy ints -> int, other -> str
            if isinstance(k, (np.integer,)):
                k2 = int(k)
            elif isinstance(k, (int, float, str, bool)) or k is None:
                k2 = k
            else:
                k2 = str(k)
            new[k2] = make_json_serializable(v)
        return new
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

# ---------------- RF training / eval ----------------
def rf_train_eval_once(feats, labels, groups, seed, n_estimators=DEFAULT_N_EST):
    tr_idx, val_idx, test_idx = group_stratified_split(groups, labels, seed=seed)
    X_tr, y_tr = feats[tr_idx], labels[tr_idx]
    X_te, y_te = feats[test_idx], labels[test_idx]
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    f1m = f1_score(y_te, preds, average='macro')
    rep = classification_report(y_te, preds, digits=4, zero_division=0)
    cm = confusion_matrix(y_te, preds).tolist()
    support = {
        'train': {int(k): int(v) for k, v in dict(Counter(y_tr)).items()},
        'val':   {int(k): int(v) for k, v in dict(Counter(labels[val_idx])).items()},
        'test':  {int(k): int(v) for k, v in dict(Counter(y_te)).items()}
    }
    return {'acc': float(acc), 'macro_f1': float(f1m), 'classification_report': rep, 'confusion_matrix': cm, 'class_support': support}

def rf_train_on_source_and_test_target(src_feats, src_labels, src_groups, tgt_feats, tgt_labels, seed, n_estimators=DEFAULT_N_EST):
    tr_idx, val_idx, _ = group_stratified_split(src_groups, src_labels, seed=seed)
    X_tr, y_tr = src_feats[tr_idx], src_labels[tr_idx]
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    preds = clf.predict(tgt_feats)
    acc = accuracy_score(tgt_labels, preds)
    f1m = f1_score(tgt_labels, preds, average='macro')
    rep = classification_report(tgt_labels, preds, digits=4, zero_division=0)
    cm = confusion_matrix(tgt_labels, preds).tolist()
    support = {
        'train': {int(k): int(v) for k, v in dict(Counter(y_tr)).items()},
        'val':   {int(k): int(v) for k, v in dict(Counter(src_labels[val_idx])).items()},
        'target': int(len(tgt_labels))
    }
    return {'acc': float(acc), 'macro_f1': float(f1m), 'classification_report': rep, 'confusion_matrix': cm, 'class_support': support}

# ---------------- main runner ----------------
def run(args):
    print("Building windows (combined)...")
    raw_all, labels_all, groups_all = build_raw_windows([args.wide, args.narrow])
    print(f"Combined: windows={raw_all.shape[0]}")

    print("Building windows (wide)...")
    raw_w, labels_w, groups_w = build_raw_windows([args.wide])
    print(f"Wide: windows={raw_w.shape[0]}")

    print("Building windows (narrow)...")
    raw_n, labels_n, groups_n = build_raw_windows([args.narrow])
    print(f"Narrow: windows={raw_n.shape[0]}")

    # feature extraction
    print("Extracting features (combined)...")
    feats_all = extract_handcrafted_features(raw_all)
    feats_w = extract_handcrafted_features(raw_w)
    feats_n = extract_handcrafted_features(raw_n)

    results = {'runs': []}
    for seed in args.seeds:
        set_seed(seed)
        print(f"\n=== Seed {seed} ===")
        comb = rf_train_eval_once(feats_all, labels_all, groups_all, seed=seed, n_estimators=args.n_estimators)
        print(f"[RF combined] acc {comb['acc']:.6f} macroF1 {comb['macro_f1']:.6f}")
        w2n = rf_train_on_source_and_test_target(feats_w, labels_w, groups_w, feats_n, labels_n, seed=seed, n_estimators=args.n_estimators)
        print(f"[RF w->n     ] acc {w2n['acc']:.6f} macroF1 {w2n['macro_f1']:.6f}")
        n2w = rf_train_on_source_and_test_target(feats_n, labels_n, groups_n, feats_w, labels_w, seed=seed, n_estimators=args.n_estimators)
        print(f"[RF n->w     ] acc {n2w['acc']:.6f} macroF1 {n2w['macro_f1']:.6f}")
        results['runs'].append({'seed': int(seed), 'combined': comb, 'w2n': w2n, 'n2w': n2w})

    # aggregate combined stats for table
    comb_accs = [r['combined']['acc'] for r in results['runs']]
    comb_f1s  = [r['combined']['macro_f1'] for r in results['runs']]
    summary = {
        'combined_acc_mean': float(np.mean(comb_accs)),
        'combined_acc_std': float(np.std(comb_accs, ddof=0)),
        'combined_f1_mean': float(np.mean(comb_f1s)),
        'combined_f1_std': float(np.std(comb_f1s, ddof=0)),
        'per_run_count': len(results['runs'])
    }
    out = {'results': results, 'summary': summary}

    # JSON-safe dump
    out_serial = make_json_serializable(out)
    with open(args.out, 'w') as f:
        json.dump(out_serial, f, indent=2)
    print(f"\nSaved RF baseline JSON to: {args.out}")

    # print LaTeX-ready summary
    print("\n=== Summary (combined dataset) ===")
    print("Random Forest (handcrafted features): "
          f"Acc = {summary['combined_acc_mean']:.4f} ± {summary['combined_acc_std']:.4f}, "
          f"Macro-F1 = {summary['combined_f1_mean']:.4f} ± {summary['combined_f1_std']:.4f}")
    print("\nLaTeX row (copy-paste):")
    print(f"Random Forest (handcrafted features) & {summary['combined_acc_mean']:.4f} $\\pm$ {summary['combined_acc_std']:.4f} & {summary['combined_f1_mean']:.4f} $\\pm$ {summary['combined_f1_std']:.4f} \\\\")

if __name__ == "__main__":
    args = parse_args()
    run(args)
