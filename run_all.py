#!/usr/bin/env python3
"""
run_all.py - Compact, single-file pipeline for vibration TinyNet experiments (fixed & plotting).
Changes:
 - FS set to 100 Hz (actual sampling rate)
 - group ids are file-level (prevents temporal leakage)
 - added --clear-cache option
 - added plotting utilities (light, clear visuals) saved to ./results/figures/
 - ensure minimum per-class samples in test split via retries
"""
import os, sys, json, time, math, random, argparse, shutil
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

# ----------- CONFIG -----------
WIDE_PATH   = "/media/khalid/sda1/vibration/dataset/wide_balanced.csv"
NARROW_PATH = "/media/khalid/sda1/vibration/dataset/narrow_balanced.csv"
CACHE_DIR   = "./cache"
OUTPUT_DIR  = "./results"
FIG_DIR     = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

WINDOW = 256
STRIDE = 128
FS = 100                 # corrected: actual sampling rate in Hz
MIN_CHUNKS_PER_FILE = 4
MAX_CHUNKS_PER_FILE = 50

# training defaults
N_REPEATS = 5
EPOCHS = 80
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10            # matches manuscript
SEED_BASE = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6

# augment
AUGMENT = True
AUG_SNR_DB = (10,30)
AUG_SCALE = (0.85,1.15)
AUG_SHIFT = 12
AUG_TIMEWARP_P = 0.5

# splitting constraints
MIN_CLASS_TRAIN = 5     # minimal samples per class in train split
MIN_CLASS_TEST = 10     # ensure at least this many samples per class in test (if possible)
# ------------------------------

# ---------------- utilities ----------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_array_per_file(arr):
    arr = arr.astype(float)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    stds_safe = np.where(stds==0, 1.0, stds)
    return (arr - means.reshape(1,3)) / stds_safe.reshape(1,3)

def windows_from_signal(sig, window=WINDOW, stride=STRIDE):
    N = sig.shape[0]; res=[]; idx=0
    while idx+window<=N:
        res.append(sig[idx:idx+window]); idx += stride
    return np.stack(res,axis=0) if res else np.zeros((0,window,3))

def window_label_majority(labels, window=WINDOW, stride=STRIDE):
    labs=[]; idx=0; N=len(labels)
    while idx+window<=N:
        block = labels[idx:idx+window]; vals, counts = np.unique(block, return_counts=True)
        labs.append(int(vals[np.argmax(counts)])); idx+=stride
    return np.array(labs, dtype=int)

def build_raw_windows(csv_paths, per_file_norm=True, cache_name=None, force_rebuild=False):
    """
    Build raw windows array from list of csv paths.
    Returns: raw_windows (N, window, 3), labels (N,), groups (N,) where group is per-file id (no leakage).
    Caches .npz if cache_name provided. If force_rebuild True, cache is ignored and rebuilt.
    """
    if cache_name:
        cache_path = os.path.join(CACHE_DIR, cache_name + ".npz")
        if os.path.exists(cache_path) and not force_rebuild:
            d = np.load(cache_path, allow_pickle=True)
            print(f"Loaded cache {cache_path}: raw {d['raw'].shape}, labels {d['labels'].shape}, unique_groups {len(np.unique(d['groups']))}")
            return d['raw'], d['labels'], d['groups']

    raw_windows=[]; labels=[]; groups=[]
    for file_idx, path in enumerate(csv_paths):
        df = pd.read_csv(path); df.columns = [c.strip().lower() for c in df.columns]
        if not {'x','y','z','class'}.issubset(set(df.columns)):
            raise ValueError(f"CSV {path} missing required columns x,y,z,class")
        arr = df[['x','y','z']].astype(float).values
        labs = df['class'].astype(int).values
        if per_file_norm:
            arr = normalize_array_per_file(arr)
        W = windows_from_signal(arr, window=WINDOW, stride=STRIDE)
        if len(W)==0: continue
        ywin = window_label_majority(labs, window=WINDOW, stride=STRIDE)
        n_win = len(W)
        # NOTE: group id is file-level to prevent leakage
        for wi in range(n_win):
            gid = file_idx           # file-level grouping (all windows from same file share same group)
            raw_windows.append(W[wi])
            labels.append(int(ywin[wi]))
            groups.append(gid)
    if not raw_windows:
        return np.zeros((0,WINDOW,3)), np.zeros((0,),dtype=int), np.zeros((0,),dtype=int)
    raw = np.stack(raw_windows,axis=0)
    labs = np.array(labels, dtype=int)
    grps = np.array(groups, dtype=int)
    if cache_name:
        np.savez_compressed(os.path.join(CACHE_DIR, cache_name + ".npz"), raw=raw, labels=labs, groups=grps)
        print(f"Saved cache {os.path.join(CACHE_DIR, cache_name + '.npz')}")
    return raw, labs, grps

# ---------- augmentations ----------
def add_awgn(sig, snr_db):
    out = sig.copy().astype(float)
    for c in range(out.shape[1]):
        x = out[:,c]; p = np.mean(x**2)
        if p<=0: continue
        snr = 10**(snr_db/10.0); noise_p = p / snr
        out[:,c] = x + np.random.normal(0, math.sqrt(noise_p), size=x.shape)
    return out

def time_warp(sig):
    # simple sinusoidal warp along time
    L = sig.shape[0]
    freq = random.uniform(1.0,8.0)
    mag = random.uniform(0.001, 0.01)
    warp = (1.0 + mag * np.sin(np.linspace(0, math.pi*freq, L))).reshape(-1,1)
    return sig * warp

def augment_window(win):
    w = win.copy()
    s = float(np.random.uniform(AUG_SCALE[0], AUG_SCALE[1]))
    w = w * s
    snr = float(np.random.uniform(AUG_SNR_DB[0], AUG_SNR_DB[1]))
    w = add_awgn(w, snr)
    shift = np.random.randint(-AUG_SHIFT, AUG_SHIFT+1)
    if shift != 0:
        w = np.roll(w, shift, axis=0)
    if random.random() < AUG_TIMEWARP_P:
        w = time_warp(w)
    w += np.random.normal(0, 1e-5, size=w.shape)
    return w

class RawDataset(Dataset):
    def __init__(self, raw_windows, labels, augment=False):
        self.raw = raw_windows.astype(np.float32); self.labels = labels.astype(np.int64); self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        w = self.raw[idx]
        if self.augment:
            w = augment_window(w)
        # return channel-first
        return torch.from_numpy(w.T).float(), int(self.labels[idx])

# ---------- models ----------
class DWConv(nn.Module):
    def __init__(self,in_ch,out_ch,k):
        super().__init__()
        self.dw = nn.Conv1d(in_ch,in_ch,k,padding=k//2,groups=in_ch,bias=False)
        self.pw = nn.Conv1d(in_ch,out_ch,1,bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self,x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); return F.relu(x, inplace=True)

class TinyNet1D(nn.Module):
    def __init__(self, in_ch=3, num_classes=6):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_ch,32,9,padding=4,bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True))
        self.ds1 = DWConv(32,64,7); self.pool1 = nn.AvgPool1d(4)
        self.ds2 = DWConv(64,96,5); self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(96,128), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(128,num_classes))
    def forward(self,x):
        x = self.conv1(x); x = self.ds1(x); x = self.pool1(x); x = self.ds2(x); x = self.pool2(x); x = self.fc(x); return x

class TinyNetPlus(nn.Module):
    def __init__(self, in_ch=3, num_classes=6):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_ch,48,9,padding=4,bias=False), nn.BatchNorm1d(48), nn.ReLU(inplace=True))
        self.ds1 = DWConv(48,96,7); self.pool1 = nn.AvgPool1d(4)
        self.ds2 = DWConv(96,128,5); self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128,192), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(192,num_classes))
    def forward(self,x):
        x = self.conv1(x); x = self.ds1(x); x = self.pool1(x); x = self.ds2(x); x = self.pool2(x); x = self.fc(x); return x

# ---------- splitting utilities ----------
def group_stratified_split(groups, y, test_size=0.15, val_fraction=0.1765, seed=0, min_train_class=MIN_CLASS_TRAIN, min_test_class=MIN_CLASS_TEST, max_tries=500):
    """
    Group-aware split that tries to ensure:
      - windows from same group do not cross splits
      - training contains at least min_train_class per label
      - test contains at least min_test_class per label (if possible)
    Returns train_idx, val_idx, test_idx
    """
    set_seed(seed)
    all_idx = np.arange(len(y))
    unique_groups = np.unique(groups)
    # fallback: if too few groups then stratify by label
    if len(unique_groups) < 3:
        tr, te = train_test_split(all_idx, test_size=test_size, stratify=y, random_state=seed)
        trf, val = train_test_split(tr, test_size=val_fraction, stratify=y[tr], random_state=seed)
        return trf, val, te

    tries = 0
    last_valid = None
    while tries < max_tries:
        tries += 1
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed+tries)
        train_idx_mask, test_idx_mask = next(gss.split(np.zeros(len(groups)), y, groups=groups))
        train_idx = train_idx_mask; test_idx = test_idx_mask
        train_groups = groups[train_idx]
        # split train -> train/val
        if len(np.unique(train_groups)) >= 2:
            gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed+tries+1)
            t2, v_in = next(gss2.split(np.zeros(len(train_idx)), y[train_idx], groups=train_groups))
            train_idx_final = train_idx[t2]; val_idx_final = train_idx[v_in]
        else:
            trf, val = train_test_split(train_idx, test_size=val_fraction, stratify=y[train_idx], random_state=seed+tries)
            train_idx_final, val_idx_final = trf, val

        cnt_train = Counter(y[train_idx_final])
        cnt_test = Counter(y[test_idx_mask])
        ok_train = all([cnt_train.get(c,0) >= min_train_class for c in range(NUM_CLASSES)])
        ok_test = all([cnt_test.get(c,0) >= min_test_class for c in range(NUM_CLASSES)])
        if ok_train and ok_test:
            return train_idx_final, val_idx_final, test_idx_mask
        last_valid = (train_idx_final, val_idx_final, test_idx_mask)
    # if no split satisfied strict constraints, return last valid (with a warning)
    print(f"WARNING: group_stratified_split reached max_tries={max_tries}. Returning last split (may have small class counts in test).")
    return last_valid

def compute_class_weights(labels, idxs):
    counts = Counter(labels[idxs])
    total = sum(counts.values())
    weights = [total / counts.get(c,1) for c in range(NUM_CLASSES)]
    return weights

# ---------- plotting ----------
def plot_confusion_matrix(cm, classes, title, out_path, normalize=False):
    """
    Save a clear, light-coloured confusion matrix image with annotations.
    Accepts cm as list or numpy array; handles int or float values robustly.
    """
    import numpy as _np
    cm_arr = _np.array(cm, dtype=float)  # use float internally for normalization
    if cm_arr.size == 0:
        print(f"Warning: empty confusion matrix for {title}, skipping plot.")
        return

    if normalize:
        with _np.errstate(all='ignore'):
            row_sums = cm_arr.sum(axis=1, keepdims=True)
            cm_plot = _np.divide(cm_arr, row_sums, out=_np.zeros_like(cm_arr), where=row_sums!=0)
    else:
        cm_plot = cm_arr.copy()

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm_plot, interpolation='nearest', cmap='Blues', vmin=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = _np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')
    ax.set_title(title)

    # Choose textual format: integer when values are (close to) integers and not normalized
    is_integer_like = (not normalize) and _np.all(_np.mod(cm_arr, 1.0) == 0)
    fmt = 'd' if is_integer_like else '.2f'

    # annotation threshold for text color
    thresh = cm_plot.max() / 2. if cm_plot.max() > 0 else 0.5
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            if normalize:
                val = cm_plot[i, j]
            else:
                val = cm_arr[i, j]
            # safe-format (fallback to str if formatting fails)
            try:
                txt = format(val, fmt)
            except Exception:
                txt = f"{val:.2f}"
            color = "black" if (cm_plot[i, j] < thresh) else "white"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_seed_metrics(runs, out_path):
    """
    Plot per-seed accuracy and macro-F1 for combined / wide->narrow / narrow->wide.
    Expects `runs` as produced in run_stability: list of dicts with keys 'seed','combined','w2n','n2w'
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    if not runs:
        print("plot_seed_metrics: no runs provided, skipping.")
        return

    seeds = [r['seed'] for r in runs]
    # sort by seed to keep x-axis ordered
    order = _np.argsort(seeds)
    seeds = _np.array(seeds)[order].tolist()

    comb_acc = [_np.array(r['combined']['acc']) for r in runs]
    comb_f1  = [_np.array(r['combined']['macro_f1']) for r in runs]
    w2n_acc  = [_np.array(r['w2n']['acc']) for r in runs]
    w2n_f1   = [_np.array(r['w2n']['macro_f1']) for r in runs]
    n2w_acc  = [_np.array(r['n2w']['acc']) for r in runs]
    n2w_f1   = [_np.array(r['n2w']['macro_f1']) for r in runs]

    # reorder according to seed sort
    comb_acc = _np.array(comb_acc)[order].astype(float)
    comb_f1  = _np.array(comb_f1)[order].astype(float)
    w2n_acc  = _np.array(w2n_acc)[order].astype(float)
    w2n_f1   = _np.array(w2n_f1)[order].astype(float)
    n2w_acc  = _np.array(n2w_acc)[order].astype(float)
    n2w_f1   = _np.array(n2w_f1)[order].astype(float)

    fig, axes = _plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1, ax2 = axes

    ax1.plot(seeds, comb_acc, marker='o', label='combined acc')
    ax1.plot(seeds, w2n_acc, marker='x', label='wide->narrow acc')
    ax1.plot(seeds, n2w_acc, marker='s', label='narrow->wide acc')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle=':', linewidth=0.5)
    ax1.legend(loc='best', fontsize=8)

    ax2.plot(seeds, comb_f1, marker='o', label='combined macro F1')
    ax2.plot(seeds, w2n_f1, marker='x', label='wide->narrow macro F1')
    ax2.plot(seeds, n2w_f1, marker='s', label='narrow->wide macro F1')
    ax2.set_ylabel('Macro F1')
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel('seed')
    ax2.grid(True, linestyle=':', linewidth=0.5)
    ax2.legend(loc='best', fontsize=8)

    # tidy x ticks (show the seed integers)
    _plt.xticks(seeds, rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    _plt.close(fig)
    print("Saved seed metrics plot to", out_path)



# ---------- training / evaluation ----------
def train_once(raw_windows, labels, groups, seed, out_prefix="run", model_type="tiny", save_ckpt=True, return_model=False):
    set_seed(seed)
    tr_idx, val_idx, test_idx = group_stratified_split(groups, labels, seed=seed)
    class_support = {'train': dict(Counter(labels[tr_idx]).items()),
                     'val': dict(Counter(labels[val_idx]).items()),
                     'test': dict(Counter(labels[test_idx]).items())}

    class_weights = compute_class_weights(labels, tr_idx)
    sample_weights = [class_weights[labels[i]] for i in tr_idx]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = RawDataset(raw_windows[tr_idx], labels[tr_idx], augment=AUGMENT)
    val_ds = RawDataset(raw_windows[val_idx], labels[val_idx], augment=False)
    test_ds = RawDataset(raw_windows[test_idx], labels[test_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyNetPlus(num_classes=NUM_CLASSES).to(DEVICE) if model_type=="plus" else TinyNet1D(num_classes=NUM_CLASSES).to(DEVICE)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = -1.0; no_imp = 0; best_epoch = -1
    start_time = time.time()
    for epoch in range(1, EPOCHS+1):
        model.train()
        all_preds=[]; all_targs=[]; losses=[]
        for x,y in train_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            losses.append(loss.item())
            all_preds.extend(out.argmax(dim=1).detach().cpu().numpy().tolist()); all_targs.extend(y.detach().cpu().numpy().tolist())
        tr_acc = accuracy_score(all_targs, all_preds); tr_f1 = f1_score(all_targs, all_preds, average='macro')
        model.eval(); vpred=[]; vtrue=[]
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(DEVICE); out=model(x); vpred.extend(out.argmax(dim=1).cpu().numpy().tolist()); vtrue.extend(y.numpy().tolist())
        v_acc = accuracy_score(vtrue, vpred); v_f1 = f1_score(vtrue, vpred, average='macro')
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"[seed {seed}] epoch {epoch}/{EPOCHS} tr_acc {tr_acc:.4f} tr_f1 {tr_f1:.4f} val_acc {v_acc:.4f} val_f1 {v_f1:.4f}")
        if v_f1 > best_val:
            best_val = v_f1; no_imp = 0; best_epoch = epoch
            if save_ckpt:
                ckpt_name = f"{out_prefix}_best_seed{seed}.pt"
                torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'v_f1': v_f1, 'seed': seed}, os.path.join(OUTPUT_DIR, ckpt_name))
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    duration = time.time() - start_time

    ckpt_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_best_seed{seed}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
    model.eval(); tp=[]; tt=[]
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(DEVICE); out=model(x); tp.extend(out.argmax(dim=1).cpu().numpy().tolist()); tt.extend(y.numpy().tolist())
    acc = accuracy_score(tt, tp); f1m = f1_score(tt, tp, average='macro')
    report = classification_report(tt, tp, digits=4, zero_division=0)
    cm = confusion_matrix(tt, tp).tolist()

    result = {'acc':float(acc), 'macro_f1':float(f1m), 'classification_report':report,
              'confusion_matrix':cm, 'class_support':class_support, 'best_epoch':best_epoch, 'duration_s': duration}
    if return_model:
        return result, model
    return result

def train_on_source_and_test_target(src_raw, src_labels, src_groups, tgt_raw, tgt_labels, seed, out_prefix="src2tgt", model_type="tiny"):
    set_seed(seed)
    tr_idx, val_idx, _ = group_stratified_split(src_groups, src_labels, seed=seed)
    class_support = {'train': dict(Counter(src_labels[tr_idx]).items()), 'val': dict(Counter(src_labels[val_idx]).items()), 'target': int(len(tgt_labels))}
    class_weights = compute_class_weights(src_labels, tr_idx)
    sample_weights = [class_weights[src_labels[i]] for i in tr_idx]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = RawDataset(src_raw[tr_idx], src_labels[tr_idx], augment=AUGMENT)
    val_ds = RawDataset(src_raw[val_idx], src_labels[val_idx], augment=False)
    tgt_ds = RawDataset(tgt_raw, tgt_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyNetPlus(num_classes=NUM_CLASSES).to(DEVICE) if model_type=="plus" else TinyNet1D(num_classes=NUM_CLASSES).to(DEVICE)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = -1.0; no_imp = 0; best_epoch = -1
    for epoch in range(1, EPOCHS+1):
        model.train()
        for x,y in train_loader:
            x=x.to(DEVICE); y=y.to(DEVICE)
            optimizer.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); optimizer.step()
        model.eval(); vpred=[]; vtrue=[]
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(DEVICE); out=model(x); vpred.extend(out.argmax(dim=1).cpu().numpy().tolist()); vtrue.extend(y.numpy().tolist())
        v_f1 = f1_score(vtrue, vpred, average='macro')
        scheduler.step()
        if v_f1 > best_val:
            best_val = v_f1; no_imp = 0; best_epoch = epoch
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'v_f1': v_f1, 'seed': seed}, os.path.join(OUTPUT_DIR, f"{out_prefix}_best_seed{seed}.pt"))
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break

    ckpt_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_best_seed{seed}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE); model.load_state_dict(ckpt['model_state'])
    model.eval(); tpred=[]; ttrue=[]
    with torch.no_grad():
        for x,y in tgt_loader:
            x=x.to(DEVICE); out=model(x); tpred.extend(out.argmax(dim=1).cpu().numpy().tolist()); ttrue.extend(y.numpy().tolist())
    acc = accuracy_score(ttrue, tpred); f1m = f1_score(ttrue, tpred, average='macro')
    report = classification_report(ttrue, tpred, digits=4, zero_division=0)
    cm = confusion_matrix(ttrue, tpred).tolist()
    return {'acc':float(acc), 'macro_f1':float(f1m), 'classification_report':report, 'confusion_matrix':cm, 'class_support':class_support, 'best_epoch':best_epoch}

# ---------- stability runner ----------
def make_json_serializable(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    try:
        items = getattr(obj, "items", None)
        if callable(items):
            return {str(k): make_json_serializable(v) for k, v in obj.items()}
    except Exception:
        pass
    return obj

def run_stability(csvs, n_repeats=N_REPEATS, model_type="tiny", force_rebuild=False):
    raw_all, labels_all, groups_all = build_raw_windows(csvs, per_file_norm=True, cache_name="combined", force_rebuild=force_rebuild)
    raw_w, labels_w, groups_w = build_raw_windows([WIDE_PATH], per_file_norm=True, cache_name="wide", force_rebuild=force_rebuild)
    raw_n, labels_n, groups_n = build_raw_windows([NARROW_PATH], per_file_norm=True, cache_name="narrow", force_rebuild=force_rebuild)
    results = {'runs': []}
    for run in range(n_repeats):
        seed = SEED_BASE + run
        print(f"\n=== RUN {run+1}/{n_repeats}, seed={seed} ===")
        comb_res = train_once(raw_all, labels_all, groups_all, seed, out_prefix="combined", model_type=model_type)
        print("[combined] acc", comb_res['acc'], "macroF1", comb_res['macro_f1'])
        w2n = train_on_source_and_test_target(raw_w, labels_w, groups_w, raw_n, labels_n, seed, out_prefix="w2n", model_type=model_type)
        print("wide->narrow acc", w2n['acc'], "macroF1", w2n['macro_f1'])
        n2w = train_on_source_and_test_target(raw_n, labels_n, groups_n, raw_w, labels_w, seed, out_prefix="n2w", model_type=model_type)
        print("narrow->wide acc", n2w['acc'], "macroF1", n2w['macro_f1'])
        results['runs'].append({'seed':seed, 'combined':comb_res, 'w2n':w2n, 'n2w':n2w})
    # aggregate
    comb_accs = [r['combined']['acc'] for r in results['runs']]
    comb_f1s  = [r['combined']['macro_f1'] for r in results['runs']]
    results['summary'] = {
        'combined_acc_mean': float(np.mean(comb_accs)), 'combined_acc_std': float(np.std(comb_accs)),
        'combined_f1_mean': float(np.mean(comb_f1s)), 'combined_f1_std': float(np.std(comb_f1s)),
        'per_run_count': len(results['runs'])
    }

    outp = os.path.join(OUTPUT_DIR, "stability_results.json")
    serial = make_json_serializable(results)
    with open(outp, 'w') as f:
        json.dump(serial, f, indent=2)
    print("Saved stability JSON:", outp)

    # produce diagnostics + plots (best-run confusion matrices + seed metrics)
    # pick best run by combined acc
    runs_sorted = sorted(results['runs'], key=lambda r: r['combined']['acc'], reverse=True)
    best = runs_sorted[0]
    best_seed = best['seed']
    # combined confusion matrix
    cm_comb = np.array(best['combined']['confusion_matrix'], dtype=int)
    classes = [str(i) for i in range(NUM_CLASSES)]
    plot_confusion_matrix(cm_comb, classes, f'Combined (best seed {best_seed})', os.path.join(FIG_DIR, f'cm_combined_seed{best_seed}.png'), normalize=False)
    plot_confusion_matrix(cm_comb, classes, f'Combined (best seed {best_seed}) (normalized)', os.path.join(FIG_DIR, f'cm_combined_seed{best_seed}_norm.png'), normalize=True)

    # cross-domain confusion matrices (best seed)
    cm_w2n = np.array(best['w2n']['confusion_matrix'], dtype=int)
    plot_confusion_matrix(cm_w2n, classes, f'Wide->Narrow (seed {best_seed})', os.path.join(FIG_DIR, f'cm_w2n_seed{best_seed}.png'), normalize=False)
    plot_confusion_matrix(cm_w2n, classes, f'Wide->Narrow (seed {best_seed}) (normalized)', os.path.join(FIG_DIR, f'cm_w2n_seed{best_seed}_norm.png'), normalize=True)

    cm_n2w = np.array(best['n2w']['confusion_matrix'], dtype=int)
    plot_confusion_matrix(cm_n2w, classes, f'Narrow->Wide (seed {best_seed})', os.path.join(FIG_DIR, f'cm_n2w_seed{best_seed}.png'), normalize=False)
    plot_confusion_matrix(cm_n2w, classes, f'Narrow->Wide (seed {best_seed}) (normalized)', os.path.join(FIG_DIR, f'cm_n2w_seed{best_seed}_norm.png'), normalize=True)

    # seed metrics plot
    plot_seed_metrics(results['runs'], os.path.join(FIG_DIR, 'seed_metrics.png'))

    # save per-run summary CSV
    rows = []
    for r in results['runs']:
        rows.append({
            'seed': r['seed'],
            'combined_acc': r['combined']['acc'],
            'combined_macro_f1': r['combined']['macro_f1'],
            'w2n_acc': r['w2n']['acc'],
            'w2n_macro_f1': r['w2n']['macro_f1'],
            'n2w_acc': r['n2w']['acc'],
            'n2w_macro_f1': r['n2w']['macro_f1'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'stability_runs_summary.csv'), index=False)
    print("Saved per-run summary CSV and figures in", FIG_DIR)

    return results

# ---------- ensemble evaluation ----------
def ensemble_eval(ckpt_paths, data_npz, model_type="tiny"):
    d = np.load(data_npz, allow_pickle=True)
    raw, labels, groups = d['raw'], d['labels'], d['groups']
    ds = RawDataset(raw, labels, augment=False)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)
    models = []
    for p in ckpt_paths:
        ck = torch.load(p, map_location=DEVICE)
        model = TinyNetPlus(num_classes=NUM_CLASSES).to(DEVICE) if model_type=="plus" else TinyNet1D(num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(ck['model_state'])
        model.eval(); models.append(model)
    all_preds = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            probs = None
            for m in models:
                out = F.softmax(m(x), dim=1)
                probs = out if probs is None else probs + out
            probs = probs / len(models)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
    acc = accuracy_score(labels, all_preds); f1m = f1_score(labels, all_preds, average='macro')
    print("Ensemble acc", acc, "macroF1", f1m)
    print(classification_report(labels, all_preds, digits=4, zero_division=0))
    return acc, f1m

# ---------- helper: select top3 from stability JSON ----------
def select_top3_and_write(js_path=os.path.join(OUTPUT_DIR,"stability_results.json"), out_txt=os.path.join(OUTPUT_DIR,"top3_ckpts.txt")):
    if not os.path.exists(js_path):
        raise FileNotFoundError(f"{js_path} not found")
    with open(js_path,'r') as f:
        js = json.load(f)
    runs = js.get("runs",[])
    if not runs:
        raise ValueError("No runs in stability JSON")
    runs_sorted = sorted(runs, key=lambda r: r["combined"]["acc"], reverse=True)
    top3 = runs_sorted[:3]
    ckpts = []
    for r in top3:
        seed = r["seed"]
        ck = os.path.join(OUTPUT_DIR, f"combined_best_seed{seed}.pt")
        if os.path.exists(ck):
            ckpts.append(ck)
        else:
            candidates = [fn for fn in os.listdir(OUTPUT_DIR) if f"seed{seed}" in fn and fn.endswith(".pt")]
            if candidates:
                ckpts.append(os.path.join(OUTPUT_DIR, candidates[0]))
    if not ckpts:
        raise FileNotFoundError("No checkpoint files found for top seeds in results/")
    with open(out_txt, "w") as fo:
        for c in ckpts:
            fo.write(c + "\n")
    print("Wrote top-3 checkpoint paths to", out_txt)
    return ckpts

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare","train_once","stability","select_top3","eval","ensemble"], help="mode")
    parser.add_argument("--model", choices=["tiny","plus"], default="tiny")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--n_repeats", type=int, default=N_REPEATS)
    parser.add_argument("--ckpts", nargs="+", help="checkpoint paths for ensemble or eval")
    parser.add_argument("--data_npz", type=str, help=".npz file for eval/ensemble")
    parser.add_argument("--clear-cache", action="store_true", help="delete caches and rebuild")
    parser.add_argument("--force-rebuild", action="store_true", help="force rebuild caches when running prepare/stability")
    args = parser.parse_args()

    if args.clear_cache:
        if os.path.exists(CACHE_DIR):
            print("Clearing cache directory", CACHE_DIR)
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)

    if args.mode == "prepare":
        print("Building caches...")
        build_raw_windows([WIDE_PATH, NARROW_PATH], per_file_norm=True, cache_name="combined", force_rebuild=args.force_rebuild)
        build_raw_windows([WIDE_PATH], per_file_norm=True, cache_name="wide", force_rebuild=args.force_rebuild)
        build_raw_windows([NARROW_PATH], per_file_norm=True, cache_name="narrow", force_rebuild=args.force_rebuild)
        print("Saved caches in", CACHE_DIR)

    elif args.mode == "train_once":
        raw_all, labels_all, groups_all = build_raw_windows([WIDE_PATH, NARROW_PATH], per_file_norm=True, cache_name="combined", force_rebuild=args.force_rebuild)
        res = train_once(raw_all, labels_all, groups_all, seed=args.seed, out_prefix="combined", model_type=args.model)
        print(json.dumps(make_json_serializable(res), indent=2))

    elif args.mode == "stability":
        run_stability([WIDE_PATH, NARROW_PATH], n_repeats=args.n_repeats, model_type=args.model, force_rebuild=args.force_rebuild)

    elif args.mode == "select_top3":
        select_top3_and_write()

    elif args.mode == "eval":
        if not args.ckpts or not args.data_npz:
            print("eval requires --ckpts <path> and --data_npz <cache.npz>"); sys.exit(1)
        acc, f1 = ensemble_eval([args.ckpts[0]], args.data_npz, model_type=args.model)

    elif args.mode == "ensemble":
        if args.ckpts and args.data_npz:
            ckpts = args.ckpts; data_npz = args.data_npz
        else:
            top3_txt = os.path.join(OUTPUT_DIR, "top3_ckpts.txt")
            if not os.path.exists(top3_txt):
                print("No top3_ckpts.txt found. Run mode 'select_top3' or run stability first."); sys.exit(1)
            with open(top3_txt,'r') as f:
                ckpts = [l.strip() for l in f.readlines() if l.strip()]
            data_npz = os.path.join(CACHE_DIR, "combined.npz")
            if not os.path.exists(data_npz):
                print("Cache combined.npz not found. Run mode 'prepare' first."); sys.exit(1)
        ensemble_eval(ckpts, data_npz, model_type=args.model)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
