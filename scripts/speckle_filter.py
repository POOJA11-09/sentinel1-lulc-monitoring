#!/usr/bin/env python3
"""
Speckle-filter experiment: create a copy of the 10 m patches with a Lee filter applied to VV and VH.

The original monthly composites only use temporal multi-looking (mean of 2-8 scenes). This script adds the
classic Lee filter (Lee, 1980) in linear power on top, so that a model trained on filtered inputs can be
compared with the unfiltered baseline under otherwise identical conditions (same patches, labels, splits).

Per-pixel noise level: a single Sentinel-1 IW GRD-HR scene has an equivalent number of looks (ENL) of
about 4.4; averaging n scenes gives ENL ~ 4.4 * n, where n is the per-pixel observation count (band 3).

    python3 scripts/speckle_filter.py --window 5

Output: patches_lee/10m/{train,val,test}_X.npy (filtered), labels/metadata linked from patches/10m,
        patches_lee/40m -> link to patches/40m (unchanged; only needed for the shared split index),
        patches_lee/10m/norm_stats_train.json (normalisation statistics of the filtered training data).
"""
import argparse, json, os

import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import uniform_filter

PROJ = "/home/jovyan/vault/heidelberg_lulc_2025"
SRC, ENL_SINGLE = f"{PROJ}/patches", 4.4


def lee(db, n_obs, w):
    """db: (N, H, W) backscatter in dB; n_obs: (N, H, W) observation count. Returns filtered dB."""
    lin = np.power(10.0, db / 10.0, dtype=np.float64)
    size = (1, w, w)
    mean = uniform_filter(lin, size=size, mode="reflect")
    mean_sq = uniform_filter(lin * lin, size=size, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    cu2 = 1.0 / (ENL_SINGLE * np.maximum(n_obs, 1.0))                  # squared noise coefficient of variation
    var_x = np.maximum((var - mean * mean * cu2) / (1.0 + cu2), 0.0)
    k = var_x / np.maximum(var, 1e-12)
    out = mean + k * (lin - mean)
    return (10.0 * np.log10(np.maximum(out, 1e-10))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--out", default=f"{PROJ}/patches_lee")
    a = ap.parse_args()
    d10 = f"{a.out}/10m"
    os.makedirs(d10, exist_ok=True)
    if not os.path.exists(f"{a.out}/40m"):
        os.symlink(f"{SRC}/40m", f"{a.out}/40m")

    for split in ("train", "val", "test"):
        for suffix in ("_Y.npy", "_AUX.npy", "_meta.csv"):
            dst = f"{d10}/{split}{suffix}"
            if not os.path.exists(dst):
                os.symlink(f"{SRC}/10m/{split}{suffix}", dst)
        X = np.load(f"{SRC}/10m/{split}_X.npy", mmap_mode="r")
        out = open_memmap(f"{d10}/{split}_X.npy", mode="w+", dtype=np.float32, shape=X.shape)
        diffs = []
        for i in range(0, len(X), 32):
            x = np.asarray(X[i:i + 32], dtype=np.float32)
            y = x.copy()
            for c in (0, 1):                                           # VV, VH; channel 2 = n_obs unchanged
                y[:, c] = lee(x[:, c], x[:, 2], a.window)
            out[i:i + 32] = y
            diffs.append(np.abs(y[:, :2] - x[:, :2]).mean())
        out.flush(); del out
        print(f"{split:5s}: {len(X)} patches filtered (Lee {a.window}x{a.window}), "
              f"mean |change| {np.mean(diffs):.3f} dB", flush=True)

    # normalisation statistics of the filtered training data (same procedure as 04_patch_preparation)
    X = np.load(f"{d10}/train_X.npy", mmap_mode="r")
    sample = np.asarray(X[np.sort(np.random.default_rng(0).choice(len(X), min(300, len(X)), replace=False))])
    stats = {}
    for c, name in enumerate(["VV", "VH", "mean_n_obs"]):
        v = sample[:, c].ravel(); v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [0.5, 99.5]); vc = np.clip(v, lo, hi)
        stats[name] = {"clip_low": float(lo), "clip_high": float(hi), "mean": float(vc.mean()), "std": float(vc.std())}
    json.dump(stats, open(f"{d10}/norm_stats_train.json", "w"), indent=2)
    print("normalisation (filtered train):", {k: {kk: round(vv, 3) for kk, vv in s.items()} for k, s in stats.items()})

    # speckle check: coefficient of variation in 7x7 windows over the training sample, before vs after
    Xo = np.asarray(np.load(f"{SRC}/10m/train_X.npy", mmap_mode="r")[:50, 0], dtype=np.float64)
    Xf = np.asarray(X[:50, 0], dtype=np.float64)
    def cv(db):
        lin = 10 ** (db / 10); m = uniform_filter(lin, (1, 7, 7)); s = np.sqrt(np.maximum(uniform_filter(lin * lin, (1, 7, 7)) - m * m, 0))
        return float(np.median(s / m))
    print(f"median local coefficient of variation (VV, 7x7): before {cv(Xo):.3f}, after {cv(Xf):.3f}")
    print("Done:", a.out)


if __name__ == "__main__":
    main()
