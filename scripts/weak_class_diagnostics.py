#!/usr/bin/env python3
"""
Weak-class diagnostics for one trained model (default: the SegFormer 10 m baseline).

Read-only with respect to the baseline: the model checkpoint and all existing evaluation outputs are only read.
Everything is written to   outputs/diagnostics/weak_classes/<run>/

What it computes (test split unless stated otherwise):
  1. class support per split (pixels, %) and spatial concentration (number of 2.56 km blocks containing a class)
  2. class weights actually used in training, and predicted-vs-reference area bias
  3. recall and key confusions stratified by
       - Dynamic World label confidence (10 m band "confidence")
       - Dynamic World observation count (10 m band "n_obs")
       - 40 m class purity (heterogeneity of the 4x4 block around the pixel)
       - reference class boundary (pixel has a different reference class in its 3x3 neighbourhood)
       - month
  4. metrics on a "clean" subset (confidence >= 0.6, purity == 1, not a boundary) vs all pixels
  5. Sentinel-1 separability (VV, VH in dB) of the confused class pairs: medians, IQR and
     Jeffries-Matusita distance (Gaussian approximation), for all months, winter (DJF) and summer (JJA)

Usage:
    python3 scripts/weak_class_diagnostics.py                       # baseline SegFormer 10 m
    python3 scripts/weak_class_diagnostics.py --run <run_name> --gpu 0
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from evaluate import load_model, predict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = ["Water", "Trees", "Grass", "FloodVeg", "Crops", "Shrub", "Built-up", "Bare", "Snow"]
NC, IGN = 9, 255
PAIRS = [("Grass", "Crops"), ("Shrub", "Trees"), ("Bare", "Built-up"), ("FloodVeg", "Water"),
         ("Snow", "Crops"), ("Snow", "Grass"), ("Snow", "Trees"), ("Trees", "Shrub"), ("Crops", "Snow")]
C = {n: i for i, n in enumerate(NAMES)}


def cm_of(y, p):
    m = y != IGN
    return np.bincount(y[m].astype(np.int64) * NC + p[m], minlength=NC * NC).reshape(NC, NC)


def metrics(cm):
    tp, gt, pr = np.diag(cm).astype(float), cm.sum(1).astype(float), cm.sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec, prec = tp / gt, tp / pr
        iou, f1 = tp / (gt + pr - tp), 2 * tp / (gt + pr)
    return pd.DataFrame({"support": gt.astype(int), "recall": rec, "precision": prec, "F1": f1, "IoU": iou},
                        index=NAMES)


def stratify(y, p, var, bins, labels, name):
    """For each reference class and bin: share of class pixels in the bin, recall, and rates of key confusions."""
    rows = []
    b = np.digitize(var, bins)
    for ci, cname in enumerate(NAMES):
        sel_c = y == ci
        n_c = int(sel_c.sum())
        if n_c == 0:
            continue
        for k, lab in enumerate(labels):
            sel = sel_c & (b == k)
            n = int(sel.sum())
            row = {"class": cname, name: lab, "pixels": n, "share_of_class_%": 100 * n / n_c,
                   "recall_%": 100 * (p[sel] == ci).mean() if n else np.nan}
            for a, t in PAIRS:
                if a == cname:
                    row[f"->{t}_%"] = 100 * (p[sel] == C[t]).mean() if n else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def jm_distance(a, b):
    """Jeffries-Matusita distance between two 2-D Gaussian samples (0 = identical, 2 = fully separable)."""
    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a.T), np.cov(b.T)
    s = (s1 + s2) / 2
    d = m1 - m2
    bh = d @ np.linalg.solve(s, d) / 8 + 0.5 * np.log(np.linalg.det(s) / np.sqrt(np.linalg.det(s1) * np.linalg.det(s2)))
    return float(2 * (1 - np.exp(-bh)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="segformer_train10_VV-VH-RATIO_s42")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    out = f"{T.PROJ}/outputs/diagnostics/weak_classes/{a.run}"
    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    device = torch.device(f"cuda:{a.gpu}")

    model, cfg, best_epoch = load_model(a.run, device)
    T.PATCH_DIR = cfg.get("patch_dir") or T.PATCH_DIR
    info10 = json.load(open(f"{T.PATCH_DIR}/10m/dataset_info.json"))
    info40 = json.load(open(f"{T.PATCH_DIR}/40m/dataset_info.json"))
    i_conf, i_nobs = info10["aux"].index("confidence"), info10["aux"].index("n_obs")
    i_pur = info40["aux"].index("purity")
    print(f"run {a.run} (best epoch {best_epoch}); 10 m aux {info10['aux']}; 40 m aux {info40['aux']}", flush=True)

    # ---------------------------------------------------------------- 1. support per split + spatial concentration
    rows, blocks = [], []
    for split in ("train", "val", "test"):
        idx = T.common_indices(split)["10"]
        Y = np.load(f"{T.PATCH_DIR}/10m/{split}_Y.npy", mmap_mode="r")
        meta = pd.read_csv(f"{T.PATCH_DIR}/10m/{split}_meta.csv").set_index("patch_index")
        per_patch = np.stack([np.bincount(np.asarray(Y[j]).ravel(), minlength=256)[:NC] for j in idx])
        tot = per_patch.sum(0)
        for c in range(NC):
            rows.append({"split": split, "class": NAMES[c], "pixels": int(tot[c]), "percent": 100 * tot[c] / tot.sum()})
        blk = meta.loc[idx, "block_id"].values
        dfb = pd.DataFrame(per_patch, columns=NAMES); dfb["block"] = blk
        per_block = dfb.groupby("block").sum()
        blocks.append({"split": split, "n_blocks": len(per_block),
                       **{c: int((per_block[c] >= 100).sum()) for c in NAMES}})
    support = pd.DataFrame(rows).pivot(index="class", columns="split", values=["pixels", "percent"]).loc[NAMES]
    blocks = pd.DataFrame(blocks).set_index("split")
    support.round(3).to_csv(f"{out}/1_support_per_split.csv"); blocks.to_csv(f"{out}/1_blocks_containing_class.csv")
    print("\n=== 1. Class support per split ===\n", support.round(3).to_string())
    print("\n=== 1b. Number of 2.56 km blocks with >= 100 px of the class (any month) ===\n", blocks.to_string())

    # ---------------------------------------------------------------- 2. predictions on the test split
    stats = cfg.get("stats") or T.shared_stats()
    idx = T.common_indices("test")
    ds = T.PatchSet("10", "test", idx["10"], stats, cfg["inputs"], augment=False)
    pred = predict(model, ds, device, batch=16)
    y = np.asarray(ds.Y[idx["10"]])
    aux = np.load(f"{T.PATCH_DIR}/10m/test_AUX.npy", mmap_mode="r")
    conf = np.asarray(aux[idx["10"], i_conf], dtype=np.float32)
    nobs = np.asarray(aux[idx["10"], i_nobs], dtype=np.float32)
    aux40 = np.load(f"{T.PATCH_DIR}/40m/test_AUX.npy", mmap_mode="r")
    pur = np.asarray(aux40[idx["40"], i_pur], dtype=np.float32).repeat(4, axis=1).repeat(4, axis=2)
    meta = pd.read_csv(f"{T.PATCH_DIR}/10m/test_meta.csv").set_index("patch_index")
    month = np.broadcast_to(meta.loc[idx["10"], "month"].values[:, None, None], y.shape)
    X = np.load(f"{T.PATCH_DIR}/10m/test_X.npy", mmap_mode="r")
    vv = np.asarray(X[idx["10"], 0], dtype=np.float32); vh = np.asarray(X[idx["10"], 1], dtype=np.float32)

    # reference class boundary: any 8-neighbour with a different (valid) reference class
    yp = np.pad(y, ((0, 0), (1, 1), (1, 1)), mode="edge")
    boundary = np.zeros(y.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nb = yp[:, 1 + dy:1 + dy + y.shape[1], 1 + dx:1 + dx + y.shape[2]]
                boundary |= (nb != y) & (nb != IGN)

    cm = cm_of(y, pred)
    m_all = metrics(cm)
    print(f"\n=== 2. Test metrics recomputed from per-pixel predictions: OA {np.trace(cm)/cm.sum():.4f}, "
          f"mIoU {m_all.IoU.mean():.4f}, macro F1 {m_all.F1.mean():.4f} (must match the reported test metrics) ===")
    w = pd.Series(stats["class_weights"], index=NAMES, name="CE weight")
    bias = pd.DataFrame({"ref_%": 100 * cm.sum(1) / cm.sum(), "pred_%": 100 * cm.sum(0) / cm.sum()}, index=NAMES)
    bias["bias_pp"] = bias["pred_%"] - bias["ref_%"]; bias["pred/ref"] = bias["pred_%"] / bias["ref_%"]
    bias = bias.join(w)
    bias.round(4).to_csv(f"{out}/2_area_bias_and_class_weights.csv")
    print("\n=== 2b. Area bias vs. class weight used in the loss ===\n", bias.round(3).to_string())
    print(f"Spearman correlation (CE weight vs pred/ref ratio): "
          f"{bias['CE weight'].rank().corr(bias['pred/ref'].rank()):.2f}")

    # ---------------------------------------------------------------- 3. stratified analyses
    valid = y != IGN
    yv, pv = y[valid], pred[valid]
    strata = {
        "confidence": stratify(yv, pv, conf[valid], [0.4, 0.5, 0.6, 0.7], ["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", ">=0.7"], "confidence"),
        "n_obs": stratify(yv, pv, nobs[valid], [1.5, 2.5, 4.5], ["1", "2", "3-4", ">=5"], "n_obs"),
        "purity40": stratify(yv, pv, pur[valid], [0.5, 0.75, 0.999], ["<0.5", "0.5-0.75", "0.75-<1", "1.0"], "purity40"),
        "boundary": stratify(yv, pv, boundary[valid].astype(float), [0.5], ["interior", "boundary"], "boundary"),
        "month": stratify(yv, pv, month[valid].astype(float), np.arange(1.5, 12.5, 1.0), [f"{m:02d}" for m in range(1, 13)], "month"),
    }
    weak = ["Grass", "FloodVeg", "Shrub", "Bare", "Snow", "Trees", "Crops"]
    for k, df in strata.items():
        df.round(2).to_csv(f"{out}/3_stratified_by_{k}.csv", index=False)
        print(f"\n=== 3. Stratified by {k} (weak classes + Trees/Crops) ===")
        print(df[df["class"].isin(weak)].round(1).to_string(index=False))

    # share of each class's reference pixels that are low-confidence / boundary / impure
    summ = pd.DataFrame({
        "conf<0.5_%": [100 * (conf[valid][yv == c] < 0.5).mean() for c in range(NC)],
        "n_obs==1_%": [100 * (nobs[valid][yv == c] <= 1).mean() for c in range(NC)],
        "purity<1_%": [100 * (pur[valid][yv == c] < 0.999).mean() for c in range(NC)],
        "boundary_%": [100 * boundary[valid][yv == c].mean() for c in range(NC)],
        "median_conf": [np.median(conf[valid][yv == c]) for c in range(NC)]}, index=NAMES)
    summ.round(2).to_csv(f"{out}/3_label_quality_per_class.csv")
    print("\n=== 3b. Label quality per reference class ===\n", summ.round(2).to_string())

    # ---------------------------------------------------------------- 4. clean subset
    clean = valid & (conf >= 0.6) & (pur >= 0.999) & ~boundary
    m_clean = metrics(cm_of(np.where(clean, y, IGN), pred))
    comp = m_all[["support", "recall", "IoU"]].add_suffix("_all").join(m_clean[["support", "recall", "IoU"]].add_suffix("_clean"))
    comp["clean_share_%"] = 100 * comp.support_clean / comp.support_all
    comp.round(3).to_csv(f"{out}/4_all_vs_clean_pixels.csv")
    print(f"\n=== 4. All pixels vs clean pixels (conf>=0.6, purity==1, not boundary; "
          f"{100*clean.sum()/valid.sum():.1f} % of labelled pixels) ===\n", comp.round(3).to_string())
    print(f"mIoU all {m_all.IoU.mean():.3f} | mIoU clean (classes present) {m_clean.IoU[m_clean.support > 0].mean():.3f}")

    # ---------------------------------------------------------------- 5. Sentinel-1 separability
    rng = np.random.default_rng(0)
    seasons = {"all": np.ones_like(month, bool), "DJF": np.isin(month, [12, 1, 2]), "JJA": np.isin(month, [6, 7, 8])}
    srows = []
    for a_, b_ in [("Grass", "Crops"), ("Shrub", "Trees"), ("Bare", "Built-up"), ("FloodVeg", "Water"),
                   ("Snow", "Crops"), ("Snow", "Grass"), ("Snow", "Trees"), ("Trees", "Crops"), ("Water", "Built-up")]:
        for s, ms in seasons.items():
            sa, sb = valid & ms & (y == C[a_]) & np.isfinite(vv), valid & ms & (y == C[b_]) & np.isfinite(vv)
            if sa.sum() < 50 or sb.sum() < 50:
                srows.append({"pair": f"{a_} vs {b_}", "season": s, "n_a": int(sa.sum()), "n_b": int(sb.sum())}); continue
            A = np.stack([vv[sa], vh[sa]], 1); B = np.stack([vv[sb], vh[sb]], 1)
            A = A[rng.choice(len(A), min(len(A), 200000), replace=False)]
            B = B[rng.choice(len(B), min(len(B), 200000), replace=False)]
            srows.append({"pair": f"{a_} vs {b_}", "season": s, "n_a": int(sa.sum()), "n_b": int(sb.sum()),
                          "VV_med_a": np.median(A[:, 0]), "VV_med_b": np.median(B[:, 0]),
                          "VH_med_a": np.median(A[:, 1]), "VH_med_b": np.median(B[:, 1]),
                          "VV_IQR_a": np.subtract(*np.percentile(A[:, 0], [75, 25])),
                          "VH_IQR_a": np.subtract(*np.percentile(A[:, 1], [75, 25])),
                          "JM_distance": jm_distance(A, B)})
    sep = pd.DataFrame(srows)
    sep.round(3).to_csv(f"{out}/5_sar_separability.csv", index=False)
    print("\n=== 5. Sentinel-1 separability (dB; JM: 0 = identical, 2 = separable) ===\n", sep.round(2).to_string(index=False))

    # ---------------------------------------------------------------- figures
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    for ax, (k, col) in zip(axs, [("confidence", "confidence"), ("purity40", "purity40"), ("month", "month")]):
        df = strata[k]
        for cname in ["Grass", "Shrub", "Bare", "Snow", "Trees", "Crops", "Built-up", "Water"]:
            d = df[df["class"] == cname]
            ax.plot(d[col], d["recall_%"], "-o", label=cname)
        ax.set_title(f"Recall by {k}"); ax.set_ylabel("recall (%)"); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out}/fig_recall_stratified.png", dpi=130)
    print("\nAll outputs written to:", out)


if __name__ == "__main__":
    main()
