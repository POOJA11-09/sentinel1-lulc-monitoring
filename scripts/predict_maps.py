#!/usr/bin/env python3
"""
Predict full-AOI monthly LULC maps (12 months of 2025) with a trained model.

The aligned Sentinel-1 rasters (outputs/aligned/<res>m/) are processed in overlapping
tiles; softmax probabilities are blended with a smooth window, so there are no tile seams.

Usage:
    python3 scripts/predict_maps.py --run segformer_train10_VV-VH-RATIO_s42 --input-res 10
    python3 scripts/predict_maps.py --run unet_train40_VV-VH-RATIO_s42      --input-res 40
    python3 scripts/predict_maps.py --run unet_train40_VV-VH-RATIO_s42      --input-res 10   # Exp. C

Outputs:
    outputs/maps/<run>/input<res>m/LULC_2025_MM.tif   (uint8 class IDs 0-8, 255 = no data, colour table)
    outputs/maps/<run>/input<res>m/overview_12_months.png
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from evaluate import load_model

PALETTE = ["#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
           "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def normalise(s1, inputs, stats):
    chans = []
    for name in inputs:
        c = {"VV": s1[0], "VH": s1[1], "RATIO": s1[0] - s1[1], "mean_n_obs": s1[2]}[name]
        st = stats[name]
        chans.append((np.clip(c, st["clip_low"], st["clip_high"]) - st["mean"]) / st["std"])
    x = np.stack(chans).astype(np.float32)
    return np.nan_to_num(x, nan=0.0)


def window(tile):
    w = np.hanning(tile + 2)[1:-1]
    return np.maximum(np.outer(w, w), 1e-3).astype(np.float32)


@torch.no_grad()
def predict_raster(model, x, tile, stride, device, batch=4):
    c, h, w = x.shape
    ph = max(tile, int(np.ceil((h - tile) / stride)) * stride + tile)
    pw = max(tile, int(np.ceil((w - tile) / stride)) * stride + tile)
    xp = np.pad(x, ((0, 0), (0, ph - h), (0, pw - w)), mode="reflect")
    prob = np.zeros((T.NC, ph, pw), dtype=np.float32)
    weight = np.zeros((ph, pw), dtype=np.float32)
    win = window(tile)
    coords = [(r, q) for r in range(0, ph - tile + 1, stride) for q in range(0, pw - tile + 1, stride)]
    for i in range(0, len(coords), batch):
        cc = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([xp[:, r:r + tile, q:q + tile] for r, q in cc])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb)
        out = T.fit_size(out, xb[:, 0]).float().softmax(1).cpu().numpy()
        for k, (r, q) in enumerate(cc):
            prob[:, r:r + tile, q:q + tile] += out[k] * win
            weight[r:r + tile, q:q + tile] += win
    prob = prob[:, :h, :w] / weight[None, :h, :w]
    return prob.argmax(0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--input-res", required=True, choices=["10", "40"])
    ap.add_argument("--months", default="1-12")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}")
    model, cfg, best_epoch = load_model(a.run, device)
    stats = T.shared_stats()
    tile = 512 if a.input_res == "10" else 256
    stride = tile * 3 // 4
    lo, hi = (int(v) for v in a.months.split("-"))
    months = list(range(lo, hi + 1))

    out_dir = f"{T.PROJ}/outputs/maps/{a.run}/input{a.input_res}m"
    os.makedirs(out_dir, exist_ok=True)
    cmap_table = {i: tuple(int(PALETTE[i][j:j + 2], 16) for j in (1, 3, 5)) + (255,) for i in range(T.NC)}
    print(f"run {a.run} (best epoch {best_epoch}) | input {a.input_res} m | tile {tile}, stride {stride}", flush=True)

    maps = {}
    for m in months:
        src_path = f"{T.PROJ}/outputs/aligned/{a.input_res}m/S1_2025_{m:02d}_{a.input_res}m.tif"
        with rasterio.open(src_path) as src:
            s1 = src.read().astype(np.float32)
            profile = src.profile
        valid = np.isfinite(s1[:2]).all(0)
        pred = predict_raster(model, normalise(s1, cfg["inputs"], stats), tile, stride, device)
        pred[~valid] = 255
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate", predictor=1)
        out_path = f"{out_dir}/LULC_2025_{m:02d}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(pred, 1)
            dst.write_colormap(1, cmap_table)
            dst.set_band_description(1, "lulc")
            dst.update_tags(run=a.run, month=m, input_res_m=a.input_res, best_epoch=best_epoch,
                            classes=",".join(T.CLASS_NAMES))
        maps[m] = pred
        print(f"  month {m:02d} -> {out_path}", flush=True)

    # overview figure
    cmap = ListedColormap(PALETTE)
    norm = BoundaryNorm(np.arange(-0.5, T.NC + 0.5), cmap.N)
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    for ax, m in zip(axs.ravel(), range(1, 13)):
        ax.axis("off")
        if m in maps:
            show = maps[m].astype(float)
            show[show == 255] = np.nan
            ax.imshow(show[::2, ::2] if a.input_res == "10" else show, cmap=cmap, norm=norm,
                      interpolation="nearest")
            ax.set_title(MONTH_NAMES[m - 1] + " 2025")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in range(T.NC)]
    fig.legend(handles, T.CLASS_NAMES, loc="lower center", ncol=9, fontsize=11)
    fig.suptitle(f"Predicted LULC — {a.run} — input {a.input_res} m", fontsize=15)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    plt.savefig(f"{out_dir}/overview_12_months.png", dpi=110)
    print("Saved overview:", f"{out_dir}/overview_12_months.png", flush=True)


if __name__ == "__main__":
    main()
