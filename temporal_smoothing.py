#!/usr/bin/env python3
"""
Temporal smoothing of the 12 monthly LULC maps and its effect on change statistics.

Monthly maps from a per-month classifier "flicker": a pixel can switch class from one month to the next
because of classification error, not because the land cover changed. This script measures that effect and
tests two simple post-processing rules:

  * majority3   : each month takes the most frequent class of (previous, same, next) month
                  (first/last month: 2-month window; ties keep the original class)
  * persistence : a pixel only changes class if the new class is also present in the following month
                  (last month: only if it equals the previous month)

For every result folder (results/lat…_lon…_YYYY-MM/) it writes, to results/temporal_smoothing/:
  - <site>_change_statistics.csv  gross change (pixels switching class) and net change per month pair
  - <site>_total_change.csv       first → last month net change per class, raw vs smoothed
  - <site>_smoothing.png          figure: gross change per month pair + total change per class + example maps
  - summary.csv                   one line per site and method

Run from the prototype folder:  python temporal_smoothing.py
"""
import glob, json, os

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

CLASSES = ["Water", "Trees", "Grass", "Flooded vegetation", "Crops",
           "Shrub and scrub", "Built-up", "Bare ground", "Snow and ice"]
PALETTE = ["#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
           "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"]
NC, NODATA = 9, 255
HERE = os.path.dirname(os.path.abspath(__file__))
SITE_NAMES = {"lat49.4875_lon8.4660": "Mannheim", "lat50.1109_lon8.6821": "Frankfurt",
              "lat48.1374_lon11.5755": "Munich", "lat30.0444_lon31.2357": "Cairo",
              "lat49.3980_lon8.6750": "Heidelberg", "lat48.7758_lon9.1829": "Stuttgart",
              "lat52.5200_lon13.4050": "Berlin"}


# ------------------------------------------------------------------ smoothing rules
def majority3(stack):
    """stack: (T, H, W) uint8. Centred 3-month majority; ties keep the original class."""
    T = stack.shape[0]
    out = stack.copy()
    for t in range(T):
        lo, hi = max(0, t - 1), min(T, t + 2)
        win = stack[lo:hi]
        counts = np.stack([(win == c).sum(0) for c in range(NC)])          # (NC, H, W)
        best = counts.argmax(0).astype(np.uint8)
        best_n = counts.max(0)
        own_n = np.take_along_axis(counts, np.minimum(stack[t], NC - 1)[None].astype(np.int64), 0)[0]
        replace = (best_n > own_n) & (stack[t] != NODATA)                   # strict majority only
        out[t][replace] = best[replace]
    return out


def persistence(stack):
    """A change is accepted only if the new class persists into the next month."""
    T = stack.shape[0]
    out = stack.copy()
    for t in range(1, T):
        prev, cur = out[t - 1], stack[t]
        nxt = stack[t + 1] if t + 1 < T else stack[t - 1]
        keep_prev = (cur != prev) & (cur != nxt) & (prev != NODATA) & (cur != NODATA)
        out[t] = np.where(keep_prev, prev, cur)
    return out


METHODS = {"raw": lambda s: s, "majority3": majority3, "persistence": persistence}


# ------------------------------------------------------------------ statistics
def shares(a):
    v = a != NODATA
    return 100 * np.bincount(a[v], minlength=NC)[:NC] / max(int(v.sum()), 1)


def change_table(stack, labels):
    rows = []
    for t in range(1, len(labels)):
        a, b = stack[t - 1], stack[t]
        v = (a != NODATA) & (b != NODATA)
        d = shares(b) - shares(a)
        rows.append({"pair": f"{labels[t-1]}→{labels[t]}",
                     "gross_change_%": 100 * (a[v] != b[v]).mean(),
                     "net_change_pp": np.abs(d).sum() / 2,
                     "built_up_change_pp": d[6]})
    return pd.DataFrame(rows).set_index("pair")


def analyse(folder, out_dir):
    tag = os.path.basename(folder.rstrip("/\\"))
    res40 = tag.endswith("_40m")
    base = tag[:-4] if res40 else tag
    site = SITE_NAMES.get(base.rsplit("_", 1)[0], base) + (" (40 m)" if res40 else "")
    files = sorted(f for f in glob.glob(os.path.join(folder, "LULC_*.tif"))
                   if os.path.basename(f)[5:12].replace("-", "").isdigit())
    if len(files) < 3:
        return None
    labels = [os.path.basename(f)[5:12] for f in files]
    raw = np.stack([rasterio.open(f).read(1) for f in files])

    stacks = {m: f(raw) for m, f in METHODS.items()}
    tables, totals, summary = {}, {}, []
    summer = [p for p in change_table(raw, labels).index
              if p.split("→")[0][-2:] in ("04", "05", "06", "07", "08", "09")]
    for m, s in stacks.items():
        t = change_table(s, labels)
        tables[m] = t
        tot = pd.Series(shares(s[-1]) - shares(s[0]), index=CLASSES)
        totals[m] = tot
        summary.append({"site": site, "method": m,
                        "mean_gross_change_%": t["gross_change_%"].mean(),
                        "mean_gross_change_summer_%": t.loc[summer, "gross_change_%"].mean() if summer else np.nan,
                        "max_abs_monthly_built_up_pp": t["built_up_change_pp"].abs().max(),
                        "total_built_up_change_pp": tot["Built-up"],
                        "first_last_gross_change_%": 100 * (s[0] != s[-1]).mean(),
                        "pixels_modified_vs_raw_%": 100 * (s != raw).mean()})

    pd.concat({m: t for m, t in tables.items()}, axis=1).round(3).to_csv(
        os.path.join(out_dir, f"{site}_change_statistics.csv"))
    pd.DataFrame(totals).round(3).to_csv(os.path.join(out_dir, f"{site}_total_change.csv"))

    # ---- figure
    cmap, norm = ListedColormap(PALETTE), BoundaryNorm(np.arange(-0.5, NC + 0.5), NC)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, :2])
    styles = {"raw": ("#555555", "-o"), "majority3": ("#003B6F", "-s"), "persistence": ("#5B7FA6", "--^")}
    for m, t in tables.items():
        ax1.plot(range(len(t)), t["gross_change_%"], styles[m][1], color=styles[m][0], label=m)
    ax1.set_xticks(range(len(t))); ax1.set_xticklabels(t.index, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("pixels changing class (%)"); ax1.grid(alpha=0.3); ax1.legend()
    ax1.set_title(f"{site}: month-to-month gross change, raw vs. temporally smoothed")
    ax2 = fig.add_subplot(gs[0, 2])
    pd.DataFrame(totals).plot.barh(ax=ax2, color=["#555555", "#003B6F", "#5B7FA6"], width=0.8)
    ax2.axvline(0, color="k", lw=0.8); ax2.set_xlabel("percentage points")
    ax2.set_title(f"Total change {labels[0]} → {labels[-1]}"); ax2.legend(fontsize=8)
    k = int(np.argmax(tables["raw"]["gross_change_%"].values)) + 1     # month with the largest raw change
    for i, (m, title) in enumerate([("raw", "raw"), ("majority3", "3-month majority"),
                                    ("persistence", "persistence rule")]):
        ax = fig.add_subplot(gs[1, i])
        a = stacks[m][k].astype(float); a[stacks[m][k] == NODATA] = np.nan
        ax.imshow(a, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{labels[k]} — {title}"); ax.axis("off")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in PALETTE], CLASSES,
               loc="lower center", ncol=9, fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(os.path.join(out_dir, f"{site}_smoothing.png"), dpi=130)
    plt.close(fig)
    return pd.DataFrame(summary)


def main():
    res_dir = os.path.join(HERE, "results")
    out_dir = os.path.join(res_dir, "temporal_smoothing")
    os.makedirs(out_dir, exist_ok=True)
    folders = sorted(d for d in glob.glob(os.path.join(res_dir, "lat*_lon*_*")) if os.path.isdir(d))
    print("Result folders:", [os.path.basename(f) for f in folders])
    parts = [r for r in (analyse(f, out_dir) for f in folders) if r is not None]
    if not parts:
        print("No complete result folders found."); return
    summary = pd.concat(parts, ignore_index=True)
    summary.round(2).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary.round(2).to_string(index=False))
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
