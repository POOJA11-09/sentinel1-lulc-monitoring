"""
Sentinel-1 LULC prototype
=========================

Input   : latitude, longitude, start month (YYYY-MM)
Process : 1. square AOI around the point, on a 10 m UTM grid (origin snapped to 40 m)
          2. Sentinel-1 GRD (IW, VV+VH, ascending; most frequent relative orbit) from Google Earth Engine
          3. 12 monthly composites: mean in linear power -> dB  (same as the training data)
          4. trained segmentation model (Sentinel-1 only) -> 12 monthly LULC maps (9 Dynamic World classes)
Output  : class percentages per month, month-to-month change and total change (percentage points),
          12-month map figure, GeoTIFFs, and an independent sanity check against ESA WorldCover 2021.

Command line:
    python lulc_prototype.py --lat 49.4875 --lon 8.466 --start 2025-01 --project YOUR_GEE_PROJECT
"""
import argparse, datetime as dt, json, math, os, time, warnings

import numpy as np
import pandas as pd

CLASS_NAMES = ["Water", "Trees", "Grass", "Flooded vegetation", "Crops",
               "Shrub and scrub", "Built-up", "Bare ground", "Snow and ice"]
PALETTE = ["#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
           "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"]
NC, NODATA = 9, 255

# ESA WorldCover v200 code -> Dynamic World class id (used only for the independent check)
WORLDCOVER_TO_DW = {10: 1,   # Tree cover            -> Trees
                    20: 5,   # Shrubland             -> Shrub and scrub
                    30: 2,   # Grassland             -> Grass
                    40: 4,   # Cropland              -> Crops
                    50: 6,   # Built-up              -> Built-up
                    60: 7,   # Bare / sparse veg.    -> Bare ground
                    70: 8,   # Snow and ice          -> Snow and ice
                    80: 0,   # Permanent water       -> Water
                    90: 3,   # Herbaceous wetland    -> Flooded vegetation
                    95: 3,   # Mangroves             -> Flooded vegetation
                    100: 2}  # Moss and lichen       -> Grass


def log(msg, verbose=True):
    if verbose:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- grid and months
def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_grid(lat, lon, size_km=10.0, res=10):
    """Square grid centred on the point, UTM, 10 m pixels, corners on the 40 m lattice (as in training)."""
    from pyproj import Transformer
    epsg = utm_epsg(lat, lon)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    n = int(round(size_km * 1000 / 40)) * 4                    # pixels, multiple of 4
    half = n * res / 2
    left = math.floor((x - half) / 40) * 40
    top = math.ceil((y + half) / 40) * 40
    return {"epsg": epsg, "crs": f"EPSG:{epsg}", "res": res, "left": float(left), "top": float(top),
            "width": n, "height": n, "center_lat": lat, "center_lon": lon, "size_km": n * res / 1000}


def month_list(start):
    y, m = (int(v) for v in start.split("-"))
    out = [(y + (m - 1 + i) // 12, (m - 1 + i) % 12 + 1) for i in range(12)]
    last_y, last_m = out[-1]
    end_of_last = dt.date(last_y + (last_m == 12), last_m % 12 + 1, 1)
    if end_of_last > dt.date.today():
        raise ValueError(f"The 12 months starting {start} end in {last_y}-{last_m:02d}, which is not "
                         f"complete yet. Choose an earlier start month.")
    return out


# ----------------------------------------------------------------------------- Earth Engine
def init_ee(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=project)
    return ee


def grid_region(ee, grid):
    g = grid
    return ee.Geometry.Rectangle([g["left"], g["top"] - g["height"] * g["res"],
                                  g["left"] + g["width"] * g["res"], g["top"]],
                                 ee.Projection(g["crs"]), False)


def s1_collection(ee, region, start, end, orbit=None):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING")))
    if orbit is not None:
        col = col.filter(ee.Filter.eq("relativeOrbitNumber_start", int(orbit)))
    return col


def choose_orbit(ee, region, months):
    y0, m0 = months[0]
    y1, m1 = months[-1]
    start = ee.Date.fromYMD(y0, m0, 1)
    end = ee.Date.fromYMD(y1, m1, 1).advance(1, "month")
    hist = s1_collection(ee, region, start, end).aggregate_histogram("relativeOrbitNumber_start").getInfo()
    if not hist:
        raise RuntimeError("No ascending Sentinel-1 IW VV+VH scenes found for this location and period.")
    hist = {int(float(k)): int(v) for k, v in hist.items()}
    return max(hist, key=hist.get), hist


def monthly_composite(ee, region, year, month, orbit):
    """Mean of linear backscatter (sigma0) over the month, converted back to dB; plus observation count."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    col = s1_collection(ee, region, start, end, orbit)
    n_images = col.size().getInfo()
    if n_images == 0:
        return None, 0
    def common_mask(im):                       # keep pixels only where BOTH VV and VH are valid (as in training)
        vvvh = im.select(["VV", "VH"])
        return vvvh.updateMask(vvvh.select("VV").mask().And(vvvh.select("VH").mask()))
    masked = col.map(common_mask)
    lin = masked.map(lambda im: ee.Image.constant(10).pow(im.divide(10)).rename(["VV", "VH"]))
    db = lin.mean().log10().multiply(10).rename(["VV", "VH"])
    n_obs = masked.select("VV").count().rename("n_obs")
    return ee.Image.cat([db, n_obs]).toFloat(), n_images


def fetch(ee, image, grid, bands, chunk=1024):
    """Download an Earth Engine image on our exact grid as a float32 array (bands, H, W); NaN = no data."""
    H, W, res = grid["height"], grid["width"], grid["res"]
    out = np.full((len(bands), H, W), np.nan, dtype=np.float32)
    img = image.select(bands).unmask(-9999)
    for r0 in range(0, H, chunk):
        for c0 in range(0, W, chunk):
            h, w = min(chunk, H - r0), min(chunk, W - c0)
            request = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
                       "grid": {"dimensions": {"width": w, "height": h},
                                "affineTransform": {"scaleX": res, "shearX": 0, "translateX": grid["left"] + c0 * res,
                                                    "shearY": 0, "scaleY": -res, "translateY": grid["top"] - r0 * res},
                                "crsCode": grid["crs"]}}
            arr = ee.data.computePixels(request)
            block = np.stack([np.asarray(arr[b], dtype=np.float32) for b in bands])
            block[block <= -9998] = np.nan
            out[:, r0:r0 + h, c0:c0 + w] = block
    return out


def to_40m(s1):
    """Aggregate a 10 m stack (VV dB, VH dB, n_obs) to 40 m: 4x4 mean in linear power -> dB (as for the training data)."""
    c, h, w = s1.shape
    b = s1[:, :h - h % 4, :w - w % 4].reshape(c, h // 4, 4, w // 4, 4)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lin = np.power(10.0, b[:2] / 10.0)
        db = 10.0 * np.log10(np.nanmean(lin, axis=(2, 4)))
        nobs = np.nanmean(b[2:3], axis=(2, 4))
    return np.concatenate([db, nobs]).astype(np.float32)


def block_mode(a, nodata):
    """Majority class of each 4x4 block (labels), ignoring no-data."""
    h, w = a.shape
    b = a[:h - h % 4, :w - w % 4].reshape(h // 4, 4, w // 4, 4)
    counts = np.stack([(b == k).sum(axis=(1, 3)) for k in range(NC)])
    out = counts.argmax(0).astype(np.uint8)
    out[counts.sum(0) == 0] = nodata
    return out


# ----------------------------------------------------------------------------- model
def load_model(weights_path, device):
    import torch, torch.nn as nn
    import segmentation_models_pytorch as smp
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    in_ch = len(cfg["inputs"])
    if cfg["arch"] == "segformer" and in_ch != 3:
        core = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"], encoder_weights=None, in_channels=3, classes=NC)
        model = nn.Sequential(nn.Conv2d(in_ch, 3, kernel_size=1), core)
    else:
        model = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"], encoder_weights=None,
                                 in_channels=in_ch, classes=NC)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def normalise(s1, inputs, stats):
    chans = []
    for name in inputs:
        c = {"VV": s1[0], "VH": s1[1], "RATIO": s1[0] - s1[1], "mean_n_obs": s1[2]}[name]
        st = stats[name]
        chans.append((np.clip(c, st["clip_low"], st["clip_high"]) - st["mean"]) / st["std"])
    return np.nan_to_num(np.stack(chans).astype(np.float32), nan=0.0)


def predict(model, x, device, tile=512, stride=384, batch=4):
    import torch
    import torch.nn.functional as F
    c, h, w = x.shape
    tile = min(tile, max(64, 32 * math.ceil(max(h, w) / 32)))
    stride = min(stride, tile)
    ph = max(tile, math.ceil((h - tile) / stride) * stride + tile)
    pw = max(tile, math.ceil((w - tile) / stride) * stride + tile)
    xp = np.pad(x, ((0, 0), (0, ph - h), (0, pw - w)), mode="reflect")
    prob = np.zeros((NC, ph, pw), dtype=np.float32)
    weight = np.zeros((ph, pw), dtype=np.float32)
    hw = np.hanning(tile + 2)[1:-1]
    win = np.maximum(np.outer(hw, hw), 1e-3).astype(np.float32)
    coords = [(r, q) for r in range(0, ph - tile + 1, stride) for q in range(0, pw - tile + 1, stride)]
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            cc = coords[i:i + batch]
            xb = torch.from_numpy(np.stack([xp[:, r:r + tile, q:q + tile] for r, q in cc])).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(xb)
            if out.shape[-2:] != xb.shape[-2:]:
                out = F.interpolate(out, size=xb.shape[-2:], mode="bilinear", align_corners=False)
            out = out.float().softmax(1).cpu().numpy()
            for k, (r, q) in enumerate(cc):
                prob[:, r:r + tile, q:q + tile] += out[k] * win
                weight[r:r + tile, q:q + tile] += win
    prob = prob[:, :h, :w] / weight[None, :h, :w]
    return prob.argmax(0).astype(np.uint8)


# ----------------------------------------------------------------------------- statistics
def class_statistics(maps, labels):
    """Percentage of the valid area per class and month; month-to-month and total change in pp."""
    rows = []
    for lab, a in zip(labels, maps):
        valid = a != NODATA
        cnt = np.bincount(a[valid], minlength=NC)[:NC] if valid.any() else np.zeros(NC)
        n = max(int(valid.sum()), 1)
        rows.append({"month": lab, **{c: 100.0 * cnt[i] / n for i, c in enumerate(CLASS_NAMES)}})
    pct = pd.DataFrame(rows).set_index("month")
    mom = pct.diff()
    total = (pct.iloc[-1] - pct.iloc[0]).rename(f"{labels[0]} → {labels[-1]} (pp)")
    return pct, mom, total


def annual_mode(maps):
    stack = np.stack(maps)
    counts = np.stack([(stack == c).sum(0) for c in range(NC)])
    mode = counts.argmax(0).astype(np.uint8)
    mode[counts.sum(0) == 0] = NODATA
    return mode


def compare_with_reference(pred, ref):
    m = (pred != NODATA) & (ref != NODATA)
    cm = np.bincount(ref[m].astype(np.int64) * NC + pred[m], minlength=NC * NC).reshape(NC, NC)
    tp, gt, pr = np.diag(cm), cm.sum(1), cm.sum(0)
    iou = np.where(gt + pr - tp > 0, tp / np.maximum(gt + pr - tp, 1), np.nan)
    table = pd.DataFrame({"prototype_%": 100 * pr / cm.sum(), "WorldCover_%": 100 * gt / cm.sum(),
                          "difference_pp": 100 * (pr - gt) / cm.sum(), "IoU": iou}, index=CLASS_NAMES)
    return {"agreement": float(tp.sum() / cm.sum()), "table": table, "confusion": cm}


# ----------------------------------------------------------------------------- outputs
def save_geotiff(path, arr, grid, tags=None):
    import rasterio
    from rasterio.transform import Affine
    profile = {"driver": "GTiff", "width": grid["width"], "height": grid["height"], "count": 1, "dtype": "uint8",
               "crs": grid["crs"], "transform": Affine(grid["res"], 0, grid["left"], 0, -grid["res"], grid["top"]),
               "nodata": NODATA, "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.write_colormap(1, {i: tuple(int(PALETTE[i][j:j + 2], 16) for j in (1, 3, 5)) + (255,) for i in range(NC)})
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def plot_maps(maps, labels, title, path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap, norm = ListedColormap(PALETTE), BoundaryNorm(np.arange(-0.5, NC + 0.5), NC)
    fig, axs = plt.subplots(3, 4, figsize=(18, 14))
    for ax, lab, a in zip(axs.ravel(), labels, maps):
        show = a.astype(float)
        show[a == NODATA] = np.nan
        ax.imshow(show, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(lab)
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in range(NC)]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=9, fontsize=10)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    if path:
        plt.savefig(path, dpi=110)
    return fig


def plot_shares(pct, total, path=None):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    for i, c in enumerate(CLASS_NAMES):
        axs[0].plot(pct.index, pct[c], marker="o", color=PALETTE[i], label=c)
    axs[0].set_ylabel("% of area"); axs[0].set_title("Class share per month")
    axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8, ncol=2); axs[0].tick_params(axis="x", rotation=45)
    axs[1].bar(CLASS_NAMES, total.values, color=PALETTE)
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_ylabel("percentage points"); axs[1].set_title(f"Total change {total.name}")
    axs[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=130)
    return fig


def plot_reference(mode_map, ref_map, cmp, path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap, norm = ListedColormap(PALETTE), BoundaryNorm(np.arange(-0.5, NC + 0.5), NC)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax, a, t in [(axs[0], mode_map, "Prototype: most frequent class over the 12 months"),
                     (axs[1], ref_map, "ESA WorldCover 2021 (mapped to the 9 classes)")]:
        show = a.astype(float)
        show[a == NODATA] = np.nan
        ax.imshow(show, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(t); ax.axis("off")
    cmp["table"][["prototype_%", "WorldCover_%"]].plot.bar(ax=axs[2], color=["#555555", "#bbbbbb"])
    axs[2].set_ylabel("% of area")
    axs[2].set_title(f"Class shares — pixel agreement {100 * cmp['agreement']:.1f} %")
    axs[2].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=120)
    return fig


# ----------------------------------------------------------------------------- full pipeline
def run(lat, lon, start_month, weights, project, size_km=10.0, out_dir="results",
        device=None, orbit=None, worldcover=True, verbose=True, input_res=10):
    import torch
    t0 = time.time()
    months = month_list(start_month)
    labels = [f"{y}-{m:02d}" for y, m in months]
    grid10 = make_grid(lat, lon, size_km)                      # Sentinel-1 is always retrieved at 10 m
    if input_res == 40:                                        # ... and aggregated 4x4 for the 40 m mode
        grid = dict(grid10, res=40, width=grid10["width"] // 4, height=grid10["height"] // 4)
    else:
        grid = grid10
    name = f"lat{lat:.4f}_lon{lon:.4f}_{start_month}" + ("_40m" if input_res == 40 else "")
    out = os.path.join(out_dir, name)
    os.makedirs(out, exist_ok=True)
    log(f"AOI {grid['size_km']:.2f} km x {grid['size_km']:.2f} km around ({lat}, {lon}), {grid['crs']}, "
        f"{grid['width']}x{grid['height']} px | months {labels[0]} … {labels[-1]}", verbose)

    ee = init_ee(project)
    region = grid_region(ee, grid)
    if orbit is None:
        orbit, hist = choose_orbit(ee, region, months)
        log(f"ascending relative orbits found (scenes): {hist} -> using orbit {orbit}", verbose)
    else:
        hist = {}

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(weights, device)
    log(f"model {cfg['arch']} / {cfg['encoder']} (trained at {cfg['train_res']} m) on {device}; "
        f"input resolution {input_res} m ({grid['width']}x{grid['height']} px)", verbose)

    maps, n_images = [], []
    for (y, m), lab in zip(months, labels):
        img, n = monthly_composite(ee, region, y, m, orbit)
        n_images.append(n)
        if img is None:
            log(f"  {lab}: no Sentinel-1 scene -> map left empty", verbose)
            maps.append(np.full((grid["height"], grid["width"]), NODATA, np.uint8))
            continue
        s1 = fetch(ee, img, grid10, ["VV", "VH", "n_obs"])
        if input_res == 40:
            s1 = to_40m(s1)
        valid = np.isfinite(s1[:2]).all(0)
        pred = predict(model, normalise(s1, cfg["inputs"], cfg["stats"]), device)
        pred[~valid] = NODATA
        maps.append(pred)
        save_geotiff(os.path.join(out, f"LULC_{lab}.tif"), pred, grid, {"month": lab, "orbit": orbit, "images": n})
        log(f"  {lab}: {n} scene(s), valid {100 * valid.mean():.1f} %, mapped", verbose)

    pct, mom, total = class_statistics(maps, labels)
    pct.round(3).to_csv(os.path.join(out, "class_percent_per_month.csv"))
    mom.round(3).to_csv(os.path.join(out, "month_to_month_change_pp.csv"))
    total.round(3).to_csv(os.path.join(out, "total_change_pp.csv"))
    fig_maps = plot_maps(maps, labels, f"Sentinel-1 LULC — ({lat:.4f}, {lon:.4f}) — orbit {orbit} ascending",
                         os.path.join(out, "maps_12_months.png"))
    fig_shares = plot_shares(pct, total, os.path.join(out, "class_shares.png"))

    result = {"grid": grid, "labels": labels, "maps": maps, "pct": pct, "mom": mom, "total": total,
              "orbit": orbit, "orbit_histogram": hist, "n_images": n_images, "out_dir": out,
              "figures": {"maps": fig_maps, "shares": fig_shares}}

    if worldcover:
        wc = fetch(ee, ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").toFloat(), grid10, ["Map"])[0]
        ref = np.full(wc.shape, NODATA, np.uint8)
        for code, cls in WORLDCOVER_TO_DW.items():
            ref[wc == code] = cls
        if input_res == 40:
            ref = block_mode(ref, NODATA)
        mode = annual_mode([a for a in maps if (a != NODATA).any()])
        cmp = compare_with_reference(mode, ref)
        cmp["table"].round(3).to_csv(os.path.join(out, "worldcover_comparison.csv"))
        save_geotiff(os.path.join(out, "LULC_annual_mode.tif"), mode, grid)
        save_geotiff(os.path.join(out, "WorldCover2021_as_9class.tif"), ref, grid)
        result.update({"annual_mode": mode, "worldcover": ref, "worldcover_cmp": cmp})
        result["figures"]["worldcover"] = plot_reference(mode, ref, cmp, os.path.join(out, "worldcover_check.png"))
        log(f"WorldCover check: pixel agreement {100 * cmp['agreement']:.1f} %", verbose)

    info = {"lat": lat, "lon": lon, "start_month": start_month, "input_res_m": input_res, "months": labels, "grid": grid,
            "orbit": orbit, "orbit_histogram": hist, "images_per_month": dict(zip(labels, n_images)),
            "model": {k: cfg[k] for k in ("arch", "encoder", "train_res", "inputs")},
            "runtime_min": round((time.time() - t0) / 60, 2)}
    json.dump(info, open(os.path.join(out, "run_info.json"), "w"), indent=2)
    log(f"done in {info['runtime_min']} min -> {out}", verbose)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sentinel-1 monthly LULC prototype")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start", required=True, help="start month YYYY-MM")
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--weights", default="model/best_model.pt")
    ap.add_argument("--size-km", type=float, default=10.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--orbit", type=int, default=None, help="force a relative orbit (default: most frequent)")
    ap.add_argument("--no-worldcover", action="store_true")
    ap.add_argument("--input-res", type=int, default=10, choices=[10, 40],
                    help="40: aggregate Sentinel-1 to 40 m (use a model trained at 40 m or mixed)")
    a = ap.parse_args()
    r = run(a.lat, a.lon, a.start, a.weights, a.project, a.size_km, a.out, orbit=a.orbit,
            worldcover=not a.no_worldcover, input_res=a.input_res)
    pd.set_option("display.width", 200)
    print("\nClass share per month (%):\n", r["pct"].round(2).to_string())
    print("\nMonth-to-month change (pp):\n", r["mom"].round(2).to_string())
    print("\nTotal change (pp):\n", r["total"].round(2).to_string())
