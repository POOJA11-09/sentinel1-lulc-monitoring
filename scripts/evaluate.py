#!/usr/bin/env python3
"""
Evaluate trained models on the TEST split (run once, after training is finished).

For every run in models/ (except smoke_* tests) three evaluations are made:
    test10        : 10 m input  -> prediction compared with 10 m labels
    test40        : 40 m input  -> prediction compared with 40 m labels
    test40_from10 : 10 m input  -> prediction aggregated by 4x4 majority -> compared with 40 m labels

Usage:
    python3 scripts/evaluate.py                 # all finished runs
    python3 scripts/evaluate.py --runs unet_train10_VV-VH-RATIO_s42

Outputs (outputs/evaluation/):
    <run>/metrics.json, <run>/confusion_<mode>.csv, <run>/monthly_<mode>.csv
    summary_overall.csv   (OA, mIoU, mean F1 per run and mode)
    summary_per_class.csv (IoU and F1 per class, run and mode)
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T                       # shared data / statistics / metric code (no training is started)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

PROJ = T.PROJ
EVAL_DIR = f"{PROJ}/outputs/evaluation"
NC, IGNORE = T.NC, T.IGNORE


def load_model(run, device):
    ck = torch.load(f"{PROJ}/models/{run}/best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    in_ch = len(cfg["inputs"])
    if cfg["arch"] == "segformer" and in_ch != 3:
        core = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"], encoder_weights=None,
                                in_channels=3, classes=NC)
        model = nn.Sequential(nn.Conv2d(in_ch, 3, kernel_size=1), core)
    else:
        model = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"], encoder_weights=None,
                                 in_channels=in_ch, classes=NC)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, int(ck["epoch"])


def test_months(split="test"):
    m10 = pd.read_csv(f"{T.PATCH_DIR}/10m/{split}_meta.csv")
    m40 = pd.read_csv(f"{T.PATCH_DIR}/40m/{split}_meta.csv")
    keys = sorted(set(zip(m10.month, m10.block_id)) & set(zip(m40.month, m40.block_id)))
    return np.array([k[0] for k in keys])           # same order as T.common_indices("test")


@torch.no_grad()
def predict(model, ds, device, batch):
    out_all = []
    for i in range(0, len(ds), batch):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out = T.fit_size(out, x[:, 0]).float()
        out_all.append(out.argmax(1).to(torch.uint8).cpu())
    return torch.cat(out_all).numpy()


def confusion(y, p):
    m = y != IGNORE
    return np.bincount(y[m].astype(np.int64) * NC + p[m].astype(np.int64),
                       minlength=NC * NC).reshape(NC, NC)


def majority4(p):
    n, h, w = p.shape
    b = p.reshape(n, h // 4, 4, w // 4, 4)
    counts = np.stack([(b == c).sum(axis=(2, 4)) for c in range(NC)], axis=1)
    return counts.argmax(1).astype(np.uint8)


def metrics(cm):
    return T.cm_metrics(torch.tensor(cm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--include-unfinished", action="store_true")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="patch split to evaluate (default: test)")
    ap.add_argument("--out-subdir", default="",
                    help="write results to outputs/evaluation/<subdir>/ (e.g. 'ablation') so the main tables stay unchanged")
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    global EVAL_DIR
    if a.out_subdir:
        EVAL_DIR = f"{EVAL_DIR}/{a.out_subdir}"
    if a.split != "test":
        EVAL_DIR = f"{EVAL_DIR}/{a.split}"
    os.makedirs(EVAL_DIR, exist_ok=True)

    if a.runs:
        runs = a.runs
    else:
        runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{PROJ}/models/*/best.pt"))
        runs = [r for r in runs if not r.startswith("smoke_")]
        if not a.include_unfinished:
            runs = [r for r in runs if os.path.exists(f"{PROJ}/models/{r}/summary.json")]
    print("Runs to evaluate:", runs, flush=True)

    overall_rows, class_rows = [], []
    default_patch_dir = T.PATCH_DIR

    for run in runs:
        model, cfg, best_epoch = load_model(run, device)
        inputs = cfg["inputs"]
        T.PATCH_DIR = cfg.get("patch_dir") or default_patch_dir      # e.g. speckle-filtered patches
        stats = cfg.get("stats") or T.shared_stats()                 # statistics the model was trained with
        idx = T.common_indices(a.split)
        months = test_months(a.split)
        ds = {r: T.PatchSet(r, a.split, idx[r], stats, inputs, augment=False) for r in ("10", "40")}
        y = {r: np.asarray(ds[r].Y[ds[r].idx]) for r in ("10", "40")}

        pred10 = predict(model, ds["10"], device, batch=16)
        pred40 = predict(model, ds["40"], device, batch=64)
        modes = {"test10": (y["10"], pred10),
                 "test40": (y["40"], pred40),
                 "test40_from10": (y["40"], majority4(pred10))}

        out_dir = f"{EVAL_DIR}/{run}"
        os.makedirs(out_dir, exist_ok=True)
        result = {"run": run, "arch": cfg["arch"], "encoder": cfg["encoder"], "train_res": cfg["train_res"],
                  "val_res": cfg["val_res"], "inputs": inputs, "best_epoch": best_epoch,
                  "split": a.split, "n_pairs": int(len(idx["10"])), "modes": {}}

        for mode, (yy, pp) in modes.items():
            cm = confusion(yy, pp)
            m = metrics(cm)
            result["modes"][mode] = m
            pd.DataFrame(cm, index=[f"true_{c}" for c in T.CLASS_NAMES],
                         columns=[f"pred_{c}" for c in T.CLASS_NAMES]).to_csv(f"{out_dir}/confusion_{mode}.csv")

            monthly = []
            for mo in range(1, 13):
                sel = months == mo
                if sel.any():
                    mm = metrics(confusion(yy[sel], pp[sel]))
                    monthly.append({"month": mo, "patches": int(sel.sum()), "oa": mm["oa"],
                                    "miou": mm["miou"], "mf1": mm["mf1"]})
            pd.DataFrame(monthly).to_csv(f"{out_dir}/monthly_{mode}.csv", index=False)

            overall_rows.append({"run": run, "arch": cfg["arch"], "train_res": cfg["train_res"], "mode": mode,
                                 "best_epoch": best_epoch, "OA": m["oa"], "mIoU": m["miou"], "mF1": m["mf1"]})
            for c in range(NC):
                class_rows.append({"run": run, "arch": cfg["arch"], "train_res": cfg["train_res"], "mode": mode,
                                   "class": T.CLASS_NAMES[c], "IoU": m["iou"][c], "F1": m["f1"][c],
                                   "support_px": m["support"][c]})
            print(f"{run:40s} {mode:14s} OA {m['oa']:.4f}  mIoU {m['miou']:.4f}  mF1 {m['mf1']:.4f}", flush=True)

        json.dump(result, open(f"{out_dir}/metrics.json", "w"), indent=2)
        del model
        torch.cuda.empty_cache()

    # merge with earlier results so runs can be evaluated in several sessions
    for name, rows in (("summary_overall.csv", overall_rows), ("summary_per_class.csv", class_rows)):
        path = f"{EVAL_DIR}/{name}"
        new = pd.DataFrame(rows)
        if os.path.exists(path):
            old = pd.read_csv(path)
            new = pd.concat([old[~old.run.isin(new.run.unique())], new], ignore_index=True)
        new.to_csv(path, index=False)
    print("\nSaved:", f"{EVAL_DIR}/summary_overall.csv", "and summary_per_class.csv", flush=True)


if __name__ == "__main__":
    main()
