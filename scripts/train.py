#!/usr/bin/env python3
"""
Train one Sentinel-1 LULC segmentation model (Heidelberg 2025, 9 Dynamic World classes).

Examples
--------
Smoke test (1 short epoch):
    python3 train.py --arch unet --train-res 10 --epochs 1 --max-batches 20 --run-name smoke_unet_10

Full runs:
    python3 train.py --arch unet          --train-res 10
    python3 train.py --arch deeplabv3plus --train-res 40
    python3 train.py --arch segformer     --train-res mix      # mixed 10 m + 40 m training

Outputs
-------
models/<run>/best.pt, last.pt, config.json, summary.json
logs/<run>/history.csv
"""
import argparse, json, os, random, site, sys, time

PROJ = "/home/jovyan/vault/heidelberg_lulc_2025"
site.addsitedir(f"{PROJ}/pylibs/lib/python3.11/site-packages")
os.environ.setdefault("HF_HOME", f"{PROJ}/cache/huggingface")      # keep downloaded weights in vault
os.environ.setdefault("TORCH_HOME", f"{PROJ}/cache/torch")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

CLASS_NAMES = ["Water", "Trees", "Grass", "Flooded vegetation", "Crops",
               "Shrub and scrub", "Built-up", "Bare ground", "Snow and ice"]
NC, IGNORE = 9, 255
PATCH_DIR = f"{PROJ}/patches"
DEFAULT_ENCODER = {"unet": "resnet34", "deeplabv3plus": "resnet34",
                   "fpn": "resnet34", "segformer": "mit_b2"}
DEFAULTS = {  # epochs, train batch (10 m), train batch (40 m)
    "10": dict(epochs=50), "40": dict(epochs=80), "mix": dict(epochs=50)}
BATCH = {"10": 16, "40": 64}
VAL_BATCH = {"10": 32, "40": 128}


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=list(DEFAULT_ENCODER))
    p.add_argument("--train-res", required=True, choices=["10", "40", "mix"])
    p.add_argument("--val-res", default=None, choices=["10", "40", "both"],
                   help="resolution used for model selection (default: same as training; 'both' for mix)")
    p.add_argument("--inputs", default="VV,VH,RATIO", help="comma list from VV,VH,RATIO,mean_n_obs")
    p.add_argument("--encoder", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-10", type=int, default=BATCH["10"])
    p.add_argument("--batch-40", type=int, default=BATCH["40"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12, help="early stopping (epochs without improvement)")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 = main process; container has tiny shared memory)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0, help="limit batches per epoch (smoke tests)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--patch-dir", default=PATCH_DIR, help="patch folder (default: patches/; e.g. patches_lee/)")
    p.add_argument("--scheduler", default="onecycle", choices=["onecycle", "cosine", "plateau"],
                   help="onecycle = original runs; cosine = cosine decay with 1 warm-up epoch; "
                        "plateau = ReduceLROnPlateau on validation mIoU")
    p.add_argument("--aug", default="geom", choices=["geom", "geom_rad", "none"],
                   help="geom = flips + 90° rotations (original runs); geom_rad = geom + radiometric "
                        "(±1 dB per-channel offset); none = no augmentation")
    p.add_argument("--class-weight", default="sqrt", choices=["sqrt", "none"],
                   help="cross-entropy class weights: sqrt = inverse square-root frequency (baseline), none = uniform")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="TRAINING ONLY: ignore label pixels whose Dynamic World confidence is below this value "
                        "(0 = baseline). Validation and test labels are never changed.")
    p.add_argument("--conf-weight", action="store_true",
                   help="TRAINING ONLY: weight each pixel's cross-entropy by its Dynamic World confidence "
                        "(all labels kept). Validation/test unchanged.")
    p.add_argument("--train-eval-n", type=int, default=240,
                   help="number of training patch pairs scored each epoch in eval mode without augmentation "
                        "(0 = off). Only for monitoring; does not change training.")
    p.add_argument("--no-pretrain", action="store_true",
                   help="train the encoder from random weights instead of ImageNet pre-training")
    p.add_argument("--resume", action="store_true", help="continue from models/<run>/last.pt")
    a = p.parse_args()
    a.encoder = a.encoder or DEFAULT_ENCODER[a.arch]
    a.val_res = a.val_res or ("both" if a.train_res == "mix" else a.train_res)
    a.epochs = a.epochs or DEFAULTS[a.train_res]["epochs"]
    a.inputs = [s.strip() for s in a.inputs.split(",") if s.strip()]
    for s in a.inputs:
        assert s in ("VV", "VH", "RATIO", "mean_n_obs", "MSIN", "MCOS"), f"unknown input {s}"
    a.run_name = a.run_name or f"{a.arch}_train{a.train_res}_{'-'.join(a.inputs)}_s{a.seed}"
    return a


def log(*msg):
    print(time.strftime("%H:%M:%S"), *msg, flush=True)


# ----------------------------------------------------------------------------- shared statistics
def shared_stats():
    """Normalisation + class weights computed ONCE from the 10 m training split; used by every experiment."""
    path = f"{PATCH_DIR}/norm_stats_shared.json"
    if os.path.exists(path):
        return json.load(open(path))
    base = json.load(open(f"{PATCH_DIR}/10m/norm_stats_train.json"))
    X = np.load(f"{PATCH_DIR}/10m/train_X.npy", mmap_mode="r")
    Y = np.load(f"{PATCH_DIR}/10m/train_Y.npy", mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(X), min(300, len(X)), replace=False))
    s = np.asarray(X[idx])
    r = (s[:, 0] - s[:, 1]).ravel()
    r = r[np.isfinite(r)]
    lo, hi = np.percentile(r, [0.5, 99.5])
    rc = np.clip(r, lo, hi)
    base["RATIO"] = {"clip_low": float(lo), "clip_high": float(hi), "mean": float(rc.mean()), "std": float(rc.std())}

    counts = np.zeros(256, dtype=np.int64)
    for i in range(0, len(Y), 64):
        counts += np.bincount(np.asarray(Y[i:i + 64]).ravel(), minlength=256)
    freq = counts[:NC] / counts[:NC].sum()
    w = np.where(freq > 0, 1.0 / np.sqrt(np.maximum(freq, 1e-12)), 0.0)
    w = w / w[freq > 0].mean()
    w = np.clip(w, 0.1, 10.0)
    base["class_freq_10m_train"] = freq.tolist()
    base["class_weights"] = w.tolist()
    base["note"] = "computed from the 10 m TRAIN split only; shared by all experiments"
    json.dump(base, open(path, "w"), indent=2)
    return base


def common_indices(split):
    """Patch indices of the (month, block) pairs present at BOTH resolutions, in the same order."""
    m10 = pd.read_csv(f"{PATCH_DIR}/10m/{split}_meta.csv")
    m40 = pd.read_csv(f"{PATCH_DIR}/40m/{split}_meta.csv")
    keys = sorted(set(zip(m10.month, m10.block_id)) & set(zip(m40.month, m40.block_id)))
    i10 = m10.set_index(["month", "block_id"]).loc[keys, "patch_index"].to_numpy()
    i40 = m40.set_index(["month", "block_id"]).loc[keys, "patch_index"].to_numpy()
    return {"10": i10, "40": i40}


# ----------------------------------------------------------------------------- data
class PatchSet(Dataset):
    def __init__(self, res, split, indices, stats, inputs, augment, radiometric=False, min_conf=0.0,
                 return_conf=False):
        self.X = np.load(f"{PATCH_DIR}/{res}m/{split}_X.npy", mmap_mode="r")
        self.Y = np.load(f"{PATCH_DIR}/{res}m/{split}_Y.npy", mmap_mode="r")
        self.idx = np.asarray(indices)
        self.inputs = inputs
        self.norm = [(-1.0, 1.0, 0.0, 1.0) if n in ("MSIN", "MCOS") else
                     (stats[n]["clip_low"], stats[n]["clip_high"], stats[n]["mean"], stats[n]["std"]) for n in inputs]
        self.month = None
        if any(n in ("MSIN", "MCOS") for n in inputs):          # season encoding: month of each patch
            meta = pd.read_csv(f"{PATCH_DIR}/{res}m/{split}_meta.csv").set_index("patch_index")
            self.month = meta["month"]
        self.min_conf, self.A, self.return_conf = min_conf, None, return_conf
        if min_conf > 0 or return_conf:                          # label confidence (training only)
            info = json.load(open(f"{PATCH_DIR}/{res}m/dataset_info.json"))
            self.A = np.load(f"{PATCH_DIR}/{res}m/{split}_AUX.npy", mmap_mode="r")
            self.i_conf = info["aux"].index("confidence")
        self.augment = augment
        self.radiometric = radiometric

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = np.array(self.X[j], dtype=np.float32)                 # copy: the memmap is read-only
        y = np.asarray(self.Y[j]).astype(np.int64)
        w = None
        if self.A is not None:
            conf = np.nan_to_num(np.asarray(self.A[j, self.i_conf], dtype=np.float32), nan=0.0)
            if self.min_conf > 0:
                y[conf < self.min_conf] = IGNORE
            if self.return_conf:
                w = conf
        if self.augment and self.radiometric:                      # small calibration/seasonal level shift
            x[0] += np.random.uniform(-1.0, 1.0)                   # VV offset in dB
            x[1] += np.random.uniform(-1.0, 1.0)                   # VH offset in dB (RATIO follows)
        chans = []
        for name, (lo, hi, mu, sd) in zip(self.inputs, self.norm):
            if name in ("MSIN", "MCOS"):
                ang = 2 * np.pi * (int(self.month.loc[j]) - 1) / 12
                c = np.full(x[0].shape, np.sin(ang) if name == "MSIN" else np.cos(ang), dtype=np.float32)
            else:
                c = {"VV": x[0], "VH": x[1], "RATIO": x[0] - x[1], "mean_n_obs": x[2]}[name]
            chans.append((np.clip(c, lo, hi) - mu) / sd)
        x = np.stack(chans).astype(np.float32)
        if self.augment:                                     # geometric only: flips + 90° rotations
            k = np.random.randint(4)
            flip = np.random.rand() < 0.5
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if w is not None:
                w = np.rot90(w, k)
            if flip:
                x, y = x[:, :, ::-1], y[:, ::-1]
                if w is not None:
                    w = w[:, ::-1]
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        if w is not None:
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(np.ascontiguousarray(w))
        return torch.from_numpy(x), torch.from_numpy(y)


def worker_init(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def make_loader(ds, batch, train, workers):
    return DataLoader(ds, batch_size=batch, shuffle=train, drop_last=train, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, worker_init_fn=worker_init)


# ----------------------------------------------------------------------------- model / metrics
def build_model(arch, encoder, in_ch, pretrained=True):
    w = "imagenet" if pretrained else None      # --no-pretrain: random initialisation (fine-tuning experiment)
    if arch == "segformer" and in_ch != 3:      # MiT encoders expect 3 channels -> learnable 1x1 adapter
        core = smp.create_model(arch, encoder_name=encoder, encoder_weights=w, in_channels=3, classes=NC)
        return nn.Sequential(nn.Conv2d(in_ch, 3, kernel_size=1), core)
    return smp.create_model(arch, encoder_name=encoder, encoder_weights=w, in_channels=in_ch, classes=NC)


def fit_size(out, y):
    if out.shape[-2:] != y.shape[-2:]:
        out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
    return out


def update_cm(cm, logits, y):
    p = logits.argmax(1)
    m = y != IGNORE
    cm += torch.bincount(y[m] * NC + p[m], minlength=NC * NC).reshape(NC, NC)


def cm_metrics(cm):
    cm = cm.double()
    tp, gt, pr = cm.diag(), cm.sum(1), cm.sum(0)
    iou = tp / (gt + pr - tp).clamp(min=1)
    f1 = 2 * tp / (gt + pr).clamp(min=1)
    present = gt > 0
    return {"oa": (tp.sum() / cm.sum().clamp(min=1)).item(),
            "miou": iou[present].mean().item(), "mf1": f1[present].mean().item(),
            "iou": iou.tolist(), "f1": f1.tolist(), "support": gt.long().tolist()}


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, criterion=None):
    """Scores a loader in eval mode without gradients; optionally also returns the mean loss."""
    model.eval()
    cm = torch.zeros(NC, NC, dtype=torch.long, device=device)
    loss_sum, n_batches = 0.0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out = fit_size(out, y).float()
        update_cm(cm, out, y)
        if criterion is not None:
            loss_sum += criterion(out, y).item(); n_batches += 1
    m = cm_metrics(cm)
    if criterion is not None:
        m["loss"] = loss_sum / max(n_batches, 1)
    return m


# ----------------------------------------------------------------------------- main
def main():
    global PATCH_DIR
    a = parse_args()
    PATCH_DIR = a.patch_dir
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{a.gpu}")
    free, total = torch.cuda.mem_get_info(device)
    log(f"run {a.run_name} | GPU {a.gpu} {torch.cuda.get_device_name(device)} | free {free/1e9:.1f} / {total/1e9:.1f} GB")

    mdir, ldir = f"{PROJ}/models/{a.run_name}", f"{PROJ}/logs/{a.run_name}"
    os.makedirs(mdir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    stats = shared_stats()

    # data
    idx = {s: common_indices(s) for s in ("train", "val")}
    train_res = ["10", "40"] if a.train_res == "mix" else [a.train_res]
    val_res = ["10", "40"] if a.val_res == "both" else [a.val_res]
    bsz = {"10": a.batch_10, "40": a.batch_40}
    train_loaders = {r: make_loader(PatchSet(r, "train", idx["train"][r], stats, a.inputs, a.aug != "none",
                                             radiometric=a.aug == "geom_rad", min_conf=a.min_conf,
                                             return_conf=a.conf_weight),
                                    bsz[r], True, a.workers) for r in train_res}
    train_eval_loaders = {}
    if a.train_eval_n:
        sel = np.sort(np.random.default_rng(0).choice(len(idx["train"]["10"]),
                                                      min(a.train_eval_n, len(idx["train"]["10"])), replace=False))
        train_eval_loaders = {r: make_loader(PatchSet(r, "train", idx["train"][r][sel], stats, a.inputs, False),
                                             VAL_BATCH[r], False, a.workers) for r in val_res}
    val_loaders = {r: make_loader(PatchSet(r, "val", idx["val"][r], stats, a.inputs, False),
                                  VAL_BATCH[r], False, a.workers) for r in val_res}
    main_res = "10" if a.train_res == "mix" else a.train_res      # epoch length follows this loader
    steps = len(train_loaders[main_res])
    if a.max_batches:
        steps = min(steps, a.max_batches)
    log(f"train res {train_res} ({len(idx['train']['10'])} patch pairs) | val res {val_res} "
        f"({len(idx['val']['10'])}) | inputs {a.inputs} | {steps} steps/epoch | {a.epochs} epochs")

    # model, loss, optimiser
    model = build_model(a.arch, a.encoder, len(a.inputs), pretrained=not a.no_pretrain).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"model {a.arch} / {a.encoder}: {n_par:.1f} M parameters")
    cw = stats["class_weights"] if a.class_weight == "sqrt" else [1.0] * NC
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device), ignore_index=IGNORE)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    if a.scheduler == "onecycle":       # original runs: warm-up 10 %, cosine annealing over ALL planned epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps, pct_start=0.1)
    elif a.scheduler == "cosine":       # 1 warm-up epoch, then cosine decay to 1 % of the learning rate
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(a.epochs - 1) * steps, eta_min=a.lr * 0.01)],
            milestones=[steps])
    else:                               # halve the learning rate when validation mIoU stalls for 4 epochs
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    step_per_batch = a.scheduler != "plateau"

    start_epoch, best, bad_epochs, history = 1, -1.0, 0, []
    if a.resume and os.path.exists(f"{mdir}/last.pt"):
        ck = torch.load(f"{mdir}/last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best, bad_epochs = ck["epoch"] + 1, ck["best"], ck["bad_epochs"]
        if os.path.exists(f"{ldir}/history.csv"):
            history = pd.read_csv(f"{ldir}/history.csv").to_dict("records")
        log(f"resumed from epoch {ck['epoch']} (best {best:.4f})")

    config = {**vars(a), "stats": stats, "class_names": CLASS_NAMES, "n_params_M": n_par,
              "n_train_pairs": int(len(idx["train"]["10"])), "n_val_pairs": int(len(idx["val"]["10"])),
              "loss": "weighted CE + Dice (ignore 255)", "optimizer": f"AdamW + {a.scheduler}",
              "augmentation": {"geom": "random 90° rotations + horizontal flip",
                               "geom_rad": "random 90° rotations + horizontal flip + ±1 dB VV/VH offsets",
                               "none": "none"}[a.aug], "amp": "bfloat16"}
    json.dump(config, open(f"{mdir}/config.json", "w"), indent=2)

    def criterion(out, y):              # same loss for training, validation loss and training-subset loss
        return ce(out, y) + dice(out, y)

    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)

    def loss_fn(x, y, w=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out = fit_size(out, y).float()
        if w is None:
            return criterion(out, y)
        valid = y != IGNORE
        nll = F.cross_entropy(out, y, weight=cw_t, ignore_index=IGNORE, reduction="none")   # = cw[y] * -log p
        denom = (cw_t[y.clamp(max=NC - 1)] * w)[valid].sum().clamp(min=1e-6)
        return (nll * w)[valid].sum() / denom + dice(out, y)

    def unpack(batch):
        if len(batch) == 3:
            return (batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True),
                    batch[2].to(device, non_blocking=True))
        return batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True), None

    t_start = time.time()
    for epoch in range(start_epoch, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        it_main = iter(train_loaders[main_res])
        it_40 = iter(train_loaders["40"]) if a.train_res == "mix" else None
        for step in range(steps):
            loss = loss_fn(*unpack(next(it_main)))
            if it_40 is not None:                                  # mixed: add one 40 m batch per step
                try:
                    b40 = next(it_40)
                except StopIteration:
                    it_40 = iter(train_loaders["40"]); b40 = next(it_40)
                loss = 0.5 * (loss + loss_fn(*unpack(b40)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step_per_batch:
                sched.step()
            losses.append(loss.item())

        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": float(np.mean(losses)),
               "train_sec": round(time.time() - t0, 1)}
        scores = []
        for r, vl in val_loaders.items():
            m = evaluate(model, vl, device, a.max_batches, criterion)
            scores.append(m["miou"])
            row.update({f"val{r}_loss": m["loss"], f"val{r}_oa": m["oa"], f"val{r}_miou": m["miou"],
                        f"val{r}_mf1": m["mf1"]})
            row.update({f"val{r}_iou_{c}": v for c, v in enumerate(m["iou"])})
            row.update({f"val{r}_f1_{c}": v for c, v in enumerate(m["f1"])})
        for r, tl in train_eval_loaders.items():       # training subset, eval mode, no augmentation
            m = evaluate(model, tl, device, a.max_batches, criterion)
            row.update({f"trainEval{r}_loss": m["loss"], f"trainEval{r}_oa": m["oa"],
                        f"trainEval{r}_miou": m["miou"], f"trainEval{r}_mf1": m["mf1"]})
        row["select_score"] = float(np.mean(scores))
        if not step_per_batch:
            sched.step(row["select_score"])
        row["epoch_sec"] = round(time.time() - t0, 1)
        history.append(row)
        pd.DataFrame(history).to_csv(f"{ldir}/history.csv", index=False)

        improved = row["select_score"] > best
        if improved:
            best, bad_epochs = row["select_score"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": best, "config": config}, f"{mdir}/best.pt")
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad_epochs": bad_epochs}, f"{mdir}/last.pt")

        val_txt = " | ".join(f"val{r} mIoU {row[f'val{r}_miou']:.4f} OA {row[f'val{r}_oa']:.4f}" for r in val_res)
        tr_txt = " | ".join(f"trainEval{r} mIoU {row[f'trainEval{r}_miou']:.4f}" for r in train_eval_loaders)
        log(f"epoch {epoch:3d}/{a.epochs} | lr {row['lr']:.2e} | loss {row['train_loss']:.4f} | {val_txt} | {tr_txt} | "
            f"{row['epoch_sec']:.0f}s {'*best*' if improved else ''}")
        if bad_epochs >= a.patience:
            log(f"early stopping: no improvement for {a.patience} epochs")
            break

    summary = {"run": a.run_name, "best_select_score": best,
               "best_epoch": int(max(history, key=lambda r: r["select_score"])["epoch"]),
               "epochs_run": len(history), "minutes": round((time.time() - t_start) / 60, 1),
               "peak_gpu_mem_GB": round(torch.cuda.max_memory_allocated(device) / 1e9, 2)}
    json.dump(summary, open(f"{mdir}/summary.json", "w"), indent=2)
    log("DONE", json.dumps(summary))


if __name__ == "__main__":
    main()
