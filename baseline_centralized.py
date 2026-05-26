import os
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from model   import (MultiTaskBreastModel, MultiTaskLoss, all_metrics)
from dataset import BreastCancerDataset


# ════════════════════════════════════════════════════════════════
# 1.  Logger
# ════════════════════════════════════════════════════════════════
class CentralLogger:

    def __init__(self, out_dir):
        self.out_dir  = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.out_dir / f"central_log_{self.run_id}.txt"
        self.csv_path = self.out_dir / "central_metrics.csv"
        self.history  = []

        self._csv_file = open(self.csv_path, "w",
                              newline="", encoding="utf-8")
        self._writer   = csv.DictWriter(self._csv_file, fieldnames=[
            "epoch",
            "train_loss", "train_dice", "train_iou",
            "train_accuracy", "train_sensitivity", "train_specificity",
            "val_loss",   "val_dice",   "val_iou",
            "val_accuracy",   "val_sensitivity",   "val_specificity",
        ])
        self._writer.writeheader()
        self._log("Centralized Training Started")
        self._log(f"Datasets : {list(C.CLIENT_MAP.values())}")
        self._log("All data pooled — no federation")
        self._log("=" * 65)

    def _log(self, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_epoch(self, epoch, train_m, val_m, is_best=False):
        tag = "  ★ NEW BEST" if is_best else ""
        self._log(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_m['loss']:.4f} "
            f"train_dice={train_m['dice']:.4f} | "
            f"val_loss={val_m['loss']:.4f} "
            f"val_dice={val_m['dice']:.4f} "
            f"val_iou={val_m['iou']:.4f} "
            f"val_acc={val_m['accuracy']:.4f} "
            f"val_sens={val_m['sensitivity']:.4f} "
            f"val_spec={val_m['specificity']:.4f}"
            f"{tag}")

        row = {"epoch": epoch}
        for k, v in train_m.items():
            row[f"train_{k}"] = round(v, 6)
        for k, v in val_m.items():
            row[f"val_{k}"] = round(v, 6)

        self._writer.writerow(row)
        self._csv_file.flush()
        self.history.append(row)

    def save_summary(self, best_metrics):
        summary = {
            "run_id"       : self.run_id,
            "method"       : "Centralized (upper bound)",
            "datasets"     : list(C.CLIENT_MAP.values()),
            "best_metrics" : best_metrics,
            "history"      : self.history,
        }
        path = self.out_dir / f"central_summary_{self.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self._log(f"Summary saved → {path}")
        self._csv_file.close()
        return summary


# ════════════════════════════════════════════════════════════════
# 2.  Pooled DataLoader
# ════════════════════════════════════════════════════════════════
def build_pooled_loader(split, batch_size=C.BATCH_SIZE,
                        image_size=C.IMAGE_SIZE):
    datasets, all_labels = [], []
    for name in C.CLIENT_MAP.values():
        try:
            ds = BreastCancerDataset(
                name, split=split,
                image_size=image_size, seed=C.SEED)
            datasets.append(ds)
            all_labels.extend(ds.labels)
            print(f"    [{split:5s}] {name:12s}: {len(ds)} samples")
        except AssertionError as e:
            print(f"    [SKIP] {name}: {e}")

    assert datasets, "No datasets loaded!"
    pooled  = ConcatDataset(datasets)
    sampler = None
    shuffle = (split == "train")

    if split == "train":
        n   = len(all_labels)
        lbl = np.array(all_labels)
        cls, cnt = np.unique(lbl, return_counts=True)
        raw  = {c: n/k for c, k in zip(cls, cnt)}
        minw = min(raw.values())
        cap  = {c: min(w, minw*3.0) for c, w in raw.items()}
        sw   = torch.tensor([cap[l] for l in lbl], dtype=torch.double)
        sampler = WeightedRandomSampler(sw, num_samples=n,
                                        replacement=True)
        shuffle = False

    return DataLoader(
        pooled,
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = shuffle,
        num_workers = 0,          # ← 0 on Windows always
        pin_memory  = True,
        drop_last   = (split == "train"))


# ════════════════════════════════════════════════════════════════
# 3.  One epoch
# ════════════════════════════════════════════════════════════════
def run_epoch(model, loader, criterion,
              optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    buf = {k: [] for k in
           ["dice","iou","accuracy","sensitivity","specificity"]}

    ctx  = torch.enable_grad() if train else torch.no_grad()
    desc = "  train" if train else "  val  "
    first_batch = True

    with ctx:
        for batch in tqdm(loader, desc=desc, ncols=75, leave=False):
            imgs   = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device,  non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            mod    = batch["modality"][0]

            # ── Debug: first batch only ──────────────────────────
            if first_batch and train:
                print(f"\n  [DEBUG] imgs.device  = {imgs.device}")
                print(f"  [DEBUG] model.device = "
                      f"{next(model.parameters()).device}")
                print(f"  [DEBUG] modality     = {mod}")
                first_batch = False

            if train:
                optimizer.zero_grad()

            sp, cp     = model(imgs, modality=mod)
            loss, _, _ = criterion(sp, masks, cp, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), C.GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            for k, v in all_metrics(sp, masks, cp, labels).items():
                buf[k].append(v)

    n = max(len(loader), 1)
    result = {"loss": total_loss / n}
    result.update({k: float(np.mean(v)) for k, v in buf.items()})
    return result


# ════════════════════════════════════════════════════════════════
# 4.  Figure Generator
# ════════════════════════════════════════════════════════════════
class CentralFigureGenerator:

    def __init__(self, out_dir):
        self.fig_dir = Path(out_dir) / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid")
        self.C = {
            "train": "#2196F3", "val"  : "#F44336",
            "dice" : "#4CAF50", "iou"  : "#FF9800",
            "acc"  : "#9C27B0", "sens" : "#E91E63",
            "spec" : "#00BCD4", "loss" : "#607D8B",
        }

    def fig1_training_curves(self, history):
        epochs     = [r["epoch"]      for r in history]
        train_loss = [r["train_loss"] for r in history]
        val_loss   = [r["val_loss"]   for r in history]
        train_dice = [r["train_dice"] for r in history]
        val_dice   = [r["val_dice"]   for r in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Centralized Training Curves",
                     fontsize=14, fontweight="bold")

        ax1.plot(epochs, train_loss, color=self.C["train"],
                 lw=2, label="Train Loss")
        ax1.plot(epochs, val_loss, color=self.C["val"],
                 lw=2, label="Val Loss", linestyle="--")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss"); ax1.legend()
        ax1.set_xlim(left=1)

        ax2.plot(epochs, train_dice, color=self.C["train"],
                 lw=2, label="Train Dice")
        ax2.plot(epochs, val_dice, color=self.C["val"],
                 lw=2, label="Val Dice", linestyle="--")
        best_dice = max(val_dice)
        best_ep   = epochs[val_dice.index(best_dice)]
        ax2.axhline(y=best_dice, color="gold", linestyle=":",
                    lw=1.5, label=f"Best={best_dice:.4f}")
        ax2.scatter([best_ep], [best_dice], color="gold",
                    s=100, zorder=5)
        ax2.set_title("Dice Score"); ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice"); ax2.legend()
        ax2.set_xlim(left=1)

        plt.tight_layout()
        path = self.fig_dir / "fig1_training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Fig 1 saved: {path.name}")

    def fig2_metric_comparison(self, history):
        epochs  = [r["epoch"] for r in history]
        metrics = [
            ("val_dice",        "Dice",        self.C["dice"]),
            ("val_iou",         "IoU",         self.C["iou"]),
            ("val_accuracy",    "Accuracy",    self.C["acc"]),
            ("val_sensitivity", "Sensitivity", self.C["sens"]),
            ("val_specificity", "Specificity", self.C["spec"]),
        ]

        fig, ax = plt.subplots(figsize=(11, 6))
        for key, label, color in metrics:
            vals = [r[key] for r in history]
            ax.plot(epochs, vals, color=color, lw=2, label=label)

        ax.set_title("Centralized — All Val Metrics over Epochs",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.legend(fontsize=10)
        ax.set_xlim(left=1); ax.set_ylim(0, 1.05)

        plt.tight_layout()
        path = self.fig_dir / "fig2_metric_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Fig 2 saved: {path.name}")

    def fig3_fl_vs_central(self, history, fl_results_dir=None):
        fl_csv = None
        if fl_results_dir:
            fl_csv_path = Path(fl_results_dir) / "global_metrics.csv"
            if fl_csv_path.exists():
                import csv as _csv
                fl_csv = []
                with open(fl_csv_path, newline="",
                          encoding="utf-8") as f:
                    for row in _csv.DictReader(f):
                        fl_csv.append({
                            k: float(v) if v else 0.0
                            for k, v in row.items()
                            if k != "round"
                        } | {"round": int(row["round"])})

        metrics_to_compare = [
            ("val_dice",     "dice",     "Dice Score"),
            ("val_iou",      "iou",      "IoU Score"),
            ("val_accuracy", "accuracy", "Accuracy"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "FL (MAFedProx) vs Centralized (Upper Bound)",
            fontsize=12, fontweight="bold")

        for ax, (c_key, fl_key, title) in zip(axes, metrics_to_compare):
            c_epochs = [r["epoch"] for r in history]
            c_vals   = [r[c_key]  for r in history]
            ax.plot(c_epochs, c_vals, color=self.C["val"],
                    lw=2.5, label="Centralized (upper bound)")

            if fl_csv:
                fl_rounds = [r["round"] for r in fl_csv]
                fl_vals   = [r[fl_key]  for r in fl_csv]
                ax.plot(fl_rounds, fl_vals,
                        color=self.C["train"], lw=2.5,
                        linestyle="--", label="MAFedProx (Ours)")

            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch / Round")
            ax.set_ylabel(title)
            ax.legend(fontsize=9)
            ax.set_xlim(left=1)

        plt.tight_layout()
        path = self.fig_dir / "fig3_fl_vs_central.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Fig 3 saved: {path.name}")


# ════════════════════════════════════════════════════════════════
# 5.  Main
# ════════════════════════════════════════════════════════════════
def train_centralized(args):

    # ── Force CUDA ───────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — running on CPU")
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        # Warm up CUDA
        _ = torch.zeros(1).cuda()
        print(f"CUDA device : {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory : "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    out_dir = Path(args.output_dir)
    logger  = CentralLogger(out_dir)
    logger._log(f"Device : {device}")
    logger._log(f"Epochs : {args.epochs}")

    # ── Data ─────────────────────────────────────────────────────
    logger._log("\nBuilding pooled DataLoaders ...")
    train_dl = build_pooled_loader("train", args.batch_size,
                                   args.image_size)
    val_dl   = build_pooled_loader("val",   args.batch_size,
                                   args.image_size)
    logger._log(f"Train batches : {len(train_dl)}")
    logger._log(f"Val   batches : {len(val_dl)}\n")

    # ── Model — explicitly on CUDA ───────────────────────────────
    model = MultiTaskBreastModel()
    model = model.to(device)
    assert next(model.parameters()).is_cuda, \
        "Model is NOT on CUDA! Check your PyTorch installation."
    print(f"Model on: {next(model.parameters()).device} ✓")

    criterion = MultiTaskLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_dice    = 0.0
    best_metrics = {}

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_dl, criterion,
                            optimizer, device, train=True)
        val_m   = run_epoch(model, val_dl, criterion,
                            None, device, train=False)
        scheduler.step()

        is_best = val_m["dice"] > best_dice
        if is_best:
            best_dice    = val_m["dice"]
            best_metrics = {
                "epoch"            : epoch,
                "best_dice"        : val_m["dice"],
                "best_iou"         : val_m["iou"],
                "best_accuracy"    : val_m["accuracy"],
                "best_sensitivity" : val_m["sensitivity"],
                "best_specificity" : val_m["specificity"],
                "min_loss"         : val_m["loss"],
            }
            torch.save(model.state_dict(),
                       out_dir / "best_centralized.pth")

        logger.log_epoch(epoch, train_m, val_m, is_best)

    # ── Summary & Figures ─────────────────────────────────────────
    summary = logger.save_summary(best_metrics)

    logger._log("\nGenerating figures ...")
    gen = CentralFigureGenerator(out_dir)
    gen.fig1_training_curves(logger.history)
    gen.fig2_metric_comparison(logger.history)
    gen.fig3_fl_vs_central(logger.history,
                           fl_results_dir="./results/fl")
    logger._log(f"Figures saved → {gen.fig_dir}")

    print(f"\n{'='*60}")
    print(f"  Centralized Training Complete")
    print(f"{'='*60}")
    print(f"  Best Dice        : {best_metrics.get('best_dice',0):.4f}")
    print(f"  Best IoU         : {best_metrics.get('best_iou', 0):.4f}")
    print(f"  Best Accuracy    : {best_metrics.get('best_accuracy',0):.4f}")
    print(f"  Best Sensitivity : {best_metrics.get('best_sensitivity',0):.4f}")
    print(f"  Best Specificity : {best_metrics.get('best_specificity',0):.4f}")
    print(f"  Best Epoch       : {best_metrics.get('epoch',0)}")

    return logger.history


# ════════════════════════════════════════════════════════════════
# 6.  CLI
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Centralized Multimodal Baseline")
    p.add_argument("--epochs",     type=int,   default=C.CENTRAL_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=C.BATCH_SIZE)
    p.add_argument("--image_size", type=int,   default=C.IMAGE_SIZE)
    p.add_argument("--output_dir", type=str,
                   default="./results/centralized")
    args = p.parse_args()
    train_centralized(args)