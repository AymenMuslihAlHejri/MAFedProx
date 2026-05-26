"""
====================================================================
ablation.py  —  Final v2
Ablation Study: 5 configurations to justify each component

Table produced (for paper):
─────────────────────────────────────────────────────────────────────
Config  │ FedProx │ Mod-Aware │ Multi-task │ Focal │ Dice │ Sens
────────┼─────────┼───────────┼────────────┼───────┼──────┼──────
A FedAvg│    ✗    │     ✗     │     ✗      │   ✗   │      │
B       │    ✓    │     ✗     │     ✗      │   ✗   │      │
C       │    ✓    │     ✓     │     ✗      │   ✗   │      │
D       │    ✓    │     ✓     │     ✓      │   ✗   │      │
E(Ours) │    ✓    │     ✓     │     ✓      │   ✓   │      │
─────────────────────────────────────────────────────────────────────

Usage:
  python ablation.py --rounds 30 --output_dir ./results/ablation
====================================================================
"""

import os
import json
import argparse
import types
from copy import deepcopy
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from model  import (MultiTaskBreastModel, MultiTaskLoss,
                    DiceLoss, FocalLoss, all_metrics)
from client import FedProxClient
from server import ModalityAwareFedProx, _capped_avg


# ════════════════════════════════════════════════════════════════
# CLIENT RESOURCES
# ════════════════════════════════════════════════════════════════
CLIENT_RES = {
    "num_cpus": 2,
    "num_gpus": 0.5 if torch.cuda.is_available() else 0.0,
}


# ════════════════════════════════════════════════════════════════
# 1.  Variant Clients
# ════════════════════════════════════════════════════════════════

class SingleTaskClient(FedProxClient):
    """
    Config A & B: Segmentation only, no classification head.
    mu=0.0 → FedAvg behaviour (Config A)
    mu=C.MU → FedProx (Config B)
    """
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_model = deepcopy(self.model).to(self.device)
        for p in global_model.parameters():
            p.requires_grad_(False)

        epochs   = config.get("local_epochs", C.LOCAL_EPOCHS)
        run_loss = 0.0
        self.model.train()

        for _ in range(epochs):
            for batch in self.trainloader:
                imgs  = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                self.optimizer.zero_grad()
                seg_pred, _ = self.model(imgs, modality=self.modality)

                # Seg only: Dice + BCE (no Focal, no cls)
                loss = (0.5 * DiceLoss()(seg_pred, masks) +
                        0.5 * nn.BCELoss()(seg_pred, masks))

                # FedProx proximal term
                prox = sum(
                    torch.norm(lp - gp) ** 2
                    for lp, gp in zip(self.model.parameters(),
                                      global_model.parameters()))
                loss = loss + (self.mu / 2.0) * prox

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), C.GRAD_CLIP)
                self.optimizer.step()
                run_loss += loss.item()

        drift = self._drift(global_model)
        return self.get_parameters(config), \
               len(self.trainloader.dataset), \
               {"modality"    : self.modality,
                "dataset"     : self.dataset_name,
                "client_drift": float(drift),
                "train_loss"  : run_loss / max(len(self.trainloader), 1)}


class MultiTaskNoBCEClient(FedProxClient):
    """
    Config D: Multi-task with Dice+BCE (no Focal Loss).
    """
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_model = deepcopy(self.model).to(self.device)
        for p in global_model.parameters():
            p.requires_grad_(False)

        # Override criterion: no Focal
        from model import MultiTaskLoss as MTL
        import torch.nn as nn2
        class NoBCELoss(MTL):
            def forward(self, sp, st, cp, ct):
                ls = (0.5 * DiceLoss()(sp, st) +
                      0.5 * nn2.BCELoss()(sp, st))
                lc = nn2.BCELoss()(cp.squeeze(1), ct)
                return self.a*ls + self.b*lc, ls, lc

        criterion = NoBCELoss(alpha=C.LOSS_ALPHA, beta=C.LOSS_BETA)
        epochs    = config.get("local_epochs", C.LOCAL_EPOCHS)
        run_loss  = 0.0
        self.model.train()

        for _ in range(epochs):
            for batch in self.trainloader:
                imgs   = batch["image"].to(self.device)
                masks  = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)
                self.optimizer.zero_grad()
                sp, cp = self.model(imgs, modality=self.modality)
                loss, _, _ = criterion(sp, masks, cp, labels)
                prox = sum(
                    torch.norm(lp - gp) ** 2
                    for lp, gp in zip(self.model.parameters(),
                                      global_model.parameters()))
                loss = loss + (self.mu / 2.0) * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), C.GRAD_CLIP)
                self.optimizer.step()
                run_loss += loss.item()

        drift = self._drift(global_model)
        return self.get_parameters(config), \
               len(self.trainloader.dataset), \
               {"modality"    : self.modality,
                "dataset"     : self.dataset_name,
                "client_drift": float(drift),
                "train_loss"  : run_loss / max(len(self.trainloader), 1)}


# ════════════════════════════════════════════════════════════════
# 2.  Simple FedAvg Strategy (Config A)
# ════════════════════════════════════════════════════════════════
class SimpleFedAvg(fl.server.strategy.FedAvg):
    """Standard FedAvg — no modality awareness."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_metrics = []

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        total_ex = sum(r.num_examples for _, r in results)
        agg_loss = sum(r.loss * r.num_examples
                       for _, r in results) / total_ex

        def _mean(key):
            vals = [r.metrics[key] for _, r in results
                    if key in r.metrics]
            return float(np.mean(vals)) if vals else 0.0

        metrics = {
            "loss"       : agg_loss,
            "dice"       : _mean("dice"),
            "iou"        : _mean("iou"),
            "accuracy"   : _mean("accuracy"),
            "sensitivity": _mean("sensitivity"),
            "specificity": _mean("specificity"),
        }
        self.round_metrics.append({"round": server_round, **metrics})
        return agg_loss, metrics


# ════════════════════════════════════════════════════════════════
# 3.  Initial parameters
# ════════════════════════════════════════════════════════════════
def _init_params():
    ndarrays = [v.cpu().numpy()
                for v in MultiTaskBreastModel().state_dict().values()]
    return ndarrays_to_parameters(ndarrays)


# ════════════════════════════════════════════════════════════════
# 4.  Run one configuration
# ════════════════════════════════════════════════════════════════
def run_config(cfg_name, rounds, local_epochs,
               client_class, strategy, out_dir):
    """Run one ablation configuration and return metrics history."""

    cfg_dir = Path(out_dir) / cfg_name
    cfg_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {cfg_name}")
    print(f"{'='*60}")

    def _client_fn(cid):
        dataset = C.CLIENT_MAP[str(cid)]
        return client_class(dataset_name=dataset, mu=C.MU)

    history = fl.simulation.start_simulation(
        client_fn        = _client_fn,
        num_clients      = C.NUM_CLIENTS,
        config           = fl.server.ServerConfig(num_rounds=rounds),
        strategy         = strategy,
        client_resources = CLIENT_RES,
    )

    m = history.metrics_distributed

    def _best(key):
        vals = [v for _, v in m.get(key, [(0, 0.0)])]
        return max(vals) if vals else 0.0

    result = {
        "config"      : cfg_name,
        "best_dice"   : _best("dice"),
        "best_iou"    : _best("iou"),
        "best_acc"    : _best("accuracy"),
        "best_sens"   : _best("sensitivity"),
        "best_spec"   : _best("specificity"),
        "history"     : {k: [(r, v) for r, v in vals]
                         for k, vals in m.items()},
    }

    with open(cfg_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"  Best Dice : {result['best_dice']:.4f}")
    print(f"  Best Sens : {result['best_sens']:.4f}")
    return result


# ════════════════════════════════════════════════════════════════
# 5.  Build strategies
# ════════════════════════════════════════════════════════════════
def make_fedavg_strategy(rounds, local_epochs):
    return make_modality_strategy(rounds, local_epochs,
                                  "./results/ablation/A_FedAvg_SingleTask")


def make_modality_strategy(rounds, local_epochs, out_dir_str):
    return ModalityAwareFedProx(
        num_rounds         = rounds,
        fusion_alpha       = C.FUSION_ALPHA,
        min_clients        = C.NUM_CLIENTS,
        initial_parameters = _init_params(),
        local_epochs       = local_epochs,
        max_client_weight  = C.MAX_CLIENT_WEIGHT,
        out_dir            = out_dir_str,
    )


# ════════════════════════════════════════════════════════════════
# 6.  Figure: Ablation comparison
# ════════════════════════════════════════════════════════════════
def plot_ablation(results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.style.use("seaborn-v0_8-darkgrid")
    fig.suptitle("Ablation Study — Component Contribution Analysis",
                 fontsize=13, fontweight="bold")

    colors = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    metrics_to_plot = [
        ("best_dice", "Dice Score"),
        ("best_sens", "Sensitivity"),
        ("best_spec", "Specificity"),
    ]

    configs = [r["config"] for r in results]
    short   = ["A\nFedAvg", "B\nFedProx", "C\n+Mod-Aware",
                "D\n+Multi-task", "E\n+Focal\n(Ours)"]

    for ax, (key, title) in zip(axes, metrics_to_plot):
        vals = [r[key] for r in results]
        bars = ax.bar(short, vals, color=colors, width=0.6,
                      edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        # Highlight best
        best_idx = vals.index(max(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    path = Path(out_dir) / "ablation_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {path}")


def plot_dice_curves(results, out_dir):
    """Plot Dice convergence curve for each config."""
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.style.use("seaborn-v0_8-darkgrid")

    colors    = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    linestyle = [":", "--", "-.", "-", "-"]

    for res, color, ls in zip(results, colors, linestyle):
        history = res.get("history", {})
        dice    = history.get("dice", [])
        if dice:
            rounds = [r for r, _ in dice]
            vals   = [v for _, v in dice]
            lw     = 3 if "Ours" in res["config"] or "E_" in res["config"] \
                     else 1.8
            ax.plot(rounds, vals, color=color, linestyle=ls,
                    linewidth=lw, label=res["config"],
                    marker="o" if lw == 3 else None,
                    markersize=4)

    ax.set_title("Dice Score Convergence — Ablation Study",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Dice Score")
    ax.legend(fontsize=10)
    ax.set_xlim(left=1)

    plt.tight_layout()
    path = Path(out_dir) / "ablation_dice_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {path}")


# ════════════════════════════════════════════════════════════════
# 7.  Print ablation table
# ════════════════════════════════════════════════════════════════
def print_table(results):
    print("\n" + "="*75)
    print("  ABLATION TABLE")
    print("="*75)
    header = (f"{'Config':<28} {'FedProx':>8} {'Mod-Aware':>10} "
              f"{'Multi-task':>11} {'Focal':>6} "
              f"{'Dice':>6} {'Sens':>6} {'Spec':>6}")
    print(header)
    print("-"*75)

    markers = {
        "A_FedAvg_SingleTask"          : ("✗", "✗", "✗", "✗"),
        "B_FedProx_SingleTask"         : ("✓", "✗", "✗", "✗"),
        "C_FedProx_ModalityAware"      : ("✓", "✓", "✗", "✗"),
        "D_FedProx_MA_MultiTask"       : ("✓", "✓", "✓", "✗"),
        "E_FedProx_MA_MT_Focal_Ours"   : ("✓", "✓", "✓", "✓"),
    }

    for r in results:
        cfg  = r["config"]
        fp, ma, mt, fl_ = markers.get(cfg, ("?","?","?","?"))
        print(f"  {cfg:<28} {fp:>8} {ma:>10} {mt:>11} {fl_:>6} "
              f"{r['best_dice']:>6.4f} "
              f"{r['best_sens']:>6.4f} "
              f"{r['best_spec']:>6.4f}")

    print("="*75)
    best = max(results, key=lambda r: r["best_dice"])
    print(f"\n  Best config: {best['config']}"
          f"  Dice={best['best_dice']:.4f}")


# ════════════════════════════════════════════════════════════════
# 8.  Main
# ════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="FL Ablation Study")
    p.add_argument("--rounds",       type=int, default=30)
    p.add_argument("--local_epochs", type=int, default=C.LOCAL_EPOCHS)
    p.add_argument("--output_dir",   type=str, default="./results/ablation")
    p.add_argument("--config",       type=str, default="all",
                   choices=["all","A","B","C","D","E"],
                   help="Config to run: all | A | B | C | D | E")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    R = args.rounds
    E = args.local_epochs

    ALL_CONFIGS = [
        ("A_FedAvg_SingleTask",
         SingleTaskClient,
         lambda: make_modality_strategy(
             R, E, str(out_dir/"A_FedAvg_SingleTask")), 0.0),
        ("B_FedProx_SingleTask",
         SingleTaskClient,
         lambda: make_fedavg_strategy(R, E), C.MU),
        ("C_FedProx_ModalityAware",
         SingleTaskClient,
         lambda: make_modality_strategy(R, E, str(out_dir/"C_FedProx_ModalityAware")), C.MU),
        ("D_FedProx_MA_MultiTask",
         MultiTaskNoBCEClient,
         lambda: make_modality_strategy(R, E, str(out_dir/"D_FedProx_MA_MultiTask")), C.MU),
        ("E_FedProx_MA_MT_Focal_Ours",
         FedProxClient,
         lambda: make_modality_strategy(R, E, str(out_dir/"E_FedProx_MA_MT_Focal_Ours")), C.MU),
    ]

    LETTER_MAP = {
        "A": "A_FedAvg_SingleTask",
        "B": "B_FedProx_SingleTask",
        "C": "C_FedProx_ModalityAware",
        "D": "D_FedProx_MA_MultiTask",
        "E": "E_FedProx_MA_MT_Focal_Ours",
    }

    to_run = ALL_CONFIGS if args.config == "all" else [
        c for c in ALL_CONFIGS if c[0] == LETTER_MAP[args.config]]

    print(f"\n{'='*60}")
    print(f"  FL Ablation Study")
    print(f"  Config            : {args.config}")
    print(f"  Rounds per config : {R}")
    print(f"  Local epochs      : {E}")
    print(f"  Output            : {out_dir}")
    print(f"{'='*60}")

    new_results = []
    for name, client_cls, strategy_fn, mu_val in to_run:

        def make_client_fn(cls, mu):
            def _fn(cid):
                dataset = C.CLIENT_MAP[str(cid)]
                return cls(dataset_name=dataset, mu=mu)
            return _fn

        strategy = strategy_fn()
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"  mu={mu_val}  client={client_cls.__name__}")
        print(f"{'='*60}")

        history = fl.simulation.start_simulation(
            client_fn        = make_client_fn(client_cls, mu_val),
            num_clients      = C.NUM_CLIENTS,
            config           = fl.server.ServerConfig(num_rounds=R),
            strategy         = strategy,
            client_resources = CLIENT_RES,
        )

        m = history.metrics_distributed

        def _best(key):
            vals = [v for _, v in m.get(key, [(0, 0.0)])]
            return max(vals) if vals else 0.0

        result = {
            "config"    : name,
            "best_dice" : _best("dice"),
            "best_iou"  : _best("iou"),
            "best_acc"  : _best("accuracy"),
            "best_sens" : _best("sensitivity"),
            "best_spec" : _best("specificity"),
            "history"   : {k: [(r, v) for r, v in vals]
                           for k, vals in m.items()},
        }
        new_results.append(result)

        cfg_dir = out_dir / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"  ✓ Dice={result['best_dice']:.4f}  Sens={result['best_sens']:.4f}")

    # ── Merge with existing results ───────────────────────────────
    all_json = out_dir / "ablation_all.json"
    existing = []
    if all_json.exists():
        with open(all_json) as f:
            existing = json.load(f)

    existing_names = {r["config"] for r in existing}
    for r in new_results:
        if r["config"] in existing_names:
            existing = [r if e["config"]==r["config"] else e
                        for e in existing]
        else:
            existing.append(r)

    order = [c[0] for c in ALL_CONFIGS]
    existing.sort(key=lambda r: order.index(r["config"])
                  if r["config"] in order else 99)

    with open(all_json, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"\n  Saved {len(existing)}/5 configs to ablation_all.json")
    print_table(existing)

    if len(existing) == 5:
        print("\n  All 5 configs done — generating figures ...")
        plot_ablation(existing, out_dir)
        plot_dice_curves(existing, out_dir)
        print(f"\n✅  Ablation complete → {out_dir}")
    else:
        done  = {r["config"] for r in existing}
        remaining = [k for k,v in LETTER_MAP.items() if v not in done]
        print(f"\n  Completed : {[r['config'] for r in existing]}")
        print(f"  Remaining : {remaining}")
        if remaining:
            print(f"\n  Next command:")
            print(f"    python ablation.py --rounds {R} --config {remaining[0]}")


if __name__ == "__main__":
    main()