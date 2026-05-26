"""
====================================================================
main.py  —  Final v2  (with logging + figures)

Usage:
  python main.py --rounds 100 --local_epochs 5
  python main.py --rounds 100 --run_centralized
  python main.py --rounds 100 --run_all
====================================================================
"""

import os
import argparse
import json
import types
from pathlib import Path

# Fix OpenMP duplicate lib on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import flwr as fl

import config as C
from model  import MultiTaskBreastModel
from client import client_fn
from server import ModalityAwareFedProx
from logger import FLFigureGenerator
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"


def _init_params():
    import pickle
    best_path = "./results/fl/best_model.pkl"

    if os.path.exists(best_path):
        print("  Loading best model from previous run...")
        with open(best_path, "rb") as f:
            arrays = pickle.load(f)
        return fl.common.ndarrays_to_parameters(arrays)
    else:
        print("  Starting from scratch...")
        ndarrays = [v.cpu().numpy()
                    for v in MultiTaskBreastModel().state_dict().values()]
        return fl.common.ndarrays_to_parameters(ndarrays)


CLIENT_RES = {
    "num_cpus": 2,
    "num_gpus": 0.5 if torch.cuda.is_available() else 0.0,
}


def run_fl(args, out_dir):
    fl_out_dir = str(out_dir / "fl")

    print(f"\n{'='*60}")
    print(f"  FL Simulation")
    print(f"  Clients      : {C.NUM_CLIENTS}  "
          f"(US={len(C.US_DATASETS)}  Mammo={len(C.MAMMO_DATASETS)})")
    print(f"  Rounds       : {args.rounds}")
    print(f"  Local epochs : {args.local_epochs}")
    print(f"  Output       : {fl_out_dir}")
    print(f"{'='*60}\n")

    strategy = ModalityAwareFedProx(
        num_rounds         = args.rounds,
        fusion_alpha       = args.fusion_alpha,
        min_clients        = C.NUM_CLIENTS,
        initial_parameters = _init_params(),
        local_epochs       = args.local_epochs,
        max_client_weight  = args.max_client_weight,
        out_dir            = fl_out_dir,
    )

    history = fl.simulation.start_simulation(
        client_fn        = lambda cid: client_fn(str(cid)),
        num_clients      = C.NUM_CLIENTS,
        config           = fl.server.ServerConfig(
                               num_rounds=args.rounds),
        strategy         = strategy,
        client_resources = CLIENT_RES,
    )

    # ── Save summary + generate figures ──────────────────────────
    summary = strategy.finalize()

    gen = FLFigureGenerator(out_dir=fl_out_dir)
    gen.generate_all(strategy.fl_logger)

    # ── Print final results ───────────────────────────────────────
    best = summary.get("best_metrics", {})
    print(f"\n{'='*60}")
    print(f"  FL Training Complete — {args.rounds} rounds")
    print(f"{'='*60}")
    print(f"  Best Dice        : {best.get('best_dice',       0):.4f}")
    print(f"  Best IoU         : {best.get('best_iou',        0):.4f}")
    print(f"  Best Accuracy    : {best.get('best_accuracy',   0):.4f}")
    print(f"  Best Sensitivity : {best.get('best_sensitivity',0):.4f}")
    print(f"  Best Specificity : {best.get('best_specificity',0):.4f}")
    print(f"  Min  Loss        : {best.get('min_loss',        0):.4f}")

    if strategy.drift_log:
        all_d = [d for dl in strategy.drift_log.values()
                 for d in dl]
        print(f"  Mean Drift       : "
              f"{sum(all_d)/len(all_d):.6f}")

    print(f"\n  Logs    → {fl_out_dir}/fl_log_*.txt")
    print(f"  CSVs    → {fl_out_dir}/global_metrics.csv")
    print(f"           → {fl_out_dir}/per_client.csv")
    print(f"           → {fl_out_dir}/drift_log.csv")
    print(f"  Figures → {fl_out_dir}/figures/")
    print(f"  Summary → {fl_out_dir}/fl_summary_*.json")

    return summary


def main():
    p = argparse.ArgumentParser(
        description="FL Multimodal Breast Cancer — Final v2")
    p.add_argument("--rounds",            type=int,
                   default=C.FL_ROUNDS)
    p.add_argument("--local_epochs",      type=int,
                   default=C.LOCAL_EPOCHS)
    p.add_argument("--fusion_alpha",      type=float,
                   default=C.FUSION_ALPHA)
    p.add_argument("--max_client_weight", type=float,
                   default=C.MAX_CLIENT_WEIGHT)
    p.add_argument("--output_dir",        type=str,
                   default="./results")
    p.add_argument("--run_centralized",   action="store_true")
    p.add_argument("--run_ablation",      action="store_true")
    p.add_argument("--run_all",           action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_centralized or args.run_all:
        from baseline_centralized import train_centralized
        c_args = types.SimpleNamespace(
            epochs     = args.rounds,
            batch_size = C.BATCH_SIZE,
            image_size = C.IMAGE_SIZE,
            output_dir = str(out_dir / "centralized"))
        train_centralized(c_args)

    if args.run_ablation or args.run_all:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "ablation.py",
             "--rounds",     str(min(args.rounds, 50)),
             "--output_dir", str(out_dir / "ablation")],
            check=True)

    run_fl(args, out_dir)
    print(f"\n✅  All done → {out_dir}")


if __name__ == "__main__":
    main()
