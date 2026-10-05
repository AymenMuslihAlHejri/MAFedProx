#!/usr/bin/env python3
"""bmc_config.py - the frozen R5 configurations (approved matrix v2 + R5.1 decisions)."""
from __future__ import annotations
import copy, json
from pathlib import Path

MU_SELECTED = "SELECTED_FROM_FEDPROX_GRID"      # resolved only from the frozen mu record

BASE = {
    "architecture": "resnet34-encoder_2x-unet-decoder(256,128,64,32,16)_mlp-cls-head_raw-logits",
    "encoder_init": "imagenet_resnet34_rgb_conv1_summed", "image_size": 256,
    "r3": {"intensity_rule": "declared_range", "aspect_rule": "direct_resize",
           "clahe_role": "augmentation", "vflip_scope": "ultrasound_only"},
    "batch_size": 16, "drop_last_train": True,
    "sampler": "class_balanced_WeightedRandomSampler_replacement_train_only",
    "optimizer": {"name": "AdamW", "lr0": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4},
    "schedule": {"name": "exposure_indexed_cosine", "lr0": 1e-4, "lr_min": 1e-6, "T_epoch_eq": 100, "block": 2},
    "loss": {"lambda_seg": 1.0, "lambda_cls": 1.0, "dice_eps": 1.0,
             "seg": "0.5*(1-soft_dice_per_image)+0.5*mean_pixel_BCEWithLogits", "cls": "BCEWithLogits_unweighted"},
    "selection": {"criterion": "joint_val_loss_client_macro", "split": "validation", "tie": "earliest"},
    "early_stopping": None, "grad_clip": None, "mixed_precision": "bf16_autocast",
    "bn_momentum": 0.1, "training_seeds": [101, 202, 303],
    "split_seed": 20260929, "bootstrap_seed": 20260930,
}
FED = {"regime": "federated", "rounds": 50, "local_epochs": 2, "optimizer_reset_each_round": True}
AGG = {"cap": 0.4, "capped": True, "alpha": 0.5, "within_group_weights": "n_train_waterfilled_cap"}


def _m(**over):
    c = copy.deepcopy(BASE)
    for k, v in over.items():
        c[k] = copy.deepcopy(v) if isinstance(v, dict) else v
    return c


def build() -> dict:
    cfg = {
        "proposed": _m(method="proposed", **FED, mu=MU_SELECTED, aggregation=AGG),
        "fedavg": _m(method="fedavg", **FED, mu=0.0, aggregation={"rule": "update_aware_n_weighted"}),
        "fedprox": _m(method="fedprox", **FED, mu=MU_SELECTED, aggregation={"rule": "update_aware_n_weighted"}),
        "centralised": _m(method="centralised", regime="centralised", epochs=100, eval_every=2,
                          centralised_draws="exact_per_client_D_k_concatenated"),
        "local": _m(method="local", regime="local", epochs=100, eval_every=2),
        "N1_mu0": _m(method="proposed", **FED, mu=0.0, aggregation=AGG, ablation="N1"),
        "N2_uncapped": _m(method="proposed", **FED, mu=MU_SELECTED,
                          aggregation={**AGG, "capped": False, "within_group_weights": "n_train_proportional"},
                          ablation="N2"),
        "N3_cls_only": _m(method="proposed", **FED, mu=MU_SELECTED, aggregation=AGG, ablation="N3",
                          loss={**BASE["loss"], "lambda_seg": 0.0}),
        "N4_seg_only": _m(method="proposed", **FED, mu=MU_SELECTED, aggregation=AGG, ablation="N4",
                          loss={**BASE["loss"], "lambda_cls": 0.0}),
    }
    for mu in (0.001, 0.01, 0.1, 1.0):
        cfg[f"fedprox_mu_{mu:g}"] = _m(method="fedprox", **FED, mu=mu,
                                       aggregation={"rule": "update_aware_n_weighted"},
                                       purpose="mu_selection_validation_only", seeds=[101])
    return cfg


def write(out_dir: Path) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, c in build().items():
        (out / f"{name}.json").write_text(json.dumps(c, indent=2, sort_keys=True))


if __name__ == "__main__":
    write(Path(__file__).resolve().parent / "configs")
    print("configs written")
