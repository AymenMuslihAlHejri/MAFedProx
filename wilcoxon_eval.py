"""
Wilcoxon Signed-Rank Test for MAFedProx
========================================
1. Evaluates Centralized model per-client (test set)
2. Extracts per-client DSC from FL logs
3. Runs Wilcoxon signed-rank tests
4. Saves results to wilcoxon_results.json
"""

import os, sys, json, torch
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Adjust this path ──────────────────────────────────────────
BASE = Path(r"C:\Users\Asus\Python code\FL_MuL_BC")
sys.path.insert(0, str(BASE))

import config as C
from model   import MultiTaskBreastModel, all_metrics
from dataset import BreastCancerDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Client list ───────────────────────────────────────────────
CLIENTS = list(C.CLIENT_MAP.values())
print(f"Clients: {CLIENTS}")


# ════════════════════════════════════════════════════════════
# 1. Evaluate Centralized model per-client
# ════════════════════════════════════════════════════════════
def eval_model_per_client(model_path, split="test"):
    """Load model and evaluate on each client's test set."""
    model = MultiTaskBreastModel().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"\nLoaded: {model_path}")

    results = {}
    for name in CLIENTS:
        try:
            ds = BreastCancerDataset(
                name, split=split,
                image_size=C.IMAGE_SIZE, seed=C.SEED)
            if len(ds) == 0:
                print(f"  [SKIP] {name}: empty test set")
                continue

            # Determine modality
            sample = ds[0]
            mod = sample["modality"]

            loader = DataLoader(
                ds, batch_size=8,
                shuffle=False, num_workers=0,
                pin_memory=True)

            dice_list = []
            with torch.no_grad():
                for batch in loader:
                    imgs   = batch["image"].to(DEVICE, non_blocking=True)
                    masks  = batch["mask"].to(DEVICE,  non_blocking=True)
                    labels = batch["label"].to(DEVICE, non_blocking=True)
                    b_mod  = batch["modality"][0]

                    sp, cp = model(imgs, modality=b_mod)
                    m = all_metrics(sp, masks, cp, labels)
                    dice_list.append(m["dice"])

            mean_dice = float(np.mean(dice_list))
            results[name] = {
                "dice": mean_dice,
                "modality": mod,
                "n_samples": len(ds)
            }
            print(f"  {name:15s}: dice={mean_dice:.4f}  "
                  f"(n={len(ds)}, {mod})")

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results[name] = {"dice": None, "error": str(e)}

    return results


# ════════════════════════════════════════════════════════════
# 2. Per-client DSC from FL experiments (from logs)
# ════════════════════════════════════════════════════════════

# Exp 2 Best Round 88 per-client DSC
EXP2_BEST = {
    "BUSI"      : 0.8064,
    "BUS_UC"    : 0.9220,
    "BrEaST"    : 0.7624,
    "BUS_BRA"   : 0.9062,
    "INbreast"  : 0.8338,
    "DMID"      : 0.5215,
    "CBIS_DDSM" : 0.2613,
    "AIISLAB"   : 0.3884,
}

# Ablation Config B best round per-client (Round 13)
# From B_FedProx_SingleTask log - best global round
ABLATION_B = {
    "BUSI"      : 0.8237,
    "BUS_UC"    : 0.9165,
    "BrEaST"    : 0.7514,
    "BUS_BRA"   : 0.8819,
    "INbreast"  : 0.8003,
    "DMID"      : 0.4162,
    "CBIS_DDSM" : 0.2517,
    "AIISLAB"   : 0.3131,
}

# Ablation Config E best round per-client (Round 11)
ABLATION_E = {
    "BUSI"      : 0.7978,
    "BUS_UC"    : 0.9177,
    "BrEaST"    : 0.7776,
    "BUS_BRA"   : 0.8848,
    "INbreast"  : 0.7874,
    "DMID"      : 0.5160,
    "CBIS_DDSM" : 0.2560,
    "AIISLAB"   : 0.3028,
}


# ════════════════════════════════════════════════════════════
# 3. Wilcoxon signed-rank test
# ════════════════════════════════════════════════════════════
def run_wilcoxon(name_a, dice_a, name_b, dice_b):
    """Paired Wilcoxon signed-rank test (two-tailed, α=0.05)."""
    a = np.array(dice_a)
    b = np.array(dice_b)

    # Remove pairs where either is None
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 3:
        return {"error": "Too few valid pairs"}

    try:
        stat, p = wilcoxon(a, b, alternative="two-sided")
        sig = p < 0.05
        return {
            "comparison": f"{name_a} vs {name_b}",
            "W_statistic": float(stat),
            "p_value": float(p),
            "significant": bool(sig),
            "alpha": 0.05,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(a - b)),
            "n_pairs": int(len(a))
        }
    except Exception as e:
        return {"error": str(e), "comparison": f"{name_a} vs {name_b}"}


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  Step 1: Evaluating Centralized model per-client")
    print("="*60)

    central_path = BASE / "results" / "centralized" / "best_centralized.pth"
    if not central_path.exists():
        print(f"ERROR: {central_path} not found!")
        print("Looking for alternative...")
        # Search
        for p in (BASE / "results" / "centralized").glob("*.pth"):
            print(f"  Found: {p}")
        sys.exit(1)

    central_results = eval_model_per_client(central_path, split="test")

    print("\n" + "="*60)
    print("  Step 2: Building DSC arrays")
    print("="*60)

    # Build aligned arrays (same client order)
    client_order = list(EXP2_BEST.keys())

    dsc_mafedprox   = [EXP2_BEST[c]   for c in client_order]
    dsc_central     = [central_results.get(c, {}).get("dice", np.nan)
                       for c in client_order]
    dsc_ablation_b  = [ABLATION_B[c]  for c in client_order]
    dsc_ablation_e  = [ABLATION_E[c]  for c in client_order]

    print(f"\n{'Client':15s}  {'MAFedProx':>10}  {'Central':>10}  {'Config_B':>10}  {'Config_E':>10}")
    print("-"*60)
    for i, c in enumerate(client_order):
        print(f"{c:15s}  {dsc_mafedprox[i]:10.4f}  "
              f"{dsc_central[i]:10.4f}  "
              f"{dsc_ablation_b[i]:10.4f}  "
              f"{dsc_ablation_e[i]:10.4f}")

    print("\n" + "="*60)
    print("  Step 3: Wilcoxon Signed-Rank Tests")
    print("="*60)

    comparisons = [
        ("MAFedProx (Exp2)", dsc_mafedprox,
         "Centralised",      dsc_central),
        ("MAFedProx (Exp2)", dsc_mafedprox,
         "Config B (FedProx only)", dsc_ablation_b),
        ("MAFedProx (Exp2)", dsc_mafedprox,
         "Config E (Ablation)",    dsc_ablation_e),
    ]

    all_results = {}
    for name_a, dice_a, name_b, dice_b in comparisons:
        res = run_wilcoxon(name_a, dice_a, name_b, dice_b)
        key = f"{name_a} vs {name_b}"
        all_results[key] = res
        sig_str = "✅ Significant" if res.get("significant") else "❌ Not Significant"
        print(f"\n  {key}")
        print(f"    W = {res.get('W_statistic', 'N/A'):.1f}")
        print(f"    p = {res.get('p_value', 'N/A'):.4f}")
        print(f"    {sig_str}")
        print(f"    Mean diff = {res.get('mean_diff', 0):+.4f}")

    # Save results
    output = {
        "per_client_dsc": {
            "mafedprox_exp2"    : dict(zip(client_order, dsc_mafedprox)),
            "centralised"       : dict(zip(client_order, dsc_central)),
            "ablation_config_b" : dict(zip(client_order, dsc_ablation_b)),
            "ablation_config_e" : dict(zip(client_order, dsc_ablation_e)),
        },
        "wilcoxon_tests": all_results
    }

    out_path = BASE / "results" / "wilcoxon_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Results saved → {out_path}")
    print(f"{'='*60}")
