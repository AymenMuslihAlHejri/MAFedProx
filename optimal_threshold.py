"""
MAFedProx — Optimal Classification Threshold
=============================================
Finds optimal threshold via Youden Index on validation set
Then re-evaluates all external datasets with new threshold
"""

import os, sys, json, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
from sklearn.metrics import roc_curve, roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

BASE       = Path(r"C:\Users\Asus\Python code\FL_MuL_BC")
MODEL_PATH = Path(r"C:\Users\Asus\Python code\FL_MuL_BC\results_exp2\fl\best_model.pkl")
OUT_DIR    = BASE / "results" / "external_validation_ALL"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))
import config as C
from model import MultiTaskBreastModel
from dataset import BreastCancerDataset
from torch.utils.data import DataLoader

DEVICE   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
print(f"Device : {DEVICE}")

# ── Load model ────────────────────────────────────────────
print("\n[1] Loading model...")
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

model = MultiTaskBreastModel().to(DEVICE)
if isinstance(data, list):
    sd = model.state_dict()
    new_sd = {}
    for i, (k, v) in enumerate(sd.items()):
        new_sd[k] = torch.tensor(
            np.array(data[i])).reshape(v.shape)
    model.load_state_dict(new_sd)
elif isinstance(data, dict):
    model.load_state_dict(data)
else:
    model = data.to(DEVICE)
model.eval()
print("  Model loaded ✓")

# ════════════════════════════════════════════════════════
# Step 1: Collect val-set probabilities from training clients
# ════════════════════════════════════════════════════════
print("\n[2] Collecting validation probabilities...")

all_probs, all_labels = [], []

for name in C.CLIENT_MAP.values():
    try:
        ds = BreastCancerDataset(
            name, split="val",
            image_size=C.IMAGE_SIZE, seed=C.SEED)
        if len(ds) == 0:
            ds = BreastCancerDataset(
                name, split="test",
                image_size=C.IMAGE_SIZE, seed=C.SEED)
        if len(ds) == 0:
            continue

        mod = ds[0]["modality"]
        loader = DataLoader(ds, batch_size=16,
                           shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(DEVICE)
                lbls = batch["label"].cpu().numpy()
                _, cls_pred = model(imgs, modality=mod)
                p = torch.sigmoid(cls_pred).cpu().numpy()
                all_probs.extend(p.flatten().tolist())
                all_labels.extend(lbls.flatten().tolist())

        print(f"  {name}: {len(ds)} samples")
    except Exception as e:
        print(f"  {name}: {e}")

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
print(f"\n  Total: {len(all_probs)} samples")

# ════════════════════════════════════════════════════════
# Step 2: Find optimal threshold via Youden Index
# ════════════════════════════════════════════════════════
print("\n[3] Computing optimal threshold...")

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
youden = tpr - fpr
best_idx = np.argmax(youden)
optimal_tau = float(thresholds[best_idx])
auc = roc_auc_score(all_labels, all_probs)

print(f"  AUC              : {auc:.4f}")
print(f"  Optimal threshold: {optimal_tau:.4f}")
print(f"  At optimal τ:")
print(f"    Sensitivity: {tpr[best_idx]:.4f}")
print(f"    Specificity: {1-fpr[best_idx]:.4f}")
print(f"    Youden Index: {youden[best_idx]:.4f}")

# ── Plot ROC with optimal threshold marked ────────────────
fig, ax = plt.subplots(figsize=(6,6), facecolor="white")
ax.plot(fpr, tpr, color="#1F3E64", lw=2,
        label=f"MAFedProx (AUC = {auc:.3f})")
ax.scatter(fpr[best_idx], tpr[best_idx],
           color="red", s=120, zorder=5,
           label=f"Optimal τ = {optimal_tau:.3f}\n"
                 f"(Sens={tpr[best_idx]:.3f}, "
                 f"Spec={1-fpr[best_idx]:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC — MAFedProx with Optimal Threshold",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0,1); ax.set_ylim(0,1.02)
plt.tight_layout()
plt.savefig(OUT_DIR/"roc_optimal_threshold.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: roc_optimal_threshold.png")

# ════════════════════════════════════════════════════════
# Step 3: Re-evaluate external datasets with optimal τ
# ════════════════════════════════════════════════════════
print(f"\n[4] Re-evaluating with τ={optimal_tau:.4f}...")

# Load existing per-image results
json_path = OUT_DIR / "ext_validation_all.json"
with open(json_path) as f:
    existing = json.load(f)

def re_eval_cls(per_image_list, tau):
    """Re-compute classification accuracy with new threshold."""
    n_correct = 0
    benign_correct = 0
    malign_correct = 0
    n_benign = n_malign = 0

    for r in per_image_list:
        pred_label = 1 if r["cls_prob"] > tau else 0
        correct = int(pred_label == r["label"])
        n_correct += correct

        if r["label"] == 0:
            n_benign += 1
            benign_correct += correct
        else:
            n_malign += 1
            malign_correct += correct

    n = len(per_image_list)
    return {
        "tau"           : tau,
        "overall_acc"   : n_correct / n if n > 0 else 0,
        "benign_acc"    : benign_correct/n_benign if n_benign>0 else 0,
        "malignant_acc" : malign_correct/n_malign if n_malign>0 else 0,
        "n"             : n,
    }

print(f"\n  {'Dataset':30s}  {'τ=0.5':>8}  {'τ={:.3f}'.format(optimal_tau):>10}  {'Δ':>8}")
print(f"  {'-'*65}")

threshold_results = {}
for ds_name, per_image in existing["per_image"].items():
    if not per_image: continue

    # Old accuracy (τ=0.5)
    old_acc = np.mean([r["cls_correct"] for r in per_image])

    # New accuracy (optimal τ)
    new = re_eval_cls(per_image, optimal_tau)
    new_acc = new["overall_acc"]
    delta = new_acc - old_acc

    threshold_results[ds_name] = {
        "tau_05":  {"overall": old_acc},
        "tau_opt": new,
    }

    sign = "+" if delta >= 0 else ""
    print(f"  {ds_name:30s}  {old_acc:8.4f}  {new_acc:10.4f}  "
          f"{sign}{delta:7.4f}")
    print(f"    Benign acc:    {new['benign_acc']:.4f}")
    print(f"    Malignant acc: {new['malignant_acc']:.4f}")

# ════════════════════════════════════════════════════════
# Step 4: Bar chart comparison
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

ds_names  = list(threshold_results.keys())
acc_05    = [threshold_results[d]["tau_05"]["overall"]
             for d in ds_names]
acc_opt   = [threshold_results[d]["tau_opt"]["overall_acc"]
             for d in ds_names]

x = np.arange(len(ds_names))
w = 0.35
b1 = ax.bar(x-w/2, acc_05,  w, label="τ = 0.5 (default)",
            color="#90A4AE", alpha=0.85)
b2 = ax.bar(x+w/2, acc_opt, w,
            label=f"τ = {optimal_tau:.3f} (Youden optimal)",
            color="#1565C0", alpha=0.85)

for bar, val in zip(b1, acc_05):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.01,
            f"{val:.3f}", ha="center", va="bottom",
            fontsize=9, color="#555555")
for bar, val in zip(b2, acc_opt):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.01,
            f"{val:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#1565C0")

ax.set_xticks(x)
ax.set_xticklabels(ds_names, rotation=10)
ax.set_ylabel("Classification Accuracy", fontsize=12)
ax.set_title(
    "Classification Accuracy: Default vs Optimal Threshold\n"
    f"(Optimal τ = {optimal_tau:.3f}, determined via Youden Index on training validation set)",
    fontsize=11, fontweight="bold", color="#1F3E64")
ax.legend(fontsize=10)
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(OUT_DIR/"fig_threshold_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: fig_threshold_comparison.png")

# ════════════════════════════════════════════════════════
# Save final results
# ════════════════════════════════════════════════════════
final = {
    "optimal_threshold": optimal_tau,
    "auc": float(auc),
    "youden_index": float(youden[best_idx]),
    "at_optimal_tau": {
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(1-fpr[best_idx]),
    },
    "classification_results": threshold_results,
}

def convert(o):
    if isinstance(o, (np.integer, np.floating)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (bool, np.bool_)): return bool(o)
    raise TypeError(type(o))

with open(OUT_DIR/"threshold_results.json","w") as f:
    json.dump(final, f, indent=2, default=convert)

print(f"  Saved: threshold_results.json")
print(f"\n{'='*65}")
print(f"  Optimal threshold : {optimal_tau:.4f}")
print(f"  AUC               : {auc:.4f}")
print(f"{'='*65}")
