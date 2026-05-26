"""
MAFedProx — Qualitative Results Visualization
==============================================
Generates:
  Fig A: Segmentation results (US + Mammo)
  Fig B: Grad-CAM XAI
  Fig C: Failure Analysis
"""

import os, sys, json, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

BASE = Path(r"C:\Users\Asus\Python code\FL_MuL_BC")
sys.path.insert(0, str(BASE))

import config as C
from model   import MultiTaskBreastModel
from dataset import BreastCancerDataset
from torch.utils.data import DataLoader

DEVICE   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR  = BASE / "results" / "qualitative_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Colours ───────────────────────────────────────────────────
GT_COLOR   = np.array([0, 255, 0],   dtype=np.uint8)   # green
PRED_COLOR = np.array([255, 0, 0],   dtype=np.uint8)   # red
OVER_ALPHA = 0.4

# ── Load best model ───────────────────────────────────────────
import pickle

model_path = Path(r"C:/Users/Asus/Python code/FL_MuL_BC/results_exp2/fl/best_model.pkl")
if not model_path.exists():
    # Search
    for p in (BASE / "results").rglob("best_model*.pkl"):
        model_path = p
        break

print(f"Loading model: {model_path}")
with open(model_path, "rb") as f:
    data = pickle.load(f)

# Handle different save formats
if isinstance(data, list):
    # Saved as list of numpy arrays (ndarrays)
    import numpy as np
    model = MultiTaskBreastModel().to(DEVICE)
    params = dict(model.named_parameters())
    keys   = list(params.keys())
    state  = {}
    for i, (k, v) in enumerate(model.state_dict().items()):
        arr = data[i]
        state[k] = torch.tensor(arr).reshape(v.shape)
    model.load_state_dict(state)
elif isinstance(data, dict):
    model = MultiTaskBreastModel().to(DEVICE)
    model.load_state_dict(data)
else:
    # Maybe it's the model itself
    model = data
    model = model.to(DEVICE)

model.eval()

print("Model loaded ✓")


# ════════════════════════════════════════════════════════════
# Helper: denormalise image for display
# ════════════════════════════════════════════════════════════
def denorm(img_tensor):
    """[-1,1] → [0,1]"""
    img = img_tensor.cpu().numpy()
    if img.ndim == 3:
        img = img[0]  # grayscale
    img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)

def make_overlay(img, mask, color, alpha=OVER_ALPHA):
    """img: (H,W) float [0,1], mask: (H,W) bool"""
    rgb = np.stack([img]*3, axis=-1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    overlay = rgb_uint8.copy()
    overlay[mask] = (
        (1-alpha) * rgb_uint8[mask] + alpha * color
    ).astype(np.uint8)
    return overlay / 255.0


# ════════════════════════════════════════════════════════════
# Grad-CAM
# ════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, layer_name="encoder.layer4"):
        self.model    = model
        self.gradients = None
        self.activations = None
        self._hook(layer_name)

    def _hook(self, layer_name):
        # Navigate to layer
        layer = self.model
        for part in layer_name.split("."):
            layer = getattr(layer, part)

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(bwd_hook)

    def __call__(self, img_tensor, modality, target="seg"):
        self.model.zero_grad()
        img_tensor = img_tensor.clone().requires_grad_(True)

        seg_pred, cls_pred = self.model(img_tensor, modality=modality)

        if target == "seg":
            score = seg_pred.mean()
        else:
            score = cls_pred.mean()

        score.backward()

        grads = self.gradients        # (1, C, H, W)
        acts  = self.activations      # (1, C, H, W)

        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(256, 256),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


# ════════════════════════════════════════════════════════════
# Sample selection
# ════════════════════════════════════════════════════════════
def get_samples(dataset_name, modality, n_benign=2, n_malign=2,
                split="test"):
    """Get representative samples from a dataset."""
    ds = BreastCancerDataset(dataset_name, split=split,
                             image_size=C.IMAGE_SIZE, seed=SEED)
    benign_idx  = [i for i in range(len(ds))
                   if ds.labels[i] == 0]
    malign_idx  = [i for i in range(len(ds))
                   if ds.labels[i] == 1]

    # Pick with best dice if possible (use random otherwise)
    picked = []
    for label, idx_list, n in [(0, benign_idx, n_benign),
                                (1, malign_idx, n_malign)]:
        chosen = random.sample(idx_list, min(n, len(idx_list)))
        for i in chosen:
            sample = ds[i]
            picked.append({
                "image"   : sample["image"].unsqueeze(0).to(DEVICE),
                "mask"    : sample["mask"].squeeze().cpu().numpy(),
                "label"   : int(sample["label"].item()),
                "modality": modality,
                "dataset" : dataset_name,
                "idx"     : i
            })
    return picked


# ════════════════════════════════════════════════════════════
# Run inference on a sample
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def infer(sample):
    seg_pred, cls_pred = model(sample["image"],
                               modality=sample["modality"])
    pred_mask = (seg_pred.squeeze().cpu().numpy() > 0.5)
    cls_prob  = torch.sigmoid(cls_pred).item()

    # Metrics
    gt   = sample["mask"].astype(bool)
    pred = pred_mask.astype(bool)
    inter = (gt & pred).sum()
    dice  = (2*inter) / (gt.sum() + pred.sum() + 1e-6)

    return {
        "pred_mask": pred_mask,
        "cls_prob" : cls_prob,
        "dice"     : float(dice),
    }


# ════════════════════════════════════════════════════════════
# FIGURE A — Segmentation Results (4×4 grid)
# ════════════════════════════════════════════════════════════
def plot_segmentation(rows_data, title, fname):
    """
    rows_data: list of dicts with image, gt_mask, pred_mask,
               dice, cls_prob, label, dataset
    """
    n_rows = len(rows_data)
    n_cols = 4  # Image | GT | Prediction | Overlay

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 4.2 * n_rows),
        facecolor="white"
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "Ground Truth",
                  "Prediction", "Overlay (GT=green, Pred=red)"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=13, fontweight="bold",
                             color="#1F3E64", pad=8)

    for i, row in enumerate(rows_data):
        img    = denorm(row["image"].squeeze(0))
        gt     = row["gt_mask"].astype(bool)
        pred   = row["pred_mask"].astype(bool)
        dice   = row["dice"]
        prob   = row["cls_prob"]
        label  = "Malignant" if row["label"] == 1 else "Benign"
        ds     = row["dataset"]

        # Row label
        row_label = (f"{ds}\n{label}\n"
                     f"DSC={dice:.3f}\nP(mal)={prob:.3f}")
        axes[i, 0].set_ylabel(row_label, fontsize=9.5,
                              rotation=0, labelpad=90,
                              va="center", ha="right")

        # Col 0: Image
        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)

        # Col 1: GT mask
        gt_rgb = np.zeros((*img.shape, 3))
        gt_rgb[gt] = [0, 1, 0]
        axes[i, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(gt_rgb, alpha=0.5)

        # Col 2: Predicted mask
        pred_rgb = np.zeros((*img.shape, 3))
        pred_rgb[pred] = [1, 0, 0]
        axes[i, 2].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(pred_rgb, alpha=0.5)

        # Col 3: Overlay (both)
        axes[i, 3].imshow(img, cmap="gray", vmin=0, vmax=1)
        both_rgb = np.zeros((*img.shape, 3))
        both_rgb[gt]   = [0, 1, 0]
        both_rgb[pred] = [1, 0, 0]
        # TP = both → yellow
        tp = gt & pred
        both_rgb[tp] = [1, 1, 0]
        axes[i, 3].imshow(both_rgb, alpha=0.5)

        # Border colour by correct/incorrect
        border_c = "#1B5E20" if dice > 0.6 else "#B71C1C"
        for j in range(n_cols):
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor(border_c)
                spine.set_linewidth(2)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    # Legend
    patches = [
        mpatches.Patch(color="green",  alpha=0.6, label="GT only"),
        mpatches.Patch(color="red",    alpha=0.6, label="Pred only"),
        mpatches.Patch(color="yellow", alpha=0.6, label="TP (both)"),
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(title, fontsize=15, fontweight="bold",
                 color="#1F3E64", y=1.01)
    plt.tight_layout()
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved: {path.name}")


# ════════════════════════════════════════════════════════════
# FIGURE B — Grad-CAM XAI
# ════════════════════════════════════════════════════════════
def plot_gradcam(rows_data, fname):
    gradcam = GradCAM(model)
    n = len(rows_data)
    fig, axes = plt.subplots(n, 3,
                             figsize=(12, 4.2*n),
                             facecolor="white")
    if n == 1:
        axes = axes[np.newaxis, :]

    for j, ct in enumerate(["Input Image",
                             "Predicted Mask",
                             "Grad-CAM (Encoder Attention)"]):
        axes[0, j].set_title(ct, fontsize=13,
                             fontweight="bold", color="#1F3E64")

    for i, row in enumerate(rows_data):
        img   = denorm(row["image"].squeeze(0))
        pred  = row["pred_mask"]
        dice  = row["dice"]
        prob  = row["cls_prob"]
        label = "Malignant" if row["label"]==1 else "Benign"

        # Grad-CAM
        cam = gradcam(row["image"], row["modality"], target="seg")

        row_label = (f"{row['dataset']}\n{label}\n"
                     f"DSC={dice:.3f}\nP={prob:.3f}")
        axes[i, 0].set_ylabel(row_label, fontsize=9.5,
                              rotation=0, labelpad=90,
                              va="center", ha="right")

        axes[i, 0].imshow(img, cmap="gray")

        pred_rgb = np.zeros((*img.shape, 3))
        pred_rgb[pred > 0.5] = [1, 0, 0]
        axes[i, 1].imshow(img, cmap="gray")
        axes[i, 1].imshow(pred_rgb, alpha=0.5)

        axes[i, 2].imshow(img, cmap="gray")
        axes[i, 2].imshow(cam, cmap="jet", alpha=0.55, vmin=0, vmax=1)

        for j in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                sp.set_edgecolor("#2E5090"); sp.set_linewidth(2)

    # Colourbar for CAM
    sm = plt.cm.ScalarMappable(cmap="jet",
                               norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, 2], fraction=0.03,
                        pad=0.04)
    cbar.set_label("Attention intensity", fontsize=10)

    fig.suptitle(
        "Grad-CAM Encoder Attention — MAFedProx (Exp 2, Round 88)",
        fontsize=14, fontweight="bold", color="#1F3E64", y=1.01)
    plt.tight_layout()
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved: {path.name}")


# ════════════════════════════════════════════════════════════
# FIGURE C — Failure Cases
# ════════════════════════════════════════════════════════════
def plot_failures(rows_data, fname):
    n = len(rows_data)
    fig, axes = plt.subplots(n, 4,
                             figsize=(16, 4.2*n),
                             facecolor="white")
    if n == 1:
        axes = axes[np.newaxis, :]

    for j, ct in enumerate(["Input Image", "Ground Truth",
                             "Prediction", "Error Map"]):
        axes[0, j].set_title(ct, fontsize=13,
                             fontweight="bold", color="#B71C1C")

    for i, row in enumerate(rows_data):
        img  = denorm(row["image"].squeeze(0))
        gt   = row["gt_mask"].astype(bool)
        pred = row["pred_mask"].astype(bool)
        dice = row["dice"]
        prob = row["cls_prob"]
        label = "Malignant" if row["label"]==1 else "Benign"

        # Error map: FP=red, FN=blue
        err = np.zeros((*img.shape, 3))
        fp = pred & ~gt; err[fp] = [1, 0, 0]   # False Positive
        fn = gt & ~pred; err[fn] = [0, 0, 1]   # False Negative

        row_label = (f"{row['dataset']}\n{label}\n"
                     f"DSC={dice:.3f}\nP={prob:.3f}")
        axes[i, 0].set_ylabel(row_label, fontsize=9.5,
                              rotation=0, labelpad=90,
                              va="center", ha="right")

        axes[i, 0].imshow(img, cmap="gray")

        gt_rgb = np.zeros((*img.shape, 3))
        gt_rgb[gt] = [0, 1, 0]
        axes[i, 1].imshow(img, cmap="gray")
        axes[i, 1].imshow(gt_rgb, alpha=0.5)

        pred_rgb = np.zeros((*img.shape, 3))
        pred_rgb[pred] = [1, 0, 0]
        axes[i, 2].imshow(img, cmap="gray")
        axes[i, 2].imshow(pred_rgb, alpha=0.5)

        axes[i, 3].imshow(img, cmap="gray")
        axes[i, 3].imshow(err, alpha=0.6)

        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                sp.set_edgecolor("#B71C1C"); sp.set_linewidth(2)

    patches = [
        mpatches.Patch(color="red",  alpha=0.7, label="False Positive"),
        mpatches.Patch(color="blue", alpha=0.7, label="False Negative"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Failure Case Analysis — MAFedProx (Low DSC Clients)",
        fontsize=14, fontweight="bold", color="#B71C1C", y=1.01)
    plt.tight_layout()
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved: {path.name}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  Collecting samples...")
    print("="*60)

    # ── US clients ───────────────────────────────────────────
    us_samples_raw = []
    for ds_name, mod in [("BUSI","ultrasound"),
                          ("BUS_UC","ultrasound"),
                          ("BrEaST","ultrasound"),
                          ("BUS_BRA","ultrasound")]:
        try:
            s = get_samples(ds_name, mod, n_benign=1, n_malign=1)
            us_samples_raw.extend(s)
            print(f"  {ds_name}: {len(s)} samples")
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")

    # ── Mammo clients ────────────────────────────────────────
    mm_samples_raw = []
    for ds_name, mod in [("INbreast","mammogram"),
                          ("DMID","mammogram"),
                          ("CBIS_DDSM","mammogram"),
                          ("AIISLAB","mammogram")]:
        try:
            s = get_samples(ds_name, mod, n_benign=1, n_malign=1)
            mm_samples_raw.extend(s)
            print(f"  {ds_name}: {len(s)} samples")
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")

    # ── Run inference ────────────────────────────────────────
    print("\n  Running inference...")

    def process(samples):
        out = []
        for s in samples:
            res = infer(s)
            out.append({
                **s,
                "gt_mask" : s["mask"],
                "pred_mask": res["pred_mask"],
                "dice"    : res["dice"],
                "cls_prob": res["cls_prob"],
            })
        return out

    us_samples = process(us_samples_raw)
    mm_samples = process(mm_samples_raw)

    print(f"  US:    {len(us_samples)} samples processed")
    print(f"  Mammo: {len(mm_samples)} samples processed")

    # ── FIGURE A: Segmentation ───────────────────────────────
    print("\n" + "="*60)
    print("  Generating Figure A — Segmentation Results")
    print("="*60)

    # Pick 4 best US (highest dice) + 4 best Mammo
    us_sorted = sorted(us_samples, key=lambda x: -x["dice"])[:4]
    mm_sorted = sorted(mm_samples, key=lambda x: -x["dice"])[:4]

    plot_segmentation(
        us_sorted,
        "Figure A1 — MAFedProx Segmentation: Ultrasound Clients\n"
        "(Best 4 samples by DSC, Exp 2 Round 88)",
        "figA1_segmentation_US.png"
    )
    plot_segmentation(
        mm_sorted,
        "Figure A2 — MAFedProx Segmentation: Mammography Clients\n"
        "(Best 4 samples by DSC, Exp 2 Round 88)",
        "figA2_segmentation_Mammo.png"
    )

    # Combined: 1 best US + 1 best Mammo per class
    combined = (
        sorted([s for s in us_samples if s["label"]==0],
               key=lambda x: -x["dice"])[:1] +
        sorted([s for s in us_samples if s["label"]==1],
               key=lambda x: -x["dice"])[:1] +
        sorted([s for s in mm_samples if s["label"]==0],
               key=lambda x: -x["dice"])[:1] +
        sorted([s for s in mm_samples if s["label"]==1],
               key=lambda x: -x["dice"])[:1]
    )
    plot_segmentation(
        combined,
        "Figure A — MAFedProx Qualitative Results\n"
        "(US Benign | US Malignant | Mammo Benign | Mammo Malignant)",
        "figA_segmentation_combined.png"
    )

    # ── FIGURE B: Grad-CAM ───────────────────────────────────
    print("\n" + "="*60)
    print("  Generating Figure B — Grad-CAM XAI")
    print("="*60)

    # 2 US + 2 Mammo (1 benign + 1 malignant each)
    gradcam_samples = (
        sorted([s for s in us_samples if s["label"]==0],
               key=lambda x: -x["dice"])[:1] +
        sorted([s for s in us_samples if s["label"]==1],
               key=lambda x: -x["dice"])[:1] +
        sorted([s for s in mm_samples if s["label"]==0],
               key=lambda x: -x["dice"])[:1] +
        sorted([s for s in mm_samples if s["label"]==1],
               key=lambda x: -x["dice"])[:1]
    )
    plot_gradcam(gradcam_samples, "figB_gradcam_XAI.png")

    # ── FIGURE C: Failure Cases ──────────────────────────────
    print("\n" + "="*60)
    print("  Generating Figure C — Failure Analysis")
    print("="*60)

    # Worst performing samples
    all_samples  = us_samples + mm_samples
    worst = sorted(all_samples, key=lambda x: x["dice"])[:4]
    plot_failures(worst, "figC_failure_analysis.png")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  DICE SUMMARY")
    print("="*60)
    all_dice = [s["dice"] for s in all_samples]
    us_dice  = [s["dice"] for s in us_samples]
    mm_dice  = [s["dice"] for s in mm_samples]
    print(f"  All:   mean={np.mean(all_dice):.4f}  "
          f"min={np.min(all_dice):.4f}  max={np.max(all_dice):.4f}")
    print(f"  US:    mean={np.mean(us_dice):.4f}")
    print(f"  Mammo: mean={np.mean(mm_dice):.4f}")
    print(f"\n  Figures saved → {OUT_DIR}")

