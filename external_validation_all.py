"""
MAFedProx — Extended External Validation
=========================================
Three external datasets never seen during training:

  1. BUS-UCLM   (Ultrasound, Spain, 2025)
     264 images with lesion (174 benign, 90 malignant)
     RGB masks: Green=Benign, Red=Malignant

  2. BLUIS/Iran (Ultrasound, Iran, 2023)
     ~232 images (109 benign, 123 malignant)
     BMP images + TIF binary masks

  3. CDD-CESM   (Mammography, Egypt, 2022)
     601 images (276 benign, 325 malignant)
     PNG grayscale masks (0/255)

Usage:
  python external_validation_all.py
"""

import os, sys, json, pickle, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

# ── Paths ─────────────────────────────────────────────────────
BASE       = Path(r"C:\Users\Asus\Python code\FL_MuL_BC")
MODEL_PATH = Path(r"C:\Users\Asus\Python code\FL_MuL_BC\results_exp2\fl\best_model.pkl")
OUT_DIR    = BASE / "results" / "external_validation_ALL"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUSUCLM_ROOT = Path(r"D:\PhD\sable maam proposal 2025\Paper 2\05 BUS-UCLM Breast US\BUS-UCLM")
BLUIS_ROOT   = Path(r"H:\BC datasets\US\02 Ultrasound dataset Iran")
CESM_ROOT    = Path(r"H:\BC datasets\Mammo\05 CDD-CESM\Prepared")

sys.path.insert(0, str(BASE))
from model import MultiTaskBreastModel

DEVICE   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
print(f"Device : {DEVICE}")

# ════════════════════════════════════════════════════════════
# 1. Load model
# ════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════
# 2. Helpers
# ════════════════════════════════════════════════════════════
def preprocess_image(img_path, size=IMG_SIZE):
    img = np.array(Image.open(img_path).convert("L"))
    img = cv2.resize(img, (size, size),
                     interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 127.5 - 1.0

def preprocess_mask(mask_path, size=IMG_SIZE):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = cv2.resize(mask, (size, size),
                      interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(bool)

def infer_sample(img_arr, gt_mask, label, modality):
    img_t = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)\
                  .float().to(DEVICE)
    with torch.no_grad():
        seg_pred, cls_pred = model(img_t, modality=modality)

    pred = (seg_pred.squeeze().cpu().numpy() > 0.5)
    prob = torch.sigmoid(cls_pred).item()

    gt   = gt_mask.astype(bool)
    inter = (gt & pred).sum()
    union = (gt | pred).sum()
    tp = inter
    fp = pred.sum() - inter
    fn = gt.sum() - inter
    tn = (~gt & ~pred).sum()

    dice = (2*inter) / (gt.sum() + pred.sum() + 1e-6)
    iou  = inter / (union + 1e-6)
    sens = tp / (tp + fn + 1e-6)
    spec = tn / (tn + fp + 1e-6)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    pred_label = 1 if prob > 0.5 else 0
    cls_ok = int(pred_label == label)

    return {
        "dice": float(dice), "iou": float(iou),
        "sens": float(sens), "spec": float(spec),
        "acc":  float(acc),  "cls_prob": float(prob),
        "cls_correct": cls_ok, "pred_mask": pred,
        "gt_mask": gt, "img": img_arr,
    }

def summarise(results, name):
    if not results:
        return {}
    keys = ["dice","iou","sens","spec","acc","cls_correct"]
    return {
        "subset": name,
        "n": len(results),
        **{f"{k}_mean": float(np.mean([r[k] for r in results]))
           for k in keys},
        "dice_std": float(np.std([r["dice"] for r in results])),
    }

def print_summary(s):
    if not s: return
    print(f"  {s['subset']:20s}: n={s['n']:4d}  "
          f"DSC={s['dice_mean']:.4f}±{s['dice_std']:.4f}  "
          f"IoU={s['iou_mean']:.4f}  "
          f"Sens={s['sens_mean']:.4f}  "
          f"Spec={s['spec_mean']:.4f}  "
          f"ClsAcc={s['cls_correct_mean']:.4f}")

# ════════════════════════════════════════════════════════════
# 3. BUS-UCLM Parser
# ════════════════════════════════════════════════════════════
def parse_busuclm():
    print("\n[2a] Parsing BUS-UCLM...")
    img_dir  = BUSUCLM_ROOT / "images"
    mask_dir = BUSUCLM_ROOT / "masks"
    results  = []
    n_skip   = 0

    for img_path in sorted(img_dir.glob("*.png")):
        name = img_path.stem
        mask_path = mask_dir / f"{name}.png"
        if not mask_path.exists():
            continue

        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        green = (mask_rgb[:,:,1] > 100) & (mask_rgb[:,:,0] < 100)
        red   = (mask_rgb[:,:,0] > 100) & (mask_rgb[:,:,1] < 100)
        binary = (green | red).astype(np.uint8)

        if red.sum() > 10:
            label = 1
        elif green.sum() > 10:
            label = 0
        else:
            n_skip += 1
            continue

        gt   = cv2.resize(binary,(IMG_SIZE,IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        img  = preprocess_image(img_path)
        r    = infer_sample(img, gt, label, "ultrasound")
        r["name"] = name
        r["label"] = label
        results.append(r)

    print(f"  Processed: {len(results)} (skipped {n_skip} normal)")
    return results

# ════════════════════════════════════════════════════════════
# 4. BLUIS/Iran Parser
# ════════════════════════════════════════════════════════════
def parse_bluis():
    print("\n[2b] Parsing BLUIS (Iran)...")
    results = []

    for label, folder, keyword in [
        (0, BLUIS_ROOT / "Benign",    "Benign"),
        (1, BLUIS_ROOT / "Malignant", "Malignant"),
    ]:
        if not folder.exists():
            print(f"  [SKIP] {folder} not found")
            continue

        # Find all image files (not Lesion, not Mask)
        img_files = sorted([
            f for f in folder.glob("*.bmp")
            if "Image" in f.name
        ], key=lambda x: int(
            re.search(r'(\d+)', x.name).group(1)
        ))

        for img_path in img_files:
            # Find matching mask (.tif)
            num = re.search(r'(\d+)', img_path.name).group(1)
            mask_path = folder / f"{num} {keyword} Mask.tif"
            if not mask_path.exists():
                continue

            img  = preprocess_image(img_path)
            gt   = preprocess_mask(mask_path)
            r    = infer_sample(img, gt, label, "ultrasound")
            r["name"]  = img_path.stem
            r["label"] = label
            results.append(r)

    n_b = sum(1 for r in results if r["label"]==0)
    n_m = sum(1 for r in results if r["label"]==1)
    print(f"  Processed: {len(results)} "
          f"(benign={n_b}, malignant={n_m})")
    return results

# ════════════════════════════════════════════════════════════
# 5. CDD-CESM Parser
# ════════════════════════════════════════════════════════════
def parse_cddcesm():
    print("\n[2c] Parsing CDD-CESM...")
    results = []

    for label, folder in [
        (0, CESM_ROOT / "Benign"),
        (1, CESM_ROOT / "Malignant"),
    ]:
        img_dir  = folder / "Images"
        mask_dir = folder / "Masks"

        if not img_dir.exists():
            print(f"  [SKIP] {img_dir} not found")
            continue

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in \
               [".png",".jpg",".jpeg",".bmp",".tif"]:
                continue

            # Find mask with same stem or similar name
            stem = img_path.stem
            # Try exact match first
            mask_path = None
            for ext in [".png",".jpg",".tif",".bmp"]:
                candidate = mask_dir / f"{stem}_mask{ext}"
                if candidate.exists():
                    mask_path = candidate; break
                candidate = mask_dir / f"{stem}{ext}"
                if candidate.exists():
                    mask_path = candidate; break

            if mask_path is None:
                # Try replacing image indicator with mask
                # e.g. P100_L_DM_CC.jpg → P100_L_DM_CC_mask.png
                for mfile in mask_dir.glob(
                    f"{stem.split('_')[0]}*"
                ):
                    mask_path = mfile; break

            if mask_path is None:
                continue

            try:
                img = preprocess_image(img_path)
                gt  = preprocess_mask(mask_path)
                r   = infer_sample(img, gt, label, "mammogram")
                r["name"]  = stem
                r["label"] = label
                results.append(r)
            except Exception as e:
                print(f"  [ERR] {stem}: {e}")

    n_b = sum(1 for r in results if r["label"]==0)
    n_m = sum(1 for r in results if r["label"]==1)
    print(f"  Processed: {len(results)} "
          f"(benign={n_b}, malignant={n_m})")
    return results

# ════════════════════════════════════════════════════════════
# 6. Run all
# ════════════════════════════════════════════════════════════
busuclm = parse_busuclm()
bluis   = parse_bluis()
cesm    = parse_cddcesm()

# ════════════════════════════════════════════════════════════
# 7. Summaries
# ════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  EXTERNAL VALIDATION SUMMARY")
print("="*70)

datasets = {
    "BUS-UCLM (US, Spain)": busuclm,
    "BLUIS (US, Iran)":     bluis,
    "CDD-CESM (Mammo, Egypt)": cesm,
}

all_summaries = {}
for name, results in datasets.items():
    if not results: continue
    benign  = [r for r in results if r["label"]==0]
    malign  = [r for r in results if r["label"]==1]

    s_all    = summarise(results, "Overall")
    s_benign = summarise(benign,  "Benign")
    s_malign = summarise(malign,  "Malignant")

    print(f"\n  ── {name} ──")
    print_summary(s_all)
    print_summary(s_benign)
    print_summary(s_malign)

    all_summaries[name] = {
        "overall": s_all,
        "benign":  s_benign,
        "malignant": s_malign,
    }

# ════════════════════════════════════════════════════════════
# 8. Combined Figure — DSC comparison across datasets
# ════════════════════════════════════════════════════════════
print("\n[3] Generating figures...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                          facecolor="white")
fig.suptitle(
    "MAFedProx Extended External Validation\n"
    "(Model: Exp 2, Round 88 — never seen during training)",
    fontsize=13, fontweight="bold", color="#1F3E64")

colors = {"BUS-UCLM (US, Spain)":    "#1565C0",
          "BLUIS (US, Iran)":         "#2E7D32",
          "CDD-CESM (Mammo, Egypt)":  "#C62828"}

metrics = [("Dice Score","dice"),
           ("Sensitivity","sens"),
           ("Specificity","spec")]

for ax, (metric_name, metric_key) in zip(axes, metrics):
    vals, labels, cols = [], [], []
    for ds_name, results in datasets.items():
        if not results: continue
        v = np.mean([r[metric_key] for r in results])
        vals.append(v)
        labels.append(ds_name.split("(")[0].strip())
        cols.append(colors[ds_name])

    bars = ax.bar(labels, vals, color=cols, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(metric_name, fontweight="bold",
                 color="#1F3E64")
    ax.tick_params(axis="x", rotation=15)
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_ext_comparison.png",
            dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"  Saved: fig_ext_comparison.png")

# ════════════════════════════════════════════════════════════
# 9. DSC Distribution boxplot
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
data_box, tick_labels, box_colors = [], [], []

for ds_name, results in datasets.items():
    if not results: continue
    data_box.append([r["dice"] for r in results])
    tick_labels.append(ds_name)
    box_colors.append(colors[ds_name])

bp = ax.boxplot(data_box, patch_artist=True,
                tick_labels=tick_labels,
                medianprops=dict(color="gold", linewidth=2))
for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)

ax.set_ylabel("Dice Score", fontsize=12)
ax.set_title(
    "DSC Distribution — External Validation Datasets",
    fontsize=13, fontweight="bold", color="#1F3E64")
ax.set_ylim(-0.05, 1.05)
ax.tick_params(axis="x", rotation=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig_ext_dsc_boxplot.png",
            dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"  Saved: fig_ext_dsc_boxplot.png")

# ════════════════════════════════════════════════════════════
# 10. Save JSON
# ════════════════════════════════════════════════════════════
def convert(o):
    if isinstance(o, (np.integer, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bool):
        return bool(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))

output = {
    "model": "MAFedProx Exp2 Round88",
    "summaries": all_summaries,
    "per_image": {
        "BUS-UCLM": [
            {k: v for k, v in r.items()
             if k not in ["pred_mask","gt_mask","img"]}
            for r in busuclm
        ],
        "BLUIS": [
            {k: v for k, v in r.items()
             if k not in ["pred_mask","gt_mask","img"]}
            for r in bluis
        ],
        "CDD-CESM": [
            {k: v for k, v in r.items()
             if k not in ["pred_mask","gt_mask","img"]}
            for r in cesm
        ],
    }
}

with open(OUT_DIR / "ext_validation_all.json","w") as f:
    json.dump(output, f, indent=2, default=convert)

print(f"\n  Saved: ext_validation_all.json")
print(f"\n  All results → {OUT_DIR}")
print("="*70)
