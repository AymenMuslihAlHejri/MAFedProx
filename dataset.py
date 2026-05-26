"""
====================================================================
dataset.py  —  Final v2
8 datasets: 4 Ultrasound + 4 Mammogram
Modality-Specific Augmentation
Capped Balanced Sampler

Scanner logic per dataset
─────────────────────────────────────────────────────────────────
BUSI      : same folder | img (no _mask) + stem_mask.*
BLUIS     : same folder | "N Image.BMP" + "N Lesion.BMP"
BrEaST    : same folder | img (no _tumor) + stem_tumor.*
BUS_BRA   : Images/ + Masks/ | bus_*.png + mask_*.png
INbreast  : Images/ + Masks/ | img_*.png + img_*.png (same name)
DMID      : same folder | IMG*.png + IMG*_mask.png
CBIS_DDSM : Images/ + Masks/ | Ben/Mal_*.jpg + mask_Ben/Mal_*.jpg
AIISLAB   : same folder | img (no "mask") + file with "mask"+stem
====================================================================
"""

import os, glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config as C

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# ════════════════════════════════════════════════════════════════
# 1.  Modality-Specific Augmentation
# ════════════════════════════════════════════════════════════════
def get_transforms(split, modality="ultrasound",
                   image_size=C.IMAGE_SIZE):
    """
    Ultrasound : speckle noise, blur, shadow simulation
    Mammogram  : CLAHE, gamma, subtle noise, sharpening
    """
    resize    = [A.Resize(image_size, image_size)]
    normalise = [A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()]

    common_geo = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.4),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    ]

    us_intensity = [
        A.GaussNoise(p=0.45),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.45),
        A.RandomShadow(p=0.2),
    ]

    mammo_intensity = [
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.2, p=0.35),
        A.GaussNoise(p=0.25),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.2),
    ]

    if split == "train":
        intensity = us_intensity if modality == "ultrasound" \
                    else mammo_intensity
        return A.Compose(resize + common_geo + intensity + normalise,
                 is_check_shapes=False)
    return A.Compose(resize + normalise,
                 is_check_shapes=False)


# ════════════════════════════════════════════════════════════════
# 2.  I/O helpers
# ════════════════════════════════════════════════════════════════
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _is_img(f):
    return f.lower().endswith(IMG_EXTS)

def _load_gray(path):
    return np.array(Image.open(path).convert("L"))

def _load_mask(path):
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


# ════════════════════════════════════════════════════════════════
# 3.  Scanners
# ════════════════════════════════════════════════════════════════

# ── US-01: BUSI ──────────────────────────────────────────────────
def _scan_BUSI(root):
    """
    US/01 Dataset_BUSI_with_GT/
        benign/    benign (N).png   benign (N)_mask.png
        malignant/ malignant (N).png malignant (N)_mask.png
    """
    out, base = [], os.path.join(root, "01 Dataset_BUSI_with_GT")
    for cls, lbl in [("benign", 0), ("malignant", 1)]:
        d = os.path.join(base, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if "_mask" in f or not _is_img(f):
                continue
            stem  = os.path.splitext(f)[0]
            masks = glob.glob(os.path.join(d, stem + "_mask.*"))
            if masks:
                out.append(dict(image=os.path.join(d, f),
                                mask=masks[0], label=lbl,
                                dataset="BUSI"))
    return out


# ── US-02: BLUIS (Iran) ──────────────────────────────────────────
def _scan_BLUIS(root):
    """
    US/02 Ultrasound dataset Iran/
        Benign/    "N Benign Image.BMP"     "N Benign Lesion.BMP"
        Malignant/ "N Malignant Image.BMP"  "N Malignant Lesion.BMP"
    """
    out, base = [], os.path.join(root, "02 Ultrasound dataset Iran")
    for cls, lbl in [("Benign", 0), ("Malignant", 1)]:
        d = os.path.join(base, cls)
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        for f in files:
            if "Image" not in f or not _is_img(f):
                continue
            num     = f.split()[0]
            lesions = [x for x in files
                       if x.startswith(num + " ") and "Lesion" in x]
            if lesions:
                out.append(dict(image=os.path.join(d, f),
                                mask=os.path.join(d, lesions[0]),
                                label=lbl, dataset="BLUIS"))
    return out


# ── US-03: BrEaST ────────────────────────────────────────────────
def _scan_BrEaST(root):
    """
    US/03 BReaST/
        Benign/    case001.png   case001_tumor.png
        Malignant/ case005.png   case005_tumor.png
    """
    out, base = [], os.path.join(root, "03 BReaST")
    for cls, lbl in [("Benign", 0), ("Malignant", 1)]:
        d = os.path.join(base, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if "_tumor" in f or not _is_img(f):
                continue
            stem  = os.path.splitext(f)[0]
            masks = glob.glob(os.path.join(d, stem + "_tumor.*"))
            if masks:
                out.append(dict(image=os.path.join(d, f),
                                mask=masks[0], label=lbl,
                                dataset="BrEaST"))
    return out


# ── US-04: BUS_BRA ───────────────────────────────────────────────
def _scan_BUS_BRA(root):
    """
    US/04 BUS_BRA/
        benign/
            Images/  bus_0002-l.png
            Masks/   mask_0002-l.png   (بادئة mask_ بدون bus)
        malignant/
            Images/  bus_0001-l.png
            Masks/   mask_0001-l.png
    """
    out, base = [], os.path.join(root, "04 BUS_BRA")
    for cls, lbl in [("benign", 0), ("malignant", 1)]:
        img_dir  = os.path.join(base, cls, "Images")
        mask_dir = os.path.join(base, cls, "Masks")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if not _is_img(f):
                continue
            stem = os.path.splitext(f)[0]          # bus_0002-l

            # mask_0002-l.png  →  نزيل "bus_" ونضع "mask_"
            stem_no_bus = stem.replace("bus_", "")  # 0002-l
            masks = (
                glob.glob(os.path.join(mask_dir, "mask_" + stem_no_bus + ".*")) or
                glob.glob(os.path.join(mask_dir, "mask_" + stem + ".*")) or
                glob.glob(os.path.join(mask_dir, stem_no_bus + ".*"))
            )
            if masks:
                out.append(dict(image=os.path.join(img_dir, f),
                                mask=masks[0], label=lbl,
                                dataset="BUS_BRA"))
    return out



# ── US-04b: BUS_UC ───────────────────────────────────────────────
def _scan_BUS_UC(root):
    """
    US/BUS_UC/
        Benign/
            images/  01.png
            masks/   01.png   ← نفس الاسم تماماً
        Malignant/
            images/  001.png
            masks/   001.png
    """
    out, base = [], os.path.join(root, "BUS_UC")
    for cls, lbl in [("Benign", 0), ("Malignant", 1)]:
        img_dir  = os.path.join(base, cls, "images")
        mask_dir = os.path.join(base, cls, "masks")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if not _is_img(f):
                continue
            stem  = os.path.splitext(f)[0]
            masks = glob.glob(os.path.join(mask_dir, stem + ".*"))
            if masks:
                out.append(dict(image=os.path.join(img_dir, f),
                                mask=masks[0], label=lbl,
                                dataset="BUS_UC"))
    return out

# ── Mammo-01: INbreast ───────────────────────────────────────────
def _scan_INbreast(root):
    """
    Mammo/01 INbreast/
        Benign/
            Images/  img_1_B.png
            Masks/   img_1_B.png   ← نفس الاسم تماماً
        Malignant/
            Images/  img_2_M.png
            Masks/   img_2_M.png
    """
    out, base = [], os.path.join(root, "01 INbreast")
    for cls, lbl in [("Benign", 0), ("Malignant", 1)]:
        img_dir  = os.path.join(base, cls, "Images")
        mask_dir = os.path.join(base, cls, "Masks")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if not _is_img(f):
                continue
            stem  = os.path.splitext(f)[0]
            # نفس الاسم في مجلد Masks
            masks = glob.glob(os.path.join(mask_dir, stem + ".*"))
            if masks:
                out.append(dict(image=os.path.join(img_dir, f),
                                mask=masks[0], label=lbl,
                                dataset="INbreast"))
    return out


# ── Mammo-02: DMID ───────────────────────────────────────────────
def _scan_DMID(root):
    """
    Mammo/02 DMID/
        benign/    IMG001.png   IMG001_mask.png
        malignant/ IMG001.png   IMG001_mask.png
    """
    out, base = [], os.path.join(root, "02 DMID")
    for cls, lbl in [("benign", 0), ("malignant", 1)]:
        d = os.path.join(base, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if "_mask" in f.lower() or not _is_img(f):
                continue
            stem  = os.path.splitext(f)[0]
            masks = (
                glob.glob(os.path.join(d, stem + "_mask.*")) or
                glob.glob(os.path.join(d, stem.upper() + "_mask.*"))
            )
            if masks:
                out.append(dict(image=os.path.join(d, f),
                                mask=masks[0], label=lbl,
                                dataset="DMID"))
    return out


# ── Mammo-03: CBIS_DDSM ──────────────────────────────────────────
def _scan_CBIS_DDSM(root):
    """
    Mammo/03 CBIS-DDSM/
        Benign/
            Images/  Ben_0001.jpg
            Masks/   mask_Ben_0001.jpg   ← بادئة mask_
        Malignant/
            Images/  Mal_0001.jpg
            Masks/   mask_Mal_0001.jpg
    """
    out, base = [], os.path.join(root, "03 CBIS-DDSM")
    for cls, lbl in [("Benign", 0), ("Malignant", 1)]:
        img_dir  = os.path.join(base, cls, "Images")
        mask_dir = os.path.join(base, cls, "Masks")
        if not os.path.isdir(img_dir):
            print(f"  [CBIS_DDSM] '{img_dir}' not found — skipping")
            continue
        for f in os.listdir(img_dir):
            if not _is_img(f):
                continue
            stem  = os.path.splitext(f)[0]   # Ben_0001
            ext   = os.path.splitext(f)[1]   # .jpg
            # mask اسمه mask_ + نفس اسم الصورة
            mask_name = "mask_" + stem + ext  # mask_Ben_0001.jpg
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                out.append(dict(image=os.path.join(img_dir, f),
                                mask=mask_path, label=lbl,
                                dataset="CBIS_DDSM"))
    return out


# ── Mammo-04: AIISLAB ────────────────────────────────────────────
def _scan_AIISLAB(root):
    """
    Mammo/04 AIISLAB/
        Benign/
            102139_L_CC.jpg        ← صورة (لا تحتوي "mask")
            mask_M102139_L_CC.jpg  ← mask  (يحتوي "mask" + stem)
        Malignant/
            42574_L_CC.jpg
            mask_42574_L_CC.jpg
    """
    out, base = [], os.path.join(root, "04 AIISLAB")
    for cls, lbl in [("Benign", 0), ("Malignant", 1)]:
        d = os.path.join(base, cls)
        if not os.path.isdir(d):
            continue
        all_files = os.listdir(d)

        # الصور: ملفات لا تحتوي "mask" في الاسم
        img_files = [f for f in all_files
                     if "mask" not in f.lower() and _is_img(f)]

        for f in img_files:
            stem = os.path.splitext(f)[0]   # e.g. 102139_L_CC

            # ابحث عن mask يحتوي "mask" في الاسم ويحتوي stem
            masks = [x for x in all_files
                     if "mask" in x.lower()
                     and stem in x
                     and _is_img(x)]
            if masks:
                out.append(dict(image=os.path.join(d, f),
                                mask=os.path.join(d, masks[0]),
                                label=lbl, dataset="AIISLAB"))
    return out


# ════════════════════════════════════════════════════════════════
# 4.  Scanner registry
# ════════════════════════════════════════════════════════════════
SCANNERS = {
    "BUSI"     : (_scan_BUSI,      "us"),
    #"BLUIS"    : (_scan_BLUIS,     "us"),
    "BUS_UC"   : (_scan_BUS_UC,    "us"),
    "BrEaST"   : (_scan_BrEaST,    "us"),
    "BUS_BRA"  : (_scan_BUS_BRA,   "us"),
    "INbreast" : (_scan_INbreast,  "mammo"),
    "DMID"     : (_scan_DMID,      "mammo"),
    "CBIS_DDSM": (_scan_CBIS_DDSM, "mammo"),
    "AIISLAB"  : (_scan_AIISLAB,   "mammo"),
}

def _root(name):
    _, rk = SCANNERS[name]
    return C.US_ROOT if rk == "us" else C.MAMMO_ROOT


# ════════════════════════════════════════════════════════════════
# 5.  PyTorch Dataset
# ════════════════════════════════════════════════════════════════
class BreastCancerDataset(Dataset):
    """
    One FL client = one dataset.
    Returns: image (1,H,W), mask (1,H,W), label, dataset, modality
    """

    def __init__(self, dataset_name,
                 split="train",
                 us_root=C.US_ROOT,
                 mammo_root=C.MAMMO_ROOT,
                 image_size=C.IMAGE_SIZE,
                 val_ratio=C.VAL_RATIO,
                 test_ratio=C.TEST_RATIO,
                 seed=C.SEED):

        if dataset_name not in SCANNERS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Options: {list(SCANNERS)}")

        self.dataset_name = dataset_name
        self.modality     = C.MODALITY_MAP[dataset_name]

        # scan
        fn, rk = SCANNERS[dataset_name]
        root   = us_root if rk == "us" else mammo_root
        all_s  = fn(root)

        if not all_s:
            raise AssertionError(
                f"No samples found for '{dataset_name}' in '{root}'")

        # split
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(all_s))
        n   = len(idx)
        nt  = max(1, int(n * test_ratio))
        nv  = max(1, int(n * val_ratio))
        chosen = {"train": idx[:n-nv-nt],
                  "val"  : idx[n-nv-nt:n-nt],
                  "test" : idx[n-nt:]}[split]
        self.samples = [all_s[i] for i in chosen]

        # modality-specific transform
        self.transform = get_transforms(split, self.modality, image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        image = _load_gray(s["image"])
        mask  = _load_mask(s["mask"])
        aug   = self.transform(image=image[..., np.newaxis], mask=mask)
        return {
            "image"   : aug["image"].float(),
            "mask"    : aug["mask"].float().unsqueeze(0),
            "label"   : torch.tensor(s["label"], dtype=torch.float32),
            "dataset" : s["dataset"],
            "modality": self.modality,
        }

    @property
    def labels(self):
        return [s["label"] for s in self.samples]


# ════════════════════════════════════════════════════════════════
# 6.  Capped Balanced Sampler
# ════════════════════════════════════════════════════════════════
def make_balanced_sampler(dataset, cap_ratio=3.0):
    labels = np.array(dataset.labels)
    n      = len(labels)
    cls, counts = np.unique(labels, return_counts=True)
    raw    = {c: n / cnt for c, cnt in zip(cls, counts)}
    minw   = min(raw.values())
    capped = {c: min(w, minw * cap_ratio) for c, w in raw.items()}
    sw     = torch.tensor([capped[l] for l in labels],
                          dtype=torch.double)
    return WeightedRandomSampler(sw, num_samples=n, replacement=True)


# ════════════════════════════════════════════════════════════════
# 7.  DataLoader factory
# ════════════════════════════════════════════════════════════════
def get_dataloader(dataset_name,
                   split        = "train",
                   us_root      = C.US_ROOT,
                   mammo_root   = C.MAMMO_ROOT,
                   batch_size   = C.BATCH_SIZE,
                   image_size   = C.IMAGE_SIZE,
                   num_workers  = C.NUM_WORKERS,
                   seed         = C.SEED,
                   use_balanced = True):

    ds = BreastCancerDataset(
        dataset_name = dataset_name,
        split        = split,
        us_root      = us_root,
        mammo_root   = mammo_root,
        image_size   = image_size,
        seed         = seed,
    )
    sampler = None
    shuffle = (split == "train")
    if split == "train" and use_balanced:
        sampler = make_balanced_sampler(ds)
        shuffle = False

    return DataLoader(ds,
                      batch_size  = batch_size,
                      sampler     = sampler,
                      shuffle     = shuffle,
                      num_workers = num_workers,
                      pin_memory  = True,
                      drop_last   = (split == "train"))


# ════════════════════════════════════════════════════════════════
# 8.  Statistics helper
# ════════════════════════════════════════════════════════════════
def print_stats():
    print("\n" + "="*65)
    print(f"  {'Dataset':12s} {'Modality':12s} "
          f"{'Train':>6} {'Val':>5} {'Test':>5} {'Total':>6}")
    print("-"*65)
    grand = 0
    for name in SCANNERS:
        try:
            tr = BreastCancerDataset(name, "train")
            vl = BreastCancerDataset(name, "val")
            te = BreastCancerDataset(name, "test")
            tot = len(tr) + len(vl) + len(te)
            grand += tot
            print(f"  {name:12s} {C.MODALITY_MAP[name]:12s} "
                  f"{len(tr):>6} {len(vl):>5} {len(te):>5} {tot:>6}")
        except AssertionError as e:
            print(f"  {name:12s} {'MISSING':12s}   {e}")
    print("-"*65)
    print(f"  {'TOTAL':12s} {'':12s} {'':>6} {'':>5} {'':>5} {grand:>6}")
    print("="*65 + "\n")


# ════════════════════════════════════════════════════════════════
# 9.  Sanity check
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print_stats()
    print("Batch check:")
    for name in SCANNERS:
        try:
            loader = get_dataloader(name, split="train", batch_size=2)
            b      = next(iter(loader))
            lbl    = b["label"].tolist()
            print(f"  [OK]  {name:12s} | {b['modality'][0]:12s} | "
                  f"img={tuple(b['image'].shape)} | labels={lbl}")
        except Exception as e:
            print(f"  [ERR] {name:12s}: {e}")
