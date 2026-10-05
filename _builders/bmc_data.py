#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bmc_data.py
===========

R3 - DATA LOADER FOR THE BMC REVISION

The only source this module knows is the frozen split manifest written by R2. It
never scans a directory, never opens the raw drive, and never decides a split. A
loader is built for exactly one (dataset, split) pair and can yield nothing else.

Four preprocessing choices are NOT decided here (see R3_PREPROCESSING_SPEC.md). They
are required constructor arguments with no default, so the loader refuses to build
until each has been stated by the person running it:

    intensity_rule  : "declared_range" | "per_image_minmax"
    aspect_rule     : "direct_resize" | "pad_to_square"
    clahe_role      : "augmentation" | "preprocessing" | "off"
    vflip_scope     : "all" | "ultrasound_only" | "off"

Run the self-tests with:
    python bmc_data.py --selftest
    python bmc_data.py --selftest --prep-root "<dataset preparing>"
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
SPLITS_DIR = "_bmc_splits"
MANIFEST_FILE = "BMC_SPLITS_MANIFEST.csv"
SUMMARY_FILE = "TABLE_4_split_summary.csv"
INTERNAL_SPLITS = ("train", "validation", "test")
EXTERNAL_SPLIT = "EXTERNAL_TEST_ONLY"
IMAGE_SIZE = 256

# R3 decisions, APPROVED and locked. The constructor still requires them to be
# passed explicitly; this constant is the single place they are recorded.
APPROVED_R3 = {"intensity_rule": "declared_range",      # D-R3-1
               "aspect_rule": "direct_resize",          # D-R3-2
               "clahe_role": "augmentation",            # D-R3-3a: train-only, mammography
               "vflip_scope": "ultrasound_only"}        # D-R3-3b

# declared content range per dataset (D-R3-1 option "a"); from the canonical bank
DECLARED_MAX: Dict[str, int] = {
    "BUSI": 255, "BUS-UC": 255, "BrEaST": 255, "BUS-BRA": 255,
    "DMID": 255, "AIISLAB": 255, "BLUIS": 255, "BUS-UCLM": 255, "CDD-CESM": 255,
    "CBIS-DDSM": 65535,     # 16-bit content
    "INbreast": 4095,       # 12-bit content in a 16-bit container
}


def read_rows(p: Path) -> List[dict]:
    with p.open(encoding="utf-8-sig", errors="replace") as fh:
        s = fh.read(8192)
        fh.seek(0)
        d = ";" if s.count(";") > s.count(",") else ","
        return list(csv.DictReader(fh, delimiter=d))


# ══════════════════════════════════════════════════════════════════════════════
# Reading and scaling
# ══════════════════════════════════════════════════════════════════════════════
SIXTEEN_BIT_MODES = ("I;16", "I;16B", "I;16L", "I")


def decode_image(im) -> np.ndarray:
    """Decode an open PIL image to one float32 channel at its NATIVE depth. A 16-bit
    image is taken as-is; it is never routed through mode 'L', which truncates."""
    if im.mode in SIXTEEN_BIT_MODES:
        return np.asarray(im, dtype=np.float32)
    if im.mode == "L":
        return np.asarray(im, dtype=np.float32)
    return np.asarray(im.convert("L"), dtype=np.float32)          # luminance


def scale_image(arr: np.ndarray, dataset: str, intensity_rule: str) -> np.ndarray:
    """Apply the intensity rule. Values are clipped to [0, 1] only as a guard; the
    self-test separately proves no value exceeds its declared range."""
    if intensity_rule == "declared_range":
        mx = DECLARED_MAX.get(dataset)
        if mx is None:
            raise ValueError(f"no declared range for {dataset}")
        arr = arr / float(mx)
    elif intensity_rule == "per_image_minmax":
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    else:
        raise ValueError(f"unknown intensity_rule {intensity_rule!r}")
    return np.clip(arr, 0.0, 1.0)


def read_image(path: Path, dataset: str, intensity_rule: str) -> np.ndarray:
    """One channel, float32 in [0, 1]."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        arr = decode_image(im)
    return scale_image(arr, dataset, intensity_rule)


def read_mask(path: Path) -> np.ndarray:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        m = np.asarray(im.convert("L"))
    return (m > 127).astype(np.uint8)


def pad_to_square(a: np.ndarray, value: float = 0.0) -> np.ndarray:
    h, w = a.shape[:2]
    s = max(h, w)
    out = np.full((s, s) + a.shape[2:], value, dtype=a.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = a
    return out


def resize_pair(img: np.ndarray, mask: np.ndarray, size: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Image bilinear, mask NEAREST. The mask stays exactly binary."""
    import cv2
    im = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mk = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return im, mk


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation (training only)
# ══════════════════════════════════════════════════════════════════════════════
def build_augmentation(modality: str, clahe_role: str, vflip_scope: str):
    """The legacy augmentation set, kept as design evidence, with the two
    D-R3-3 decisions applied. Geometric transforms are shared with the mask by
    albumentations' mask handling (nearest interpolation for masks)."""
    import warnings
    import albumentations as A
    warnings.filterwarnings("ignore", message=".*ShiftScaleRotate.*")
    vflip = (vflip_scope == "all"
             or (vflip_scope == "ultrasound_only" and modality == "ultrasound"))
    geo = [A.HorizontalFlip(p=0.5)]
    if vflip:
        geo.append(A.VerticalFlip(p=0.3))
    geo += [A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                               border_mode=0, p=0.4),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2)]
    if modality == "ultrasound":
        inten = [A.GaussNoise(p=0.45), A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                 A.RandomBrightnessContrast(0.3, 0.3, p=0.45), A.RandomShadow(p=0.2)]
    else:
        inten = []
        if clahe_role == "augmentation":
            inten.append(A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5))
        inten += [A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                  A.RandomBrightnessContrast(0.15, 0.2, p=0.35),
                  A.GaussNoise(p=0.25),
                  A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.2)]
    return A.Compose(geo + inten, is_check_shapes=False)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class BMCSplitDataset:
    """One (dataset, split). Reads the frozen manifest and nothing else."""

    def __init__(self, dataset: str, split: str, *, prep_root: Path,
                 intensity_rule: str, aspect_rule: str, clahe_role: str,
                 vflip_scope: str, image_size: int = IMAGE_SIZE,
                 manifest_rows: Optional[List[dict]] = None):
        for name, val, allowed in (("intensity_rule", intensity_rule,
                                     ("declared_range", "per_image_minmax")),
                                    ("aspect_rule", aspect_rule,
                                     ("direct_resize", "pad_to_square")),
                                    ("clahe_role", clahe_role,
                                     ("augmentation", "preprocessing", "off")),
                                    ("vflip_scope", vflip_scope,
                                     ("all", "ultrasound_only", "off"))):
            if val not in allowed:
                raise ValueError(f"{name} must be one of {allowed}; got {val!r}. "
                                 "This is a D-R3 decision and has no default.")
        self.dataset, self.split = dataset, split
        self.intensity_rule, self.aspect_rule = intensity_rule, aspect_rule
        self.clahe_role, self.vflip_scope = clahe_role, vflip_scope
        self.image_size = image_size
        rows = manifest_rows if manifest_rows is not None else read_rows(
            Path(prep_root) / BANK_NAME / SPLITS_DIR / MANIFEST_FILE)
        mine = [r for r in rows if r["dataset"] == dataset]
        if not mine:
            raise ValueError(f"dataset {dataset!r} is not in the manifest")
        role = mine[0]["role"]
        if role == "EXTERNAL" and split != EXTERNAL_SPLIT:
            raise ValueError(f"{dataset} is EXTERNAL and may only be built as "
                             f"{EXTERNAL_SPLIT}; {split!r} was requested")
        if role == "CLIENT" and split not in INTERNAL_SPLITS:
            raise ValueError(f"{dataset} is a client; split must be one of "
                             f"{INTERNAL_SPLITS}; got {split!r}")
        self.rows = sorted((r for r in mine if r["split"] == split),
                           key=lambda r: r["canonical_record_id"])
        if not self.rows:
            raise ValueError(f"no rows for {dataset}/{split}")
        self.modality = self.rows[0]["modality"].lower()
        self.is_train = split == "train"
        self.aug = (build_augmentation(self.modality, clahe_role, vflip_scope)
                    if self.is_train else None)

    def __len__(self) -> int:
        return len(self.rows)

    def record_ids(self) -> List[str]:
        return [r["canonical_record_id"] for r in self.rows]

    def labels(self) -> np.ndarray:
        return np.array([int(r["label_binary"]) for r in self.rows], dtype=np.int64)

    def class_weights_for_sampler(self) -> np.ndarray:
        """Per-sample weights from THIS partition's labels only."""
        y = self.labels()
        counts = np.bincount(y, minlength=2).astype(np.float64)
        w = 1.0 / np.maximum(counts, 1.0)
        return w[y]

    def load_pair(self, i: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        r = self.rows[i]
        img = read_image(Path(r["canonical_image_abspath"]), self.dataset,
                         self.intensity_rule)
        mask = read_mask(Path(r["canonical_mask_abspath"]))
        if img.shape != mask.shape:
            raise RuntimeError(f"{r['canonical_record_id']}: image {img.shape} vs "
                               f"mask {mask.shape}")
        if self.aspect_rule == "pad_to_square":
            img, mask = pad_to_square(img), pad_to_square(mask, 0)
        if self.clahe_role == "preprocessing" and self.modality != "ultrasound":
            import cv2
            u8 = (img * 255).astype(np.uint8)
            img = cv2.createCLAHE(3.0, (8, 8)).apply(u8).astype(np.float32) / 255.0
        img, mask = resize_pair(img, mask, self.image_size)
        return img, mask, r

    def __getitem__(self, i: int) -> Dict[str, Any]:
        img, mask, r = self.load_pair(i)
        if self.aug is not None:
            u8 = (img * 255.0).round().astype(np.uint8)   # intensity aug needs uint8
            out = self.aug(image=u8, mask=mask)
            img = out["image"].astype(np.float32) / 255.0
            mask = out["mask"].astype(np.uint8)
        x = ((img - 0.5) / 0.5).astype(np.float32)[None]      # (1, H, W) in [-1, 1]
        m = mask.astype(np.float32)[None]                      # (1, H, W) in {0, 1}
        return {"image": x, "mask": m, "label": np.float32(int(r["label_binary"])),
                "modality": self.modality, "dataset": self.dataset,
                "record_id": r["canonical_record_id"], "group_key": r["group_key"]}


def make_torch_loader(ds: BMCSplitDataset, batch_size: int, num_workers: int = 0,
                      seed: int = 0):
    """Train: weighted sampler from the train partition only. Others: sequential."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    class _Wrap(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            s = ds[i]
            return {"image": torch.from_numpy(s["image"]),
                    "mask": torch.from_numpy(s["mask"]),
                    "label": torch.tensor(s["label"]),
                    "modality": s["modality"], "dataset": s["dataset"],
                    "record_id": s["record_id"], "group_key": s["group_key"]}

    if ds.is_train:
        g = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            torch.as_tensor(ds.class_weights_for_sampler(), dtype=torch.double),
            num_samples=len(ds), replacement=True, generator=g)
        return DataLoader(_Wrap(), batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, drop_last=False)
    return DataLoader(_Wrap(), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=False)


# ══════════════════════════════════════════════════════════════════════════════
# Self-tests
# ══════════════════════════════════════════════════════════════════════════════
def selftest(prep_root: Path, sample_per_split: int = 8) -> int:
    root = Path(prep_root) / BANK_NAME / SPLITS_DIR
    rows = read_rows(root / MANIFEST_FILE)
    summ = {(r["dataset"], r["role"]): r for r in read_rows(root / SUMMARY_FILE)}
    T: List[Tuple[str, str, bool, str]] = []

    def add(i, d, ok, b):
        T.append((i, d, ok, b))

    # S01 counts per (dataset, split) == TABLE_4
    bad = []
    for (ds, role), s in summ.items():
        want = ({"train": int(s["train_images"]), "validation": int(s["val_images"]),
                 "test": int(s["test_images"])} if role == "CLIENT"
                else {EXTERNAL_SPLIT: int(s["test_images"])})
        for sp, n in want.items():
            got = sum(1 for r in rows if r["dataset"] == ds and r["split"] == sp)
            if got != n:
                bad.append(f"{ds}/{sp}: manifest {got} vs table {n}")
    add("S01", "every partition count equals TABLE_4_split_summary", not bad,
        f"{len(summ)} dataset(s); {len(bad)} mismatch" + (f": {bad[:2]}" if bad else ""))

    # S02 every path exists (full scan, cheap)
    miss = [r["canonical_record_id"] for r in rows
            if not Path(r["canonical_image_abspath"]).is_file()
            or not Path(r["canonical_mask_abspath"]).is_file()]
    add("S02", "every image and mask path in the manifest exists", not miss,
        f"{len(rows)} record(s); {len(miss)} missing")

    # S05 group disjointness re-verified from the manifest
    leaks = []
    for ds in sorted({r["dataset"] for r in rows if r["role"] == "CLIENT"}):
        k = {sp: {r["group_key"] for r in rows if r["dataset"] == ds and r["split"] == sp}
             for sp in INTERNAL_SPLITS}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if k[a] & k[b]:
                leaks.append(f"{ds}: {a}∩{b}={len(k[a] & k[b])}")
    add("S05", "group keys are pairwise disjoint across train/validation/test, "
               "re-verified here", not leaks, f"{len(leaks)} leak(s)")

    # S10 CBIS-DDSM abnormality-level structure, measured on the BUILDER'S OWN
    # mammogram definition. BMC_SPLITS_MANIFEST.csv does not carry mammogram_key, so
    # it is read from the frozen CBIS canonical manifest, where the builder wrote
    # mammogram_key = full_image_path_source, and joined on canonical_record_id.
    # Nothing is reconstructed: a record that does not join is a failure.
    cb = [r for r in rows if r["dataset"] == "CBIS-DDSM"]
    if cb:
        cman = Path(prep_root) / BANK_NAME / "CBIS_DDSM" / "manifests" / \
            "canonical_manifest.csv"
        mk_by_id: Dict[str, List[str]] = defaultdict(list)
        if cman.is_file():
            for x in read_rows(cman):
                mk_by_id[x.get("canonical_record_id", "")].append(
                    x.get("mammogram_key", ""))
        unjoined = [r["canonical_record_id"] for r in cb
                    if len(mk_by_id.get(r["canonical_record_id"], [])) != 1
                    or not mk_by_id[r["canonical_record_id"]][0]]
        mk = {r["canonical_record_id"]: mk_by_id[r["canonical_record_id"]][0]
              for r in cb if r["canonical_record_id"] not in unjoined}
        ids = {r["canonical_record_id"] for r in cb}
        files = {r["canonical_image_abspath"] for r in cb}
        mams = set(mk.values())
        pats = {r["group_key"] for r in cb}
        split_of_mam: Dict[str, set] = defaultdict(set)
        split_of_pat: Dict[str, set] = defaultdict(set)
        for r in cb:
            if r["canonical_record_id"] in mk:
                split_of_mam[mk[r["canonical_record_id"]]].add(r["split"])
            split_of_pat[r["group_key"]].add(r["split"])
        mam_cross = [k for k, v in split_of_mam.items() if len(v) > 1]
        pat_cross = [k for k, v in split_of_pat.items() if len(v) > 1]
        facts = {
            "joint-task records": (len(cb), 2805),
            "unique canonical_record_id": (len(ids), 2805),
            "canonical image files": (len(files), 2805),
            "unique mammogram_key": (len(mams), 2625),
            "patient groups": (len(pats), 1389),
            "records sharing a source mammogram": (len(cb) - len(mams), 180),
            "mammogram_key crossing partitions": (len(mam_cross), 0),
            "patient group_key crossing partitions": (len(pat_cross), 0),
            "records not joined to mammogram_key": (len(unjoined), 0),
        }
        wrong = [f"{k}: got {g}, expected {e}" for k, (g, e) in facts.items() if g != e]
        add("S10", "CBIS-DDSM abnormality-level structure on the builder's "
                   "mammogram_key (read from the frozen CBIS canonical manifest)",
            cman.is_file() and not wrong,
            ("; ".join(f"{k}={g}" for k, (g, _e) in facts.items()))
            + ("" if cman.is_file() else f" | CBIS canonical manifest not found: {cman}")
            + (f" | MISMATCH: {wrong}" if wrong else "")
            + (f" | unjoined: {unjoined[:3]}" if unjoined else "")
            + (f" | mammogram crossing: {mam_cross[:2]}" if mam_cross else ""))

    # ---- EXHAUSTIVE pixel verification: every record in the manifest ------------
    # One read per file. The checks call the production decode_image, scale_image
    # and resize_pair on that read, so what is verified is exactly what training and
    # evaluation will execute on the deterministic path (the approved rules; no
    # augmentation, which is stochastic and checked separately in S07).
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    kw = dict(prep_root=prep_root, manifest_rows=rows, **APPROVED_R3)
    checked = 0
    fail03: List[str] = []           # native dimensions disagree
    fail04a: List[str] = []          # canonical mask not strictly {0, 255}
    fail04b: List[str] = []          # mask not binary after deterministic resize
    fail09a: List[str] = []          # scaled image not finite or outside [0, 1]
    fail09b: List[str] = []          # 16-bit record truncated or out of declared range
    fail09c: List[str] = []          # 8-bit record read in the wrong mode
    n16 = 0
    readable = [r for r in rows if Path(r["canonical_image_abspath"]).is_file()
                and Path(r["canonical_mask_abspath"]).is_file()]
    total = len(readable)
    print(f"\nexhaustive pixel pass over {total} record(s) ...")
    for n, r in enumerate(readable, 1):
        rid = f"{r['dataset']}::{r['canonical_record_id']}"
        ds = r["dataset"]
        with Image.open(r["canonical_image_abspath"]) as im:
            mode = im.mode
            raw = decode_image(im)
        with Image.open(r["canonical_mask_abspath"]) as mi:
            mraw = np.asarray(mi.convert("L"))
        checked += 1
        # S03 native dimensions
        if raw.shape[:2] != mraw.shape[:2]:
            fail03.append(f"{rid} image {raw.shape[:2]} vs mask {mraw.shape[:2]}")
            continue
        # S04a the canonical mask itself is strictly {0, 255}
        if set(np.unique(mraw).tolist()) - {0, 255}:
            fail04a.append(f"{rid} values {sorted(np.unique(mraw).tolist())[:6]}")
        mask = (mraw > 127).astype(np.uint8)
        # S09 scaling
        mx = DECLARED_MAX.get(ds)
        if mx is None:
            fail09a.append(f"{rid} no declared range for {ds}")
            continue
        if mx > 255:
            n16 += 1
            # a 16-bit record must decode in a 16-bit mode, and no raw value may
            # exceed the declared content range (which would be clipped away)
            if mode not in SIXTEEN_BIT_MODES:
                fail09b.append(f"{rid} decoded in mode {mode!r}, not 16-bit")
            elif float(raw.max()) > mx:
                fail09b.append(f"{rid} raw max {int(raw.max())} exceeds declared {mx}")
        else:
            if mode in SIXTEEN_BIT_MODES:
                fail09c.append(f"{rid} 8-bit dataset decoded in 16-bit mode {mode!r}")
            elif float(raw.max()) > 255:
                fail09c.append(f"{rid} raw max {int(raw.max())} exceeds 255")
        img = scale_image(raw, ds, APPROVED_R3["intensity_rule"])
        if not np.isfinite(img).all() or img.min() < 0.0 or img.max() > 1.0:
            fail09a.append(f"{rid} range [{img.min()}, {img.max()}]")
        # the scaled image must equal raw / declared EXACTLY (no hidden clipping)
        if not np.allclose(img, raw / float(mx), rtol=0, atol=1e-7):
            fail09b.append(f"{rid} scaled image differs from raw/{mx}")
        # S04b deterministic path: direct resize, mask NEAREST
        _im, mk = resize_pair(img, mask, IMAGE_SIZE)
        if set(np.unique(mk).tolist()) - {0, 1}:
            fail04b.append(f"{rid} after resize {sorted(np.unique(mk).tolist())[:6]}")
        if n % 500 == 0 or n == total:
            print(f"   {n}/{total}")
    unreadable = len(rows) - total

    add("S03", f"native image and mask dimensions match, for EVERY record "
               f"({checked} checked)",
        not fail03 and checked == len(rows),
        f"{len(fail03)} mismatch; {unreadable} not readable (see S02)"
        + (f": {fail03[:3]}" if fail03 else ""))
    add("S04", f"every canonical mask is strictly {{0,255}} and stays binary after the "
               f"deterministic path ({checked} checked)",
        not fail04a and not fail04b and checked == len(rows),
        f"{len(fail04a)} non-binary canonical mask(s), {len(fail04b)} non-binary after "
        f"resize" + (f": {(fail04a + fail04b)[:3]}" if fail04a or fail04b else ""))
    add("S09", f"every image scales to finite values in [0,1]; every 16-bit record "
               f"decodes at 16 bits within its declared range ({checked} checked, "
               f"{n16} sixteen-bit)",
        not fail09a and not fail09b and not fail09c and checked == len(rows),
        f"{len(fail09a)} out of range, {len(fail09b)} 16-bit fault(s), "
        f"{len(fail09c)} 8-bit fault(s)"
        + (f": {(fail09a + fail09b + fail09c)[:3]}"
           if fail09a or fail09b or fail09c else ""))
    failing_ids = sorted({x.split(" ")[0] for x in
                          fail03 + fail04a + fail04b + fail09a + fail09b + fail09c})

    # S06 the train loader cannot reach a non-train record
    bad6 = []
    for ds in sorted({r["dataset"] for r in rows if r["role"] == "CLIENT"}):
        # identity is (dataset, record id): a record is the same record only within
        # its own dataset, and a loader is filtered on the dataset first
        tr = {(ds, i) for i in BMCSplitDataset(ds, "train", **kw).record_ids()}
        others = {(r["dataset"], r["canonical_record_id"]) for r in rows
                  if r["dataset"] == ds and r["split"] != "train"}
        ext = {(r["dataset"], r["canonical_record_id"]) for r in rows
               if r["role"] == "EXTERNAL"}
        if tr & others or tr & ext:
            bad6.append(ds)
    add("S06", "the training dataset contains no validation, test or external record",
        not bad6, f"{len(bad6)} dataset(s) leak")

    # S06b externals refuse an internal split
    refused = False
    ext_ds = next((r["dataset"] for r in rows if r["role"] == "EXTERNAL"), None)
    if ext_ds:
        try:
            BMCSplitDataset(ext_ds, "train", **kw)
        except ValueError:
            refused = True
    add("S06b", "an external dataset refuses to be built as train or validation",
        refused, f"{ext_ds}: train request refused" if refused else "NOT refused")

    # S07 geometric augmentation moves image and mask identically
    try:
        import albumentations as A
        rng = np.random.default_rng(0)
        img = rng.random((64, 64), dtype=np.float32)
        mk = np.zeros((64, 64), np.uint8)
        mk[10:30, 5:40] = 1
        u8 = (img * 255).astype(np.uint8)
        ok7 = True
        for t in (A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0),
                  A.Rotate(limit=(90, 90), p=1.0)):
            out = A.Compose([t])(image=u8, mask=mk)
            ref = A.Compose([t])(image=mk * 255, mask=mk)   # mask through image path
            if not np.array_equal((ref["image"] > 127).astype(np.uint8), out["mask"]):
                ok7 = False
        add("S07", "flip and rotate move image and mask identically (mask stays "
                   "aligned and binary)", ok7, "HFlip, VFlip, Rotate90 checked")
    except ImportError:
        add("S07", "albumentations available for augmentation", False, "not installed")

    # S08 val/test are ordered and deterministic
    ds0 = sorted({r["dataset"] for r in rows if r["role"] == "CLIENT"})[0]
    a1 = BMCSplitDataset(ds0, "test", **kw).record_ids()
    a2 = BMCSplitDataset(ds0, "test", **kw).record_ids()
    add("S08", "validation/test order is sorted and identical across constructions",
        a1 == a2 == sorted(a1), f"{ds0}/test: {len(a1)} record(s)")

    # S11 sampler weights come from the train partition only
    d = BMCSplitDataset(ds0, "train", **kw)
    w = d.class_weights_for_sampler()
    y = d.labels()
    add("S11", "sampler weights are 1/count of the TRAIN partition's own labels",
        np.isclose(w[y == 0].sum(), w[y == 1].sum()) if (y == 0).any() and (y == 1).any()
        else False, f"{ds0}: total weight per class equalised")

    # S12 the four decisions are mandatory
    try:
        BMCSplitDataset(ds0, "train", prep_root=prep_root, intensity_rule="x",
                        aspect_rule="direct_resize", clahe_role="off",
                        vflip_scope="off", manifest_rows=rows)
        forced = False
    except ValueError:
        forced = True
    add("S12", "an unstated D-R3 decision refuses to build", forced, "")

    print("=" * 90)
    print("R3 - DATA LOADER SELF-TESTS")
    print("=" * 90)
    for i, dsc, ok, b in sorted(T, key=lambda t: t[0]):
        print(f"  {i:5s} [{'PASS' if ok else 'FAIL'}]  {dsc}")
        if b:
            print(f"        {b}")
    fails = [t for t in T if not t[2]]
    print("=" * 90)
    print(f"records in manifest       : {len(rows)}")
    print(f"records pixel-checked     : {checked}")
    print(f"failing canonical_record_id: {len(failing_ids)}")
    for x in failing_ids[:40]:
        print(f"   {x}")
    print(f"PASS = {len(T) - len(fails)}   FAIL = {len(fails)}")
    print("R3_LOADER_VERIFIED" if not fails else "R3_LOADER_BLOCKED")
    return 0 if not fails else 4


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(Path(a.prep_root))
    ap.print_help()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
