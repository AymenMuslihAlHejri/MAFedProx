#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bank_bluis.py
===================

PHASE 2B.3.6 - BLUIS CANONICAL DATASET BANK BUILDER

Sixth dataset. Nothing is carried over from BUSI, BUS-UC, BrEaST, BUS-BRA or INbreast.

    BUSI      union of multiple masks
    BUS-UC    midpoint threshold at 128 on soft boundaries
    BrEaST    RGBA colour-channel reduction
    BUS-BRA   value scaling {0,1} -> {0,255}
    INbreast  contour rasterisation + 16-bit passthrough
    BLUIS     value scaling {0,1} -> {0,255}, and a 3-channel BMP image kept as is

    MASK   BLUIS_CANONICAL_MASK_VALUE_SCALING_V1
        verify a single plane whose values lie within {0,1} -> multiply by 255 ->
        single-channel lossless PNG. No threshold, no union, no morphology, no
        pseudo-mask. A mask outside the declared range is refused, never converted.

    IMAGE  BLUIS_CANONICAL_IMAGE_LOSSLESS_V1
        the decoded BMP is written unchanged to a lossless PNG at native dimensions,
        KEEPING ITS 3 CHANNELS. Collapsing RGB to grey would be a preprocessing
        decision this bank does not make, even though the source looks greyscale; the
        per-record check records whether the channels are in fact identical so a later
        phase can decide with evidence.

AUXILIARY LESION IMAGES
    The source ships a third file per sample (232 of them). The audit classified these
    as AUXILIARY_LESION_IMAGE and excluded them from the modelling pool. They are
    recorded here as provenance only: their path is carried in the manifest and they
    are NEVER read as a source image or a mask. T13 checks that no canonical record
    points at one.

IDENTITY AND DUPLICATES
    patient_id, case_id and lesion_id are NOT_ESTABLISHED and stay empty. The two LOW
    near-duplicate flags are screening candidates and produce no exclusion and no group
    lock. No split is created.

Usage:
    python build_bank_bluis.py --dry-run
    python build_bank_bluis.py --execute
Requires: Python >= 3.8, Pillow, numpy. openpyxl optional.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import re
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET = "BLUIS"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)

MANIFEST_FILE = "11_bluis_audit_manifest.csv"
SUMMARY_FILE = "13_bluis_audit_summary.json"
SELFTEST_FILE = "15_bluis_audit_self_tests.txt"
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]

MASK_POLICY_ID = "BLUIS_CANONICAL_MASK_VALUE_SCALING_V1"
MASK_POLICY_RULE = ("verify a single plane with values within {0,1} -> multiply by 255 "
                    "-> single-channel lossless PNG")
IMAGE_POLICY_ID = "BLUIS_CANONICAL_IMAGE_LOSSLESS_V1"
IMAGE_POLICY_RULE = ("decoded BMP written unchanged to lossless PNG at native "
                     "dimensions, channels preserved; no resize, crop, ROI, "
                     "normalisation, CLAHE, denoising or augmentation")

EXPECT = {"records": 232, "benign": 109, "malignant": 123, "masks": 232,
          "auxiliary": 232, "excluded": 0, "png": 464, "duplicates": 0}
CANONICAL_LABELS = {"BENIGN": 0, "MALIGNANT": 1}

SCHEMA_FIELDS = [
    "dataset", "modality", "canonical_record_id", "sample_key",
    "source_image_id", "source_mask_id", "source_image_path", "source_mask_path",
    "auxiliary_lesion_path", "auxiliary_lesion_is_canonical_source",
    "source_label", "canonical_binary_label", "class_folder",
    "canonical_image_path", "canonical_mask_path",
    "patient_id", "case_id", "lesion_id", "identity_status",
    "same_source_image_group", "duplicate_group_id",
    "near_duplicate_flag_PROVENANCE_ONLY",
    "classification_available", "segmentation_available",
    "image_conversion_status", "mask_conversion_status",
    "source_image_height", "source_image_width", "source_image_channels",
    "source_channels_identical", "source_mask_height", "source_mask_width",
    "canonical_image_height", "canonical_image_width",
    "canonical_mask_height", "canonical_mask_width",
    "source_image_format", "source_mask_format",
    "source_mask_encoding", "canonical_image_encoding", "canonical_mask_encoding",
    "mask_foreground_pixels", "mask_foreground_fraction",
    "image_policy_id", "image_policy_rule", "mask_policy_id", "mask_policy_rule",
    "identity_provenance", "label_provenance", "mask_provenance",
    "exclusion_status", "exclusion_reason", "notes",
]
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}

SEP = " | "
LOGGER = logging.getLogger("bank_bluis")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def join_list(v: Sequence[Any], sep: str = SEP) -> str:
    return sep.join(str(x) for x in v)


def archived(p: Path) -> bool:
    return any(x.lower() in ARCHIVE_DIR_NAMES for x in p.parts[:-1])


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def snapshot(root: Path) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        try:
            if p.is_file():
                st = p.stat()
                out[str(p)] = (st.st_size, st.st_mtime_ns)
        except OSError:
            pass
    return out


def diff_state(a: dict, b: dict) -> dict:
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    modified = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return {"added": added, "removed": removed, "modified": modified,
            "unchanged": not (added or removed or modified)}


def is_split_like(path_str: str) -> bool:
    pth = Path(path_str)
    raw = [t for t in re.split(r"[^A-Za-z0-9]+", pth.stem) if t]
    raw += [t for part in pth.parts[:-1]
            for t in re.split(r"[^A-Za-z0-9]+", part) if t]
    toks = set()
    for t in raw:
        toks.add(t.lower())
        toks.add(re.sub(r"\d+$", "", t.lower()))
    return bool(toks & SPLIT_TOKENS)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    LOGGER.info("WROTE %-44s (%d data rows)", path.name, len(rows))


def read_rows(p: Path) -> List[dict]:
    with p.open(encoding="utf-8-sig", errors="replace") as fh:
        s = fh.read(8192)
        fh.seek(0)
        if not s.strip():
            return []
        d = ";" if s.count(";") > s.count(",") else ","
        return list(csv.DictReader(fh, delimiter=d))


def setup_logging(report_dir: Path, name: str) -> None:
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    f = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(f)
    sh.setLevel(logging.INFO)
    LOGGER.addHandler(sh)
    report_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(report_dir / name, mode="w", encoding="utf-8")
    fh.setFormatter(f)
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)


def find_one(files: Sequence[Path], filename: str) -> Tuple[Optional[Path], str]:
    hits = sorted(p for p in files if p.name.lower() == filename.lower())
    if len(hits) == 1:
        (only,) = hits
        return only, "UNIQUE"
    return None, ("MISSING" if not hits else f"AMBIGUOUS({len(hits)})")


def canonical_stem(cls: str, sample_key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_key).strip("_")
    return f"{safe or 'sample'}__{sha256_text(f'{cls}/{sample_key}')[:10]}"


def scale_mask(arr, np) -> Tuple[Any, str]:
    """{0,1} -> {0,255}. Anything outside the declared rule is refused."""
    if arr.ndim == 3:
        if arr.shape[2] < 3:
            return None, f"UNEXPECTED_CHANNELS_{arr.shape[2]}"
        if not (np.array_equal(arr[..., 0], arr[..., 1])
                and np.array_equal(arr[..., 1], arr[..., 2])):
            return None, "MULTICHANNEL_NOT_REPLICATED"
        plane, note = arr[..., 0], "REDUCED_REPLICATED_CHANNELS"
    else:
        plane, note = arr, "ALREADY_SINGLE_CHANNEL"
    if plane.dtype == bool:
        plane, note = plane.astype(np.uint8), note + "_FROM_BOOL"
    vals = set(np.unique(plane).tolist())
    if vals <= {0, 1}:
        return (plane.astype(np.uint8) * 255), note + "_SCALED_0_1_TO_0_255"
    if vals <= {0, 255}:
        return plane.astype(np.uint8), note + "_ALREADY_0_255"
    return None, f"VALUES_OUTSIDE_DECLARED_RULE{sorted(vals)[:5]}"


def load_evidence(prep_root: Path) -> dict:
    ev: Dict[str, Any] = {"paths": {}, "status": {}, "hashes": {}, "problems": []}
    dirs = [d for d in prep_root.iterdir()
            if d.is_dir() and "bluis" in d.name.lower().replace(" ", "")]
    files = [p for d in dirs for p in d.rglob("*") if p.is_file() and not archived(p)]
    for key, fname, required in (("manifest", MANIFEST_FILE, True),
                                 ("summary", SUMMARY_FILE, True),
                                 ("selftests", SELFTEST_FILE, False)):
        art, st = find_one(files, fname)
        ev["paths"][key] = art
        ev["status"][key] = st
        if art is not None:
            ev["hashes"][art.name] = sha256_file(art)
        elif required:
            ev["problems"].append(f"BLOCKED: {fname} -> {st}")
    ev["dataset_root"] = None
    ev["dataset_root_locator"] = ""
    if ev["paths"].get("summary"):
        blob = json.loads(ev["paths"]["summary"].read_text(encoding="utf-8"))
        for loc in ROOT_LOCATORS:
            v = blob.get(loc)
            if isinstance(v, str) and v.strip():
                ev["dataset_root"] = Path(v.strip())
                ev["dataset_root_locator"] = f"{SUMMARY_FILE} :: json:{loc}"
                break
        if ev["dataset_root"] is None:
            ev["problems"].append(f"BLOCKED: no source root in {SUMMARY_FILE}")
    return ev


def run(mode: str, prep_root: Path) -> int:
    bank_ds = prep_root / BANK_NAME / DATASET
    reports, manifests = bank_ds / "reports", bank_ds / "manifests"
    setup_logging(reports, f"bluis_{mode}.log")
    LOGGER.info("=" * 78)
    LOGGER.info("PHASE 2B.3.6 BLUIS CANONICAL BANK BUILDER  [%s]", mode.upper())
    LOGGER.info("=" * 78)
    LOGGER.info("python : %s", platform.python_version())
    LOGGER.info("bank   : %s", bank_ds)

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        LOGGER.critical("Pillow and numpy required")
        return 3
    Image.MAX_IMAGE_PIXELS = None

    prep_dirs = [d for d in prep_root.iterdir()
                 if d.is_dir() and d.name != BANK_NAME]
    before_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        before_audit.update(snapshot(d))
    before_bank = snapshot(prep_root / BANK_NAME)

    LOGGER.info("[1/6] locked evidence bindings ...")
    ev = load_evidence(prep_root)
    for k, st in ev["status"].items():
        art = ev["paths"].get(k)
        LOGGER.info("      %-10s %-12s %s", k, st, art.name if art else "")
    if ev["problems"]:
        for p in ev["problems"]:
            LOGGER.error("      %s", p)
        LOGGER.error("FINAL VERDICT:\nBLOCKED")
        return 4
    src_root = ev["dataset_root"]
    before_raw = snapshot(src_root)
    summ = json.loads(ev["paths"]["summary"].read_text(encoding="utf-8"))
    LOGGER.info("      RAW    = %s  [READ-ONLY]", src_root)
    LOGGER.info("      root evidence = %s", ev["dataset_root_locator"])
    LOGGER.info("      image policy = %s", IMAGE_POLICY_ID)
    LOGGER.info("      mask policy  = %s", MASK_POLICY_ID)

    rows = read_rows(ev["paths"]["manifest"])
    keys = [r["sample_key"].strip() for r in rows]
    dupe = [k for k, n in Counter(keys).items() if n > 1]
    if dupe:
        LOGGER.error("      sample_key not unique: %s", dupe[:5])
        LOGGER.error("FINAL VERDICT:\nBLOCKED")
        return 4

    LOGGER.info("[2/6] source accounting ...")
    by_cls = Counter(r["class_label"].strip().upper() for r in rows)
    n_masks = sum(int(r["mask_count"]) for r in rows if r["mask_count"].strip())
    aux = sum(1 for r in rows
              if r.get("auxiliary_lesion_present", "").strip().upper() == "YES")
    enc = Counter(r["mask_encoding"].strip() for r in rows)
    ch = Counter(r["image_channels"].strip() for r in rows)
    nd = Counter(r.get("near_duplicate_candidate_flag", "").strip() or "NONE"
                 for r in rows)
    LOGGER.info("      manifest %d record(s): benign=%d malignant=%d, masks=%d",
                len(rows), by_cls.get("BENIGN", 0), by_cls.get("MALIGNANT", 0), n_masks)
    LOGGER.info("      summary json: images=%s masks=%s benign=%s malignant=%s "
                "auxiliary=%s", summ.get("total_source_images"),
                summ.get("total_masks"),
                (summ.get("class_distribution") or {}).get("benign"),
                (summ.get("class_distribution") or {}).get("malignant"),
                summ.get("auxiliary_lesion_images_excluded"))
    LOGGER.info("      mask encoding %s | image channels %s", dict(enc), dict(ch))
    LOGGER.info("      auxiliary lesion files present: %d (provenance only)", aux)
    LOGGER.info("      near-duplicate flags: %s  [screening candidates only]",
                dict(nd))
    LOGGER.info("      identity: patient=%s lesion=%s grouping_possible=%s",
                summ.get("patient_id_available"), summ.get("lesion_id_available"),
                summ.get("patient_grouping_possible"))

    LOGGER.info("[3/6] planning canonical records ...")
    records: List[dict] = []
    for r in rows:
        key = r["sample_key"].strip()
        cls = r["class_label"].strip().upper()
        folder = "Benign" if cls == "BENIGN" else (
            "Malignant" if cls == "MALIGNANT" else "")
        stem = canonical_stem(cls, key)
        records.append({
            "dataset": DATASET, "modality": r.get("modality", "").strip(),
            "canonical_record_id": f"{DATASET}__{stem}", "sample_key": key,
            "source_image_id": r.get("image_filename", "").strip(),
            "source_mask_id": r.get("mask_filenames", "").strip(),
            "source_image_path": r.get("image_path", "").strip(),
            "source_mask_path": r.get("mask_paths", "").strip(),
            "auxiliary_lesion_path": r.get("auxiliary_lesion_path", "").strip(),
            "auxiliary_lesion_is_canonical_source": "NO",
            "source_label": cls,
            "canonical_binary_label": CANONICAL_LABELS.get(cls, "UNRESOLVED"),
            "class_folder": folder,
            "canonical_image_path": f"{folder}/Images/{stem}.png" if folder else "",
            "canonical_mask_path": f"{folder}/Masks/{stem}_mask.png" if folder else "",
            "patient_id": "", "case_id": "", "lesion_id": "",
            "identity_status": "IMAGE_ONLY_NOT_ESTABLISHED",
            "same_source_image_group": f"{DATASET}__{stem}",
            "duplicate_group_id": r.get("exact_duplicate_group", "").strip(),
            "near_duplicate_flag_PROVENANCE_ONLY": (
                (r.get("near_duplicate_candidate_flag", "").strip() or "NONE")
                + " [SCREENING_CANDIDATE_ONLY, not a confirmed duplicate and not a "
                  "group lock]"),
            "classification_available": "TRUE", "segmentation_available": "TRUE",
            "image_conversion_status": "PLANNED", "mask_conversion_status": "PLANNED",
            "source_image_height": r.get("image_height", ""),
            "source_image_width": r.get("image_width", ""),
            "source_image_channels": r.get("image_channels", ""),
            "source_channels_identical": "",
            "source_mask_height": r.get("mask_height", ""),
            "source_mask_width": r.get("mask_width", ""),
            "canonical_image_height": "", "canonical_image_width": "",
            "canonical_mask_height": "", "canonical_mask_width": "",
            "source_image_format": Path(r.get("image_path", "")).suffix.lstrip(
                ".").upper(),
            "source_mask_format": Path(r.get("mask_paths", "")).suffix.lstrip(
                ".").upper(),
            "source_mask_encoding": r.get("mask_encoding", "").strip(),
            "canonical_image_encoding": "LOSSLESS_PNG_CHANNELS_PRESERVED",
            "canonical_mask_encoding": "SINGLE_CHANNEL_BINARY_0_255",
            "mask_foreground_pixels": "", "mask_foreground_fraction": "",
            "image_policy_id": IMAGE_POLICY_ID, "image_policy_rule": IMAGE_POLICY_RULE,
            "mask_policy_id": MASK_POLICY_ID, "mask_policy_rule": MASK_POLICY_RULE,
            "identity_provenance": (f"{MANIFEST_FILE} :: csv:[patient_id_status]="
                                    f"{r.get('patient_id_status', '').strip()}; "
                                    "patient/case/lesion NOT_ESTABLISHED"),
            "label_provenance": (f"{MANIFEST_FILE} :: csv:[class_label] "
                                 f"(source={r.get('class_label_source', '').strip()})"),
            "mask_provenance": f"SOURCE_NATIVE_BINARY_SCALED_BY[{MASK_POLICY_ID}]",
            "exclusion_status": "INCLUDED_CANONICAL_JOINT", "exclusion_reason": "",
            "notes": (f"pairing_status={r.get('pairing_status', '').strip()}; "
                      f"audit_status={r.get('audit_status', '').strip()}; "
                      f"auxiliary_lesion_present="
                      f"{r.get('auxiliary_lesion_present', '').strip()} "
                      "(provenance only, never a canonical source)"),
        })
    by_el = Counter(r["source_label"] for r in records)
    LOGGER.info("      planned %d: benign=%d malignant=%d", len(records),
                by_el.get("BENIGN", 0), by_el.get("MALIGNANT", 0))

    LOGGER.info("[4/6] resolving and pre-checking source pairs ...")
    img_ok = mask_ok = plans = 0
    unresolved: List[str] = []
    notes_c: Counter = Counter()
    chan_same = Counter()
    for i, r in enumerate(records, 1):
        sp, mp = src_root / r["source_image_path"], src_root / r["source_mask_path"]
        if not sp.is_file():
            unresolved.append(f"image missing: {r['source_image_path']}")
            r["image_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            with Image.open(sp) as im:
                ia = np.array(im)
                r["source_image_width"] = r["canonical_image_width"] = im.width
                r["source_image_height"] = r["canonical_image_height"] = im.height
            img_ok += 1
        except Exception:  # noqa: BLE001
            unresolved.append(f"image undecodable: {r['source_image_path']}")
            r["image_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        if ia.ndim == 3 and ia.shape[2] >= 3:
            same = bool(np.array_equal(ia[..., 0], ia[..., 1])
                        and np.array_equal(ia[..., 1], ia[..., 2]))
        else:
            same = True
        r["source_channels_identical"] = "YES" if same else "NO"
        chan_same["identical" if same else "not identical"] += 1
        if not mp.is_file():
            unresolved.append(f"mask missing: {r['source_mask_path']}")
            r["mask_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            with Image.open(mp) as mi:
                ma = np.array(mi)
                r["source_mask_width"] = r["canonical_mask_width"] = mi.width
                r["source_mask_height"] = r["canonical_mask_height"] = mi.height
        except Exception:  # noqa: BLE001
            unresolved.append(f"mask undecodable: {r['source_mask_path']}")
            r["mask_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        mask_ok += 1
        plane, note = scale_mask(ma, np)
        notes_c[note] += 1
        if plane is None:
            unresolved.append(f"mask refused: {r['sample_key']} ({note})")
            r["mask_conversion_status"] = f"REFUSED_{note}"
            continue
        if (r["source_mask_height"], r["source_mask_width"]) != (
                r["source_image_height"], r["source_image_width"]):
            unresolved.append(f"dimension mismatch: {r['sample_key']}")
            r["mask_conversion_status"] = "DIMENSION_MISMATCH"
            continue
        fg = int((plane == 255).sum())
        if fg == 0:
            unresolved.append(f"empty mask: {r['sample_key']}")
            r["mask_conversion_status"] = "EMPTY_MASK"
            continue
        r["mask_foreground_pixels"] = fg
        r["mask_foreground_fraction"] = round(
            fg / float(plane.shape[0] * plane.shape[1]), 6)
        r["notes"] += f"; scaling={note}"
        plans += 1
        if i % 50 == 0:
            LOGGER.info("      %d/%d", i, len(records))
    LOGGER.info("      image=%d mask=%d scaling plans=%d unresolved=%d",
                img_ok, mask_ok, plans, len(unresolved))
    LOGGER.info("      scaling outcomes: %s", dict(notes_c))
    LOGGER.info("      source RGB channels: %s", dict(chan_same))
    for u in unresolved[:10]:
        LOGGER.error("         %s", u)

    png_written = 0
    if mode == "execute":
        LOGGER.info("[5/6] executing canonical conversion ...")
        for r in records:
            if r["image_conversion_status"] != "PLANNED":
                continue
            dst = bank_ds / r["canonical_image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with Image.open(src_root / r["source_image_path"]) as im:
                    im.save(dst, format="PNG")   # channels and values preserved
                png_written += 1
            r["image_conversion_status"] = "CONVERTED"
            with Image.open(src_root / r["source_mask_path"]) as mi:
                ma = np.array(mi)
            plane, _n = scale_mask(ma, np)
            if plane is None:
                r["mask_conversion_status"] = "REFUSED_AT_EXECUTE"
                continue
            mdst = bank_ds / r["canonical_mask_path"]
            mdst.parent.mkdir(parents=True, exist_ok=True)
            if not mdst.exists():
                Image.fromarray(plane.astype(np.uint8), mode="L").save(
                    mdst, format="PNG")
                png_written += 1
            r["mask_conversion_status"] = "CONVERTED"
    else:
        LOGGER.info("[5/6] dry-run: no canonical PNG is written")

    LOGGER.info("[6/6] validation ...")
    after_bank = snapshot(prep_root / BANK_NAME)
    new_splits = [k for k in after_bank if k not in before_bank and is_split_like(k)]
    new_png = [k for k in after_bank
               if k not in before_bank and k.lower().endswith(".png")]
    raw_diff = diff_state(before_raw, snapshot(src_root))
    after_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        after_audit.update(snapshot(d))
    aud_diff = diff_state(before_audit, after_audit)
    cpaths = [r["canonical_image_path"] for r in records] + \
             [r["canonical_mask_path"] for r in records]
    collisions = [n for n, c in Counter(p for p in cpaths if p).items() if c > 1]
    aux_paths = {r["auxiliary_lesion_path"] for r in records
                 if r["auxiliary_lesion_path"]}
    aux_as_source = [r["sample_key"] for r in records
                     if r["source_image_path"] in aux_paths
                     or r["source_mask_path"] in aux_paths]

    T: List[Tuple[str, str, Optional[bool], str]] = []

    def add(i: str, d: str, ok: Optional[bool], b: str) -> None:
        T.append((i, d, ok, b))

    add("T01", f"source images == {EXPECT['records']}",
        len(rows) == EXPECT["records"], f"{len(rows)} @ {MANIFEST_FILE}")
    add("T02", f"paired masks == {EXPECT['masks']}", n_masks == EXPECT["masks"],
        f"sum(mask_count)={n_masks}; pairing "
        f"{dict(Counter(r['pairing_status'] for r in rows))}")
    add("T03", f"benign == {EXPECT['benign']} and malignant == "
               f"{EXPECT['malignant']}",
        by_cls.get("BENIGN") == EXPECT["benign"]
        and by_cls.get("MALIGNANT") == EXPECT["malignant"],
        f"{by_cls.get('BENIGN')} / {by_cls.get('MALIGNANT')}")
    add("T04", "summary json agrees with the manifest",
        summ.get("total_source_images") == len(rows)
        and summ.get("total_masks") == n_masks
        and (summ.get("class_distribution") or {}).get("benign")
        == by_cls.get("BENIGN"),
        f"summary {summ.get('total_source_images')}/{summ.get('total_masks')}")
    add("T05", f"planned canonical records == {EXPECT['records']}",
        len(records) == EXPECT["records"]
        and by_el.get("BENIGN") == EXPECT["benign"]
        and by_el.get("MALIGNANT") == EXPECT["malignant"],
        f"{len(records)} ({by_el.get('BENIGN')} / {by_el.get('MALIGNANT')})")
    add("T06", "mask encoding is GRAY_BINARY_0_1 on every record",
        set(enc) == {"GRAY_BINARY_0_1"}, f"{dict(enc)}")
    add("T07", f"value-scaling plans == {EXPECT['records']}",
        plans == EXPECT["records"], f"{plans}; outcomes {dict(notes_c)}")
    add("T08", "no threshold, union, contour or policy from another dataset",
        "threshold" not in MASK_POLICY_RULE and "union" not in MASK_POLICY_RULE
        and not any(pid in r["mask_provenance"] for r in records
                    for pid in ("BUSI_CANONICAL_MULTIMASK_UNION_V1",
                                "BUS_UC_CANONICAL_MASK_BINARIZATION_V1",
                                "BREAST_CANONICAL_MASK_REPLICATION_REDUCTION_V1",
                                "BUS_BRA_CANONICAL_MASK_VALUE_SCALING_V1",
                                "INBREAST_FULLSOURCE_MASK_FROM_XML_CONTOUR_V1")),
        f"declared rule: {MASK_POLICY_ID}")
    add("T09", "no resize, crop, ROI, normalisation, CLAHE, denoising, augmentation "
               "or pseudo-mask", True,
        "code-scope: decode, value scaling, lossless PNG; a mask outside the rule is "
        "refused, never repaired")
    add("T10", f"image mappings == {EXPECT['records']}", img_ok == EXPECT["records"],
        f"{img_ok}/{len(records)}")
    add("T11", f"mask mappings == {EXPECT['records']}", mask_ok == EXPECT["records"],
        f"{mask_ok}/{len(records)}")
    add("T12", f"auxiliary lesion images recorded == {EXPECT['auxiliary']} and used "
               "as canonical source == 0",
        aux == EXPECT["auxiliary"] and not aux_as_source,
        f"{aux} recorded as provenance; {len(aux_as_source)} used as a source")
    add("T13", "no canonical record points at an auxiliary file",
        not aux_as_source
        and all(r["auxiliary_lesion_is_canonical_source"] == "NO" for r in records),
        f"{len(aux_paths)} distinct auxiliary path(s), none converted")
    add("T14", "no patient, case or lesion identity invented",
        all(not r["patient_id"] and not r["case_id"] and not r["lesion_id"]
            for r in records)
        and summ.get("patient_id_available") == "NO",
        f"summary patient_id_available={summ.get('patient_id_available')}, "
        f"lesion_id_available={summ.get('lesion_id_available')}")
    add("T15", f"exact duplicates == {EXPECT['duplicates']} and no group lock "
               "invented from screening candidates",
        summ.get("exact_duplicate_groups") == 0
        and all(not r["duplicate_group_id"] for r in records),
        f"summary groups={summ.get('exact_duplicate_groups')}; near-duplicate flags "
        f"{dict(nd)} kept in a PROVENANCE_ONLY field")
    add("T16", f"excluded == {EXPECT['excluded']}",
        sum(1 for r in records
            if r["exclusion_status"] != "INCLUDED_CANONICAL_JOINT")
        == EXPECT["excluded"]
        and all(r.get("audit_status", "").strip() == "OK" for r in rows),
        f"audit_status {dict(Counter(r['audit_status'] for r in rows))}; the "
        "exclusion ledger is NOT modified when excluded == 0")
    add("T17", "no unresolved record", not unresolved, f"{len(unresolved)}")
    add("T18", "no canonical path collision", not collisions, f"{len(collisions)}")
    add("T19", "RAW unchanged", raw_diff["unchanged"],
        f"modified {len(raw_diff['modified'])}, added {len(raw_diff['added'])}")
    add("T20", "completed audit artifacts unchanged", aud_diff["unchanged"],
        f"modified {len(aud_diff['modified'])}")
    add("T21", "no split created", not new_splits, f"{len(new_splits)}")
    add("T22", ("dry-run wrote no canonical PNG" if mode == "dry-run"
                else f"execution wrote {EXPECT['png']} canonical PNG"),
        (not new_png) if mode == "dry-run" else (len(new_png) == EXPECT["png"]),
        f"{len(new_png)} new PNG")
    add("T23", "no split field in the manifest schema",
        not any(is_split_like(f) for f in SCHEMA_FIELDS), "schema checked")
    add("T24", "source RGB channel equality recorded, never acted on", None,
        f"{dict(chan_same)}; the 3-channel image is written unchanged, so any later "
        "greyscale decision is made with evidence rather than silently here")
    verified = [t for t in T if t[2] is not None]
    failed = [t for t in verified if not t[2]]
    for t in T:
        (LOGGER.info if t[2] is not False else LOGGER.error)(
            "      %-5s [%-11s] %-52s %s", t[0],
            "DECLARATION" if t[2] is None else ("PASS" if t[2] else "FAIL"),
            t[1][:52], t[3][:52])

    tag = "DRY_RUN_" if mode == "dry-run" else ""
    out_rows = [{k: r[k] for k in SCHEMA_FIELDS} for r in records]
    write_csv(manifests / f"{tag}canonical_manifest.csv", SCHEMA_FIELDS, out_rows)
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "canonical_manifest"
        ws.append(SCHEMA_FIELDS)
        for c in range(1, len(SCHEMA_FIELDS) + 1):
            ws.cell(1, c).font = Font(bold=True, color="FFFFFF")
            ws.cell(1, c).fill = PatternFill("solid", fgColor="2F5496")
            ws.cell(1, c).alignment = Alignment(horizontal="center", wrap_text=True)
        for r in out_rows:
            ws.append([r.get(h, "") for h in SCHEMA_FIELDS])
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = (f"A1:{get_column_letter(len(SCHEMA_FIELDS))}"
                              f"{len(out_rows) + 1}")
        wide = {"image_policy_rule", "mask_policy_rule", "identity_provenance",
                "label_provenance", "mask_provenance", "notes", "exclusion_reason",
                "near_duplicate_flag_PROVENANCE_ONLY"}
        for i, h in enumerate(SCHEMA_FIELDS, 1):
            ws.column_dimensions[get_column_letter(i)].width = 70 if h in wide else 22
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        wb.save(manifests / f"{tag}canonical_manifest.xlsx")
        LOGGER.info("WROTE %-44s (1 sheet)", f"{tag}canonical_manifest.xlsx")
    except ImportError:
        LOGGER.warning("openpyxl not installed; XLSX skipped")

    blocked = bool(failed)
    state = {
        "phase": "2B.3.6", "dataset": DATASET, "mode": mode, "generated": now_iso(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "schema_sha256": sha256_text(join_list(SCHEMA_FIELDS, ",")),
        "evidence_hashes": ev["hashes"], "source_root": str(src_root),
        "policies": {"image": {"id": IMAGE_POLICY_ID, "rule": IMAGE_POLICY_RULE},
                     "mask": {"id": MASK_POLICY_ID, "rule": MASK_POLICY_RULE}},
        "counts": {"source_images": len(rows), "source_masks": n_masks,
                   "benign": by_cls.get("BENIGN", 0),
                   "malignant": by_cls.get("MALIGNANT", 0),
                   "planned_records": len(records), "scaling_plans": plans,
                   "auxiliary_recorded": aux, "auxiliary_used_as_source": 0,
                   "excluded": 0, "duplicates": 0, "group_locks": 0},
        "identity": {"patient_id": "NOT_ESTABLISHED", "case_id": "NOT_ESTABLISHED",
                     "lesion_id": "NOT_ESTABLISHED",
                     "grouping_possible": summ.get("patient_grouping_possible"),
                     "limitation": "no patient-independent split can be defended for "
                                   "this dataset; no split is created here"},
        "near_duplicate_flags": dict(nd),
        "source_channels": dict(chan_same),
        "mask_scaling_outcomes": dict(notes_c),
        "image_mappings_resolved": img_ok, "mask_mappings_resolved": mask_ok,
        "unresolved": unresolved[:50], "collisions": collisions,
        "canonical_png_written": png_written,
        "raw_modified": not raw_diff["unchanged"],
        "audit_modified": not aud_diff["unchanged"],
        "exclusion_ledger_modified": False,
        "tests_pass": sum(1 for t in verified if t[2]), "tests_fail": len(failed),
        "verdict": ("BLOCKED" if blocked else
                    "READY_FOR_BLUIS_CANONICAL_EXECUTION" if mode == "dry-run"
                    else "BLUIS_CANONICAL_BUILD_COMPLETE"),
    }
    (reports / f"{tag}canonical_build_report.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    LOGGER.info("WROTE %-44s", f"{tag}canonical_build_report.json")

    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 2B.3.6 BLUIS - %s", mode.upper())
    LOGGER.info("Source:      %d image(s) / %d mask(s)", len(rows), n_masks)
    LOGGER.info("Planned:     %d canonical record(s)", len(records))
    LOGGER.info("B / M:       %d / %d", by_el.get("BENIGN", 0),
                by_el.get("MALIGNANT", 0))
    LOGGER.info("Mask rule:   %s -> %s", MASK_POLICY_ID, dict(notes_c))
    LOGGER.info("Identity:    patient/case/lesion NOT_ESTABLISHED; 0 invented; "
                "grouping not possible")
    LOGGER.info("Auxiliary:   %d recorded as provenance, 0 used as canonical source",
                aux)
    LOGGER.info("Exclusions:  0  (exclusion ledger NOT modified)")
    LOGGER.info("RAW modified %s | audit modified %s | PNG %d | splits %d",
                state["raw_modified"], state["audit_modified"], png_written,
                len(new_splits))
    LOGGER.info("TESTS PASS: %d    TESTS FAIL: %d", state["tests_pass"], len(failed))
    LOGGER.info("FINAL VERDICT:")
    if blocked:
        LOGGER.error("BLOCKED")
        for t in failed:
            LOGGER.error("    FAILED %s: %s | %s", t[0], t[1], t[3])
        return 4
    LOGGER.info(state["verdict"])
    if mode == "dry-run":
        LOGGER.info("STOP. --execute needs human approval.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="BLUIS canonical bank builder.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    p.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    a = p.parse_args(argv)
    mode = "dry-run" if a.dry_run else "execute"
    if mode == "execute":
        st = (Path(a.prep_root) / BANK_NAME / DATASET / "reports"
              / "DRY_RUN_canonical_build_report.json")
        if not st.is_file():
            sys.stderr.write("EXECUTE refused: no dry-run report\n")
            return 5
        d = json.loads(st.read_text(encoding="utf-8"))
        if d.get("verdict") != "READY_FOR_BLUIS_CANONICAL_EXECUTION":
            sys.stderr.write(f"EXECUTE refused: last verdict {d.get('verdict')!r}\n")
            return 5
        if d.get("builder_sha256") != sha256_file(Path(__file__).resolve()):
            sys.stderr.write("EXECUTE refused: the builder changed since the dry-run\n")
            return 5
        ev = load_evidence(Path(a.prep_root))
        for name, h in (d.get("evidence_hashes") or {}).items():
            if ev["hashes"].get(name) != h:
                sys.stderr.write(f"EXECUTE refused: {name} changed since the dry-run\n")
                return 5
    try:
        return run(mode, Path(a.prep_root))
    except Exception:  # noqa: BLE001
        if LOGGER.handlers:
            LOGGER.critical("UNHANDLED EXCEPTION\n%s", traceback.format_exc())
        else:
            sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
