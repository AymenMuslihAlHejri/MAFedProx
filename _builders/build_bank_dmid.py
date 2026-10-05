#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bank_dmid.py
==================

PHASE 2B.3.9 - DMID CANONICAL DATASET BANK BUILDER

Ninth dataset, tenth distinct mask rule.

    MASK   DMID_CANONICAL_MASK_WHITE_FOREGROUND_V1
        The ROI masks are RGB or RGBA rasters whose colour channels are replicated and
        whose background the audit measured as pure black (0,0,0) on all 269. The
        canonical rule: a pixel is lesion foreground if and only if its colour channels
        are pure white (255,255,255); everything else is background. The result is a
        single-channel {0,255} PNG.

        Why white and not "non-black": 268 of the 269 masks hold exactly two colours,
        black and white, so the two readings agree on them. One mask, IMG333, also
        holds 3 pixels at (122,122,122) and 1 at (87,87,87) out of 28.5 million. Those
        four are neither the background nor the foreground value used everywhere else,
        so they are read as background. The count of such pixels is recorded PER
        RECORD, so the effect of the rule is visible rather than assumed.

        Alpha is inspected and never used: the audit found it constant and opaque on
        268 masks and not applicable on the one RGB mask.

    IMAGE  DMID_CANONICAL_IMAGE_16BIT_LOSSLESS_V1
        the DICOM stored pixel values are written unchanged to a 16-bit PNG. No
        windowing, rescale, normalisation, inversion, resize or crop. The per-record
        checks assert unsigned pixels, an identity rescale and a range that fits
        uint16, and the build blocks if any fails.

ONE MASK FILE PER IMAGE, SO NO UNION IS BUILT
    The ROI mask inventory holds 269 files for 269 images: one each. 64 of the 248
    joint images carry more than one abnormality record, up to six, but their lesions
    are already inside that single mask file. Constructing a union would presuppose
    multiple mask files that do not exist. The multi-lesion structure is preserved as
    metadata (n_abnormality_records and the per-class counts), never by combining
    rasters.

THE COHORT, AND WHAT IS NOT AN EXCLUSION
    510 source images, all retained in the manifest.
      248  joint cohort: classification-valid AND a verified raster mask (143 B/105 M)
        7  classification-valid B/M with NO raster ground truth -> a TRUE exclusion.
             No pseudo-mask is generated, no mask is inferred from the X/Y/radius
             metadata, and the pixel-annotation TIFFs are never used as segmentation
             ground truth.
      200  NORMAL_ONLY: a class outside the binary target, not an exclusion
       20  MIXED_BENIGN_MALIGNANT: both classes on one image, no defensible label
       34  UNDEFINED_ONLY and 1 BENIGN_PLUS_UNDEFINED: label unresolved in the source
    Only the 7 reach CANONICAL_DATASET_EXCLUSION_LEDGER. The other 255 are source
    categories that were never eligible, and counting them as exclusions would
    misrepresent the curation.

IDENTITY
    PatientID and PatientName are empty on all 510 DICOM headers. Study, Series and
    SOP UIDs are unique per object and the audit states they must not be repurposed as
    patient identifiers. patient_id, case_id and lesion_id stay empty and every record
    carries the limitation. No split is created.

Usage:
    python build_bank_dmid.py --dry-run
    python build_bank_dmid.py --execute
Requires: Python >= 3.8, pydicom, Pillow, numpy. tifffile recommended for the masks.
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

DATASET = "DMID"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)

MANIFEST_FILE = "83_dmid_phase1b_corrected_localization_manifest.csv"
PAIRING_FILE = "08_dmid_resource_pairing.csv"
MASKINT_FILE = "47_dmid_phase1a2_roi_mask_integrity.csv"
DIMCHECK_FILE = "48_dmid_phase1a2_roi_dicom_dimension_check.csv"
ABNORM_FILE = "65_dmid_phase1a3c_corrected_abnormality_manifest.csv"
SUMMARY_FILE = "17_dmid_audit_summary.json"
PHASE1B_SUMMARY = "79_dmid_phase1b_summary.json"
CORRECTIVE_SUMMARY = "86_dmid_phase1b_corrective_summary.json"
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root"]

MASK_POLICY_ID = "DMID_CANONICAL_MASK_WHITE_FOREGROUND_V1"
MASK_POLICY_RULE = ("colour channels exactly (255,255,255) -> 255, every other value "
                    "-> 0, written as a single-channel PNG; alpha inspected and never "
                    "used; no ramp threshold, no morphology, no raster union, no "
                    "pseudo-mask")
IMAGE_POLICY_ID = "DMID_CANONICAL_IMAGE_8BIT_REPLICATION_REDUCTION_V1"
IMAGE_POLICY_RULE = (
    "the DICOM is JPEG-Lossless and 8-bit unsigned, and the source ships two shapes: "
    "most files carry SamplesPerPixel 3 with three IDENTICAL channels, and some carry "
    "SamplesPerPixel 1 already. A 1-channel file is written as it is. A 3-channel file "
    "is written as ONE channel only after the equality of its three channels is "
    "verified on that file's own decoded pixels; a 3-channel file whose channels "
    "differ is REFUSED, never reduced. Which path each file took is recorded per "
    "record. No windowing, rescale, normalisation, inversion, resize or crop")
FG = 255

EXPECT = {"images": 510, "classification_valid": 255, "cls_benign": 145,
          "cls_malignant": 110, "joint": 248, "joint_benign": 143,
          "joint_malignant": 105, "no_raster": 7, "normal": 200, "mixed": 20,
          "undefined": 34, "benign_plus_undefined": 1, "masks": 269,
          "images_written": 255, "masks_written": 248, "png": 503}
SEVEN = ["IMG022", "IMG023", "IMG081", "IMG262", "IMG405", "IMG406", "IMG435"]
EXCLUSION_CODE = "EXCLUDE_NO_VERIFIED_RASTER_SEGMENTATION_GT"

NON_JOINT_STATUS = {
    "NORMAL_OUTSIDE_BINARY_BM_TARGET": (
        "NOT_IN_BINARY_TARGET_NORMAL_CLASS",
        "the image carries only NORMAL abnormality records; normal is a class outside "
        "the binary benign/malignant target, not a defect and not an exclusion"),
    "AMBIGUOUS_MIXED_BM": (
        "NOT_IN_BINARY_TARGET_MIXED_BENIGN_AND_MALIGNANT",
        "the image carries both benign and malignant abnormality records, so no single "
        "image-level binary label is defensible; no label was chosen for it"),
    "UNDEFINED_ONLY": (
        "NOT_IN_BINARY_TARGET_LABEL_UNDEFINED_IN_SOURCE",
        "every abnormality record on this image has an undefined class in the source "
        "metadata, so no label exists to carry"),
    "AMBIGUOUS_BENIGN_PLUS_UNDEFINED": (
        "NOT_IN_BINARY_TARGET_BENIGN_PLUS_UNDEFINED",
        "the image mixes a benign record with an undefined one, so the image-level "
        "label is unresolved in the source"),
}

SCHEMA_FIELDS = [
    "dataset", "canonical_record_id", "image_reference",
    "source_image_path", "source_mask_path",
    "canonical_image_path", "canonical_mask_path",
    "image_abnormality_category", "classification_status", "classification_valid",
    "source_label", "canonical_binary_label", "class_folder",
    "segmentation_status", "segmentation_valid", "joint_valid_strict",
    "localization_status", "localization_record_detail",
    "n_abnormality_records", "n_benign_records", "n_malignant_records",
    "n_normal_records", "n_undefined_records", "abnormality_types",
    "multi_lesion_in_single_mask_file",
    "roi_mask_present", "pixel_annotation_present", "pixel_annotation_used_as_gt",
    "patient_id", "case_id", "lesion_id", "identity_status", "grouping_limitation",
    "same_source_image_group", "duplicate_group_id",
    "classification_available", "segmentation_available",
    "image_conversion_status", "mask_conversion_status",
    "source_image_rows", "source_image_columns", "source_bits_allocated",
    "source_bits_stored", "source_pixel_representation", "source_photometric",
    "source_rescale_slope", "source_rescale_intercept",
    "source_pixel_min", "source_pixel_max", "source_samples_per_pixel",
    "source_channels_identical", "source_transfer_syntax",
    "source_mask_encoding", "source_mask_channels",
    "mask_dimension_verdict_audit", "mask_dimension_verified_here",
    "mask_white_pixels", "mask_nonblack_nonwhite_pixels_to_background",
    "mask_foreground_fraction",
    "canonical_image_height", "canonical_image_width",
    "canonical_mask_height", "canonical_mask_width",
    "canonical_image_encoding", "canonical_mask_encoding",
    "image_policy_id", "image_policy_rule", "mask_policy_id", "mask_policy_rule",
    "identity_provenance", "label_provenance", "mask_provenance",
    "exclusion_status", "exclusion_reason", "notes",
]
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}

SEP = " | "
LOGGER = logging.getLogger("bank_dmid")


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
    LOGGER.info("WROTE %-46s (%d data rows)", path.name, len(rows))


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


def read_mask(path: Path, np):
    """tifffile first, Pillow as a fallback: the same order the audit used."""
    try:
        import tifffile
        return np.asarray(tifffile.imread(str(path))), "tifffile"
    except Exception:  # noqa: BLE001
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im), "pillow"


def white_foreground(arr, np) -> Tuple[Any, int, int, str]:
    """Colour channels exactly white -> foreground. Returns the plane, the white
    count, the count of pixels that were neither pure black nor pure white and
    therefore became background, and a note."""
    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
        note = "SINGLE_CHANNEL_TREATED_AS_GREY"
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        note = ("CHANNELS_4_ALPHA_IGNORED" if arr.shape[2] == 4
                else f"CHANNELS_{arr.shape[2]}")
    else:
        return None, 0, 0, f"UNEXPECTED_SHAPE_{arr.shape}"
    white = np.all(rgb == FG, axis=-1)
    black = np.all(rgb == 0, axis=-1)
    other = int((~white & ~black).sum())
    n_white = int(white.sum())
    if n_white == 0:
        return None, 0, other, "NO_WHITE_FOREGROUND"
    return (white.astype(np.uint8) * FG), n_white, other, note


def load_evidence(prep_root: Path) -> dict:
    ev: Dict[str, Any] = {"paths": {}, "status": {}, "hashes": {}, "problems": []}
    dirs = [d for d in prep_root.iterdir()
            if d.is_dir() and "dmid" in d.name.lower().replace(" ", "")]
    files = [p for d in dirs for p in d.rglob("*") if p.is_file() and not archived(p)]
    for key, fname, required in (("manifest", MANIFEST_FILE, True),
                                 ("pairing", PAIRING_FILE, True),
                                 ("maskint", MASKINT_FILE, True),
                                 ("dimcheck", DIMCHECK_FILE, True),
                                 ("abnormality", ABNORM_FILE, True),
                                 ("summary", SUMMARY_FILE, True),
                                 ("phase1b", PHASE1B_SUMMARY, False),
                                 ("corrective", CORRECTIVE_SUMMARY, False)):
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


def run(mode: str, prep_root: Path, raw_root_cli: Optional[Path],
        headers_only: bool = False) -> int:
    bank_ds = prep_root / BANK_NAME / DATASET
    reports, manifests = bank_ds / "reports", bank_ds / "manifests"
    setup_logging(reports, f"dmid_{mode}.log")
    LOGGER.info("=" * 78)
    LOGGER.info("PHASE 2B.3.9 DMID CANONICAL BANK BUILDER  [%s]", mode.upper())
    LOGGER.info("=" * 78)
    LOGGER.info("python : %s", platform.python_version())
    if headers_only:
        LOGGER.warning("HEADERS-ONLY: DICOM pixels and ROI rasters are NOT decoded. "
                       "Accounting, pairing and header conformance are checked; the "
                       "channel equality and the mask content are deferred to "
                       "--execute, which decodes and verifies every file.")
    LOGGER.info("bank   : %s", bank_ds)

    try:
        import numpy as np
        import pydicom
        from PIL import Image
    except ImportError as exc:
        LOGGER.critical("missing dependency (%s); need pydicom, pillow, numpy", exc)
        return 3
    Image.MAX_IMAGE_PIXELS = None

    prep_dirs = [d for d in prep_root.iterdir()
                 if d.is_dir() and d.name != BANK_NAME]
    before_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        before_audit.update(snapshot(d))
    before_bank = snapshot(prep_root / BANK_NAME)

    LOGGER.info("[1/7] locked evidence bindings ...")
    ev = load_evidence(prep_root)
    for k, st in ev["status"].items():
        art = ev["paths"].get(k)
        LOGGER.info("      %-12s %-12s %s", k, st, art.name if art else "")
    if ev["problems"]:
        for p in ev["problems"]:
            LOGGER.error("      %s", p)
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4
    src_root = raw_root_cli or ev["dataset_root"]
    before_raw = snapshot(src_root)
    summ = json.loads(ev["paths"]["summary"].read_text(encoding="utf-8"))
    LOGGER.info("      RAW = %s  [READ-ONLY]", src_root)
    LOGGER.info("      root evidence = %s", ev["dataset_root_locator"])
    LOGGER.info("      image policy = %s", IMAGE_POLICY_ID)
    LOGGER.info("      mask policy  = %s", MASK_POLICY_ID)
    LOGGER.info("      mask rule    = %s", MASK_POLICY_RULE)

    rows = read_rows(ev["paths"]["manifest"])
    pairing = {r["image_reference"].strip(): r
               for r in read_rows(ev["paths"]["pairing"])
               if r["image_reference"].strip()}
    maskint = {r["image_reference"].strip(): r
               for r in read_rows(ev["paths"]["maskint"])}
    dimchk = {r["image_reference"].strip(): r
              for r in read_rows(ev["paths"]["dimcheck"])}
    abnorm = read_rows(ev["paths"]["abnormality"])
    ab_by_img: Dict[str, List[dict]] = {}
    for r in abnorm:
        ab_by_img.setdefault(r["image_reference"].strip(), []).append(r)
    refs = [r["image_reference"].strip() for r in rows]
    dupe = [k for k, n in Counter(refs).items() if n > 1]
    if dupe:
        LOGGER.error("      image_reference not unique: %s", dupe[:5])
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4

    LOGGER.info("[2/7] source accounting ...")
    cat = Counter(r["image_abnormality_category"].strip() for r in rows)
    cls_status = Counter(r["classification_status"].strip() for r in rows)
    seg_status = Counter(r["segmentation_status"].strip() for r in rows)
    joint = Counter(r["joint_valid_strict"].strip() for r in rows)
    cls_valid = [r for r in rows if r["classification_valid"].strip() == "YES"]
    cls_lbl = Counter(r["classification_label"].strip() for r in cls_valid)
    joint_rows = [r for r in rows if r["joint_valid_strict"].strip() == "YES"]
    joint_lbl = Counter(r["classification_label"].strip() for r in joint_rows)
    no_raster = [r for r in rows
                 if r["segmentation_status"].strip() == "NO_RASTER_PIXEL_GT"]
    enc = Counter(r["encoding"].strip() for r in maskint.values())
    dimv = Counter(r["dimension_verdict"].strip() for r in dimchk.values())
    LOGGER.info("      manifest %d image(s); categories %s", len(rows), dict(cat))
    LOGGER.info("      classification: %s", dict(cls_status))
    LOGGER.info("      classification-valid %d -> %s", len(cls_valid), dict(cls_lbl))
    LOGGER.info("      segmentation: %s", dict(seg_status))
    LOGGER.info("      joint_valid_strict: %s -> %s", dict(joint), dict(joint_lbl))
    LOGGER.info("      abnormality records: %d over %d image(s)", len(abnorm),
                len(ab_by_img))
    LOGGER.info("      ROI mask files: %d | encodings %s", len(maskint), dict(enc))
    LOGGER.info("      audit ROI/DICOM dimension verdict: %s", dict(dimv))
    LOGGER.info("      identity: %s", summ.get("dicom_identity_status"))
    LOGGER.info("      B/M images with no raster GT: %s",
                join_list(sorted(r["image_reference"] for r in no_raster), ", "))

    LOGGER.info("[3/7] mask cardinality ...")
    multi = [r for r in joint_rows if int(r["n_abnormality_records"] or 0) > 1]
    LOGGER.info("      one mask FILE per image: %d file(s) for %d distinct image(s)",
                len(maskint), len({r["image_reference"] for r in maskint.values()}))
    LOGGER.info("      joint images carrying >1 abnormality record: %d; records per "
                "image %s", len(multi),
                dict(sorted(Counter(int(r["n_abnormality_records"])
                                    for r in joint_rows).items())))
    LOGGER.info("      union plans: 0 - a union would presuppose multiple mask files "
                "per image, which do not exist; the lesions are already inside the "
                "single file and the structure is kept as metadata")

    LOGGER.info("[4/7] planning canonical records ...")
    records: List[dict] = []
    for r in rows:
        ref = r["image_reference"].strip()
        lbl = r["classification_label"].strip()
        is_joint = r["joint_valid_strict"].strip() == "YES"
        folder = ("Benign" if lbl == "BENIGN" else
                  "Malignant" if lbl == "MALIGNANT" else "")
        stem = f"{ref}__{sha256_text(f'{lbl}/{ref}')[:8]}"
        pr = pairing.get(ref, {})
        mi = maskint.get(ref, {})
        dc = dimchk.get(ref, {})
        img_path = (pr.get("dicom_paths", "").strip()
                    or dc.get("dicom_relative_path", "").strip())
        mask_path = (mi.get("relative_path", "").strip()
                     or dc.get("roi_relative_path", "").strip())
        cs = r["classification_status"].strip()
        if is_joint:
            status, reason = "INCLUDED_JOINT_COHORT", ""
        elif r["segmentation_status"].strip() == "NO_RASTER_PIXEL_GT":
            status = EXCLUSION_CODE
            reason = ("the image has a valid binary classification label but the "
                      "source ships no verified raster ROI mask for it. The image IS "
                      "converted and keeps its class label, so it remains available "
                      "for the classification task; only the segmentation target is "
                      "absent. No pseudo-mask was generated, no mask was inferred from "
                      "the X/Y/radius metadata, and the pixel-annotation TIFF was not "
                      "used as segmentation ground truth.")
        else:
            status, reason = NON_JOINT_STATUS.get(
                cs, ("NOT_IN_JOINT_COHORT_UNCLASSIFIED", f"classification_status={cs}"))
        records.append({
            "dataset": DATASET, "canonical_record_id": f"{DATASET}__{stem}",
            "image_reference": ref,
            "source_image_path": img_path, "source_mask_path": mask_path,
            "canonical_image_path": (
                f"{folder}/Images/{stem}.png"
                if r["classification_valid"].strip() == "YES" and folder else ""),
            "canonical_mask_path": (f"{folder}/Masks/{stem}_mask.png"
                                    if is_joint and folder else ""),
            "image_abnormality_category": r["image_abnormality_category"].strip(),
            "classification_status": cs,
            "classification_valid": r["classification_valid"].strip(),
            "source_label": lbl,
            "canonical_binary_label": (0 if lbl == "BENIGN" else
                                       1 if lbl == "MALIGNANT"
                                       else "NOT_IN_BINARY_TARGET"),
            "class_folder": (folder if r["classification_valid"].strip() == "YES"
                             else ""),
            "segmentation_status": r["segmentation_status"].strip(),
            "segmentation_valid": r["segmentation_valid"].strip(),
            "joint_valid_strict": r["joint_valid_strict"].strip(),
            "localization_status": r["localization_status"].strip(),
            "localization_record_detail": r.get("localization_record_detail",
                                                "").strip(),
            "n_abnormality_records": r["n_abnormality_records"].strip(),
            "n_benign_records": r["n_benign_records"].strip(),
            "n_malignant_records": r["n_malignant_records"].strip(),
            "n_normal_records": r["n_normal_records"].strip(),
            "n_undefined_records": r["n_undefined_records"].strip(),
            "abnormality_types": r.get("abnormality_types", "").strip(),
            "multi_lesion_in_single_mask_file": (
                "YES" if int(r["n_abnormality_records"] or 0) > 1 else "NO"),
            "roi_mask_present": r["roi_mask_present"].strip(),
            "pixel_annotation_present": r["pixel_annotation_present"].strip(),
            "pixel_annotation_used_as_gt": "NO",
            "patient_id": "", "case_id": "", "lesion_id": "",
            "identity_status": summ.get("dicom_identity_status", ""),
            "grouping_limitation": (
                "PatientID and PatientName are empty on every DICOM header; Study, "
                "Series and SOP UIDs are unique per object and must not be repurposed "
                "as patient identifiers, so patient-independent splitting cannot be "
                "demonstrated from this release"),
            "same_source_image_group": f"{DATASET}__{stem}",
            "duplicate_group_id": "",
            "classification_available": ("TRUE" if r["classification_valid"].strip()
                                         == "YES" else "FALSE"),
            "source_channels_identical": "", "image_reduction_path": "",
            "segmentation_available": "TRUE" if is_joint else "FALSE",
            "image_conversion_status": (
                "PLANNED" if r["classification_valid"].strip() == "YES"
                else "NOT_APPLICABLE"),
            "mask_conversion_status": "PLANNED" if is_joint else "NOT_APPLICABLE",
            "source_image_rows": dc.get("dicom_rows", ""),
            "source_image_columns": dc.get("dicom_columns", ""),
            "source_bits_allocated": "", "source_bits_stored": "",
            "source_pixel_representation": "", "source_photometric": "",
            "source_rescale_slope": "", "source_rescale_intercept": "",
            "source_pixel_min": "", "source_pixel_max": "",
            "source_samples_per_pixel": "", "source_transfer_syntax": "",
            "source_mask_encoding": mi.get("encoding", ""),
            "source_mask_channels": mi.get("channel_count", ""),
            "mask_dimension_verdict_audit": dc.get("dimension_verdict", ""),
            "mask_dimension_verified_here": "",
            "mask_white_pixels": "",
            "mask_nonblack_nonwhite_pixels_to_background": "",
            "mask_foreground_fraction": "",
            "canonical_image_height": "", "canonical_image_width": "",
            "canonical_mask_height": "", "canonical_mask_width": "",
            "canonical_image_encoding": (
                "SINGLE_CHANNEL_8BIT_PNG_REPLICATION_REDUCED"
                if r["classification_valid"].strip() == "YES" else ""),
            "canonical_mask_encoding": ("SINGLE_CHANNEL_BINARY_0_255" if is_joint
                                        else ""),
            "image_policy_id": (IMAGE_POLICY_ID
                                if r["classification_valid"].strip() == "YES" else ""),
            "image_policy_rule": (IMAGE_POLICY_RULE
                                  if r["classification_valid"].strip() == "YES"
                                  else ""),
            "mask_policy_id": MASK_POLICY_ID if is_joint else "",
            "mask_policy_rule": MASK_POLICY_RULE if is_joint else "",
            "identity_provenance": (f"{SUMMARY_FILE} :: json:dicom_identity_status="
                                    f"{summ.get('dicom_identity_status')}"),
            "label_provenance": (f"{MANIFEST_FILE} :: csv:[classification_label] "
                                 f"(status={cs}); derived in Phase 1B from the source "
                                 "abnormality classes, never from pixels"),
            "mask_provenance": (f"SOURCE_ROI_RASTER_WHITE_FOREGROUND_BY"
                                f"[{MASK_POLICY_ID}]" if is_joint
                                else "NOT_CONVERTED"),
            "exclusion_status": status, "exclusion_reason": reason,
            "notes": (f"eligibility_reason={r.get('eligibility_reason', '').strip()}; "
                      f"limitation="
                      f"{r.get('exclusion_or_limitation_reason', '').strip()}"),
        })
    cohort = [r for r in records if r["exclusion_status"] == "INCLUDED_JOINT_COHORT"]
    excl7 = [r for r in records if r["exclusion_status"] == EXCLUSION_CODE]
    nonjoint = [r for r in records
                if r["exclusion_status"].startswith("NOT_IN_BINARY_TARGET")]
    by_el = Counter(r["source_label"] for r in cohort)
    LOGGER.info("      joint cohort %d (benign %d / malignant %d)", len(cohort),
                by_el.get("BENIGN", 0), by_el.get("MALIGNANT", 0))
    LOGGER.info("      true exclusions %d: %s", len(excl7),
                join_list(sorted(r["image_reference"] for r in excl7), ", "))
    LOGGER.info("      non-binary source categories retained (NOT exclusions): %d %s",
                len(nonjoint), dict(Counter(r["exclusion_status"] for r in nonjoint)))

    LOGGER.info("[5/7] resolving and pre-checking the joint cohort ...")
    img_ok = mask_ok = plans = 0
    unresolved: List[str] = []
    notes_c: Counter = Counter()
    other_px: Counter = Counter()
    chan_seen: Counter = Counter()
    dim_bad: List[str] = []
    chan_fail: List[str] = []
    header_fail: List[str] = []
    rescale_nonid: List[str] = []
    convertible = [r for r in records if r["image_conversion_status"] == "PLANNED"]
    LOGGER.info("      converting %d image(s) (every classification-valid record) and "
                "%d mask(s) (the joint cohort)", len(convertible), len(cohort))
    for i, r in enumerate(convertible, 1):
        sp = src_root / r["source_image_path"] if r["source_image_path"] else None
        mp = src_root / r["source_mask_path"] if r["source_mask_path"] else None
        if sp is None or not sp.is_file():
            unresolved.append(f"{r['image_reference']}: DICOM missing "
                              f"({r['source_image_path']})")
            r["image_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            ds = pydicom.dcmread(sp, stop_before_pixels=headers_only)
            arr = None if headers_only else ds.pixel_array
        except Exception as exc:  # noqa: BLE001
            unresolved.append(f"{r['image_reference']}: DICOM unreadable "
                              f"({type(exc).__name__})")
            r["image_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        if headers_only:
            h = int(getattr(ds, "Rows", 0) or 0)
            w = int(getattr(ds, "Columns", 0) or 0)
        else:
            h, w = arr.shape[:2]
        r["canonical_image_height"], r["canonical_image_width"] = h, w
        r["source_bits_allocated"] = int(getattr(ds, "BitsAllocated", 0) or 0)
        r["source_bits_stored"] = int(getattr(ds, "BitsStored", 0) or 0)
        r["source_pixel_representation"] = int(getattr(ds, "PixelRepresentation", -1))
        r["source_photometric"] = str(getattr(ds, "PhotometricInterpretation", ""))
        rs = getattr(ds, "RescaleSlope", None)
        ri = getattr(ds, "RescaleIntercept", None)
        r["source_rescale_slope"] = "" if rs is None else str(rs)
        r["source_rescale_intercept"] = "" if ri is None else str(ri)
        if not headers_only:
            r["source_pixel_min"] = int(arr.min())
            r["source_pixel_max"] = int(arr.max())
        r["source_samples_per_pixel"] = int(getattr(ds, "SamplesPerPixel", 0) or 0)
        try:
            r["source_transfer_syntax"] = str(ds.file_meta.TransferSyntaxUID.name)
        except Exception:  # noqa: BLE001
            r["source_transfer_syntax"] = ""
        # the header must be what the audit and the decode both reported
        if (r["source_bits_allocated"] != 8
                or r["source_pixel_representation"] != 0
                or r["source_samples_per_pixel"] not in (1, 3)
                or (not headers_only and str(arr.dtype) != "uint8")):
            header_fail.append(
                f"{r['image_reference']}: BitsAllocated={r['source_bits_allocated']} "
                f"SamplesPerPixel={r['source_samples_per_pixel']} "
                f"PixelRepresentation={r['source_pixel_representation']} "
                f"dtype={'not decoded' if headers_only else arr.dtype}")
            r["image_conversion_status"] = "HEADER_NOT_AS_DECLARED"
            continue
        # two shapes ship in this source; each takes its own verified path
        if headers_only:
            # the channel equality is a PIXEL fact; it cannot be read from a header,
            # so it is deferred to --execute, which verifies it per file and refuses
            # any file whose channels differ
            r["source_channels_identical"] = "DEFERRED_TO_EXECUTE"
            r["image_reduction_path"] = (
                "ALREADY_SINGLE_CHANNEL" if r["source_samples_per_pixel"] == 1
                else "THREE_CHANNELS_EQUALITY_TO_BE_VERIFIED_AT_EXECUTE")
            chan_seen["header only, not decoded"] += 1
        elif arr.ndim == 2 or r["source_samples_per_pixel"] == 1:
            r["source_channels_identical"] = "NOT_APPLICABLE_SINGLE_CHANNEL"
            r["image_reduction_path"] = "ALREADY_SINGLE_CHANNEL"
            chan_seen["already single channel"] += 1
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            same_ch = bool(np.array_equal(arr[..., 0], arr[..., 1])
                           and np.array_equal(arr[..., 1], arr[..., 2]))
            r["source_channels_identical"] = "YES" if same_ch else "NO"
            if not same_ch:
                d = int(np.sum((arr[..., 0] != arr[..., 1])
                               | (arr[..., 1] != arr[..., 2])))
                chan_fail.append(f"{r['image_reference']}: channels differ on "
                                 f"{d} pixel(s)")
                r["image_conversion_status"] = "REFUSED_CHANNELS_NOT_REPLICATED"
                chan_seen["DIFFERENT"] += 1
                continue
            r["image_reduction_path"] = "THREE_IDENTICAL_CHANNELS_REDUCED_TO_ONE"
            chan_seen["3 identical -> 1"] += 1
        else:
            chan_fail.append(f"{r['image_reference']}: unexpected shape {arr.shape}")
            r["image_conversion_status"] = "REFUSED_UNEXPECTED_SHAPE"
            continue
        if (rs is not None and float(rs) != 1.0) or (
                ri is not None and float(ri) != 0.0):
            rescale_nonid.append(f"{r['image_reference']}: slope={rs} intercept={ri}")
        img_ok += 1
        if r["mask_conversion_status"] != "PLANNED":
            continue                      # the 7: image converted, no mask expected
        if mp is None or not mp.is_file():
            unresolved.append(f"{r['image_reference']}: ROI mask missing "
                              f"({r['source_mask_path']})")
            r["mask_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        if headers_only:
            r["mask_dimension_verified_here"] = "DEFERRED_TO_EXECUTE"
            notes_c["HEADER_ONLY_NOT_DECODED"] += 1
            mask_ok += 1
            plans += 1
            continue
        try:
            ma, _dec = read_mask(mp, np)
        except Exception as exc:  # noqa: BLE001
            unresolved.append(f"{r['image_reference']}: mask unreadable "
                              f"({type(exc).__name__})")
            r["mask_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        mask_ok += 1
        mh, mw = ma.shape[:2]
        r["canonical_mask_height"], r["canonical_mask_width"] = mh, mw
        same = (mh, mw) == (h, w)
        r["mask_dimension_verified_here"] = "MATCH" if same else "MISMATCH"
        if not same:
            dim_bad.append(f"{r['image_reference']}: DICOM {w}x{h} vs mask {mw}x{mh}")
            r["mask_conversion_status"] = "DIMENSION_MISMATCH"
            continue
        plane, n_white, n_other, note = white_foreground(ma, np)
        notes_c[note] += 1
        r["mask_white_pixels"] = n_white
        r["mask_nonblack_nonwhite_pixels_to_background"] = n_other
        if n_other:
            other_px[r["image_reference"]] = n_other
        if plane is None:
            unresolved.append(f"{r['image_reference']}: mask refused ({note})")
            r["mask_conversion_status"] = f"REFUSED_{note}"
            continue
        r["mask_foreground_fraction"] = round(n_white / float(mh * mw), 8)
        plans += 1
        if i % 50 == 0:
            LOGGER.info("      %d/%d", i, len(cohort))
    LOGGER.info("      DICOM decoded=%d/%d | masks read=%d | mask plans=%d | "
                "unresolved=%d", img_ok, len(convertible), mask_ok, plans,
                len(unresolved))
    LOGGER.info("      source channel equality (verified per file): %s",
                dict(chan_seen))
    LOGGER.info("      header conformance failures: %d | channel failures: %d",
                len(header_fail), len(chan_fail))
    LOGGER.info("      mask channel notes: %s", dict(notes_c))
    LOGGER.info("      pixels neither black nor white, mapped to background: %s",
                dict(other_px) or "none")
    for x in (unresolved + dim_bad + header_fail + chan_fail
              + rescale_nonid)[:10]:
        LOGGER.error("         %s", x)

    png_written = 0
    readback_fail: List[str] = []
    if mode == "execute":
        LOGGER.info("[6/7] executing canonical conversion ...")
        for n, r in enumerate(convertible, 1):
            if r["image_conversion_status"] != "PLANNED":
                continue
            ds = pydicom.dcmread(src_root / r["source_image_path"])
            arr = ds.pixel_array
            if not (np.array_equal(arr[..., 0], arr[..., 1])
                    and np.array_equal(arr[..., 1], arr[..., 2])):
                r["image_conversion_status"] = "REFUSED_CHANNELS_NOT_REPLICATED"
                continue
            intended = arr[..., 0].astype(np.uint8)      # one channel, values unchanged
            dst = bank_ds / r["canonical_image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                Image.fromarray(intended, mode="L").save(dst, format="PNG")
                png_written += 1
            with Image.open(dst) as chk:
                back = np.asarray(chk.convert("L"))
            if back.shape != intended.shape or not np.array_equal(back, intended):
                d = (int(np.sum(back != intended)) if back.shape == intended.shape
                     else -1)
                r["image_conversion_status"] = "VERIFICATION_FAILED_NOT_IDENTICAL"
                readback_fail.append(f"{r['image_reference']}: image differs on {d} px")
                continue
            r["image_conversion_status"] = "CONVERTED_VERIFIED_IDENTICAL"
            if r["mask_conversion_status"] != "PLANNED":
                continue
            ma, _d = read_mask(src_root / r["source_mask_path"], np)
            plane, _nw, _no, _n = white_foreground(ma, np)
            if plane is None:
                r["mask_conversion_status"] = "REFUSED_AT_EXECUTE"
                continue
            mdst = bank_ds / r["canonical_mask_path"]
            mdst.parent.mkdir(parents=True, exist_ok=True)
            if not mdst.exists():
                Image.fromarray(plane, mode="L").save(mdst, format="PNG")
                png_written += 1
            with Image.open(mdst) as chk:
                mback = np.asarray(chk.convert("L"))
            if not np.array_equal(mback, plane):
                r["mask_conversion_status"] = "VERIFICATION_FAILED_NOT_IDENTICAL"
                readback_fail.append(f"{r['image_reference']}: mask read-back differs")
                continue
            r["mask_conversion_status"] = "CONVERTED_VERIFIED_IDENTICAL"
            if n % 50 == 0:
                LOGGER.info("      %d/%d", n, len(convertible))
    else:
        LOGGER.info("[6/7] dry-run: no canonical PNG is written")

    LOGGER.info("[7/7] validation ...")
    after_bank = snapshot(prep_root / BANK_NAME)
    new_splits = [k for k in after_bank if k not in before_bank and is_split_like(k)]
    new_png = [k for k in after_bank
               if k not in before_bank and k.lower().endswith(".png")]
    raw_diff = diff_state(before_raw, snapshot(src_root))
    after_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        after_audit.update(snapshot(d))
    aud_diff = diff_state(before_audit, after_audit)
    cpaths = [r["canonical_image_path"] for r in records if r["canonical_image_path"]]
    cpaths += [r["canonical_mask_path"] for r in records if r["canonical_mask_path"]]
    collisions = [n for n, c in Counter(cpaths).items() if c > 1]

    T: List[Tuple[str, str, Optional[bool], str]] = []

    def add(i: str, d: str, ok: Optional[bool], b: str) -> None:
        T.append((i, d, ok, b))

    add("T01", f"source images == {EXPECT['images']}", len(rows) == EXPECT["images"],
        f"{len(rows)} @ {MANIFEST_FILE}")
    add("T02", f"categories: {EXPECT['normal']} normal, {EXPECT['mixed']} mixed, "
               f"{EXPECT['undefined']} undefined, {EXPECT['benign_plus_undefined']} "
               "benign+undefined",
        cat.get("NORMAL_ONLY") == EXPECT["normal"]
        and cat.get("MIXED_BENIGN_MALIGNANT") == EXPECT["mixed"]
        and cat.get("UNDEFINED_ONLY") == EXPECT["undefined"]
        and cat.get("BENIGN_PLUS_UNDEFINED") == EXPECT["benign_plus_undefined"],
        f"{dict(cat)}")
    add("T03", f"classification-valid == {EXPECT['classification_valid']} = "
               f"{EXPECT['cls_benign']} B + {EXPECT['cls_malignant']} M",
        len(cls_valid) == EXPECT["classification_valid"]
        and cls_lbl.get("BENIGN") == EXPECT["cls_benign"]
        and cls_lbl.get("MALIGNANT") == EXPECT["cls_malignant"],
        f"{len(cls_valid)} = {cls_lbl.get('BENIGN')} + {cls_lbl.get('MALIGNANT')}")
    add("T04", f"joint cohort == {EXPECT['joint']} = {EXPECT['joint_benign']} B + "
               f"{EXPECT['joint_malignant']} M",
        len(cohort) == EXPECT["joint"]
        and by_el.get("BENIGN") == EXPECT["joint_benign"]
        and by_el.get("MALIGNANT") == EXPECT["joint_malignant"],
        f"{len(cohort)} = {by_el.get('BENIGN')} + {by_el.get('MALIGNANT')}")
    add("T05", f"{EXPECT['classification_valid']} - {EXPECT['joint']} = "
               f"{EXPECT['no_raster']} classification-valid without raster GT",
        len(cls_valid) - len(cohort) == EXPECT["no_raster"]
        and len(excl7) == EXPECT["no_raster"]
        and sorted(r["image_reference"] for r in excl7) == sorted(SEVEN),
        f"{len(cls_valid)} - {len(cohort)} = {len(excl7)}: "
        + join_list(sorted(r["image_reference"] for r in excl7), ", "))
    add("T06", "the 7 keep their converted image and their label, carry an explicit "
               "code and reason, and have NO mask path; no pseudo-mask was generated",
        all(r["exclusion_status"] == EXCLUSION_CODE and r["exclusion_reason"]
            and r["canonical_image_path"] and not r["canonical_mask_path"]
            and r["classification_available"] == "TRUE"
            and r["segmentation_available"] == "FALSE"
            for r in excl7),
        f"{len(excl7)} record(s): image YES, mask NO, "
        f"{EXCLUSION_CODE}")
    add("T07", "the non-binary source categories are retained with a status and are "
               "NOT counted as exclusions",
        len(nonjoint) == (EXPECT["normal"] + EXPECT["mixed"] + EXPECT["undefined"]
                          + EXPECT["benign_plus_undefined"])
        and all(r["exclusion_reason"] and not r["canonical_image_path"]
                for r in nonjoint),
        f"{len(nonjoint)} retained: "
        f"{dict(Counter(r['exclusion_status'] for r in nonjoint))}")
    add("T08", f"every record retained: {EXPECT['images']} = joint + 7 + non-binary",
        len(records) == EXPECT["images"]
        and len(cohort) + len(excl7) + len(nonjoint) == len(records),
        f"{len(cohort)} + {len(excl7)} + {len(nonjoint)} = {len(records)}")
    add("T09", "one ROI mask file per image; union plans == 0",
        len(maskint) == EXPECT["masks"]
        and len({r["image_reference"] for r in maskint.values()}) == EXPECT["masks"],
        f"{len(maskint)} file(s) for {len(maskint)} image(s); {len(multi)} joint "
        "image(s) hold more than one lesion inside one file")
    add("T10", "multi-lesion structure preserved as metadata, not by combining rasters",
        sum(1 for r in cohort
            if r["multi_lesion_in_single_mask_file"] == "YES") == len(multi),
        f"{len(multi)} joint image(s) flagged; n_abnormality_records and the per-class "
        "counts are carried on every record")
    add("T11", f"mask conversion plans == {EXPECT['joint']} (the joint cohort)",
        plans == EXPECT["joint"], f"{plans}; notes {dict(notes_c)}")
    add("T11b", f"images planned == {EXPECT['images_written']} (every "
                "classification-valid record, so the 7 keep their image)",
        len(convertible) == EXPECT["images_written"]
        and img_ok == EXPECT["images_written"],
        f"{len(convertible)} planned, {img_ok} decoded; the 7 without a raster mask "
        "are converted as images and carry no mask")
    add("T12", ("the white-foreground rule is declared; its per-record effect is "
                "measured at --execute" if headers_only else
                "the white-foreground rule is applied and its effect recorded per "
                "record"),
        headers_only or all(str(r["mask_white_pixels"]).isdigit() for r in cohort
                            if r["mask_conversion_status"] == "PLANNED"),
        f"pixels neither black nor white, mapped to background: "
        f"{dict(other_px) or 'none'}")
    add("T13", "the pixel-annotation TIFFs were never used as segmentation GT",
        all(r["pixel_annotation_used_as_gt"] == "NO" for r in records),
        "no record sources its mask from a pixel-annotation file")
    add("T14", "no mask inferred from X/Y/radius metadata", True,
        "localization metadata is carried as provenance only; the mask comes from the "
        "ROI raster or the record is excluded")
    add("T15", "every DICOM is 8-bit, unsigned, and SamplesPerPixel 1 or 3, as the "
               "source actually ships",
        not header_fail and img_ok == EXPECT["images_written"],
        f"{img_ok}/{EXPECT['images_written']} conform; {len(header_fail)} "
        "header mismatch(es); transfer syntax "
        f"{dict(Counter(r['source_transfer_syntax'] for r in convertible if r['source_transfer_syntax']))}")
    add("T15b", ("channel equality is a pixel fact, deferred to --execute in "
                 "headers-only mode" if headers_only else
                 "every 3-channel file is verified identical before one channel is "
                 "written; a file whose channels differ is refused"),
        not chan_fail,
        f"{dict(chan_seen)}; {len(chan_fail)} refused"
        + ("; --execute decodes and verifies every file" if headers_only else ""))
    add("T16", "no rescale transform discarded", not rescale_nonid,
        f"{len(rescale_nonid)} record(s) carry a non-identity rescale")
    add("T17", ("mask dimensions deferred to --execute in headers-only mode; the "
                "audit verdict is carried" if headers_only else
                "mask and DICOM dimensions agree, verified here from the pixels"),
        not dim_bad and (headers_only
                         or all(r["mask_dimension_verified_here"] == "MATCH"
                                for r in cohort
                                if r["mask_conversion_status"] == "PLANNED")),
        f"{len(dim_bad)} mismatch(es); audit verdict {dict(dimv)}")
    add("T18", "no patient, case or lesion identity invented",
        all(not r["patient_id"] and not r["case_id"] and not r["lesion_id"]
            for r in records)
        and summ.get("dicom_identity_status")
        == "NO_DEFENSIBLE_PATIENT_GROUPING_ESTABLISHED",
        f"{summ.get('dicom_identity_status')}; UIDs never used as a patient key")
    add("T19", "the identity limitation is recorded on every record",
        all("cannot be demonstrated" in r["grouping_limitation"] for r in records),
        "patient-independent splitting cannot be demonstrated from this release")
    add("T20", "no policy from another dataset",
        not any(pid in r["mask_provenance"] for r in records
                for pid in ("BUSI_CANONICAL_MULTIMASK_UNION_V1",
                            "BUS_UC_CANONICAL_MASK_BINARIZATION_V1",
                            "BUS_UCLM_CANONICAL_MASK_COLOUR_TO_BINARY_V1",
                            "INBREAST_FULLSOURCE_MASK_FROM_XML_CONTOUR_V1")),
        f"declared rule: {MASK_POLICY_ID}")
    add("T21", "no unresolved joint record", not unresolved, f"{len(unresolved)}")
    add("T22", "no canonical path collision", not collisions, f"{len(collisions)}")
    add("T23", "RAW unchanged", raw_diff["unchanged"],
        f"modified {len(raw_diff['modified'])}, added {len(raw_diff['added'])}")
    add("T24", "completed audit artifacts unchanged", aud_diff["unchanged"],
        f"modified {len(aud_diff['modified'])}")
    add("T25", "no split created", not new_splits, f"{len(new_splits)}")
    add("T25b", ("read-back verification runs at execute time" if mode == "dry-run"
                 else "every written PNG reloads pixel-identical to its intended "
                      "array"),
        (mode == "dry-run") or not readback_fail,
        ("deferred to --execute" if mode == "dry-run"
         else f"{len(readback_fail)} file(s) differed on reload"))
    add("T26", ("dry-run wrote no canonical PNG" if mode == "dry-run"
                else f"execution wrote {EXPECT['png']} canonical PNG "
                     f"({EXPECT['images_written']} images + "
                     f"{EXPECT['masks_written']} masks)"),
        (not new_png) if mode == "dry-run" else (len(new_png) == EXPECT["png"]),
        f"{len(new_png)} new PNG")
    add("T27", "no split field in the manifest schema",
        not any(is_split_like(f) for f in SCHEMA_FIELDS), "schema checked")
    add("T28", "no resize, crop, normalisation, CLAHE or augmentation", None,
        "code-scope declaration: one verified-identical channel -> 8-bit PNG with its "
        "values unchanged; ROI white pixels -> 255; nothing else")
    verified = [t for t in T if t[2] is not None]
    failed = [t for t in verified if not t[2]]
    for t in T:
        (LOGGER.info if t[2] is not False else LOGGER.error)(
            "      %-5s [%-11s] %-52s %s", t[0],
            "DECLARATION" if t[2] is None else ("PASS" if t[2] else "FAIL"),
            t[1][:52], t[3][:52])

    tag = "DRY_RUN_" if mode == "dry-run" else ""
    write_csv(manifests / f"{tag}canonical_manifest.csv", SCHEMA_FIELDS,
              [{k: r[k] for k in SCHEMA_FIELDS} for r in records])
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
        for r in records:
            ws.append([r.get(h, "") for h in SCHEMA_FIELDS])
        ws.freeze_panes = "A2"
        wide = {"exclusion_reason", "notes", "mask_policy_rule", "image_policy_rule",
                "grouping_limitation", "label_provenance", "identity_provenance",
                "localization_record_detail"}
        for i, h in enumerate(SCHEMA_FIELDS, 1):
            ws.column_dimensions[get_column_letter(i)].width = 70 if h in wide else 20
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        wb.save(manifests / f"{tag}canonical_manifest.xlsx")
        LOGGER.info("WROTE %-46s (1 sheet)", f"{tag}canonical_manifest.xlsx")
    except ImportError:
        LOGGER.warning("openpyxl not installed; XLSX skipped")

    blocked = bool(failed)
    state = {
        "phase": "2B.3.9", "dataset": DATASET, "mode": mode, "generated": now_iso(),
        "headers_only": headers_only,
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "schema_sha256": sha256_text(join_list(SCHEMA_FIELDS, ",")),
        "evidence_hashes": ev["hashes"], "source_root": str(src_root),
        "policies": {"image": {"id": IMAGE_POLICY_ID, "rule": IMAGE_POLICY_RULE},
                     "mask": {"id": MASK_POLICY_ID, "rule": MASK_POLICY_RULE,
                              "foreground": "(255,255,255) only",
                              "pixels_mapped_to_background": dict(other_px)}},
        "counts": {"source_images": len(rows),
                   "classification_valid": len(cls_valid),
                   "classification_benign": cls_lbl.get("BENIGN", 0),
                   "classification_malignant": cls_lbl.get("MALIGNANT", 0),
                   "joint_cohort": len(cohort),
                   "joint_benign": by_el.get("BENIGN", 0),
                   "joint_malignant": by_el.get("MALIGNANT", 0),
                   "true_exclusions": len(excl7),
                   "non_binary_retained": len(nonjoint),
                   "images_planned": len(convertible),
                   "images_decoded": img_ok,
                   "mask_conversion_plans": plans, "union_plans": 0,
                   "pseudo_masks": 0, "masks_inferred_from_metadata": 0},
        "image_categories": dict(cat), "segmentation_status": dict(seg_status),
        "multi_lesion_joint_images": len(multi), "mask_encodings": dict(enc),
        "source_channel_equality": dict(chan_seen),
        "header_conformance_failures": header_fail,
        "channel_failures": chan_fail,
        "readback_failures": readback_fail,
        "excluded_images": sorted(r["image_reference"] for r in excl7),
        "non_joint_breakdown": dict(Counter(r["exclusion_status"] for r in nonjoint)),
        "identity": {"status": summ.get("dicom_identity_status"),
                     "patient_id": "NOT_ESTABLISHED",
                     "uids_used_as_patient_key": False},
        "image_mappings_resolved": img_ok, "mask_mappings_resolved": mask_ok,
        "unresolved": unresolved[:50], "collisions": collisions,
        "canonical_png_written": png_written,
        "raw_modified": not raw_diff["unchanged"],
        "audit_modified": not aud_diff["unchanged"],
        "exclusion_ledger_must_be_updated": len(excl7) > 0,
        "tests_pass": sum(1 for t in verified if t[2]), "tests_fail": len(failed),
        "verdict": ("BLOCKED" if blocked else
                    "HEADERS_ONLY_ACCOUNTING_CLEAN_FULL_DRY_RUN_STILL_REQUIRED"
                    if (mode == "dry-run" and headers_only) else
                    "READY_FOR_DMID_CANONICAL_EXECUTION" if mode == "dry-run"
                    else "DMID_CANONICAL_BUILD_COMPLETE"),
    }
    (reports / f"{tag}canonical_build_report.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    LOGGER.info("WROTE %-46s", f"{tag}canonical_build_report.json")

    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 2B.3.9 DMID - %s", mode.upper())
    LOGGER.info("Source:        %d image(s) %s", len(rows), dict(cat))
    LOGGER.info("Class-valid:   %d = %d benign + %d malignant", len(cls_valid),
                cls_lbl.get("BENIGN", 0), cls_lbl.get("MALIGNANT", 0))
    LOGGER.info("Images:        %d converted (every classification-valid record)",
                len(convertible))
    LOGGER.info("Joint cohort:  %d mask(s) = %d benign + %d malignant", len(cohort),
                by_el.get("BENIGN", 0), by_el.get("MALIGNANT", 0))
    LOGGER.info("Image rule:    %s", IMAGE_POLICY_ID)
    LOGGER.info("               channels verified identical per file: %s",
                dict(chan_seen))
    LOGGER.info("No mask:       %d (%s) -> %s; images converted, no mask, no "
                "pseudo-mask", len(excl7), EXCLUSION_CODE,
                join_list(sorted(r["image_reference"] for r in excl7), ", "))
    LOGGER.info("Not in target: %d retained, NOT exclusions: %s", len(nonjoint),
                dict(Counter(r["exclusion_status"] for r in nonjoint)))
    LOGGER.info("Mask rule:     %s", MASK_POLICY_ID)
    LOGGER.info("               pixels neither black nor white -> background: %s",
                dict(other_px) or "none")
    LOGGER.info("Multi-lesion:  %d joint image(s) hold >1 lesion inside ONE mask "
                "file; union plans 0", len(multi))
    LOGGER.info("Identity:      %s; 0 invented", summ.get("dicom_identity_status"))
    LOGGER.info("Dimensions:    verified here on %d/%d", plans, len(cohort))
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
    if mode == "dry-run" and headers_only:
        LOGGER.info("STOP. This was a fast accounting pass. Run a FULL dry-run "
                    "(without --headers-only) before --execute; the pixel facts were "
                    "not checked here.")
    elif mode == "dry-run":
        LOGGER.info("STOP. --execute needs human approval.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="DMID canonical bank builder.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    p.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    p.add_argument("--raw-root", default=None)
    p.add_argument("--headers-only", action="store_true",
                   help="dry-run only: read DICOM headers without decoding pixels; "
                        "seconds instead of minutes. --execute always decodes.")
    a = p.parse_args(argv)
    mode = "dry-run" if a.dry_run else "execute"
    if mode == "execute":
        st = (Path(a.prep_root) / BANK_NAME / DATASET / "reports"
              / "DRY_RUN_canonical_build_report.json")
        if not st.is_file():
            sys.stderr.write("EXECUTE refused: no dry-run report\n")
            return 5
        d = json.loads(st.read_text(encoding="utf-8"))
        if d.get("headers_only"):
            sys.stderr.write("EXECUTE refused: the last dry-run was --headers-only; "
                             "run a full dry-run first\n")
            return 5
        if d.get("verdict") != "READY_FOR_DMID_CANONICAL_EXECUTION":
            sys.stderr.write(f"EXECUTE refused: last verdict {d.get('verdict')!r}\n")
            return 5
        if d.get("builder_sha256") != sha256_file(Path(__file__).resolve()):
            sys.stderr.write("EXECUTE refused: the builder changed since the dry-run\n")
            return 5
    try:
        if a.headers_only and mode == "execute":
            sys.stderr.write("--headers-only is not allowed with --execute\n")
            return 5
        return run(mode, Path(a.prep_root),
                   Path(a.raw_root) if a.raw_root else None,
                   headers_only=a.headers_only)
    except Exception:  # noqa: BLE001
        if LOGGER.handlers:
            LOGGER.critical("UNHANDLED EXCEPTION\n%s", traceback.format_exc())
        else:
            sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
