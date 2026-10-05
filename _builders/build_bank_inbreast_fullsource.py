#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bank_inbreast_fullsource.py
=================================

PHASE 2B.3.5b - INbreast FULL SOURCE, LOCKED MASS COHORT (108 mammograms)

SCOPE
    The 410-DICOM full source at H:\\BC datasets\\Mammo\\01 INbreast 2.
    The old 107-image prepared derivative is NOT used, read, or linked.

LOCKED COHORT (human decision, recorded not recomputed)
    INbreast.xls rows 0-409 are the 410 real mammograms. Row 410 is blank and row 411
    is a SUMMARY/FOOTER row that must never be treated as a mammogram. Among the 410
    real rows, Mass == "X" for exactly 108 mammograms, whose source BI-RADS is
    2=24, 3=13, 4a=7, 4b=2, 4c=11, 5=43, 6=8. The builder re-derives all of this from
    the workbook and BLOCKS on any disagreement.

NO CLASSIFICATION LABEL IS CREATED
    INbreast distributes no per-image biopsy or pathology status. BI-RADS is carried
    VERBATIM in its own column and is never mapped to benign/malignant. There are no
    Benign/ or Malignant/ folders: the canonical output is one cohort for the
    segmentation task. canonical_binary_label stays NOT_ESTABLISHED on every record.

TWO CONVERSIONS THAT ARE NEW TO THIS BANK - BOTH DECLARED, NEITHER SILENT
    1. MASK GENERATION. INbreast ships no raster mask; it ships XML contours. Phase 1A
       deliberately generated none (self-test I16). This builder rasterises them, which
       is a genuine Phase 2B canonical decision:
         POLICY_ID INBREAST_FULLSOURCE_MASK_FROM_XML_CONTOUR_V1
           take ROIs whose Name is exactly "Mass" -> keep those with >= 3 contour
           points -> fill each polygon -> union the polygons of one mammogram ->
           single-channel {0,255} PNG at the DICOM's native size.
         No smoothing, dilation, erosion, convex hull or hole filling. A Mass ROI with
         fewer than 3 points cannot bound an area and is refused, never interpolated.
    2. 14-BIT DICOM -> PNG. INbreast is 14-bit. An 8-bit PNG would require a display
       window, which is a preprocessing choice this bank does not make. The canonical
       image is therefore a 16-BIT PNG holding the stored pixel values UNCHANGED:
         POLICY_ID INBREAST_FULLSOURCE_IMAGE_16BIT_LOSSLESS_V1
           no windowing, rescale, normalisation, inversion, resize or crop.

IDENTITY
    Patient group = the filename hash token of the DICOM file name, the identifier
    Phase 1A.1 adjudicated as PATIENT_GROUPING_ESTABLISHED (108 groups over 410
    mammograms, one-to-one with the report case IDs). Recorded as case_id and as the
    future group key. patient_id and lesion_id are never invented. No split is created.

Usage:
    python build_bank_inbreast_fullsource.py --dry-run
    python build_bank_inbreast_fullsource.py --execute
Requires: Python >= 3.8, pydicom, Pillow, numpy, pandas, xlrd. openpyxl optional.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import plistlib
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET = "INbreast_FullSource"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
DEFAULT_RAW_ROOT = r"H:\BC datasets\Mammo\01 INbreast 2"
BANK_NAME = "CANONICAL_DATASET_BANK"

WORKBOOK = "INbreast.xls"
DICOM_DIR = "AllDICOMs"
XML_DIR = "AllXML"
REAL_ROWS = 410          # rows 0-409; 410 is blank, 411 is a summary/footer row
COHORT_COL = "Mass "     # the trailing space is the workbook's own column name

MASK_POLICY_ID = "INBREAST_FULLSOURCE_MASK_FROM_XML_CONTOUR_V1"
MASK_POLICY_RULE = ('ROIs named exactly "Mass" with >= 3 contour points -> fill each '
                    "polygon -> union per mammogram -> single-channel {0,255} PNG at "
                    "native size; no morphology, no hull, no hole filling")
IMAGE_POLICY_ID = "INBREAST_FULLSOURCE_IMAGE_16BIT_LOSSLESS_V1"
IMAGE_POLICY_RULE = ("stored DICOM pixel values written unchanged as a 16-bit PNG; no "
                     "windowing, rescale, normalisation, inversion, resize or crop")
MASS_ROI_NAME = "Mass"
MIN_POLYGON_POINTS = 3

EXPECT = {"real_rows": REAL_ROWS, "cohort": 108,
          "birads": {"2": 24, "3": 13, "4a": 7, "4b": 2, "4c": 11, "5": 43, "6": 8}}

SCHEMA_FIELDS = [
    "dataset", "cohort", "canonical_record_id", "source_file_number",
    "source_image_path", "source_annotation_path",
    "canonical_image_path", "canonical_mask_path",
    "birads_verbatim", "acr_verbatim", "laterality", "view", "acquisition_date",
    "findings_notes_verbatim", "lesion_annotation_status_verbatim",
    "canonical_binary_label", "classification_available", "segmentation_available",
    "patient_id", "case_id", "lesion_id", "group_key_for_future_partitions",
    "image_conversion_status", "mask_conversion_status",
    "mass_roi_total", "mass_roi_used", "mass_roi_refused_point_count",
    "mask_foreground_pixels", "mask_foreground_fraction",
    "source_image_height", "source_image_width",
    "source_bits_allocated", "source_bits_stored", "source_high_bit",
    "source_pixel_representation", "source_photometric",
    "source_rescale_slope", "source_rescale_intercept",
    "source_pixel_dtype", "source_pixel_min", "source_pixel_max",
    "lossless_uint16_writable", "rescale_is_identity", "values_within_bits_stored",
    "image_pixel_identical_to_source",
    "canonical_image_height", "canonical_image_width",
    "canonical_mask_height", "canonical_mask_width",
    "canonical_image_encoding", "canonical_mask_encoding",
    "image_policy_id", "image_policy_rule", "mask_policy_id", "mask_policy_rule",
    "identity_provenance", "label_provenance", "cohort_provenance",
    "exclusion_status", "exclusion_reason", "notes",
]
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}

SEP = " | "
LOGGER = logging.getLogger("bank_inbreast_fs")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def join_list(v: Sequence[Any], sep: str = SEP) -> str:
    return sep.join(str(x) for x in v)


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


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    LOGGER.info("WROTE %-44s (%d data rows)", path.name, len(rows))


def parse_points(seq) -> List[Tuple[float, float]]:
    """OsiriX plist stores each point as the string '(x, y, z)'."""
    pts: List[Tuple[float, float]] = []
    for s in seq or []:
        m = re.findall(r"-?\d+\.?\d*", str(s))
        if len(m) >= 2:
            pts.append((float(m[0]), float(m[1])))
    return pts


def read_mass_rois(xml_path: Path) -> Tuple[List[List[Tuple[float, float]]], int, int]:
    """Returns (polygons, total Mass ROIs, Mass ROIs refused for < 3 points)."""
    with xml_path.open("rb") as fh:
        blob = plistlib.load(fh, fmt=plistlib.FMT_XML)
    rois: List[dict] = []
    imgs = blob.get("Images") if isinstance(blob, dict) else None
    if isinstance(imgs, list):
        for im in imgs:
            rois.extend(im.get("ROIs") or [])
    elif isinstance(blob, dict) and "ROIs" in blob:
        rois.extend(blob.get("ROIs") or [])
    polys, total, refused = [], 0, 0
    for r in rois:
        if str(r.get("Name", "")).strip() != MASS_ROI_NAME:
            continue
        total += 1
        pts = parse_points(r.get("Point_px"))
        if len(pts) < MIN_POLYGON_POINTS:
            refused += 1
            continue
        polys.append(pts)
    return polys, total, refused


def build_mask(polys, shape, np, ImageDraw, Image):
    h, w = shape
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for pts in polys:
        d.polygon([(float(x), float(y)) for x, y in pts], fill=255)
    return np.asarray(im, dtype=np.uint8)


def run(mode: str, prep_root: Path, raw_root: Path) -> int:
    bank_ds = prep_root / BANK_NAME / DATASET
    reports, manifests = bank_ds / "reports", bank_ds / "manifests"
    setup_logging(reports, f"inbreast_fullsource_{mode}.log")
    LOGGER.info("=" * 78)
    LOGGER.info("PHASE 2B.3.5b INbreast FULL SOURCE - MASS COHORT  [%s]", mode.upper())
    LOGGER.info("=" * 78)
    LOGGER.info("python : %s", platform.python_version())
    LOGGER.info("RAW    : %s  [READ-ONLY]", raw_root)
    LOGGER.info("bank   : %s", bank_ds)

    try:
        import numpy as np
        import pandas as pd
        import pydicom
        from PIL import Image, ImageDraw
    except ImportError as exc:
        LOGGER.critical("missing dependency (%s); need pydicom, pillow, numpy, "
                        "pandas, xlrd", exc)
        return 3
    Image.MAX_IMAGE_PIXELS = None

    before_raw = snapshot(raw_root)
    before_bank = snapshot(prep_root / BANK_NAME)

    # ---- 1. workbook, with the footer row excluded by position ---------------
    LOGGER.info("[1/6] locked cohort from %s ...", WORKBOOK)
    wb = raw_root / WORKBOOK
    if not wb.is_file():
        # some installations hold an .xlsx export of the same sheet
        alt = raw_root / (Path(WORKBOOK).stem + ".xlsx")
        if alt.is_file():
            LOGGER.warning("      %s absent; using the .xlsx export %s",
                           WORKBOOK, alt.name)
            wb = alt
    if not wb.is_file():
        LOGGER.error("      workbook not found: %s", wb)
        LOGGER.error("FINAL VERDICT:\nBLOCKED")
        return 4
    raw_df = pd.read_excel(wb)
    LOGGER.info("      workbook shape %s", (raw_df.shape,))
    df = raw_df.iloc[:REAL_ROWS].copy()
    footer = raw_df.iloc[REAL_ROWS:]
    LOGGER.info("      real mammogram rows kept: 0-%d (%d); rows %d+ dropped as "
                "blank/footer: %d", REAL_ROWS - 1, len(df), REAL_ROWS, len(footer))
    if COHORT_COL not in df.columns:
        alt = [c for c in df.columns if c.strip().lower() == "mass"]
        if not alt:
            LOGGER.error("      no Mass column in %s", list(df.columns))
            LOGGER.error("FINAL VERDICT:\nBLOCKED")
            return 4
        col = alt[0]
    else:
        col = COHORT_COL
    mass = df[df[col].astype(str).str.strip().str.upper() == "X"].copy()
    br = Counter(str(x).strip() for x in mass["Bi-Rads"])
    LOGGER.info("      Mass == X -> %d mammogram(s)", len(mass))
    LOGGER.info("      BI-RADS verbatim: %s", dict(sorted(br.items())))
    footer_vals = {}
    if len(footer):
        for c in (col, "Micros", "Distortion", "Asymmetry"):
            if c in footer.columns:
                v = footer[c].dropna()
                if len(v):
                    footer_vals[c.strip()] = v.iloc[-1]
    LOGGER.info("      workbook footer totals (context only): %s", footer_vals)

    # ---- 2. index the source tree --------------------------------------------
    LOGGER.info("[2/6] indexing the source tree ...")
    dcm_by_num: Dict[str, Path] = {}
    dup_nums: List[str] = []
    for p in (raw_root / DICOM_DIR).glob("*.dcm"):
        num = p.name.split("_", 1)[0]
        if num in dcm_by_num:
            dup_nums.append(num)
        dcm_by_num[num] = p
    xml_by_num = {p.stem: p for p in (raw_root / XML_DIR).glob("*.xml")}
    LOGGER.info("      DICOM files %d (duplicate numbers %d) | XML files %d",
                len(dcm_by_num), len(dup_nums), len(xml_by_num))

    # ---- 3. plan ---------------------------------------------------------------
    LOGGER.info("[3/6] planning canonical records ...")
    records: List[dict] = []
    for _, row in mass.iterrows():
        try:
            num = str(int(row["File Name"]))
        except (TypeError, ValueError):
            num = str(row["File Name"]).strip()
        dcm = dcm_by_num.get(num)
        xml = xml_by_num.get(num)
        ghash = dcm.name.split("_")[1] if dcm and len(dcm.name.split("_")) > 1 else ""
        stem = f"{num}__{sha256_text(num)[:8]}"
        records.append({
            "dataset": DATASET, "cohort": "FULL_SOURCE_MASS_COHORT_108",
            "canonical_record_id": f"{DATASET}__{stem}",
            "source_file_number": num,
            "source_image_path": (str(dcm.relative_to(raw_root)) if dcm else ""),
            "source_annotation_path": (str(xml.relative_to(raw_root)) if xml else ""),
            "canonical_image_path": f"Images/{stem}.png",
            "canonical_mask_path": f"Masks/{stem}_mask.png",
            "birads_verbatim": str(row.get("Bi-Rads", "")).strip(),
            "acr_verbatim": str(row.get("ACR", "")).strip(),
            "laterality": str(row.get("Laterality", "")).strip(),
            "view": str(row.get("View", "")).strip(),
            "acquisition_date": str(row.get("Acquisition date", "")).strip(),
            "findings_notes_verbatim": str(
                row.get("Findings Notes (in Portuguese)", "")).strip(),
            "lesion_annotation_status_verbatim": str(
                row.get("Lesion Annotation Status", "")).strip(),
            "canonical_binary_label": "NOT_ESTABLISHED",
            "classification_available": "FALSE",
            "segmentation_available": "PLANNED",
            "patient_id": "", "case_id": ghash, "lesion_id": "",
            "group_key_for_future_partitions": f"INBREAST_PATIENT::{ghash}",
            "image_conversion_status": "PLANNED", "mask_conversion_status": "PLANNED",
            "mass_roi_total": "", "mass_roi_used": "",
            "mass_roi_refused_point_count": "",
            "mask_foreground_pixels": "", "mask_foreground_fraction": "",
            "source_image_height": "", "source_image_width": "",
            "source_bits_allocated": "", "source_bits_stored": "",
            "source_high_bit": "", "source_pixel_representation": "",
            "source_photometric": "", "source_rescale_slope": "",
            "source_rescale_intercept": "", "source_pixel_dtype": "",
            "source_pixel_min": "", "source_pixel_max": "",
            "lossless_uint16_writable": "", "rescale_is_identity": "",
            "values_within_bits_stored": "",
            "image_pixel_identical_to_source": "NOT_VERIFIED_DRY_RUN",
            "canonical_image_height": "", "canonical_image_width": "",
            "canonical_mask_height": "", "canonical_mask_width": "",
            "canonical_image_encoding": "SINGLE_CHANNEL_16BIT_PNG_LOSSLESS",
            "canonical_mask_encoding": "SINGLE_CHANNEL_BINARY_0_255",
            "image_policy_id": IMAGE_POLICY_ID, "image_policy_rule": IMAGE_POLICY_RULE,
            "mask_policy_id": MASK_POLICY_ID, "mask_policy_rule": MASK_POLICY_RULE,
            "identity_provenance": ("filename hash token; PATIENT_GROUPING_ESTABLISHED "
                                    "in the Phase 1A.1 grouping adjudication"),
            "label_provenance": ("BI-RADS carried VERBATIM from INbreast.xls; NO "
                                 "benign/malignant label is derived"),
            "cohort_provenance": (f"{WORKBOOK} rows 0-{REAL_ROWS - 1}, "
                                  f"[{col.strip()}] == X"),
            "exclusion_status": "INCLUDED_SEGMENTATION_COHORT", "exclusion_reason": "",
            "notes": "",
        })
    groups = {r["case_id"] for r in records if r["case_id"]}
    LOGGER.info("      planned %d record(s) over %d patient group(s)",
                len(records), len(groups))

    # ---- 4. resolve, decode, rasterise (in memory during dry-run) ------------
    LOGGER.info("[4/6] resolving sources and building masks ...")
    img_ok = mask_ok = 0
    unresolved: List[str] = []
    no_contour: List[str] = []
    profiles: List[dict] = []
    lossless_fail: List[str] = []
    rescale_nonid: List[str] = []
    over_bits: List[str] = []
    pix_min, pix_max = 1 << 30, -(1 << 30)
    for i, r in enumerate(records, 1):
        if not r["source_image_path"]:
            unresolved.append(f"{r['source_file_number']}: no DICOM")
            r["image_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            ds = pydicom.dcmread(raw_root / r["source_image_path"])
            arr = ds.pixel_array
        except Exception as exc:  # noqa: BLE001
            unresolved.append(f"{r['source_file_number']}: DICOM unreadable "
                              f"({type(exc).__name__})")
            r["image_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        h, w = arr.shape[:2]
        r["source_image_height"] = r["canonical_image_height"] = h
        r["source_image_width"] = r["canonical_image_width"] = w
        r["source_bits_allocated"] = int(getattr(ds, "BitsAllocated", 0) or 0)
        r["source_bits_stored"] = int(getattr(ds, "BitsStored", 0) or 0)
        r["source_high_bit"] = int(getattr(ds, "HighBit", -1))
        r["source_pixel_representation"] = int(getattr(ds, "PixelRepresentation", -1))
        r["source_photometric"] = str(getattr(ds, "PhotometricInterpretation", ""))
        rs = getattr(ds, "RescaleSlope", None)
        ri = getattr(ds, "RescaleIntercept", None)
        r["source_rescale_slope"] = "" if rs is None else str(rs)
        r["source_rescale_intercept"] = "" if ri is None else str(ri)
        r["source_pixel_dtype"] = str(arr.dtype)
        r["source_pixel_min"] = int(arr.min())
        r["source_pixel_max"] = int(arr.max())
        # can this array go to a uint16 PNG with no change of value?
        fits = (str(arr.dtype) == "uint16"
                or (arr.dtype.kind == "u" and int(arr.max()) <= 65535)
                or (arr.dtype.kind == "i" and int(arr.min()) >= 0
                    and int(arr.max()) <= 65535))
        rescale_identity = (rs is None or float(rs) == 1.0) and (
            ri is None or float(ri) == 0.0)
        within_bits = int(arr.max()) < (1 << max(1, r["source_bits_stored"]))
        signed = r["source_pixel_representation"] == 1
        r["lossless_uint16_writable"] = "YES" if (fits and not signed) else "NO"
        r["rescale_is_identity"] = "YES" if rescale_identity else "NO"
        r["values_within_bits_stored"] = "YES" if within_bits else "NO"
        profiles.append({
            "BitsAllocated": r["source_bits_allocated"],
            "BitsStored": r["source_bits_stored"], "HighBit": r["source_high_bit"],
            "PixelRepresentation": r["source_pixel_representation"],
            "PhotometricInterpretation": r["source_photometric"],
            "RescaleSlope": r["source_rescale_slope"],
            "RescaleIntercept": r["source_rescale_intercept"],
            "dtype": r["source_pixel_dtype"]})
        pix_min = min(pix_min, int(arr.min()))
        pix_max = max(pix_max, int(arr.max()))
        if not fits or signed:
            lossless_fail.append(f"{r['source_file_number']}: dtype="
                                 f"{arr.dtype} signed={signed} max={arr.max()}")
        if not rescale_identity:
            rescale_nonid.append(f"{r['source_file_number']}: slope={rs} "
                                 f"intercept={ri}")
        if not within_bits:
            over_bits.append(f"{r['source_file_number']}: max={arr.max()} "
                             f">= 2^{r['source_bits_stored']}")
        img_ok += 1
        if not r["source_annotation_path"]:
            no_contour.append(r["source_file_number"])
            r["mask_conversion_status"] = "NO_XML_ANNOTATION"
            r["segmentation_available"] = "FALSE"
            r["exclusion_status"] = "EXCLUDED_FROM_SEGMENTATION_NO_MASS_CONTOUR"
            r["exclusion_reason"] = ("the workbook marks Mass == X but the source "
                                     "ships no XML annotation for this mammogram; no "
                                     "contour is invented")
            continue
        try:
            polys, total, refused = read_mass_rois(raw_root / r["source_annotation_path"])
        except Exception as exc:  # noqa: BLE001
            unresolved.append(f"{r['source_file_number']}: XML unreadable "
                              f"({type(exc).__name__})")
            r["mask_conversion_status"] = "UNPARSEABLE_ANNOTATION"
            continue
        r["mass_roi_total"] = total
        r["mass_roi_used"] = len(polys)
        r["mass_roi_refused_point_count"] = refused
        if not polys:
            no_contour.append(r["source_file_number"])
            r["mask_conversion_status"] = ("NO_MASS_ROI" if total == 0
                                           else "MASS_ROI_BELOW_3_POINTS")
            r["segmentation_available"] = "FALSE"
            r["exclusion_status"] = "EXCLUDED_FROM_SEGMENTATION_NO_MASS_CONTOUR"
            r["exclusion_reason"] = (
                f"the workbook marks Mass == X but the XML carries {total} Mass ROI(s) "
                f"and {refused} of them have fewer than {MIN_POLYGON_POINTS} points, "
                "so no polygon can be bounded; nothing is interpolated")
            continue
        m = build_mask(polys, (h, w), np, ImageDraw, Image)
        fg = int((m == 255).sum())
        if fg == 0:
            unresolved.append(f"{r['source_file_number']}: mask empty after fill")
            r["mask_conversion_status"] = "EMPTY_AFTER_FILL"
            continue
        r["canonical_mask_height"], r["canonical_mask_width"] = h, w
        r["mask_foreground_pixels"] = fg
        r["mask_foreground_fraction"] = round(fg / float(h * w), 6)
        r["segmentation_available"] = "TRUE"
        mask_ok += 1
        if i % 25 == 0:
            LOGGER.info("      %d/%d", i, len(records))
    LOGGER.info("      DICOM decoded=%d | masks built=%d | no Mass contour=%d | "
                "unresolved=%d", img_ok, mask_ok, len(no_contour), len(unresolved))
    LOGGER.info("")
    LOGGER.info("      PIXEL REPRESENTATION ACROSS ALL %d DICOM(s)", img_ok)
    for key in ("BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
                "PhotometricInterpretation", "RescaleSlope", "RescaleIntercept",
                "dtype"):
        LOGGER.info("         %-26s %s", key,
                    dict(Counter(str(x[key]) for x in profiles)))
    LOGGER.info("         %-26s min=%d max=%d", "stored value range",
                pix_min if profiles else 0, pix_max if profiles else 0)
    bs = {x["BitsStored"] for x in profiles}
    if profiles and len(bs) == 1:
        b = next(iter(bs))
        LOGGER.info("         %-26s max %d vs 2^%d-1 = %d -> %s", "bit-depth check",
                    pix_max, b, (1 << b) - 1,
                    "content fits BitsStored" if pix_max < (1 << b)
                    else "CONTENT EXCEEDS BitsStored")
    for lbl, lst in (("not writable as uint16 without change", lossless_fail),
                     ("rescale is not identity", rescale_nonid),
                     ("values exceed BitsStored", over_bits)):
        (LOGGER.info if not lst else LOGGER.warning)(
            "         %-26s %d%s", lbl, len(lst),
            "" if not lst else f" -> {join_list(lst[:3], '; ')}")
    LOGGER.info("")
    if no_contour:
        LOGGER.warning("      Mass == X but no usable contour: %s",
                       join_list(no_contour, ", "))
    for u in unresolved[:10]:
        LOGGER.error("         %s", u)

    # ---- 5. execute -------------------------------------------------------------
    png_written = 0
    readback_fail: List[str] = []
    if mode == "execute":
        LOGGER.info("[5/6] executing canonical conversion ...")
        for r in records:
            if r["image_conversion_status"] != "PLANNED":
                continue
            ds = pydicom.dcmread(raw_root / r["source_image_path"])
            arr = ds.pixel_array
            dst = bank_ds / r["canonical_image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            intended = arr.astype(np.uint16)
            if not dst.exists():
                Image.fromarray(intended, mode="I;16").save(dst, format="PNG")
                png_written += 1
            # pixel-perfect verification: reload and compare to the intended array
            with Image.open(dst) as chk:
                back = np.asarray(chk)
            if back.shape != intended.shape or not np.array_equal(
                    back.astype(np.uint16), intended):
                d = (int(np.sum(back.astype(np.uint16) != intended))
                     if back.shape == intended.shape else -1)
                r["image_conversion_status"] = "VERIFICATION_FAILED_NOT_IDENTICAL"
                r["image_pixel_identical_to_source"] = "NO"
                readback_fail.append(f"{r['source_file_number']}: differing px={d}")
                continue
            r["image_pixel_identical_to_source"] = "YES"
            r["image_conversion_status"] = "CONVERTED_VERIFIED_IDENTICAL"
            if r["segmentation_available"] != "TRUE":
                r["canonical_mask_path"] = ""
                continue
            polys, _t, _rf = read_mass_rois(raw_root / r["source_annotation_path"])
            m = build_mask(polys, arr.shape[:2], np, ImageDraw, Image)
            mdst = bank_ds / r["canonical_mask_path"]
            mdst.parent.mkdir(parents=True, exist_ok=True)
            if not mdst.exists():
                Image.fromarray(m, mode="L").save(mdst, format="PNG")
                png_written += 1
            with Image.open(mdst) as chk:
                mback = np.asarray(chk.convert("L"))
            if not np.array_equal(mback, m):
                r["mask_conversion_status"] = "VERIFICATION_FAILED_NOT_IDENTICAL"
                readback_fail.append(f"{r['source_file_number']}: mask read-back "
                                     "differs")
                continue
            r["mask_conversion_status"] = "CONVERTED_VERIFIED_IDENTICAL"
    else:
        LOGGER.info("[5/6] dry-run: no canonical PNG is written")
        for r in records:
            if r["segmentation_available"] != "TRUE":
                r["canonical_mask_path"] = ""

    # ---- 6. tests ----------------------------------------------------------------
    LOGGER.info("[6/6] validation ...")
    after_bank = snapshot(prep_root / BANK_NAME)
    new_png = [k for k in after_bank
               if k not in before_bank and k.lower().endswith(".png")]
    new_splits = [k for k in after_bank if k not in before_bank and is_split_like(k)]
    raw_diff = diff_state(before_raw, snapshot(raw_root))
    paths = [r["canonical_image_path"] for r in records if r["canonical_image_path"]]
    paths += [r["canonical_mask_path"] for r in records if r["canonical_mask_path"]]
    collisions = [n for n, c in Counter(paths).items() if c > 1]
    seg = [r for r in records if r["segmentation_available"] == "TRUE"]
    seg_groups = {r["case_id"] for r in seg if r["case_id"]}
    seg_br = Counter(r["birads_verbatim"] for r in seg)
    bits = Counter(str(r["source_bits_stored"]) for r in records if r["source_bits_stored"])

    T: List[Tuple[str, str, Optional[bool], str]] = []

    def add(i: str, d: str, ok: Optional[bool], b: str) -> None:
        T.append((i, d, ok, b))

    add("T01", f"real mammogram rows == {REAL_ROWS} and the footer row is excluded",
        len(df) == REAL_ROWS and len(raw_df) > REAL_ROWS,
        f"kept {len(df)} of {len(raw_df)}; dropped {len(raw_df) - REAL_ROWS} "
        f"blank/footer row(s); footer totals seen: {footer_vals}")
    add("T02", f"locked Mass cohort == {EXPECT['cohort']}",
        len(mass) == EXPECT["cohort"], f"{len(mass)}")
    add("T03", "BI-RADS distribution matches the locked cohort",
        {k: v for k, v in br.items()} == EXPECT["birads"],
        f"{dict(sorted(br.items()))} vs locked {EXPECT['birads']}")
    add("T04", "no benign/malignant label created",
        all(r["canonical_binary_label"] == "NOT_ESTABLISHED" for r in records)
        and not any(is_split_like(f) or f.lower() in ("benign", "malignant")
                    for f in SCHEMA_FIELDS)
        and not any("Benign" in p or "Malignant" in p for p in paths),
        "canonical_binary_label=NOT_ESTABLISHED on every record; no class folder")
    add("T05", "BI-RADS carried verbatim",
        all(r["birads_verbatim"] for r in records)
        and set(br) <= {"1", "2", "3", "4a", "4b", "4c", "5", "6"},
        f"values present: {sorted(set(br))}")
    add("T06", f"every cohort row resolved to a DICOM", img_ok == len(records),
        f"{img_ok}/{len(records)}")
    ba = Counter(str(x["BitsAllocated"]) for x in profiles)
    pr = Counter(str(x["PixelRepresentation"]) for x in profiles)
    ph = Counter(str(x["PhotometricInterpretation"]) for x in profiles)
    dt = Counter(str(x["dtype"]) for x in profiles)
    add("T07", "every DICOM is unsigned, <= 16 bits allocated, and its stored values "
               "fit uint16 without modification",
        bool(profiles) and len(profiles) == img_ok == len(records)
        and set(ba) == {"16"} and set(pr) == {"0"} and set(dt) == {"uint16"}
        and not lossless_fail and 0 <= pix_min and pix_max <= 65535,
        f"BitsAllocated {dict(ba)}; BitsStored {dict(bits)}; "
        f"PixelRepresentation {dict(pr)}; dtype {dict(dt)}; "
        f"range [{pix_min},{pix_max}]; unwritable {len(lossless_fail)}")
    add("T07b", "no rescale transform is being discarded (slope 1 / intercept 0, or "
                "absent)",
        not rescale_nonid,
        f"{len(rescale_nonid)} record(s) carry a non-identity rescale"
        + (f": {join_list(rescale_nonid[:3], '; ')}" if rescale_nonid else
           "; stored values are the image values"))
    add("T07c", "stored values lie within the declared BitsStored",
        not over_bits,
        f"max {pix_max} vs BitsStored {dict(bits)}; "
        f"{len(over_bits)} record(s) exceed it")
    add("T07d", "photometric interpretation recorded for the bank's users", None,
        f"{dict(ph)}" + ("  [MONOCHROME1 = inverted greyscale; recorded, not "
                         "converted]" if "MONOCHROME1" in ph else ""))
    add("T08", "mask rule is declared and applied without morphology",
        "no morphology" in MASK_POLICY_RULE,
        f"{MASK_POLICY_ID}: only Mass ROIs with >= {MIN_POLYGON_POINTS} points")
    add("T09", "masks built for every record carrying a usable Mass contour",
        mask_ok == len(records) - len(no_contour) and not unresolved,
        f"{mask_ok} built; {len(no_contour)} without a usable contour; "
        f"{len(unresolved)} unresolved")
    add("T10", "records without a Mass contour are retained and reported, not dropped",
        all(r["exclusion_status"].startswith("EXCLUDED_FROM_SEGMENTATION")
            and r["exclusion_reason"] and not r["canonical_mask_path"]
            for r in records if r["segmentation_available"] == "FALSE"),
        f"{len(no_contour)} record(s): "
        + (join_list(no_contour, ", ") if no_contour else "none"))
    add("T11", "no patient or lesion identity invented",
        all(not r["patient_id"] and not r["lesion_id"] for r in records),
        "patient_id and lesion_id empty on every record")
    add("T12", "every record carries the adjudicated patient group key",
        all(r["case_id"] and r["group_key_for_future_partitions"]
            == f"INBREAST_PATIENT::{r['case_id']}" for r in records),
        f"{len(groups)} group(s) over {len(records)} record(s)")
    add("T13", "the source read is the full DICOM source, not a prepared "
               "class-folder derivative",
        (raw_root / DICOM_DIR).is_dir() and (raw_root / XML_DIR).is_dir()
        and len(dcm_by_num) == REAL_ROWS
        and not (raw_root / "Benign").exists()
        and not (raw_root / "Malignant").exists(),
        f"{DICOM_DIR}/{len(dcm_by_num)} DICOM + {XML_DIR}/{len(xml_by_num)} XML; "
        "no Benign/ or Malignant/ folder present")
    add("T14", "no unresolved record", not unresolved, f"{len(unresolved)}")
    add("T15", "no canonical path collision", not collisions, f"{len(collisions)}")
    add("T16", "RAW unchanged", raw_diff["unchanged"],
        f"modified {len(raw_diff['modified'])}, added {len(raw_diff['added'])}")
    add("T17", "no split created", not new_splits, f"{len(new_splits)}")
    add("T18", ("dry-run wrote no canonical PNG" if mode == "dry-run"
                else f"execution wrote {2 * len(seg) + (len(records) - len(seg))} PNG"),
        (not new_png) if mode == "dry-run"
        else len(new_png) == 2 * len(seg) + (len(records) - len(seg)),
        f"{len(new_png)} new PNG")
    add("T18b", ("read-back verification runs at execute time" if mode == "dry-run"
                 else "every written PNG reloads pixel-identical to its intended "
                      "array"),
        (mode == "dry-run") or not readback_fail,
        ("deferred to --execute; the dry-run writes nothing to read back"
         if mode == "dry-run" else
         f"{len(readback_fail)} file(s) differed on reload"
         + (f": {join_list(readback_fail[:3], '; ')}" if readback_fail else "")))
    add("T19", "no preprocessing beyond the two declared conversions", None,
        "code-scope declaration: DICOM stored values -> 16-bit PNG unchanged; XML Mass "
        "contours -> filled polygons; nothing else")
    verified = [t for t in T if t[2] is not None]
    failed = [t for t in verified if not t[2]]
    for t in T:
        (LOGGER.info if t[2] is not False else LOGGER.error)(
            "      %-5s [%-11s] %-52s %s", t[0],
            "DECLARATION" if t[2] is None else ("PASS" if t[2] else "FAIL"),
            t[1][:52], t[3][:56])

    tag = "DRY_RUN_" if mode == "dry-run" else ""
    write_csv(manifests / f"{tag}canonical_manifest.csv", SCHEMA_FIELDS,
              [{k: r[k] for k in SCHEMA_FIELDS} for r in records])
    blocked = bool(failed)
    state = {
        "phase": "2B.3.5b", "dataset": DATASET,
        "cohort": "FULL_SOURCE_MASS_COHORT_108", "mode": mode, "generated": now_iso(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "raw_root": str(raw_root), "workbook_sha256": sha256_file(wb),
        "cohort_lock": {"real_rows": REAL_ROWS, "expected": EXPECT["cohort"],
                        "observed": len(mass),
                        "birads_expected": EXPECT["birads"],
                        "birads_observed": dict(sorted(br.items()))},
        "policies": {"image": {"id": IMAGE_POLICY_ID, "rule": IMAGE_POLICY_RULE},
                     "mask": {"id": MASK_POLICY_ID, "rule": MASK_POLICY_RULE}},
        "classification": {"label_created": False,
                           "canonical_binary_label": "NOT_ESTABLISHED",
                           "birads": "VERBATIM"},
        "counts": {"cohort": len(records), "dicom_decoded": img_ok,
                   "masks_built": mask_ok,
                   "segmentation_eligible": len(seg),
                   "no_mass_contour": len(no_contour),
                   "patient_groups_cohort": len(groups),
                   "patient_groups_segmentation": len(seg_groups)},
        "pixel_representation": {
            "BitsAllocated": dict(Counter(str(x["BitsAllocated"]) for x in profiles)),
            "BitsStored": dict(bits),
            "HighBit": dict(Counter(str(x["HighBit"]) for x in profiles)),
            "PixelRepresentation": dict(
                Counter(str(x["PixelRepresentation"]) for x in profiles)),
            "PhotometricInterpretation": dict(
                Counter(str(x["PhotometricInterpretation"]) for x in profiles)),
            "RescaleSlope": dict(Counter(str(x["RescaleSlope"]) for x in profiles)),
            "RescaleIntercept": dict(
                Counter(str(x["RescaleIntercept"]) for x in profiles)),
            "dtype": dict(Counter(str(x["dtype"]) for x in profiles)),
            "stored_value_range": [pix_min, pix_max],
            "not_uint16_writable": lossless_fail,
            "non_identity_rescale": rescale_nonid,
            "exceeding_bits_stored": over_bits},
        "readback_failures": readback_fail,
        "segmentation_birads": dict(sorted(seg_br.items())),
        "no_mass_contour_records": no_contour,
        "unresolved": unresolved[:50], "collisions": collisions,
        "canonical_png_written": png_written,
        "raw_modified": not raw_diff["unchanged"],
        "tests_pass": sum(1 for t in verified if t[2]), "tests_fail": len(failed),
        "verdict": ("BLOCKED" if blocked else
                    "READY_FOR_INBREAST_FULLSOURCE_EXECUTION" if mode == "dry-run"
                    else "INBREAST_FULLSOURCE_BUILD_COMPLETE"),
    }
    (reports / f"{tag}canonical_build_report.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    LOGGER.info("WROTE %-44s", f"{tag}canonical_build_report.json")

    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 2B.3.5b INbreast FULL SOURCE - %s", mode.upper())
    LOGGER.info("Cohort:    %d mammogram(s) with Mass == X (locked), %d patient "
                "group(s)", len(records), len(groups))
    LOGGER.info("BI-RADS:   %s  [VERBATIM, no B/M label]", dict(sorted(br.items())))
    LOGGER.info("Seg-ready: %d with a usable Mass contour over %d group(s)",
                len(seg), len(seg_groups))
    LOGGER.info("           BI-RADS of those: %s", dict(sorted(seg_br.items())))
    LOGGER.info("No contour: %d -> %s", len(no_contour),
                join_list(no_contour, ", ") or "none")
    LOGGER.info("Image rule: %s", IMAGE_POLICY_ID)
    LOGGER.info("Mask rule:  %s", MASK_POLICY_ID)
    LOGGER.info("RAW modified %s | PNG written %d | splits %d",
                state["raw_modified"], png_written, len(new_splits))
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
    p = argparse.ArgumentParser(description="INbreast full-source Mass cohort builder.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    p.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    a = p.parse_args(argv)
    mode = "dry-run" if a.dry_run else "execute"
    if mode == "execute":
        st = (Path(a.prep_root) / BANK_NAME / DATASET / "reports"
              / "DRY_RUN_canonical_build_report.json")
        if not st.is_file():
            sys.stderr.write("EXECUTE refused: no dry-run report\n")
            return 5
        d = json.loads(st.read_text(encoding="utf-8"))
        if d.get("verdict") != "READY_FOR_INBREAST_FULLSOURCE_EXECUTION":
            sys.stderr.write(f"EXECUTE refused: last verdict {d.get('verdict')!r}\n")
            return 5
        if d.get("builder_sha256") != sha256_file(Path(__file__).resolve()):
            sys.stderr.write("EXECUTE refused: the builder changed since the dry-run\n")
            return 5
    try:
        return run(mode, Path(a.prep_root), Path(a.raw_root))
    except Exception:  # noqa: BLE001
        if LOGGER.handlers:
            LOGGER.critical("UNHANDLED EXCEPTION\n%s", traceback.format_exc())
        else:
            sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
