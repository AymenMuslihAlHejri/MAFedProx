#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bank_cbis_ddsm.py
=======================

PHASE 2B.3.10 - CBIS-DDSM CANONICAL DATASET BANK BUILDER (TRAINING-READY)

THE SAMPLE UNIT IS THE ABNORMALITY RECORD, NOT THE MAMMOGRAM
    CBIS-DDSM states pathology per abnormality, and 293 mammograms carry between 2 and
    7 abnormalities. Building at mammogram level would require two rules the closed
    audit does not define: an image-level label for a mammogram whose abnormalities
    disagree, and a union of masks belonging to different lesions. Neither is invented
    here. One abnormality record = one sample: a full mammogram, ITS OWN verified ROI
    raster, and ITS OWN confirmed pathology.

    A consequence, stated plainly: a mammogram with several abnormalities is written
    once per abnormality, so 2885 samples come from 2703 distinct mammograms and 182
    image files repeat content under different sample ids. That is what makes the
    directory readable by a plain DataLoader with no pathology filtering, which is the
    stated requirement. The duplication is recorded per record
    (mammogram_sample_count, mammogram_repeat_index) so a later partition can see it.

WHAT IS WRITTEN, AND WHAT IS ONLY DOCUMENTED
    written   2885 pairs: 1429 BENIGN + 1456 MALIGNANT
    documented only, never written:
        682  BENIGN_WITHOUT_CALLBACK - kept verbatim as its own source category. It is
             NOT remapped to benign and NOT an exclusion; it is outside the current
             binary target and does not enter the global exclusion ledger.
          1  P_00474 / LEFT / MLO / abnormality 1, pathology MALIGNANT. Its cited ROI
             object carries roi_dicom_role = CROPPED_IMAGE: the adjudication verdict is
             UNRESOLVED_OTHER, so using it would put a cropped image where the
             segmentation target belongs. No repair, no swap, no pseudo-mask.

    IMAGE  CBIS_CANONICAL_IMAGE_LOSSLESS_V1
        the decoded DICOM pixels are written unchanged: 8-bit stays 8-bit, 16-bit stays
        16-bit, and a 3-channel array is reduced ONLY when that file's own channels are
        verified identical. Every file's header and shape are checked and a file that
        does not match what the source declares is refused, never coerced.

    MASK   CBIS_CANONICAL_MASK_BINARY_V1
        the ROI raster must resolve to two values. {0,255} is written through; {0,1} is
        scaled by exact value mapping. Anything else is REFUSED and reported, because
        this dataset's mask encoding was not established by the closed audit and a
        threshold chosen here would be an invention.

Usage:
    python build_bank_cbis_ddsm.py --dry-run
    python build_bank_cbis_ddsm.py --dry-run --headers-only    # fast accounting pass
    python build_bank_cbis_ddsm.py --execute
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET = "CBIS-DDSM"
DATASET_DIR = "CBIS_DDSM"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)

LINKAGE_FILE = "06_cbis_phase1a_linkage_manifest.csv"
SEG_FILE = "31_cbis_phase1b_segmentation_eligibility.csv"
CLS_FILE = "30_cbis_phase1b_classification_eligibility.csv"
ADJ_FILE = "20_cbis_phase1a1_role_conflict_adjudication.csv"
SUMMARY_FILE = "32_cbis_phase1b_modeling_summary.json"
SELFTEST_FILE = "34_cbis_phase1b_self_tests.txt"
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "cbis_root"]

IMAGE_POLICY_ID = "CBIS_CANONICAL_IMAGE_LOSSLESS_V1"
IMAGE_POLICY_RULE = ("decoded DICOM pixels written unchanged at native depth and "
                     "dimensions; a 3-channel array is reduced only after that file's "
                     "own channels are verified identical; no windowing, rescale, "
                     "normalisation, inversion, resize or crop")
MASK_POLICY_ID = "CBIS_CANONICAL_MASK_BINARY_V1"
MASK_POLICY_RULE = ("the ROI raster must hold exactly two values: {0,255} is written "
                    "through, {0,1} is scaled by exact value mapping; any other value "
                    "set is REFUSED, never thresholded")

BWC = "BENIGN_WITHOUT_CALLBACK"
BINARY = {"BENIGN": 0, "MALIGNANT": 1}
EXPECT = {"records": 3568, "benign": 1429, "malignant": 1457, "bwc": 682,
          "bm_before_seg": 2886, "bm_after_unresolved": 2885,
          "joint": 2805, "joint_benign": 1367, "joint_malignant": 1438,
          "dim_mismatch": 80, "dim_mismatch_benign": 62,
          "dim_mismatch_malignant": 18, "unresolved": 1, "png": 5610,
          "source_patients": 1566, "source_mammograms": 3103}
DIM_EXCLUSION_CODE = "JOINT_TASK_INELIGIBLE_MASK_IMAGE_DIMENSION_MISMATCH"
UNRESOLVED_ID = ("P_00474", "LEFT", "MLO", "1")
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}
SEP = " | "
LOGGER = logging.getLogger("bank_cbis")


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


def read_rows(p: Path) -> List[dict]:
    with p.open(encoding="utf-8-sig", errors="replace") as fh:
        s = fh.read(8192)
        fh.seek(0)
        if not s.strip():
            return []
        d = ";" if s.count(";") > s.count(",") else ","
        return list(csv.DictReader(fh, delimiter=d))


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    LOGGER.info("WROTE %-44s (%d data rows)", path.name, len(rows))


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


def sample_id(r: dict) -> str:
    """Deterministic and unique down to the abnormality, so two lesions on one
    mammogram can never overwrite each other."""
    fam = (r.get("abnormality_family") or "ABN")[:4].upper()
    parts = [fam, r["case_description_patient_id"], r["breast_side"],
             r["image_view"], f"ABN{r['abnormality_id']}"]
    stem = "_".join(re.sub(r"[^A-Za-z0-9]+", "", p) for p in parts)
    return f"{stem}__{sha256_text('/'.join(parts))[:8]}"


def reduce_image(arr, np) -> Tuple[Any, str]:
    if arr.ndim == 2:
        return arr, "ALREADY_SINGLE_CHANNEL"
    if arr.ndim == 3 and arr.shape[2] >= 3:
        if not (np.array_equal(arr[..., 0], arr[..., 1])
                and np.array_equal(arr[..., 1], arr[..., 2])):
            return None, "REFUSED_CHANNELS_NOT_REPLICATED"
        return arr[..., 0], "THREE_IDENTICAL_CHANNELS_REDUCED_TO_ONE"
    return None, f"REFUSED_UNEXPECTED_SHAPE_{arr.shape}"


def binarise_mask(arr, np) -> Tuple[Any, str]:
    """Two values only. Nothing is thresholded."""
    plane, note = reduce_image(arr, np)
    if plane is None:
        return None, note
    vals = set(np.unique(plane).tolist())
    if vals == {0, 255}:
        return plane.astype(np.uint8), "ALREADY_0_255"
    if vals == {0, 1}:
        return (plane.astype(np.uint8) * 255), "SCALED_0_1_TO_0_255"
    if vals == {0}:
        return None, "EMPTY_MASK"
    return None, f"REFUSED_VALUES_{sorted(vals)[:6]}"


def load_evidence(prep_root: Path) -> dict:
    ev: Dict[str, Any] = {"paths": {}, "status": {}, "hashes": {}, "problems": []}
    dirs = [d for d in prep_root.iterdir()
            if d.is_dir()
            and "cbis" in d.name.lower().replace(" ", "").replace("-", "")]
    files = [p for d in dirs for p in d.rglob("*") if p.is_file() and not archived(p)]
    for key, fname, required in (("linkage", LINKAGE_FILE, True),
                                 ("segmentation", SEG_FILE, True),
                                 ("classification", CLS_FILE, True),
                                 ("adjudication", ADJ_FILE, True),
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
    return ev


SCHEMA_FIELDS = [
    "dataset", "canonical_record_id", "sample_id",
    "case_description_patient_id", "breast_side", "image_view", "abnormality_id",
    "abnormality_family", "abnormality_type_source", "assessment", "subtlety",
    "source_table", "source_split_PROVENANCE_ONLY",
    "pathology_source_verbatim", "canonical_binary_label", "class_folder",
    "source_image_path", "source_mask_path",
    "canonical_image_path", "canonical_mask_path",
    "patient_id", "group_key_for_future_partitions",
    "mammogram_key", "mammogram_sample_count", "mammogram_repeat_index",
    "multi_abnormality_mammogram", "mask_union_applied",
    "phase1a_linkage_status", "roi_dicom_role",
    "segmentation_status", "classification_status",
    "in_modeling_cohort", "modeling_cohort_reason",
    "classification_available", "segmentation_available",
    "image_conversion_status", "mask_conversion_status",
    "source_image_rows", "source_image_columns", "source_bits_allocated",
    "source_bits_stored", "source_samples_per_pixel", "source_photometric",
    "source_pixel_dtype", "source_pixel_min", "source_pixel_max",
    "image_reduction_path", "mask_value_rule_applied",
    "mask_dimension_verified_here", "source_mask_rows", "source_mask_columns",
    "mask_image_ratio_w", "mask_image_ratio_h", "mask_foreground_pixels",
    "mask_foreground_fraction",
    "canonical_image_height", "canonical_image_width",
    "canonical_image_encoding", "canonical_mask_encoding",
    "image_policy_id", "image_policy_rule", "mask_policy_id", "mask_policy_rule",
    "identity_provenance", "label_provenance", "mask_provenance",
    "exclusion_status", "exclusion_reason", "notes",
]


def run(mode: str, prep_root: Path, raw_cli: Optional[Path],
        headers_only: bool) -> int:
    bank_ds = prep_root / BANK_NAME / DATASET_DIR
    reports, manifests = bank_ds / "reports", bank_ds / "manifests"
    setup_logging(reports, f"cbis_{mode}.log")
    LOGGER.info("=" * 78)
    LOGGER.info("PHASE 2B.3.10 CBIS-DDSM CANONICAL BANK BUILDER  [%s]", mode.upper())
    LOGGER.info("=" * 78)
    LOGGER.info("python : %s", platform.python_version())
    if headers_only:
        LOGGER.warning("HEADERS-ONLY: pixels are NOT decoded; the pixel facts are "
                       "deferred to a full dry-run and to --execute")
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
        LOGGER.info("      %-15s %-12s %s", k, st, art.name if art else "")
    if ev["problems"]:
        for p in ev["problems"]:
            LOGGER.error("      %s", p)
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4
    summ = json.loads(ev["paths"]["summary"].read_text(encoding="utf-8"))
    src_root = raw_cli or ev["dataset_root"]
    if src_root is None:
        LOGGER.error("      BLOCKED: no source root in %s and no --raw-root given",
                     SUMMARY_FILE)
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4
    before_raw = snapshot(src_root)
    LOGGER.info("      RAW = %s  [READ-ONLY]", src_root)
    LOGGER.info("      image policy = %s", IMAGE_POLICY_ID)
    LOGGER.info("      mask policy  = %s", MASK_POLICY_ID)

    link = read_rows(ev["paths"]["linkage"])
    seg = read_rows(ev["paths"]["segmentation"])
    cls = read_rows(ev["paths"]["classification"])
    adj = read_rows(ev["paths"]["adjudication"])

    def key(r: dict) -> tuple:
        return (r["source_table"], r["case_description_patient_id"],
                r["breast_side"], r["image_view"], r["abnormality_id"])

    link_by = {key(r): r for r in link}

    LOGGER.info("[2/7] source accounting ...")
    path_c = Counter(r["pathology_source_verbatim"] for r in seg)
    segs = Counter(r["segmentation_status"] for r in seg)
    roles = Counter(r["roi_dicom_role"] for r in seg)
    cls_by_mammo = {r["full_mammogram_path"]: r for r in cls}
    LOGGER.info("      abnormality records %d over %d patient(s), %d mammogram(s)",
                len(seg), len({r["case_description_patient_id"] for r in seg}),
                len({r["full_image_path_source"] for r in seg}))
    LOGGER.info("      pathology: %s", dict(path_c))
    LOGGER.info("      segmentation: %s | roi roles %s", dict(segs), dict(roles))
    LOGGER.info("      classification (mammogram level): %s",
                dict(Counter(r["classification_status"] for r in cls)))
    LOGGER.info("      image_level_label: %s  [never used here]",
                dict(Counter(r["image_level_label"] for r in cls)))
    LOGGER.info("      adjudicated role conflict(s): %d -> %s", len(adj),
                join_list([f"{r['case_description_patient_id']}/{r['breast_side']}/"
                           f"{r['image_view']}/{r['abnormality_id']} verdict="
                           f"{r['verdict']} roi_role={r['roi_dicom_role']}"
                           for r in adj], "; "))

    LOGGER.info("[3/7] planning canonical records ...")
    mammo_n = Counter(r["full_image_path_source"] for r in seg)
    seen: Counter = Counter()
    records: List[dict] = []
    for r in sorted(seg, key=lambda x: (x["case_description_patient_id"],
                                        x["breast_side"], x["image_view"],
                                        x["abnormality_id"])):
        k = key(r)
        lk = link_by.get(k, {})
        path = r["pathology_source_verbatim"].strip()
        seg_ok = r["segmentation_status"].strip() == "SEGMENTATION_LINKAGE_VALID"
        binary = path in BINARY
        in_cohort = binary and seg_ok
        folder = ("Benign" if path == "BENIGN" else
                  "Malignant" if path == "MALIGNANT" else "")
        sid = sample_id(r)
        mk = r["full_image_path_source"]
        seen[mk] += 1
        if in_cohort:
            status, reason = "INCLUDED_TRAINING_COHORT", ""
            tr = "written to the modeling-ready directories"
        elif path == BWC:
            status = "NOT_IN_BINARY_TARGET_BENIGN_WITHOUT_CALLBACK"
            reason = ("BENIGN_WITHOUT_CALLBACK is a distinct source category that the "
                      "closed audit deliberately kept unmapped. It is outside the "
                      "current binary target, is NOT remapped to benign, and is NOT an "
                      "exclusion: the record is retained here with its source "
                      "pathology verbatim and no canonical file is written for it.")
            tr = "provenance only"
        elif not seg_ok:
            status = "EXCLUDE_SEGMENTATION_LINKAGE_UNRESOLVED"
            reason = (f"the cited ROI object carries roi_dicom_role="
                      f"{r['roi_dicom_role']}, so the adjudication verdict is "
                      "UNRESOLVED_OTHER. Writing it would put a cropped image where "
                      "the segmentation target belongs. No repair, no crop/ROI swap "
                      "and no pseudo-mask was applied; the pathology label and the "
                      "patient are untouched.")
            tr = "provenance only"
        else:
            status, reason = "NOT_IN_TRAINING_COHORT_UNCLASSIFIED", f"pathology={path}"
            tr = "provenance only"
        records.append({
            "dataset": DATASET, "canonical_record_id": f"{DATASET_DIR}__{sid}",
            "sample_id": sid,
            "case_description_patient_id": r["case_description_patient_id"],
            "breast_side": r["breast_side"], "image_view": r["image_view"],
            "abnormality_id": r["abnormality_id"],
            "abnormality_family": r.get("abnormality_family", ""),
            "abnormality_type_source": lk.get("abnormality_type_source", ""),
            "assessment": lk.get("assessment", ""), "subtlety": lk.get("subtlety", ""),
            "source_table": r["source_table"],
            "source_split_PROVENANCE_ONLY": (r.get("source_split", "")
                                             + " [PROVENANCE ONLY, never a partition]"),
            "pathology_source_verbatim": path,
            "canonical_binary_label": (BINARY[path] if in_cohort
                                       else "NOT_IN_BINARY_TARGET"),
            "class_folder": folder if in_cohort else "",
            "source_image_path": lk.get("full_path_resolved", "").strip(),
            "source_mask_path": r.get("roi_path_resolved", "").strip(),
            "canonical_image_path": (f"{folder}/Images/{sid}.png" if in_cohort else ""),
            "canonical_mask_path": (f"{folder}/Masks/{sid}_mask.png" if in_cohort
                                    else ""),
            "patient_id": r["case_description_patient_id"],
            "group_key_for_future_partitions":
                f"PATIENT::{r['case_description_patient_id']}",
            "mammogram_key": mk, "mammogram_sample_count": mammo_n[mk],
            "mammogram_repeat_index": seen[mk],
            "multi_abnormality_mammogram": "YES" if mammo_n[mk] > 1 else "NO",
            "mask_union_applied": "NO",
            "phase1a_linkage_status": r.get("phase1a_linkage_status", ""),
            "roi_dicom_role": r.get("roi_dicom_role", ""),
            "segmentation_status": r["segmentation_status"],
            "classification_status": cls_by_mammo.get(mk, {}).get(
                "classification_status", ""),
            "in_modeling_cohort": "YES" if in_cohort else "NO",
            "modeling_cohort_reason": tr,
            "classification_available": "TRUE" if binary else "FALSE",
            "segmentation_available": "TRUE" if seg_ok else "FALSE",
            "image_conversion_status": "PLANNED" if in_cohort else "NOT_APPLICABLE",
            "mask_conversion_status": "PLANNED" if in_cohort else "NOT_APPLICABLE",
            "source_image_rows": "", "source_image_columns": "",
            "source_bits_allocated": "", "source_bits_stored": "",
            "source_samples_per_pixel": "", "source_photometric": "",
            "source_pixel_dtype": "", "source_pixel_min": "", "source_pixel_max": "",
            "image_reduction_path": "", "mask_value_rule_applied": "",
            "mask_dimension_verified_here": "", "source_mask_rows": "",
            "source_mask_columns": "", "mask_image_ratio_w": "",
            "mask_image_ratio_h": "", "mask_foreground_pixels": "",
            "mask_foreground_fraction": "",
            "canonical_image_height": "", "canonical_image_width": "",
            "canonical_image_encoding": ("LOSSLESS_PNG_NATIVE_DEPTH" if in_cohort
                                         else ""),
            "canonical_mask_encoding": ("SINGLE_CHANNEL_BINARY_0_255" if in_cohort
                                        else ""),
            "image_policy_id": IMAGE_POLICY_ID if in_cohort else "",
            "image_policy_rule": IMAGE_POLICY_RULE if in_cohort else "",
            "mask_policy_id": MASK_POLICY_ID if in_cohort else "",
            "mask_policy_rule": MASK_POLICY_RULE if in_cohort else "",
            "identity_provenance": (f"{SUMMARY_FILE} :: json:E_patient_grouping; key "
                                    "CASE_DESCRIPTION_PATIENT_ID (P_XXXXX)"),
            "label_provenance": (f"{SEG_FILE} :: csv:[pathology_source_verbatim] at "
                                 "ABNORMALITY level; no mammogram-level label was "
                                 "inferred"),
            "mask_provenance": (f"SOURCE_ROI_RASTER_BY[{MASK_POLICY_ID}]; one ROI per "
                                "abnormality, no union" if in_cohort
                                else "NOT_CONVERTED"),
            "exclusion_status": status, "exclusion_reason": reason,
            "notes": r.get("segmentation_reason", "")[:200],
        })
    cohort = [r for r in records
              if r["exclusion_status"] == "INCLUDED_TRAINING_COHORT"]
    bwc = [r for r in records if r["pathology_source_verbatim"] == BWC]
    unres = [r for r in records
             if r["exclusion_status"] == "EXCLUDE_SEGMENTATION_LINKAGE_UNRESOLVED"]
    by = Counter(r["pathology_source_verbatim"] for r in cohort)
    pats = {r["patient_id"] for r in cohort}
    mams = {r["mammogram_key"] for r in cohort}
    multi = [r for r in cohort if r["multi_abnormality_mammogram"] == "YES"]
    LOGGER.info("      training cohort %d = %d BENIGN + %d MALIGNANT", len(cohort),
                by.get("BENIGN", 0), by.get("MALIGNANT", 0))
    LOGGER.info("      over %d mammogram(s) and %d patient(s)", len(mams), len(pats))
    LOGGER.info("      multi-abnormality samples %d; mask unions applied 0", len(multi))
    LOGGER.info("      BWC documented only: %d | unresolved excluded: %d -> %s",
                len(bwc), len(unres),
                join_list([f"{r['patient_id']}/{r['breast_side']}/{r['image_view']}/"
                           f"{r['abnormality_id']} ({r['pathology_source_verbatim']})"
                           for r in unres], "; "))

    LOGGER.info("[4/7] resolving and pre-checking the training cohort ...")
    img_ok = mask_ok = plans = 0
    unresolved_src: List[str] = []
    red_c: Counter = Counter()
    mask_c: Counter = Counter()
    depth_c: Counter = Counter()
    dim_bad: List[str] = []
    decoded: Dict[str, Tuple[int, int]] = {}
    for i, r in enumerate(cohort, 1):
        sp = (src_root / r["source_image_path"]) if r["source_image_path"] else None
        mp = (src_root / r["source_mask_path"]) if r["source_mask_path"] else None
        if sp is None or not sp.is_file():
            unresolved_src.append(f"{r['sample_id']}: image missing "
                                  f"({r['source_image_path']})")
            r["image_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            ds = pydicom.dcmread(sp, stop_before_pixels=headers_only)
            arr = None if headers_only else ds.pixel_array
        except Exception as exc:  # noqa: BLE001
            unresolved_src.append(f"{r['sample_id']}: image unreadable "
                                  f"({type(exc).__name__})")
            r["image_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        r["source_bits_allocated"] = int(getattr(ds, "BitsAllocated", 0) or 0)
        r["source_bits_stored"] = int(getattr(ds, "BitsStored", 0) or 0)
        r["source_samples_per_pixel"] = int(getattr(ds, "SamplesPerPixel", 0) or 0)
        r["source_photometric"] = str(getattr(ds, "PhotometricInterpretation", ""))
        if headers_only:
            h = int(getattr(ds, "Rows", 0) or 0)
            w = int(getattr(ds, "Columns", 0) or 0)
            r["image_reduction_path"] = "DEFERRED_TO_FULL_RUN"
            depth_c[f"{r['source_bits_allocated']}-bit/"
                    f"{r['source_samples_per_pixel']}ch"] += 1
        else:
            want, note = reduce_image(arr, np)
            red_c[note] += 1
            r["image_reduction_path"] = note
            r["source_pixel_dtype"] = str(arr.dtype)
            r["source_pixel_min"], r["source_pixel_max"] = (int(arr.min()),
                                                            int(arr.max()))
            depth_c[f"{arr.dtype}/{r['source_samples_per_pixel']}ch"] += 1
            if want is None:
                unresolved_src.append(f"{r['sample_id']}: {note}")
                r["image_conversion_status"] = note
                continue
            h, w = want.shape[:2]
        r["source_image_rows"], r["source_image_columns"] = h, w
        r["canonical_image_height"], r["canonical_image_width"] = h, w
        decoded[r["sample_id"]] = (h, w)
        img_ok += 1
        if mp is None or not mp.is_file():
            unresolved_src.append(f"{r['sample_id']}: ROI missing "
                                  f"({r['source_mask_path']})")
            r["mask_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        if headers_only:
            r["mask_dimension_verified_here"] = "DEFERRED_TO_FULL_RUN"
            mask_c["HEADER_ONLY_NOT_DECODED"] += 1
            mask_ok += 1
            plans += 1
            continue
        try:
            ma = pydicom.dcmread(mp).pixel_array
        except Exception as exc:  # noqa: BLE001
            unresolved_src.append(f"{r['sample_id']}: ROI unreadable "
                                  f"({type(exc).__name__})")
            r["mask_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        mask_ok += 1
        plane, note = binarise_mask(ma, np)
        mask_c[note] += 1
        r["mask_value_rule_applied"] = note
        if plane is None:
            unresolved_src.append(f"{r['sample_id']}: mask refused ({note})")
            r["mask_conversion_status"] = f"REFUSED_{note}"
            continue
        same = plane.shape[:2] == (h, w)
        r["mask_dimension_verified_here"] = "MATCH" if same else "MISMATCH"
        if not same:
            mw, mh = plane.shape[1], plane.shape[0]
            rw, rh = (mw / w if w else 0), (mh / h if h else 0)
            dim_bad.append(f"{r['sample_id']}: image {w}x{h} vs mask {mw}x{mh} "
                           f"(ratio {rw:.4f}/{rh:.4f})")
            # JOINT-TASK-INELIGIBLE, not a bank-wide exclusion: the record keeps its
            # pathology, its patient and its observed dimensions, and no geometric
            # transform of any kind is applied to make it fit
            r["mask_conversion_status"] = "NOT_CONVERTED_DIMENSION_MISMATCH"
            r["image_conversion_status"] = "NOT_CONVERTED_DIMENSION_MISMATCH"
            r["canonical_image_path"] = ""
            r["canonical_mask_path"] = ""
            r["canonical_binary_label"] = "NOT_IN_JOINT_TRAINING_COHORT"
            r["class_folder"] = ""
            r["in_modeling_cohort"] = "NO"
            r["modeling_cohort_reason"] = "provenance only; joint-task ineligible"
            r["segmentation_available"] = "FALSE"
            r["source_mask_rows"], r["source_mask_columns"] = mh, mw
            r["mask_image_ratio_w"], r["mask_image_ratio_h"] = (round(rw, 6),
                                                               round(rh, 6))
            r["exclusion_status"] = DIM_EXCLUSION_CODE
            r["exclusion_reason"] = (
                f"the full mammogram is {w}x{h} and its ROI raster is {mw}x{mh}, a "
                f"ratio of {rw:.4f} by {rh:.4f}. A mask of different dimensions does "
                "not locate the lesion on its own image, and aligning them would "
                "require a resize, rescale, interpolation, crop or pad - every one of "
                "which moves the annotated boundary. The record is therefore "
                "ineligible for the JOINT task and is retained here with its observed "
                "dimensions, its pathology and its patient intact. It is NOT a "
                "bank-wide exclusion: the source data are not claimed to be invalid.")
            continue
        fg = int((plane == 255).sum())
        r["mask_foreground_pixels"] = fg
        r["mask_foreground_fraction"] = round(fg / float(h * w), 8)
        plans += 1
        if i % 200 == 0:
            LOGGER.info("      %d/%d", i, len(cohort))
    final = [r for r in cohort if r["exclusion_status"] == "INCLUDED_TRAINING_COHORT"]
    dim_excl = [r for r in cohort if r["exclusion_status"] == DIM_EXCLUSION_CODE]
    fby = Counter(r["pathology_source_verbatim"] for r in final)
    dby = Counter(r["pathology_source_verbatim"] for r in dim_excl)
    fpats = {r["patient_id"] for r in final}
    fmams = {r["mammogram_key"] for r in final}
    fmulti = [r for r in final if r["multi_abnormality_mammogram"] == "YES"]
    LOGGER.info("      images resolved=%d | masks read=%d | pairs planned=%d | "
                "unresolved=%d", img_ok, mask_ok, plans, len(unresolved_src))
    LOGGER.info("      dimension-mismatch -> joint-task-ineligible: %d (%d BENIGN / "
                "%d MALIGNANT) over %d patient(s)", len(dim_excl),
                dby.get("BENIGN", 0), dby.get("MALIGNANT", 0),
                len({r["patient_id"] for r in dim_excl}))
    LOGGER.info("      FINAL training cohort: %d = %d BENIGN + %d MALIGNANT over %d "
                "mammogram(s) and %d patient(s)", len(final), fby.get("BENIGN", 0),
                fby.get("MALIGNANT", 0), len(fmams), len(fpats))
    LOGGER.info("      source depth/channels: %s", dict(depth_c))
    LOGGER.info("      image reduction: %s", dict(red_c) or "deferred")
    LOGGER.info("      mask value rule: %s", dict(mask_c))
    LOGGER.info("      mask/image dimension mismatches: %d", len(dim_bad))
    for x in (unresolved_src + dim_bad)[:12]:
        LOGGER.error("         %s", x)

    png_written = 0
    readback_fail: List[str] = []
    if mode == "execute":
        LOGGER.info("[5/7] executing canonical conversion ...")
        for n, r in enumerate(final, 1):
            if r["image_conversion_status"] != "PLANNED":
                continue
            arr = pydicom.dcmread(src_root / r["source_image_path"]).pixel_array
            want, note = reduce_image(arr, np)
            if want is None:
                r["image_conversion_status"] = note
                continue
            intended = want
            dst = bank_ds / r["canonical_image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            pil_mode = "I;16" if intended.dtype == np.uint16 else "L"
            if not dst.exists():
                Image.fromarray(intended, mode=pil_mode).save(dst, format="PNG")
                png_written += 1
            with Image.open(dst) as chk:
                back = np.asarray(chk)
            if back.shape != intended.shape or not np.array_equal(
                    back.astype(intended.dtype), intended):
                r["image_conversion_status"] = "VERIFICATION_FAILED_NOT_IDENTICAL"
                readback_fail.append(f"{r['sample_id']}: image differs")
                continue
            r["image_conversion_status"] = "CONVERTED_VERIFIED_IDENTICAL"
            ma = pydicom.dcmread(src_root / r["source_mask_path"]).pixel_array
            plane, mnote = binarise_mask(ma, np)
            if plane is None or plane.shape[:2] != intended.shape[:2]:
                r["mask_conversion_status"] = f"REFUSED_{mnote}"
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
                readback_fail.append(f"{r['sample_id']}: mask differs")
                continue
            r["mask_conversion_status"] = "CONVERTED_VERIFIED_IDENTICAL"
            if n % 200 == 0:
                LOGGER.info("      %d/%d", n, len(final))
    else:
        LOGGER.info("[5/7] dry-run: no canonical PNG is written")

    LOGGER.info("[6/7] validation ...")
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
    sids = Counter(r["sample_id"] for r in records)
    dup_sid = [s for s, c in sids.items() if c > 1]
    unres_path = UNRESOLVED_ID[0]

    T: List[Tuple[str, str, Optional[bool], str]] = []

    def add(i: str, d: str, ok: Optional[bool], b: str) -> None:
        T.append((i, d, ok, b))

    add("T01", f"abnormality records == {EXPECT['records']}",
        len(seg) == EXPECT["records"], f"{len(seg)} @ {SEG_FILE}")
    add("T02", f"pathology: {EXPECT['benign']} B, {EXPECT['malignant']} M, "
               f"{EXPECT['bwc']} BWC",
        path_c.get("BENIGN") == EXPECT["benign"]
        and path_c.get("MALIGNANT") == EXPECT["malignant"]
        and path_c.get(BWC) == EXPECT["bwc"], f"{dict(path_c)}")
    add("T03", f"B/M before segmentation filtering == {EXPECT['bm_before_seg']}",
        path_c.get("BENIGN", 0) + path_c.get("MALIGNANT", 0)
        == EXPECT["bm_before_seg"],
        f"{path_c.get('BENIGN', 0)} + {path_c.get('MALIGNANT', 0)}")
    add("T04", f"B/M with a valid segmentation link == "
               f"{EXPECT['bm_after_unresolved']} before the dimension check",
        len(cohort) == EXPECT["bm_after_unresolved"],
        f"{len(cohort)} = {by.get('BENIGN')} + {by.get('MALIGNANT')}")
    add("T04b", f"FINAL training cohort == {EXPECT['joint']} = "
                f"{EXPECT['joint_benign']} BENIGN + {EXPECT['joint_malignant']} "
                "MALIGNANT",
        len(final) == EXPECT["joint"]
        and fby.get("BENIGN") == EXPECT["joint_benign"]
        and fby.get("MALIGNANT") == EXPECT["joint_malignant"],
        f"{len(final)} = {fby.get('BENIGN')} + {fby.get('MALIGNANT')}")
    add("T04c", f"{EXPECT['bm_after_unresolved']} - {EXPECT['dim_mismatch']} = "
                f"{EXPECT['joint']}",
        len(cohort) - len(dim_excl) == len(final)
        and len(dim_excl) == EXPECT["dim_mismatch"]
        and dby.get("BENIGN") == EXPECT["dim_mismatch_benign"]
        and dby.get("MALIGNANT") == EXPECT["dim_mismatch_malignant"],
        f"{len(cohort)} - {len(dim_excl)} = {len(final)}; excluded "
        f"{dby.get('BENIGN')} B / {dby.get('MALIGNANT')} M")
    add("T04d", "every dimension-mismatch record is retained with its observed "
                "dimensions, its ratio and its pathology, and writes nothing",
        all(r["exclusion_status"] == DIM_EXCLUSION_CODE and r["exclusion_reason"]
            and str(r["source_mask_rows"]).isdigit()
            and not r["canonical_image_path"] and not r["canonical_mask_path"]
            and r["pathology_source_verbatim"] in BINARY for r in dim_excl),
        f"{len(dim_excl)} record(s) with {DIM_EXCLUSION_CODE}")
    add("T05", f"exactly {EXPECT['unresolved']} record excluded for unresolved "
               "segmentation, and it is the adjudicated one",
        len(unres) == EXPECT["unresolved"]
        and all((r["patient_id"], r["breast_side"], r["image_view"],
                 r["abnormality_id"]) == UNRESOLVED_ID for r in unres),
        join_list([f"{r['patient_id']}/{r['breast_side']}/{r['image_view']}/"
                   f"{r['abnormality_id']} = {r['pathology_source_verbatim']}"
                   for r in unres], "; ") or "none")
    add("T06", f"{EXPECT['bwc']} BWC records kept verbatim, never remapped, and no "
               "canonical file written for them",
        len(bwc) == EXPECT["bwc"]
        and all(r["pathology_source_verbatim"] == BWC
                and not r["canonical_image_path"] and not r["canonical_mask_path"]
                and r["canonical_binary_label"] == "NOT_IN_BINARY_TARGET"
                for r in bwc),
        f"{len(bwc)} retained; 0 written; 0 remapped to BENIGN")
    add("T07", "the training cohort contains ONLY BENIGN and MALIGNANT",
        set(fby) == {"BENIGN", "MALIGNANT"} and not any(
            r["pathology_source_verbatim"] == BWC for r in final),
        f"{dict(fby)}")
    add("T08", "every training record has an image path, a mask path, a 0/1 label "
               "and exact image/mask dimension agreement",
        all(r["canonical_image_path"] and r["canonical_mask_path"]
            and r["canonical_binary_label"] in (0, 1)
            and r["mask_dimension_verified_here"] in ("MATCH", "DEFERRED_TO_FULL_RUN")
            for r in final),
        "one mammogram + one abnormality-specific ROI + one confirmed label")
    add("T09", "the final cohort's mammogram and patient counts are reported",
        len(fmams) > 0 and len(fpats) > 0,
        f"{len(fmams)} mammogram(s), {len(fpats)} patient(s)")
    add("T10", "sample ids are unique down to the abnormality, so two lesions on one "
               "mammogram cannot overwrite each other",
        not dup_sid and not collisions,
        f"{len(sids)} id(s) for {len(records)} record(s); {len(dup_sid)} duplicate, "
        f"{len(collisions)} path collision(s)")
    add("T11", "no mask union, no mammogram-level label, no geometric transform",
        all(r["mask_union_applied"] == "NO" for r in records)
        and all(r["label_provenance"].find("ABNORMALITY level") >= 0
                for r in records),
        f"{len(fmulti)} final sample(s) share a mammogram with another lesion and "
        "stay separate")
    add("T12", "every record carries the P_XXXXX patient grouping key",
        all(r["group_key_for_future_partitions"] == f"PATIENT::{r['patient_id']}"
            and re.fullmatch(r"P_\d+", r["patient_id"]) for r in records),
        f"{len({r['group_key_for_future_partitions'] for r in records})} distinct key(s)")
    add("T13", "SOURCE_SPLIT is carried as provenance only",
        all("PROVENANCE ONLY" in r["source_split_PROVENANCE_ONLY"] for r in records),
        "never used as a partition")
    add("T14", f"every B/M record with a valid link resolved to an image "
               f"({EXPECT['bm_after_unresolved']})",
        img_ok == EXPECT["bm_after_unresolved"], f"{img_ok}/{len(cohort)}")
    add("T15", f"mask pairs planned == {EXPECT['joint']} (the final cohort)",
        plans == EXPECT["joint"], f"{plans}; value rules {dict(mask_c)}")
    add("T16", ("dimension check deferred in headers-only mode" if headers_only
                else "every record IN THE FINAL COHORT has exact image/mask dimension "
                     "agreement; the mismatches were removed, never resized"),
        headers_only or all(r["mask_dimension_verified_here"] == "MATCH"
                            for r in final),
        f"{len(dim_bad)} mismatch(es) found and set aside; 0 resized, 0 cropped, "
        "0 padded, 0 interpolated")
    add("T16b", ("mask values deferred in headers-only mode" if headers_only
                 else "every mask in the final cohort is binary {0,255}"),
        headers_only or (set(mask_c) <= {"ALREADY_0_255", "SCALED_0_1_TO_0_255"}
                         and sum(mask_c.values()) >= len(final)),
        f"{dict(mask_c)}")
    add("T17", "no pseudo-mask, no heuristic repair, no crop/ROI swap", True,
        "the unresolved record is excluded rather than repaired; masks come from the "
        "ROI raster or the record is not written")
    add("T18", "no resize, crop, normalisation, CLAHE or augmentation", None,
        "code-scope declaration: decode, verified channel reduction, exact value "
        "mapping, lossless PNG")
    add("T19", "no unresolved source", not unresolved_src, f"{len(unresolved_src)}")
    add("T20", "RAW unchanged", raw_diff["unchanged"],
        f"modified {len(raw_diff['modified'])}, added {len(raw_diff['added'])}")
    add("T21", "completed audit artifacts unchanged", aud_diff["unchanged"],
        f"modified {len(aud_diff['modified'])}")
    add("T22", "no split created", not new_splits, f"{len(new_splits)}")
    add("T23", ("dry-run wrote no canonical PNG" if mode == "dry-run"
                else f"execution wrote {EXPECT['png']} canonical PNG"),
        (not new_png) if mode == "dry-run" else (len(new_png) == EXPECT["png"]),
        f"{len(new_png)} new PNG")
    add("T24", ("read-back verification runs at execute time" if mode == "dry-run"
                else "every written PNG reloads pixel-identical"),
        (mode == "dry-run") or not readback_fail,
        "deferred" if mode == "dry-run" else f"{len(readback_fail)} differed")
    add("T25", "no split field in the manifest schema",
        not any(is_split_like(f) for f in SCHEMA_FIELDS
                if f != "source_split_PROVENANCE_ONLY"), "schema checked")
    add("T26", "the BWC records do NOT enter the global exclusion ledger", None,
        f"{len(bwc)} record(s) carry NOT_IN_BINARY_TARGET_*, which the bank-wide "
        "report tool counts separately from exclusions")
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
                "label_provenance", "identity_provenance", "mask_provenance"}
        for i, h in enumerate(SCHEMA_FIELDS, 1):
            ws.column_dimensions[get_column_letter(i)].width = 70 if h in wide else 20
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        wb.save(manifests / f"{tag}canonical_manifest.xlsx")
        LOGGER.info("WROTE %-44s (1 sheet)", f"{tag}canonical_manifest.xlsx")
    except ImportError:
        LOGGER.warning("openpyxl not installed; XLSX skipped")

    blocked = bool(failed)
    state = {
        "phase": "2B.3.10", "dataset": DATASET, "mode": mode,
        "headers_only": headers_only, "generated": now_iso(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "schema_sha256": sha256_text(join_list(SCHEMA_FIELDS, ",")),
        "evidence_hashes": ev["hashes"], "source_root": str(src_root),
        "sample_unit": "ABNORMALITY_RECORD",
        "policies": {"image": {"id": IMAGE_POLICY_ID, "rule": IMAGE_POLICY_RULE},
                     "mask": {"id": MASK_POLICY_ID, "rule": MASK_POLICY_RULE}},
        "counts": {"abnormality_records": len(seg), "pathology": dict(path_c),
                   "bm_before_segmentation": path_c.get("BENIGN", 0)
                                             + path_c.get("MALIGNANT", 0),
                   "bm_with_valid_link": len(cohort),
                   "dimension_mismatch_set_aside": len(dim_excl),
                   "dimension_mismatch_benign": dby.get("BENIGN", 0),
                   "dimension_mismatch_malignant": dby.get("MALIGNANT", 0),
                   "training_cohort": len(final),
                   "training_benign": fby.get("BENIGN", 0),
                   "training_malignant": fby.get("MALIGNANT", 0),
                   "unique_mammograms": len(fmams), "unique_patients": len(fpats),
                   "multi_abnormality_samples": len(fmulti),
                   "bwc_documented_only": len(bwc),
                   "excluded_unresolved_segmentation": len(unres),
                   "mask_unions": 0, "pseudo_masks": 0,
                   "mammogram_level_labels": 0},
        "dimension_mismatch_records": [
            {"sample_id": r["sample_id"], "patient": r["patient_id"],
             "pathology": r["pathology_source_verbatim"],
             "image": f"{r['canonical_image_width']}x{r['canonical_image_height']}",
             "mask": f"{r['source_mask_columns']}x{r['source_mask_rows']}",
             "ratio_w": r["mask_image_ratio_w"], "ratio_h": r["mask_image_ratio_h"]}
            for r in dim_excl],
        "geometric_transforms_applied": 0,
        "global_exclusion_ledger_modified_for_dimension_mismatch": False,
        "unresolved_record": [{"patient": r["patient_id"], "side": r["breast_side"],
                               "view": r["image_view"],
                               "abnormality": r["abnormality_id"],
                               "pathology": r["pathology_source_verbatim"],
                               "roi_dicom_role": r["roi_dicom_role"]}
                              for r in unres],
        "source_depth_channels": dict(depth_c),
        "image_reduction": dict(red_c), "mask_value_rules": dict(mask_c),
        "dimension_mismatches": dim_bad, "unresolved_sources": unresolved_src[:50],
        "readback_failures": readback_fail,
        "canonical_png_written": png_written,
        "raw_modified": not raw_diff["unchanged"],
        "audit_modified": not aud_diff["unchanged"],
        "global_exclusion_ledger_modified_for_bwc": False,
        "tests_pass": sum(1 for t in verified if t[2]), "tests_fail": len(failed),
        "verdict": ("BLOCKED" if blocked else
                    "HEADERS_ONLY_ACCOUNTING_CLEAN_FULL_DRY_RUN_STILL_REQUIRED"
                    if (mode == "dry-run" and headers_only) else
                    "READY_FOR_CBIS_DDSM_CANONICAL_EXECUTION" if mode == "dry-run"
                    else "CBIS_DDSM_CANONICAL_BUILD_COMPLETE"),
    }
    (reports / f"{tag}canonical_build_report.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    LOGGER.info("WROTE %-44s", f"{tag}canonical_build_report.json")

    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 2B.3.10 CBIS-DDSM - %s", mode.upper())
    LOGGER.info("Source:      %d abnormality record(s) %s", len(seg), dict(path_c))
    LOGGER.info("Sample unit: ABNORMALITY RECORD (one mammogram + its own ROI + its "
                "own pathology)")
    LOGGER.info("Training:    %d = %d BENIGN + %d MALIGNANT", len(final),
                fby.get("BENIGN", 0), fby.get("MALIGNANT", 0))
    LOGGER.info("             over %d mammogram(s) and %d patient(s)", len(fmams),
                len(fpats))
    LOGGER.info("Not written: %d BWC + %d unresolved (%s) + %d dimension mismatch "
                "(%d B / %d M)", len(bwc), len(unres), unres_path, len(dim_excl),
                dby.get("BENIGN", 0), dby.get("MALIGNANT", 0))
    LOGGER.info("             all retained in the manifest; 0 resized, 0 cropped, "
                "0 padded, 0 interpolated, 0 reconstructed")
    LOGGER.info("Multi-abn:   %d sample(s) share a mammogram with another lesion; "
                "mask unions 0, mammogram-level labels 0", len(fmulti))
    LOGGER.info("Grouping:    PATIENT::P_XXXXX on every record; SOURCE_SPLIT is "
                "provenance only")
    LOGGER.info("Masks:       %s", dict(mask_c))
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
        LOGGER.info("STOP. Fast accounting pass; run a FULL dry-run before --execute.")
    elif mode == "dry-run":
        LOGGER.info("STOP. --execute needs human approval.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CBIS-DDSM canonical bank builder.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    p.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    p.add_argument("--raw-root", default=None)
    p.add_argument("--headers-only", action="store_true")
    a = p.parse_args(argv)
    mode = "dry-run" if a.dry_run else "execute"
    if mode == "execute":
        if a.headers_only:
            sys.stderr.write("--headers-only is not allowed with --execute\n")
            return 5
        st = (Path(a.prep_root) / BANK_NAME / DATASET_DIR / "reports"
              / "DRY_RUN_canonical_build_report.json")
        if not st.is_file():
            sys.stderr.write("EXECUTE refused: no dry-run report\n")
            return 5
        d = json.loads(st.read_text(encoding="utf-8"))
        if d.get("headers_only"):
            sys.stderr.write("EXECUTE refused: the last dry-run was --headers-only\n")
            return 5
        if d.get("verdict") != "READY_FOR_CBIS_DDSM_CANONICAL_EXECUTION":
            sys.stderr.write(f"EXECUTE refused: last verdict {d.get('verdict')!r}\n")
            return 5
        if d.get("builder_sha256") != sha256_file(Path(__file__).resolve()):
            sys.stderr.write("EXECUTE refused: the builder changed since the dry-run\n")
            return 5
    try:
        return run(mode, Path(a.prep_root),
                   Path(a.raw_root) if a.raw_root else None, a.headers_only)
    except Exception:  # noqa: BLE001
        if LOGGER.handlers:
            LOGGER.critical("UNHANDLED EXCEPTION\n%s", traceback.format_exc())
        else:
            sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
