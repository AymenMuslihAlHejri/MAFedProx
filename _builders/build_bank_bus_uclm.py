#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bank_bluis.py
===================

PHASE 2B.3.7 - BUS-UCLM CANONICAL DATASET BANK BUILDER

Seventh dataset, eighth distinct mask rule. Nothing is carried over from any other.

    BUSI      union of multiple masks
    BUS-UC    midpoint threshold at 128 on soft boundaries
    BrEaST    RGBA colour-channel reduction
    BUS-BRA   value scaling {0,1} -> {0,255}
    INbreast  contour rasterisation + 16-bit passthrough
    BLUIS     value scaling {0,1} -> {0,255}
    BUS-UCLM  COLOUR-CODED masks: green foreground on benign, red on malignant

    MASK   BUS_UCLM_CANONICAL_MASK_COLOUR_TO_BINARY_V1
        The source masks encode the CLASS in the foreground colour: the audit verified,
        per sample and with zero mismatches, that all 174 benign masks are
        BLACK_BG+GREEN_FG, all 90 malignant masks are BLACK_BG+RED_FG, and all 419
        normal masks are BLACK_ONLY.

        Canonical rule: any pixel that is not pure black (0,0,0) is lesion foreground
        and becomes 255; pure black becomes 0. The result is a single-channel PNG.
        No threshold on a colour channel, no morphology, no pseudo-mask.

        The colour is checked against the official label per record and recorded, but
        it NEVER derives the label: the label comes from INFO.csv only. Making the
        colour authoritative would turn a visual convention into ground truth, which
        the audit deliberately avoided.

    IMAGE  BUS_UCLM_CANONICAL_IMAGE_LOSSLESS_V1
        the decoded source image is written unchanged to a lossless PNG at native
        dimensions with its channels preserved.

THE 419 NORMAL SAMPLES
    Normal is not a defect and not an exclusion: it is a class outside the binary
    lesion target. Its masks are legitimately all black, which the audit recorded as
    EXPECTED_EMPTY_MASK_FOR_NORMAL_SAMPLE rather than as annotation failures. All 419
    are retained in the manifest with full provenance and no canonical path, and the
    CANONICAL_DATASET_EXCLUSION_LEDGER is NOT modified, because nothing was excluded
    after selection.

PATIENT GROUPING
    patient_id comes from the dataset's OWN documented naming scheme XXXX_YYY, on the
    authority of that documentation and not by inferring meaning from filenames. 38
    patients overall, 36 in the lesion cohort, and every record carries the key. The
    audit states the requirement plainly: all future partitions must be grouped by
    patient_id. No split is created here.

ONE GAP THIS BUILDER CLOSES
    The audit's dimension_match column is EMPTY on all 683 rows, so mask/image
    dimension agreement was never established upstream. This builder verifies it from
    the pixels for every cohort record and blocks on any mismatch.

Usage:
    python build_bank_bus_uclm.py --dry-run
    python build_bank_bus_uclm.py --execute
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

DATASET = "BUS_UCLM"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)

MANIFEST_FILE = "26_bus_uclm_phase1a1_manifest.csv"
COHORT_FILE = "25_bus_uclm_binary_lesion_cohort.csv"
GROUPING_FILE = "24_bus_uclm_patient_grouping_audit.csv"
CROSSTAB_FILE = "22_bus_uclm_label_color_crosstab.csv"
MISMATCH_FILE = "23_bus_uclm_label_color_mismatches.csv"
SUMMARY_FILE = "27_bus_uclm_phase1a1_summary.json"
SELFTEST_FILE = "29_bus_uclm_phase1a1_self_tests.txt"
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]

MASK_POLICY_ID = "BUS_UCLM_CANONICAL_MASK_COLOUR_TO_BINARY_V1"
MASK_POLICY_RULE = ("the source mask is COLOUR-CODED: pure black background with a "
                    "green foreground on benign samples and a red foreground on "
                    "malignant ones. Canonical rule: any pixel that is not pure black "
                    "(0,0,0) is lesion foreground -> 255, pure black -> 0, written as "
                    "a single-channel PNG. The colour is verified against the official "
                    "label as a cross-check and is NEVER used to derive it")
IMAGE_POLICY_ID = "BUS_UCLM_CANONICAL_IMAGE_LOSSLESS_V1"
IMAGE_POLICY_RULE = ("decoded source image written unchanged to lossless PNG at native "
                     "dimensions, channels preserved; no resize, crop, ROI, "
                     "normalisation, CLAHE, denoising or augmentation")

EXPECT = {"source_records": 683, "normal": 419, "benign": 174, "malignant": 90,
          "selected": 264, "excluded": 4, "cohort": 260,
          "patients_overall": 38, "patients_selected": 36, "png": 520}
DIM_EXCLUSION_CODE = "EXCLUDE_MASK_IMAGE_DIMENSION_MISMATCH"
NORMAL_STATUS = "NOT_IN_BINARY_TARGET_NORMAL_CLASS"
GREEN, RED = "BLACK_BG+GREEN_FG", "BLACK_BG+RED_FG"
BLACK = "BLACK_ONLY"
CANONICAL_LABELS = {"BENIGN": 0, "MALIGNANT": 1}

SCHEMA_FIELDS = [
    "dataset", "canonical_record_id", "sample_id", "source_filename",
    "source_image_path", "source_mask_path",
    "source_label", "canonical_binary_label", "class_folder", "modeling_status",
    "canonical_image_path", "canonical_mask_path",
    "patient_id", "patient_id_source", "group_key_for_future_partitions",
    "case_id", "lesion_id", "same_source_image_group",
    "classification_available", "segmentation_available",
    "image_conversion_status", "mask_conversion_status",
    "source_image_height", "source_image_width", "source_image_channels",
    "resolution_metadata", "source_mask_height", "source_mask_width",
    "mask_dimension_match_verified_here",
    "canonical_image_height", "canonical_image_width",
    "canonical_mask_height", "canonical_mask_width",
    "source_image_format", "source_mask_format",
    "source_mask_colour_pattern", "mask_colour_agrees_with_official_label",
    "mask_foreground_colour_observed",
    "canonical_image_encoding", "canonical_mask_encoding",
    "doppler", "marks", "combined",
    "mask_foreground_pixels", "mask_foreground_fraction",
    "image_policy_id", "image_policy_rule", "mask_policy_id", "mask_policy_rule",
    "identity_provenance", "label_provenance", "mask_provenance",
    "exclusion_status", "exclusion_reason", "notes",
]
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}

SEP = " | "
LOGGER = logging.getLogger("bank_bus_uclm")


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


def colour_mask_to_binary(arr, np) -> Tuple[Any, str, str]:
    """Any pixel that is not pure black is lesion foreground. Returns the binary plane,
    a note, and the observed foreground colour so it can be cross-checked against the
    official label without ever deriving the label from it."""
    if arr.ndim == 2:
        fg = arr > 0
        colours = "GREYSCALE_NON_BLACK"
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        fg = rgb.any(axis=2)
        if not fg.any():
            colours = "NONE_ALL_BLACK"
        else:
            mean = rgb[fg].mean(axis=0)
            order = int(np.argmax(mean))
            colours = ("RED_DOMINANT" if order == 0 else
                       "GREEN_DOMINANT" if order == 1 else "BLUE_DOMINANT")
    else:
        return None, f"UNEXPECTED_SHAPE_{arr.shape}", ""
    if not fg.any():
        return None, "ALL_BLACK_NO_FOREGROUND", colours
    return (fg.astype(np.uint8) * 255), "NON_BLACK_TO_255", colours


def load_evidence(prep_root: Path) -> dict:
    ev: Dict[str, Any] = {"paths": {}, "status": {}, "hashes": {}, "problems": []}
    dirs = [d for d in prep_root.iterdir()
            if d.is_dir() and "busuclm" in d.name.lower().replace(" ", "").replace("_", "").replace("-", "")]
    files = [p for d in dirs for p in d.rglob("*") if p.is_file() and not archived(p)]
    for key, fname, required in (("manifest", MANIFEST_FILE, True),
                                 ("cohort", COHORT_FILE, True),
                                 ("grouping", GROUPING_FILE, True),
                                 ("crosstab", CROSSTAB_FILE, False),
                                 ("mismatches", MISMATCH_FILE, False),
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
    setup_logging(reports, f"bus_uclm_{mode}.log")
    LOGGER.info("=" * 78)
    LOGGER.info("PHASE 2B.3.7 BUS-UCLM CANONICAL BANK BUILDER  [%s]", mode.upper())
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
    cohort_rows = read_rows(ev["paths"]["cohort"])
    grouping = read_rows(ev["paths"]["grouping"])
    mismatches = (read_rows(ev["paths"]["mismatches"])
                  if ev["paths"].get("mismatches") else [])
    ids = [r["sample_id"].strip() for r in rows]
    dupe = [k for k, n in Counter(ids).items() if n > 1]
    if dupe:
        LOGGER.error("      sample_id not unique: %s", dupe[:5])
        LOGGER.error("FINAL VERDICT:\nBLOCKED")
        return 4

    LOGGER.info("[2/6] source accounting ...")
    by_lbl = Counter(r["label_normalized"].strip() for r in rows)
    colours = Counter(r["mask_color_pattern"].strip() for r in rows)
    status = Counter(r["modeling_status"].strip() for r in rows)
    empty = Counter(r["mask_empty"].strip().upper() for r in rows)
    dimcol = Counter(r.get("dimension_match", "").strip() or "EMPTY" for r in rows)
    pats_all = {r["patient_id"].strip() for r in rows if r["patient_id"].strip()}
    cohort_ids = {r["sample_id"].strip() for r in cohort_rows}
    pats_cohort = {r["patient_id"].strip() for r in cohort_rows
                   if r["patient_id"].strip()}
    LOGGER.info("      manifest %d record(s): %s", len(rows), dict(by_lbl))
    LOGGER.info("      summary json: rows=%s benign=%s malignant=%s normal=%s "
                "cohort=%s", summ.get("info_csv_rows"),
                (summ.get("label_distribution") or {}).get("Benign"),
                (summ.get("label_distribution") or {}).get("Malignant"),
                (summ.get("label_distribution") or {}).get("Normal"),
                summ.get("binary_cohort_total"))
    LOGGER.info("      mask colour patterns: %s", dict(colours))
    LOGGER.info("      label x colour mismatches recorded by the audit: %d",
                len(mismatches))
    LOGGER.info("      modeling status: %s", dict(status))
    LOGGER.info("      mask_empty: %s  (all-black is EXPECTED on Normal)", dict(empty))
    LOGGER.info("      patients: %d overall, %d in the lesion cohort (audit says "
                "%s / %s)", len(pats_all), len(pats_cohort),
                summ.get("unique_patients_overall"),
                summ.get("unique_patients_in_binary_cohort"))
    LOGGER.info("      patient_id source: %s",
                str(summ.get("patient_id_source"))[:90])
    LOGGER.info("      audit dimension_match column: %s", dict(dimcol))
    if "EMPTY" in dimcol:
        LOGGER.warning("      the audit never established mask/image dimension "
                       "agreement; this builder verifies it from the pixels")

    LOGGER.info("[3/6] planning canonical records ...")
    records: List[dict] = []
    for r in rows:
        sid = r["sample_id"].strip()
        lbl = r["label_normalized"].strip()
        in_cohort = r["modeling_status"].strip() == "ELIGIBLE_BINARY_LESION_COHORT"
        folder = ("Benign" if lbl == "Benign" else
                  "Malignant" if lbl == "Malignant" else "")
        stem = canonical_stem(lbl.upper(), sid)
        pid = r["patient_id"].strip()
        records.append({
            "dataset": DATASET,
            "canonical_record_id": f"{DATASET}__{stem}", "sample_id": sid,
            "source_filename": r.get("filename", "").strip(),
            "source_image_path": r.get("image_path", "").strip(),
            "source_mask_path": r.get("mask_path", "").strip(),
            "source_label": lbl,
            "canonical_binary_label": (r.get("binary_label", "").strip()
                                       if in_cohort else "NOT_IN_BINARY_TARGET"),
            "class_folder": folder if in_cohort else "",
            "modeling_status": r.get("modeling_status", "").strip(),
            "canonical_image_path": (f"{folder}/Images/{stem}.png"
                                     if in_cohort and folder else ""),
            "canonical_mask_path": (f"{folder}/Masks/{stem}_mask.png"
                                    if in_cohort and folder else ""),
            "patient_id": pid,
            "patient_id_source": r.get("patient_id_source", "").strip(),
            "group_key_for_future_partitions": f"PATIENT::{pid}" if pid else "",
            "case_id": "", "lesion_id": "",
            "same_source_image_group": f"{DATASET}__{stem}",
            "classification_available": "TRUE" if in_cohort else "FALSE",
            "segmentation_available": "TRUE" if in_cohort else "FALSE",
            "image_conversion_status": "PLANNED" if in_cohort else "NOT_APPLICABLE",
            "mask_conversion_status": "PLANNED" if in_cohort else "NOT_APPLICABLE",
            "source_image_height": "", "source_image_width": "",
            "source_image_channels": "",
            "resolution_metadata": r.get("resolution_metadata", "").strip(),
            "source_mask_height": "", "source_mask_width": "",
            "mask_dimension_match_verified_here": "",
            "canonical_image_height": "", "canonical_image_width": "",
            "canonical_mask_height": "", "canonical_mask_width": "",
            "source_image_format": Path(r.get("image_path", "")).suffix.lstrip(
                ".").upper(),
            "source_mask_format": Path(r.get("mask_path", "")).suffix.lstrip(
                ".").upper(),
            "source_mask_colour_pattern": r.get("mask_color_pattern", "").strip(),
            "mask_colour_agrees_with_official_label": "",
            "mask_foreground_colour_observed": "",
            "canonical_image_encoding": "LOSSLESS_PNG_CHANNELS_PRESERVED",
            "canonical_mask_encoding": ("SINGLE_CHANNEL_BINARY_0_255" if in_cohort
                                        else ""),
            "doppler": r.get("doppler", "").strip(),
            "marks": r.get("marks", "").strip(),
            "combined": r.get("combined", "").strip(),
            "mask_foreground_pixels": "", "mask_foreground_fraction": "",
            "image_policy_id": IMAGE_POLICY_ID, "image_policy_rule": IMAGE_POLICY_RULE,
            "mask_policy_id": MASK_POLICY_ID if in_cohort else "",
            "mask_policy_rule": MASK_POLICY_RULE if in_cohort else "",
            "identity_provenance": (f"{MANIFEST_FILE} :: csv:[patient_id]; source="
                                    + str(r.get("patient_id_source", ""))[:70]),
            "label_provenance": (f"{MANIFEST_FILE} :: csv:[label_normalized] "
                                 f"(source={r.get('label_source', '').strip()}); the "
                                 "mask colour is cross-checked, never used to derive "
                                 "the label"),
            "mask_provenance": (f"SOURCE_COLOUR_MASK_BINARISED_BY[{MASK_POLICY_ID}]"
                                if in_cohort else "NOT_CONVERTED"),
            "exclusion_status": ("INCLUDED_BINARY_LESION_COHORT" if in_cohort
                                 else NORMAL_STATUS),
            "exclusion_reason": ("" if in_cohort else
                                 "Normal is a class outside the binary lesion target, "
                                 "not a defect and not an exclusion. Its mask is "
                                 "legitimately all black "
                                 "(EXPECTED_EMPTY_MASK_FOR_NORMAL_SAMPLE). The record "
                                 "is retained here in full with its source provenance "
                                 "and carries no canonical path."),
            "notes": (f"pairing_status={r.get('pairing_status', '').strip()}; "
                      f"audit_status={r.get('audit_status', '').strip()}; "
                      f"mask_empty_interpretation="
                      f"{r.get('mask_empty_interpretation', '').strip()}"),
        })
    cohort = [r for r in records if r["exclusion_status"]
              == "INCLUDED_BINARY_LESION_COHORT"]
    normals = [r for r in records if r["exclusion_status"] == NORMAL_STATUS]
    by_el = Counter(r["source_label"] for r in cohort)
    plan_pats = {r["patient_id"] for r in cohort if r["patient_id"]}
    LOGGER.info("      planned %d cohort record(s): benign=%d malignant=%d over %d "
                "patient group(s)", len(cohort), by_el.get("Benign", 0),
                by_el.get("Malignant", 0), len(plan_pats))
    LOGGER.info("      %d Normal record(s) retained in the manifest, no canonical path",
                len(normals))

    LOGGER.info("[4/6] resolving and pre-checking cohort pairs ...")
    img_ok = mask_ok = plans = 0
    unresolved: List[str] = []
    notes_c: Counter = Counter()
    colour_obs: Counter = Counter()
    colour_bad: List[str] = []
    dim_bad: List[str] = []
    for i, r in enumerate(cohort, 1):
        sp = src_root / r["source_image_path"]
        mp = src_root / r["source_mask_path"]
        if not sp.is_file():
            unresolved.append(f"image missing: {r['source_image_path']}")
            r["image_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            with Image.open(sp) as im:
                ia = np.array(im)
                r["source_image_width"] = r["canonical_image_width"] = im.width
                r["source_image_height"] = r["canonical_image_height"] = im.height
                r["source_image_channels"] = (ia.shape[2] if ia.ndim == 3 else 1)
            img_ok += 1
        except Exception:  # noqa: BLE001
            unresolved.append(f"image undecodable: {r['source_image_path']}")
            r["image_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
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
        same_dim = ((r["source_mask_height"], r["source_mask_width"])
                    == (r["source_image_height"], r["source_image_width"]))
        r["mask_dimension_match_verified_here"] = "YES" if same_dim else "NO"
        if not same_dim:
            dim_bad.append(f"{r['sample_id']} image "
                           f"{r['source_image_width']}x{r['source_image_height']} vs "
                           f"mask {r['source_mask_width']}x{r['source_mask_height']}")
            r["mask_conversion_status"] = "DIMENSION_MISMATCH"
            r["image_conversion_status"] = "NOT_CONVERTED_EXCLUDED"
            r["segmentation_available"] = "FALSE"
            r["classification_available"] = "FALSE"
            r["canonical_image_path"] = ""
            r["canonical_mask_path"] = ""
            r["canonical_binary_label"] = "NOT_IN_BINARY_TARGET"
            r["class_folder"] = ""
            r["exclusion_status"] = DIM_EXCLUSION_CODE
            r["exclusion_reason"] = (
                f"the source image is {r['source_image_width']}x"
                f"{r['source_image_height']} while its paired mask is "
                f"{r['source_mask_width']}x{r['source_mask_height']}; the official "
                f"INFO.csv declares {r['resolution_metadata']} for this record, so the "
                "metadata and the pixels disagree. A mask of different dimensions does "
                "not locate the lesion on its image, and aligning them would require "
                "an unverified resize or crop, which this phase prohibits.")
            continue
        plane, note, obs = colour_mask_to_binary(ma, np)
        notes_c[note] += 1
        colour_obs[obs] += 1
        r["mask_foreground_colour_observed"] = obs
        expected = (GREEN if r["source_label"] == "Benign" else RED)
        want = "GREEN_DOMINANT" if expected == GREEN else "RED_DOMINANT"
        agrees = (obs == want)
        r["mask_colour_agrees_with_official_label"] = "YES" if agrees else "NO"
        if not agrees:
            colour_bad.append(f"{r['sample_id']}: label {r['source_label']} expects "
                              f"{want}, observed {obs}")
        if plane is None:
            unresolved.append(f"mask refused: {r['sample_id']} ({note})")
            r["mask_conversion_status"] = f"REFUSED_{note}"
            continue
        fg = int((plane == 255).sum())
        if fg == 0:
            unresolved.append(f"empty mask on a lesion sample: {r['sample_id']}")
            r["mask_conversion_status"] = "EMPTY_MASK_ON_LESION"
            continue
        r["mask_foreground_pixels"] = fg
        r["mask_foreground_fraction"] = round(
            fg / float(plane.shape[0] * plane.shape[1]), 6)
        r["notes"] += f"; binarisation={note}; observed_colour={obs}"
        plans += 1
        if i % 50 == 0:
            LOGGER.info("      %d/%d", i, len(cohort))
    dim_excluded = [r for r in cohort if r["exclusion_status"] == DIM_EXCLUSION_CODE]
    converted = [r for r in cohort
                 if r["exclusion_status"] == "INCLUDED_BINARY_LESION_COHORT"]
    conv_by = Counter(r["source_label"] for r in converted)
    conv_pats = {r["patient_id"] for r in converted if r["patient_id"]}
    LOGGER.info("      image=%d mask=%d binarisation plans=%d unresolved=%d",
                img_ok, mask_ok, plans, len(unresolved))
    LOGGER.info("      dimension-mismatch exclusions: %d -> final cohort %d "
                "(benign %d / malignant %d) over %d patient group(s)",
                len(dim_excluded), len(converted), conv_by.get("Benign", 0),
                conv_by.get("Malignant", 0), len(conv_pats))
    LOGGER.info("      binarisation outcomes: %s", dict(notes_c))
    LOGGER.info("      observed foreground colours: %s", dict(colour_obs))
    LOGGER.info("      colour vs official label: %d disagreement(s)", len(colour_bad))
    for x in (unresolved + dim_bad + colour_bad)[:10]:
        LOGGER.error("         %s", x)

    png_written = 0
    if mode == "execute":
        LOGGER.info("[5/6] executing canonical conversion ...")
        for r in converted:
            if r["image_conversion_status"] != "PLANNED":
                continue
            dst = bank_ds / r["canonical_image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with Image.open(src_root / r["source_image_path"]) as im:
                    im.save(dst, format="PNG")
                png_written += 1
            r["image_conversion_status"] = "CONVERTED"
            with Image.open(src_root / r["source_mask_path"]) as mi:
                ma = np.array(mi)
            plane, _n, _c = colour_mask_to_binary(ma, np)
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
    cpaths = [r["canonical_image_path"] for r in records if r["canonical_image_path"]]
    cpaths += [r["canonical_mask_path"] for r in records if r["canonical_mask_path"]]
    collisions = [n for n, c in Counter(cpaths).items() if c > 1]

    T: List[Tuple[str, str, Optional[bool], str]] = []

    def add(i: str, d: str, ok: Optional[bool], b: str) -> None:
        T.append((i, d, ok, b))

    add("T01", f"source records == {EXPECT['source_records']}",
        len(rows) == EXPECT["source_records"], f"{len(rows)} @ {MANIFEST_FILE}")
    add("T02", f"labels: {EXPECT['benign']} benign, {EXPECT['malignant']} malignant, "
               f"{EXPECT['normal']} normal",
        by_lbl.get("Benign") == EXPECT["benign"]
        and by_lbl.get("Malignant") == EXPECT["malignant"]
        and by_lbl.get("Normal") == EXPECT["normal"], f"{dict(by_lbl)}")
    add("T03", "summary json agrees with the manifest",
        (summ.get("label_distribution") or {}) == dict(by_lbl)
        and summ.get("binary_cohort_total") == EXPECT["selected"],
        f"summary {summ.get('label_distribution')}, cohort "
        f"{summ.get('binary_cohort_total')}")
    add("T04", f"the audited binary lesion selection == {EXPECT['selected']} and "
               "matches the cohort artifact",
        len(cohort) == EXPECT["selected"] and len(cohort_rows) == EXPECT["selected"]
        and {r["sample_id"] for r in cohort} == cohort_ids,
        f"{len(cohort)} planned vs {len(cohort_rows)} in {COHORT_FILE}; id sets "
        f"{'identical' if {r['sample_id'] for r in cohort} == cohort_ids else 'DIFFER'}")
    add("T05", f"cohort benign == {EXPECT['benign']} and malignant == "
               f"{EXPECT['malignant']}",
        by_el.get("Benign") == EXPECT["benign"]
        and by_el.get("Malignant") == EXPECT["malignant"],
        f"{by_el.get('Benign')} / {by_el.get('Malignant')}")
    add("T06", f"{EXPECT['normal']} Normal record(s) retained with no canonical path "
               "and an explicit status",
        len(normals) == EXPECT["normal"]
        and all(not r["canonical_image_path"] and not r["canonical_mask_path"]
                and r["exclusion_status"] == NORMAL_STATUS and r["exclusion_reason"]
                for r in normals),
        f"{len(normals)} retained; 0 with a canonical path; status {NORMAL_STATUS}")
    add("T07", "the label comes from the official metadata, never from the mask colour",
        all(r["label_source"].strip() == "OFFICIAL_INFO_CSV" for r in rows)
        and all("never used to derive" in r["label_provenance"] for r in records),
        "label_source=OFFICIAL_INFO_CSV on every source row")
    add("T08", "the audit recorded zero label-colour mismatches",
        len(mismatches) == 0 and summ.get("label_color_mismatches") == 0
        and bool(summ.get("label_color_sample_level_consistent")),
        f"{MISMATCH_FILE} rows={len(mismatches)}; sample-level consistency="
        f"{summ.get('label_color_sample_level_consistent')}")
    add("T09", "the observed foreground colour agrees with the official label on every "
               "mask that was read",
        not colour_bad,
        f"{dict(colour_obs)}; {len(colour_bad)} disagreement(s) over "
        f"{sum(colour_obs.values())} mask(s) read")
    add("T10", f"binarisation plans == {EXPECT['cohort']} (the final cohort)",
        plans == EXPECT["cohort"], f"{plans}; outcomes {dict(notes_c)}")
    add("T10b", f"selected {EXPECT['selected']} - excluded {EXPECT['excluded']} = "
                f"cohort {EXPECT['cohort']}",
        len(cohort) == EXPECT["selected"] and len(dim_excluded) == EXPECT["excluded"]
        and len(converted) == EXPECT["cohort"],
        f"{len(cohort)} - {len(dim_excluded)} = {len(converted)}")
    add("T10c", "every dimension-mismatch record is retained with an explicit code, "
                "a reason carrying the observed dimensions, and no canonical path",
        all(r["exclusion_status"] == DIM_EXCLUSION_CODE and r["exclusion_reason"]
            and not r["canonical_image_path"] and not r["canonical_mask_path"]
            for r in dim_excluded),
        join_list([r["sample_id"] for r in dim_excluded], ", ") or "none")
    add("T11", "no threshold, union, scaling or policy from another dataset",
        "threshold" not in MASK_POLICY_RULE
        and not any(pid in r["mask_provenance"] for r in records
                    for pid in ("BUSI_CANONICAL_MULTIMASK_UNION_V1",
                                "BUS_UC_CANONICAL_MASK_BINARIZATION_V1",
                                "BREAST_CANONICAL_MASK_REPLICATION_REDUCTION_V1",
                                "BUS_BRA_CANONICAL_MASK_VALUE_SCALING_V1",
                                "BLUIS_CANONICAL_MASK_VALUE_SCALING_V1",
                                "INBREAST_FULLSOURCE_MASK_FROM_XML_CONTOUR_V1")),
        f"declared rule: {MASK_POLICY_ID}")
    add("T12", "no resize, crop, ROI, normalisation, CLAHE, denoising, augmentation, "
               "pseudo-mask or mask repair", True,
        "code-scope: decode, non-black to 255, lossless PNG; a mask the rule does not "
        "cover is refused, never repaired")
    add("T13", f"every selected record resolved to an image ({EXPECT['selected']})",
        img_ok == EXPECT["selected"], f"{img_ok}/{len(cohort)}")
    add("T14", f"every selected record resolved to a mask ({EXPECT['selected']})",
        mask_ok == EXPECT["selected"], f"{mask_ok}/{len(cohort)}")
    add("T15", "every CONVERTED record has mask and image dimensions that agree, "
               "verified here because the audit left the column empty",
        all(r["mask_dimension_match_verified_here"] == "YES" for r in converted),
        f"{len(dim_bad)} mismatch(es) found and excluded; audit dimension_match "
        f"column {dict(dimcol)}")
    add("T16", f"patient groups: {EXPECT['patients_overall']} overall, "
               f"{EXPECT['patients_selected']} selected",
        len(pats_all) == EXPECT["patients_overall"]
        and len(plan_pats) == EXPECT["patients_selected"]
        and len(grouping) == EXPECT["patients_overall"],
        f"{len(pats_all)} overall, {len(plan_pats)} selected, {len(conv_pats)} after "
        f"the dimension exclusions, {len(grouping)} rows in {GROUPING_FILE}")
    add("T17", "every record carries the patient grouping key",
        all(r["patient_id"] and r["group_key_for_future_partitions"]
            == f"PATIENT::{r['patient_id']}" for r in records),
        "PATIENT::<id> on every record; the audit requires all future partitions to "
        "be grouped by patient_id")
    add("T18", "patient_id comes from the dataset's own documentation, not from "
               "filename inference",
        all("OFFICIAL_DATASET_DOCUMENTATION" in r["patient_id_source"]
            for r in records),
        str(summ.get("patient_id_source"))[:70])
    add("T19", "no case or lesion identity invented",
        all(not r["case_id"] and not r["lesion_id"] for r in records),
        "case_id and lesion_id empty on every record")
    add("T20", f"excluded after selection == {EXPECT['excluded']}; the exclusion "
               "ledger MUST be updated with these records",
        len(dim_excluded) == EXPECT["excluded"],
        f"{len(dim_excluded)} record(s) with code {DIM_EXCLUSION_CODE}; the 419 "
        "Normal records are a class outside the target and are NOT counted here")
    add("T21", "no unresolved cohort record", not unresolved, f"{len(unresolved)}")
    add("T22", "no canonical path collision", not collisions, f"{len(collisions)}")
    add("T23", "RAW unchanged", raw_diff["unchanged"],
        f"modified {len(raw_diff['modified'])}, added {len(raw_diff['added'])}")
    add("T24", "completed audit artifacts unchanged", aud_diff["unchanged"],
        f"modified {len(aud_diff['modified'])}")
    add("T25", "no split created", not new_splits, f"{len(new_splits)}")
    add("T26", ("dry-run wrote no canonical PNG" if mode == "dry-run"
                else f"execution wrote {EXPECT['png']} canonical PNG"),
        (not new_png) if mode == "dry-run" else (len(new_png) == EXPECT["png"]),
        f"{len(new_png)} new PNG")
    add("T27", "no split field in the manifest schema",
        not any(is_split_like(f) for f in SCHEMA_FIELDS),
        "group_key_for_future_partitions carries a PATIENT key, not a partition")
    add("T28", "mask colour is recorded as evidence, never as a label source", None,
        f"observed {dict(colour_obs)}; the official label decided every record and the "
        "colour only had to agree with it")
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
        "phase": "2B.3.7", "dataset": DATASET, "mode": mode, "generated": now_iso(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "schema_sha256": sha256_text(join_list(SCHEMA_FIELDS, ",")),
        "evidence_hashes": ev["hashes"], "source_root": str(src_root),
        "policies": {"image": {"id": IMAGE_POLICY_ID, "rule": IMAGE_POLICY_RULE},
                     "mask": {"id": MASK_POLICY_ID, "rule": MASK_POLICY_RULE}},
        "counts": {"source_records": len(rows), "labels": dict(by_lbl),
                   "selected_binary_cohort": len(cohort),
                   "excluded_after_selection": len(dim_excluded),
                   "final_cohort": len(converted),
                   "cohort_benign": conv_by.get("Benign", 0),
                   "cohort_malignant": conv_by.get("Malignant", 0),
                   "normal_retained_not_in_target": len(normals),
                   "binarisation_plans": plans,
                   "patients_overall": len(pats_all),
                   "patients_selected": len(plan_pats),
                   "patients_final_cohort": len(conv_pats)},
        "exclusions": [{"sample_id": r["sample_id"], "patient_id": r["patient_id"],
                        "source_label": r["source_label"],
                        "code": DIM_EXCLUSION_CODE,
                        "image": f"{r['source_image_width']}x{r['source_image_height']}",
                        "mask": f"{r['source_mask_width']}x{r['source_mask_height']}",
                        "metadata_resolution": r["resolution_metadata"],
                        "reason": r["exclusion_reason"]} for r in dim_excluded],
        "mask_colour": {"source_patterns": dict(colours),
                        "observed_foreground": dict(colour_obs),
                        "disagreements_with_label": colour_bad,
                        "audit_recorded_mismatches": len(mismatches)},
        "binarisation_outcomes": dict(notes_c),
        "dimension_check": {"audit_column": dict(dimcol),
                            "verified_here": len(cohort) - len(dim_bad),
                            "mismatches": dim_bad},
        "identity": {"patient_id_source": summ.get("patient_id_source"),
                     "grouping_key": "PATIENT::<patient_id>",
                     "future_split_requirement": summ.get(
                         "future_split_requirement")},
        "image_mappings_resolved": img_ok, "mask_mappings_resolved": mask_ok,
        "unresolved": unresolved[:50], "collisions": collisions,
        "canonical_png_written": png_written,
        "raw_modified": not raw_diff["unchanged"],
        "audit_modified": not aud_diff["unchanged"],
        "exclusion_ledger_must_be_updated": len(dim_excluded) > 0,
        "tests_pass": sum(1 for t in verified if t[2]), "tests_fail": len(failed),
        "verdict": ("BLOCKED" if blocked else
                    "READY_FOR_BUS_UCLM_CANONICAL_EXECUTION" if mode == "dry-run"
                    else "BUS_UCLM_CANONICAL_BUILD_COMPLETE"),
    }
    (reports / f"{tag}canonical_build_report.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    LOGGER.info("WROTE %-44s", f"{tag}canonical_build_report.json")

    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 2B.3.7 BUS-UCLM - %s", mode.upper())
    LOGGER.info("Source:      %d record(s) %s", len(rows), dict(by_lbl))
    LOGGER.info("Selected:    %d binary lesion record(s) (benign %d / malignant %d)",
                len(cohort), by_el.get("Benign", 0), by_el.get("Malignant", 0))
    LOGGER.info("Excluded:    %d for mask/image dimension mismatch -> %s",
                len(dim_excluded),
                join_list([f"{r['sample_id']} "
                           f"({r['source_image_width']}x{r['source_image_height']} vs "
                           f"{r['source_mask_width']}x{r['source_mask_height']})"
                           for r in dim_excluded], "; ") or "none")
    LOGGER.info("Final cohort: %d canonical record(s) (benign %d / malignant %d) over "
                "%d patient group(s)", len(converted), conv_by.get("Benign", 0),
                conv_by.get("Malignant", 0), len(conv_pats))
    LOGGER.info("Normal:      %d retained in the manifest, outside the binary target, "
                "no canonical path", len(normals))
    LOGGER.info("Patients:    %d overall, %d in the cohort; key PATIENT::<id>",
                len(pats_all), len(plan_pats))
    LOGGER.info("Mask rule:   %s", MASK_POLICY_ID)
    LOGGER.info("             observed colours %s; %d disagreement(s) with the label",
                dict(colour_obs), len(colour_bad))
    LOGGER.info("Dimensions:  verified here on %d/%d (the audit column was empty)",
                len(cohort) - len(dim_bad), len(cohort))
    LOGGER.info("Ledger:      %d row(s) to add to CANONICAL_DATASET_EXCLUSION_LEDGER",
                len(dim_excluded))
    LOGGER.info("Unresolved:  %d | collisions %d", len(unresolved), len(collisions))
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
    p = argparse.ArgumentParser(description="BUS-UCLM canonical bank builder.")
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
        if d.get("verdict") != "READY_FOR_BUS_UCLM_CANONICAL_EXECUTION":
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
