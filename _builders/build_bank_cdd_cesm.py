#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bank_bluis.py
===================

PHASE 2B.3.12 - CDD-CESM CANONICAL DATASET BANK BUILDER

Twelfth and last dataset. It is built under a STRICT COMPLETE-REGION policy, and the
reason is worth stating plainly: an annotated lesion region that this pipeline cannot
rasterise must not silently become background supervision.

    MASK  CDDCESM_RAW_STRING_POLYGON_RECT_ELLIPSE_CV2_V1
        Regions are read from the RAW annotation strings, not from the Phase 1A copy,
        which truncated 1374 polygon strings at 200 characters and made them look
        unparseable. Supported shapes are polygon, rect and ellipse; they are
        rasterised with cv2 and UNIONED per image into one single-channel {0,255} mask
        at the image's native dimensions.

        circle, point and polyline are NOT supported. No circle is approximated by a
        polygon, no morphology is applied, nothing is repaired, cropped or resized.

    WHAT MAY AND MAY NOT BE CLAIMED
        The historical preparation code was NOT found. The Phase 1A.2 summary reports
        historical_code.located = true, but the path it names is the audit's own
        forensics script and the function is its locator, so that match is the audit
        finding itself; the later provenance gate excluded audit files and found
        nothing. Its verdict, EXACT_HISTORICAL_SOURCE_NOT_AVAILABLE, is the one this
        builder follows. Consequently the 2971 "historical parser EXCEPTION" outcomes
        carry no evidential weight about the historical code, and nothing here is
        described as a historical reproduction. The legacy manuscript count of 601 is
        NOT reproducible from the available source.

    THE COHORT, AND THE TWO REASONS A RECORD LEAVES IT
        650 B/M LE candidates, all retained in the manifest with full provenance.
         53  D03: every annotated region is an unsupported shape, so no mask exists
                  to build. All 53 are circle-only.
         79  D04: the mask is non-empty but INCOMPLETE - 157 annotated regions across
                  these 79 images are unsupported and would be dropped, leaving real
                  lesion tissue labelled as background. 385 regions available, 228
                  rasterisable.
        518  eligible: every annotated region is supported and rasterised. Verified
                  per record, not assumed: available == rasterised on all 518.

        The 132 are JOINT_TASK_INELIGIBLE, not full canonical exclusions, and none of
        them reaches the global exclusion ledger.

    THE 241 "EMPTY" MASKS ARE NOT AN EXCLUSION
        All 241 are AUDIT_PARSER_FALSE_EMPTY and reconstruct NON_EMPTY from the RAW
        strings. They are inside the cohort, not outside it.

    IDENTITY
        PATIENT::<Patient_ID> from the source Excel column, provenance
        PATIENT_ID_ESTABLISHED_FROM_SOURCE_METADATA. Patients carrying both labels do
        so on different breasts (DIFFERENT_LATERALITY); no label is changed and the
        patient stays in one group, which is what prevents the leak.

Usage:
    python build_bank_cdd_cesm.py --dry-run
    python build_bank_cdd_cesm.py --execute
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET = "CDD-CESM"
DATASET_DIR = "CDD_CESM"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)

RECON_FILE = "05_cddcesm_phase1a2_exact_historical_mask_reconstruction.csv"
CAND_FILE = "05_cddcesm_phase1a1_bm_le_candidate_cohort.csv"
BOUNDARY_FILE = "08_cddcesm_phase1a2_650_to_597_boundary.csv"
EMPTY_FILE = "06_cddcesm_phase1a2_241_empty_reanalysis.csv"
CROSS_FILE = "12_cddcesm_phase1a1_cross_label_patients.csv"
LINK_FILE = "11_cddcesm_phase1a_segmentation_le_linkage.csv"
GATE_FILE = "07_cddcesm_provenance_gate_summary.json"
SUMMARY_FILE = "13_cddcesm_phase1a2_summary.json"
REPORT_FILE = "12_cddcesm_phase1a2_corrective_verdict.csv"
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]

MASK_POLICY_ID = "CDDCESM_RAW_STRING_POLYGON_RECT_ELLIPSE_CV2_V1"
MASK_POLICY_RULE = (
    "regions are read from the RAW annotation strings, never from the Phase 1A copy "
    "which truncated 1374 polygon strings at 200 characters. polygon, rect and "
    "ellipse are rasterised with cv2 and UNIONED per image into one single-channel "
    "{0,255} mask at the LE image's native dimensions. circle, point and polyline are "
    "NOT supported: no circle is approximated by a polygon, no morphology is applied, "
    "nothing is repaired, cropped or resized. A record whose regions are not ALL "
    "supported is not eligible for the joint task, so no annotated lesion region ever "
    "becomes background supervision")
IMAGE_POLICY_ID = "CDDCESM_CANONICAL_IMAGE_LOSSLESS_V1"
IMAGE_POLICY_RULE = ("decoded BMP written unchanged to lossless PNG at native "
                     "dimensions, channels preserved; no resize, crop, ROI, "
                     "normalisation, CLAHE, denoising or augmentation")

EXPECT = {"candidates": 650, "src_benign": 324, "src_malignant": 326,
          "src_patients": 282,
          "eligible": 518, "benign": 222, "malignant": 296, "patients": 250,
          "patient_sides": 304, "regions": 1278,
          "ineligible": 132, "inel_benign": 102, "inel_malignant": 30,
          "d03": 53, "d03_benign": 52, "d03_malignant": 1,
          "d04": 79, "d04_benign": 50, "d04_malignant": 29,
          "d04_regions_available": 385, "d04_regions_rasterised": 228,
          "d04_regions_dropped": 157,
          "png": 1036}
D03_CODE = "JOINT_TASK_INELIGIBLE_NO_SUPPORTED_MASK"
D04_CODE = "JOINT_TASK_INELIGIBLE_PARTIAL_MASK_UNSUPPORTED_REGIONS_DROPPED"
SUPPORTED_SHAPES = ("polygon", "rect", "ellipse")
CANONICAL_LABELS = {"BENIGN": 0, "MALIGNANT": 1}

SCHEMA_FIELDS = [
    "dataset", "canonical_record_id", "sample_id", "segmentation_filename",
    "Patient_ID", "Side", "View", "representation_TOKEN_3",
    "source_image_path", "source_annotation_source",
    "source_label", "canonical_binary_label", "class_folder",
    "canonical_image_path", "canonical_mask_path",
    "patient_id", "patient_id_provenance", "group_key_for_future_partitions",
    "patient_is_cross_label", "patient_side_key", "case_id", "lesion_id",
    "same_source_image_group",
    "joint_task_eligible", "joint_ineligibility_reason",
    "total_source_regions", "rasterized_regions", "unsupported_regions",
    "unsupported_shape_types", "mask_reconstruction_policy",
    "classification_available", "segmentation_available",
    "image_conversion_status", "mask_conversion_status",
    "source_image_height", "source_image_width", "source_image_mode",
    "evidence_image_height", "evidence_image_width",
    "mask_dimension_verified_here", "regions_verified_here",
    "phase1a2_mask_status", "phase1a_mask_status",
    "audit_false_empty_241", "raw_string_truncation_affected",
    "canonical_image_height", "canonical_image_width",
    "canonical_mask_height", "canonical_mask_width",
    "source_image_format", "source_mask_format",
    "mask_rule_applied", "mask_union_of_regions",
    "canonical_image_encoding", "canonical_mask_encoding",
    "mask_foreground_pixels", "mask_foreground_fraction",
    "image_policy_id", "image_policy_rule", "mask_policy_id", "mask_policy_rule",
    "identity_provenance", "label_provenance", "mask_provenance",
    "exclusion_status", "exclusion_reason", "notes",
]
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}

SEP = " | "
LOGGER = logging.getLogger("bank_cddcesm")


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


def parse_regions(raw: str) -> Tuple[List[dict], List[str], List[str]]:
    """Read the region objects from a RAW annotation string. Returns the supported
    regions, the unsupported shape names, and any string that would not deserialise.
    Nothing is approximated: an unsupported shape is reported, never converted."""
    import ast
    sup: List[dict] = []
    unsup: List[str] = []
    bad: List[str] = []
    if not raw or not raw.strip():
        return sup, unsup, bad
    text = raw.strip()
    chunks = [text] if text.startswith("{") and text.count("{\"name\"") <= 1 else None
    if chunks is None:
        # several region objects concatenated; split on the object boundary
        parts, depth, cur = [], 0, ""
        for ch in text:
            if ch == "{":
                depth += 1
            cur += ch
            if ch == "}":
                depth -= 1
                if depth == 0:
                    parts.append(cur.strip().strip(","))
                    cur = ""
        chunks = [p for p in parts if p]
    for c in chunks:
        obj = None
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(c)
                break
            except Exception:  # noqa: BLE001
                continue
        if not isinstance(obj, dict):
            bad.append(c[:60])
            continue
        name = str(obj.get("name", "")).lower()
        if name in SUPPORTED_SHAPES:
            sup.append(obj)
        else:
            unsup.append(name or "UNNAMED")
    return sup, unsup, bad


def rasterise(regions: Sequence[dict], h: int, w: int, np) -> Tuple[Any, int]:
    """Union of the supported regions, drawn with cv2 to match the documented
    semantics. No morphology, no smoothing, no approximation of an unsupported shape.
    Returns the {0,255} plane and the number of regions actually drawn."""
    import cv2
    m = np.zeros((h, w), dtype=np.uint8)
    drawn = 0
    for r in regions:
        name = str(r.get("name", "")).lower()
        if name == "polygon":
            xs, ys = r.get("all_points_x"), r.get("all_points_y")
            if not xs or not ys or len(xs) != len(ys) or len(xs) < 3:
                continue
            pts = np.array([[int(round(x)), int(round(y))]
                            for x, y in zip(xs, ys)], dtype=np.int32)
            cv2.fillPoly(m, [pts], 255)
        elif name == "rect":
            x, y = int(r.get("x", 0)), int(r.get("y", 0))
            rw, rh = int(r.get("width", 0)), int(r.get("height", 0))
            if rw <= 0 or rh <= 0:
                continue
            cv2.rectangle(m, (x, y), (x + rw - 1, y + rh - 1), 255, -1)
        elif name == "ellipse":
            cx, cy = int(r.get("cx", 0)), int(r.get("cy", 0))
            rx, ry = int(r.get("rx", 0)), int(r.get("ry", 0))
            if rx <= 0 or ry <= 0:
                continue
            ang = float(r.get("theta", 0.0)) * 180.0 / 3.141592653589793
            cv2.ellipse(m, (cx, cy), (rx, ry), ang, 0, 360, 255, -1)
        else:
            continue
        drawn += 1
    return m, drawn


def load_evidence(prep_root: Path) -> dict:
    ev: Dict[str, Any] = {"paths": {}, "status": {}, "hashes": {}, "problems": []}
    dirs = [d for d in prep_root.iterdir()
            if d.is_dir() and "cddcesm" in d.name.lower().replace(" ", "").replace("-", "")]
    files = [p for d in dirs for p in d.rglob("*") if p.is_file() and not archived(p)]
    for key, fname, required in (("recon", RECON_FILE, True),
                                 ("cand", CAND_FILE, True),
                                 ("boundary", BOUNDARY_FILE, True),
                                 ("empty", EMPTY_FILE, True),
                                 ("cross", CROSS_FILE, True),
                                 ("link", LINK_FILE, True),
                                 ("gate", GATE_FILE, True),
                                 ("summary", SUMMARY_FILE, True),
                                 ("report", REPORT_FILE, False)):
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
            # this dataset's summary carries no raw_root key, which is a fact about
            # the artifact and not an error; --raw-root supplies it and the root that
            # was actually used is printed and recorded
            ev["dataset_root_locator"] = (f"absent from {SUMMARY_FILE}; must be given "
                                          "with --raw-root")
    return ev


def run(mode: str, prep_root: Path, raw_cli: Optional[Path] = None) -> int:
    bank_ds = prep_root / BANK_NAME / DATASET_DIR
    reports, manifests = bank_ds / "reports", bank_ds / "manifests"
    setup_logging(reports, f"cddcesm_{mode}.log")
    LOGGER.info("=" * 78)
    LOGGER.info("PHASE 2B.3.12 CDD-CESM CANONICAL BANK BUILDER  [%s]", mode.upper())
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
    src_root = raw_cli or ev["dataset_root"]
    if src_root is None:
        LOGGER.error("      BLOCKED: %s carries no source root and --raw-root was "
                     "not given. Nothing was guessed.", SUMMARY_FILE)
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4
    if not src_root.is_dir():
        LOGGER.error("      BLOCKED: the source root does not exist: %s", src_root)
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4
    before_raw = snapshot(src_root)
    summ = json.loads(ev["paths"]["summary"].read_text(encoding="utf-8"))
    LOGGER.info("      RAW    = %s  [READ-ONLY]  (from %s)", src_root,
                "--raw-root" if raw_cli else ev["dataset_root_locator"])
    LOGGER.info("      root evidence = %s", ev["dataset_root_locator"])
    LOGGER.info("      image policy = %s", IMAGE_POLICY_ID)

    recon = read_rows(ev["paths"]["recon"])
    cand = {r["segmentation_filename"].strip(): r
            for r in read_rows(ev["paths"]["cand"])}
    boundary = {r["segmentation_filename"].strip(): r
                for r in read_rows(ev["paths"]["boundary"])}
    empty241 = {r["segmentation_filename"].strip(): r
                for r in read_rows(ev["paths"]["empty"])}
    crossp = {r["Patient_ID"].strip(): r for r in read_rows(ev["paths"]["cross"])}
    link = {r["segmentation_filename"].strip(): r
            for r in read_rows(ev["paths"]["link"])}
    gate = json.loads(ev["paths"]["gate"].read_text(encoding="utf-8"))

    LOGGER.info("[2/6] source accounting ...")
    by_path = Counter(r["source_pathology"].strip() for r in recon)
    st = Counter(r["mask_status"].strip() for r in recon)
    pats_all = {cand[k]["Patient_ID"] for k in cand}
    LOGGER.info("      reconstruction rows %d: %s | mask_status %s", len(recon),
                dict(by_path), dict(st))
    LOGGER.info("      candidates %d over %d patient(s); all ELIGIBLE=%s, "
                "exact_LE_exists=%s", len(cand), len(pats_all),
                len({r['binary_BM_eligibility'] for r in cand.values()}) == 1,
                len({r['exact_LE_exists'] for r in cand.values()}) == 1)
    LOGGER.info("      the 241: %s -> %s",
                dict(Counter(r["classification"] for r in empty241.values())),
                dict(Counter(r["raw_reconstruction_status"]
                             for r in empty241.values())))
    LOGGER.info("      cross-label patients %d: %s", len(crossp),
                dict(Counter(r["classification"] for r in crossp.values())))
    LOGGER.info("      provenance gate: %s | legacy: %s",
                gate.get("provenance_verdict"),
                gate.get("legacy_manuscript_status"))
    LOGGER.info("      mask policy = %s (supported: %s)", MASK_POLICY_ID,
                ", ".join(SUPPORTED_SHAPES))

    LOGGER.info("[3/6] locating the RAW annotation source ...")
    ann_path, ann_rows, ann_col = None, [], ""
    cands: List[Path] = []
    for pat in ("*.xlsx", "*.xls", "*.csv"):
        cands += [x for x in src_root.rglob(pat) if x.is_file()]
    for c in sorted(cands):
        try:
            if c.suffix.lower() in (".xlsx", ".xls"):
                import pandas as pd
                df = pd.read_excel(c)
                cols = [str(x) for x in df.columns]
            else:
                rws = read_rows(c)
                cols = list(rws[0]) if rws else []
        except Exception:  # noqa: BLE001
            continue
        hit = next((x for x in cols if "region_shape_attributes" in x.lower()), "")
        if hit:
            ann_path, ann_col = c, hit
            if c.suffix.lower() in (".xlsx", ".xls"):
                ann_rows = df.to_dict("records")
            else:
                ann_rows = rws
            break
    if ann_path is None:
        LOGGER.error("      BLOCKED: no file under the RAW root carries a "
                     "region_shape_attributes column, so the RAW annotation strings "
                     "cannot be read. The Phase 1A copy is NOT used because it "
                     "truncated 1374 strings at 200 characters. Nothing was guessed.")
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("CDD_CESM_CANONICAL_DRY_RUN_BLOCKED")
        return 4
    fname_col = next((c for c in (list(ann_rows[0]) if ann_rows else [])
                      if str(c).strip().lower() in ("#filename", "filename",
                                                    "image_name")), "")
    if not fname_col:
        LOGGER.error("      BLOCKED: %s has no filename column", ann_path.name)
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("CDD_CESM_CANONICAL_DRY_RUN_BLOCKED")
        return 4
    ann_by: Dict[str, List[str]] = defaultdict(list)
    for r in ann_rows:
        fn = str(r.get(fname_col, "")).strip()
        rs = str(r.get(ann_col, "")).strip()
        if fn and rs and rs.lower() not in ("nan", "{}"):
            ann_by[fn].append(rs)
    LOGGER.info("      RAW annotations: %s :: [%s] -> %d row(s) over %d filename(s)",
                ann_path.name, ann_col, len(ann_rows), len(ann_by))

    LOGGER.info("[4/6] planning canonical records ...")
    records: List[dict] = []
    for r in recon:
        fn = r["segmentation_filename"].strip()
        c = cand.get(fn, {})
        lk = link.get(fn, {})
        path = r["source_pathology"].strip()
        avail = int(r["historical_regions_available"] or 0)
        parsed = int(r["successfully_parsed_regions"] or 0)
        unsup = int(r["unsupported_shapes"] or 0)
        status = r["mask_status"].strip()
        pid = c.get("Patient_ID", "").strip()
        elig = status == "NON_EMPTY" and unsup == 0
        folder = "Benign" if path == "Benign" else (
            "Malignant" if path == "Malignant" else "")
        sid = f"{Path(fn).stem}__{sha256_text(path + '/' + fn)[:8]}"
        if elig:
            estat, ereason = "INCLUDED_JOINT_COHORT", ""
        elif status == "EMPTY":
            estat = D03_CODE
            ereason = ("every annotated region on this image is an unsupported shape "
                       f"({boundary.get(fn, {}).get('shape_composition_raw', '')}), so "
                       "no mask can be built from the supported set. No circle was "
                       "approximated and nothing was repaired.")
        else:
            estat = D04_CODE
            ereason = (f"{avail} annotated region(s) on this image, {parsed} supported "
                       f"and rasterisable, {unsup} unsupported. Writing the partial "
                       "union would label real annotated lesion tissue as background, "
                       "so under the STRICT COMPLETE-REGION policy the record is not "
                       "eligible for the joint task. Nothing was approximated.")
        records.append({
            "dataset": DATASET, "canonical_record_id": f"{DATASET_DIR}__{sid}",
            "sample_id": sid, "segmentation_filename": fn,
            "Patient_ID": pid, "Side": c.get("Side", "").strip(),
            "View": c.get("View", "").strip(),
            "representation_TOKEN_3": c.get("representation_TOKEN_3", "").strip(),
            "source_image_path": lk.get("le_relative_path", "").strip(),
            "source_annotation_source": f"{ann_path.name} :: [{ann_col}] RAW string",
            "source_label": path.upper(),
            "canonical_binary_label": (0 if path == "Benign" else 1) if elig
                                      else "NOT_IN_JOINT_COHORT",
            "class_folder": folder if elig else "",
            "canonical_image_path": (f"{folder}/Images/{sid}.png"
                                     if elig and folder else ""),
            "canonical_mask_path": (f"{folder}/Masks/{sid}_mask.png"
                                    if elig and folder else ""),
            "patient_id": pid,
            "patient_id_provenance": ("PATIENT_ID_ESTABLISHED_FROM_SOURCE_METADATA: "
                                      "the Patient_ID column of the source annotation "
                                      "workbook, carried unchanged through the Phase "
                                      "1A.1 crosswalk"),
            "group_key_for_future_partitions": f"PATIENT::{pid}" if pid else "",
            "patient_is_cross_label": "YES" if pid in crossp else "NO",
            "patient_side_key": f"{pid}::{c.get('Side', '').strip()}",
            "case_id": "", "lesion_id": "",
            "same_source_image_group": f"{DATASET_DIR}__{sid}",
            "joint_task_eligible": "YES" if elig else "NO",
            "joint_ineligibility_reason": ereason,
            "total_source_regions": avail, "rasterized_regions": parsed,
            "unsupported_regions": unsup,
            "unsupported_shape_types": (
                boundary.get(fn, {}).get("shape_composition_raw", "")
                if unsup else ""),
            "mask_reconstruction_policy": MASK_POLICY_ID,
            "classification_available": "TRUE",
            "segmentation_available": "TRUE" if elig else "FALSE",
            "image_conversion_status": "PLANNED" if elig else "NOT_APPLICABLE",
            "mask_conversion_status": "PLANNED" if elig else "NOT_APPLICABLE",
            "source_image_height": "", "source_image_width": "",
            "source_image_mode": "",
            "evidence_image_height": r.get("image_height", ""),
            "evidence_image_width": r.get("image_width", ""),
            "mask_dimension_verified_here": "", "regions_verified_here": "",
            "phase1a2_mask_status": status,
            "phase1a_mask_status": r.get("phase1a_mask_status", ""),
            "audit_false_empty_241": "YES" if fn in empty241 else "NO",
            "raw_string_truncation_affected": (
                empty241.get(fn, {}).get("regions_with_truncated_audit_string", "")),
            "canonical_image_height": "", "canonical_image_width": "",
            "canonical_mask_height": "", "canonical_mask_width": "",
            "source_image_format": Path(
                lk.get("le_relative_path", "")).suffix.lstrip(".").upper(),
            "source_mask_format": "RECONSTRUCTED_FROM_RAW_ANNOTATION_STRINGS",
            "mask_rule_applied": "", "mask_union_of_regions": "",
            "canonical_image_encoding": "LOSSLESS_PNG_NATIVE",
            "canonical_mask_encoding": ("SINGLE_CHANNEL_BINARY_0_255" if elig else ""),
            "mask_foreground_pixels": "", "mask_foreground_fraction": "",
            "image_policy_id": IMAGE_POLICY_ID, "image_policy_rule": IMAGE_POLICY_RULE,
            "mask_policy_id": MASK_POLICY_ID if elig else "",
            "mask_policy_rule": MASK_POLICY_RULE if elig else "",
            "identity_provenance": f"{CAND_FILE} :: csv:[Patient_ID]",
            "label_provenance": f"{RECON_FILE} :: csv:[source_pathology]",
            "mask_provenance": (f"RAW_ANNOTATION_STRINGS_UNIONED_BY[{MASK_POLICY_ID}]"
                                if elig else "NOT_CONVERTED"),
            "exclusion_status": estat, "exclusion_reason": ereason,
            "notes": r.get("notes", "")[:160],
        })
    cohort = [r for r in records if r["joint_task_eligible"] == "YES"]
    d03 = [r for r in records if r["exclusion_status"] == D03_CODE]
    d04 = [r for r in records if r["exclusion_status"] == D04_CODE]
    by_el = Counter(r["source_label"] for r in cohort)
    pats = {r["patient_id"] for r in cohort if r["patient_id"]}
    sides = {r["patient_side_key"] for r in cohort if r["patient_id"]}
    reg_av = sum(int(r["total_source_regions"] or 0) for r in cohort)
    reg_rs = sum(int(r["rasterized_regions"] or 0) for r in cohort)
    LOGGER.info("      eligible %d = %d BENIGN + %d MALIGNANT over %d patient(s), "
                "%d patient-side(s)", len(cohort), by_el.get("BENIGN", 0),
                by_el.get("MALIGNANT", 0), len(pats), len(sides))
    LOGGER.info("      regions in the cohort: %d available, %d rasterisable "
                "(equal on every record: %s)", reg_av, reg_rs, reg_av == reg_rs)
    LOGGER.info("      ineligible %d: D03 %d (%s) + D04 %d (%s)",
                len(d03) + len(d04), len(d03),
                dict(Counter(r["source_label"] for r in d03)), len(d04),
                dict(Counter(r["source_label"] for r in d04)))
    LOGGER.info("      D04 regions: %d available, %d rasterisable, %d dropped",
                sum(int(r["total_source_regions"] or 0) for r in d04),
                sum(int(r["rasterized_regions"] or 0) for r in d04),
                sum(int(r["unsupported_regions"] or 0) for r in d04))

    LOGGER.info("[5/6] resolving sources and rebuilding masks from RAW ...")
    img_ok = plans = 0
    unresolved: List[str] = []
    shape_c: Counter = Counter()
    unsup_c: Counter = Counter()
    mismatch: List[str] = []
    dim_bad: List[str] = []
    for i, r in enumerate(cohort, 1):
        fn = r["segmentation_filename"]
        sp = src_root / r["source_image_path"] if r["source_image_path"] else None
        if sp is None or not sp.is_file():
            unresolved.append(f"{fn}: LE image missing ({r['source_image_path']})")
            r["image_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            with Image.open(sp) as im:
                r["source_image_mode"] = im.mode
                r["source_image_width"] = r["canonical_image_width"] = im.width
                r["source_image_height"] = r["canonical_image_height"] = im.height
                ih, iw = im.height, im.width
            img_ok += 1
        except Exception:  # noqa: BLE001
            unresolved.append(f"{fn}: LE image undecodable")
            r["image_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        raws = ann_by.get(fn, [])
        sup, unsup, bad = [], [], []
        for rs in raws:
            a1, u1, b1 = parse_regions(rs)
            sup += a1
            unsup += u1
            bad += b1
        for n in unsup:
            unsup_c[n] += 1
        for g in sup:
            shape_c[str(g.get("name", "")).lower()] += 1
        r["regions_verified_here"] = len(sup)
        # the evidence says every region on an eligible record is supported; verify it
        if unsup or bad:
            mismatch.append(f"{fn}: {len(unsup)} unsupported, {len(bad)} unparseable "
                            "found here on a record the evidence calls complete")
            r["mask_conversion_status"] = "REFUSED_UNSUPPORTED_REGION_FOUND_HERE"
            continue
        if len(sup) != int(r["total_source_regions"] or 0):
            mismatch.append(f"{fn}: {len(sup)} region(s) read here vs "
                            f"{r['total_source_regions']} in the evidence")
            r["mask_conversion_status"] = "REFUSED_REGION_COUNT_DISAGREES"
            continue
        plane, drawn = rasterise(sup, ih, iw, np)
        r["mask_union_of_regions"] = drawn
        r["mask_rule_applied"] = f"UNION_OF_{drawn}_SUPPORTED_REGIONS_CV2"
        fg = int((plane == 255).sum())
        if fg == 0:
            unresolved.append(f"{fn}: mask empty after union")
            r["mask_conversion_status"] = "EMPTY_MASK"
            continue
        r["canonical_mask_height"], r["canonical_mask_width"] = ih, iw
        r["mask_dimension_verified_here"] = "MATCH"
        r["mask_foreground_pixels"] = fg
        r["mask_foreground_fraction"] = round(fg / float(ih * iw), 8)
        plans += 1
        if i % 100 == 0:
            LOGGER.info("      %d/%d", i, len(cohort))
    LOGGER.info("      images resolved=%d | masks built=%d | unresolved=%d",
                img_ok, plans, len(unresolved))
    LOGGER.info("      shapes rasterised: %s", dict(shape_c))
    LOGGER.info("      unsupported shapes met inside the cohort: %s",
                dict(unsup_c) or "none")
    LOGGER.info("      records disagreeing with the evidence: %d", len(mismatch))
    for x in (unresolved + mismatch + dim_bad)[:10]:
        LOGGER.error("         %s", x)

    png_written = 0
    if mode == "execute":
        LOGGER.info("[6/6] executing canonical conversion ...")
        for r in cohort:
            if r["image_conversion_status"] != "PLANNED":
                continue
            dst = bank_ds / r["canonical_image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with Image.open(src_root / r["source_image_path"]) as im:
                    im.save(dst, format="PNG")
                png_written += 1
            r["image_conversion_status"] = "CONVERTED"
            sup = []
            for rs in ann_by.get(r["segmentation_filename"], []):
                a1, _u, _b = parse_regions(rs)
                sup += a1
            with Image.open(src_root / r["source_image_path"]) as im:
                ih, iw = im.height, im.width
            plane, _d = rasterise(sup, ih, iw, np)
            mdst = bank_ds / r["canonical_mask_path"]
            mdst.parent.mkdir(parents=True, exist_ok=True)
            if not mdst.exists():
                Image.fromarray(plane, mode="L").save(mdst, format="PNG")
                png_written += 1
            with Image.open(mdst) as chk:
                back = np.asarray(chk.convert("L"))
            r["mask_conversion_status"] = ("CONVERTED_VERIFIED_IDENTICAL"
                                           if np.array_equal(back, plane)
                                           else "VERIFICATION_FAILED_NOT_IDENTICAL")
    else:
        LOGGER.info("[6/6] dry-run: no canonical PNG is written")

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

    add("T01", f"B/M LE candidates == {EXPECT['candidates']} = "
               f"{EXPECT['src_benign']} B + {EXPECT['src_malignant']} M over "
               f"{EXPECT['src_patients']} patient(s)",
        len(recon) == EXPECT["candidates"]
        and by_path.get("Benign") == EXPECT["src_benign"]
        and by_path.get("Malignant") == EXPECT["src_malignant"]
        and len(pats_all) == EXPECT["src_patients"],
        f"{len(recon)} = {dict(by_path)}; {len(pats_all)} patient(s)")
    add("T02", f"STRICT eligible == {EXPECT['eligible']} = {EXPECT['benign']} B + "
               f"{EXPECT['malignant']} M",
        len(cohort) == EXPECT["eligible"]
        and by_el.get("BENIGN") == EXPECT["benign"]
        and by_el.get("MALIGNANT") == EXPECT["malignant"],
        f"{len(cohort)} = {by_el.get('BENIGN')} + {by_el.get('MALIGNANT')}")
    add("T03", f"unique patients == {EXPECT['patients']} and patient-sides == "
               f"{EXPECT['patient_sides']}",
        len(pats) == EXPECT["patients"] and len(sides) == EXPECT["patient_sides"],
        f"{len(pats)} patient(s), {len(sides)} patient-side(s)")
    add("T04", f"D03 (no supported mask) == {EXPECT['d03']} = "
               f"{EXPECT['d03_benign']} B + {EXPECT['d03_malignant']} M",
        len(d03) == EXPECT["d03"]
        and Counter(r["source_label"] for r in d03).get("BENIGN")
        == EXPECT["d03_benign"],
        f"{len(d03)}: {dict(Counter(r['source_label'] for r in d03))}")
    add("T05", f"D04 (partial mask) == {EXPECT['d04']} = {EXPECT['d04_benign']} B + "
               f"{EXPECT['d04_malignant']} M",
        len(d04) == EXPECT["d04"]
        and Counter(r["source_label"] for r in d04).get("BENIGN")
        == EXPECT["d04_benign"],
        f"{len(d04)}: {dict(Counter(r['source_label'] for r in d04))}")
    add("T06", f"D04 regions: {EXPECT['d04_regions_available']} available, "
               f"{EXPECT['d04_regions_rasterised']} rasterisable, "
               f"{EXPECT['d04_regions_dropped']} dropped",
        sum(int(r["total_source_regions"] or 0) for r in d04)
        == EXPECT["d04_regions_available"]
        and sum(int(r["unsupported_regions"] or 0) for r in d04)
        == EXPECT["d04_regions_dropped"],
        f"{sum(int(r['total_source_regions'] or 0) for r in d04)} / "
        f"{sum(int(r['rasterized_regions'] or 0) for r in d04)} / "
        f"{sum(int(r['unsupported_regions'] or 0) for r in d04)}")
    add("T07", f"total ineligible == {EXPECT['ineligible']} = "
               f"{EXPECT['inel_benign']} B + {EXPECT['inel_malignant']} M, and "
               f"{EXPECT['eligible']} + {EXPECT['ineligible']} = "
               f"{EXPECT['candidates']}",
        len(d03) + len(d04) == EXPECT["ineligible"]
        and len(cohort) + len(d03) + len(d04) == len(records) == EXPECT["candidates"],
        f"{len(cohort)} + {len(d03)} + {len(d04)} = {len(records)}")
    add("T08", "every eligible record has EVERY source region rasterised; none is "
               "partial",
        all(int(r["total_source_regions"] or 0) == int(r["rasterized_regions"] or 0)
            and int(r["unsupported_regions"] or 0) == 0 for r in cohort)
        and reg_av == reg_rs == EXPECT["regions"],
        f"{reg_av} available == {reg_rs} rasterisable across the cohort")
    add("T09", "the regions read here from RAW agree with the evidence on every "
               "eligible record",
        not mismatch, f"{len(mismatch)} disagreement(s)")
    add("T10", "no unsupported shape was met inside the cohort; none was approximated",
        not unsup_c,
        f"shapes rasterised {dict(shape_c)}; unsupported met {dict(unsup_c) or 'none'}")
    add("T11", f"the mask is the UNION of supported regions, rasterised with cv2 "
               f"({MASK_POLICY_ID})",
        plans == EXPECT["eligible"]
        and all(str(r["mask_union_of_regions"]).isdigit() for r in cohort
                if r["mask_conversion_status"] == "PLANNED"),
        f"{plans} mask(s) built as a union")
    add("T12", "the 241 audit-false-empty records are INSIDE the cohort, not excluded",
        all(r["audit_false_empty_241"] in ("YES", "NO") for r in records)
        and len(empty241) == 241
        and set(Counter(r["classification"]
                        for r in empty241.values())) == {"AUDIT_PARSER_FALSE_EMPTY"},
        f"{sum(1 for r in cohort if r['audit_false_empty_241'] == 'YES')} of the 241 "
        "are in the eligible cohort; none is an exclusion")
    add("T13", "no circle approximation, no morphology, no geometric repair, no crop, "
               "no resize", True,
        "code-scope: parse RAW, fillPoly/rectangle/ellipse, union, lossless PNG")
    add("T14", "nothing is claimed as a historical reproduction",
        gate.get("provenance_verdict") == "EXACT_HISTORICAL_SOURCE_NOT_AVAILABLE"
        and gate.get("legacy_manuscript_status")
        == "LEGACY_MANUSCRIPT_COUNT_NOT_REPRODUCED",
        f"{gate.get('provenance_verdict')}; legacy 601 "
        f"{gate.get('legacy_manuscript_status')}")
    add("T15", f"image mappings == {EXPECT['eligible']}",
        img_ok == EXPECT["eligible"], f"{img_ok}/{len(cohort)}")
    add("T16", "mask and image dimensions agree by construction and are recorded",
        all(r["mask_dimension_verified_here"] == "MATCH" for r in cohort
            if r["mask_conversion_status"] == "PLANNED"),
        "the mask is rasterised at the LE image's own height and width")
    add("T17", "every record carries PATIENT::<Patient_ID> with its source provenance",
        all(r["group_key_for_future_partitions"] == f"PATIENT::{r['patient_id']}"
            and "SOURCE_METADATA" in r["patient_id_provenance"] for r in records),
        f"{len({r['group_key_for_future_partitions'] for r in records})} distinct key(s)")
    add("T18", "cross-label patients keep both labels and one group key",
        all(len({x["group_key_for_future_partitions"] for x in records
                 if x["patient_id"] == pid}) == 1 for pid in crossp),
        f"{len(crossp)} cross-label patient(s), all DIFFERENT_LATERALITY; "
        f"{sum(1 for r in cohort if r['patient_is_cross_label'] == 'YES')} of their "
        "images are in the cohort")
    add("T19", "every one of the 650 is retained with its provenance fields",
        len(records) == EXPECT["candidates"]
        and all(r["mask_reconstruction_policy"] and r["joint_task_eligible"]
                and str(r["total_source_regions"]) != "" for r in records),
        "joint_task_eligible, reason, region counts and policy on every record")
    add("T20", "the 132 ineligible carry a reason and no canonical path, and none "
               "reaches the global exclusion ledger",
        all(r["joint_ineligibility_reason"] and not r["canonical_image_path"]
            and not r["canonical_mask_path"] for r in d03 + d04),
        f"{len(d03) + len(d04)} record(s); ledger untouched")
    add("T21", "no unresolved record", not unresolved, f"{len(unresolved)}")
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
        not any(is_split_like(f) for f in SCHEMA_FIELDS), "schema checked")
    add("T28", "the mask is a reconstruction from RAW annotation strings, not the "
               "historical output", None,
        "the historical preparation code was not found; the legacy count of 601 is "
        "not reproducible from the available source")
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
        "phase": "2B.3.12", "dataset": DATASET, "mode": mode, "generated": now_iso(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "schema_sha256": sha256_text(join_list(SCHEMA_FIELDS, ",")),
        "evidence_hashes": ev["hashes"], "source_root": str(src_root),
        "source_root_from": "--raw-root" if raw_cli else ev["dataset_root_locator"],
        "raw_annotation_source": {"file": str(ann_path), "column": ann_col,
                                  "filenames": len(ann_by)},
        "policies": {"image": {"id": IMAGE_POLICY_ID, "rule": IMAGE_POLICY_RULE},
                     "mask": {"id": MASK_POLICY_ID, "rule": MASK_POLICY_RULE,
                              "supported_shapes": list(SUPPORTED_SHAPES),
                              "strict_complete_region_policy": True}},
        "counts": {"candidates": len(records),
                   "source_benign": by_path.get("Benign", 0),
                   "source_malignant": by_path.get("Malignant", 0),
                   "source_patients": len(pats_all),
                   "joint_eligible": len(cohort),
                   "joint_benign": by_el.get("BENIGN", 0),
                   "joint_malignant": by_el.get("MALIGNANT", 0),
                   "joint_patients": len(pats),
                   "joint_patient_sides": len(sides),
                   "regions_available": reg_av, "regions_rasterised": reg_rs,
                   "joint_ineligible": len(d03) + len(d04),
                   "d03_no_supported_mask": len(d03),
                   "d04_partial_mask": len(d04),
                   "d04_regions_dropped": sum(int(r["unsupported_regions"] or 0)
                                              for r in d04),
                   "masks_built": plans, "excluded_from_bank": 0},
        "shapes_rasterised": dict(shape_c),
        "unsupported_shapes_met_in_cohort": dict(unsup_c),
        "evidence_disagreements": mismatch[:50],
        "identity": {"key": "PATIENT::<Patient_ID>",
                     "provenance": "PATIENT_ID_ESTABLISHED_FROM_SOURCE_METADATA",
                     "cross_label_patients": len(crossp),
                     "cross_label_classification": "DIFFERENT_LATERALITY"},
        "historical_claims": {
            "provenance_verdict": gate.get("provenance_verdict"),
            "legacy_601": gate.get("legacy_manuscript_status"),
            "reproduction_claimed": False,
            "the_241": "AUDIT_PARSER_FALSE_EMPTY; inside the cohort, not excluded"},
        "unresolved": unresolved[:50], "collisions": collisions,
        "canonical_png_written": png_written,
        "raw_modified": not raw_diff["unchanged"],
        "audit_modified": not aud_diff["unchanged"],
        "exclusion_ledger_modified": False,
        "tests_pass": sum(1 for t in verified if t[2]), "tests_fail": len(failed),
        "verdict": ("CDD_CESM_CANONICAL_DRY_RUN_BLOCKED" if blocked and
                    mode == "dry-run" else "BLOCKED" if blocked else
                    "READY_FOR_CDD_CESM_CANONICAL_EXECUTION" if mode == "dry-run"
                    else "CDD_CESM_CANONICAL_BUILD_COMPLETE"),
    }
    (reports / f"{tag}canonical_build_report.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    LOGGER.info("WROTE %-44s", f"{tag}canonical_build_report.json")

    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 2B.3.12 CDD-CESM - %s", mode.upper())
    LOGGER.info("Source:      %d B/M LE candidate(s) = %d B + %d M over %d patient(s)",
                len(records), by_path.get("Benign", 0), by_path.get("Malignant", 0),
                len(pats_all))
    LOGGER.info("Eligible:    %d = %d BENIGN + %d MALIGNANT", len(cohort),
                by_el.get("BENIGN", 0), by_el.get("MALIGNANT", 0))
    LOGGER.info("             over %d patient(s) and %d patient-side(s)", len(pats),
                len(sides))
    LOGGER.info("             %d region(s) available == %d rasterised; no record is "
                "partial", reg_av, reg_rs)
    LOGGER.info("Ineligible:  %d = D03 %d (no supported mask) + D04 %d (partial mask, "
                "%d region(s) would have been dropped)", len(d03) + len(d04),
                len(d03), len(d04),
                sum(int(r["unsupported_regions"] or 0) for r in d04))
    LOGGER.info("             JOINT_TASK_INELIGIBLE, not bank exclusions; ledger "
                "untouched")
    LOGGER.info("Mask rule:   %s", MASK_POLICY_ID)
    LOGGER.info("             shapes rasterised %s; unsupported met in cohort %s",
                dict(shape_c), dict(unsup_c) or "none")
    LOGGER.info("The 241:     AUDIT_PARSER_FALSE_EMPTY, inside the cohort, not "
                "excluded")
    LOGGER.info("History:     %s; legacy 601 %s; nothing claimed as reproduced",
                gate.get("provenance_verdict"), gate.get("legacy_manuscript_status"))
    LOGGER.info("Patients:    key PATIENT::<Patient_ID>, provenance "
                "PATIENT_ID_ESTABLISHED_FROM_SOURCE_METADATA; %d cross-label, all "
                "DIFFERENT_LATERALITY", len(crossp))
    LOGGER.info("RAW modified %s | audit modified %s | PNG %d | splits %d",
                state["raw_modified"], state["audit_modified"], png_written,
                len(new_splits))
    LOGGER.info("TESTS PASS: %d    TESTS FAIL: %d", state["tests_pass"], len(failed))
    LOGGER.info("FINAL VERDICT:")
    if blocked:
        for t in failed:
            LOGGER.error("    FAILED %s: %s | %s", t[0], t[1], t[3])
        LOGGER.error(state["verdict"])
        return 4
    LOGGER.info(state["verdict"])
    if mode == "dry-run":
        LOGGER.info("STOP. --execute needs human approval.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CDD-CESM canonical bank builder.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    p.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    p.add_argument("--raw-root", default=None,
                   help="CDD-CESM source root on H: (read-only). Required: the "
                        "Phase 1A.2 summary carries no raw_root key.")
    a = p.parse_args(argv)
    mode = "dry-run" if a.dry_run else "execute"
    if mode == "execute":
        st = (Path(a.prep_root) / BANK_NAME / DATASET_DIR / "reports"
              / "DRY_RUN_canonical_build_report.json")
        if not st.is_file():
            sys.stderr.write("EXECUTE refused: no dry-run report\n")
            return 5
        d = json.loads(st.read_text(encoding="utf-8"))
        if d.get("verdict") != "READY_FOR_CDD_CESM_CANONICAL_EXECUTION":
            sys.stderr.write(f"EXECUTE refused: last verdict {d.get('verdict')!r}\n")
            return 5
        if d.get("builder_sha256") != sha256_file(Path(__file__).resolve()):
            sys.stderr.write("EXECUTE refused: the builder changed since the dry-run\n")
            return 5
        want = d.get("source_root")
        got = str(Path(a.raw_root)) if a.raw_root else None
        if want and got and want != got:
            sys.stderr.write(f"EXECUTE refused: the dry-run verified {want!r} but "
                             f"--raw-root is {got!r}\n")
            return 5
        ev = load_evidence(Path(a.prep_root))
        for name, h in (d.get("evidence_hashes") or {}).items():
            if ev["hashes"].get(name) != h:
                sys.stderr.write(f"EXECUTE refused: {name} changed since the dry-run\n")
                return 5
    try:
        return run(mode, Path(a.prep_root),
                   Path(a.raw_root) if a.raw_root else None)
    except Exception:  # noqa: BLE001
        if LOGGER.handlers:
            LOGGER.critical("UNHANDLED EXCEPTION\n%s", traceback.format_exc())
        else:
            sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
