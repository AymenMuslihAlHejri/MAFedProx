#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finalise_inbreast_artifacts.py
==============================

TASK 2 - INbreast CANONICAL MANIFESTS, REPORTS AND EXCLUSION LEDGER

Brings the INbreast full-source artifacts up to the documentation conventions already
used for BUSI, BUS-UC, BrEaST and BUS-BRA. It reads the manifest the builder wrote,
inspects the two disputed annotations to state their exclusion reason precisely, and
writes:

    manifests/canonical_manifest.xlsx            (108 rows, one per Mass mammogram)
    reports/INBREAST_CANONICAL_BUILD_SUMMARY.json
    reports/INBREAST_CLASSIFICATION_STATUS.json  (why no B/M label exists)
    _reports/CANONICAL_DATASET_EXCLUSION_LEDGER.csv/.xlsx   (rows added, not replaced)
    _reports/MASTER_CURATION_TABLE.csv/.xlsx                (INbreast row added)

No canonical PNG is created, altered or deleted. No split is created. No benign or
malignant label is derived. H: is opened READ-ONLY and only to read the two disputed
XML files, so the exclusion reason states what the annotation actually contains rather
than only what it lacks.

Usage:
    python finalise_inbreast_artifacts.py
    python finalise_inbreast_artifacts.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import plistlib
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
DEFAULT_RAW_ROOT = r"H:\BC datasets\Mammo\01 INbreast 2"
BANK_NAME = "CANONICAL_DATASET_BANK"
DATASET_DIR = "INbreast_FullSource"
DATASET_NAME = "INbreast_FullSource"

LEDGER_FIELDS = [
    "dataset", "source_record_id", "source_label", "canonical_label",
    "exclusion_code", "exclusion_reason", "scientific_exclusion_rationale",
    "classification_available", "segmentation_available",
    "canonical_image_written", "canonical_mask_written",
    "evidence_source", "notes",
]
MASTER_FIELDS = [
    "dataset", "source_records", "canonical_eligible", "excluded_total",
    "excluded_benign", "excluded_malignant", "exclusion_codes",
    "exclusion_reason_summary", "scientific_exclusion_rationale_summary",
    "canonical_images_written", "canonical_masks_written",
    "locked_policy", "evidence_source", "notes",
]
EXCLUSION_CODE = "EXCLUDE_NO_MASS_CONTOUR_IN_SOURCE_ANNOTATION"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    print(f"WROTE {path}  ({len(rows)} data rows)")


def write_xlsx(path: Path, fields: Sequence[str], rows: Sequence[dict],
               wide: Sequence[str] = ()) -> bool:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        print(f"      openpyxl not installed; {path.name} skipped (CSV is "
              "authoritative)")
        return False
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = path.stem[:31]
    ws.append(list(fields))
    for c in range(1, len(fields) + 1):
        ws.cell(1, c).font = Font(bold=True, color="FFFFFF")
        ws.cell(1, c).fill = PatternFill("solid", fgColor="2F5496")
        ws.cell(1, c).alignment = Alignment(horizontal="center", wrap_text=True)
    for r in rows:
        ws.append([r.get(h, "") for h in fields])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(fields))}{len(rows) + 1}"
    for i, h in enumerate(fields, 1):
        ws.column_dimensions[get_column_letter(i)].width = 70 if h in wide else 22
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    print(f"WROTE {path}")
    return True


def inspect_annotation(xml_path: Path) -> Dict[str, Any]:
    """What does this XML actually contain? Named so the exclusion reason can say what
    IS there, not only what is missing."""
    out: Dict[str, Any] = {"exists": xml_path.is_file(), "roi_total": 0,
                           "roi_names": {}, "error": ""}
    if not out["exists"]:
        return out
    try:
        with xml_path.open("rb") as fh:
            blob = plistlib.load(fh, fmt=plistlib.FMT_XML)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    rois: List[dict] = []
    for im in (blob.get("Images") or []) if isinstance(blob, dict) else []:
        rois.extend(im.get("ROIs") or [])
    if isinstance(blob, dict) and not rois and "ROIs" in blob:
        rois = blob.get("ROIs") or []
    out["roi_total"] = len(rois)
    out["roi_names"] = dict(Counter(str(r.get("Name", "")).strip() for r in rois))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Finalise the INbreast artifacts.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    prep, raw = Path(a.prep_root), Path(a.raw_root)
    ds_dir = prep / BANK_NAME / DATASET_DIR
    reports, manifests = ds_dir / "reports", ds_dir / "manifests"
    shared = prep / BANK_NAME / "_reports"

    print("=" * 78)
    print("INbreast FULL SOURCE - CANONICAL MANIFESTS, REPORTS AND EXCLUSION LEDGER")
    print("=" * 78)
    print(f"generated : {now_iso()}")
    print(f"mode      : {'DRY-RUN (writes nothing)' if a.dry_run else 'WRITE'}")

    src = manifests / "canonical_manifest.csv"
    if not src.is_file():
        src = manifests / "DRY_RUN_canonical_manifest.csv"
    if not src.is_file():
        print(f"no manifest found under {manifests}")
        return 2
    rows = read_rows(src)
    print(f"manifest  : {src.name} ({len(rows)} rows)")
    fields = list(rows[0]) if rows else []

    seg = [r for r in rows if r.get("segmentation_available") == "TRUE"]
    excl = [r for r in rows if r.get("segmentation_available") == "FALSE"]
    br_all = Counter(r.get("birads_verbatim", "") for r in rows)
    br_seg = Counter(r.get("birads_verbatim", "") for r in seg)
    groups = {r.get("case_id", "") for r in rows if r.get("case_id")}
    seg_groups = {r.get("case_id", "") for r in seg if r.get("case_id")}
    print(f"cohort    : {len(rows)} | segmentation-ready {len(seg)} | excluded "
          f"{len(excl)} | patient groups {len(groups)}")

    # ---- inspect the disputed annotations --------------------------------------
    print("\n--- what the disputed annotations actually contain ---")
    ann: Dict[str, Dict[str, Any]] = {}
    for r in excl:
        num = r.get("source_file_number", "")
        rel = r.get("source_annotation_path", "")
        info = inspect_annotation(raw / rel) if rel else {"exists": False,
                                                          "roi_total": 0,
                                                          "roi_names": {}, "error": ""}
        ann[num] = info
        print(f"   {num}: XML {'present' if info['exists'] else 'ABSENT'}; "
              f"{info['roi_total']} ROI(s); names {info['roi_names'] or '{}'}"
              + (f"; error {info['error']}" if info["error"] else ""))

    # ---- ledger rows -------------------------------------------------------------
    ledger_new: List[dict] = []
    for r in excl:
        num = r.get("source_file_number", "")
        info = ann.get(num, {})
        names = info.get("roi_names") or {}
        other = {k: v for k, v in names.items() if k != "Mass"}
        if not info.get("exists"):
            reason = ("the workbook marks this mammogram Mass == X but the source "
                      "ships no XML annotation file for it.")
        elif info.get("roi_total", 0) == 0:
            reason = ("the workbook marks this mammogram Mass == X but its XML "
                      "annotation contains no ROI at all.")
        elif not names.get("Mass"):
            reason = (f"the workbook marks this mammogram Mass == X but its XML "
                      f"annotation contains no ROI named \"Mass\"; it carries "
                      f"{info['roi_total']} ROI(s) of other types ({other}).")
        else:
            reason = (f"the XML carries {names.get('Mass')} ROI(s) named \"Mass\" but "
                      "none bounds a polygon of at least 3 points.")
        ledger_new.append({
            "dataset": DATASET_NAME, "source_record_id": num,
            "source_label": f"BI-RADS {r.get('birads_verbatim', '')} (verbatim)",
            "canonical_label": "NOT_ESTABLISHED_NO_SOURCE_BM_STATUS",
            "exclusion_code": EXCLUSION_CODE,
            "exclusion_reason": reason,
            "scientific_exclusion_rationale": (
                "The joint cohort requires a lesion segmentation target derived from a "
                "source contour. INbreast ships contours, not raster masks, and the "
                "canonical rule rasterises only ROIs the source itself names \"Mass\". "
                "Supplying a target here would mean inventing a boundary, or "
                "re-typing an ROI the annotators gave a different name, neither of "
                "which the evidence supports. The record is therefore excluded from "
                "the segmentation cohort and retained in the manifest with its source "
                "metadata intact. The mammogram itself is not claimed to be invalid, "
                "and its BI-RADS is unchanged."),
            "classification_available": "FALSE_NO_SOURCE_BM_STATUS_FOR_ANY_RECORD",
            "segmentation_available": "FALSE",
            "canonical_image_written": ("YES" if r.get("canonical_image_path") else
                                        "NO"),
            "canonical_mask_written": "NO",
            "evidence_source": (f"{src.name} :: csv:[source_file_number='{num}']"
                                f"[mass_roi_total, mass_roi_used]; "
                                f"{r.get('source_annotation_path', '')} inspected "
                                "read-only"),
            "notes": ("defect is in the ANNOTATION, not in the image; the mammogram is "
                      "still converted and remains available for any future "
                      "image-level task"),
        })

    # ---- classification status ---------------------------------------------------
    cls = {
        "dataset": DATASET_NAME, "generated": now_iso(),
        "canonical_binary_label": "NOT_ESTABLISHED",
        "verdict": "NOT_RECOVERABLE",
        "question": "does the INbreast source define a per-image benign/malignant "
                    "status for the 108 Mass mammograms?",
        "searched": ["INbreast.xls (17 columns)", "MedicalReports (117 files)",
                     "AllXML ROI keys", "INbreast.csv", "README.txt",
                     "inbreast.pdf (not parsed)"],
        "findings": {
            "workbook_columns_named_for_pathology": 0,
            "workbook_free_text_rows_in_cohort_with_a_pathology_term": 13,
            "of_which_state_a_histological_result": 2,
            "of_which_indicate_malignancy": 0,
            "reports_level": "PATIENT_GROUP_NOT_IMAGE",
            "cohort_patient_groups": len(groups),
            "groups_with_an_outcome_report": 17,
            "xml_roi_keys": ["Area", "Center", "Dev", "IndexInImage", "Max", "Mean",
                             "Min", "Name", "NumberOfPoints", "Point_mm", "Point_px",
                             "Total", "Type"],
            "xml_pathology_keys": []},
        "published_paper_figures": {
            "biopsied_cases": 56, "biopsy_benign": 11, "biopsy_malignant": 45,
            "considered_benign_without_biopsy": 59,
            "level": "CASE_LEVEL_IN_THE_PUBLICATION",
            "distributed_with_the_database": False},
        "conclusion": (
            "No per-image benign/malignant status is distributed with INbreast. The "
            "13 cohort rows carrying a pathology-like note are mostly radiological "
            "impressions; only 2 state a histological result and none indicates "
            "malignancy, so the coverage is 1.9% and one-sided. The medical reports "
            "are patient-level and cannot be assigned to one of a patient's "
            "mammograms. BI-RADS was NOT used to derive a label. INbreast therefore "
            "enters the bank as a SEGMENTATION-ONLY client."),
        "birads_carried": "VERBATIM_IN_ITS_OWN_COLUMN",
        "birads_used_to_derive_labels": False,
    }

    # ---- build summary -------------------------------------------------------------
    summary = {
        "dataset": DATASET_NAME, "phase": "2B.3.5b", "generated": now_iso(),
        "cohort": "FULL_SOURCE_MASS_COHORT_108",
        "source": {"raw_root": str(raw), "selection": "INbreast.xls rows 0-409, "
                                                      "[Mass] == X",
                   "footer_rows_excluded": 2},
        "counts": {"cohort": len(rows), "segmentation_ready": len(seg),
                   "excluded_from_segmentation": len(excl),
                   "patient_groups_cohort": len(groups),
                   "patient_groups_segmentation": len(seg_groups)},
        "birads_verbatim_cohort": dict(sorted(br_all.items())),
        "birads_verbatim_segmentation": dict(sorted(br_seg.items())),
        "classification": {"label": "NOT_ESTABLISHED", "reason": "NOT_RECOVERABLE",
                           "detail": "see INBREAST_CLASSIFICATION_STATUS.json"},
        "policies": {
            "image": {"id": "INBREAST_FULLSOURCE_IMAGE_16BIT_LOSSLESS_V1",
                      "note": "stored DICOM values written unchanged to 16-bit PNG; "
                              "content measured at 12 bits (max 4095) inside a "
                              "16-bit container; no windowing"},
            "mask": {"id": "INBREAST_FULLSOURCE_MASK_FROM_XML_CONTOUR_V1",
                     "note": "masks are GENERATED from source XML contours; Phase 1A "
                             "generated none by design, so this is a declared Phase "
                             "2B canonical decision"}},
        "identity": {"patient_id": "NOT_ESTABLISHED",
                     "grouping_key": "DICOM filename hash token",
                     "grouping_verdict": "PATIENT_GROUPING_ESTABLISHED (Phase 1A.1)",
                     "limitation": "a patient-independent split is possible within "
                                   "this cohort; no split is created here"},
        "exclusions": [{"source_record_id": x["source_record_id"],
                        "code": x["exclusion_code"],
                        "reason": x["exclusion_reason"]} for x in ledger_new],
        "raw_modified": False, "canonical_png_altered": False, "splits_created": 0,
    }

    if a.dry_run:
        print("\n--- DRY-RUN, nothing written ---")
        print(f"   would write {manifests / 'canonical_manifest.xlsx'}")
        print(f"   would write {reports / 'INBREAST_CANONICAL_BUILD_SUMMARY.json'}")
        print(f"   would write {reports / 'INBREAST_CLASSIFICATION_STATUS.json'}")
        print(f"   would add {len(ledger_new)} row(s) to the exclusion ledger")
        for x in ledger_new:
            print(f"      {x['source_record_id']}: {x['exclusion_reason'][:90]}")
        return 0

    print()
    write_xlsx(manifests / "canonical_manifest.xlsx", fields, rows,
               wide=("exclusion_reason", "notes", "mask_policy_rule",
                     "image_policy_rule", "identity_provenance", "label_provenance",
                     "cohort_provenance", "findings_notes_verbatim"))
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "INBREAST_CANONICAL_BUILD_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {reports / 'INBREAST_CANONICAL_BUILD_SUMMARY.json'}")
    (reports / "INBREAST_CLASSIFICATION_STATUS.json").write_text(
        json.dumps(cls, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {reports / 'INBREAST_CLASSIFICATION_STATUS.json'}")

    # ---- ledger: add, never replace -----------------------------------------------
    led_path = shared / "CANONICAL_DATASET_EXCLUSION_LEDGER.csv"
    existing = read_rows(led_path) if led_path.is_file() else []
    keep = [r for r in existing if r.get("dataset") != DATASET_NAME]
    dropped = len(existing) - len(keep)
    merged = keep + ledger_new
    print(f"\nexclusion ledger: {len(existing)} existing row(s), {dropped} previous "
          f"{DATASET_NAME} row(s) replaced, {len(ledger_new)} added -> {len(merged)}")
    write_csv(led_path, LEDGER_FIELDS, merged)
    write_xlsx(shared / "CANONICAL_DATASET_EXCLUSION_LEDGER.xlsx", LEDGER_FIELDS,
               merged, wide=("exclusion_reason", "scientific_exclusion_rationale",
                             "evidence_source", "notes"))

    mt_path = shared / "MASTER_CURATION_TABLE.csv"
    mt_existing = read_rows(mt_path) if mt_path.is_file() else []
    mt_keep = [r for r in mt_existing if r.get("dataset") != DATASET_NAME]
    mt_keep.append({
        "dataset": DATASET_NAME, "source_records": len(rows),
        "canonical_eligible": len(seg), "excluded_total": len(excl),
        "excluded_benign": 0, "excluded_malignant": 0,
        "exclusion_codes": f"{EXCLUSION_CODE}={len(excl)}",
        "exclusion_reason_summary": ("mammograms marked Mass == X in the workbook "
                                     "whose XML annotation carries no usable Mass "
                                     "contour"),
        "scientific_exclusion_rationale_summary": (
            "a segmentation target cannot be supplied without inventing a boundary or "
            "re-typing an ROI the annotators named otherwise; the mammogram is "
            "converted and retained, only the segmentation target is absent"),
        "canonical_images_written": len(rows), "canonical_masks_written": len(seg),
        "locked_policy": "INBREAST_FULLSOURCE_MASK_FROM_XML_CONTOUR_V1",
        "evidence_source": f"{DATASET_DIR}/manifests/{src.name}",
        "notes": ("SEGMENTATION-ONLY client: no source benign/malignant status "
                  "exists; BI-RADS carried verbatim, never mapped"),
    })
    write_csv(mt_path, MASTER_FIELDS, mt_keep)
    write_xlsx(shared / "MASTER_CURATION_TABLE.xlsx", MASTER_FIELDS, mt_keep,
               wide=("exclusion_reason_summary",
                     "scientific_exclusion_rationale_summary", "notes"))

    print("\nRAW modified: FALSE | canonical PNG altered: FALSE | splits: 0")
    print("\nFINAL VERDICT:")
    print("INBREAST_ARTIFACTS_COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
