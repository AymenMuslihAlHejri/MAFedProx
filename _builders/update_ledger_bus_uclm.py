#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_ledger_bus_uclm.py
=========================

Adds the BUS-UCLM dimension-mismatch exclusions to CANONICAL_DATASET_EXCLUSION_LEDGER
and refreshes MASTER_CURATION_TABLE, using the same field conventions as the datasets
already recorded there.

    4 records excluded: HESN_002, HESN_003, HESN_004, HESN_005
    code EXCLUDE_MASK_IMAGE_DIMENSION_MISMATCH

It reads the facts from the canonical manifest the builder wrote, never from constants,
so the ledger cannot drift from what was actually built. Existing rows for other
datasets are preserved untouched; any previous BUS_UCLM rows are replaced so re-running
is safe.

The 419 Normal records are NOT written to the ledger. Normal is a class outside the
binary lesion target, not an exclusion, and its masks are legitimately all black
(EXPECTED_EMPTY_MASK_FOR_NORMAL_SAMPLE). They appear in the master table's own column.

Usage:
    python update_ledger_bus_uclm.py --dry-run
    python update_ledger_bus_uclm.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
DATASET_DIR = DATASET_NAME = "BUS_UCLM"
EXCLUSION_CODE = "EXCLUDE_MASK_IMAGE_DIMENSION_MISMATCH"
NORMAL_STATUS = "NOT_IN_BINARY_TARGET_NORMAL_CLASS"
EXPECT = {"records": 683, "normal": 419, "selected": 264, "excluded": 4,
          "cohort": 260, "benign": 170, "malignant": 90}

LEDGER_FIELDS = [
    "dataset", "source_record_id", "source_label", "canonical_label",
    "exclusion_code", "exclusion_reason", "scientific_exclusion_rationale",
    "classification_available", "segmentation_available",
    "canonical_image_written", "canonical_mask_written", "evidence_source", "notes"]
MASTER_FIELDS = [
    "dataset", "source_records", "canonical_eligible", "excluded_total",
    "excluded_benign", "excluded_malignant", "exclusion_codes",
    "exclusion_reason_summary", "scientific_exclusion_rationale_summary",
    "canonical_images_written", "canonical_masks_written",
    "locked_policy", "evidence_source", "notes"]

RATIONALE = (
    "The joint cohort pairs every image with a segmentation target drawn on that same "
    "image. Here the paired mask has different pixel dimensions from its image, so it "
    "does not locate the lesion on the image it belongs to. The official INFO.csv "
    "declares one resolution for all 683 records, and these four contradict it, so the "
    "disagreement is in the source metadata rather than in the conversion. Aligning "
    "them would require an unverified resize or crop, which this phase prohibits, and "
    "any such correction would move the annotated boundary. The records are therefore "
    "excluded from the canonical cohort and retained in the manifest with their source "
    "provenance and the observed dimensions. Neither the images nor their official "
    "labels are claimed to be invalid.")


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
    print(f"WROTE {path}  ({len(rows)} rows)")


def write_xlsx(path: Path, fields: Sequence[str], rows: Sequence[dict],
               wide: Sequence[str] = ()) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        print(f"      openpyxl missing; {path.name} skipped")
        return
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Update the ledger for BUS-UCLM.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    prep = Path(a.prep_root)
    bank = prep / BANK_NAME / DATASET_DIR
    shared = prep / BANK_NAME / "_reports"
    man = bank / "manifests" / "canonical_manifest.csv"

    print("=" * 78)
    print("BUS-UCLM - EXCLUSION LEDGER AND MASTER CURATION TABLE")
    print("=" * 78)
    print(f"generated : {now_iso()}")
    print(f"mode      : {'DRY-RUN (writes nothing)' if a.dry_run else 'WRITE'}")
    if not man.is_file():
        print(f"manifest not found: {man}")
        return 2
    rows = read_rows(man)
    excl = [r for r in rows if r.get("exclusion_status") == EXCLUSION_CODE]
    normals = [r for r in rows if r.get("exclusion_status") == NORMAL_STATUS]
    cohort = [r for r in rows
              if r.get("exclusion_status") == "INCLUDED_BINARY_LESION_COHORT"]
    by = Counter(r.get("source_label") for r in cohort)
    print(f"manifest  : {len(rows)} records | cohort {len(cohort)} "
          f"({by.get('Benign', 0)} benign / {by.get('Malignant', 0)} malignant) | "
          f"excluded {len(excl)} | normal {len(normals)}")

    problems = []
    if len(rows) != EXPECT["records"]:
        problems.append(f"records {len(rows)} != {EXPECT['records']}")
    if len(excl) != EXPECT["excluded"]:
        problems.append(f"excluded {len(excl)} != {EXPECT['excluded']}")
    if len(cohort) != EXPECT["cohort"]:
        problems.append(f"cohort {len(cohort)} != {EXPECT['cohort']}")
    if by.get("Benign") != EXPECT["benign"] or by.get("Malignant") != EXPECT["malignant"]:
        problems.append(f"class split {dict(by)} != "
                        f"{EXPECT['benign']}/{EXPECT['malignant']}")
    if len(normals) != EXPECT["normal"]:
        problems.append(f"normal {len(normals)} != {EXPECT['normal']}")
    if problems:
        print("\nBLOCKED - the manifest does not match the approved build:")
        for p in problems:
            print(f"   {p}")
        return 4

    print("\n--- exclusion rows to add ---")
    new_rows: List[dict] = []
    for r in excl:
        sid = r.get("sample_id", "")
        iw, ih = r.get("source_image_width", ""), r.get("source_image_height", "")
        mw, mh = r.get("source_mask_width", ""), r.get("source_mask_height", "")
        reason = (f"the source image is {iw}x{ih} while its paired mask is {mw}x{mh}; "
                  f"the official INFO.csv declares "
                  f"{r.get('resolution_metadata', '')} for this record, so the "
                  "metadata and the pixels disagree")
        print(f"   {sid} ({r.get('source_label')}, patient {r.get('patient_id')}): "
              f"image {iw}x{ih} vs mask {mw}x{mh}")
        new_rows.append({
            "dataset": DATASET_NAME, "source_record_id": sid,
            "source_label": r.get("source_label", ""),
            "canonical_label": "NOT_IN_BINARY_TARGET_NO_USABLE_SEGMENTATION_TARGET",
            "exclusion_code": EXCLUSION_CODE,
            "exclusion_reason": reason,
            "scientific_exclusion_rationale": RATIONALE,
            "classification_available": "FALSE",
            "segmentation_available": "FALSE",
            "canonical_image_written": "NO", "canonical_mask_written": "NO",
            "evidence_source": (f"{DATASET_DIR}/manifests/canonical_manifest.csv :: "
                                f"[sample_id='{sid}']"
                                "[source_image_width, source_image_height, "
                                "source_mask_width, source_mask_height, "
                                "mask_dimension_match_verified_here]; dimensions read "
                                "from the pixels because the audit's dimension_match "
                                "column was empty on all 683 rows"),
            "notes": (f"patient {r.get('patient_id', '')}; the defect is a "
                      "mask/image geometry disagreement in the source, not a defect of "
                      "the image or its label")})

    led = shared / "CANONICAL_DATASET_EXCLUSION_LEDGER.csv"
    existing = read_rows(led) if led.is_file() else []
    keep = [x for x in existing if x.get("dataset") != DATASET_NAME]
    replaced = len(existing) - len(keep)
    merged = keep + new_rows
    print(f"\nledger: {len(existing)} existing row(s), {replaced} previous "
          f"{DATASET_NAME} row(s) replaced, {len(new_rows)} added -> {len(merged)}")
    print(f"        by dataset: {dict(Counter(x.get('dataset') for x in merged))}")

    mt = shared / "MASTER_CURATION_TABLE.csv"
    mt_existing = read_rows(mt) if mt.is_file() else []
    mt_keep = [x for x in mt_existing if x.get("dataset") != DATASET_NAME]
    mt_row = {
        "dataset": DATASET_NAME, "source_records": len(rows),
        "canonical_eligible": len(cohort), "excluded_total": len(excl),
        "excluded_benign": sum(1 for r in excl if r.get("source_label") == "Benign"),
        "excluded_malignant": sum(1 for r in excl
                                  if r.get("source_label") == "Malignant"),
        "exclusion_codes": f"{EXCLUSION_CODE}={len(excl)}",
        "exclusion_reason_summary": ("mask and image pixel dimensions disagree, "
                                     "contradicting the single resolution the official "
                                     "INFO.csv declares for every record"),
        "scientific_exclusion_rationale_summary": (
            "a mask of different dimensions does not locate the lesion on its image, "
            "and aligning them would require an unverified resize or crop that would "
            "move the annotated boundary"),
        "canonical_images_written": len(cohort),
        "canonical_masks_written": len(cohort),
        "locked_policy": ("BUS_UCLM_CANONICAL_MASK_COLOUR_TO_BINARY_V1 + "
                          "BUS_UCLM_CANONICAL_IMAGE_LOSSLESS_V1"),
        "evidence_source": f"{DATASET_DIR}/manifests/canonical_manifest.csv",
        "notes": (f"{len(normals)} Normal records are a class outside the binary "
                  "target, retained in the manifest with no canonical path and NOT "
                  "counted as exclusions; labels come from the official INFO.csv and "
                  "the colour-coded mask was only cross-checked against them")}
    mt_keep.append(mt_row)

    if a.dry_run:
        print("\nDRY-RUN: nothing written")
        print(f"   would write {led} ({len(merged)} rows) and its .xlsx")
        print(f"   would write {mt} ({len(mt_keep)} rows) and its .xlsx")
        return 0

    write_csv(led, LEDGER_FIELDS, merged)
    write_xlsx(shared / "CANONICAL_DATASET_EXCLUSION_LEDGER.xlsx", LEDGER_FIELDS,
               merged, wide=("exclusion_reason", "scientific_exclusion_rationale",
                             "evidence_source", "notes"))
    write_csv(mt, MASTER_FIELDS, mt_keep)
    write_xlsx(shared / "MASTER_CURATION_TABLE.xlsx", MASTER_FIELDS, mt_keep,
               wide=("exclusion_reason_summary",
                     "scientific_exclusion_rationale_summary", "notes",
                     "locked_policy"))
    print("\ncanonical files touched: NONE | RAW touched: NONE")
    print("\nFINAL VERDICT:")
    print("BUS_UCLM_LEDGER_UPDATED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
