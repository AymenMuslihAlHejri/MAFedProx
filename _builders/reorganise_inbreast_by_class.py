#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reorganise_inbreast_by_class.py
===============================

INbreast FULL SOURCE - CLASS FOLDERS, VIEWABLE PREVIEWS, UPDATED WORKBOOKS

Reorganises the already-built canonical output into Benign/ and Malignant/, adds a
viewable 8-bit preview beside every canonical image, and refreshes the manifest, the
exclusion ledger and the master curation table.

    Benign/Images      35   16-bit canonical, pixel-identical to the DICOM
    Benign/Masks       35   binary {0,255}
    Benign/Preview     35   8-bit, FOR VIEWING ONLY
    Malignant/Images   71
    Malignant/Masks    71
    Malignant/Preview  71
    _excluded_from_segmentation/Images   2  (+ Preview)  retained, never deleted

THE LABEL IS AN ASSIGNMENT, NOT A SOURCE FACT
    INbreast distributes no per-image benign/malignant status; the audit of
    INbreast.xls, the 117 medical reports and the XML confirmed it. The class folders
    are therefore filled by a PRE-SPECIFIED assignment from source BI-RADS:
        BI-RADS 2, 3            -> Benign     (0)
        BI-RADS 4a, 4b, 4c, 5, 6 -> Malignant (1)
    Every record records label_provenance = BIRADS_DERIVED_ASSIGNMENT_NOT_SOURCE_
    PATHOLOGY, and BI-RADS stays verbatim in its own column. No model performance was
    used and the rule is fixed before any training.

WHY A SEPARATE PREVIEW RATHER THAN BRIGHTER IMAGES
    The canonical image holds the DICOM's stored values unchanged - content measured
    at 12 bits (max 4095) inside a 16-bit container - so a viewer that assumes a
    0-65535 range renders it almost black. The file is correct; the viewer is not.
    Rewriting it brighter would destroy the losslessness just verified pixel by pixel.
    So the canonical image is left untouched and a separate 8-bit preview is written:
        preview = round(value / 4095 * 255), clipped to [0, 255]
    One fixed divisor for every image, so previews stay comparable. Previews are
    DISPLAY ONLY and are marked as such in the manifest; they are never a training
    input and never enter the verifier's pixel checks.

Nothing is deleted. Canonical pixel content is not altered. H: is not touched.

Usage:
    python reorganise_inbreast_by_class.py --dry-run
    python reorganise_inbreast_by_class.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
BANK_NAME = "CANONICAL_DATASET_BANK"
DATASET_DIR = "INbreast_FullSource"
DATASET_NAME = "INbreast_FullSource"

BENIGN_BIRADS = {"2", "3"}
MALIGNANT_BIRADS = {"4a", "4b", "4c", "5", "6"}
LABEL_POLICY_ID = "INBREAST_BIRADS_ASSIGNMENT_V1"
LABEL_POLICY_RULE = ("BI-RADS 2,3 -> Benign (0); BI-RADS 4a,4b,4c,5,6 -> Malignant "
                     "(1); pre-specified, never tuned, never source pathology")
PREVIEW_DIVISOR = 4095
PREVIEW_LINEAR_RULE = (f"8-bit DISPLAY ONLY, linear: round(value / {PREVIEW_DIVISOR} "
                       "* 255), clipped; faithful to the full stored range but dim")
PREVIEW_STRETCH_RULE = ("8-bit DISPLAY ONLY, breast-range stretch: the 1st and 99th "
                        "percentile of the NON-ZERO pixels (the breast, excluding the "
                        "air background) are mapped to 0 and 255 and the result "
                        "clipped; per-image window, recorded per record")
PREVIEW_RULE = PREVIEW_LINEAR_RULE   # kept for the manifest column name
EXCLUDED_DIR = "_excluded_from_segmentation"
EXPECT = {"benign": 35, "malignant": 71, "excluded": 2}

NEW_COLS = ["canonical_binary_label", "class_folder", "label_policy_id",
            "label_policy_rule", "label_provenance",
            "canonical_preview_linear_path", "canonical_preview_stretched_path",
            "preview_linear_rule", "preview_stretched_rule",
            "preview_stretch_low", "preview_stretch_high",
            "preview_is_display_only",
            "mask_lesion_count", "mask_crosscheck_legacy_method"]
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


def legacy_mass_polygons(xml_path: Path) -> list:
    """Mass polygons built the way the user's existing loader builds them: each point
    string "(x, y)" is returned as (y, x). Used ONLY to cross-check our mask."""
    import plistlib
    with xml_path.open("rb") as fh:
        plist = plistlib.load(fh, fmt=plistlib.FMT_XML)
    img = (plist.get("Images") or [{}])[0]
    out = []
    for roi in (img.get("ROIs") or []):
        if str(roi.get("Name", "")).strip() != "Mass":
            continue
        pts = []
        for ps in (roi.get("Point_px") or []):
            try:
                x, y = tuple(float(n) for n in str(ps).strip("()").split(",")[:2])
            except ValueError:
                continue
            pts.append((y, x))
        if len(pts) > 2:
            out.append(pts)
    return out


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
    ap = argparse.ArgumentParser(description="Reorganise INbreast into class folders.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--raw-root",
                    default=r"H:\BC datasets\Mammo\01 INbreast 2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    prep = Path(a.prep_root)
    raw_root_for_xml = Path(a.raw_root) if a.raw_root else None
    bank = prep / BANK_NAME / DATASET_DIR
    shared = prep / BANK_NAME / "_reports"
    man_path = bank / "manifests" / "canonical_manifest.csv"

    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        sys.stderr.write(f"missing dependency: {exc}\n")
        return 3
    Image.MAX_IMAGE_PIXELS = None

    print("=" * 78)
    print("INbreast FULL SOURCE - CLASS FOLDERS, PREVIEWS, UPDATED WORKBOOKS")
    print("=" * 78)
    print(f"generated : {now_iso()}")
    print(f"mode      : {'DRY-RUN (writes nothing)' if a.dry_run else 'WRITE'}")
    print(f"label rule: {LABEL_POLICY_ID} -> {LABEL_POLICY_RULE}")
    print(f"preview   : {PREVIEW_RULE}")
    if not man_path.is_file():
        print(f"manifest not found: {man_path}")
        return 2
    rows = read_rows(man_path)
    fields = list(rows[0]) if rows else []
    for c in NEW_COLS:
        if c not in fields:
            fields.append(c)

    # ---- assign, plan the moves -------------------------------------------------
    plan: List[dict] = []
    bad_birads: List[str] = []
    for r in rows:
        num = r.get("source_file_number", "")
        br = r.get("birads_verbatim", "").strip()
        seg_ok = r.get("segmentation_available") == "TRUE"
        if br in BENIGN_BIRADS:
            cls, lab = "Benign", 0
        elif br in MALIGNANT_BIRADS:
            cls, lab = "Malignant", 1
        else:
            bad_birads.append(f"{num}: BI-RADS {br!r}")
            cls, lab = "", "UNRESOLVED"
        folder = cls if seg_ok else EXCLUDED_DIR
        stem = Path(r.get("canonical_image_path", "")).name
        mstem = Path(r.get("canonical_mask_path", "")).name if r.get(
            "canonical_mask_path") else ""
        plan.append({
            "row": r, "num": num, "birads": br, "class": cls, "label": lab,
            "folder": folder, "seg": seg_ok,
            "old_img": r.get("canonical_image_path", ""),
            "old_mask": r.get("canonical_mask_path", ""),
            "new_img": f"{folder}/Images/{stem}" if stem else "",
            "new_mask": f"{folder}/Masks/{mstem}" if (mstem and seg_ok) else "",
            "new_prev_lin": f"{folder}/Preview_linear/{stem}" if stem else "",
            "new_prev_str": f"{folder}/Preview_stretched/{stem}" if stem else ""})
    counts = Counter(p["class"] for p in plan if p["seg"])
    print(f"\ncohort {len(rows)}: segmentation-ready "
          f"{sum(1 for p in plan if p['seg'])}, excluded "
          f"{sum(1 for p in plan if not p['seg'])}")
    print(f"   Benign    {counts.get('Benign', 0)}  (BI-RADS "
          f"{sorted(BENIGN_BIRADS)})")
    print(f"   Malignant {counts.get('Malignant', 0)}  (BI-RADS "
          f"{sorted(MALIGNANT_BIRADS)})")
    for p in plan:
        if not p["seg"]:
            print(f"   excluded  {p['num']} BI-RADS {p['birads']} -> "
                  f"{EXCLUDED_DIR}/ (image kept, no mask)")
    if bad_birads:
        print(f"   UNMAPPED BI-RADS: {bad_birads}")
        print("   BLOCKED: every record must map to exactly one class")
        return 4
    if (counts.get("Benign") != EXPECT["benign"]
            or counts.get("Malignant") != EXPECT["malignant"]):
        print(f"   BLOCKED: expected {EXPECT['benign']} benign and "
              f"{EXPECT['malignant']} malignant, got {counts.get('Benign')} and "
              f"{counts.get('Malignant')}")
        return 4

    missing = [p["num"] for p in plan
               if p["old_img"] and not (bank / p["old_img"]).is_file()]
    if missing:
        print(f"   BLOCKED: {len(missing)} canonical image(s) not on disk: "
              f"{missing[:5]}")
        return 4

    if a.dry_run:
        print("\nDRY-RUN: would create")
        for c in ("Benign", "Malignant"):
            print(f"   {c}/Images, {c}/Masks, {c}/Preview_linear, "
                  f"{c}/Preview_stretched")
        print(f"   {EXCLUDED_DIR}/Images, {EXCLUDED_DIR}/Preview_linear, "
              f"{EXCLUDED_DIR}/Preview_stretched")
        print("   and refresh canonical_manifest.csv/.xlsx, the exclusion ledger and "
              "the master curation table")
        return 0

    # ---- move, then write the preview ------------------------------------------
    print("\nmoving canonical files into class folders and writing previews ...")
    moved = previews = 0
    for i, p in enumerate(plan, 1):
        src_img = bank / p["old_img"]
        dst_img = bank / p["new_img"]
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if src_img.resolve() != dst_img.resolve():
            shutil.move(str(src_img), str(dst_img))
            moved += 1
        if p["new_mask"]:
            src_m, dst_m = bank / p["old_mask"], bank / p["new_mask"]
            dst_m.parent.mkdir(parents=True, exist_ok=True)
            if src_m.is_file() and src_m.resolve() != dst_m.resolve():
                shutil.move(str(src_m), str(dst_m))
                moved += 1
        with Image.open(dst_img) as im:
            arr = np.asarray(im).astype(np.float64)
        # A. linear over the full stored range - faithful, dim
        lin = bank / p["new_prev_lin"]
        lin.parent.mkdir(parents=True, exist_ok=True)
        v8 = np.clip(np.rint(arr / PREVIEW_DIVISOR * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(v8, mode="L").save(lin, format="PNG")
        # B. stretch over the BREAST range only: the air background sits at 0 and
        #    would otherwise dominate the window and keep the tissue dark
        nz = arr[arr > 0]
        if nz.size:
            lo, hi = float(np.percentile(nz, 1)), float(np.percentile(nz, 99))
        else:
            lo, hi = 0.0, float(PREVIEW_DIVISOR)
        if hi <= lo:
            lo, hi = float(arr.min()), float(max(arr.max(), arr.min() + 1))
        strc = bank / p["new_prev_str"]
        strc.parent.mkdir(parents=True, exist_ok=True)
        v8s = np.clip(np.rint((arr - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(v8s, mode="L").save(strc, format="PNG")
        previews += 2
        # cross-check the mask against the legacy (y, x) construction: OsiriX stores
        # Point_px as "(x, y)" while an array is indexed [row, col], so a transposed
        # polygon is the classic failure. Building it the other way and comparing is
        # the only way to be sure ours is not the transposed one.
        lesions, xchk = "", ""
        if p["new_mask"]:
            xrel = p["row"].get("source_annotation_path", "").strip()
            xp = raw_root_for_xml / xrel if (raw_root_for_xml and xrel) else None
            if xp is not None and xp.is_file():
                try:
                    polys = legacy_mass_polygons(xp)
                    lesions = len(polys)
                    with Image.open(bank / p["new_mask"]) as mm:
                        ours = np.asarray(mm.convert("L"))
                    leg = Image.new("L", (ours.shape[1], ours.shape[0]), 0)
                    dr = ImageDraw.Draw(leg)
                    for pts in polys:                       # pts are (y, x) here
                        dr.polygon([(float(x), float(y)) for y, x in pts], fill=255)
                    lega = np.asarray(leg)
                    xchk = ("IDENTICAL" if np.array_equal(lega, ours)
                            else f"DIFFERS_{int(np.sum(lega != ours))}px")
                except Exception as exc:  # noqa: BLE001
                    xchk = f"ERROR_{type(exc).__name__}"
        r = p["row"]
        r["mask_lesion_count"] = lesions
        r["mask_crosscheck_legacy_method"] = xchk
        r["canonical_image_path"] = p["new_img"]
        r["canonical_mask_path"] = p["new_mask"]
        r["canonical_preview_linear_path"] = p["new_prev_lin"]
        r["canonical_preview_stretched_path"] = p["new_prev_str"]
        r["preview_stretch_low"] = int(lo)
        r["preview_stretch_high"] = int(hi)
        r["canonical_binary_label"] = p["label"] if p["seg"] else "NOT_IN_CLASS_COHORT"
        r["class_folder"] = p["folder"]
        r["label_policy_id"] = LABEL_POLICY_ID
        r["label_policy_rule"] = LABEL_POLICY_RULE
        r["label_provenance"] = ("BIRADS_DERIVED_ASSIGNMENT_NOT_SOURCE_PATHOLOGY; "
                                 "INbreast distributes no per-image benign/malignant "
                                 "status")
        r["preview_linear_rule"] = PREVIEW_LINEAR_RULE
        r["preview_stretched_rule"] = PREVIEW_STRETCH_RULE
        r["preview_is_display_only"] = "TRUE"
        r["classification_available"] = "TRUE_BY_ASSIGNMENT" if p["seg"] else "FALSE"
        if i % 25 == 0:
            print(f"   {i}/{len(plan)}")
    for d in (bank / "Images", bank / "Masks"):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    print(f"   moved {moved} file(s); wrote {previews} preview(s)")
    xc = Counter(r.get("mask_crosscheck_legacy_method", "") for r in rows
                 if r.get("mask_crosscheck_legacy_method"))
    print(f"   mask cross-check vs the legacy (y,x) construction: {dict(xc)}")
    if any(k.startswith("DIFFERS") for k in xc):
        print("   WARNING: our polygon and the legacy polygon disagree; one of the "
              "two transposes the contour. Inspect before using these masks.")
    les = Counter(str(r.get("mask_lesion_count", "")) for r in rows
                  if str(r.get("mask_lesion_count", "")).isdigit())
    print(f"   Mass lesions per mammogram: {dict(sorted(les.items()))}")

    on_disk = {c: {s: len(list((bank / c / s).glob("*.png")))
                   for s in ("Images", "Masks", "Preview_linear",
                             "Preview_stretched")
                   if (bank / c / s).is_dir()}
               for c in ("Benign", "Malignant", EXCLUDED_DIR)
               if (bank / c).is_dir()}
    print(f"   on disk: {json.dumps(on_disk)}")

    write_csv(man_path, fields, rows)
    write_xlsx(bank / "manifests" / "canonical_manifest.xlsx", fields, rows,
               wide=("exclusion_reason", "notes", "mask_policy_rule",
                     "image_policy_rule", "label_policy_rule", "label_provenance",
                     "identity_provenance", "cohort_provenance",
                     "preview_linear_rule", "preview_stretched_rule",
                     "findings_notes_verbatim"))

    # ---- ledger ------------------------------------------------------------------
    led = shared / "CANONICAL_DATASET_EXCLUSION_LEDGER.csv"
    existing = read_rows(led) if led.is_file() else []
    keep = [r for r in existing if r.get("dataset") != DATASET_NAME]
    new_rows = []
    for p in plan:
        if p["seg"]:
            continue
        r = p["row"]
        new_rows.append({
            "dataset": DATASET_NAME, "source_record_id": p["num"],
            "source_label": f"BI-RADS {p['birads']} (verbatim)",
            "canonical_label": "NOT_IN_CLASS_COHORT_NO_SEGMENTATION_TARGET",
            "exclusion_code": "EXCLUDE_NO_MASS_CONTOUR_IN_SOURCE_ANNOTATION",
            "exclusion_reason": r.get("exclusion_reason", ""),
            "scientific_exclusion_rationale": (
                "The joint cohort pairs each image with a lesion segmentation target "
                "derived from a source contour. The workbook marks this mammogram "
                "Mass == X, but its XML annotation carries only Calcification ROIs, "
                "so the source contradicts itself and no Mass boundary exists to "
                "rasterise. Supplying one would mean inventing a boundary or "
                "re-typing an ROI the annotators named otherwise. The record is "
                "excluded from the class cohort and its canonical image is retained "
                f"under {EXCLUDED_DIR}/ with full provenance. Neither the mammogram "
                "nor its BI-RADS is claimed to be invalid."),
            "classification_available": "FALSE",
            "segmentation_available": "FALSE",
            "canonical_image_written": "YES", "canonical_mask_written": "NO",
            "evidence_source": (f"{DATASET_DIR}/manifests/canonical_manifest.csv :: "
                                f"[source_file_number='{p['num']}']"
                                "[mass_roi_total=0]"),
            "notes": ("defect is in the ANNOTATION, not the image; the image and its "
                      "preview are kept and nothing was deleted")})
    merged = keep + new_rows
    write_csv(led, LEDGER_FIELDS, merged)
    write_xlsx(shared / "CANONICAL_DATASET_EXCLUSION_LEDGER.xlsx", LEDGER_FIELDS,
               merged, wide=("exclusion_reason", "scientific_exclusion_rationale",
                             "evidence_source", "notes"))

    mt = shared / "MASTER_CURATION_TABLE.csv"
    mt_rows = [r for r in (read_rows(mt) if mt.is_file() else [])
               if r.get("dataset") != DATASET_NAME]
    mt_rows.append({
        "dataset": DATASET_NAME, "source_records": len(rows),
        "canonical_eligible": counts.get("Benign", 0) + counts.get("Malignant", 0),
        "excluded_total": len(new_rows),
        "excluded_benign": sum(1 for p in plan
                               if not p["seg"] and p["class"] == "Benign"),
        "excluded_malignant": sum(1 for p in plan
                                  if not p["seg"] and p["class"] == "Malignant"),
        "exclusion_codes": f"EXCLUDE_NO_MASS_CONTOUR_IN_SOURCE_ANNOTATION={len(new_rows)}",
        "exclusion_reason_summary": ("marked Mass == X in the workbook but the XML "
                                     "annotation carries only Calcification ROIs"),
        "scientific_exclusion_rationale_summary": (
            "the source contradicts itself and no Mass contour exists to rasterise; "
            "the image is retained, only the segmentation target is absent"),
        "canonical_images_written": len(rows),
        "canonical_masks_written": sum(1 for p in plan if p["new_mask"]),
        "locked_policy": (f"{LABEL_POLICY_ID} + "
                          "INBREAST_FULLSOURCE_MASK_FROM_XML_CONTOUR_V1 + "
                          "INBREAST_FULLSOURCE_IMAGE_16BIT_LOSSLESS_V1"),
        "evidence_source": f"{DATASET_DIR}/manifests/canonical_manifest.csv",
        "notes": (f"class folders filled by {LABEL_POLICY_ID}, a pre-specified "
                  "BI-RADS assignment, NOT source pathology; 8-bit previews are "
                  "display-only")})
    write_csv(mt, MASTER_FIELDS, mt_rows)
    write_xlsx(shared / "MASTER_CURATION_TABLE.xlsx", MASTER_FIELDS, mt_rows,
               wide=("exclusion_reason_summary",
                     "scientific_exclusion_rationale_summary", "notes",
                     "locked_policy"))

    (bank / "reports" / "INBREAST_LABEL_ASSIGNMENT_POLICY.json").write_text(
        json.dumps({
            "dataset": DATASET_NAME, "generated": now_iso(),
            "policy_id": LABEL_POLICY_ID, "policy_rule": LABEL_POLICY_RULE,
            "source_defined_labels": False,
            "derived_from": "source BI-RADS carried verbatim",
            "model_performance_used": False, "fixed_before_training": True,
            "class_counts": {"Benign": counts.get("Benign", 0),
                             "Malignant": counts.get("Malignant", 0),
                             "excluded": len(new_rows)},
            "birads_to_class": {**{b: "Benign" for b in sorted(BENIGN_BIRADS)},
                                **{b: "Malignant" for b in sorted(MALIGNANT_BIRADS)}},
            "preview": {"linear_rule": PREVIEW_LINEAR_RULE,
                        "stretched_rule": PREVIEW_STRETCH_RULE,
                        "display_only": True,
                        "canonical_image_unchanged": True,
                        "note": "two renderings are written so the user can choose; "
                                "neither is a training input"},
            "mask_crosscheck_vs_legacy_method": dict(
                Counter(r.get("mask_crosscheck_legacy_method", "") for r in rows
                        if r.get("mask_crosscheck_legacy_method"))),
            "declaration_required_in_manuscript": (
                "INbreast binary labels are assigned from source BI-RADS by a "
                "pre-specified rule, not from pathology; the database distributes no "
                "per-image benign/malignant status, as confirmed by auditing "
                "INbreast.xls, the 117 medical reports and the XML annotations.")},
            indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {bank / 'reports' / 'INBREAST_LABEL_ASSIGNMENT_POLICY.json'}")

    print("\ncanonical pixel content altered: FALSE | files deleted: 0 | "
          "H: touched: FALSE")
    print("\nFINAL VERDICT:")
    print("INBREAST_CLASS_REORGANISATION_COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
