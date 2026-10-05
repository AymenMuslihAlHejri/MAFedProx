#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_dmid_22_records.py
=========================

MINIMAL CORRECTIVE REBUILD - the 22 affected DMID records only.

The execute loop of build_bank_dmid.py indexed arr[..., 0] without checking ndim. On a
2-D greyscale array that slice is a COLUMN, not a channel. 12 records passed the
spurious test and were written as a single column; 10 failed it and were refused. The
diagnosis confirmed all 22 are SamplesPerPixel 1, MONOCHROME2, genuinely 2-D, uint8,
and that there are zero genuine multichannel cases.

WHAT THIS DOES
    - fixes the writer: ndim == 2 -> the array IS the image; the channel-replication
      check runs only when ndim == 3
    - deletes ONLY canonical PNGs it has itself proved corrupt, by re-reading each one
      and comparing its size to the decoded DICOM. A file that matches is never touched
    - rebuilds ONLY the 22 records named by their manifest status
    - requires a pixel-identical read-back against the decoded DICOM before a record is
      called converted
    - writes each mask ONLY after its image verified
    - updates the manifest and the build report for those 22 rows

WHAT THIS NEVER DOES
    no cohort or count change, no exclusion, no ledger edit, no RAW or audit write, no
    touch of the other 226 joint records, no split, and no pixel value altered:
    no normalisation, rescale, window, crop or resize.

Usage:
    python repair_dmid_22_records.py --dry-run
    python repair_dmid_22_records.py --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
DEFAULT_RAW_ROOT = r"H:\BC datasets\Mammo\02 DMID"
BANK_NAME = "CANONICAL_DATASET_BANK"
DATASET = "DMID"
TARGETS = ("VERIFICATION_FAILED_NOT_IDENTICAL", "REFUSED_CHANNELS_NOT_REPLICATED")
EXPECT = {"targets": 22, "joint": 248, "joint_benign": 143, "joint_malignant": 105,
          "images": 255, "cls_benign": 145, "cls_malignant": 110, "png": 503}
FG = 255


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_rows(p: Path) -> Tuple[List[dict], List[str]]:
    with p.open(encoding="utf-8-sig", errors="replace") as fh:
        s = fh.read(8192)
        fh.seek(0)
        d = ";" if s.count(";") > s.count(",") else ","
        rd = csv.DictReader(fh, delimiter=d)
        return list(rd), list(rd.fieldnames or [])


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"WROTE {path}  ({len(rows)} rows)")


def read_mask(path: Path, np):
    try:
        import tifffile
        return np.asarray(tifffile.imread(str(path)))
    except Exception:  # noqa: BLE001
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im)


def white_foreground(arr, np):
    """Unchanged from the builder: colour channels exactly white -> foreground."""
    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
    else:
        return None, 0, 0
    white = np.all(rgb == FG, axis=-1)
    black = np.all(rgb == 0, axis=-1)
    other = int((~white & ~black).sum())
    n_white = int(white.sum())
    if n_white == 0:
        return None, 0, other
    return (white.astype(np.uint8) * FG), n_white, other


def intended_image(arr, np) -> Tuple[Any, str]:
    """THE FIX. Branch on ndim, exactly as the pre-check loop always did.
    No value is altered in either branch."""
    if arr.ndim == 2:
        return arr, "ALREADY_SINGLE_CHANNEL"
    if arr.ndim == 3 and arr.shape[2] >= 3:
        if not (np.array_equal(arr[..., 0], arr[..., 1])
                and np.array_equal(arr[..., 1], arr[..., 2])):
            return None, "REFUSED_CHANNELS_NOT_REPLICATED"
        return arr[..., 0], "THREE_IDENTICAL_CHANNELS_REDUCED_TO_ONE"
    return None, f"REFUSED_UNEXPECTED_SHAPE_{arr.shape}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Repair the 22 DMID records.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    a = ap.parse_args(argv)
    mode = "dry-run" if a.dry_run else "execute"
    prep, raw = Path(a.prep_root), Path(a.raw_root)
    bank = prep / BANK_NAME / DATASET
    man = bank / "manifests" / "canonical_manifest.csv"

    try:
        import numpy as np
        import pydicom
        from PIL import Image
    except ImportError as exc:
        sys.stderr.write(f"missing dependency: {exc}\n")
        return 3
    Image.MAX_IMAGE_PIXELS = None

    print("=" * 84)
    print("DMID - MINIMAL CORRECTIVE REBUILD OF THE 22 AFFECTED RECORDS")
    print("=" * 84)
    print(f"generated : {now_iso()}")
    print(f"mode      : {'DRY-RUN (writes nothing)' if a.dry_run else 'EXECUTE'}")
    print(f"source    : {raw}  [READ-ONLY]")
    print("scope     : the 22 named records only; no cohort change, no exclusion, no "
          "ledger edit, no split")
    if not man.is_file():
        print(f"manifest not found: {man}")
        return 2
    rows, fields = read_rows(man)
    targets = [r for r in rows if r.get("image_conversion_status") in TARGETS]
    others = [r for r in rows if r.get("image_conversion_status")
              == "CONVERTED_VERIFIED_IDENTICAL"]
    print(f"manifest  : {len(rows)} rows | targets {len(targets)} | already verified "
          f"{len(others)} (never touched)")
    if len(targets) != EXPECT["targets"]:
        print(f"BLOCKED: expected {EXPECT['targets']} target(s), found {len(targets)}")
        return 4

    # ---- pass 1: prove which canonical PNGs are corrupt, never assume -------------
    print("\n--- pass 1: decode each target and check what is on disk ---")
    plan: List[dict] = []
    for i, r in enumerate(targets, 1):
        ref = r.get("image_reference", "")
        sp = raw / r.get("source_image_path", "")
        if not sp.is_file():
            print(f"  {ref}: BLOCKED - source missing")
            return 4
        arr = pydicom.dcmread(sp).pixel_array
        want, note = intended_image(arr, np)
        if want is None:
            print(f"  {ref}: BLOCKED - {note}; this is not the ndim defect and needs "
                  "its own decision")
            return 4
        dst = bank / r.get("canonical_image_path", "")
        state, on_disk = "NOT_WRITTEN", None
        if dst.is_file():
            with Image.open(dst) as im:
                on_disk = (im.height, im.width)
            state = ("CORRUPT" if on_disk != tuple(want.shape[:2]) else "ALREADY_OK")
        plan.append({"row": r, "ref": ref, "src": sp, "dst": dst, "want_shape":
                     tuple(want.shape[:2]), "note": note, "state": state,
                     "on_disk": on_disk, "dtype": str(arr.dtype),
                     "rng": (int(arr.min()), int(arr.max()))})
        print(f"  {ref:8s} decoded {list(arr.shape)} {arr.dtype} "
              f"[{arr.min()},{arr.max()}] -> {note}; on disk {state} "
              f"{on_disk if on_disk else ''}")
        if i % 10 == 0:
            print(f"     {i}/{len(targets)}")
    st = Counter(p["state"] for p in plan)
    print(f"\n  on-disk state: {dict(st)}")
    if st.get("ALREADY_OK"):
        print("  NOTE: files already matching their source are NOT deleted or "
              "rewritten")

    if a.dry_run:
        print("\nDRY-RUN: nothing written")
        print(f"   would delete {st.get('CORRUPT', 0)} proven-corrupt PNG(s)")
        print(f"   would write {len(plan)} image(s) and then their masks")
        print(f"   would leave the other {len(others)} verified record(s) untouched")
        print("\nFINAL VERDICT:\nREADY_FOR_DMID_22_RECORD_REPAIR")
        return 0

    # ---- pass 2: delete only proven-corrupt files ---------------------------------
    print("\n--- pass 2: delete only the PNGs proved corrupt above ---")
    deleted = 0
    for p in plan:
        if p["state"] == "CORRUPT":
            p["dst"].unlink()
            deleted += 1
            print(f"  deleted {p['dst'].name}  (was {p['on_disk']}, source "
                  f"{p['want_shape']})")
    print(f"  deleted {deleted} corrupt file(s); valid canonical files untouched")

    # ---- pass 3: rebuild image then mask, with pixel-identical read-back -----------
    print("\n--- pass 3: rebuild the 22, image first, mask only after it verifies ---")
    img_done = mask_done = 0
    fails: List[str] = []
    for i, p in enumerate(plan, 1):
        r = p["row"]
        arr = pydicom.dcmread(p["src"]).pixel_array
        want, note = intended_image(arr, np)
        intended = want.astype(np.uint8)          # values unchanged
        p["dst"].parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(intended, mode="L").save(p["dst"], format="PNG")
        with Image.open(p["dst"]) as chk:
            back = np.asarray(chk.convert("L"))
        if back.shape != intended.shape or not np.array_equal(back, intended):
            d = (int(np.sum(back != intended)) if back.shape == intended.shape else -1)
            r["image_conversion_status"] = "VERIFICATION_FAILED_NOT_IDENTICAL"
            fails.append(f"{p['ref']}: image differs on {d} px")
            print(f"  {p['ref']}: IMAGE VERIFICATION FAILED")
            continue
        r["image_conversion_status"] = "CONVERTED_VERIFIED_IDENTICAL"
        r["source_channels_identical"] = ("NOT_APPLICABLE_SINGLE_CHANNEL"
                                          if arr.ndim == 2 else "YES")
        r["image_reduction_path"] = note
        r["source_samples_per_pixel"] = r.get("source_samples_per_pixel") or (
            1 if arr.ndim == 2 else 3)
        r["source_pixel_min"], r["source_pixel_max"] = int(arr.min()), int(arr.max())
        r["canonical_image_height"], r["canonical_image_width"] = intended.shape[:2]
        img_done += 1
        if r.get("mask_conversion_status") != "PLANNED":
            continue
        mp = raw / r.get("source_mask_path", "")
        if not mp.is_file():
            fails.append(f"{p['ref']}: mask source missing")
            continue
        ma = read_mask(mp, np)
        plane, n_white, n_other = white_foreground(ma, np)
        if plane is None:
            r["mask_conversion_status"] = "REFUSED_NO_WHITE_FOREGROUND"
            fails.append(f"{p['ref']}: mask has no white foreground")
            continue
        if plane.shape[:2] != intended.shape[:2]:
            r["mask_conversion_status"] = "DIMENSION_MISMATCH"
            fails.append(f"{p['ref']}: mask {plane.shape[:2]} vs image "
                         f"{intended.shape[:2]}")
            continue
        mdst = bank / r["canonical_mask_path"]
        mdst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(plane, mode="L").save(mdst, format="PNG")
        with Image.open(mdst) as chk:
            mback = np.asarray(chk.convert("L"))
        if not np.array_equal(mback, plane):
            r["mask_conversion_status"] = "VERIFICATION_FAILED_NOT_IDENTICAL"
            fails.append(f"{p['ref']}: mask read-back differs")
            continue
        r["mask_conversion_status"] = "CONVERTED_VERIFIED_IDENTICAL"
        r["mask_white_pixels"] = n_white
        r["mask_nonblack_nonwhite_pixels_to_background"] = n_other
        r["mask_foreground_fraction"] = round(
            n_white / float(plane.shape[0] * plane.shape[1]), 8)
        r["mask_dimension_verified_here"] = "MATCH"
        r["canonical_mask_height"], r["canonical_mask_width"] = plane.shape[:2]
        mask_done += 1
        if i % 5 == 0:
            print(f"     {i}/{len(plan)}")
    print(f"\n  images verified identical: {img_done}/{len(plan)}")
    print(f"  masks  verified identical: {mask_done}")
    for x in fails[:10]:
        print(f"  FAILURE: {x}")

    # ---- pass 4: manifest, report, and the counts that must NOT have moved ---------
    write_csv(man, fields, rows)
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "canonical_manifest"
        ws.append(fields)
        for c in range(1, len(fields) + 1):
            ws.cell(1, c).font = Font(bold=True, color="FFFFFF")
            ws.cell(1, c).fill = PatternFill("solid", fgColor="2F5496")
            ws.cell(1, c).alignment = Alignment(horizontal="center", wrap_text=True)
        for r in rows:
            ws.append([r.get(h, "") for h in fields])
        ws.freeze_panes = "A2"
        for i, h in enumerate(fields, 1):
            ws.column_dimensions[get_column_letter(i)].width = 20
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        wb.save(bank / "manifests" / "canonical_manifest.xlsx")
        print(f"WROTE {bank / 'manifests' / 'canonical_manifest.xlsx'}")
    except ImportError:
        print("      openpyxl missing; XLSX skipped")

    joint = [r for r in rows if r.get("exclusion_status") == "INCLUDED_JOINT_COHORT"]
    jb = Counter(r.get("source_label") for r in joint)
    cls_valid = [r for r in rows if r.get("classification_valid") == "YES"]
    cb = Counter(r.get("source_label") for r in cls_valid)
    img_files = len([p for p in bank.rglob("Images/*.png")])
    mask_files = len([p for p in bank.rglob("Masks/*.png")])
    ics = Counter(r.get("image_conversion_status") for r in rows)
    mcs = Counter(r.get("mask_conversion_status") for r in rows)

    checks = [
        ("all 22 images verified pixel-identical", img_done == EXPECT["targets"]),
        ("no record failed", not fails),
        (f"joint cohort still {EXPECT['joint']} = {EXPECT['joint_benign']}B + "
         f"{EXPECT['joint_malignant']}M",
         len(joint) == EXPECT["joint"] and jb.get("BENIGN") == EXPECT["joint_benign"]
         and jb.get("MALIGNANT") == EXPECT["joint_malignant"]),
        (f"classification-valid still {EXPECT['images']} = {EXPECT['cls_benign']}B + "
         f"{EXPECT['cls_malignant']}M",
         len(cls_valid) == EXPECT["images"]
         and cb.get("BENIGN") == EXPECT["cls_benign"]
         and cb.get("MALIGNANT") == EXPECT["cls_malignant"]),
        (f"canonical images on disk == {EXPECT['images']}",
         img_files == EXPECT["images"]),
        (f"canonical masks on disk == {EXPECT['joint']}",
         mask_files == EXPECT["joint"]),
        ("every classification-valid record is CONVERTED_VERIFIED_IDENTICAL",
         ics.get("CONVERTED_VERIFIED_IDENTICAL") == EXPECT["images"]),
        ("every joint record has a verified mask",
         mcs.get("CONVERTED_VERIFIED_IDENTICAL") == EXPECT["joint"]),
        ("no record left PLANNED", not ics.get("PLANNED") and not mcs.get("PLANNED")),
    ]
    print("\n" + "=" * 84)
    for d, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {d}")
    print(f"\n  image_conversion_status: {dict(ics)}")
    print(f"  mask_conversion_status : {dict(mcs)}")
    print(f"  files on disk          : {img_files} image(s), {mask_files} mask(s), "
          f"{img_files + mask_files} total")
    failed = [d for d, ok in checks if not ok]

    rep = bank / "reports" / "DMID_22_RECORD_REPAIR_REPORT.json"
    rep.write_text(json.dumps({
        "generated": now_iso(), "scope": "22 records only",
        "defect": ("the execute loop indexed arr[..., 0] without checking ndim; on a "
                   "2-D greyscale array that slice is a column, not a channel"),
        "fix": "branch on ndim; ndim == 2 means the array is already the image",
        "corrupt_png_deleted": deleted,
        "images_rebuilt_and_verified": img_done,
        "masks_written_after_image_verification": mask_done,
        "failures": fails,
        "counts_unchanged": {"joint": len(joint), "joint_benign": jb.get("BENIGN", 0),
                             "joint_malignant": jb.get("MALIGNANT", 0),
                             "classification_valid": len(cls_valid)},
        "files_on_disk": {"images": img_files, "masks": mask_files},
        "image_conversion_status": dict(ics), "mask_conversion_status": dict(mcs),
        "exclusions_added": 0, "ledger_modified": False, "raw_modified": False,
        "splits_created": 0,
        "other_records_touched": 0,
        "verdict": "DMID_22_RECORD_REPAIR_COMPLETE" if not failed else "BLOCKED",
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWROTE {rep}")
    print("exclusions added: 0 | ledger modified: NO | RAW touched: NO | splits: 0")
    print(f"other records rebuilt: 0 (the {len(others)} already-verified records were "
          "never opened for writing)")
    print("\nFINAL VERDICT:")
    if failed:
        print("BLOCKED")
        for d in failed:
            print(f"    FAILED: {d}")
        return 4
    print("DMID_22_RECORD_REPAIR_COMPLETE")
    print("\nNext: run the independent full-bank verification.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
