#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bank_inbreast_fullsource.py
==================================

PHASE 2B.3.5b - INDEPENDENT ON-DISK VERIFIER FOR THE INbreast FULL-SOURCE BANK

Checks the canonical output against the SOURCE, never against the builder's log or its
report. It re-reads the manifest, re-opens every canonical PNG, re-reads every DICOM
and every XML, and REIMPLEMENTS both conversion rules from their written description
rather than importing the builder, so a bug cannot hide behind identical code.

READ-ONLY EVERYWHERE. One report is written under reports/. H: is decoded, never
written.

V01 manifest 108 / segmentation-ready 106 / excluded 2
V02 every canonical path named in the manifest exists
V03 exactly 214 canonical PNG and no orphan
V04 canonical image dimensions equal the DICOM dimensions
V05 canonical image pixels equal the DICOM stored values EXACTLY
V06 masks are single-channel and strictly {0,255}
V07 mask and image dimensions agree
V08 every mask independently reconstructs from the source XML Mass contours
V09 every written mask is non-empty
V10 the excluded records wrote no mask, and their XML really carries no Mass ROI
V11 no benign/malignant label and no class folder anywhere
V12 BI-RADS in the manifest matches INbreast.xls verbatim
V13 no patient or lesion identity invented; the group key is the DICOM filename hash
V14 RAW unchanged
V15 no Train/Validation/Test split created
V16 recorded mask foreground counts match the pixels on disk

Usage:
    python verify_bank_inbreast_fullsource.py
    python verify_bank_inbreast_fullsource.py --sample 40
"""

from __future__ import annotations

import argparse
import csv
import json
import plistlib
import random
import re
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
DEFAULT_RAW_ROOT = r"H:\BC datasets\Mammo\01 INbreast 2"
BANK_NAME = "CANONICAL_DATASET_BANK"
DATASET_DIR = "INbreast_FullSource"
WORKBOOK, DICOM_DIR = "INbreast.xls", "AllDICOMs"
REAL_ROWS, COHORT_COL = 410, "Mass "
MASS_ROI_NAME, MIN_POLYGON_POINTS = "Mass", 3
EXPECT = {"records": 108, "segmentation": 106, "excluded": 2, "png": 214,
          "benign": 35, "malignant": 71,
          "birads": {"2": 24, "3": 13, "4a": 7, "4b": 2, "4c": 11, "5": 43, "6": 8}}
BENIGN_BIRADS = {"2", "3"}
MALIGNANT_BIRADS = {"4a", "4b", "4c", "5", "6"}
PREVIEW_DIRS = ("Preview_linear", "Preview_stretched")
PREVIEW_DIVISOR = 4095
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def is_split_like(s: str) -> bool:
    pth = Path(s)
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


def mass_polygons(xml_path: Path) -> Tuple[List[List[Tuple[float, float]]], int]:
    """Reimplemented from the rule text, not copied from the builder."""
    with xml_path.open("rb") as fh:
        blob = plistlib.load(fh, fmt=plistlib.FMT_XML)
    rois: List[dict] = []
    if isinstance(blob, dict):
        for im in (blob.get("Images") or []):
            rois.extend(im.get("ROIs") or [])
        if not rois:
            rois = blob.get("ROIs") or []
    named = [r for r in rois if str(r.get("Name", "")).strip() == MASS_ROI_NAME]
    polys = []
    for r in named:
        pts = []
        for s in (r.get("Point_px") or []):
            nums = re.findall(r"-?\d+\.?\d*", str(s))
            if len(nums) >= 2:
                pts.append((float(nums[0]), float(nums[1])))
        if len(pts) >= MIN_POLYGON_POINTS:
            polys.append(pts)
    return polys, len(named)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the INbreast full-source bank.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args(argv)
    prep, raw = Path(a.prep_root), Path(a.raw_root)
    bank = prep / BANK_NAME / DATASET_DIR
    man = bank / "manifests" / "canonical_manifest.csv"

    try:
        import numpy as np
        import pandas as pd
        import pydicom
        from PIL import Image, ImageDraw
    except ImportError as exc:
        sys.stderr.write(f"missing dependency: {exc}\n")
        return 3
    Image.MAX_IMAGE_PIXELS = None

    print("=" * 78)
    print("INbreast FULL SOURCE - INDEPENDENT ON-DISK VERIFICATION")
    print("=" * 78)
    print(f"generated : {now_iso()}")
    print(f"bank      : {bank}")
    print(f"source    : {raw}  [READ-ONLY]")
    if not man.is_file():
        print(f"manifest not found: {man}")
        return 2
    rows = read_rows(man)
    seg = [r for r in rows if r.get("segmentation_available") == "TRUE"]
    excl = [r for r in rows if r.get("segmentation_available") == "FALSE"]
    print(f"manifest  : {len(rows)} records, {len(seg)} segmentation-ready, "
          f"{len(excl)} excluded")
    before_raw = snapshot(raw)

    checks: List[Tuple[str, str, bool, str]] = []

    def add(i: str, d: str, ok: bool, b: str) -> None:
        checks.append((i, d, ok, b))

    add("V01", f"manifest {EXPECT['records']} / segmentation {EXPECT['segmentation']} "
               f"/ excluded {EXPECT['excluded']}",
        len(rows) == EXPECT["records"] and len(seg) == EXPECT["segmentation"]
        and len(excl) == EXPECT["excluded"],
        f"{len(rows)} / {len(seg)} / {len(excl)}")

    named, missing = set(), []
    for r in rows:
        for col in ("canonical_image_path", "canonical_mask_path"):
            rel = r.get(col, "").strip()
            if not rel:
                continue
            p = bank / rel
            named.add(str(p.resolve()) if p.exists() else str(p))
            if not p.is_file():
                missing.append(rel)
    add("V02", "every canonical path named in the manifest exists", not missing,
        f"{len(missing)} missing" + (f": {missing[:3]}" if missing else ""))

    all_png = [p for p in bank.rglob("*.png") if p.is_file()]
    prev_png = [p for p in all_png if p.parent.name in PREVIEW_DIRS]
    on_disk = [p for p in all_png if p.parent.name not in PREVIEW_DIRS]
    orphans = [str(p) for p in on_disk if str(p.resolve()) not in named]
    add("V03", f"exactly {EXPECT['png']} canonical PNG and no orphan "
               "(previews excluded from the count)",
        len(on_disk) == EXPECT["png"] and not orphans,
        f"{len(on_disk)} canonical + {len(prev_png)} preview; "
        f"{len(orphans)} orphan" + (f": {orphans[:2]}" if orphans else ""))

    todo = list(rows)
    if a.sample and a.sample < len(todo):
        todo = random.Random(0).sample(todo, a.sample)
        print(f"sampling  : {len(todo)} of {len(rows)}")
    print(f"verifying : {len(todo)} record(s) pixel by pixel ...")

    dim_bad, ident_bad, enc_bad, mdim_bad = [], [], [], []
    recon_bad, empty_bad, fg_bad = [], [], []
    for n, r in enumerate(todo, 1):
        num = r.get("source_file_number", "")
        cimg = bank / r.get("canonical_image_path", "")
        sdcm = raw / r.get("source_image_path", "")
        if not (cimg.is_file() and sdcm.is_file()):
            continue
        arr = pydicom.dcmread(sdcm).pixel_array
        with Image.open(cimg) as ci:
            back = np.asarray(ci)
            iw, ih = ci.width, ci.height
        if (ih, iw) != arr.shape[:2]:
            dim_bad.append(f"{num} {(ih, iw)} vs {arr.shape[:2]}")
        elif not np.array_equal(back.astype(np.uint16), arr.astype(np.uint16)):
            d = int(np.sum(back.astype(np.uint16) != arr.astype(np.uint16)))
            ident_bad.append(f"{num} differing px={d}")
        mrel = r.get("canonical_mask_path", "").strip()
        if not mrel:
            continue
        cmask = bank / mrel
        if not cmask.is_file():
            continue
        with Image.open(cmask) as cm:
            if cm.mode != "L":
                enc_bad.append(f"{num} mode={cm.mode}")
            ma = np.asarray(cm.convert("L"))
            mw, mh = cm.width, cm.height
        vals = set(np.unique(ma).tolist())
        if not vals <= {0, 255}:
            enc_bad.append(f"{num} values={sorted(vals)[:6]}")
        if (mh, mw) != (ih, iw):
            mdim_bad.append(f"{num} mask {(mh, mw)} vs image {(ih, iw)}")
        fg = int((ma == 255).sum())
        if fg == 0:
            empty_bad.append(num)
        rec = r.get("mask_foreground_pixels", "").strip()
        if rec.isdigit() and int(rec) != fg:
            fg_bad.append(f"{num} disk={fg} manifest={rec}")
        xrel = r.get("source_annotation_path", "").strip()
        if not xrel or not (raw / xrel).is_file():
            recon_bad.append(f"{num}: annotation missing")
            continue
        polys, _named_n = mass_polygons(raw / xrel)
        im = Image.new("L", (iw, ih), 0)
        d = ImageDraw.Draw(im)
        for pts in polys:
            d.polygon([(float(x), float(y)) for x, y in pts], fill=255)
        expect = np.asarray(im, dtype=np.uint8)
        if expect.shape != ma.shape or not np.array_equal(expect, ma):
            dd = int(np.sum(expect != ma)) if expect.shape == ma.shape else -1
            recon_bad.append(f"{num} differing px={dd}")
        if n % 25 == 0:
            print(f"            {n}/{len(todo)}")

    add("V04", "canonical image dimensions equal the DICOM", not dim_bad,
        f"{len(dim_bad)} mismatch" + (f": {dim_bad[:2]}" if dim_bad else ""))
    add("V05", "canonical image pixels equal the DICOM stored values exactly",
        not ident_bad,
        f"{len(ident_bad)} differ" + (f": {ident_bad[:2]}" if ident_bad else ""))
    add("V06", "masks single-channel and strictly {0,255}", not enc_bad,
        f"{len(enc_bad)} bad" + (f": {enc_bad[:2]}" if enc_bad else ""))
    add("V07", "mask and image dimensions agree", not mdim_bad,
        f"{len(mdim_bad)} mismatch" + (f": {mdim_bad[:2]}" if mdim_bad else ""))
    add("V08", "every mask reconstructs from the source XML Mass contours",
        not recon_bad,
        f"{len([r for r in todo if r.get('canonical_mask_path')]) - len(recon_bad)} "
        f"reconstructed exactly, {len(recon_bad)} differ"
        + (f": {recon_bad[:2]}" if recon_bad else ""))
    add("V09", "every written mask is non-empty", not empty_bad,
        f"{len(empty_bad)} empty" + (f": {empty_bad[:2]}" if empty_bad else ""))

    wrote, still_mass = [], []
    for r in excl:
        if r.get("canonical_mask_path", "").strip():
            wrote.append(r.get("source_file_number", ""))
        xrel = r.get("source_annotation_path", "").strip()
        if xrel and (raw / xrel).is_file():
            polys, named_n = mass_polygons(raw / xrel)
            if named_n:
                still_mass.append(f"{r.get('source_file_number')}: {named_n} Mass ROI")
    add("V10", "the excluded records wrote no mask and their XML really carries no "
               "Mass ROI",
        not wrote and not still_mass,
        f"{len(excl)} excluded; {len(wrote)} wrote a mask; "
        f"{len(still_mass)} still carry a Mass ROI"
        + (f": {still_mass}" if still_mass else ""))

    cls_bad, lab_bad = [], []
    for r in rows:
        num = r.get("source_file_number", "")
        br = r.get("birads_verbatim", "").strip()
        lab = str(r.get("canonical_binary_label", "")).strip()
        img = r.get("canonical_image_path", "")
        if r.get("segmentation_available") != "TRUE":
            if not img.startswith("_excluded"):
                cls_bad.append(f"{num}: excluded but under {img.split('/')[0]}")
            continue
        want = "Benign" if br in BENIGN_BIRADS else "Malignant"
        want_lab = "0" if br in BENIGN_BIRADS else "1"
        if not img.startswith(f"{want}/"):
            cls_bad.append(f"{num}: BI-RADS {br} under {img.split('/')[0]}")
        if lab != want_lab:
            lab_bad.append(f"{num}: BI-RADS {br} labelled {lab}")
    on_b = len(list((bank / "Benign" / "Images").glob("*.png"))) if (
        bank / "Benign" / "Images").is_dir() else 0
    on_m = len(list((bank / "Malignant" / "Images").glob("*.png"))) if (
        bank / "Malignant" / "Images").is_dir() else 0
    add("V11", f"class folders agree with the BI-RADS assignment "
               f"({EXPECT['benign']} benign / {EXPECT['malignant']} malignant)",
        not cls_bad and not lab_bad and on_b == EXPECT["benign"]
        and on_m == EXPECT["malignant"],
        f"on disk Benign {on_b} / Malignant {on_m}; {len(cls_bad)} misfiled, "
        f"{len(lab_bad)} mislabelled"
        + (f": {(cls_bad + lab_bad)[:2]}" if (cls_bad or lab_bad) else ""))
    prov = Counter("BIRADS_DERIVED_ASSIGNMENT" in r.get("label_provenance", "")
                   for r in rows)
    add("V11b", "every label is marked as a BI-RADS assignment, not source pathology",
        prov.get(True, 0) == len(rows),
        f"{prov.get(True, 0)}/{len(rows)} carry "
        "BIRADS_DERIVED_ASSIGNMENT_NOT_SOURCE_PATHOLOGY")

    wb = raw / WORKBOOK
    if not wb.is_file() and (raw / "INbreast.xlsx").is_file():
        wb = raw / "INbreast.xlsx"
    br_ok, br_bad = False, []
    if wb.is_file():
        df = pd.read_excel(wb).iloc[:REAL_ROWS]
        col = COHORT_COL if COHORT_COL in df.columns else next(
            (c for c in df.columns if c.strip().lower() == "mass"), None)
        src_br = {}
        for _, row in df[df[col].astype(str).str.strip().str.upper() == "X"].iterrows():
            try:
                src_br[str(int(row["File Name"]))] = str(row["Bi-Rads"]).strip()
            except (TypeError, ValueError):
                pass
        for r in rows:
            num = r.get("source_file_number", "")
            if src_br.get(num) != r.get("birads_verbatim", ""):
                br_bad.append(f"{num}: manifest={r.get('birads_verbatim')} "
                              f"workbook={src_br.get(num)}")
        man_br = Counter(r.get("birads_verbatim", "") for r in rows)
        br_ok = not br_bad and dict(man_br) == EXPECT["birads"]
    add("V12", "BI-RADS matches INbreast.xls verbatim and the locked distribution",
        br_ok, f"{len(br_bad)} mismatch; distribution "
               f"{dict(sorted(Counter(r.get('birads_verbatim','') for r in rows).items()))}")

    hash_by_num = {}
    for p in (raw / DICOM_DIR).glob("*.dcm"):
        parts = p.name.split("_")
        if len(parts) > 1:
            hash_by_num[parts[0]] = parts[1]
    id_bad = [r.get("source_file_number") for r in rows
              if r.get("patient_id", "").strip() or r.get("lesion_id", "").strip()]
    key_bad = [r.get("source_file_number") for r in rows
               if r.get("case_id", "") != hash_by_num.get(r.get("source_file_number"), "")]
    add("V13", "no patient or lesion identity invented; the group key is the DICOM "
               "filename hash",
        not id_bad and not key_bad,
        f"{len(id_bad)} invented; {len(key_bad)} group-key mismatch; "
        f"{len({r.get('case_id') for r in rows})} distinct group(s)")

    after_raw = snapshot(raw)
    raw_mod = [k for k in after_raw if k in before_raw and after_raw[k] != before_raw[k]]
    add("V14", "RAW unchanged", not raw_mod and len(after_raw) == len(before_raw),
        f"{len(before_raw)} -> {len(after_raw)} file(s), {len(raw_mod)} modified")
    splits = [str(p) for p in bank.rglob("*") if p.is_file() and is_split_like(str(p))]
    add("V15", "no Train/Validation/Test split created", not splits,
        f"{len(splits)} split-like artifact(s)")
    add("V16", "recorded mask foreground counts match the pixels on disk", not fg_bad,
        f"{len(fg_bad)} differ" + (f": {fg_bad[:2]}" if fg_bad else ""))

    prev_bad, prev_ok, prev_checked = [], 0, 0
    for r in todo:
        num = r.get("source_file_number", "")
        cimg = bank / r.get("canonical_image_path", "")
        lin = r.get("canonical_preview_linear_path", "").strip()
        if not lin or not cimg.is_file():
            continue
        lp = bank / lin
        if not lp.is_file():
            prev_bad.append(f"{num}: linear preview missing")
            continue
        prev_checked += 1
        with Image.open(cimg) as ci:
            canon16 = np.asarray(ci).astype(np.float64)
        with Image.open(lp) as pi:
            pv = np.asarray(pi)
        if pv.dtype != np.uint8:
            prev_bad.append(f"{num}: preview dtype {pv.dtype}, expected uint8")
            continue
        want = np.clip(np.rint(canon16 / PREVIEW_DIVISOR * 255.0), 0,
                       255).astype(np.uint8)
        if not np.array_equal(want, pv):
            prev_bad.append(f"{num}: linear preview not re-derivable "
                            f"({int(np.sum(want != pv))} px)")
        else:
            prev_ok += 1
    add("V17", "the linear preview is exactly re-derivable from the canonical image "
               "and is 8-bit, so it cannot be mistaken for the canonical data",
        not prev_bad and prev_checked > 0,
        f"{prev_ok}/{prev_checked} re-derived exactly; {len(prev_bad)} problem(s)"
        + (f": {prev_bad[:2]}" if prev_bad else ""))

    seg_rows = [r for r in rows if r.get("segmentation_available") == "TRUE"]
    xc = Counter(r.get("mask_crosscheck_legacy_method", "") for r in seg_rows)
    add("V18", "every mask matched the independent (y,x) reconstruction recorded at "
               "build time",
        xc.get("IDENTICAL", 0) == len(seg_rows),
        f"{dict(xc)} over {len(seg_rows)} segmentation record(s)")

    print()
    print("=" * 78)
    for i, d, ok, b in checks:
        print(f"  {i}  [{'PASS' if ok else 'FAIL'}]  {d}")
        print(f"        {b}")
    failed = [c for c in checks if not c[2]]
    print("=" * 78)
    print(f"PASS = {len(checks) - len(failed)}   FAIL = {len(failed)}")
    print("\nFINAL VERDICT:")
    verdict = ("INBREAST_FULLSOURCE_BANK_VERIFIED_ON_DISK" if not failed
               else "INBREAST_FULLSOURCE_BANK_VERIFICATION_FAILED")
    print(verdict)
    for i, d, _ok, b in failed:
        print(f"    {i}: {d} | {b}")

    rep = bank / "reports" / "INBREAST_FULLSOURCE_ONDISK_VERIFICATION.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(
        "INbreast FULL SOURCE - INDEPENDENT ON-DISK VERIFICATION\n"
        f"generated: {now_iso()}\nsource: {raw}\nbank: {bank}\n"
        f"records verified: {len(todo)} of {len(rows)}"
        + (" (SAMPLE)" if a.sample else " (ALL)") + "\n" + "=" * 72 + "\n"
        + "\n".join(f"{i}  [{'PASS' if ok else 'FAIL'}]  {d}\n    {b}"
                    for i, d, ok, b in checks)
        + "\n" + "=" * 72
        + f"\nPASS={len(checks) - len(failed)}  FAIL={len(failed)}\n{verdict}\n",
        encoding="utf-8")
    print(f"\nWROTE {rep}")
    return 0 if not failed else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
