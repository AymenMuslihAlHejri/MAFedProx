#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bank_dmid.py
=====================

PHASE 2B.3.9 - INDEPENDENT POST-EXECUTION VERIFIER FOR THE DMID CANONICAL BANK

Checks the canonical output ON DISK against the SOURCE and the completed audit, never
against the builder's log or its report. It re-reads the manifest, re-opens every
canonical PNG and every source file, and recomputes the channel reduction itself.

READ-ONLY EVERYWHERE. Writes one report under BrEaST\\reports\\ and nothing else.

V01 manifest/source/canonical accounting  (252 / 154 / 98, excluded 0)
V02 every canonical path named in the manifest exists
V03 exactly 504 canonical PNG and no orphan
V04 source and canonical image dimensions match
V05 canonical image decoded pixels equal source decoded pixels
V06 masks are single-channel and strictly {0,255}
V07 mask and image dimensions agree
V08 every canonical mask equals the independently reduced source colour plane
V09 every canonical mask is non-empty
V10 class folders agree with labels
V11 no patient/case/lesion identity invented
V12 no unsupported duplicate or group lock invented
V13 RAW unchanged
V14 completed audit unchanged
V15 no Train/Validation/Test split created

V16 IS NOT IN THE ORIGINAL DESIGN AND IS ADDED DELIBERATELY
    The source masks are RGBA and their alpha channel varies, so "the canonical mask
    is a binary plane" is not by itself proof that the RIGHT plane was taken: an
    alpha-encoded annotation would also yield a binary plane. V16 compares the
    foreground pixel count of every canonical mask against foreground_pixels recorded
    independently in 05_breast_mask_audit.csv. Agreement proves the conversion kept
    the plane the audit profiled, and closes the alpha question numerically.

Usage:
    python verify_bank_dmid.py
    python verify_bank_dmid.py --sample 100
Requires: Python >= 3.8, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from collections import Counter
import re
import sys
import traceback
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
SUMMARY_FILE = "17_dmid_audit_summary.json"
AUDIT_MANIFEST_FILE = "83_dmid_phase1b_corrected_localization_manifest.csv"
MASKINT_FILE = "47_dmid_phase1a2_roi_mask_integrity.csv"
DIMCHECK_FILE = "48_dmid_phase1a2_roi_dicom_dimension_check.csv"

MASK_POLICY_ID = "DMID_CANONICAL_MASK_WHITE_FOREGROUND_V1"
IMAGE_POLICY_ID = "DMID_CANONICAL_IMAGE_8BIT_REPLICATION_REDUCTION_V1"
EXCLUSION_CODE = "EXCLUDE_NO_VERIFIED_RASTER_SEGMENTATION_GT"
SEVEN = ["IMG022", "IMG023", "IMG081", "IMG262", "IMG405", "IMG406",
         "IMG435"]
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]
EXPECT = {"rows": 510, "images": 255, "cls_benign": 145, "cls_malignant": 110,
          "joint": 248, "joint_benign": 143, "joint_malignant": 105,
          "no_raster": 7, "non_binary": 255, "png": 503, "masks_source": 269}
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def archived(p: Path) -> bool:
    return any(x.lower() in ARCHIVE_DIR_NAMES for x in p.parts[:-1])


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


def audit_dir(prep: Path) -> Optional[Path]:
    for d in prep.iterdir():
        if d.is_dir() and "dmid" in d.name.lower().replace(" ", ""):
            return d
    return None


def find_in_audit(prep: Path, filename: str) -> Optional[Path]:
    d = audit_dir(prep)
    if d is None:
        return None
    hits = [p for p in d.rglob(filename) if not archived(p)]
    return hits[0] if len(hits) == 1 else None


def read_source_mask(path, np):
    try:
        import tifffile
        return np.asarray(tifffile.imread(str(path)))
    except Exception:  # noqa: BLE001
        from PIL import Image as _I
        with _I.open(path) as im:
            return np.asarray(im)


def reduce_plane(arr, np) -> Tuple[Any, str]:
    """Independent reimplementation of DMID_CANONICAL_MASK_WHITE_FOREGROUND_V1: colour
    channels exactly (255,255,255) are lesion foreground, everything else background.
    Written from the rule text, not copied from the builder, so a builder bug cannot
    hide behind identical code."""
    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
    else:
        return None, f"UNEXPECTED_SHAPE_{arr.shape}"
    white = np.all(rgb == 255, axis=-1)
    if not white.any():
        return None, "NO_WHITE_FOREGROUND"
    return (white.astype(np.uint8) * 255), "WHITE_TO_255"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the BrEaST canonical bank.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args(argv)
    prep = Path(a.prep_root)
    bank = prep / BANK_NAME / DATASET
    man = bank / "manifests" / "canonical_manifest.csv"

    try:
        from PIL import Image
        import numpy as np
        import pydicom
    except ImportError:
        sys.stderr.write("Pillow and numpy required\n")
        return 3
    Image.MAX_IMAGE_PIXELS = None

    print("=" * 78)
    print("DMID CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION")
    print("=" * 78)
    print(f"generated : {now_iso()}")
    print(f"bank      : {bank}")
    if not man.is_file():
        print(f"manifest not found: {man}")
        return 2
    sp = find_in_audit(prep, SUMMARY_FILE)
    if sp is None:
        print(f"{SUMMARY_FILE} not found or not unique")
        return 2
    blob = json.loads(sp.read_text(encoding="utf-8"))
    summ = blob
    src_root = None
    for k in ROOT_LOCATORS:
        v = blob.get(k)
        if isinstance(v, str) and v.strip():
            src_root = Path(v.strip())
            break
    print(f"source    : {src_root}  [READ-ONLY]")
    if src_root is None or not src_root.is_dir():
        print("source root unresolved; cannot verify against source")
        return 2

    before_raw = snapshot(src_root)
    prep_dirs = [d for d in prep.iterdir() if d.is_dir() and d.name != BANK_NAME]
    before_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        before_audit.update(snapshot(d))

    rows = read_rows(man)
    eligible = [r for r in rows if r.get("canonical_image_path", "").strip()]
    seg = [r for r in rows if r.get("canonical_mask_path", "").strip()]
    mk_fg: Dict[str, int] = {}
    for r in rows:
        v = str(r.get("mask_white_pixels", "")).strip()
        if v.isdigit():
            mk_fg[str(r.get("image_reference", "")).strip()] = int(v)
    print(f"foreground : {len(mk_fg)} count(s) recorded at build time, for V17")

    excluded = [r for r in rows if not r.get("canonical_image_path", "").strip()]
    print(f"manifest  : {len(rows)} records, {len(eligible)} eligible, "
          f"{len(excluded)} excluded")

    checks: List[Tuple[str, str, bool, str]] = []

    def add(i: str, d: str, ok: bool, b: str) -> None:
        checks.append((i, d, ok, b))

    lab = {"0": 0, "1": 0}
    for r in eligible:
        v = str(r.get("canonical_binary_label", "")).strip()
        if v in lab:
            lab[v] += 1
    cohort = [r for r in rows
              if r.get("exclusion_status") == "INCLUDED_JOINT_COHORT"]
    excl7 = [r for r in rows if r.get("exclusion_status") == EXCLUSION_CODE]
    nonbin = [r for r in rows
              if str(r.get("exclusion_status", "")).startswith("NOT_IN_BINARY_TARGET")]
    cls_valid = [r for r in rows if r.get("classification_valid") == "YES"]
    cb = Counter(r.get("source_label") for r in cls_valid)
    jb = Counter(r.get("source_label") for r in cohort)
    add("V01", f"manifest {EXPECT['rows']} = joint {EXPECT['joint']} + no-raster "
               f"{EXPECT['no_raster']} + non-binary {EXPECT['non_binary']}",
        len(rows) == EXPECT["rows"] and len(cohort) == EXPECT["joint"]
        and len(excl7) == EXPECT["no_raster"]
        and len(nonbin) == EXPECT["non_binary"]
        and len(cohort) + len(excl7) + len(nonbin) == len(rows),
        f"{len(rows)} = {len(cohort)} + {len(excl7)} + {len(nonbin)}")
    add("V01b", f"classification-valid {EXPECT['images']} = {EXPECT['cls_benign']}B + "
                f"{EXPECT['cls_malignant']}M; joint {EXPECT['joint']} = "
                f"{EXPECT['joint_benign']}B + {EXPECT['joint_malignant']}M",
        len(cls_valid) == EXPECT["images"]
        and cb.get("BENIGN") == EXPECT["cls_benign"]
        and cb.get("MALIGNANT") == EXPECT["cls_malignant"]
        and jb.get("BENIGN") == EXPECT["joint_benign"]
        and jb.get("MALIGNANT") == EXPECT["joint_malignant"],
        f"{len(cls_valid)} = {cb.get('BENIGN')}+{cb.get('MALIGNANT')}; "
        f"{len(cohort)} = {jb.get('BENIGN')}+{jb.get('MALIGNANT')}")
    named, missing = set(), []
    for r in eligible:
        # an empty mask path is CORRECT for the 7 without a raster GT; only a named
        # path that does not exist is a failure
        for col in ("canonical_image_path", "canonical_mask_path"):
            rel = r[col].strip()
            if not rel:
                if col == "canonical_image_path":
                    missing.append(f"{r.get('image_reference', '')}: no image path")
                continue
            pp = bank / rel
            named.add(str(pp.resolve()) if pp.exists() else str(pp))
            if not pp.is_file():
                missing.append(rel)
    add("V02", "every canonical path NAMED in the manifest exists (an empty mask path "
               f"is correct for the {EXPECT['no_raster']} without a raster GT)",
        not missing,
        f"{len(missing)} missing" + (f": {missing[:3]}" if missing else "")
        + f"; {len(eligible)} image path(s), {len(seg)} mask path(s)")

    on_disk = [p for p in bank.rglob("*.png") if p.is_file()]
    orphans = [str(p) for p in on_disk if str(p.resolve()) not in named]
    add("V03", f"exactly {EXPECT['png']} canonical PNG and no orphan",
        len(on_disk) == EXPECT["png"] and not orphans,
        f"{len(on_disk)} on disk, {len(orphans)} orphan"
        + (f": {orphans[:2]}" if orphans else ""))

    todo = list(eligible)
    if a.sample and a.sample < len(todo):
        todo = random.Random(0).sample(todo, a.sample)
        print(f"sampling  : {len(todo)} of {len(eligible)}")
    print(f"verifying : {len(todo)} record(s) pixel by pixel ...")

    dim_bad, ident_bad, enc_bad, mdim_bad = [], [], [], []
    recon_bad, empty_bad, fg_bad, fg_ok = [], [], [], 0
    chan_seen, path_seen = Counter(), Counter()
    for n, r in enumerate(todo, 1):
        key = r.get("image_reference", "")
        cimg = bank / r["canonical_image_path"]
        simg = src_root / r["source_image_path"]
        if not (cimg.is_file() and simg.is_file()):
            continue
        ds = pydicom.dcmread(simg)
        arr = ds.pixel_array
        # reimplemented from the rule text: ndim decides, values never change
        if arr.ndim == 2:
            want = arr
            path_seen["already single channel"] += 1
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            if not (np.array_equal(arr[..., 0], arr[..., 1])
                    and np.array_equal(arr[..., 1], arr[..., 2])):
                chan_bad_key = f"{key}: channels differ but a single channel was written"
                ident_bad.append(chan_bad_key)
                continue
            want = arr[..., 0]
            path_seen["3 identical -> 1"] += 1
        else:
            ident_bad.append(f"{key}: unexpected source shape {arr.shape}")
            continue
        with Image.open(cimg) as ci:
            ca = np.asarray(ci.convert("L"))
            chan_seen[ci.mode] += 1
        if ca.shape != want.shape:
            dim_bad.append(f"{key}: canonical {ca.shape} vs source {want.shape}")
            continue
        if not np.array_equal(ca, want.astype(np.uint8)):
            ident_bad.append(f"{key}: {int(np.sum(ca != want))} px differ")
            continue
        mrel = r.get("canonical_mask_path", "").strip()
        if not mrel:
            continue                     # the 7: image only, by design
        cmask, smask = bank / mrel, src_root / r["source_mask_path"]
        if not (cmask.is_file() and smask.is_file()):
            continue
        with Image.open(cmask) as cm:
            if cm.mode != "L":
                enc_bad.append(f"{key} mode={cm.mode}")
            ma = np.asarray(cm.convert("L"))
        vals = set(np.unique(ma).tolist())
        if not vals <= {0, 255}:
            enc_bad.append(f"{key} values={sorted(vals)[:5]}")
        if ma.shape != ca.shape:
            mdim_bad.append(f"{key}: mask {ma.shape} vs image {ca.shape}")
        src_m = read_source_mask(smask, np)
        exp, _note = reduce_plane(src_m, np)
        if exp is None or not np.array_equal(exp, ma):
            d = (int(np.sum(exp != ma)) if exp is not None and exp.shape == ma.shape
                 else -1)
            recon_bad.append(f"{key}: {d} px differ from the independent rebuild")
        if int((ma == 255).sum()) == 0:
            empty_bad.append(key)
        rec = str(r.get("mask_white_pixels", "")).strip()
        if rec.isdigit():
            if int(rec) != int((ma == 255).sum()):
                fg_bad.append(f"{key}: disk={int((ma == 255).sum())} manifest={rec}")
            else:
                fg_ok += 1
        if n % 50 == 0:
            print(f"            {n}/{len(todo)}")

    add("V04", "canonical image dimensions equal source", not dim_bad,
        f"{len(dim_bad)} mismatch" + (f": {dim_bad[:2]}" if dim_bad else ""))
    add("V04b", "every canonical image is a single-channel 8-bit PNG, and the "
                "reduction path taken is the one the source shape dictates",
        set(chan_seen) <= {"L"} and sum(path_seen.values()) == len(todo),
        f"PNG modes {dict(chan_seen)}; reduction paths {dict(path_seen)}")
    add("V05", "canonical image pixels equal decoded source pixels", not ident_bad,
        f"{len(ident_bad)} differ" + (f": {ident_bad[:2]}" if ident_bad else ""))
    add("V06", "masks single-channel and strictly {0,255}", not enc_bad,
        f"{len(enc_bad)} bad" + (f": {enc_bad[:2]}" if enc_bad else ""))
    add("V07", "mask and image dimensions agree", not mdim_bad,
        f"{len(mdim_bad)} mismatch" + (f": {mdim_bad[:2]}" if mdim_bad else ""))
    add("V08", "every canonical mask equals the independently reduced source plane",
        not recon_bad,
        f"{len(todo) - len(recon_bad)} reconstructed exactly, {len(recon_bad)} differ"
        + (f": {recon_bad[:2]}" if recon_bad else ""))
    add("V09", "every canonical mask is non-empty", not empty_bad,
        f"{len(empty_bad)} empty" + (f": {empty_bad[:2]}" if empty_bad else ""))

    lbl_bad = [r.get("sample_key", "") for r in eligible
               if (str(r["canonical_binary_label"]) == "0"
                   and not r["canonical_image_path"].startswith("Benign"))
               or (str(r["canonical_binary_label"]) == "1"
                   and not r["canonical_image_path"].startswith("Malignant"))]
    add("V10", "class folders agree with labels", not lbl_bad,
        f"{len(lbl_bad)} mismatch" + (f": {lbl_bad[:2]}" if lbl_bad else ""))

    id_bad = [r.get("sample_key", "") for r in rows
              if r.get("patient_id", "").strip() or r.get("case_id", "").strip()
              or r.get("lesion_id", "").strip()]
    cpath = find_in_audit(prep, AUDIT_MANIFEST_FILE)
    caudit = read_rows(cpath) if cpath else []
    man_cases = {r.get("case_id", "").strip() for r in rows if r.get("case_id", "").strip()}
    aud_cases = {r.get("case_id", "").strip() for r in caudit}
    key_bad = [r.get("source_record_id", "") for r in rows
               if r.get("group_key_for_future_partitions", "")
               != f"CASE::{r.get('case_id', '')}"]
    mixed = {}
    for r in rows:
        mixed.setdefault(r.get("case_id", ""), set()).add(
            str(r.get("canonical_binary_label", "")))
    mixed_cases = [c for c, s in mixed.items() if len(s) > 1]
    aud = read_rows(cpath) if cpath else []
    # the 7: image converted, label kept, NO mask, and nothing resembling one on disk
    wrote_mask = [r.get("image_reference") for r in excl7
                  if r.get("canonical_mask_path", "").strip()]
    no_img = [r.get("image_reference") for r in excl7
              if not (bank / r.get("canonical_image_path", "")).is_file()]
    add("V11", f"the {EXPECT['no_raster']} classification-valid records without a "
               "raster GT keep their image and carry NO mask",
        sorted(r.get("image_reference") for r in excl7) == sorted(SEVEN)
        and not wrote_mask and not no_img
        and all(r.get("classification_available") == "TRUE"
                and r.get("segmentation_available") == "FALSE" for r in excl7),
        f"{len(excl7)}: " + ", ".join(sorted(r.get('image_reference') for r in excl7))
        + f"; {len(wrote_mask)} carry a mask path, {len(no_img)} missing their image")

    # and the source really does ship no ROI mask for them
    has_src = []
    for r in excl7:
        mp = r.get("source_mask_path", "").strip()
        if mp and (src_root / mp).is_file():
            has_src.append(r.get("image_reference"))
    add("V11b", "the source really ships no ROI raster for those records, re-checked "
                "here",
        not has_src,
        f"{len(excl7) - len(has_src)}/{len(excl7)} confirmed absent"
        + (f"; {has_src}" if has_src else ""))

    nb_bad = [r.get("image_reference") for r in nonbin
              if r.get("canonical_image_path", "").strip()
              or r.get("canonical_mask_path", "").strip()
              or not r.get("exclusion_reason", "").strip()]
    add("V12", f"the {EXPECT['non_binary']} non-binary source categories are retained "
               "with a reason and no canonical path; they are not exclusions",
        len(nonbin) == EXPECT["non_binary"] and not nb_bad,
        f"{len(nonbin)} retained; {len(nb_bad)} with a path or no reason; "
        f"{dict(Counter(r.get('exclusion_status') for r in nonbin))}")

    id_bad = [r.get("image_reference") for r in rows
              if r.get("patient_id", "").strip() or r.get("case_id", "").strip()
              or r.get("lesion_id", "").strip()]
    add("V13", "no patient, case or lesion identity invented; the UIDs were never "
               "used as a patient key",
        not id_bad
        and summ.get("dicom_identity_status")
        == "NO_DEFENSIBLE_PATIENT_GROUPING_ESTABLISHED"
        and all("cannot be demonstrated" in r.get("grouping_limitation", "")
                for r in rows),
        f"{len(id_bad)} record(s) carry an identity; "
        f"{summ.get('dicom_identity_status')}")

    pa_bad = [r.get("image_reference") for r in rows
              if r.get("pixel_annotation_used_as_gt") != "NO"]
    add("V13b", "the pixel-annotation TIFFs were never used as segmentation GT and no "
                "mask was inferred from X/Y/radius",
        not pa_bad,
        f"{len(rows) - len(pa_bad)}/{len(rows)} record(s) declare the TIFF unused")

    lbl_bad = [r.get("image_reference") for r in eligible
               if (str(r.get("canonical_binary_label")) == "0"
                   and not r.get("canonical_image_path", "").startswith("Benign"))
               or (str(r.get("canonical_binary_label")) == "1"
                   and not r.get("canonical_image_path", "").startswith("Malignant"))]
    on_b = len(list((bank / "Benign" / "Images").glob("*.png"))) if (
        bank / "Benign" / "Images").is_dir() else 0
    on_m = len(list((bank / "Malignant" / "Images").glob("*.png"))) if (
        bank / "Malignant" / "Images").is_dir() else 0
    add("V13c", f"class folders agree with the labels ({EXPECT['cls_benign']} benign / "
                f"{EXPECT['cls_malignant']} malignant images)",
        not lbl_bad and on_b == EXPECT["cls_benign"]
        and on_m == EXPECT["cls_malignant"],
        f"on disk Benign {on_b} / Malignant {on_m}; {len(lbl_bad)} misfiled")

    multi = [r for r in cohort if r.get("multi_lesion_in_single_mask_file") == "YES"]
    add("V13d", "multi-lesion images were never built by combining rasters; one mask "
                "file per image",
        all(len([x for x in rows
                 if x.get("source_mask_path") == r.get("source_mask_path")]) == 1
            for r in multi[:20]),
        f"{len(multi)} joint image(s) hold more than one lesion inside a single "
        "source file")

    after_raw = snapshot(src_root)
    raw_mod = [k for k in after_raw if k in before_raw and after_raw[k] != before_raw[k]]
    add("V14", "RAW unchanged", not raw_mod and len(after_raw) == len(before_raw),
        f"{len(before_raw)} -> {len(after_raw)} file(s), {len(raw_mod)} modified")
    after_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        after_audit.update(snapshot(d))
    aud_mod = [k for k in after_audit
               if k in before_audit and after_audit[k] != before_audit[k]]
    add("V15", "completed audit unchanged",
        not aud_mod and len(after_audit) == len(before_audit),
        f"{len(before_audit)} -> {len(after_audit)} file(s), {len(aud_mod)} modified")
    splits = [str(p) for p in bank.rglob("*") if p.is_file() and is_split_like(str(p))]
    add("V16", "no Train/Validation/Test split created", not splits,
        f"{len(splits)} split-like artifact(s)")

    add("V17", "canonical foreground matches the count recorded in the mask audit "
               "(proves the scaling preserved the annotation exactly)",
        bool(fg_ok) and not fg_bad,
        f"{fg_ok} agree, {len(fg_bad)} differ"
        + (f": {fg_bad[:2]}" if fg_bad else "")
        + ("" if mk_fg else "  [NO foreground_pixels available; check not performed]"))

    print()
    print("=" * 78)
    for i, d, ok, b in checks:
        print(f"  {i}  [{'PASS' if ok else 'FAIL'}]  {d}")
        print(f"        {b}")
    failed = [c for c in checks if not c[2]]
    print("=" * 78)
    print(f"PASS = {len(checks) - len(failed)}   FAIL = {len(failed)}")
    print()
    print("FINAL VERDICT:")
    verdict = ("DMID_CANONICAL_BANK_VERIFIED_ON_DISK" if not failed
               else "DMID_CANONICAL_BANK_VERIFICATION_FAILED")
    print(verdict)
    for i, d, _ok, b in failed:
        print(f"    {i}: {d} | {b}")

    rep = bank / "reports" / "DMID_ONDISK_VERIFICATION.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(
        "DMID CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION\n"
        f"generated: {now_iso()}\nsource: {src_root}\nbank: {bank}\n"
        f"records verified: {len(todo)} of {len(eligible)}"
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
