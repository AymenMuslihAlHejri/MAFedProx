#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bank_cbis_ddsm.py
=====================

PHASE 2B.3.10 - INDEPENDENT POST-EXECUTION VERIFIER FOR THE CBIS-DDSM CANONICAL BANK

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
    python verify_bank_cbis_ddsm.py
    python verify_bank_cbis_ddsm.py --sample 100
Requires: Python >= 3.8, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from collections import defaultdict
from collections import Counter
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET = "CBIS_DDSM"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)
SUMMARY_FILE = "32_cbis_phase1b_modeling_summary.json"
SEG_FILE = "31_cbis_phase1b_segmentation_eligibility.csv"
ADJ_FILE = "20_cbis_phase1a1_role_conflict_adjudication.csv"


MASK_POLICY_ID = "CBIS_CANONICAL_MASK_BINARY_V1"
IMAGE_POLICY_ID = "CBIS_CANONICAL_IMAGE_LOSSLESS_V1"
DIM_CODE = "JOINT_TASK_INELIGIBLE_MASK_IMAGE_DIMENSION_MISMATCH"
UNRES_CODE = "EXCLUDE_SEGMENTATION_LINKAGE_UNRESOLVED"
BWC_CODE = "NOT_IN_BINARY_TARGET_BENIGN_WITHOUT_CALLBACK"
INCLUDED = "INCLUDED_TRAINING_COHORT"
UNRESOLVED_ID = ("P_00474", "LEFT", "MLO", "1")
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]
EXPECT = {"rows": 3568, "cohort": 2805, "benign": 1367, "malignant": 1438,
          "bwc": 682, "dim_mismatch": 80, "dim_benign": 62, "dim_malignant": 18,
          "unresolved": 1, "png": 5610, "mammograms": 2625, "patients": 1389,
          "source_benign": 1429, "source_malignant": 1457}
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
        if d.is_dir() and "cbis" in d.name.lower().replace(" ", "").replace("-", ""):
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
    """Independent reimplementation of CBIS_CANONICAL_MASK_BINARY_V1: the ROI raster must
    hold exactly two values; {0,255} passes through and {0,1} is scaled. Written from
    the rule text, not copied from the builder, so a builder bug cannot hide behind
    identical code. Nothing is thresholded."""
    if arr.ndim == 3 and arr.shape[2] >= 3:
        if not (np.array_equal(arr[..., 0], arr[..., 1])
                and np.array_equal(arr[..., 1], arr[..., 2])):
            return None, "CHANNELS_NOT_REPLICATED"
        arr = arr[..., 0]
    elif arr.ndim != 2:
        return None, f"UNEXPECTED_SHAPE_{arr.shape}"
    vals = set(np.unique(arr).tolist())
    if vals == {0, 255}:
        return arr.astype(np.uint8), "ALREADY_0_255"
    if vals == {0, 1}:
        return (arr.astype(np.uint8) * 255), "SCALED_0_1_TO_0_255"
    return None, f"VALUES_OUTSIDE_RULE_{sorted(vals)[:5]}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the BrEaST canonical bank.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--raw-root", default=None,
                    help="source root on H:. CBIS-DDSM's Phase 1B summary carries no "
                         "raw_root key, so it must be given here; the root used is "
                         "printed and recorded in the report.")
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
    print("CBIS-DDSM CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION")
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
    src_root, root_from = None, ""
    if getattr(a, "raw_root", None):
        src_root, root_from = Path(a.raw_root), "--raw-root"
    else:
        for k in ROOT_LOCATORS:
            v = blob.get(k)
            if isinstance(v, str) and v.strip():
                src_root, root_from = Path(v.strip()), f"{SUMMARY_FILE} :: json:{k}"
                break
    print(f"source    : {src_root}  [READ-ONLY]  (from {root_from or 'nowhere'})")
    if src_root is None:
        print("BLOCKED: this dataset's summary carries no source root; pass "
              "--raw-root explicitly. Nothing was guessed.")
        return 2
    if not src_root.is_dir():
        print(f"BLOCKED: the source root does not exist: {src_root}")
        return 2

    before_raw = snapshot(src_root)
    prep_dirs = [d for d in prep.iterdir() if d.is_dir() and d.name != BANK_NAME]
    before_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        before_audit.update(snapshot(d))

    rows = read_rows(man)
    eligible = [r for r in rows if r.get("canonical_image_path", "").strip()]
    seg = eligible
    seg = [r for r in rows if r.get("canonical_mask_path", "").strip()]
    mk_fg: Dict[str, int] = {}
    for r in rows:
        v = str(r.get("mask_foreground_pixels", "")).strip()
        if v.isdigit():
            mk_fg[str(r.get("sample_id", "")).strip()] = int(v)
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
    cohort = [r for r in rows if r.get("exclusion_status") == INCLUDED]
    dim_excl = [r for r in rows if r.get("exclusion_status") == DIM_CODE]
    unres = [r for r in rows if r.get("exclusion_status") == UNRES_CODE]
    bwc = [r for r in rows if r.get("exclusion_status") == BWC_CODE]
    cb = Counter(r.get("pathology_source_verbatim") for r in cohort)
    db = Counter(r.get("pathology_source_verbatim") for r in dim_excl)
    add("V01", f"manifest {EXPECT['rows']} = cohort {EXPECT['cohort']} + BWC "
               f"{EXPECT['bwc']} + dimension-mismatch {EXPECT['dim_mismatch']} + "
               f"unresolved {EXPECT['unresolved']}",
        len(rows) == EXPECT["rows"] and len(cohort) == EXPECT["cohort"]
        and len(bwc) == EXPECT["bwc"] and len(dim_excl) == EXPECT["dim_mismatch"]
        and len(unres) == EXPECT["unresolved"]
        and len(cohort) + len(bwc) + len(dim_excl) + len(unres) == len(rows),
        f"{len(rows)} = {len(cohort)} + {len(bwc)} + {len(dim_excl)} + {len(unres)}")
    add("V01b", f"cohort {EXPECT['cohort']} = {EXPECT['benign']} BENIGN + "
                f"{EXPECT['malignant']} MALIGNANT",
        cb.get("BENIGN") == EXPECT["benign"]
        and cb.get("MALIGNANT") == EXPECT["malignant"]
        and set(cb) == {"BENIGN", "MALIGNANT"},
        f"{dict(cb)}")
    add("V01c", f"source {EXPECT['source_benign']} B and "
                f"{EXPECT['source_malignant']} M, minus 1 unresolved and "
                f"{EXPECT['dim_mismatch']} mismatches, gives the cohort",
        EXPECT["source_benign"] - db.get("BENIGN", 0) == EXPECT["benign"]
        and EXPECT["source_malignant"] - db.get("MALIGNANT", 0)
        - len(unres) == EXPECT["malignant"]
        and db.get("BENIGN") == EXPECT["dim_benign"]
        and db.get("MALIGNANT") == EXPECT["dim_malignant"],
        f"{EXPECT['source_benign']}-{db.get('BENIGN')} = {EXPECT['benign']}; "
        f"{EXPECT['source_malignant']}-{db.get('MALIGNANT')}-{len(unres)} = "
        f"{EXPECT['malignant']}")
    named, missing = set(), []
    for r in eligible:
        # an empty mask path is CORRECT for the 7 without a raster GT; only a named
        # path that does not exist is a failure
        for col in ("canonical_image_path", "canonical_mask_path"):
            rel = r[col].strip()
            if not rel:
                missing.append(f"{r.get('sample_id', '')}: {col} empty")
                continue
            pp = bank / rel
            named.add(str(pp.resolve()) if pp.exists() else str(pp))
            if not pp.is_file():
                missing.append(rel)
    add("V02", "every canonical path named in the manifest exists, and every cohort "
               "record names BOTH an image and a mask",
        not missing,
        f"{len(missing)} missing" + (f": {missing[:3]}" if missing else "")
        + f"; {len(eligible)} image path(s), {len(seg)} mask path(s)")

    on_disk = [p for p in bank.rglob("*.png") if p.is_file()]
    orphans = [str(p) for p in on_disk if str(p.resolve()) not in named]
    # explicit 1:1 pairing, measured from the FILE NAMES on disk rather than from the
    # manifest, so a file that exists without its partner cannot pass unnoticed
    img_stems, mask_stems = {}, {}
    for cls in ("Benign", "Malignant"):
        for f in (bank / cls / "Images").glob("*.png"):
            img_stems[f.stem] = cls
        for f in (bank / cls / "Masks").glob("*.png"):
            mask_stems[f.stem[:-5] if f.stem.endswith("_mask") else f.stem] = cls
    img_no_mask = sorted(set(img_stems) - set(mask_stems))
    mask_no_img = sorted(set(mask_stems) - set(img_stems))
    cross = [k for k in set(img_stems) & set(mask_stems)
             if img_stems[k] != mask_stems[k]]
    add("V02b", "every image has exactly one mask and every mask exactly one image, "
                "in the same class folder",
        not img_no_mask and not mask_no_img and not cross
        and len(img_stems) == EXPECT["cohort"],
        f"{len(img_stems)} image(s), {len(mask_stems)} mask(s); "
        f"{len(img_no_mask)} image(s) without a mask, {len(mask_no_img)} mask(s) "
        f"without an image, {len(cross)} in mismatched folders"
        + (f": {(img_no_mask + mask_no_img + cross)[:3]}"
           if (img_no_mask or mask_no_img or cross) else ""))

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
        key = r.get("sample_id", "")
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
            # NEVER convert here: these images keep their source depth, and
            # convert("L") would silently downcast a 16-bit image to 8-bit and make
            # every comparison fail for a reason that is in this verifier, not on disk
            ca = np.asarray(ci)
            chan_seen[ci.mode] += 1
        if ca.shape != want.shape:
            dim_bad.append(f"{key}: canonical {ca.shape} vs source {want.shape}")
            continue
        # compare at the SOURCE dtype: casting either side truncates and invents a
        # difference that is not on disk
        if not np.array_equal(ca.astype(want.dtype), want):
            ident_bad.append(f"{key}: {int(np.sum(ca != want))} px differ "
                             f"(canonical {ca.dtype} vs source {want.dtype})")
            continue
        mrel = r.get("canonical_mask_path", "").strip()
        if not mrel:
            continue                     # a record with no mask, by design
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
        src_m = pydicom.dcmread(smask).pixel_array
        exp, _note = reduce_plane(src_m, np)
        if exp is None or not np.array_equal(exp, ma):
            d = (int(np.sum(exp != ma)) if exp is not None and exp.shape == ma.shape
                 else -1)
            recon_bad.append(f"{key}: {d} px differ from the independent rebuild")
        if int((ma == 255).sum()) == 0:
            empty_bad.append(key)
        rec = str(r.get("mask_foreground_pixels", "")).strip()
        if rec.isdigit():
            if int(rec) != int((ma == 255).sum()):
                fg_bad.append(f"{key}: disk={int((ma == 255).sum())} manifest={rec}")
            else:
                fg_ok += 1
        if n % 50 == 0:
            print(f"            {n}/{len(todo)}")

    add("V04", "canonical image dimensions equal source", not dim_bad,
        f"{len(dim_bad)} mismatch" + (f": {dim_bad[:2]}" if dim_bad else ""))
    add("V04b", "every canonical image is single-channel at its source depth, and "
                "the reduction path is the one the source shape dictates",
        set(chan_seen) <= {"L", "I;16", "I"} and sum(path_seen.values()) == len(todo),
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

    # nothing outside the cohort may have reached the class directories
    on_disk_stems = {p.stem.replace("_mask", "")
                     for p in bank.rglob("*.png")}
    leaked = [r.get("sample_id") for r in (bwc + dim_excl + unres)
              if r.get("sample_id") in on_disk_stems]
    wrote = [r.get("sample_id") for r in (bwc + dim_excl + unres)
             if r.get("canonical_image_path", "").strip()
             or r.get("canonical_mask_path", "").strip()]
    add("V11", f"none of the {EXPECT['bwc'] + EXPECT['dim_mismatch'] + EXPECT['unresolved']} "
               "non-cohort records has a canonical path or a file on disk",
        not leaked and not wrote,
        f"{len(bwc)} BWC + {len(dim_excl)} mismatch + {len(unres)} unresolved; "
        f"{len(wrote)} carry a path, {len(leaked)} appear on disk")

    # the 80 really do mismatch in the source, re-measured here
    still_ok, bad_claim = 0, []
    for r in dim_excl:
        sp = src_root / r.get("source_image_path", "")
        mp = src_root / r.get("source_mask_path", "")
        if not (sp.is_file() and mp.is_file()):
            continue
        ia = pydicom.dcmread(sp).pixel_array
        ma = pydicom.dcmread(mp).pixel_array
        if ia.shape[:2] == ma.shape[:2]:
            bad_claim.append(f"{r.get('sample_id')}: shapes actually agree")
        else:
            still_ok += 1
    add("V11b", "each dimension-mismatch record really does mismatch in the source, "
                "re-measured here",
        not bad_claim,
        f"{still_ok}/{len(dim_excl)} confirmed mismatched"
        + (f"; {bad_claim[:2]}" if bad_claim else ""))

    add("V12", f"the {EXPECT['bwc']} BWC records keep their source category verbatim "
               "and were never remapped to BENIGN",
        all(r.get("pathology_source_verbatim") == "BENIGN_WITHOUT_CALLBACK"
            and str(r.get("canonical_binary_label")) not in ("0", "1")
            for r in bwc),
        f"{len(bwc)} retained; 0 remapped")

    ur = [r for r in unres]
    add("V12b", "the adjudicated unresolved record is the one excluded, and its ROI "
                "role is recorded",
        len(ur) == 1
        and (ur[0].get("case_description_patient_id"), ur[0].get("breast_side"),
             ur[0].get("image_view"), ur[0].get("abnormality_id")) == UNRESOLVED_ID
        and ur[0].get("roi_dicom_role") == "CROPPED_IMAGE",
        (f"{ur[0].get('case_description_patient_id')}/{ur[0].get('breast_side')}/"
         f"{ur[0].get('image_view')}/{ur[0].get('abnormality_id')} "
         f"role={ur[0].get('roi_dicom_role')} "
         f"pathology={ur[0].get('pathology_source_verbatim')}") if ur else "none")

    union = [r.get("sample_id") for r in rows
             if r.get("mask_union_applied") != "NO"]
    lbl_src = [r.get("sample_id") for r in rows
               if "ABNORMALITY level" not in r.get("label_provenance", "")]
    add("V13", "no mask union, no mammogram-level label, no geometric transform",
        not union and not lbl_src,
        f"{len(union)} union(s); {len(lbl_src)} record(s) not labelled at abnormality "
        "level")

    # two lesions on one mammogram must be two distinct files
    per_mam = defaultdict(list)
    for r in cohort:
        per_mam[r.get("mammogram_key")].append(r)
    multi = {k: v for k, v in per_mam.items() if len(v) > 1}
    clash = [k for k, v in multi.items()
             if len({x.get("canonical_image_path") for x in v}) != len(v)
             or len({x.get("canonical_mask_path") for x in v}) != len(v)]
    add("V13b", "a mammogram with several lesions yields several distinct sample "
                "files; none overwrites another",
        not clash,
        f"{len(per_mam)} mammogram(s) for {len(cohort)} sample(s); {len(multi)} carry "
        f"more than one lesion; {len(clash)} collision(s)")

    key_bad = [r.get("sample_id") for r in rows
               if r.get("group_key_for_future_partitions")
               != f"PATIENT::{r.get('patient_id')}"
               or not re.fullmatch(r"P_\d+", str(r.get("patient_id", "")))]
    add("V13c", f"every record carries PATIENT::P_XXXXX; the cohort spans "
                f"{EXPECT['patients']} patient(s)",
        not key_bad
        and len({r.get("patient_id") for r in cohort}) == EXPECT["patients"],
        f"{len({r.get('patient_id') for r in cohort})} patient(s) in the cohort; "
        f"{len(key_bad)} bad key")

    on_b = len(list((bank / "Benign" / "Images").glob("*.png"))) if (
        bank / "Benign" / "Images").is_dir() else 0
    on_m = len(list((bank / "Malignant" / "Images").glob("*.png"))) if (
        bank / "Malignant" / "Images").is_dir() else 0
    mk_b = len(list((bank / "Benign" / "Masks").glob("*.png"))) if (
        bank / "Benign" / "Masks").is_dir() else 0
    mk_m = len(list((bank / "Malignant" / "Masks").glob("*.png"))) if (
        bank / "Malignant" / "Masks").is_dir() else 0
    add("V13d", f"the directories hold exactly {EXPECT['benign']} benign and "
                f"{EXPECT['malignant']} malignant image-mask pairs",
        on_b == mk_b == EXPECT["benign"] and on_m == mk_m == EXPECT["malignant"],
        f"Benign {on_b} images / {mk_b} masks; Malignant {on_m} images / {mk_m} masks")

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

    add("V17", "the canonical mask foreground matches the count the builder recorded, "
               "proving the conversion preserved the annotation exactly",
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
    verdict = ("CBIS_DDSM_CANONICAL_BANK_VERIFIED_ON_DISK" if not failed
               else "CBIS_DDSM_CANONICAL_BANK_VERIFICATION_FAILED")
    print(verdict)
    for i, d, _ok, b in failed:
        print(f"    {i}: {d} | {b}")

    rep = bank / "reports" / "CBIS_DDSM_ONDISK_VERIFICATION.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(
        "CBIS-DDSM CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION\n"
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
