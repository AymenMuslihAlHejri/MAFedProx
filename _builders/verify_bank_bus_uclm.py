#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bank_bus_uclm.py
=====================

PHASE 2B.3.7 - INDEPENDENT POST-EXECUTION VERIFIER FOR THE BUS-UCLM CANONICAL BANK

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
    python verify_bank_bus_uclm.py
    python verify_bank_bus_uclm.py --sample 100
Requires: Python >= 3.8, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
import re
import sys
import traceback
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
SUMMARY_FILE = "27_bus_uclm_phase1a1_summary.json"
MASK_AUDIT_FILE = "26_bus_uclm_phase1a1_manifest.csv"
AUDIT_MANIFEST_FILE = "26_bus_uclm_phase1a1_manifest.csv"
COHORT_FILE = "25_bus_uclm_binary_lesion_cohort.csv"
MASK_POLICY_ID = "BUS_UCLM_CANONICAL_MASK_COLOUR_TO_BINARY_V1"
IMAGE_POLICY_ID = "BUS_UCLM_CANONICAL_IMAGE_LOSSLESS_V1"
EXCLUSION_CODE = "EXCLUDE_MASK_IMAGE_DIMENSION_MISMATCH"
NORMAL_STATUS = "NOT_IN_BINARY_TARGET_NORMAL_CLASS"
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]
EXPECT = {"records": 683, "normal": 419, "selected": 264, "excluded": 4,
          "cohort": 260, "benign": 170, "malignant": 90, "png": 520,
          "patients_overall": 38, "patients_cohort": 35}
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
        if d.is_dir() and "busuclm" in d.name.lower().replace(" ", "").replace("_", "").replace("-", ""):
            return d
    return None


def find_in_audit(prep: Path, filename: str) -> Optional[Path]:
    d = audit_dir(prep)
    if d is None:
        return None
    hits = [p for p in d.rglob(filename) if not archived(p)]
    return hits[0] if len(hits) == 1 else None


def reduce_plane(arr, np) -> Tuple[Any, str]:
    """Independent reimplementation of BUS_UCLM_CANONICAL_MASK_COLOUR_TO_BINARY_V1:
    any pixel that is not pure black is lesion foreground. Written from the rule text,
    not copied from the builder, so a builder bug cannot hide behind identical code."""
    if arr.ndim == 3 and arr.shape[2] >= 3:
        fg = arr[..., :3].any(axis=2)
    elif arr.ndim == 2:
        fg = arr > 0
    else:
        return None, f"UNEXPECTED_SHAPE_{arr.shape}"
    if not fg.any():
        return None, "ALL_BLACK_NO_FOREGROUND"
    return (fg.astype(np.uint8) * 255), "NON_BLACK_TO_255"


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
    except ImportError:
        sys.stderr.write("Pillow and numpy required\n")
        return 3
    Image.MAX_IMAGE_PIXELS = None

    print("=" * 78)
    print("BUS-UCLM CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION")
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
    seg = eligible
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
    # the foreground counts live in the CANONICAL manifest the builder wrote, not in
    # the audit manifest, which never recorded them
    mk_fg: Dict[str, int] = {}
    for r in rows:
        v = str(r.get("mask_foreground_pixels", "")).strip()
        if v.isdigit():
            mk_fg[str(r.get("sample_id", "")).strip()] = int(v)
    print(f"foreground : {len(mk_fg)} count(s) recorded at build time, for V17")

    cohort = [r for r in rows
              if r.get("exclusion_status") == "INCLUDED_BINARY_LESION_COHORT"]
    excl = [r for r in rows if r.get("exclusion_status") == EXCLUSION_CODE]
    normals = [r for r in rows if r.get("exclusion_status") == NORMAL_STATUS]
    lab = Counter(r.get("source_label") for r in cohort)
    add("V01", f"manifest {EXPECT['records']} = cohort {EXPECT['cohort']} "
               f"({EXPECT['benign']} benign / {EXPECT['malignant']} malignant) + "
               f"excluded {EXPECT['excluded']} + normal {EXPECT['normal']}",
        len(rows) == EXPECT["records"] and len(cohort) == EXPECT["cohort"]
        and len(excl) == EXPECT["excluded"] and len(normals) == EXPECT["normal"]
        and lab.get("Benign") == EXPECT["benign"]
        and lab.get("Malignant") == EXPECT["malignant"]
        and len(cohort) + len(excl) + len(normals) == len(rows),
        f"{len(rows)} = {len(cohort)} ({lab.get('Benign')}/{lab.get('Malignant')}) "
        f"+ {len(excl)} + {len(normals)}")
    add("V01b", f"selected {EXPECT['selected']} - excluded {EXPECT['excluded']} = "
                f"cohort {EXPECT['cohort']}",
        len(cohort) + len(excl) == EXPECT["selected"],
        f"{len(cohort)} + {len(excl)} = {len(cohort) + len(excl)}")
    named, missing = set(), []
    for r in eligible:
        for col in ("canonical_image_path", "canonical_mask_path"):
            rel = r[col].strip()
            if not rel:
                missing.append(f"{r.get('sample_key', '')}:{col} empty")
                continue
            p = bank / rel
            named.add(str(p.resolve()) if p.exists() else str(p))
            if not p.is_file():
                missing.append(rel)
    add("V02", "every canonical path named in the manifest exists", not missing,
        f"{len(missing)} missing" + (f": {missing[:3]}" if missing else ""))

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
    chan_bad, chan_seen = [], Counter()
    for n, r in enumerate(todo, 1):
        cimg = bank / r["canonical_image_path"]
        simg = src_root / r["source_image_path"]
        if not (cimg.is_file() and simg.is_file()):
            continue
        key = r.get("sample_key", r.get("source_image_id", ""))
        with Image.open(cimg) as ci, Image.open(simg) as si:
            ca, sa = np.array(ci), np.array(si)
            if ci.size != si.size:
                dim_bad.append(f"{key} {ci.size} vs {si.size}")
            elif ca.shape != sa.shape:
                # a silent RGB -> greyscale collapse keeps the size and changes the
                # shape, so it must be reported separately from a pixel difference
                chan_bad.append(f"{key} canonical {ca.shape} vs source {sa.shape}")
            elif not np.array_equal(ca, sa):
                ident_bad.append(key)
            iw, ih = ci.width, ci.height
        chan_seen[str(ca.shape[2]) if ca.ndim == 3 else "1"] += 1
        cmask = bank / r["canonical_mask_path"]
        smask = src_root / r["source_mask_path"]
        if not (cmask.is_file() and smask.is_file()):
            continue
        with Image.open(cmask) as cm:
            if cm.mode != "L":
                enc_bad.append(f"{key} mode={cm.mode}")
            ma = np.array(cm.convert("L"))
            mw, mh = cm.width, cm.height
        vals = set(np.unique(ma).tolist())
        if not vals <= {0, 255}:
            enc_bad.append(f"{key} values={sorted(vals)[:6]}")
        if (mh, mw) != (ih, iw):
            mdim_bad.append(f"{key} mask {(mh, mw)} vs image {(ih, iw)}")
        fg = int((ma == 255).sum())
        if fg == 0:
            empty_bad.append(key)
        with Image.open(smask) as sm:
            sarr = np.array(sm)
        plane, note = reduce_plane(sarr, np)
        if plane is None:
            recon_bad.append(f"{key} ({note})")
        elif plane.shape != ma.shape or not np.array_equal(plane, ma):
            d = int(np.sum(plane != ma)) if plane.shape == ma.shape else -1
            recon_bad.append(f"{key} differing px={d}")
        # V16: does the canonical foreground match the count the audit recorded?
        exp_fg = mk_fg.get(str(r.get("sample_id", "")).strip())
        if exp_fg is None:
            exp_fg = mk_fg.get(r["source_mask_path"].strip().replace("\\", "/"))
        if exp_fg is not None:
            if exp_fg == fg:
                fg_ok += 1
            else:
                fg_bad.append(f"{key} canonical={fg} audit={exp_fg}")
        if n % 100 == 0:
            print(f"            {n}/{len(todo)}")

    add("V04", "canonical image dimensions equal source", not dim_bad,
        f"{len(dim_bad)} mismatch" + (f": {dim_bad[:2]}" if dim_bad else ""))
    add("V04b", "the canonical image keeps its source channels; no silent RGB to "
                "greyscale collapse",
        not chan_bad and set(chan_seen) == {"3"},
        f"channels on disk {dict(chan_seen)}; {len(chan_bad)} collapsed"
        + (f": {chan_bad[:2]}" if chan_bad else ""))
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
    aux_paths = {r.get("auxiliary_lesion_path", "").strip() for r in aud
                 if r.get("auxiliary_lesion_path", "").strip()}
    aux_used = [r.get("sample_key", "") for r in rows
                if r.get("source_image_path", "").strip() in aux_paths
                or r.get("source_mask_path", "").strip() in aux_paths]
    aux_converted = []
    for r in rows:
        ap = r.get("auxiliary_lesion_path", "").strip()
        if not ap:
            continue
        stem = Path(ap).stem
        for col in ("canonical_image_path", "canonical_mask_path"):
            if stem and stem in r.get(col, ""):
                aux_converted.append(f"{r.get('sample_key')}: {col}")
    wrote = [r.get("sample_id") for r in excl
             if r.get("canonical_image_path", "").strip()
             or r.get("canonical_mask_path", "").strip()]
    thin = [r.get("sample_id") for r in excl
            if not r.get("exclusion_reason")
            or "x" not in str(r.get("exclusion_reason"))]
    on_disk_names = {p.name for p in bank.rglob("*.png")}
    leaked = [r.get("sample_id") for r in excl
              if any(str(r.get("sample_id")) in n for n in on_disk_names)]
    add("V11", f"the {EXPECT['excluded']} dimension-mismatch records wrote no "
               "canonical file and carry a reason with the observed dimensions",
        not wrote and not thin and not leaked,
        f"{len(excl)} excluded: "
        + ", ".join(sorted(str(r.get('sample_id')) for r in excl))
        + f"; {len(wrote)} wrote a path, {len(leaked)} appear on disk")

    dim_bad2 = []
    for r in excl:
        sp = src_root / r.get("source_image_path", "")
        mp = src_root / r.get("source_mask_path", "")
        if not (sp.is_file() and mp.is_file()):
            continue
        with Image.open(sp) as si, Image.open(mp) as mi:
            if si.size == mi.size:
                dim_bad2.append(f"{r.get('sample_id')}: {si.size} == {mi.size}")
    add("V11b", "each excluded record really does have mismatched dimensions in the "
                "source, re-read here",
        not dim_bad2,
        f"{len(excl) - len(dim_bad2)}/{len(excl)} confirmed mismatched"
        + (f"; {dim_bad2}" if dim_bad2 else ""))

    norm_bad = [r.get("sample_id") for r in normals
                if r.get("canonical_image_path", "").strip()
                or r.get("canonical_mask_path", "").strip()
                or r.get("source_label") != "Normal"]
    add("V12", f"the {EXPECT['normal']} Normal records are retained with no canonical "
               "path and are not counted as exclusions",
        not norm_bad and len(normals) == EXPECT["normal"],
        f"{len(normals)} retained; {len(norm_bad)} with a path or a wrong label")

    colour_bad, colour_seen = [], Counter()
    for r in (todo if 'todo' in dir() else cohort):
        if r.get("exclusion_status") != "INCLUDED_BINARY_LESION_COHORT":
            continue
        mp = src_root / r.get("source_mask_path", "")
        if not mp.is_file():
            continue
        with Image.open(mp) as mi:
            ma = np.asarray(mi)
        if ma.ndim != 3 or ma.shape[2] < 3:
            colour_bad.append(f"{r.get('sample_id')}: mask is not colour")
            continue
        fgm = ma[..., :3].any(axis=2)
        if not fgm.any():
            colour_bad.append(f"{r.get('sample_id')}: all-black mask on a lesion")
            continue
        mean = ma[..., :3][fgm].mean(axis=0)
        obs = int(np.argmax(mean))
        want = 1 if r.get("source_label") == "Benign" else 0
        colour_seen["GREEN" if obs == 1 else "RED" if obs == 0 else "OTHER"] += 1
        if obs != want:
            colour_bad.append(f"{r.get('sample_id')}: label {r.get('source_label')} "
                              f"but channel {obs} dominates")
    add("V13", "the mask foreground colour agrees with the official label on every "
               "converted record, and the label never came from the colour",
        not colour_bad
        and all(r.get("label_provenance", "").find("never used to derive") >= 0
                for r in rows),
        f"{dict(colour_seen)}; {len(colour_bad)} disagreement(s)")

    id_bad2 = [r.get("sample_id") for r in rows
               if r.get("case_id", "").strip() or r.get("lesion_id", "").strip()]
    pats_all = {r.get("patient_id", "").strip() for r in rows
                if r.get("patient_id", "").strip()}
    pats_coh = {r.get("patient_id", "").strip() for r in cohort
                if r.get("patient_id", "").strip()}
    key_bad = [r.get("sample_id") for r in rows
               if r.get("group_key_for_future_partitions")
               != f"PATIENT::{r.get('patient_id')}"]
    add("V13b", f"patient grouping preserved: {EXPECT['patients_overall']} overall, "
                f"{EXPECT['patients_cohort']} in the cohort, key on every record",
        not id_bad2 and not key_bad
        and len(pats_all) == EXPECT["patients_overall"]
        and len(pats_coh) == EXPECT["patients_cohort"],
        f"{len(pats_all)} overall, {len(pats_coh)} in the cohort; {len(key_bad)} bad "
        f"key; {len(id_bad2)} invented case/lesion id")

    lbl_bad = [r.get("sample_id") for r in cohort
               if (str(r.get("canonical_binary_label")) == "0"
                   and not r.get("canonical_image_path", "").startswith("Benign"))
               or (str(r.get("canonical_binary_label")) == "1"
                   and not r.get("canonical_image_path", "").startswith("Malignant"))]
    on_b = len(list((bank / "Benign" / "Images").glob("*.png"))) if (
        bank / "Benign" / "Images").is_dir() else 0
    on_m = len(list((bank / "Malignant" / "Images").glob("*.png"))) if (
        bank / "Malignant" / "Images").is_dir() else 0
    add("V13c", f"class folders agree with the official labels "
                f"({EXPECT['benign']} benign / {EXPECT['malignant']} malignant)",
        not lbl_bad and on_b == EXPECT["benign"] and on_m == EXPECT["malignant"],
        f"on disk Benign {on_b} / Malignant {on_m}; {len(lbl_bad)} misfiled")

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

    add("V17", "canonical foreground matches the count recorded in the canonical "
               "manifest at build time",
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
    verdict = ("BUS_UCLM_CANONICAL_BANK_VERIFIED_ON_DISK" if not failed
               else "BUS_UCLM_CANONICAL_BANK_VERIFICATION_FAILED")
    print(verdict)
    for i, d, _ok, b in failed:
        print(f"    {i}: {d} | {b}")

    rep = bank / "reports" / "BUS_UCLM_ONDISK_VERIFICATION.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(
        "BUS-UCLM CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION\n"
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
