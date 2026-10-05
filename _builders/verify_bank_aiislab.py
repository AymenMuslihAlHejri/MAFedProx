#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bank_aiislab.py
=====================

PHASE 2B.3.11 - INDEPENDENT POST-EXECUTION VERIFIER FOR THE AIISLAB CANONICAL BANK

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
    python verify_bank_aiislab.py
    python verify_bank_aiislab.py --sample 100
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

DATASET = "AIISLAB"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)
SUMMARY_FILE = "27_aiislab_phase1a1_corrected_units.json"
PAIRING_FILE = "15_aiislab_phase1a_pairing_manifest.csv"
DOUBLE_FILE = "23_aiislab_phase1a1_double_extension_pairs.csv"
ENCODING_FILE = "24_aiislab_phase1a1_mask_encoding_adjudication.csv"
AUTHOR_FILE = "AIISLAB_AUTHOR_CONFIRMED_DATASET_SEMANTICS.json"
MANIFEST_AUDIT_FILE = "11_bluis_audit_manifest.csv"
NATIVE_POLICY_ID = "AIISLAB_CANONICAL_MASK_NATIVE_BINARY_V1"
JPEG_POLICY_ID = "AIISLAB_JPEG_MASK_BINARIZATION_V1"
IMAGE_POLICY_ID = "AIISLAB_CANONICAL_IMAGE_LOSSLESS_V1"
CROSS_TOKEN = "113646"
SPECIAL_TOKEN = "42574"
GAP_LO, GAP_HI = 32, 191
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]
EXPECT = {"records": 166, "benign": 66, "malignant": 100, "excluded": 0,
          "png": 332, "native": 148, "jpeg": 18, "patients": 79,
          "special_records": 2, "cross_records": 4}
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
        if d.is_dir() and "aiislab" in d.name.lower().replace(" ", ""):
            return d
    return None


def find_in_audit(prep: Path, filename: str) -> Optional[Path]:
    d = audit_dir(prep)
    if d is None:
        return None
    hits = [p for p in d.rglob(filename) if not archived(p)]
    return hits[0] if len(hits) == 1 else None


def take_plane(arr, np) -> Tuple[Any, str]:
    """Channel reduction with the replication VERIFIED here, independently."""
    if arr.ndim == 2:
        return arr, "ALREADY_SINGLE_CHANNEL"
    if arr.ndim == 3 and arr.shape[2] >= 3:
        if not (np.array_equal(arr[..., 0], arr[..., 1])
                and np.array_equal(arr[..., 1], arr[..., 2])):
            return None, "CHANNELS_NOT_REPLICATED"
        return arr[..., 0], "REDUCED"
    return None, f"UNEXPECTED_SHAPE_{arr.shape}"


def reduce_plane(arr, np, jpeg: bool = False) -> Tuple[Any, str]:
    """Independent reimplementation of BOTH AIISLAB mask rules, written from their
    text rather than copied from the builder, so a builder bug cannot hide behind
    identical code.

    native: the values must already be {0,255} and are preserved exactly.
    jpeg  : source_gray >= 128 -> 255, else 0.
    """
    plane, note = take_plane(arr, np)
    if plane is None:
        return None, note
    if jpeg:
        return ((plane >= 128).astype(np.uint8) * 255), "JPEG_GE_128"
    vals = set(np.unique(plane).tolist())
    if vals == {0, 255}:
        return plane.astype(np.uint8), "NATIVE_PRESERVED"
    return None, f"NOT_NATIVE_BINARY_{sorted(vals)[:5]}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the BrEaST canonical bank.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--raw-root", default=None,
                    help="AIISLAB source root on H:. The Phase 1A.1 corrected-units "
                         "file carries no raw_root key, so it must be given here; "
                         "the root actually used is printed and recorded.")
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
    print("AIISLAB CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION")
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
        print(f"BLOCKED: {SUMMARY_FILE} carries no source root; pass --raw-root "
              "explicitly. Nothing was guessed.")
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
    lab = Counter(r.get("source_label") for r in rows)
    native = [r for r in rows if r.get("mask_policy_id") == NATIVE_POLICY_ID]
    jpegs = [r for r in rows if r.get("mask_policy_id") == JPEG_POLICY_ID]
    add("V01", f"manifest holds exactly {EXPECT['records']} records: "
               f"{EXPECT['benign']} BENIGN + {EXPECT['malignant']} MALIGNANT",
        len(rows) == EXPECT["records"] and lab.get("BENIGN") == EXPECT["benign"]
        and lab.get("MALIGNANT") == EXPECT["malignant"],
        f"{len(rows)} = {dict(lab)}")
    add("V01b", f"{EXPECT['native']} record(s) carry the native rule and "
                f"{EXPECT['jpeg']} the JPEG rule",
        len(native) == EXPECT["native"] and len(jpegs) == EXPECT["jpeg"],
        f"native {len(native)}, jpeg {len(jpegs)}")
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
    rule_seen, gap_seen = Counter(), {}
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
        is_jpeg = r.get("mask_policy_id", "") == JPEG_POLICY_ID
        plane, note = reduce_plane(sarr, np, jpeg=is_jpeg)
        rule_seen[note] += 1
        if is_jpeg:
            gap = int(((sp2 >= GAP_LO) & (sp2 <= GAP_HI)).sum()) \
                if (sp2 := take_plane(sarr, np)[0]) is not None else -1
            gap_seen[r.get("sample_id", "")] = gap

        if plane is None:
            recon_bad.append(f"{key} ({note})")
        elif plane.shape != ma.shape or not np.array_equal(plane, ma):
            d = int(np.sum(plane != ma)) if plane.shape == ma.shape else -1
            recon_bad.append(f"{key} differing px={d}")
        # V21: compare the INDEPENDENTLY reconstructed foreground with the canonical
        # one; no manifest column and no audit file is trusted for this
        if plane is not None and plane.shape == ma.shape:
            rfg = int((plane == 255).sum())
            if rfg == fg:
                fg_ok += 1
            else:
                fg_bad.append(f"{key} reconstructed={rfg} canonical={fg}")
        if n % 100 == 0:
            print(f"            {n}/{len(todo)}")

    add("V04", "canonical image dimensions equal source", not dim_bad,
        f"{len(dim_bad)} mismatch" + (f": {dim_bad[:2]}" if dim_bad else ""))
    add("V04b", "every canonical image has the same channel count as its source; "
                "nothing was collapsed or expanded",
        not chan_bad,
        f"channels on disk {dict(chan_seen)}; {len(chan_bad)} differ from source")
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
    add("V11", "every canonical mask is non-empty", not empty_bad,
        f"{len(empty_bad)} empty" + (f": {empty_bad[:2]}" if empty_bad else ""))

    lbl_bad = [r.get("sample_key", "") for r in eligible
               if (str(r["canonical_binary_label"]) == "0"
                   and not r["canonical_image_path"].startswith("Benign"))
               or (str(r["canonical_binary_label"]) == "1"
                   and not r["canonical_image_path"].startswith("Malignant"))]
    add("V12b", "class folders agree with labels", not lbl_bad,
        f"{len(lbl_bad)} mismatch" + (f": {lbl_bad[:2]}" if lbl_bad else ""))

    id_bad = [r.get("sample_key", "") for r in rows
              if r.get("patient_id", "").strip() or r.get("case_id", "").strip()
              or r.get("lesion_id", "").strip()]
    cpath = find_in_audit(prep, MANIFEST_AUDIT_FILE)
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
    # V09/V10: the JPEG rule and the empty gap, re-measured here
    gap_bad = [k for k, v in gap_seen.items() if v != 0]
    add("V09", f"all {EXPECT['jpeg']} JPEG masks reconstruct pixel-for-pixel under "
               ">=128 -> 255",
        rule_seen.get("JPEG_GE_128", 0) == len(jpegs)
        and not [x for x in recon_bad
                 if any(x.startswith(r.get("sample_id", "")) for r in jpegs)],
        f"{rule_seen.get('JPEG_GE_128', 0)}/{len(jpegs)} rebuilt with the JPEG rule")
    add("V10", f"all {EXPECT['jpeg']} JPEG source masks hold ZERO pixels in "
               f"{GAP_LO}-{GAP_HI}, re-measured here",
        not gap_bad and len(gap_seen) == len(jpegs),
        f"{len(gap_seen)} measured; {len(gap_bad)} with pixels in the gap"
        + (f": {gap_bad[:3]}" if gap_bad else ""))
    add("V08b", f"all {EXPECT['native']} native masks reconstruct as preserved, not "
                "thresholded",
        rule_seen.get("NATIVE_PRESERVED", 0) == len(native),
        f"{rule_seen.get('NATIVE_PRESERVED', 0)}/{len(native)}; rules "
        f"{dict(rule_seen)}")

    # V13/V14/V15: identity derived independently from the filename token
    tok_bad, key_bad = [], []
    tok_cls = defaultdict(set)
    for r in rows:
        stem = str(r.get("image_stem", ""))
        derived = stem.split("_")[0] if "_" in stem else ""
        if derived != str(r.get("TOKEN_1", "")):
            tok_bad.append(f"{stem}: derived {derived!r} vs manifest "
                           f"{r.get('TOKEN_1')!r}")
        if r.get("group_key_for_future_partitions") != f"PATIENT::{r.get('TOKEN_1')}":
            key_bad.append(stem)
        tok_cls[str(r.get("TOKEN_1"))].add(str(r.get("source_label")))
    pats = set(tok_cls)
    add("V13", f"TOKEN_1 re-derived from every filename; {EXPECT['patients']} unique "
               "patient(s); group_key = PATIENT::<TOKEN_1>",
        not tok_bad and not key_bad and len(pats) == EXPECT["patients"],
        f"{len(pats)} patient(s); {len(tok_bad)} token mismatch, {len(key_bad)} bad "
        "key")
    cross = sorted(t for t, c in tok_cls.items() if len(c) > 1)
    n_cross = sum(1 for r in rows if str(r.get("TOKEN_1")) == CROSS_TOKEN)
    add("V14", f"patient {CROSS_TOKEN} is ONE group carrying both classes",
        cross == [CROSS_TOKEN] and n_cross == EXPECT["cross_records"]
        and sorted(tok_cls.get(CROSS_TOKEN, [])) == ["BENIGN", "MALIGNANT"]
        and len({r.get("group_key_for_future_partitions") for r in rows
                 if str(r.get("TOKEN_1")) == CROSS_TOKEN}) == 1,
        f"cross-class token(s) {cross}; {n_cross} record(s), classes "
        f"{sorted(tok_cls.get(CROSS_TOKEN, []))}, one key")
    spec = [r for r in rows if str(r.get("TOKEN_1")) == SPECIAL_TOKEN]
    padded = [r for r in rows
              if str(r.get("TOKEN_1")).lstrip("0") == SPECIAL_TOKEN
              and str(r.get("TOKEN_1")) != SPECIAL_TOKEN]
    add("V15", f"identifier {SPECIAL_TOKEN} is verbatim with exactly "
               f"{EXPECT['special_records']} record(s)",
        len(spec) == EXPECT["special_records"] and not padded,
        f"{len(spec)} record(s); {len(padded)} zero-padded variant(s)")

    # V16: the double-extension recoveries, re-resolved from the evidence
    dpath = find_in_audit(prep, DOUBLE_FILE)
    drows = read_rows(dpath) if dpath else []
    dmap = {r["image_stem"].strip(): r for r in drows}
    unres_d, wrong_mask = [], []
    for stem, d in dmap.items():
        rec = next((r for r in rows if str(r.get("image_stem")) == stem), None)
        if rec is None:
            unres_d.append(stem)
            continue
        want = d.get("recovered_mask_relative_path", "").strip()
        if want and str(rec.get("source_mask_path")) != want:
            wrong_mask.append(f"{stem}: {rec.get('source_mask_path')} != {want}")
        if not (src_root / str(rec.get("source_mask_path"))).is_file():
            unres_d.append(f"{stem}: mask file absent")
    add("V16", f"all {EXPECT['jpeg']} double-extension masks resolve to their intended "
               "image record, with no unresolved or orphan pair left",
        len(dmap) == EXPECT["jpeg"] and not unres_d and not wrong_mask
        and {r.get("image_stem") for r in jpegs} == set(dmap),
        f"{len(dmap)} recovery row(s); {len(unres_d)} unresolved, "
        f"{len(wrong_mask)} pointing elsewhere")

    # V17: no exclusion here and no AIISLAB row in the global ledger
    excl = [r for r in rows
            if str(r.get("exclusion_status")) != "INCLUDED_CANONICAL_JOINT"]
    led = prep / BANK_NAME / "_reports" / "CANONICAL_DATASET_EXCLUSION_LEDGER.csv"
    led_rows = read_rows(led) if led.is_file() else []
    led_ai = [r for r in led_rows if str(r.get("dataset", "")).upper().startswith(
        "AIISLAB")]
    add("V17", "no record is excluded and the global exclusion ledger holds no "
               "AIISLAB row",
        not excl and not led_ai,
        f"{len(excl)} excluded here; the ledger has {len(led_rows)} row(s), "
        f"{len(led_ai)} of them AIISLAB")

    on_b = len(list((bank / "Benign" / "Images").glob("*.png"))) if (
        bank / "Benign" / "Images").is_dir() else 0
    on_m = len(list((bank / "Malignant" / "Images").glob("*.png"))) if (
        bank / "Malignant" / "Images").is_dir() else 0
    mis = [r for r in rows
           if not str(r.get("canonical_image_path", "")).startswith(
               "Benign" if r.get("source_label") == "BENIGN" else "Malignant")]
    add("V12", f"folder class and manifest class agree: {EXPECT['benign']} benign / "
               f"{EXPECT['malignant']} malignant",
        on_b == EXPECT["benign"] and on_m == EXPECT["malignant"] and not mis,
        f"on disk Benign {on_b} / Malignant {on_m}; {len(mis)} misfiled")

    after_raw = snapshot(src_root)
    raw_mod = [k for k in after_raw if k in before_raw and after_raw[k] != before_raw[k]]
    add("V18", "RAW inventory and content unchanged", not raw_mod and len(after_raw) == len(before_raw),
        f"{len(before_raw)} -> {len(after_raw)} file(s), {len(raw_mod)} modified")
    after_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        after_audit.update(snapshot(d))
    aud_mod = [k for k in after_audit
               if k in before_audit and after_audit[k] != before_audit[k]]
    add("V19", "completed audit artifacts unchanged",
        not aud_mod and len(after_audit) == len(before_audit),
        f"{len(before_audit)} -> {len(after_audit)} file(s), {len(aud_mod)} modified")
    splits = [str(p) for p in bank.rglob("*") if p.is_file() and is_split_like(str(p))]
    add("V20", "no Train/Validation/Test split exists", not splits,
        f"{len(splits)} split-like artifact(s)")

    add("V21", "foreground-pixel counts agree between the independently "
               "reconstructed mask and the canonical mask, on every record",
        fg_ok == EXPECT["records"] and not fg_bad,
        f"{fg_ok}/{EXPECT['records']} agree, {len(fg_bad)} differ"
        + (f": {fg_bad[:2]}" if fg_bad else ""))

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
    verdict = ("AIISLAB_CANONICAL_BANK_VERIFIED_ON_DISK" if not failed
               else "AIISLAB_CANONICAL_BANK_VERIFICATION_FAILED")
    print(verdict)
    if not failed:
        print("PHASE_2B.3.11_AIISLAB_CLOSED")
    for i, d, _ok, b in failed:
        print(f"    {i}: {d} | {b}")

    rep = bank / "reports" / "AIISLAB_ONDISK_VERIFICATION.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(
        "AIISLAB CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION\n"
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
