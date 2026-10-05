#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bank_bus_uc.py
=====================

PHASE 2B.3.2 - INDEPENDENT POST-EXECUTION VERIFIER FOR THE BUS-UC CANONICAL BANK

Checks the canonical output ON DISK against the SOURCE and the LOCKED POLICY, never
against the builder's log or its report. It re-reads the manifest, re-opens every
canonical PNG and every source file, and recomputes the binarisation itself.

READ-ONLY EVERYWHERE. It writes one report under BUS_UC\\reports\\ and nothing else.
RAW is opened for decoding only; no canonical file is modified or deleted.

V01 manifest = 811 / eligible = 808 / excluded = 3
V02 every canonical path named in the manifest exists
V03 exactly 1616 canonical PNG files and no orphan
V04 canonical image dimensions equal source dimensions
V05 canonical image pixel content equals decoded source pixels
V06 masks are single-channel and strictly {0,255}
V07 mask and image dimensions agree
V08 every canonical mask independently reconstructs as 255 * (source_gray >= 128),
    with the threshold READ FROM THE LOCKED POLICY, not from this file
V09 all 808 canonical masks are non-empty
V10 the three integrity-excluded records wrote no canonical file
V11 class folders agree with the canonical labels
V12 no patient, case or lesion identity was invented
V13 no duplicate group or split lock was invented from screening candidates
V14 RAW unchanged
V15 completed audit artifacts unchanged
V16 no Train/Validation/Test split created

V05 and V08 are the decisive ones: they are the only checks that could catch a silent
resize, a wrong channel reduction, or a threshold applied differently from the policy.

Usage:
    python verify_bank_bus_uc.py
    python verify_bank_bus_uc.py --sample 150
Requires: Python >= 3.8, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET = "BUS_UC"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)
SUMMARY_FILE = "16_bus_uc_audit_summary.json"
POLICY_FILE = "BUS_UC_CANONICAL_MASK_POLICY_V1.json"
REQUIRED_POLICY_ID = "BUS_UC_CANONICAL_MASK_BINARIZATION_V1"
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]
EXPECT = {"records": 811, "eligible": 808, "excluded": 3, "png": 1616}
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


def find_source_root(prep: Path) -> Optional[Path]:
    for d in prep.iterdir():
        if not d.is_dir() or "busuc" not in d.name.lower().replace(" ", "").replace("_", ""):
            continue
        for p in d.rglob(SUMMARY_FILE):
            if archived(p):
                continue
            blob = json.loads(p.read_text(encoding="utf-8"))
            for k in ROOT_LOCATORS:
                v = blob.get(k)
                if isinstance(v, str) and v.strip():
                    return Path(v.strip())
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the BUS-UC canonical bank.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args(argv)
    prep = Path(a.prep_root)
    bank = prep / BANK_NAME / DATASET
    man = bank / "manifests" / "canonical_manifest.csv"
    pol_path = prep / BANK_NAME / "_policies" / POLICY_FILE

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        sys.stderr.write("Pillow and numpy required\n")
        return 3
    Image.MAX_IMAGE_PIXELS = None

    print("=" * 78)
    print("BUS-UC CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION")
    print("=" * 78)
    print(f"generated : {now_iso()}")
    print(f"bank      : {bank}")
    if not man.is_file():
        print(f"manifest not found: {man}")
        return 2
    if not pol_path.is_file():
        print(f"locked policy not found: {pol_path}")
        return 2
    pol = json.loads(pol_path.read_text(encoding="utf-8"))
    if pol.get("policy_id") != REQUIRED_POLICY_ID:
        print(f"unexpected policy_id {pol.get('policy_id')!r}")
        return 2
    THRESHOLD = int(pol["canonical_threshold"])
    src_root = find_source_root(prep)
    print(f"source    : {src_root}  [READ-ONLY]")
    print(f"policy    : {pol['policy_id']} threshold={THRESHOLD} "
          f"({pol.get('threshold_provenance')})")
    if src_root is None or not src_root.is_dir():
        print("source root unresolved; cannot verify against source")
        return 2
    before_raw = snapshot(src_root)
    prep_dirs = [d for d in prep.iterdir() if d.is_dir() and d.name != BANK_NAME]
    before_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        before_audit.update(snapshot(d))

    with man.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    eligible = [r for r in rows if r["canonical_image_path"]]
    excluded = [r for r in rows if not r["canonical_image_path"]]
    print(f"manifest  : {len(rows)} records, {len(eligible)} eligible, "
          f"{len(excluded)} excluded")

    checks: List[Tuple[str, str, bool, str]] = []

    def add(i: str, d: str, ok: bool, b: str) -> None:
        checks.append((i, d, ok, b))

    add("V01", f"manifest {EXPECT['records']} / eligible {EXPECT['eligible']} / "
               f"excluded {EXPECT['excluded']}",
        len(rows) == EXPECT["records"] and len(eligible) == EXPECT["eligible"]
        and len(excluded) == EXPECT["excluded"],
        f"{len(rows)} / {len(eligible)} / {len(excluded)}")

    named, missing = set(), []
    for r in eligible:
        for col in ("canonical_image_path", "canonical_mask_path"):
            rel = r[col].strip()
            if not rel:
                missing.append(f"{r['sample_key']}:{col} empty")
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

    dim_bad, ident_bad, enc_bad, mdim_bad, recon_bad, empty_bad = [], [], [], [], [], []
    for n, r in enumerate(todo, 1):
        cimg, simg = bank / r["canonical_image_path"], src_root / r["source_image_path"]
        if not (cimg.is_file() and simg.is_file()):
            continue
        with Image.open(cimg) as ci, Image.open(simg) as si:
            ca, sa = np.array(ci), np.array(si)
            if ci.size != si.size:
                dim_bad.append(f"{r['sample_key']} {ci.size} vs {si.size}")
            elif ca.shape != sa.shape or not np.array_equal(ca, sa):
                ident_bad.append(r["sample_key"])
            iw, ih = ci.width, ci.height
        cmask = bank / r["canonical_mask_path"]
        smask = src_root / r["source_mask_path"]
        if not (cmask.is_file() and smask.is_file()):
            continue
        with Image.open(cmask) as cm:
            if cm.mode != "L":
                enc_bad.append(f"{r['sample_key']} mode={cm.mode}")
            ma = np.array(cm.convert("L"))
            mw, mh = cm.width, cm.height
        vals = set(np.unique(ma).tolist())
        if not vals <= {0, 255}:
            enc_bad.append(f"{r['sample_key']} values={sorted(vals)[:6]}")
        if (mh, mw) != (ih, iw):
            mdim_bad.append(f"{r['sample_key']} mask {(mh, mw)} vs image {(ih, iw)}")
        if int((ma == 255).sum()) == 0:
            empty_bad.append(r["sample_key"])
        # independent reconstruction, using the POLICY threshold
        with Image.open(smask) as sm:
            sarr = np.array(sm)
        gray = sarr[..., 0] if sarr.ndim == 3 else sarr
        expect = (255 * (gray >= THRESHOLD)).astype(np.uint8)
        if expect.shape != ma.shape or not np.array_equal(expect, ma):
            diff = (int(np.sum(expect != ma)) if expect.shape == ma.shape else -1)
            recon_bad.append(f"{r['sample_key']} differing px={diff}")
        if n % 150 == 0:
            print(f"            {n}/{len(todo)}")

    add("V04", "canonical image dimensions equal source", not dim_bad,
        f"{len(dim_bad)} mismatch" + (f": {dim_bad[:2]}" if dim_bad else ""))
    add("V05", "canonical image pixels equal decoded source pixels", not ident_bad,
        f"{len(ident_bad)} differ" + (f": {ident_bad[:2]}" if ident_bad else ""))
    add("V06", "masks single-channel and strictly {0,255}", not enc_bad,
        f"{len(enc_bad)} bad" + (f": {enc_bad[:2]}" if enc_bad else ""))
    add("V07", "mask and image dimensions agree", not mdim_bad,
        f"{len(mdim_bad)} mismatch" + (f": {mdim_bad[:2]}" if mdim_bad else ""))
    add("V08", f"every mask reconstructs as 255 * (source_gray >= {THRESHOLD})",
        not recon_bad,
        f"{len(todo) - len(recon_bad)} reconstructed exactly, {len(recon_bad)} differ"
        + (f": {recon_bad[:2]}" if recon_bad else ""))
    add("V09", "every canonical mask is non-empty", not empty_bad,
        f"{len(empty_bad)} empty" + (f": {empty_bad[:2]}" if empty_bad else ""))

    wrote = [r["sample_key"] for r in excluded
             if r["canonical_image_path"].strip() or r["canonical_mask_path"].strip()]
    statuses = {r["sample_key"]: r["exclusion_status"] for r in excluded}
    add("V10", "the three excluded records wrote no canonical file", not wrote,
        f"{len(excluded)} excluded: {statuses}")

    lbl_bad = [r["sample_key"] for r in eligible
               if (str(r["canonical_binary_label"]) == "0"
                   and not r["canonical_image_path"].startswith("Benign"))
               or (str(r["canonical_binary_label"]) == "1"
                   and not r["canonical_image_path"].startswith("Malignant"))]
    add("V11", "class folders agree with canonical labels", not lbl_bad,
        f"{len(lbl_bad)} mismatch" + (f": {lbl_bad[:2]}" if lbl_bad else ""))

    id_bad = [r["sample_key"] for r in rows
              if r.get("patient_id", "").strip() or r.get("case_id", "").strip()
              or r.get("lesion_id", "").strip()]
    add("V12", "no patient, case or lesion identity invented", not id_bad,
        f"{len(id_bad)} record(s) carry an identity")

    cols = list(rows[0]) if rows else []
    grp_cols = [c for c in cols
                if "duplicate_group" in c.lower() or "group_lock" in c.lower()
                or "split_group" in c.lower()]
    cand_notes = sum(1 for r in rows
                     if "SCREENING_CANDIDATE_ONLY" in r.get("notes", ""))
    add("V13", "no duplicate group or split lock invented from screening candidates",
        not grp_cols,
        f"no group column in the manifest; {cand_notes} candidate flag(s) preserved "
        "in notes only")

    after_raw = snapshot(src_root)
    raw_mod = [k for k in after_raw if k in before_raw and after_raw[k] != before_raw[k]]
    add("V14", "RAW unchanged", not raw_mod and len(after_raw) == len(before_raw),
        f"{len(before_raw)} -> {len(after_raw)} file(s), {len(raw_mod)} modified")
    after_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        after_audit.update(snapshot(d))
    aud_mod = [k for k in after_audit
               if k in before_audit and after_audit[k] != before_audit[k]]
    add("V15", "completed audit artifacts unchanged",
        not aud_mod and len(after_audit) == len(before_audit),
        f"{len(before_audit)} -> {len(after_audit)} file(s), {len(aud_mod)} modified")
    splits = [str(p) for p in bank.rglob("*") if p.is_file() and is_split_like(str(p))]
    add("V16", "no Train/Validation/Test split created", not splits,
        f"{len(splits)} split-like artifact(s)"
        + (f": {splits[:2]}" if splits else ""))

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
    verdict = ("BUS_UC_CANONICAL_BANK_VERIFIED_ON_DISK" if not failed
               else "BUS_UC_CANONICAL_BANK_VERIFICATION_FAILED")
    print(verdict)
    for i, d, _ok, b in failed:
        print(f"    {i}: {d} | {b}")

    rep = bank / "reports" / "BUS_UC_ONDISK_VERIFICATION.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(
        "BUS-UC CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION\n"
        f"generated: {now_iso()}\nsource: {src_root}\nbank: {bank}\n"
        f"policy: {pol['policy_id']} threshold={THRESHOLD}\n"
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
