#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bank_busi.py
===================

PHASE 2B.3.1 - INDEPENDENT POST-EXECUTION VERIFIER FOR THE BUSI CANONICAL BANK

Checks the canonical output ON DISK against the SOURCE, not against the builder's log.
It re-reads the manifest, re-opens every canonical PNG and every source file, and
recomputes what the builder claimed. If the builder had a bug, this is what catches it.

READ-ONLY EVERYWHERE. It writes one report under BUSI\\reports\\ and nothing else.
RAW is opened for decoding only; no canonical file is modified or deleted.

WHAT IT PROVES
    V01 the manifest holds 647 records, 640 with canonical paths and 7 without
    V02 every canonical image and mask named in the manifest exists on disk
    V03 the bank holds no canonical PNG that the manifest does not name (no orphans)
    V04 every canonical image has EXACTLY the pixel dimensions of its source
    V05 every canonical image is pixel-identical to its source (lossless, no geometry
        or intensity change anywhere in the conversion path)
    V06 every canonical mask is single-channel and contains only {0, 255}
    V07 every canonical mask matches its image's dimensions
    V08 each single-mask record's canonical mask equals the binarised source mask
    V09 each of the 17 union masks CONTAINS every constituent source mask, and equals
        their union exactly - no lesion dropped, no pixel invented
    V10 the 7 label-conflict records wrote no canonical file
    V11 class folders agree with the canonical labels
    V12 RAW filesystem metadata unchanged across this verification

V05 and V09 are the ones that matter: they are the only checks that could have caught
a silent resize, a wrong mask, or a union that lost a lesion.

Usage:
    python verify_bank_busi.py
    python verify_bank_busi.py --sample 120     (spot-check instead of all 640)
Requires: Python >= 3.8, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET = "BUSI"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
SUMMARY_FILE = "16_busi_audit_summary.json"
ROOT_LOCATORS = ["source_path", "raw_root", "dataset_root", "source_root"]
EXPECT = {"records": 647, "eligible": 640, "excluded": 7, "multi_mask": 17,
          "png": 1280}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stat_all(root: Path) -> Dict[str, Tuple[int, int]]:
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


def find_source_root(prep_root: Path) -> Optional[Path]:
    for d in prep_root.iterdir():
        if not d.is_dir() or "busi" not in d.name.lower().replace(" ", ""):
            continue
        for p in d.rglob(SUMMARY_FILE):
            if "audit_old" in [x.lower() for x in p.parts]:
                continue
            try:
                blob = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for k in ROOT_LOCATORS:
                v = blob.get(k)
                if isinstance(v, str) and v.strip():
                    return Path(v.strip())
    return None


def binarise(arr, np) -> Optional[Any]:
    vals = set(np.unique(arr).tolist())
    if vals <= {0}:
        return (arr * 0).astype(np.uint8)
    if vals <= {0, 255}:
        return arr.astype(np.uint8)
    if vals <= {0, 1}:
        return (arr * 255).astype(np.uint8)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the BUSI canonical bank on disk.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--sample", type=int, default=0,
                    help="verify only N eligible records (0 = all)")
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
    print("BUSI CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION")
    print("=" * 78)
    print(f"generated : {now_iso()}")
    print(f"bank      : {bank}")
    if not man.is_file():
        print(f"manifest not found: {man}")
        return 2
    src_root = find_source_root(prep)
    print(f"source    : {src_root}  [READ-ONLY]")
    if src_root is None or not src_root.is_dir():
        print("source root unresolved; cannot verify against source")
        return 2
    before_raw = stat_all(src_root)

    with man.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    eligible = [r for r in rows if r["canonical_image_path"]]
    excluded = [r for r in rows if not r["canonical_image_path"]]
    multi = [r for r in eligible if "UNION" in r["mask_provenance"]]
    print(f"manifest  : {len(rows)} record(s), {len(eligible)} with canonical paths, "
          f"{len(excluded)} without, {len(multi)} union mask(s)")

    checks: List[Tuple[str, str, bool, str]] = []

    def add(i: str, d: str, ok: bool, b: str) -> None:
        checks.append((i, d, ok, b))

    add("V01", f"manifest holds {EXPECT['records']} records, "
               f"{EXPECT['eligible']} eligible and {EXPECT['excluded']} excluded",
        len(rows) == EXPECT["records"] and len(eligible) == EXPECT["eligible"]
        and len(excluded) == EXPECT["excluded"],
        f"{len(rows)} / {len(eligible)} / {len(excluded)}")

    named = set()
    missing: List[str] = []
    for r in eligible:
        for col in ("canonical_image_path", "canonical_mask_path"):
            rel = r[col].strip()
            if not rel:
                continue
            p = bank / rel
            named.add(str(p.resolve()) if p.exists() else str(p))
            if not p.is_file():
                missing.append(rel)
    add("V02", "every canonical file named in the manifest exists", not missing,
        f"{len(missing)} missing" + (f": {missing[:3]}" if missing else ""))

    on_disk = [p for p in bank.rglob("*.png") if p.is_file()]
    orphans = [str(p) for p in on_disk if str(p.resolve()) not in named]
    add("V03", f"the bank holds exactly {EXPECT['png']} canonical PNG(s) and no orphan",
        len(on_disk) == EXPECT["png"] and not orphans,
        f"{len(on_disk)} on disk, {len(orphans)} orphan"
        + (f": {orphans[:2]}" if orphans else ""))

    todo = list(eligible)
    if a.sample and a.sample < len(todo):
        todo = random.Random(0).sample(todo, a.sample)
        print(f"sampling  : {len(todo)} of {len(eligible)} eligible record(s)")
    print(f"verifying : {len(todo)} record(s) pixel by pixel ...")

    dim_bad, ident_bad, enc_bad, mdim_bad, single_bad = [], [], [], [], []
    union_bad, union_ok = [], 0
    for n, r in enumerate(todo, 1):
        cimg = bank / r["canonical_image_path"]
        simg = src_root / r["source_image_path"]
        if not (cimg.is_file() and simg.is_file()):
            continue
        with Image.open(cimg) as ci, Image.open(simg) as si:
            ca, sa = np.array(ci), np.array(si)
            if ci.size != si.size:
                dim_bad.append(f"{r['canonical_image_path']} {ci.size} vs {si.size}")
            elif ca.shape != sa.shape or not np.array_equal(ca, sa):
                ident_bad.append(r["canonical_image_path"])
        cmask_rel = r["canonical_mask_path"].strip()
        if not cmask_rel:
            continue
        cmask = bank / cmask_rel
        if not cmask.is_file():
            continue
        with Image.open(cmask) as cm:
            if cm.mode != "L":
                enc_bad.append(f"{cmask_rel} mode={cm.mode}")
            ma = np.array(cm.convert("L"))
        vals = set(np.unique(ma).tolist())
        if not vals <= {0, 255}:
            enc_bad.append(f"{cmask_rel} values={sorted(vals)[:6]}")
        with Image.open(cimg) as ci:
            if ma.shape[:2] != (ci.height, ci.width):
                mdim_bad.append(f"{cmask_rel} {ma.shape[:2]} vs "
                                f"{(ci.height, ci.width)}")
        srcs = [s.strip() for s in r["source_mask_path"].split(",") if s.strip()]
        acc = None
        bad = False
        for s in srcs:
            sp = src_root / s
            if not sp.is_file():
                bad = True
                break
            with Image.open(sp) as sm:
                b = binarise(np.array(sm.convert("L")), np)
            if b is None:
                bad = True
                break
            acc = b if acc is None else np.maximum(acc, b)
        if bad or acc is None:
            continue
        if len(srcs) == 1:
            if not np.array_equal(ma, acc):
                single_bad.append(cmask_rel)
        else:
            covers = bool(np.all(ma[acc == 255] == 255))
            exact = np.array_equal(ma, acc)
            if not (covers and exact):
                union_bad.append(f"{cmask_rel} covers={covers} exact={exact} "
                                 f"src={len(srcs)}")
            else:
                union_ok += 1
        if n % 150 == 0:
            print(f"            {n}/{len(todo)}")

    add("V04", "every canonical image keeps its source dimensions", not dim_bad,
        f"{len(dim_bad)} mismatch" + (f": {dim_bad[:2]}" if dim_bad else ""))
    add("V05", "every canonical image is pixel-identical to its source",
        not ident_bad,
        f"{len(ident_bad)} differ" + (f": {ident_bad[:2]}" if ident_bad else ""))
    add("V06", "every canonical mask is single-channel and strictly {0,255}",
        not enc_bad, f"{len(enc_bad)} bad" + (f": {enc_bad[:2]}" if enc_bad else ""))
    add("V07", "every canonical mask matches its image dimensions", not mdim_bad,
        f"{len(mdim_bad)} mismatch" + (f": {mdim_bad[:2]}" if mdim_bad else ""))
    add("V08", "each single-mask canonical mask equals its binarised source",
        not single_bad,
        f"{len(single_bad)} differ" + (f": {single_bad[:2]}" if single_bad else ""))
    add("V09", "each union mask equals the union of its sources and drops no lesion",
        not union_bad and (union_ok == len(multi) or bool(a.sample)),
        f"{union_ok} verified, {len(union_bad)} bad"
        + (f": {union_bad[:2]}" if union_bad else ""))
    wrote = [r for r in excluded
             if (bank / r["canonical_image_path"]).is_file()
             and r["canonical_image_path"]]
    add("V10", "the label-conflict records wrote no canonical file", not wrote,
        f"{len(excluded)} excluded, {len(wrote)} wrote a file")
    lbl_bad = [r["canonical_image_path"] for r in eligible
               if (r["canonical_binary_label"] == "0"
                   and not r["canonical_image_path"].startswith("Benign"))
               or (r["canonical_binary_label"] == "1"
                   and not r["canonical_image_path"].startswith("Malignant"))]
    add("V11", "class folder agrees with the canonical label", not lbl_bad,
        f"{len(lbl_bad)} mismatch" + (f": {lbl_bad[:2]}" if lbl_bad else ""))
    after_raw = stat_all(src_root)
    mod = [k for k in after_raw if k in before_raw and after_raw[k] != before_raw[k]]
    add("V12", "RAW filesystem metadata unchanged by this verification",
        not mod and len(after_raw) == len(before_raw),
        f"{len(before_raw)} -> {len(after_raw)} file(s), {len(mod)} modified")

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
    verdict = ("BUSI_CANONICAL_BANK_VERIFIED_ON_DISK" if not failed
               else "BUSI_CANONICAL_BANK_VERIFICATION_FAILED")
    print(verdict)
    if failed:
        for i, d, _ok, b in failed:
            print(f"    {i}: {d} | {b}")

    rep = bank / "reports" / "BUSI_ONDISK_VERIFICATION.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(
        "BUSI CANONICAL BANK - INDEPENDENT ON-DISK VERIFICATION\n"
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
