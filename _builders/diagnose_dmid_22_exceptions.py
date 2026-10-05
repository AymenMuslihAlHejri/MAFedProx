#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_dmid_22_exceptions.py
==============================

DIAGNOSIS ONLY. Nothing is written to the bank, nothing is rebuilt, no record is
excluded, no cohort count changes, the exclusion ledger is not touched and no
verification is weakened. H: is opened READ-ONLY.

It inspects exactly the 22 records whose image conversion did not complete:
    12  VERIFICATION_FAILED_NOT_IDENTICAL
    10  REFUSED_CHANNELS_NOT_REPLICATED

For each it reports the DICOM pixel metadata, the decoded array as it really is, what
the current writer did to it, and what a lossless canonical representation would be.
It also reproduces the writer's channel test verbatim, so the failure is demonstrated
rather than asserted.

One report is written into the dataset's reports folder.

Usage:
    python diagnose_dmid_22_exceptions.py
    python diagnose_dmid_22_exceptions.py --limit 4      # fewer files, faster
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
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
DEFAULT_RAW_ROOT = r"H:\BC datasets\Mammo\02 DMID"
BANK_NAME = "CANONICAL_DATASET_BANK"
DATASET = "DMID"
TARGET_STATUSES = ("VERIFICATION_FAILED_NOT_IDENTICAL",
                   "REFUSED_CHANNELS_NOT_REPLICATED")


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose the 22 DMID exceptions.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--limit", type=int, default=0,
                    help="inspect only the first N of each group")
    a = ap.parse_args(argv)
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

    print("=" * 100)
    print("DMID - DIAGNOSIS OF THE 22 IMAGE-CONVERSION EXCEPTIONS")
    print("=" * 100)
    print(f"generated : {now_iso()}")
    print(f"source    : {raw}  [READ-ONLY]")
    print("scope     : diagnosis only; nothing is written to the bank, no record is "
          "excluded, no count changes")
    if not man.is_file():
        print(f"manifest not found: {man}")
        return 2
    rows = read_rows(man)
    groups: Dict[str, List[dict]] = {s: [] for s in TARGET_STATUSES}
    for r in rows:
        st = r.get("image_conversion_status", "")
        if st in groups:
            groups[st].append(r)
    total = sum(len(v) for v in groups.values())
    print(f"manifest  : {len(rows)} rows; targets "
          + ", ".join(f"{k}={len(v)}" for k, v in groups.items())
          + f" = {total}")

    findings: List[dict] = []
    for status, recs in groups.items():
        sel = recs[:a.limit] if a.limit else recs
        print()
        print("-" * 100)
        print(f"{status}  ({len(recs)} record(s)"
              + (f", inspecting {len(sel)}" if a.limit else "") + ")")
        print("-" * 100)
        for r in sel:
            ref = r.get("image_reference", "")
            f: Dict[str, Any] = {"image_reference": ref, "manifest_status": status}
            sp = raw / r.get("source_image_path", "")
            if not sp.is_file():
                f["error"] = f"source missing: {r.get('source_image_path')}"
                findings.append(f)
                print(f"  {ref}: {f['error']}")
                continue
            ds = pydicom.dcmread(sp)
            f["SamplesPerPixel"] = int(getattr(ds, "SamplesPerPixel", -1))
            f["PhotometricInterpretation"] = str(
                getattr(ds, "PhotometricInterpretation", ""))
            f["PlanarConfiguration"] = (int(ds.PlanarConfiguration)
                                        if "PlanarConfiguration" in ds else None)
            f["BitsAllocated"] = int(getattr(ds, "BitsAllocated", -1))
            f["BitsStored"] = int(getattr(ds, "BitsStored", -1))
            f["HighBit"] = int(getattr(ds, "HighBit", -1))
            f["PixelRepresentation"] = int(getattr(ds, "PixelRepresentation", -1))
            f["Rows"], f["Columns"] = int(ds.Rows), int(ds.Columns)
            try:
                f["TransferSyntax"] = str(ds.file_meta.TransferSyntaxUID.name)
            except Exception:  # noqa: BLE001
                f["TransferSyntax"] = ""
            try:
                arr = ds.pixel_array
            except Exception as exc:  # noqa: BLE001
                f["decode_error"] = f"{type(exc).__name__}: {exc}"
                findings.append(f)
                print(f"  {ref}: decode failed - {f['decode_error']}")
                continue
            f["array_shape"] = list(arr.shape)
            f["array_ndim"] = int(arr.ndim)
            f["array_dtype"] = str(arr.dtype)
            f["array_min"], f["array_max"] = int(arr.min()), int(arr.max())

            # --- reproduce the writer's channel test EXACTLY as the builder ran it ---
            # the execute loop did: arr[..., 0] vs arr[..., 1] vs arr[..., 2]
            try:
                c0, c1, c2 = arr[..., 0], arr[..., 1], arr[..., 2]
                f["writer_slice_shape"] = list(np.shape(c0))
                writer_equal = bool(np.array_equal(c0, c1) and np.array_equal(c1, c2))
                f["writer_channel_test_result"] = writer_equal
                f["writer_slice_meaning"] = ("a real channel" if arr.ndim == 3
                                             else "A COLUMN, not a channel")
            except Exception as exc:  # noqa: BLE001
                f["writer_channel_test_result"] = f"raised {type(exc).__name__}"
                f["writer_slice_shape"] = None
                f["writer_slice_meaning"] = "indexing failed"

            # --- what the array actually is -----------------------------------------
            if arr.ndim == 2:
                f["true_nature"] = "SINGLE CHANNEL GREYSCALE (2-D)"
                f["channels_identical"] = "NOT_APPLICABLE"
                f["lossless_canonical"] = (
                    "write the 2-D array unchanged as an 8-bit PNG; nothing to reduce")
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                same = bool(np.array_equal(arr[..., 0], arr[..., 1])
                            and np.array_equal(arr[..., 1], arr[..., 2]))
                f["true_nature"] = f"{arr.shape[2]}-CHANNEL"
                f["channels_identical"] = same
                if not same:
                    d = int(np.sum((arr[..., 0] != arr[..., 1])
                                   | (arr[..., 1] != arr[..., 2])))
                    f["differing_pixels"] = d
                    f["lossless_canonical"] = (
                        "KEEP ALL CHANNELS: reduction would discard real colour")
                else:
                    f["lossless_canonical"] = (
                        "one channel is lossless: the three are provably identical")
            else:
                f["true_nature"] = f"UNEXPECTED shape {arr.shape}"
                f["lossless_canonical"] = "undetermined; inspect before converting"

            # --- what is on disk now, if anything -----------------------------------
            cp = r.get("canonical_image_path", "")
            dst = bank / cp if cp else None
            if dst is not None and dst.is_file():
                with Image.open(dst) as im:
                    f["written_png_size"] = [im.height, im.width]
                    f["written_png_mode"] = im.mode
                exp = (list(arr.shape[:2]) if arr.ndim >= 2 else None)
                f["written_png_matches_source_shape"] = (
                    f["written_png_size"] == exp)
                f["written_png_state"] = ("CORRUPT_ON_DISK"
                                          if f["written_png_size"] != exp else "OK")
            else:
                f["written_png_size"] = None
                f["written_png_state"] = "NOT_WRITTEN"

            # --- the exact cause ------------------------------------------------------
            if arr.ndim == 2 and status == "VERIFICATION_FAILED_NOT_IDENTICAL":
                f["exact_cause"] = (
                    "the execute loop indexes arr[..., 0] unconditionally. On this 2-D "
                    f"array that slice is a COLUMN of shape {f['writer_slice_shape']}, "
                    "not a channel. Columns 0, 1 and 2 happened to be equal (a uniform "
                    "border), so the channel test passed and the single COLUMN was "
                    "written as the image. The read-back then compared a "
                    f"{f['written_png_size']} PNG with the intended "
                    f"{list(arr.shape)} array and correctly reported a difference.")
                f["action"] = ("fix the writer to branch on ndim as the pre-check loop "
                               "already does; delete the corrupt PNG and rewrite")
            elif arr.ndim == 2 and status == "REFUSED_CHANNELS_NOT_REPLICATED":
                f["exact_cause"] = (
                    "same root cause: arr[..., 0] on a 2-D array is a COLUMN. Here "
                    "columns 0, 1 and 2 were NOT equal, so the writer's test failed and "
                    "the file was refused. The refusal is spurious: the image is "
                    "single-channel greyscale and needs no channel reduction at all.")
                f["action"] = ("fix the writer to branch on ndim; then write the 2-D "
                               "array unchanged")
            elif arr.ndim == 3 and f.get("channels_identical") is False:
                f["exact_cause"] = (
                    f"genuinely {arr.shape[2]}-channel with {f.get('differing_pixels')} "
                    "pixel(s) where the channels differ; the refusal is CORRECT")
                f["action"] = ("do NOT reduce; decide whether to keep all channels or "
                               "exclude, with evidence")
            else:
                f["exact_cause"] = "not explained by the ndim defect; inspect further"
                f["action"] = "hold"
            findings.append(f)

            print(f"  {ref}")
            print(f"     DICOM      : SamplesPerPixel={f['SamplesPerPixel']} "
                  f"Photometric={f['PhotometricInterpretation']} "
                  f"PlanarConfiguration={f['PlanarConfiguration']} "
                  f"BitsAllocated={f['BitsAllocated']} BitsStored={f['BitsStored']} "
                  f"PixelRepresentation={f['PixelRepresentation']}")
            print(f"     decoded    : shape={f['array_shape']} dtype={f['array_dtype']} "
                  f"range=[{f['array_min']},{f['array_max']}] -> {f['true_nature']}")
            print(f"     writer test: arr[...,0] has shape {f['writer_slice_shape']} "
                  f"({f['writer_slice_meaning']}); equality test returned "
                  f"{f['writer_channel_test_result']}")
            print(f"     on disk    : {f['written_png_state']} "
                  f"{f['written_png_size'] or ''}")
            print(f"     cause      : {f['exact_cause']}")
            print(f"     lossless   : {f['lossless_canonical']}")
            print(f"     action     : {f['action']}")
            print()

    print("=" * 100)
    print("COMPACT TABLE")
    print("=" * 100)
    hdr = (f"{'image':9s} {'SPP':>3s} {'photometric':13s} {'decoded shape':22s} "
           f"{'dtype':7s} {'range':12s} {'on disk':16s} cause")
    print(hdr)
    print("-" * len(hdr))
    for f in findings:
        if "array_shape" not in f:
            print(f"{f['image_reference']:9s} {'-':>3s} {'-':13s} "
                  f"{f.get('error', f.get('decode_error', ''))[:60]}")
            continue
        short = ("2-D indexed as 3-D by the writer"
                 if f["array_ndim"] == 2 else
                 "real multichannel, channels differ"
                 if f.get("channels_identical") is False else "other")
        print(f"{f['image_reference']:9s} {f['SamplesPerPixel']:>3d} "
              f"{f['PhotometricInterpretation'][:13]:13s} "
              f"{str(f['array_shape']):22s} {f['array_dtype']:7s} "
              f"[{f['array_min']},{f['array_max']}]".ljust(96 - len(short))
              + f" {f['written_png_state'][:16]:16s} {short}")

    print()
    print("SUMMARY")
    nd = Counter(f.get("array_ndim") for f in findings if "array_ndim" in f)
    spp = Counter(f.get("SamplesPerPixel") for f in findings if "SamplesPerPixel" in f)
    st = Counter(f.get("written_png_state") for f in findings)
    print(f"  decoded ndim        : {dict(nd)}")
    print(f"  SamplesPerPixel     : {dict(spp)}")
    print(f"  canonical PNG state : {dict(st)}")
    twod = [f for f in findings if f.get("array_ndim") == 2]
    real3 = [f for f in findings if f.get("channels_identical") is False]
    print(f"  explained by the 2-D indexing defect : {len(twod)} of {len(findings)}")
    print(f"  genuine multichannel refusals        : {len(real3)}")
    print()
    print("VERDICT")
    if len(twod) == len(findings):
        print("  ALL 22 are the same writer defect: the execute loop indexes")
        print("  arr[..., 0] without checking ndim, so on a 2-D greyscale array it")
        print("  takes a COLUMN instead of a channel. The pre-check loop already")
        print("  branches on ndim correctly, which is why the dry-run reported")
        print("  'already single channel: 22' and the execute loop did not.")
        print("  No record needs excluding and no cohort count changes.")
    else:
        print(f"  {len(twod)} are the writer defect; {len(real3)} are genuine "
              "multichannel refusals that need their own decision.")

    out = bank / "reports" / "DMID_22_EXCEPTION_DIAGNOSIS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generated": now_iso(), "raw_root": str(raw),
                               "scope": "diagnosis only; nothing written or rebuilt",
                               "findings": findings}, indent=2, ensure_ascii=False,
                              default=str), encoding="utf-8")
    print(f"\nWROTE {out}")
    print("bank files changed: NONE | RAW touched: NONE | records excluded: NONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
