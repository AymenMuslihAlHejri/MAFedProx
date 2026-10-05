#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_aiislab_18_masks.py
============================

DIAGNOSIS ONLY. Nothing is written to the bank, no canonical mask is produced, no
threshold is chosen, no record is excluded and RAW is opened READ-ONLY.

It examines exactly the 18 AIISLAB masks the closed audit adjudicated as
JPEG_REENCODED_MASK_WITH_INTERMEDIATE_PIXEL_VALUES, and answers one question with
measurements rather than opinion: are the intermediate values confined to a thin
boundary band produced by JPEG re-encoding, or have they spread far enough that the
original binary boundary is genuinely unrecoverable?

WHAT IT MEASURES, PER MASK
    min, max, the full list of unique values, and a histogram
    counts and proportions at exactly 0 and exactly 255
    counts in 1-31, 32-63, 64-127, 128-191, 192-254
    the number and proportion of intermediate-valued pixels
    how far each intermediate pixel sits from the >=128 boundary, which is what
        decides whether the artefact is a boundary band or a spread
    connected components at thresholds 1, 64, 128, 192, 254
    foreground area at each threshold
    Dice between every pair of thresholds
    the boundary displacement implied by the area change between thresholds

WHAT IT DOES NOT DO
    it does not pick a threshold, does not write a mask, does not modify RAW, and does
    not declare the records eligible or ineligible. That decision stays with the
    reader, who will have the numbers.

Usage:
    python diagnose_aiislab_18_masks.py --raw-root "H:\\BC datasets\\Mammo\\04 AIISLAB"
    python diagnose_aiislab_18_masks.py --raw-root ... --limit 3   # fewer, faster
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
BANK_NAME = "CANONICAL_DATASET_BANK"
DATASET_DIR = "AIISLAB"
ENCODING_FILE = "24_aiislab_phase1a1_mask_encoding_adjudication.csv"
DOUBLE_FILE = "23_aiislab_phase1a1_double_extension_pairs.csv"
THRESHOLDS = (1, 64, 128, 192, 254)
BANDS = (("0", 0, 0), ("1-31", 1, 31), ("32-63", 32, 63), ("64-127", 64, 127),
         ("128-191", 128, 191), ("192-254", 192, 254), ("255", 255, 255))


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


def find_in_audit(prep: Path, name: str) -> Optional[Path]:
    for d in prep.iterdir():
        if d.is_dir() and "aiislab" in d.name.lower().replace(" ", ""):
            for p in d.rglob(name):
                if p.is_file():
                    return p
    return None


def label_components(bw, np) -> int:
    """Connected components, 8-connected. scipy if present, else an iterative
    flood fill so the number is still produced rather than skipped."""
    try:
        from scipy import ndimage
        st = np.ones((3, 3), dtype=bool)
        _lab, n = ndimage.label(bw, structure=st)
        return int(n)
    except ImportError:
        pass
    h, w = bw.shape
    seen = np.zeros_like(bw, dtype=bool)
    n = 0
    for y in range(h):
        row = bw[y]
        for x in range(w):
            if not row[x] or seen[y, x]:
                continue
            n += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and bw[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return n


def boundary_distance_profile(plane, np) -> Dict[str, Any]:
    """How far do the intermediate pixels sit from the >=128 boundary? A JPEG ringing
    artefact hugs the edge; a genuinely lost boundary does not."""
    core = plane >= 128
    inter = (plane > 0) & (plane < 255)
    n_inter = int(inter.sum())
    if n_inter == 0:
        return {"intermediate_pixels": 0}
    try:
        from scipy import ndimage
        # distance from every pixel to the nearest core pixel and to the nearest
        # background pixel; the boundary band is where both are small
        d_out = ndimage.distance_transform_edt(~core)
        d_in = ndimage.distance_transform_edt(core)
        dist = np.where(core, d_in, d_out)
        d_inter = dist[inter]
        return {"intermediate_pixels": n_inter,
                "distance_to_boundary_mean": round(float(d_inter.mean()), 3),
                "distance_to_boundary_p50": round(float(np.percentile(d_inter, 50)), 3),
                "distance_to_boundary_p95": round(float(np.percentile(d_inter, 95)), 3),
                "distance_to_boundary_max": round(float(d_inter.max()), 3),
                "within_1px_of_boundary": round(float((d_inter <= 1).mean()), 6),
                "within_2px_of_boundary": round(float((d_inter <= 2).mean()), 6),
                "within_3px_of_boundary": round(float((d_inter <= 3).mean()), 6),
                "method": "scipy distance transform"}
    except ImportError:
        # a cheap dilation-based band test, enough to separate a boundary band from a
        # spread even without scipy
        band = np.zeros_like(core)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                band |= np.roll(np.roll(core, dy, 0), dx, 1)
        band &= ~core
        near = int((inter & (band | core)).sum())
        return {"intermediate_pixels": n_inter,
                "within_1px_of_boundary": round(near / n_inter, 6),
                "method": "roll-based 1px band (scipy not installed)"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose the 18 AIISLAB masks.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--raw-root", required=True,
                    help="AIISLAB source root on H: (read-only)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    prep, raw = Path(a.prep_root), Path(a.raw_root)

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        sys.stderr.write(f"missing dependency: {exc}\n")
        return 3
    Image.MAX_IMAGE_PIXELS = None

    print("=" * 94)
    print("AIISLAB - DIAGNOSIS OF THE 18 JPEG-RE-ENCODED MASKS")
    print("=" * 94)
    print(f"generated : {now_iso()}")
    print(f"source    : {raw}  [READ-ONLY]")
    print("scope     : measurement only; no threshold chosen, no mask written, no "
          "record excluded")
    if not raw.is_dir():
        print(f"BLOCKED: source root does not exist: {raw}")
        return 2
    ep = find_in_audit(prep, ENCODING_FILE)
    if ep is None:
        print(f"BLOCKED: {ENCODING_FILE} not found")
        return 2
    rows = read_rows(ep)
    dp = find_in_audit(prep, DOUBLE_FILE)
    doubles = {r["image_stem"]: r for r in read_rows(dp)} if dp else {}
    print(f"evidence  : {ep.name} ({len(rows)} row(s))"
          + (f" + {dp.name}" if dp else ""))
    sel = rows[:a.limit] if a.limit else rows
    print(f"examining : {len(sel)} mask(s)\n")

    findings: List[dict] = []
    for i, r in enumerate(sel, 1):
        stem = r["image_stem"]
        rel = r.get("mask_relative_path", "").strip()
        mp = raw / rel
        f: Dict[str, Any] = {"image_stem": stem, "mask_relative_path": rel,
                             "audit_encoding": r.get("encoding_from_phase1a", ""),
                             "audit_unique_value_count": r.get("unique_value_count"),
                             "audit_pixels_outside_0_255":
                                 r.get("pixels_outside_0_255"),
                             "adjudication": r.get("adjudication", "")}
        if not mp.is_file():
            f["error"] = f"mask not found: {rel}"
            findings.append(f)
            print(f"  [{i}/{len(sel)}] {stem}: {f['error']}")
            continue
        with Image.open(mp) as im:
            arr = np.asarray(im)
            f["pil_mode"], f["shape"] = im.mode, list(arr.shape)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[..., :3]
            f["channels_identical"] = bool(
                np.array_equal(rgb[..., 0], rgb[..., 1])
                and np.array_equal(rgb[..., 1], rgb[..., 2]))
            plane = rgb[..., 0]
        elif arr.ndim == 2:
            f["channels_identical"] = "SINGLE_CHANNEL"
            plane = arr
        else:
            f["error"] = f"unexpected shape {arr.shape}"
            findings.append(f)
            continue
        total = int(plane.size)
        vals = np.unique(plane)
        f["min"], f["max"] = int(plane.min()), int(plane.max())
        f["unique_values"] = [int(v) for v in vals]
        f["unique_value_count"] = int(vals.size)
        hist = np.bincount(plane.ravel(), minlength=256)
        f["histogram_nonzero"] = {int(v): int(hist[v]) for v in vals}
        f["total_pixels"] = total
        f["count_0"], f["count_255"] = int(hist[0]), int(hist[255])
        f["prop_0"] = round(f["count_0"] / total, 8)
        f["prop_255"] = round(f["count_255"] / total, 8)
        bands = {}
        for name, lo, hi in BANDS:
            c = int(hist[lo:hi + 1].sum())
            bands[name] = {"count": c, "proportion": round(c / total, 8)}
        f["bands"] = bands
        inter = int(hist[1:255].sum())
        f["intermediate_pixels"] = inter
        f["intermediate_proportion"] = round(inter / total, 8)
        f["boundary_profile"] = boundary_distance_profile(plane, np)

        per_t = {}
        masks = {}
        for t in THRESHOLDS:
            bw = plane >= t
            area = int(bw.sum())
            masks[t] = bw
            per_t[t] = {"foreground_pixels": area,
                        "foreground_fraction": round(area / total, 8),
                        "connected_components": label_components(bw, np)}
        base = per_t[128]["foreground_pixels"]
        for t in THRESHOLDS:
            per_t[t]["area_change_vs_t128"] = (
                round((per_t[t]["foreground_pixels"] - base) / base, 6)
                if base else None)
        f["thresholds"] = {str(k): v for k, v in per_t.items()}

        dice, disagree = {}, {}
        for x in range(len(THRESHOLDS)):
            for y in range(x + 1, len(THRESHOLDS)):
                t1, t2 = THRESHOLDS[x], THRESHOLDS[y]
                a1, a2 = masks[t1], masks[t2]
                inter_n = int((a1 & a2).sum())
                s = int(a1.sum()) + int(a2.sum())
                dice[f"{t1}_vs_{t2}"] = round(2 * inter_n / s, 6) if s else None
                d = int((a1 ^ a2).sum())
                disagree[f"{t1}_vs_{t2}"] = {
                    "disagreeing_pixels": d,
                    "as_fraction_of_t128_area": (round(d / base, 6) if base
                                                 else None)}
        f["dice_between_thresholds"] = dice
        f["disagreement_between_thresholds"] = disagree
        # an equivalent-radius reading of the area change: how far the boundary would
        # move if the region were a disc of the same area
        if base:
            import math
            r128 = math.sqrt(base / math.pi)
            f["equivalent_boundary_shift_px"] = {
                str(t): round(math.sqrt(per_t[t]["foreground_pixels"] / math.pi)
                              - r128, 3) for t in THRESHOLDS}
        findings.append(f)

        bp = f["boundary_profile"]
        print(f"  [{i}/{len(sel)}] {stem}  ({f['audit_encoding']})")
        print(f"      shape {f['shape']} mode {f['pil_mode']} channels_identical="
              f"{f['channels_identical']}")
        print(f"      range [{f['min']},{f['max']}]  {f['unique_value_count']} unique "
              f"value(s): {f['unique_values'][:12]}"
              + (" ..." if f["unique_value_count"] > 12 else ""))
        print(f"      0: {f['count_0']:,} ({f['prop_0']:.6f})   255: "
              f"{f['count_255']:,} ({f['prop_255']:.6f})")
        print("      bands: " + "  ".join(
            f"{n}={bands[n]['count']:,}" for n, _l, _h in BANDS))
        print(f"      intermediate: {inter:,} ({f['intermediate_proportion']:.6f} of "
              "the image)")
        if bp.get("intermediate_pixels"):
            if "distance_to_boundary_mean" in bp:
                print(f"      distance to the >=128 boundary: mean "
                      f"{bp['distance_to_boundary_mean']}, p95 "
                      f"{bp['distance_to_boundary_p95']}, max "
                      f"{bp['distance_to_boundary_max']} px")
            print(f"      within 1px of the boundary: "
                  f"{bp.get('within_1px_of_boundary')}"
                  + (f" | 2px: {bp.get('within_2px_of_boundary')}"
                     f" | 3px: {bp.get('within_3px_of_boundary')}"
                     if "within_2px_of_boundary" in bp else ""))
        print("      threshold  area          frac       comps   area vs t128")
        for t in THRESHOLDS:
            v = per_t[t]
            print(f"        {t:>3}      {v['foreground_pixels']:>10,}  "
                  f"{v['foreground_fraction']:.6f}  {v['connected_components']:>5}   "
                  f"{v['area_change_vs_t128']:+.6f}"
                  if v["area_change_vs_t128"] is not None else f"        {t:>3}")
        print(f"      Dice 1v255-ish: 1_vs_254={dice.get('1_vs_254')}  "
              f"64_vs_192={dice.get('64_vs_192')}  128_vs_192={dice.get('128_vs_192')}"
              f"  64_vs_128={dice.get('64_vs_128')}")
        if "equivalent_boundary_shift_px" in f:
            print("      equivalent boundary shift vs t128 (px): "
                  + ", ".join(f"t{t}={f['equivalent_boundary_shift_px'][str(t)]:+}"
                              for t in THRESHOLDS))
        print()

    ok = [f for f in findings if "error" not in f and "bands" in f]
    if ok:
        print("=" * 94)
        print("SUMMARY ACROSS THE MASKS EXAMINED")
        print("=" * 94)
        ip = [f["intermediate_proportion"] for f in ok]
        print(f"  intermediate-pixel proportion: min {min(ip):.8f}  max {max(ip):.8f}"
              f"  mean {sum(ip) / len(ip):.8f}")
        d1 = [f["dice_between_thresholds"].get("1_vs_254") for f in ok
              if f["dice_between_thresholds"].get("1_vs_254") is not None]
        d2 = [f["dice_between_thresholds"].get("64_vs_192") for f in ok
              if f["dice_between_thresholds"].get("64_vs_192") is not None]
        d3 = [f["dice_between_thresholds"].get("128_vs_192") for f in ok
              if f["dice_between_thresholds"].get("128_vs_192") is not None]
        if d1:
            print(f"  Dice t1 vs t254   : min {min(d1):.6f}  max {max(d1):.6f}")
        if d2:
            print(f"  Dice t64 vs t192  : min {min(d2):.6f}  max {max(d2):.6f}")
        if d3:
            print(f"  Dice t128 vs t192 : min {min(d3):.6f}  max {max(d3):.6f}")
        w1 = [f["boundary_profile"].get("within_1px_of_boundary") for f in ok
              if f["boundary_profile"].get("within_1px_of_boundary") is not None]
        if w1:
            print(f"  intermediate pixels within 1px of the >=128 boundary: "
                  f"min {min(w1):.6f}  max {max(w1):.6f}")
        shifts = [abs(v) for f in ok
                  for k, v in f.get("equivalent_boundary_shift_px", {}).items()
                  if k != "128"]
        if shifts:
            print(f"  equivalent boundary shift across thresholds: max "
                  f"{max(shifts):.3f} px")
        comps = Counter()
        for f in ok:
            for t in THRESHOLDS:
                comps[t] = max(comps[t], f["thresholds"][str(t)]
                               ["connected_components"])
        print(f"  max connected components at each threshold: {dict(comps)}")
        print()
        print("  READ THIS AS: if the intermediate pixels sit within a pixel or two of")
        print("  the boundary, if Dice between distant thresholds stays very high, and")
        print("  if the component count does not change, then the artefact is a thin")
        print("  JPEG edge band and the choice of threshold moves the boundary by less")
        print("  than a pixel. If instead Dice falls, components appear or disappear,")
        print("  or intermediate pixels sit far from the edge, the original boundary")
        print("  is not recoverable and no threshold can be defended.")
        print()
        print("  NO THRESHOLD HAS BEEN CHOSEN. NO MASK HAS BEEN WRITTEN.")

    out = prep / BANK_NAME / DATASET_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)
    op = out / "AIISLAB_18_MASK_DIAGNOSIS.json"
    op.write_text(json.dumps({"generated": now_iso(), "raw_root": str(raw),
                              "scope": "diagnosis only; no threshold chosen, no mask "
                                       "written, RAW untouched",
                              "thresholds_tested": list(THRESHOLDS),
                              "findings": findings}, indent=2, ensure_ascii=False,
                             default=str), encoding="utf-8")
    print(f"WROTE {op}")
    print("RAW modified: FALSE | canonical masks written: 0 | records excluded: 0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
