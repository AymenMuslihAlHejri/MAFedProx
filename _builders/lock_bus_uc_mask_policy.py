#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lock_bus_uc_mask_policy.py
==========================

PHASE 2B.3.2 - BUS-UC CANONICAL MASK BINARIZATION POLICY (immutable artifact)

Writes ONE policy artifact under the canonical policy area and nothing else.
No RAW is touched, no audit artifact is modified, no image is read or written.

The figures in the policy text are ASSERTIONS. This tool re-derives the mask-encoding
facts from the completed audit artifacts and refuses to write if they do not hold.
The threshold itself is NOT derived from evidence and is not claimed to be: it is a
Phase 2B curation decision, recorded as such.

IMMUTABLE: if the artifact already exists with different content the tool refuses
rather than overwriting; a changed policy needs its own version id.

Usage:
    python lock_bus_uc_mask_policy.py --dry-run
    python lock_bus_uc_mask_policy.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)
POLICY_ID = "BUS_UC_CANONICAL_MASK_BINARIZATION_V1"
POLICY_FILE = "BUS_UC_CANONICAL_MASK_POLICY_V1.json"
THRESHOLD = 128

EVIDENCE = [
    ("16_bus_uc_audit_summary.json", "source inventory and mask encoding distribution"),
    ("05_bus_uc_mask_audit.csv", "per-mask dtype, channels, encoding, emptiness"),
    ("20_bus_uc_mask_encoding_profile.json", "soft-boundary characterisation"),
    ("19_bus_uc_mask_integrity_cases.csv", "the three mask-integrity cases"),
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def archived(p: Path) -> bool:
    return any(x.lower() in ARCHIVE_DIR_NAMES for x in p.parts[:-1])


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_rows(p: Path) -> List[dict]:
    with p.open(encoding="utf-8-sig", errors="replace") as fh:
        s = fh.read(8192); fh.seek(0)
        d = ";" if s.count(";") > s.count(",") else ","
        return list(csv.DictReader(fh, delimiter=d))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Lock the BUS-UC mask policy.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    prep = Path(a.prep_root)
    policy_dir = prep / BANK_NAME / "_policies"

    print("=" * 78)
    print("PHASE 2B.3.2 - BUS-UC CANONICAL MASK BINARIZATION POLICY")
    print("=" * 78)
    print(f"mode: {'DRY-RUN (writes nothing)' if a.dry_run else 'LOCK'}")

    dirs = [d for d in prep.iterdir()
            if d.is_dir() and "busuc" in d.name.lower().replace(" ", "").replace("_", "")]
    files = [p for d in dirs for p in d.rglob("*") if p.is_file() and not archived(p)]
    found: Dict[str, Path] = {}
    problems: List[str] = []
    print("\n--- evidence artifacts ---")
    for name, purpose in EVIDENCE:
        hits = sorted(p for p in files if p.name.lower() == name.lower())
        if len(hits) == 1:
            (only,) = hits
            found[name] = only
            print(f"  UNIQUE     {name:44s} {purpose}")
        else:
            problems.append(f"{name} -> {'MISSING' if not hits else 'AMBIGUOUS'}")
            print(f"  {'MISSING' if not hits else 'AMBIGUOUS':10s} {name}")

    verified: Dict[str, Any] = {}
    if "05_bus_uc_mask_audit.csv" in found:
        rows = read_rows(found["05_bus_uc_mask_audit.csv"])
        dt = Counter(r.get("dtype", "") for r in rows)
        ch = Counter(r.get("channels", "") for r in rows)
        rep = Counter(str(r.get("rgb_channels_identical", "")).upper() for r in rows)
        binr = Counter(str(r.get("is_binary_encoding", "")).upper() for r in rows)
        empt = Counter(str(r.get("empty_mask", "")).upper() for r in rows)
        verified.update(masks_profiled=len(rows), dtype=dict(dt), channels=dict(ch),
                        rgb_identical=dict(rep), is_binary_encoding=dict(binr),
                        empty_mask=dict(empt))
        if list(dt) != ["uint8"]:
            problems.append(f"masks are not uniformly uint8: {dict(dt)}")
        if binr.get("NO", 0) != len(rows):
            problems.append("some masks are already strictly binary; the soft-boundary "
                            "premise of this policy would not hold for them")
    if "20_bus_uc_mask_encoding_profile.json" in found:
        prof = json.loads(found["20_bus_uc_mask_encoding_profile.json"]
                          .read_text(encoding="utf-8"))
        verified.update(
            encoding_verdict=prof.get("encoding_verdict"),
            masks_with_only_0_and_255=prof.get("masks_with_only_0_and_255"),
            masks_with_intermediate_values=prof.get("masks_with_intermediate_values"),
            intermediate_boundary_fraction_min=(
                prof.get("intermediate_boundary_fraction_per_mask") or {}).get("min"),
            soft_band_halfwidth_max=(
                prof.get("soft_band_halfwidth_px") or {}).get("max"),
            rgb_replicated_for_every_pixel=prof.get(
                "rgb_channels_replicated_for_every_pixel"))
        if prof.get("encoding_verdict") != "BINARY_SEMANTIC_MASK_WITH_INTERMEDIATE_EDGE_VALUES":
            problems.append(f"unexpected encoding verdict "
                            f"{prof.get('encoding_verdict')!r}")
        if prof.get("masks_with_only_0_and_255") != 0:
            problems.append("the profile reports strictly binary masks; re-check the "
                            "premise")
    excluded: List[dict] = []
    if "19_bus_uc_mask_integrity_cases.csv" in found:
        for r in read_rows(found["19_bus_uc_mask_integrity_cases.csv"]):
            excluded.append({"sample_key": r.get("sample_key", ""),
                             "problem_flags": r.get("problem_flags", ""),
                             "mismatch_kind": r.get("mismatch_kind", "")})
        if len(excluded) != 3:
            problems.append(f"integrity cases = {len(excluded)}, expected 3")

    print("\n--- verified from the artifacts ---")
    for k, v in verified.items():
        print(f"  {k:38s} {v}")
    print(f"  integrity cases                        "
          f"{[e['sample_key'] for e in excluded]}")

    if problems:
        print("\n" + "=" * 78)
        print("BLOCKED - the policy was NOT written")
        for p in problems:
            print(f"  {p}")
        return 4

    policy = {
        "dataset": "BUS_UC",
        "policy_id": POLICY_ID,
        "policy_status": "LOCKED_PHASE2B_CANONICAL_POLICY",
        "policy_title": "BUS-UC canonical mask binarization at a pre-specified "
                        "midpoint threshold",
        "source_mask_semantics":
            "RGB_REPLICATED_GRAYSCALE_BINARY_SEMANTIC_WITH_SOFT_BOUNDARY",
        "canonical_threshold": THRESHOLD,
        "canonical_rule": "source_gray >= 128 -> 255; otherwise 0",
        "canonical_rule_expression": "canonical_mask = 255 * (source_gray >= 128)",
        "channel_reduction": ("RGB channels are verified identical per pixel, then "
                              "reduced to one grayscale plane before thresholding"),
        "output_encoding": {"background": 0, "foreground": 255, "channels": 1,
                            "format": "lossless PNG"},
        "preserve_native_dimensions": True,
        "roi_cropping": False,
        "resize_or_resample": False,
        "pseudo_mask": False,
        "union_masks": False,
        "threshold_provenance": "PHASE2B_CANONICAL_POLICY",
        "source_defined_threshold": False,
        "model_performance_used_to_select_threshold": False,
        "threshold_fixed_before_training": True,
        "rationale": ("no authoritative source threshold exists. A fixed midpoint is "
                      "pre-specified so that low-intensity edge artefacts are not "
                      "treated as lesion foreground, and so that an extreme high "
                      "threshold does not retain only the strongest foreground core. "
                      "It is a data-curation rule, not a tuned parameter."),
        "descriptive_rules_explicitly_not_adopted": {
            "pixel > 0": "used only to compute foreground fractions during the audit",
            "5/250 bounds": "used only to separate unambiguous extremes for "
                            "description; explicitly disclaimed as thresholds"},
        "verified_encoding_facts": verified,
        "excluded_mask_integrity_records": len(excluded),
        "excluded_records": excluded,
        "exclusion_policy": ("the three mask-integrity records are excluded from the "
                             "canonical joint cohort and are NOT repaired, resized, "
                             "resampled, cropped, padded, transformed, redrawn or "
                             "substituted. They are retained in the manifest with an "
                             "explicit exclusion status."),
        "scope": ("BUS_UC ONLY. This establishes no threshold for any other dataset, "
                  "and no BUSI policy applies here: BUS-UC has no multi-mask case."),
        "evidence_artifacts": [
            {"artifact": n, "purpose": p, "path": str(found[n]),
             "sha256": sha256_file(found[n])} for n, p in EVIDENCE],
        "raw_modified": False,
        "audit_reopened": False,
        "policy_date": now_iso(),
    }
    target = policy_dir / POLICY_FILE
    if target.exists():
        try:
            old = json.loads(target.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = None
        a_new = {k: v for k, v in policy.items() if k != "policy_date"}
        a_old = ({k: v for k, v in old.items() if k != "policy_date"}
                 if isinstance(old, dict) else None)
        if a_old == a_new:
            print(f"\nthe policy artifact already exists and is identical; "
                  f"nothing rewritten ({target.name})")
        else:
            print("\n" + "=" * 78)
            print("BLOCKED - an existing policy artifact differs from this one")
            print(f"  {target}")
            print("  The artifact is immutable; a changed policy needs its own "
                  "version id (…_V2).")
            return 4
    elif a.dry_run:
        print(f"\nDRY-RUN: would write {target}")
    else:
        policy_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(policy, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print(f"\nWROTE {target}")

    print("\n" + "=" * 78)
    print(f"Policy ID:               {POLICY_ID}")
    print(f"Threshold:               {THRESHOLD}  (PHASE2B_CANONICAL_POLICY, "
          "not source-defined)")
    print(f"Evidence artifacts:      {len(found)}/{len(EVIDENCE)}")
    print(f"Excluded integrity cases: {len(excluded)}")
    print("RAW modified:            FALSE")
    print("Audit modified:          FALSE")
    print("\nFINAL VERDICT:")
    print("DRY_RUN_OK_POLICY_NOT_YET_LOCKED" if a.dry_run
          else "BUS_UC_CANONICAL_MASK_POLICY_LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
