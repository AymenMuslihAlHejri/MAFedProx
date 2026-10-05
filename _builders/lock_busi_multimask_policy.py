#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lock_busi_multimask_policy.py
=============================

PHASE 2B.2 ADDENDUM A - BUSI MULTI-MASK CANONICALIZATION POLICY

Writes ONE immutable policy artifact and appends a reference to the bank policy
metadata. Nothing else.

DOES NOT reopen Phase 2A, reopen the BUSI forensic audit, modify RAW, rerun duplicate
analysis, convert any image or mask, or create any split. It writes no image.

TWO THINGS THIS TOOL DOES NOT TAKE ON TRUST
    1. The figures in the addendum text are ASSERTIONS. The 630/17 accounting and the
       union behaviour are re-read from the named evidence artifacts and compared. A
       figure that cannot be confirmed from an artifact blocks the lock rather than
       being written from the addendum.
    2. The policy artifact is IMMUTABLE. If it already exists with different content,
       this tool refuses rather than overwriting; re-locking a changed policy is a
       human decision that needs its own version.

PROVENANCE HONESTY
    The policy records that its rationale is existing curation evidence and historical
    implemented behaviour, now ELEVATED to a locked Phase 2B decision by human
    authority. It does not claim the machine audit established the rule - it did not,
    which is precisely why this addendum exists.

Usage:
    python lock_busi_multimask_policy.py
    python lock_busi_multimask_policy.py --dry-run     (verify only, write nothing)
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
POLICY_ID = "BUSI_CANONICAL_MULTIMASK_UNION_V1"
POLICY_FILE = "BUSI_CANONICAL_MULTIMASK_POLICY_V1.json"

EVIDENCE_ARTIFACTS = [
    ("03_pairing_audit.csv", "authoritative source image to source mask pairing"),
    ("05_busi_curated_manifest.csv", "curated per-record manifest"),
    ("07_curated_mask_qa.csv", "curated mask QA, including union-mask creation"),
    ("08_busi_curation_summary.json", "curation summary counts"),
    ("09_BUSI_SCIENTIFIC_CURATION_REPORT.md",
     "curation report describing M_final = M1 OR ... OR Mk"),
    ("11_phase1b_self_tests.txt", "self-tests confirming the union masks"),
]

# claims to verify against the artifacts, never to copy from the addendum text
CLAIMS = {"single_mask_images": 630, "multi_mask_images": 17,
          "union_masks_created": 17}
UNION_PATTERNS = [
    r"M_?final\s*=\s*M_?1\s*(OR|\|{1,2}|∨)", r"logical\s+union", r"union\s+of\s+all",
    r"union\s+mask", r"pixel-?wise\s+(logical\s+)?(or|union)", r"np\.maximum",
    r"logical_or", r"bitwise_or",
]


def archived(p: Path) -> bool:
    return any(part.lower() in ARCHIVE_DIR_NAMES for part in p.parts[:-1])


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scalar_paths(blob: Any, prefix: str = "", depth: int = 0,
                 out: Optional[list] = None) -> list:
    out = [] if out is None else out
    if depth > 4:
        return out
    if isinstance(blob, dict):
        for k, v in blob.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (int, float, str, bool)) or v is None:
                out.append((path, v))
            elif isinstance(v, dict):
                scalar_paths(v, path, depth + 1, out)
            elif isinstance(v, list):
                for i, el in enumerate(v[:20]):
                    if isinstance(el, dict):
                        scalar_paths(el, f"{path}[{i}]", depth + 1, out)
    return out


# deliberately EXCLUDES "tests" - a self-tests report is not a partition, and that
# coincidence is what produced the earlier false positive
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "cv", "train", "training",
                "trainset", "trainval", "val", "validation", "test", "testset",
                "testing", "holdout", "partition", "traintest"}


def is_split_like(path_str: str) -> bool:
    """Whole-token match on the filename and its parent folders, with any trailing
    digits stripped so 'fold5' still matches while 'evaluation', 'interval' and
    'SAFETY_TESTS' do not. A match is a CANDIDATE only; whether it belongs to THIS RUN
    is decided by the before/after snapshot, never by the name."""
    pth = Path(path_str)
    raw = [t for t in re.split(r"[^A-Za-z0-9]+", pth.stem) if t]
    raw += [t for part in pth.parts[:-1]
            for t in re.split(r"[^A-Za-z0-9]+", part) if t]
    tokens = set()
    for t in raw:
        low = t.lower()
        tokens.add(low)
        tokens.add(re.sub(r"\d+$", "", low))
    return bool(tokens & SPLIT_TOKENS)


def snapshot(root: Path) -> Dict[str, Tuple[int, int]]:
    """Filesystem metadata only; no file is opened."""
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


def classify(before: Dict[str, Tuple[int, int]], after: Dict[str, Tuple[int, int]],
             predicate) -> Tuple[List[str], List[str], List[str]]:
    """Return (preexisting, newly_created, modified) among paths matching predicate."""
    pre = sorted(k for k in before if predicate(k))
    new = sorted(k for k in after if k not in before and predicate(k))
    mod = sorted(k for k in after
                 if k in before and after[k] != before[k] and predicate(k))
    return pre, new, mod


def read_rows(p: Path) -> List[dict]:
    with p.open(encoding="utf-8-sig", errors="replace") as fh:
        sample = fh.read(8192); fh.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        return list(csv.DictReader(fh, delimiter=delim))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Lock the BUSI multi-mask policy.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the evidence and write nothing")
    a = ap.parse_args(argv)
    prep = Path(a.prep_root)
    bank = prep / BANK_NAME
    policy_dir = bank / "_policies"

    print("=" * 78)
    print("PHASE 2B.2 ADDENDUM A - BUSI MULTI-MASK CANONICALIZATION POLICY")
    print("=" * 78)
    print(f"mode: {'DRY-RUN (writes nothing)' if a.dry_run else 'LOCK'}")

    # snapshot the bank BEFORE this run writes anything, so split and PNG safety can
    # be attributed to THIS RUN rather than to whatever already existed
    before_bank = snapshot(bank)

    busi_dirs = [d for d in prep.iterdir()
                 if d.is_dir() and "busi" in d.name.lower().replace(" ", "")]
    if not busi_dirs:
        print(f"no BUSI folder under {prep}")
        return 2
    files = [p for d in busi_dirs for p in d.rglob("*")
             if p.is_file() and not archived(p)]

    # ---- 1. locate the named evidence artifacts, uniquely ---------------------
    print("\n--- evidence artifacts ---")
    found: Dict[str, Path] = {}
    missing: List[str] = []
    ambiguous: List[str] = []
    for name, purpose in EVIDENCE_ARTIFACTS:
        hits = sorted(p for p in files if p.name.lower() == name.lower())
        if len(hits) == 1:
            (only,) = hits
            found[name] = only
            print(f"  UNIQUE     {name:44s} {purpose}")
        elif not hits:
            missing.append(name)
            print(f"  MISSING    {name:44s} {purpose}")
        else:
            ambiguous.append(name)
            print(f"  AMBIGUOUS  {name:44s} {len(hits)} matches")

    # ---- 2. verify the claims from the artifacts, not from the addendum -------
    print("\n--- claim verification (artifact-derived) ---")
    verified: Dict[str, dict] = {}
    for p in found.values():
        suf = p.suffix.lower()
        try:
            if suf == ".json":
                for path, val in scalar_paths(json.loads(p.read_text(encoding="utf-8"))):
                    if not isinstance(val, (int, float)):
                        continue
                    low = path.lower()
                    for claim, expect in CLAIMS.items():
                        key = claim.split("_")[0]
                        if int(val) == expect and (
                                ("single" in low and claim == "single_mask_images")
                                or ("multi" in low and claim == "multi_mask_images")
                                or ("union" in low and claim == "union_masks_created")):
                            verified.setdefault(claim, {
                                "value": int(val), "artifact": p.name,
                                "locator": f"json:{path}"})
            elif suf == ".csv":
                rows = read_rows(p)
                if not rows:
                    continue
                for c in list(rows[0]):
                    cnt = Counter(str(r.get(c, "")).strip().upper() for r in rows)
                    low = c.lower()
                    if "union" in low:
                        n = sum(v for k, v in cnt.items()
                                if k in ("TRUE", "YES", "1", "UNION"))
                        if n == CLAIMS["union_masks_created"]:
                            verified.setdefault("union_masks_created", {
                                "value": n, "artifact": p.name,
                                "locator": f"csv:[{c}] truthy row count"})
                    if "mask" in low or "count" in low or "status" in low:
                        for val, n in cnt.items():
                            if n == CLAIMS["multi_mask_images"] and val:
                                verified.setdefault("multi_mask_images", {
                                    "value": n, "artifact": p.name,
                                    "locator": f"csv:[{c}=='{val}'] row count"})
                            if n == CLAIMS["single_mask_images"] and val:
                                verified.setdefault("single_mask_images", {
                                    "value": n, "artifact": p.name,
                                    "locator": f"csv:[{c}=='{val}'] row count"})
            else:
                txt = p.read_text(encoding="utf-8", errors="replace")
                for claim, expect in CLAIMS.items():
                    for m in re.finditer(rf"\b{expect}\b", txt):
                        ctx = txt[max(0, m.start() - 110):m.end() + 110]
                        cl = ctx.lower()
                        if (("single" in cl and claim == "single_mask_images")
                                or ("multi" in cl and claim == "multi_mask_images")
                                or ("union" in cl and claim == "union_masks_created")):
                            verified.setdefault(claim, {
                                "value": expect, "artifact": p.name,
                                "locator": "text: " + " ".join(ctx.split())[:110]})
                            break
        except Exception as exc:  # noqa: BLE001
            print(f"  unreadable {p.name}: {type(exc).__name__}")

    for claim, expect in CLAIMS.items():
        v = verified.get(claim)
        print(f"  {claim:24s} expected {expect:<5} "
              + (f"CONFIRMED  {v['artifact']} :: {v['locator'][:70]}" if v
                 else "NOT CONFIRMED FROM ANY NAMED ARTIFACT"))

    # ---- 3. verify the union formula is actually described ---------------------
    union_evidence: List[dict] = []
    for p in found.values():
        if p.suffix.lower() not in (".md", ".txt", ".json", ".csv"):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for pat in UNION_PATTERNS:
            m = re.search(pat, txt, re.I)
            if m:
                ctx = " ".join(txt[max(0, m.start() - 120):m.end() + 160].split())
                union_evidence.append({"artifact": p.name, "pattern": pat,
                                       "quote": ctx[:200]})
                break
    print("\n--- union-mask evidence ---")
    if union_evidence:
        for u in union_evidence:
            print(f"  {u['artifact']:44s} /{u['pattern']}/")
            print(f"      ...{u['quote'][:150]}...")
    else:
        print("  NO named artifact describes a union/OR construction")

    # ---- 4. fail-closed gate ----------------------------------------------------
    blockers: List[str] = []
    if missing:
        blockers.append("named evidence artifact(s) not found: " + ", ".join(missing))
    if ambiguous:
        blockers.append("named evidence artifact(s) ambiguous: " + ", ".join(ambiguous))
    for claim in CLAIMS:
        if claim not in verified:
            blockers.append(f"claim {claim} not confirmed from any named artifact")
    if not union_evidence:
        blockers.append("the union construction is not described in any named artifact")
    if blockers:
        print("\n" + "=" * 78)
        print("BLOCKED - the policy was NOT written")
        print("=" * 78)
        for b in blockers:
            print(f"  {b}")
        print("\nThe addendum asserts these facts; this tool requires them to come from "
              "the artifacts it binds. Nothing was written.")
        return 4

    # ---- 5. build the immutable policy artifact -----------------------------------
    policy = {
        "dataset": "BUSI",
        "policy_id": POLICY_ID,
        "policy_title": "BUSI canonical multi-mask consolidation by logical union",
        "sample_unit": "SOURCE_IMAGE",
        "single_mask_rule": ("canonical mask = binarized form of the single valid "
                             "source mask"),
        "multi_mask_rule": ("canonical mask = pixel-wise logical union of all valid "
                            "source masks of that source image: "
                            "M_canonical = OR(M_1, M_2, ..., M_K)"),
        "output_mask_encoding": {"background": 0, "foreground": 255,
                                 "channels": 1, "format": "lossless PNG"},
        "preserve_native_dimensions": True,
        "roi_cropping": False,
        "pseudo_mask": False,
        "lesion_instance_selection": "NONE; no source mask is preferentially selected",
        "source_mask_discarding": "NONE; no valid source mask is discarded",
        "union_support_constraint": ("the union contains only pixels supported by "
                                     "source-provided masks"),
        "multi_lesion_representation": ("multiple annotations within ONE source image "
                                        "do NOT become separate canonical samples"),
        "scope": ("BUSI ONLY. This establishes no global multi-mask rule; every other "
                  "dataset requires its own evidence-backed canonical policy."),
        "verified_accounting": {k: {"value": v["value"], "artifact": v["artifact"],
                                    "locator": v["locator"]}
                                for k, v in verified.items()},
        "union_evidence": union_evidence,
        "evidence_artifacts": [
            {"artifact": name, "purpose": purpose,
             "path": str(found[name]), "sha256": sha256_file(found[name])}
            for name, purpose in EVIDENCE_ARTIFACTS],
        "manifest_fields_to_preserve": [
            "original source mask filenames", "number of original source masks",
            "union_mask_created", "source to canonical provenance"],
        "provenance_of_this_policy": {
            "basis": "EXISTING_CURATION_EVIDENCE_AND_IMPLEMENTED_HISTORICAL_BEHAVIOUR",
            "elevated_by": "HUMAN_SCIENTIFIC_DECISION_PHASE_2B2_ADDENDUM_A",
            "machine_audit_established_this_rule": False,
            "note": ("the completed forensic audit COUNTED the multi-mask images but "
                     "did not decide their canonical resolution; this addendum "
                     "supplies that decision and cites the prior curation evidence as "
                     "its rationale. Historical audit conclusions are not rewritten.")},
        "policy_status": "LOCKED_PHASE2B_CANONICAL_POLICY",
        "policy_date": now_iso(),
    }
    body = json.dumps(policy, indent=2, ensure_ascii=False)

    target = policy_dir / POLICY_FILE
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = None
        comparable_new = {k: v for k, v in policy.items() if k != "policy_date"}
        comparable_old = ({k: v for k, v in existing.items() if k != "policy_date"}
                          if isinstance(existing, dict) else None)
        if comparable_old == comparable_new:
            print(f"\nthe policy artifact already exists and is identical; "
                  f"nothing rewritten ({target.name})")
        else:
            print("\n" + "=" * 78)
            print("BLOCKED - an existing policy artifact differs from this one")
            print("=" * 78)
            print(f"  {target}")
            print("  The policy artifact is immutable. Re-locking changed content is a "
                  "human decision and needs its own version id (…_V2).")
            return 4
    elif a.dry_run:
        print(f"\nDRY-RUN: would write {target}")
    else:
        policy_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        print(f"\nWROTE {target}")

    # policy registry + a reference appended to the bank policy, never a rewrite
    registry = policy_dir / "POLICY_REGISTRY.json"
    if not a.dry_run:
        reg = {"policies": []}
        if registry.exists():
            try:
                reg = json.loads(registry.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                reg = {"policies": []}
        if not any(x.get("policy_id") == POLICY_ID for x in reg.get("policies", [])):
            reg.setdefault("policies", []).append(
                {"policy_id": POLICY_ID, "dataset": "BUSI", "scope": "BUSI_ONLY",
                 "artifact": POLICY_FILE, "status": policy["policy_status"],
                 "date": policy["policy_date"]})
            registry.write_text(json.dumps(reg, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            print(f"WROTE {registry}")
        bank_policy = bank / "CANONICAL_DATASET_BANK_POLICY.md"
        if bank_policy.exists():
            txt = bank_policy.read_text(encoding="utf-8")
            if POLICY_ID not in txt:
                with bank_policy.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"\n\n## Dataset policy addendum: {POLICY_ID}\n\n"
                        f"**Scope: BUSI only.** A source image with one valid mask takes "
                        f"that mask binarised; a source image with K > 1 valid masks "
                        f"takes the pixel-wise logical union "
                        f"`M_canonical = OR(M_1, ..., M_K)`. The sample unit stays the "
                        f"SOURCE IMAGE, no lesion instance is preferred, no source mask "
                        f"is discarded, no pseudo-mask is generated and no ROI is "
                        f"cropped. Locked by Phase 2B.2 Addendum A on "
                        f"{policy['policy_date']}; see `_policies/{POLICY_FILE}`. This "
                        f"establishes no rule for any other dataset.\n")
                print(f"APPENDED reference to {bank_policy.name} (existing text "
                      "unchanged)")

    # ---- 6. safety ------------------------------------------------------------------
    # A filename token is only a CANDIDATE. What matters is whether THIS RUN created or
    # modified it, decided by the before/after snapshot - not by the name alone. The
    # previous version counted PHASE2B30_SAFETY_TESTS.txt as a split because "TESTS"
    # contains "test", which is a naming coincidence, not a partition.
    after_bank = snapshot(bank)
    split_pre, split_new, split_mod = classify(before_bank, after_bank, is_split_like)
    png_pre, png_new, png_mod = classify(
        before_bank, after_bank, lambda p: p.lower().endswith(".png"))
    print("\n--- split safety (before/after attribution) ---")
    print(f"  preexisting_split_artifacts : {len(split_pre)}")
    for f in split_pre[:10]:
        print(f"      {Path(f).name}   [filename token only; not asserted to be a "
              "split]")
    print(f"  newly_created_split_artifacts: {len(split_new)}")
    for f in split_new[:10]:
        print(f"      {f}")
    print(f"  modified_split_artifacts     : {len(split_mod)}")
    for f in split_mod[:10]:
        print(f"      {f}")
    pngs, splits = png_new, split_new
    print("\n" + "=" * 78)
    print("PHASE 2B.2 ADDENDUM A - BUSI MULTI-MASK POLICY")
    print("=" * 78)
    print(f"Policy ID:                   {POLICY_ID}")
    print(f"Evidence artifacts found:    {len(found)}/{len(EVIDENCE_ARTIFACTS)}")
    print(f"630 single-mask:             "
          f"{verified['single_mask_images']['artifact']} :: "
          f"{verified['single_mask_images']['locator'][:58]}")
    print(f"17 multi-mask:               "
          f"{verified['multi_mask_images']['artifact']} :: "
          f"{verified['multi_mask_images']['locator'][:58]}")
    print(f"Existing union-mask evidence: {len(union_evidence)} artifact(s): "
          + ", ".join(u["artifact"] for u in union_evidence))
    print(f"Policy written:              "
          + ("DRY-RUN, nothing written" if a.dry_run else str(target)))
    print("RAW modified:                FALSE  (RAW was never opened by this tool)")
    print("Audit modified:              FALSE  (only the bank policy area was written)")
    print(f"Canonical PNG written:       {len(png_new)}  "
          f"(pre-existing in bank: {len(png_pre)}, modified: {len(png_mod)})")
    print(f"preexisting_split_artifacts: {len(split_pre)}"
          + (f"  ({', '.join(Path(f).name for f in split_pre[:3])})" if split_pre
             else ""))
    print(f"newly_created_split_artifacts: {len(split_new)}")
    print(f"modified_split_artifacts:    {len(split_mod)}")
    if split_new or split_mod or png_new or png_mod:
        print("\nFINAL VERDICT:")
        print("BLOCKED_THIS_RUN_CREATED_OR_MODIFIED_SPLIT_OR_IMAGE_ARTIFACTS")
        return 4
    print("\nFINAL VERDICT:")
    if a.dry_run:
        print("DRY_RUN_OK_POLICY_NOT_YET_LOCKED")
        return 0
    print("BUSI_CANONICAL_MULTIMASK_POLICY_LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
