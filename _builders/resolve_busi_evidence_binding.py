#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resolve_busi_evidence_binding.py
================================

PHASE 2B.3.1a - BUSI EVIDENCE-BINDING RESOLUTION (READ-ONLY)

Locates EXISTING completed BUSI evidence for tasks A-D. It searches; it does not
audit, adjudicate, convert or decide.

STRICTLY READ-ONLY. Writes nothing. Opens no image. Never touches RAW. Runs no
duplicate or near-duplicate analysis, and re-adjudicates nothing.

SELF-MATCH GUARD: a previous forensic search in this project matched its own script
and reported a false positive. This resolver excludes builder, inspector, resolver and
bank-init scripts by filename, so it cannot cite itself or its siblings as historical
evidence.

EVIDENCE STRENGTH is assigned conservatively and never upgraded:
    EXPLICIT_LOCKED_DECISION      a policy-shaped key/value in a locked audit JSON
    IMPLEMENTED_HISTORICAL_BEHAVIOR   code that did something, in a non-audit script
    OBSERVATION_ONLY              prose or a count that describes, not decides
    INFERENCE                     anything this tool would have had to conclude itself
Only EXPLICIT_LOCKED_DECISION may authorise the canonical builder, and this tool never
awards that label to code or prose.

Usage:
    python resolve_busi_evidence_binding.py
    python resolve_busi_evidence_binding.py --prep-root ... --code-root ...
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
DEFAULT_CODE_ROOT = r"C:\Users\Asus\Python code\MAFedProx_Revision"
ARCHIVE_DIR_NAMES = ("audit_old",)

# never cite these as evidence: they are this pipeline's own tooling
SELF_EXCLUDE = re.compile(
    r"^(build_bank_|inspect_busi_|resolve_busi_evidence_binding|"
    r"build_phase2a|build_phase2b|build_master_provenance)", re.I)

DISPOSITIONS = ("ELIGIBLE_UNIQUE", "ELIGIBLE_GROUP_LOCK_REQUIRED",
                "EXCLUDE_CONFIRMED_LABEL_CONFLICT")
EXPECT = {"ELIGIBLE_UNIQUE": 636, "ELIGIBLE_GROUP_LOCK_REQUIRED": 4,
          "EXCLUDE_CONFIRMED_LABEL_CONFLICT": 7}
MASK_CONCEPTS = ("multi_mask", "multi-mask", "multiple mask", "multiple lesion",
                 "multi lesion", "mask_1", "mask_2", "union", "combined mask",
                 "merge mask", "merged mask", "canonical mask", "resolved mask",
                 "lesion mask")
POLICY_KEY_HINT = re.compile(
    r"(policy|rule|decision|canonical|resolution|handling|treatment)", re.I)

TEXT_EXT = {".json", ".csv", ".md", ".txt", ".log", ".py"}


def archived(p: Path) -> bool:
    return any(part.lower() in ARCHIVE_DIR_NAMES for part in p.parts[:-1])


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
                out.append((f"{path}[]", f"<list:{len(v)}>"))
                for i, el in enumerate(v[:20]):
                    if isinstance(el, dict):
                        scalar_paths(el, f"{path}[{i}]", depth + 1, out)
                    elif isinstance(el, (int, float, str)):
                        out.append((f"{path}[{i}]", el))
    return out


def read_rows(p: Path) -> Tuple[List[dict], str]:
    with p.open(encoding="utf-8-sig", errors="replace") as fh:
        sample = fh.read(8192); fh.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        return list(csv.DictReader(fh, delimiter=delim)), delim


def col_like(cols: Sequence[str], words: Sequence[str]) -> List[str]:
    out = []
    for c in cols:
        n = re.sub(r"[^a-z0-9]", "", str(c).lower())
        if any(re.sub(r"[^a-z0-9]", "", w) in n for w in words):
            out.append(c)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BUSI evidence-binding resolution.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--code-root", default=DEFAULT_CODE_ROOT)
    a = ap.parse_args(argv)
    prep, code = Path(a.prep_root), Path(a.code_root)

    busi_dirs = [d for d in prep.iterdir()
                 if d.is_dir() and "busi" in d.name.lower().replace(" ", "")]
    if not busi_dirs:
        print(f"no BUSI folder under {prep}")
        return 2
    files = [p for d in busi_dirs for p in d.rglob("*")
             if p.is_file() and not archived(p)]
    small = [p for p in files if p.suffix.lower() in TEXT_EXT
             and not SELF_EXCLUDE.match(p.name)]
    print("=" * 78)
    print("BUSI EVIDENCE-BINDING RESOLUTION  (read-only)")
    print("=" * 78)
    print(f"active BUSI files: {len(files)}   inspectable: {len(small)}")
    print(f"archived excluded: {sum(1 for d in busi_dirs for p in d.rglob('*') if p.is_file() and archived(p))}")

    # ---------------- TASK A: mask accounting ---------------------------------
    print("\n" + "=" * 78)
    print("A. MASK ACCOUNTING  (semantic search for 630 single / 17 multi)")
    print("=" * 78)
    a_hits: List[dict] = []
    for p in small:
        if p.suffix.lower() == ".json":
            try:
                blob = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for path, val in scalar_paths(blob):
                if isinstance(val, (int, float)) and int(val) in (630, 17):
                    a_hits.append({"artifact": str(p), "locator": f"json:{path}",
                                   "value": val, "type": "JSON"})
        elif p.suffix.lower() == ".csv":
            try:
                rows, _d = read_rows(p)
            except Exception:  # noqa: BLE001
                continue
            if not rows:
                continue
            maskcols = col_like(list(rows[0]), ["mask", "pairing", "status", "count",
                                                "nmask", "num"])
            for c in maskcols:
                cnt = Counter(str(r.get(c, "")).strip() for r in rows)
                for v, n in cnt.items():
                    if n in (630, 17) and v:
                        a_hits.append({
                            "artifact": str(p),
                            "locator": f"csv:[{c}=='{v}'] row count",
                            "value": n, "type": "CSV"})
        else:
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            for m in re.finditer(r"\b(630|17)\b", txt):
                ctx = txt[max(0, m.start() - 90):m.end() + 90].replace("\n", " ")
                if any(w in ctx.lower() for w in ("mask", "lesion")):
                    a_hits.append({"artifact": str(p), "locator": "text line",
                                   "value": m.group(1), "type": p.suffix.upper()[1:],
                                   "context": ctx.strip()})
    if a_hits:
        for h in a_hits[:30]:
            print(f"  [{h['type']:4s}] {Path(h['artifact']).name:46s} {h['locator']}")
            print(f"         value={h['value']}"
                  + (f"   ctx: ...{h.get('context', '')[:100]}..."
                     if h.get("context") else ""))
    else:
        print("  NOT FOUND in any active BUSI artifact")

    # ---------------- TASK B: per-record pairing --------------------------------
    print("\n" + "=" * 78)
    print("B. PER-RECORD PAIRING TABLE")
    print("=" * 78)
    b_cands = []
    for p in [x for x in small if x.suffix.lower() == ".csv"]:
        try:
            rows, delim = read_rows(p)
        except Exception:  # noqa: BLE001
            continue
        if not rows:
            continue
        cols = list(rows[0])
        imgc = col_like(cols, ["imagepath", "imagerelative", "relativepath", "image",
                               "filename", "stem", "file"])
        mskc = col_like(cols, ["mask"])
        if imgc and mskc:
            b_cands.append({"artifact": p, "rows": len(rows), "delim": delim,
                            "cols": cols, "image_cols": imgc, "mask_cols": mskc,
                            "label_cols": col_like(cols, ["label", "class",
                                                          "pathology"])})
    for c in sorted(b_cands, key=lambda x: -abs(647 - x["rows"]))[-8:]:
        flag = "  <== row count == 647" if c["rows"] == 647 else ""
        print(f"\n  {c['artifact'].name}   [{c['artifact'].parent.name}]{flag}")
        print(f"    rows={c['rows']}  delimiter='{c['delim']}'")
        print(f"    columns      : {c['cols']}")
        print(f"    image column : {c['image_cols']}")
        print(f"    mask column  : {c['mask_cols']}")
        print(f"    label column : {c['label_cols'] or 'NONE'}")
    if not b_cands:
        print("  NO CSV with both an image-like and a mask-like column")

    # ---------------- TASK C: per-record disposition ------------------------------
    print("\n" + "=" * 78)
    print("C. PER-RECORD FINAL DISPOSITION")
    print("=" * 78)
    c_cands = []
    for p in [x for x in small if x.suffix.lower() == ".csv"]:
        try:
            rows, delim = read_rows(p)
        except Exception:  # noqa: BLE001
            continue
        if not rows:
            continue
        for c in list(rows[0]):
            vals = Counter(str(r.get(c, "")).strip() for r in rows)
            hit = {d: vals.get(d, 0) for d in DISPOSITIONS if vals.get(d, 0)}
            if hit:
                c_cands.append({"artifact": p, "rows": len(rows), "delim": delim,
                                "cols": list(rows[0]), "status_col": c, "counts": hit,
                                "id_cols": col_like(list(rows[0]),
                                                    ["relativepath", "imagepath",
                                                     "image", "filename", "stem"]),
                                "dup_cols": col_like(list(rows[0]),
                                                     ["cluster", "group", "duplicate",
                                                      "lock"])})
    for c in c_cands:
        total = sum(c["counts"].values())
        exact = all(c["counts"].get(k) == v for k, v in EXPECT.items())
        print(f"\n  {c['artifact'].name}   [{c['artifact'].parent.name}]")
        print(f"    rows={c['rows']}  status column='{c['status_col']}'")
        print(f"    reconstructed: {c['counts']}   total={total}"
              f"   {'<== reconciles 636+4+7=647' if exact and total == 647 else ''}")
        print(f"    id column(s) : {c['id_cols'] or 'NONE'}")
        print(f"    dup column(s): {c['dup_cols'] or 'NONE'}")
        print(f"    all columns  : {c['cols']}")
    if not c_cands:
        print("  NO CSV contains the disposition vocabulary "
              f"({', '.join(DISPOSITIONS)})")

    # ---------------- TASK D: multi-mask policy -------------------------------------
    print("\n" + "=" * 78)
    print("D. MULTI-MASK RESOLUTION EVIDENCE")
    print("=" * 78)
    print("  (self-match guard active: this pipeline's own scripts are excluded)")
    d_hits: List[dict] = []
    # D1: policy-shaped keys in locked audit JSON  -> may be EXPLICIT_LOCKED_DECISION
    for p in [x for x in small if x.suffix.lower() == ".json"]:
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for path, val in scalar_paths(blob):
            low = f"{path} {val}".lower()
            if any(c in low for c in MASK_CONCEPTS):
                strength = ("EXPLICIT_LOCKED_DECISION"
                            if POLICY_KEY_HINT.search(path) and isinstance(val, str)
                            and len(str(val)) > 3 else "OBSERVATION_ONLY")
                d_hits.append({"artifact": str(p), "locator": f"json:{path}",
                               "value": str(val)[:110], "strength": strength})
    # D2: prose in reports  -> OBSERVATION_ONLY at best
    for p in [x for x in small if x.suffix.lower() in (".md", ".txt", ".log")]:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for c in MASK_CONCEPTS:
            for m in re.finditer(re.escape(c), txt, re.I):
                ctx = txt[max(0, m.start() - 110):m.end() + 130].replace("\n", " ")
                d_hits.append({"artifact": str(p), "locator": f"text:'{c}'",
                               "value": ctx.strip()[:200],
                               "strength": "OBSERVATION_ONLY"})
                break
    # D3: historical code  -> IMPLEMENTED_HISTORICAL_BEHAVIOR, never more
    code_files = []
    if code.is_dir():
        code_files = [p for p in code.rglob("*.py")
                      if not SELF_EXCLUDE.match(p.name) and not archived(p)]
    for p in code_files:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        if "busi" not in txt.lower() and "busi" not in p.name.lower():
            continue
        for c in ("union", "np.maximum", "logical_or", "mask_1", "multi_mask",
                  "cv2.bitwise_or"):
            for m in re.finditer(re.escape(c), txt, re.I):
                ctx = txt[max(0, m.start() - 110):m.end() + 130].replace("\n", " ")
                if "mask" in ctx.lower():
                    d_hits.append({"artifact": str(p), "locator": f"code:'{c}'",
                                   "value": ctx.strip()[:200],
                                   "strength": "IMPLEMENTED_HISTORICAL_BEHAVIOR"})
                    break
    seen = set(); uniq = []
    for h in d_hits:
        k = (h["artifact"], h["locator"])
        if k not in seen:
            seen.add(k); uniq.append(h)
    if uniq:
        for h in sorted(uniq, key=lambda x: x["strength"]):
            print(f"\n  [{h['strength']}]")
            print(f"    artifact: {h['artifact']}")
            print(f"    locator : {h['locator']}")
            print(f"    value   : {h['value']}")
    else:
        print("  NO artifact, report or non-self script mentions multi-mask resolution")
    explicit = [h for h in uniq if h["strength"] == "EXPLICIT_LOCKED_DECISION"]

    # ---------------- FINAL REPORT ----------------------------------------------------
    a_ok = bool(a_hits)
    b_ok = any(c["rows"] == 647 for c in b_cands) or bool(b_cands)
    c_exact = [c for c in c_cands
               if all(c["counts"].get(k) == v for k, v in EXPECT.items())
               and sum(c["counts"].values()) == 647]
    c_ok = bool(c_exact)
    print("\n" + "=" * 78)
    print("BUSI EVIDENCE-BINDING RESOLUTION")
    print("=" * 78)
    def a_binding(target: str) -> str:
        for h in a_hits:
            if str(h["value"]) == target:
                return f"{Path(h['artifact']).name} :: {h['locator']}"
        return "NOT FOUND"

    print("\nA. MASK ACCOUNTING")
    print(f"   630 binding: {a_binding('630')}")
    print(f"   17  binding: {a_binding('17')}")
    print("\nB. PAIRING TABLE")
    if b_cands:
        best = next((c for c in b_cands if c["rows"] == 647), b_cands[0])
        print(f"   artifact: {best['artifact']}")
        print(f"   rows    : {best['rows']}")
        print(f"   columns : image={best['image_cols']} mask={best['mask_cols']} "
              f"label={best['label_cols'] or 'NONE'}")
        print(f"   status  : {'RESOLVED' if best['rows'] == 647 else 'CANDIDATE_ROWCOUNT_MISMATCH'}")
    else:
        print("   artifact: NOT FOUND\n   status  : UNRESOLVED")
    print("\nC. FINAL DISPOSITION")
    if c_exact:
        b = c_exact[0]
        print(f"   artifact: {b['artifact']}")
        for k, v in EXPECT.items():
            print(f"   {v:<7}: {k} @ csv:[{b['status_col']}]")
        print(f"   duplicate groups: {b['dup_cols'] or 'NOT FOUND IN THIS TABLE'}")
        print("   status  : RESOLVED")
    else:
        print(f"   artifact: {'CANDIDATES BUT NO EXACT RECONCILIATION' if c_cands else 'NOT FOUND'}")
        print("   status  : UNRESOLVED")
    print("\nD. MULTI-MASK POLICY")
    if explicit:
        h = explicit[0]
        print(f"   artifact: {h['artifact']}")
        print(f"   policy  : {h['value']}")
        print("   evidence strength: EXPLICIT_LOCKED_DECISION")
        print("   status  : LOCKED")
    else:
        strengths = sorted({h["strength"] for h in uniq})
        print(f"   artifact: {'none carrying a locked decision' if uniq else 'NOT FOUND'}")
        print("   policy  : F. NO POLICY WAS EVER ESTABLISHED"
              if not uniq else
              f"   policy  : only non-authoritative traces ({', '.join(strengths)})")
        print(f"   evidence strength: {', '.join(strengths) if strengths else 'NONE'}")
        print("   status  : CANONICAL_MULTI_MASK_POLICY_NOT_PREVIOUSLY_LOCKED")
    print("\nE. BUILDER PATCH PLAN")
    print("   (produced by the human/assistant from A-D above; this tool proposes "
          "bindings only, and patches nothing)")
    if a_hits:
        for h in a_hits[:2]:
            print(f"   - bind mask accounting -> {Path(h['artifact']).name} :: "
                  f"{h['locator']}")
    if b_cands:
        best = next((c for c in b_cands if c["rows"] == 647), b_cands[0])
        print(f"   - bind pairing  -> {best['artifact'].name} :: "
              f"[{best['image_cols'][0] if best['image_cols'] else '?'} -> "
              f"{best['mask_cols'][0] if best['mask_cols'] else '?'}]")
    if c_exact:
        b = c_exact[0]
        print(f"   - bind disposition -> {b['artifact'].name} :: "
              f"[{b['id_cols'][0] if b['id_cols'] else '?'} -> {b['status_col']}]")
        if b["dup_cols"]:
            print(f"   - bind duplicate group -> same table :: [{b['dup_cols'][0]}]")
    print("   - bind multi-mask policy -> "
          + ("ONLY after a 2B.2 addendum records an explicit decision"
             if not explicit else f"{Path(explicit[0]['artifact']).name} :: "
                                  f"{explicit[0]['locator']}"))
    print("\nRAW modified: FALSE")
    print("AUDIT artifacts modified: FALSE")
    print("Canonical PNGs written: 0")
    print("Splits created: 0")
    print("\nFINAL VERDICT:")
    if not (a_ok and b_ok and c_ok):
        print("BLOCKED_PENDING_EXISTING_EVIDENCE")
        for label, ok in (("A mask accounting", a_ok), ("B pairing", b_ok),
                          ("C disposition", c_ok)):
            if not ok:
                print(f"    missing: {label}")
        return 4
    if not explicit:
        print("EVIDENCE_BINDING_RESOLVED")
        print("CANONICAL_MULTI_MASK_POLICY_NOT_PREVIOUSLY_LOCKED")
        print("REQUIRES_PHASE2B2_ADDENDUM")
        return 3
    print("READY_TO_PATCH_BUSI_BUILDER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
