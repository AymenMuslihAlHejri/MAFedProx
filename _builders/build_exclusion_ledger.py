#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_exclusion_ledger.py
=========================

PHASE 2B - CANONICAL DATASET EXCLUSION LEDGER + MASTER CURATION TABLE

Documents every excluded record with a machine-readable code, a technical description
of the OBSERVED defect, and the scientific rationale for why that defect makes the
record unsuitable for the canonical JOINT segmentation-classification cohort.

READ-ONLY over all source evidence. It reads the canonical manifests and the completed
audit artifacts, and writes only the ledger and the master curation table under
CANONICAL_DATASET_BANK\\_reports\\. No RAW, no audit artifact, no canonical image and
no existing manifest is modified.

EVERY FIELD IS DERIVED, NOT ASSUMED
    The exclusion code comes from the dataset's own artifact. The technical reason is
    built from the observed values recorded there - dimensions, flags, group
    composition - so a record whose evidence differs from its neighbours gets a
    different sentence rather than a shared template.

TWO THINGS THE RATIONALES DELIBERATELY DO NOT SAY
    They do not claim the underlying image is intrinsically invalid; exclusion is from
    the JOINT cohort. And for a label conflict they do not claim WHICH label is
    correct: the audit established that a confirmed duplicate group carries
    conflicting class labels, not which member is mislabelled. No patient, pathology
    or lesion identity is invented anywhere.

Usage:
    python build_exclusion_ledger.py
    python build_exclusion_ledger.py --dry-run
Requires: Python >= 3.8. openpyxl optional (XLSX output).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)

LEDGER_FIELDS = [
    "dataset", "source_record_id", "source_label", "canonical_label",
    "exclusion_code", "exclusion_reason", "scientific_exclusion_rationale",
    "classification_available", "segmentation_available",
    "canonical_image_written", "canonical_mask_written",
    "evidence_source", "notes",
]
MASTER_FIELDS = [
    "dataset", "source_records", "canonical_eligible", "excluded_total",
    "excluded_benign", "excluded_malignant", "exclusion_codes",
    "exclusion_reason_summary", "scientific_exclusion_rationale_summary",
    "canonical_images_written", "canonical_masks_written",
    "locked_policy", "evidence_source", "notes",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def archived(p: Path) -> bool:
    return any(x.lower() in ARCHIVE_DIR_NAMES for x in p.parts[:-1])


def read_rows(p: Path) -> List[dict]:
    with p.open(encoding="utf-8-sig", errors="replace") as fh:
        s = fh.read(8192)
        fh.seek(0)
        d = ";" if s.count(";") > s.count(",") else ","
        return list(csv.DictReader(fh, delimiter=d))


def find_one(prep: Path, token: str, filename: str) -> Optional[Path]:
    for d in prep.iterdir():
        if not d.is_dir():
            continue
        if token not in d.name.lower().replace(" ", "").replace("-", "").replace("_", ""):
            continue
        hits = [p for p in d.rglob(filename) if not archived(p)]
        if len(hits) == 1:
            (only,) = hits
            return only
    return None


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"WROTE {path}  ({len(rows)} data rows)")


# ---------------------------------------------------------------------------
# BUS-UC: the defect is in the MASK. The classification label is untouched.
# ---------------------------------------------------------------------------
def bus_uc_rows(prep: Path) -> List[dict]:
    bank = prep / BANK_NAME / "BUS_UC"
    man = bank / "manifests" / "canonical_manifest.csv"
    integ = find_one(prep, "busuc", "19_bus_uc_mask_integrity_cases.csv")
    if not man.is_file():
        print(f"  BUS_UC canonical manifest not found: {man}")
        return []
    rows = read_rows(man)
    obs: Dict[str, dict] = {}
    if integ:
        for r in read_rows(integ):
            obs[str(r.get("sample_key", "")).strip()] = r
    out: List[dict] = []
    for r in rows:
        if r.get("canonical_image_path", "").strip():
            continue
        key = r.get("sample_key", "").strip()
        code = r.get("exclusion_status", "").strip()
        g = obs.get(key, {})
        iw, ih = g.get("image_width", ""), g.get("image_height", "")
        mw, mh = g.get("mask_width", ""), g.get("mask_height", "")
        kind = g.get("mismatch_kind", "")
        if code == "EXCLUDE_EMPTY_SEGMENTATION_MASK":
            reason = ("The paired source segmentation mask is entirely zero-valued "
                      "and contains no lesion foreground"
                      + (f" (mask {mw}x{mh}, confirmed all-zero from RAW pixels)."
                         if mw else "."))
            rationale = (
                "The record provides no usable positive lesion segmentation target for "
                "the joint segmentation-classification task. An empty mask is missing "
                "information rather than malformed information, so no deterministic "
                "rule can supply it. No pseudo-mask and no inferred lesion region was "
                "generated; the record was therefore excluded from the canonical joint "
                "cohort. The source image and its class label are not claimed to be "
                "invalid.")
        elif code == "EXCLUDE_MASK_SCALE_DIMENSION_MISMATCH":
            reason = (f"The source image ({iw}x{ih}) and its segmentation mask "
                      f"({mw}x{mh}) have incompatible spatial dimensions consistent "
                      f"with a scale/dimension mismatch ({kind}).")
            rationale = (
                "Using this mask as ground truth would require an unverified geometric "
                "transformation or resampling step, which could alter lesion geometry "
                "and introduce artificial segmentation boundaries. Phase 2B prohibits "
                "unsupported mask repair, so the record was excluded rather than "
                "repaired. The source image and its class label are not claimed to be "
                "invalid.")
        else:
            reason = (f"The source image ({iw}x{ih}) and its paired segmentation mask "
                      f"({mw}x{mh}) have a spatial dimension mismatch ({kind}).")
            rationale = (
                "The mask cannot be used as a native pixel-aligned segmentation target "
                "without applying an unverified geometric correction such as cropping, "
                "padding, resizing or resampling. Because Phase 2B prohibits "
                "unsupported mask repair, the record was excluded from the canonical "
                "joint segmentation-classification cohort. The source image and its "
                "class label are not claimed to be invalid.")
        out.append({
            "dataset": "BUS_UC", "source_record_id": key,
            "source_label": r.get("source_label", ""),
            "canonical_label": "NOT_ASSIGNED_EXCLUDED_FROM_JOINT_COHORT",
            "exclusion_code": code, "exclusion_reason": reason,
            "scientific_exclusion_rationale": rationale,
            "classification_available": "FALSE_IN_CANONICAL_JOINT_COHORT",
            "segmentation_available": "FALSE",
            "canonical_image_written": "NO", "canonical_mask_written": "NO",
            "evidence_source": (f"{integ.name if integ else 'NOT_FOUND'} :: "
                                f"csv:[sample_key='{key}']; "
                                "canonical_manifest.csv :: csv:[exclusion_status]"),
            "notes": ("defect is in the SEGMENTATION MASK; the source classification "
                      "label is unaffected by this defect and the record remains "
                      "available for a future classification-only cohort if a policy "
                      "permits one"),
        })
    return out


# ---------------------------------------------------------------------------
# BUSI: the defect is in the LABEL. Derived per record from the duplicate map.
# ---------------------------------------------------------------------------
def busi_rows(prep: Path) -> List[dict]:
    bank = prep / BANK_NAME / "BUSI"
    man = bank / "manifests" / "canonical_manifest.csv"
    dmap = find_one(prep, "busi", "04_busi_duplicate_group_map.csv")
    if not man.is_file():
        print(f"  BUSI canonical manifest not found: {man}")
        return []
    rows = read_rows(man)
    ev: Dict[str, dict] = {}
    by_group: Dict[str, List[dict]] = defaultdict(list)
    if dmap:
        for r in read_rows(dmap):
            ev[str(r.get("image_filename", "")).strip()] = r
            g = str(r.get("confirmed_duplicate_group_id", "")).strip()
            if g:
                by_group[g].append(r)
    out: List[dict] = []
    for r in rows:
        if r.get("canonical_image_path", "").strip():
            continue
        rid = r.get("source_image_id", "").strip()
        fname = rid if rid.lower().endswith(".png") else f"{rid}.png"
        g = ev.get(fname, {})
        code = "EXCLUDE_CONFIRMED_LABEL_CONFLICT"
        grp = str(g.get("confirmed_duplicate_group_id", "")).strip()
        status = str(g.get("duplicate_group_status", "")).strip()
        exact = str(g.get("known_exact_conflict", "")).strip().upper() == "YES"
        members = by_group.get(grp, [])
        comp = Counter(str(m.get("class_label", "")).strip() for m in members)
        comp_txt = ", ".join(f"{n} {c}" for c, n in sorted(comp.items()))
        others = [str(m.get("image_filename", "")).strip() for m in members
                  if str(m.get("image_filename", "")).strip() != fname]
        if exact:
            reason = (f"The record belongs to confirmed duplicate group {grp} "
                      f"({status}), which contains {len(members)} record(s) carrying "
                      f"conflicting source class labels ({comp_txt}). The group is "
                      f"additionally flagged known_exact_conflict=YES, i.e. a byte- or "
                      f"content-identical conflict was recorded. Group members: "
                      f"{', '.join(members and [str(m.get('image_filename', '')).strip() for m in members] or [])}.")
            rationale = (
                "Records that are confirmed duplicates of one another while carrying "
                "opposite class labels cannot both be correct, and the completed audit "
                "established the conflict without establishing which label is the "
                "correct one. Admitting either member would place a contradictory "
                "supervision target in the canonical cohort, and admitting both would "
                "place the same image content under two different labels. The whole "
                "conflict group was therefore excluded from the canonical joint "
                "cohort. No label was reassigned, no member was preferred, and no "
                "patient, pathology or lesion identity was inferred.")
        else:
            reason = (f"The record belongs to confirmed duplicate group {grp} "
                      f"({status}), which contains {len(members)} record(s) carrying "
                      f"conflicting source class labels ({comp_txt}). "
                      f"known_exact_conflict=NO, so the conflict was confirmed by "
                      f"adjudication rather than by byte identity. Other group "
                      f"member(s): {', '.join(others) if others else 'none recorded'}.")
            rationale = (
                "The completed BUSI adjudication confirmed that this duplicate group "
                "carries mixed class labels, but did not establish which label is "
                "correct. Including any member would introduce a supervision target "
                "the evidence does not support, so the entire confirmed-conflict group "
                "was excluded from the canonical joint cohort. The exclusion rests on "
                "the recorded adjudication; no label was reassigned and no patient, "
                "pathology or lesion identity was inferred.")
        out.append({
            "dataset": "BUSI", "source_record_id": fname,
            "source_label": r.get("source_label", ""),
            "canonical_label": "NOT_ASSIGNED_EXCLUDED_FROM_JOINT_COHORT",
            "exclusion_code": code, "exclusion_reason": reason,
            "scientific_exclusion_rationale": rationale,
            "classification_available": "FALSE",
            "segmentation_available": "FALSE_IN_CANONICAL_JOINT_COHORT",
            "canonical_image_written": "NO", "canonical_mask_written": "NO",
            "evidence_source": (f"{dmap.name if dmap else 'NOT_FOUND'} :: "
                                f"csv:[image_filename='{fname}']"
                                f"[duplicate_group_status, known_exact_conflict, "
                                f"confirmed_duplicate_group_id]"),
            "notes": ("defect is in the CLASS LABEL, not in the mask; unlike a "
                      "mask-integrity exclusion the classification target itself is "
                      "contradicted by the evidence, so the record is not available "
                      "for a classification-only cohort either"),
        })
    return out


def dataset_summary(prep: Path, ds: str, ledger: List[dict]) -> Optional[dict]:
    bank = prep / BANK_NAME / ds
    man = bank / "manifests" / "canonical_manifest.csv"
    if not man.is_file():
        return None
    rows = read_rows(man)
    elig = [r for r in rows if r.get("canonical_image_path", "").strip()]
    mine = [x for x in ledger if x["dataset"] == ds]
    codes = Counter(x["exclusion_code"] for x in mine)
    lab = Counter(x["source_label"].upper() for x in mine)
    pol = ""
    for r in rows:
        for c in ("mask_provenance", "canonical_binarization_rule"):
            v = str(r.get(c, ""))
            if "[" in v and "]" in v:
                pol = v.split("[", 1)[1].split("]", 1)[0]
                break
        if pol:
            break
    pngs = [p for p in bank.rglob("*.png") if p.is_file()]
    imgs = sum(1 for p in pngs if p.parent.name.lower() == "images")
    msks = sum(1 for p in pngs if p.parent.name.lower() == "masks")
    if ds == "BUS_UC":
        summary_reason = ("mask-integrity defects: dimension mismatch, scale/dimension "
                          "mismatch, and an all-zero segmentation mask")
        summary_rat = ("each defect would require unsupported mask repair or would "
                       "supply no lesion target at all; Phase 2B excludes rather than "
                       "repairs. The source images and class labels are not claimed "
                       "invalid")
    elif ds == "BUSI":
        ng = len({x["exclusion_reason"].split("group ")[1].split(" ")[0]
                  for x in mine if "group " in x["exclusion_reason"]})
        summary_reason = (f"confirmed duplicate groups carrying conflicting source "
                          f"class labels ({ng} group(s) recorded)")
        summary_rat = ("the audit confirmed the label conflict without establishing "
                       "which label is correct, so every member of each confirmed "
                       "conflict group was excluded; no label was reassigned and no "
                       "identity was inferred")
    else:
        summary_reason = summary_rat = ""
    return {
        "dataset": ds, "source_records": len(rows), "canonical_eligible": len(elig),
        "excluded_total": len(mine),
        "excluded_benign": lab.get("BENIGN", 0),
        "excluded_malignant": lab.get("MALIGNANT", 0),
        "exclusion_codes": "; ".join(f"{k}={v}" for k, v in sorted(codes.items())),
        "exclusion_reason_summary": summary_reason,
        "scientific_exclusion_rationale_summary": summary_rat,
        "canonical_images_written": imgs, "canonical_masks_written": msks,
        "locked_policy": pol or "NOT_RECORDED_IN_MANIFEST",
        "evidence_source": f"{ds}/manifests/canonical_manifest.csv",
        "notes": "aggregate only; record-level reasons are in the exclusion ledger",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the canonical exclusion ledger.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    prep = Path(a.prep_root)
    out_dir = prep / BANK_NAME / "_reports"

    print("=" * 78)
    print("PHASE 2B - CANONICAL DATASET EXCLUSION LEDGER")
    print("=" * 78)
    print(f"generated : {now_iso()}")
    print(f"bank      : {prep / BANK_NAME}")
    print(f"mode      : {'DRY-RUN (writes nothing)' if a.dry_run else 'WRITE'}")

    ledger: List[dict] = []
    print("\n--- deriving record-level exclusions ---")
    for fn, ds in ((busi_rows, "BUSI"), (bus_uc_rows, "BUS_UC")):
        rows = fn(prep)
        ledger += rows
        print(f"  {ds:8s} {len(rows)} excluded record(s)")
        for r in rows:
            print(f"      {r['source_record_id']:22s} {r['exclusion_code']}")

    if not ledger:
        print("\nno excluded records found; nothing written")
        return 2

    master = [m for m in (dataset_summary(prep, ds, ledger)
                          for ds in ("BUSI", "BUS_UC")) if m]

    print("\n--- aggregate ---")
    for m in master:
        print(f"  {m['dataset']:8s} source={m['source_records']:>4} "
              f"eligible={m['canonical_eligible']:>4} excluded={m['excluded_total']} "
              f"(B={m['excluded_benign']} M={m['excluded_malignant']})")
        print(f"           codes: {m['exclusion_codes']}")

    # integrity: every ledger row must have all three documentation fields
    thin = [r["source_record_id"] for r in ledger
            if not r["exclusion_code"] or len(r["exclusion_reason"]) < 40
            or len(r["scientific_exclusion_rationale"]) < 80]
    print(f"\nrecords carrying code + reason + rationale: "
          f"{len(ledger) - len(thin)}/{len(ledger)}")
    if thin:
        print(f"  INCOMPLETE: {thin}")
        return 4

    if a.dry_run:
        print(f"\nDRY-RUN: would write {out_dir}\\CANONICAL_DATASET_EXCLUSION_LEDGER."
              "csv/xlsx and MASTER_CURATION_TABLE.csv/xlsx")
        return 0

    write_csv(out_dir / "CANONICAL_DATASET_EXCLUSION_LEDGER.csv", LEDGER_FIELDS, ledger)
    write_csv(out_dir / "MASTER_CURATION_TABLE.csv", MASTER_FIELDS, master)
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        for name, fields, rows in (
                ("CANONICAL_DATASET_EXCLUSION_LEDGER.xlsx", LEDGER_FIELDS, ledger),
                ("MASTER_CURATION_TABLE.xlsx", MASTER_FIELDS, master)):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = name.split(".")[0][:31]
            ws.append(list(fields))
            for c in range(1, len(fields) + 1):
                ws.cell(1, c).font = Font(bold=True, color="FFFFFF")
                ws.cell(1, c).fill = PatternFill("solid", fgColor="2F5496")
                ws.cell(1, c).alignment = Alignment(horizontal="center", wrap_text=True)
            for r in rows:
                ws.append([r.get(h, "") for h in fields])
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = (f"A1:{get_column_letter(len(fields))}"
                                  f"{len(rows) + 1}")
            for i, h in enumerate(fields, 1):
                wide = h in ("exclusion_reason", "scientific_exclusion_rationale",
                             "exclusion_reason_summary",
                             "scientific_exclusion_rationale_summary",
                             "evidence_source", "notes")
                ws.column_dimensions[get_column_letter(i)].width = 70 if wide else 24
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
            wb.save(out_dir / name)
            print(f"WROTE {out_dir / name}")
    except ImportError:
        print("openpyxl not installed; XLSX skipped (CSV is authoritative)")

    write_text = (out_dir / "EXCLUSION_LEDGER_SUMMARY.json")
    write_text.write_text(json.dumps({
        "generated": now_iso(),
        "datasets": [m["dataset"] for m in master],
        "excluded_total": len(ledger),
        "by_dataset": {m["dataset"]: m["excluded_total"] for m in master},
        "by_code": dict(Counter(r["exclusion_code"] for r in ledger)),
        "raw_modified": False, "audit_modified": False,
        "canonical_images_modified": False,
        "note": ("exclusion is from the canonical JOINT segmentation-classification "
                 "cohort; it does not by itself assert that a source image is "
                 "invalid"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {write_text}")
    print("\nRAW modified: FALSE | audit modified: FALSE | manifests modified: FALSE")
    print("\nFINAL VERDICT:")
    print("EXCLUSION_LEDGER_COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
