#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_bank_reports.py
=======================

BANK-WIDE EXCLUSION LEDGER AND MASTER CURATION TABLE

Rebuilds CANONICAL_DATASET_EXCLUSION_LEDGER and MASTER_CURATION_TABLE by reading EVERY
dataset's canonical manifest, rather than accumulating one dataset at a time. Until now
each dataset appended its own row, so the master table held only the datasets that
happened to have exclusions and the ledger depended on the order things were run.

WHY READ THE MANIFESTS AND NOT THE PREVIOUS REPORTS
    The canonical manifest is written by the builder and confirmed by the independent
    on-disk verifier, so it is the only artifact that cannot drift from what is on
    disk. This tool derives every number from those manifests and counts the PNG files
    itself, so a figure that disagrees with the bank is reported rather than repeated.

THE THREE CATEGORIES, KEPT APART
    Datasets use different words for the same three situations, so this tool
    normalises them and never merges them:

    INCLUDED        the record is in the canonical cohort
    EXCLUDED        the record was eligible and then removed for a defect. ONLY these
                    reach the exclusion ledger.
    NOT IN TARGET   the record was never eligible: a normal class, an ambiguous or
                    undefined label, a class outside the binary target. These are
                    counted in their own column of the master table and are NOT
                    exclusions. Merging them would misrepresent 512 source records as
                    curation failures.

It writes CSV and XLSX for both artifacts. It does not touch any canonical image, any
manifest, RAW, or the completed audits.

Usage:
    python rebuild_bank_reports.py --dry-run
    python rebuild_bank_reports.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
SKIP_DIRS = {"_reports", "_policies", "_builders", "_scratch"}

LEDGER_FIELDS = [
    "dataset", "source_record_id", "source_label", "canonical_label",
    "exclusion_code", "exclusion_reason", "scientific_exclusion_rationale",
    "classification_available", "segmentation_available",
    "canonical_image_written", "canonical_mask_written", "evidence_source", "notes"]
MASTER_FIELDS = [
    "dataset", "source_records", "not_in_binary_target", "eligible_after_selection",
    "task_ineligible_total", "task_ineligible_benign", "task_ineligible_malignant",
    "task_ineligibility_reasons",
    "classification_only_total", "classification_only_benign",
    "classification_only_malignant", "classification_only_reasons",
    "excluded_total", "excluded_benign", "excluded_malignant", "canonical_records",
    "canonical_benign", "canonical_malignant",
    "canonical_images_on_disk", "canonical_masks_on_disk",
    "exclusion_codes", "exclusion_reason_summary",
    "not_in_target_breakdown", "partition_unit_available",
    "locked_policy", "evidence_source", "notes"]

# per-dataset facts the manifests do not carry
GROUPING = {
    "BUSI": "Image level (no patient identifier in the release)",
    "BUS_UC": "Image level (no patient identifier in the release)",
    "BrEaST": "Image level (no patient identifier in the release)",
    "BUS_BRA": "Case level (1064 cases, 811 with more than one image)",
    "INbreast_FullSource": "Patient level (50 patient groups in the cohort)",
    "BLUIS": "Image level (no patient identifier in the release)",
    "BUS_UCLM": "Patient level (38 patients overall, 35 in the cohort)",
    "RODTOOK": "Image level; the only grouping key is the confirmed exact-duplicate "
               "group (4 groups, 8 members)",
    "DMID": "Image level; PatientID and PatientName are empty and the UIDs must not "
            "be repurposed as patient identifiers",
}
NOTES = {
    "INbreast_FullSource": ("binary labels assigned from source BI-RADS by a "
                            "pre-specified rule, NOT from pathology; the database "
                            "distributes no per-image benign/malignant status"),
    "DMID": ("7 classification-valid records have no verified raster GT: their images "
             "are converted and labelled, only the segmentation target is absent; no "
             "pseudo-mask was generated"),
    "RODTOOK": ("4 exact-duplicate groups (8 members) are RETAINED as leakage keys, "
                "not deleted; all within one class, so no label conflict"),
    "BLUIS": ("232 auxiliary lesion images ship alongside and never enter the "
              "modelling pool"),
    "BUS_UCLM": ("labels come from the official INFO.csv; the colour-coded mask was "
                 "cross-checked against them and never used to derive them"),
}

INCLUDED_PREFIXES = ("INCLUDED",)
# Later datasets named the cohort status differently, and one of them uses the word
# "EXCLUDE" for a record that is merely outside the current task. Both lists are
# explicit so no status is classified by guessing at its prefix.
INCLUDED_EXACT = ("INCLUDED_TRAINING_COHORT", "INCLUDED_JOINT_COHORT",
                  "INCLUDED_CANONICAL_JOINT", "INCLUDED_CANONICAL_BANK")
TASK_INELIGIBLE_MARKERS = ("JOINT_TASK_INELIGIBLE",)
# A record that keeps a valid classification label and loses ONLY its segmentation
# target is not excluded from the bank: it is classification-only. The status alone is
# not enough to decide this, because some of these codes begin with EXCLUDE, so the
# record's own availability fields are checked too and a record is classified this way
# only when the evidence says classification is available and segmentation is not.
SEGMENTATION_ONLY_MARKERS = ("NO_VERIFIED_RASTER_SEGMENTATION_GT",
                             "SEGMENTATION_LINKAGE_UNRESOLVED",
                             "NO_RASTER_PIXEL_GT", "NO_SEGMENTATION_GT")
NOT_IN_TARGET_MARKERS = ("NOT_IN_BINARY_TARGET", "NOT_IN_CLASS_COHORT",
                         "NOT_IN_JOINT_COHORT", "NOT_IN_TRAINING_COHORT",
                         "EXCLUDE_FROM_BINARY_LESION_COHORT")


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


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"WROTE {path}  ({len(rows)} rows)")


def write_xlsx(path: Path, fields: Sequence[str], rows: Sequence[dict],
               wide: Sequence[str] = (), total_row: bool = False) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        print(f"      openpyxl missing; {path.name} skipped")
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = path.stem[:31]
    ws.append(list(fields))
    for r in rows:
        ws.append([r.get(h, "") for h in fields])
    thin = Side(style="thin", color="BFBFBF")
    bd = Border(left=thin, right=thin, top=thin, bottom=thin)
    for c in range(1, len(fields) + 1):
        cell = ws.cell(1, c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2F5496")
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        ws.column_dimensions[get_column_letter(c)].width = (
            70 if fields[c - 1] in wide else 20)
    for row in ws.iter_rows(min_row=1, max_row=len(rows) + 1, max_col=len(fields)):
        for cell in row:
            cell.border = bd
            if cell.row > 1:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
    if total_row and rows:
        last = len(rows) + 1
        for c in range(1, len(fields) + 1):
            ws.cell(last, c).font = Font(bold=True)
            ws.cell(last, c).fill = PatternFill("solid", fgColor="D9E2F3")
    ws.freeze_panes = "B2"
    ws.row_dimensions[1].height = 42
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    print(f"WROTE {path}")


STATUS_PAT = re.compile(
    r"^(INCLUDED|EXCLUDE|NOT_IN|ELIGIBLE|JOINT_TASK_INELIGIBLE)", re.I)


def detect_status_column(rows: List[dict]) -> Tuple[str, int]:
    """Find the column holding the CANONICAL status.

    Name first, content second, and in that order for a reason. Several manifests
    carry more than one status-shaped column: BUS-UCLM has both exclusion_status and
    modeling_status, and the latter marks its 419 normal samples
    EXCLUDE_FROM_BINARY_LESION_COHORT. Choosing by content alone would read those as
    exclusions and turn a class that was never eligible into 419 curation failures.
    exclusion_status is the column this bank writes, so it wins whenever it is present
    and populated; content detection is only the fallback for an older manifest that
    lacks it."""
    def score(col: str) -> int:
        return sum(1 for r in rows
                   if STATUS_PAT.match(str(r.get(col, "") or "").strip()))
    cols = list(rows[0]) if rows else []
    for preferred in ("exclusion_status", "canonical_status", "record_status"):
        if preferred in cols:
            n = score(preferred)
            if n >= len(rows) * 0.9:
                return preferred, n
    best, best_n = "", 0
    for col in cols:
        n = score(col)
        if n > best_n:
            best, best_n = col, n
    return best, best_n


def detect_label(r: dict) -> str:
    """The binary label, from whichever column this dataset used for it."""
    for col in ("source_label", "class_folder", "class_directory",
                "label_normalized", "canonical_class"):
        v = str(r.get(col, "") or "").strip().upper()
        if v in ("BENIGN", "MALIGNANT"):
            return v
    v = str(r.get("canonical_binary_label", "") or "").strip()
    if v == "0":
        return "BENIGN"
    if v == "1":
        return "MALIGNANT"
    # last resort: the canonical path itself says which folder it was written to
    pth = str(r.get("canonical_image_path", "") or "")
    if pth.startswith("Benign"):
        return "BENIGN"
    if pth.startswith("Malignant"):
        return "MALIGNANT"
    return ""


def classify(status: str, row: Optional[dict] = None) -> str:
    s = (status or "").strip().upper()
    if not s:
        return "UNKNOWN"
    if s in INCLUDED_EXACT or any(s.startswith(p) for p in INCLUDED_PREFIXES) \
            or s.startswith("ELIGIBLE"):
        return "INCLUDED"
    # a record the current task cannot use is NOT a curation failure: it keeps its
    # source label and its files are simply not written. It is counted apart from
    # both the cohort and the exclusions, and it never reaches the ledger.
    if any(m in s for m in TASK_INELIGIBLE_MARKERS):
        return "TASK_INELIGIBLE"
    if any(m in s for m in SEGMENTATION_ONLY_MARKERS):
        cls = str((row or {}).get("classification_available", "")).strip().upper()
        seg = str((row or {}).get("segmentation_available", "")).strip().upper()
        if cls in ("TRUE", "YES") and seg in ("FALSE", "NO", ""):
            return "CLASSIFICATION_ONLY"
        # the status says segmentation, but the record's own fields do not confirm
        # that classification survived; it is not reclassified on the strength of a
        # name alone
        return "EXCLUDED"
    if any(m in s for m in NOT_IN_TARGET_MARKERS):
        return "NOT_IN_TARGET"
    if s.startswith("EXCLUDE"):
        return "EXCLUDED"
    return "UNKNOWN"


def first(r: dict, *names: str) -> str:
    for n in names:
        v = str(r.get(n, "") or "").strip()
        if v:
            return v
    return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild the bank-wide reports.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    prep = Path(a.prep_root)
    bank = prep / BANK_NAME
    shared = bank / "_reports"

    print("=" * 88)
    print("CANONICAL DATASET BANK - EXCLUSION LEDGER AND MASTER CURATION TABLE")
    print("=" * 88)
    print(f"generated : {now_iso()}")
    print(f"mode      : {'DRY-RUN (writes nothing)' if a.dry_run else 'WRITE'}")
    print(f"bank      : {bank}")
    if not bank.is_dir():
        print("bank not found")
        return 2

    datasets = sorted(d for d in bank.iterdir()
                      if d.is_dir() and d.name not in SKIP_DIRS
                      and (d / "manifests" / "canonical_manifest.csv").is_file())
    print(f"datasets  : {len(datasets)} with a canonical manifest -> "
          + ", ".join(d.name for d in datasets))
    missing = [d.name for d in bank.iterdir()
               if d.is_dir() and d.name not in SKIP_DIRS
               and not (d / "manifests" / "canonical_manifest.csv").is_file()]
    if missing:
        print(f"            (no manifest yet, skipped: {', '.join(sorted(missing))})")

    ledger: List[dict] = []
    master: List[dict] = []
    problems: List[str] = []

    for d in datasets:
        name = d.name
        rows = read_rows(d / "manifests" / "canonical_manifest.csv")
        scol, scount = detect_status_column(rows)
        how = f"status column '{scol}' on {scount}/{len(rows)} row(s)"
        if not scol or scount < len(rows) * 0.9:
            # BUSI's manifest predates the shared vocabulary: its status column is
            # exclusion_or_special_status and its values are IN_BINARY_TARGET,
            # LABEL_CONFLICT and DUPLICATE_LOCKED, none of which start with the
            # keywords the pattern looks for. Rather than hard-code that dataset's
            # words, fall back to the pair of columns every manifest carries and that
            # state the same fact unambiguously: whether the record is available for
            # classification and for segmentation. A record marked NO for both was
            # not converted; the canonical paths then say whether anything was
            # written for it at all.
            avail_cols = ("classification_available", "segmentation_available")
            # the WORD for the disposition comes from the dataset's own status column,
            # not from whatever the pattern happened to score highest: BUSI's
            # canonical_binary_label reads NOT_IN_BINARY_TARGET on its excluded rows,
            # which would misfile a real exclusion as a class that was never eligible
            cols0 = list(rows[0]) if rows else []
            wcol = next((c for c in ("exclusion_or_special_status", "exclusion_status",
                                     "special_status", "record_status")
                         if c in cols0), "")
            if not wcol:
                wcol = next((c for c in cols0
                             if "status" in c.lower()
                             and "conversion" not in c.lower()), "")
            if all(c in cols0 for c in avail_cols):
                how = (f"availability fallback ({' + '.join(avail_cols)}), with the "
                       f"disposition word from '{wcol or 'none'}'; the pattern matched "
                       f"only {scount}/{len(rows)} on '{scol or 'none'}'")
                for r in rows:
                    yes = all(str(r.get(c, "")).strip().upper() in ("YES", "TRUE")
                              for c in avail_cols)
                    raw_status = str(r.get(wcol, "") or "").strip() if wcol else ""
                    if yes:
                        r["_status"] = "INCLUDED_CANONICAL_COHORT"
                    else:
                        # not converted: an exclusion if it was eligible and removed,
                        # otherwise a class that was never in the binary target
                        r["_status"] = (raw_status
                                        if raw_status.upper().startswith(
                                            ("EXCLUDE", "NOT_IN"))
                                        else f"EXCLUDE_{raw_status}"
                                        if raw_status else "EXCLUDE_UNSPECIFIED")
                    r["_status_source"] = raw_status
                scol = "|".join(avail_cols)
            else:
                problems.append(
                    f"{name}: no column holds a canonical status on at least 90% of "
                    f"rows (best: {scol or 'none'} on {scount}/{len(rows)}) and the "
                    "availability columns are absent")
                for r in rows:
                    r["_status"] = str(r.get(scol, "") or "").strip() if scol else ""
        else:
            for r in rows:
                r["_status"] = str(r.get(scol, "") or "").strip()
        buckets: Dict[str, List[dict]] = {"INCLUDED": [], "EXCLUDED": [],
                                          "NOT_IN_TARGET": [], "TASK_INELIGIBLE": [],
                                          "CLASSIFICATION_ONLY": [], "UNKNOWN": []}
        for r in rows:
            buckets[classify(r["_status"], r)].append(r)
        if buckets["UNKNOWN"]:
            codes = Counter(r.get("_status") for r in buckets["UNKNOWN"])
            problems.append(f"{name}: {len(buckets['UNKNOWN'])} row(s) with an "
                            f"unrecognised status {dict(codes)}")
        inc, exc, nit = (buckets["INCLUDED"], buckets["EXCLUDED"],
                         buckets["NOT_IN_TARGET"])
        tin = buckets["TASK_INELIGIBLE"]
        con = buckets["CLASSIFICATION_ONLY"]

        label_of = detect_label

        cb = Counter(label_of(r) for r in inc)
        benign = cb.get("BENIGN", 0)
        malignant = cb.get("MALIGNANT", 0)
        imgs = len(list((d).rglob("Images/*.png")))
        masks = len(list((d).rglob("Masks/*.png")))
        codes = Counter(r.get("_status") for r in exc)
        nit_codes = Counter(r.get("_status") for r in nit)

        print()
        print(f"  {name}")
        print(f"     disposition read from: {how}")
        print(f"     source {len(rows)} | included {len(inc)} "
              f"({benign}B/{malignant}M) | FULL-excluded {len(exc)} | "
              f"classification-only {len(con)} | task-ineligible {len(tin)} | "
              f"not in target {len(nit)}")
        print(f"     on disk: {imgs} image(s), {masks} mask(s)")
        if len(inc) + len(exc) + len(nit) + len(tin) + len(con) \
                + len(buckets["UNKNOWN"]) != len(rows):
            problems.append(f"{name}: buckets do not sum to the row count")
        if benign + malignant != len(inc):
            problems.append(f"{name}: {benign}+{malignant} != {len(inc)} included")
        if exc:
            print(f"     exclusion codes: {dict(codes)}")
        if nit:
            print(f"     not-in-target  : {dict(nit_codes)}")
        if tin:
            print(f"     task-ineligible: "
                  f"{dict(Counter(r.get('_status') for r in tin))}")
        if con:
            print(f"     classification-only (NOT a full exclusion): "
                  f"{dict(Counter(r.get('_status') for r in con))}")

        for r in exc:
            rid = first(r, "source_record_id", "sample_id", "image_reference",
                        "source_file_number", "sample_key", "canonical_record_id")
            ledger.append({
                "dataset": name, "source_record_id": rid,
                "source_label": label_of(r) or "NOT_ESTABLISHED",
                "canonical_label": first(r, "canonical_binary_label") or
                                   "NOT_IN_CANONICAL_COHORT",
                "exclusion_code": r.get("_status", ""),
                "exclusion_reason": (first(r, "exclusion_reason")
                                     or (f"source status "
                                         f"'{r.get('_status_source')}'; see notes"
                                         if r.get("_status_source") else "")),
                "scientific_exclusion_rationale": first(
                    r, "scientific_exclusion_rationale", "notes"),
                "classification_available": first(r, "classification_available")
                                            or "FALSE",
                "segmentation_available": first(r, "segmentation_available")
                                          or "FALSE",
                "canonical_image_written": ("YES" if first(r, "canonical_image_path")
                                            else "NO"),
                "canonical_mask_written": ("YES" if first(r, "canonical_mask_path")
                                           else "NO"),
                "evidence_source": f"{name}/manifests/canonical_manifest.csv :: "
                                   f"[{rid}]",
                "notes": first(r, "notes")[:400]})

        master.append({
            "dataset": name, "source_records": len(rows),
            "not_in_binary_target": len(nit),
            "eligible_after_selection": len(inc) + len(exc) + len(tin) + len(con),
            "task_ineligible_total": len(tin),
            "task_ineligible_benign": sum(1 for r in tin if label_of(r) == "BENIGN"),
            "task_ineligible_malignant": sum(1 for r in tin
                                             if label_of(r) == "MALIGNANT"),
            "task_ineligibility_reasons": "; ".join(
                f"{k}={v}" for k, v in sorted(
                    Counter(r.get("_status") for r in tin).items())) or "none",
            "classification_only_total": len(con),
            "classification_only_benign": sum(1 for r in con
                                              if label_of(r) == "BENIGN"),
            "classification_only_malignant": sum(1 for r in con
                                                 if label_of(r) == "MALIGNANT"),
            "classification_only_reasons": "; ".join(
                f"{k}={v}" for k, v in sorted(
                    Counter(r.get("_status") for r in con).items())) or "none",
            "excluded_total": len(exc),
            "excluded_benign": sum(1 for r in exc if label_of(r) == "BENIGN"),
            "excluded_malignant": sum(1 for r in exc if label_of(r) == "MALIGNANT"),
            "canonical_records": len(inc), "canonical_benign": benign,
            "canonical_malignant": malignant,
            "canonical_images_on_disk": imgs, "canonical_masks_on_disk": masks,
            "exclusion_codes": "; ".join(f"{k}={v}" for k, v in sorted(codes.items()))
                               or "none",
            "exclusion_reason_summary": (
                "; ".join(sorted({first(r, "exclusion_reason")[:110] for r in exc}))
                or "no record was excluded after selection"),
            "not_in_target_breakdown": "; ".join(
                f"{k}={v}" for k, v in sorted(nit_codes.items())) or "none",
            "partition_unit_available": GROUPING.get(name, "not recorded"),
            "locked_policy": "; ".join(sorted(
                {first(r, "mask_policy_id") for r in inc
                 if first(r, "mask_policy_id")}
                | {first(r, "image_policy_id") for r in inc
                   if first(r, "image_policy_id")})) or "not recorded",
            "evidence_source": f"{name}/manifests/canonical_manifest.csv",
            "notes": NOTES.get(name, "")})

    T = lambda k: sum(m[k] for m in master)  # noqa: E731
    master.append({
        "dataset": "TOTAL", "source_records": T("source_records"),
        "not_in_binary_target": T("not_in_binary_target"),
        "eligible_after_selection": T("eligible_after_selection"),
        "task_ineligible_total": T("task_ineligible_total"),
        "task_ineligible_benign": T("task_ineligible_benign"),
        "task_ineligible_malignant": T("task_ineligible_malignant"),
        "task_ineligibility_reasons": "",
        "classification_only_total": T("classification_only_total"),
        "classification_only_benign": T("classification_only_benign"),
        "classification_only_malignant": T("classification_only_malignant"),
        "classification_only_reasons": "",
        "excluded_total": T("excluded_total"),
        "excluded_benign": T("excluded_benign"),
        "excluded_malignant": T("excluded_malignant"),
        "canonical_records": T("canonical_records"),
        "canonical_benign": T("canonical_benign"),
        "canonical_malignant": T("canonical_malignant"),
        "canonical_images_on_disk": T("canonical_images_on_disk"),
        "canonical_masks_on_disk": T("canonical_masks_on_disk"),
        "exclusion_codes": f"{len(ledger)} record(s) in the exclusion ledger",
        "exclusion_reason_summary": "", "not_in_target_breakdown": "",
        "partition_unit_available": "", "locked_policy": "",
        "evidence_source": "derived from every dataset's canonical manifest",
        "notes": ""})

    print()
    print("=" * 88)
    print(f"{'dataset':22s} {'source':>7s} {'notTgt':>7s} {'elig':>6s} {'tskIn':>6s} "
          f"{'clsOnly':>8s} {'FULLexcl':>9s} {'canon':>6s} {'B':>6s} {'M':>6s} "
          f"{'img':>6s} {'mask':>6s}")
    print("-" * 88)
    for m in master:
        print(f"{m['dataset']:22s} {m['source_records']:>7d} "
              f"{m['not_in_binary_target']:>7d} {m['eligible_after_selection']:>6d} "
              f"{m['task_ineligible_total']:>6d} "
              f"{m['classification_only_total']:>8d} "
              f"{m['excluded_total']:>9d} {m['canonical_records']:>6d} "
              f"{m['canonical_benign']:>6d} {m['canonical_malignant']:>6d} "
              f"{m['canonical_images_on_disk']:>6d} "
              f"{m['canonical_masks_on_disk']:>6d}")
    print("-" * 88)
    print(f"ledger rows: {len(ledger)}  "
          f"{dict(Counter(x['dataset'] for x in ledger))}")

    tot = master[-1]
    checks = [
        ("eligible - FULL-excluded - task-ineligible - classification-only == "
         "canonical, everywhere",
         all(m["eligible_after_selection"] - m["excluded_total"]
             - m["task_ineligible_total"] - m["classification_only_total"]
             == m["canonical_records"] for m in master)),
        ("no task-ineligible record reached the ledger",
         not any("JOINT_TASK_INELIGIBLE" in x["exclusion_code"] for x in ledger)),
        ("no classification-only record reached the ledger: a record that keeps its "
         "label and loses only its segmentation target is not a full exclusion",
         not any(any(m in x["exclusion_code"] for m in SEGMENTATION_ONLY_MARKERS)
                 for x in ledger)),

        ("benign + malignant == canonical records, everywhere",
         all(m["canonical_benign"] + m["canonical_malignant"]
             == m["canonical_records"] for m in master)),
        ("the ledger holds exactly the excluded records, no more",
         len(ledger) == tot["excluded_total"]),
        ("every ledger row carries a code and a reason",
         all(x["exclusion_code"] and x["exclusion_reason"] for x in ledger)),
        ("no not-in-target record reached the ledger",
         not any("NOT_IN" in x["exclusion_code"] for x in ledger)),
        ("no unrecognised status in any manifest", not problems),
    ]
    print()
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {desc}")
    for p in problems:
        print(f"     {p}")
    failed = [d for d, ok in checks if not ok]

    disk = [m for m in master[:-1]
            if m["canonical_images_on_disk"] != m["canonical_records"]]
    if disk:
        print("\n  NOTE: image files on disk differ from the canonical record count "
              "for:")
        for m in disk:
            print(f"     {m['dataset']}: {m['canonical_images_on_disk']} file(s) vs "
                  f"{m['canonical_records']} record(s)"
                  + ("  (expected: records without a mask still have an image)"
                     if m["excluded_total"] else ""))

    if a.dry_run:
        print("\nDRY-RUN: nothing written")
        print(f"   would write the ledger ({len(ledger)} rows) and the master table "
              f"({len(master)} rows), each as CSV and XLSX")
        return 0 if not failed else 4

    print()
    write_csv(shared / "CANONICAL_DATASET_EXCLUSION_LEDGER.csv", LEDGER_FIELDS, ledger)
    write_xlsx(shared / "CANONICAL_DATASET_EXCLUSION_LEDGER.xlsx", LEDGER_FIELDS,
               ledger, wide=("exclusion_reason", "scientific_exclusion_rationale",
                             "evidence_source", "notes"))
    write_csv(shared / "MASTER_CURATION_TABLE.csv", MASTER_FIELDS, master)
    write_xlsx(shared / "MASTER_CURATION_TABLE.xlsx", MASTER_FIELDS, master,
               wide=("exclusion_reason_summary", "not_in_target_breakdown",
                     "task_ineligibility_reasons", "classification_only_reasons",
                     "partition_unit_available", "locked_policy", "notes"),
               total_row=True)
    (shared / "BANK_REPORTS_REBUILD.json").write_text(json.dumps({
        "generated": now_iso(), "datasets": [d.name for d in datasets],
        "ledger_semantics": ("CANONICAL_DATASET_EXCLUSION_LEDGER holds ONLY records "
                             "fully excluded from the bank. A record that keeps a "
                             "valid classification label and lacks only a "
                             "segmentation target is classification-only and is "
                             "represented in its dataset manifest, not here."),
        "ledger_rows": len(ledger), "master_rows": len(master),
        "totals": {k: tot[k] for k in
                   ("source_records", "not_in_binary_target",
                    "eligible_after_selection", "task_ineligible_total",
                    "classification_only_total", "excluded_total",
                    "canonical_records", "canonical_benign", "canonical_malignant",
                    "canonical_images_on_disk", "canonical_masks_on_disk")},
        "checks": {d: ok for d, ok in checks}, "problems": problems,
        "canonical_files_touched": 0, "manifests_touched": 0, "raw_touched": False,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {shared / 'BANK_REPORTS_REBUILD.json'}")
    print("\ncanonical files touched: NONE | manifests touched: NONE | RAW: NONE")
    print("\nFINAL VERDICT:")
    if failed:
        print("BLOCKED")
        for d in failed:
            print(f"    FAILED: {d}")
        return 4
    print("BANK_REPORTS_REBUILT")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
