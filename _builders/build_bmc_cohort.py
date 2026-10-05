#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bmc_cohort.py
===================

R1 - BMC COHORT LOCK

Assembles the eight federated clients of the BMC revision, and the three external
validation sets, from the finished Canonical Dataset Bank. It READS the bank and
writes only its own manifest: no canonical file, no source manifest, no RAW and no
audit artifact is touched, and no split is created here.

    Client 0  BUSI          Ultrasound       Client 4  INbreast      Mammography
    Client 1  BUS-UC        Ultrasound       Client 5  DMID          Mammography
    Client 2  BrEaST        Ultrasound       Client 6  CBIS-DDSM     Mammography
    Client 3  BUS-BRA       Ultrasound       Client 7  AIISLAB       Mammography

    External  BUS-UCLM, BLUIS, CDD-CESM

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
    Reviewer 1 comment 3, Reviewer 2 comment 2 and Editor item 4 all require the same
    thing: for every dataset, the number of unique grouping units, and an explicit
    statement of whether partitioning can be patient-independent. This script produces
    that table from the manifests rather than from memory, and it REFUSES to call a
    grouping key a patient identifier unless the dataset's own evidence established
    one. Where no patient identifier exists in the public release, the manifest says
    so in plain words, because the honest answer to "were partitions patient
    independent?" is per-dataset, not a single yes.

    The grouping key each client actually gets:
        PATIENT::<id>   where the source or its author established patient identity
        CASE::<id>      where the source established a case identifier only
        GROUP::<id>     an explicit leakage lock that is not a patient claim
        IMAGE::<id>     no grouping identifier exists in the release; one image is the
                        unit, and this is a declared limitation, not a silent default

    Nothing here selects a train, validation or test partition. That is R2, and it
    will consume the group key this manifest fixes.

Usage:
    python build_bmc_cohort.py
    python build_bmc_cohort.py --prep-root "<path to dataset preparing>"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
OUT_DIRNAME = "_bmc_cohort"

# client_id, bank folder, display name, modality, paper reference id
CLIENTS: List[Tuple[int, str, str, str, str]] = [
    (0, "BUSI", "BUSI", "Ultrasound", "D1"),
    (1, "BUS_UC", "BUS-UC", "Ultrasound", "D2"),
    (2, "BrEaST", "BrEaST", "Ultrasound", "D3"),
    (3, "BUS_BRA", "BUS-BRA", "Ultrasound", "D4"),
    (4, "INbreast_FullSource", "INbreast", "Mammography", "D5"),
    (5, "DMID", "DMID", "Mammography", "D6"),
    (6, "CBIS_DDSM", "CBIS-DDSM", "Mammography", "D7"),
    (7, "AIISLAB", "AIISLAB", "Mammography", "D8"),
]
EXTERNALS: List[Tuple[str, str, str]] = [
    ("BUS_UCLM", "BUS-UCLM", "Ultrasound"),
    ("BLUIS", "BLUIS", "Ultrasound"),
    ("CDD_CESM", "CDD-CESM", "Mammography"),
]

# the columns a dataset may use for its grouping unit, in descending strength, with
# the kind of claim each one supports
GROUP_COLUMNS: List[Tuple[str, str]] = [
    ("group_key_for_future_partitions", "AS_DECLARED"),
    ("patient_id", "PATIENT"),
    ("case_id", "CASE"),
    ("duplicate_group_id", "GROUP_LOCK"),
    ("same_source_image_group", "GROUP_LOCK"),
]
INCLUDED_STATUSES = ("INCLUDED_CANONICAL_JOINT", "INCLUDED_JOINT_COHORT",
                     "INCLUDED_TRAINING_COHORT", "INCLUDED_CANONICAL_BANK")
LABEL_COLUMNS = ("source_label", "class_folder", "class_directory",
                 "label_normalized")

MANIFEST_FIELDS = [
    "client_id", "client_name", "dataset", "modality", "paper_dataset_id",
    "role", "canonical_record_id", "sample_id",
    "canonical_image_path", "canonical_mask_path",
    "canonical_image_abspath", "canonical_mask_abspath",
    "label", "label_binary",
    "group_key", "group_unit", "group_provenance",
    "patient_independent_split_possible",
    "source_record_id", "source_image_path", "source_mask_path",
    "mask_policy_id", "image_policy_id", "bank_exclusion_status", "notes",
]
SUMMARY_FIELDS = [
    "paper_dataset_id", "client_id", "dataset", "modality", "role",
    "images", "benign", "malignant", "benign_pct",
    "group_unit", "groups", "images_per_group_max",
    "patient_independent_split_possible", "group_provenance",
    "groups_with_both_labels", "mask_policy_id", "notes",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
    print(f"WROTE {path.name:44s} ({len(rows)} data rows)")


def write_xlsx(path: Path, fields: Sequence[str], rows: Sequence[dict],
               wide: Sequence[str] = ()) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
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
    for c in range(1, len(fields) + 1):
        cell = ws.cell(1, c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2F5496")
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(c)].width = (
            60 if fields[c - 1] in wide else 18)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "B2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    print(f"WROTE {path.name}")


def label_of(r: dict) -> str:
    for c in LABEL_COLUMNS:
        v = str(r.get(c, "") or "").strip().upper()
        if v in ("BENIGN", "MALIGNANT"):
            return v
    v = str(r.get("canonical_binary_label", "") or "").strip()
    if v == "0":
        return "BENIGN"
    if v == "1":
        return "MALIGNANT"
    p = str(r.get("canonical_image_path", "") or "")
    if p.startswith("Benign"):
        return "BENIGN"
    if p.startswith("Malignant"):
        return "MALIGNANT"
    return ""


def is_included(r: dict) -> bool:
    """A record is in the modelling cohort when it was written to the bank. The
    status is checked first; the canonical paths settle a manifest that words the
    status differently."""
    st = str(r.get("exclusion_status", "") or "").strip().upper()
    if st in INCLUDED_STATUSES or st.startswith("INCLUDED"):
        return True
    if st:
        return False
    return bool(str(r.get("canonical_image_path", "") or "").strip()
                and str(r.get("canonical_mask_path", "") or "").strip())


def resolve_group(rows: Sequence[dict]) -> Tuple[str, str, str]:
    """Which column carries this dataset's grouping unit, and what may be claimed of
    it. Returns (column, unit, provenance). A key is never called PATIENT unless the
    manifest itself says patient."""
    cols = list(rows[0]) if rows else []
    for col, kind in GROUP_COLUMNS:
        if col not in cols:
            continue
        vals = [str(r.get(col, "") or "").strip() for r in rows]
        if not any(vals):
            continue
        if kind == "AS_DECLARED":
            pre = {v.split("::", 1)[0].upper() for v in vals if "::" in v}
            # a builder may qualify the prefix with its dataset name, as INbreast does
            # with INBREAST_PATIENT::<id>. The claim the prefix makes is its last
            # token, so the unit is read from that rather than from an exact match,
            # which would otherwise downgrade a real patient key to a generic lock
            # and report a dataset as not patient-independent when it is.
            tail = {x.rsplit("_", 1)[-1] for x in pre}
            if tail == {"PATIENT"}:
                return col, "PATIENT", f"{col} ({'/'.join(sorted(pre))}::<id>)"
            if tail == {"CASE"}:
                return col, "CASE", f"{col} ({'/'.join(sorted(pre))}::<id>)"
            if pre and pre <= {"EXACT_DUP", "GROUP", "LOCK"}:
                return col, "GROUP_LOCK", f"{col} ({'/'.join(sorted(pre))}::<id>)"
            if pre:
                return col, "GROUP_LOCK", f"{col} ({'/'.join(sorted(pre))}::<id>)"
            continue
        if sum(1 for v in vals if v) < len(vals) * 0.9:
            continue
        return col, kind, col
    return "", "IMAGE", ("no grouping identifier is present in the public release; "
                         "one image is the unit")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="R1 - BMC cohort lock.")
    ap.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    a = ap.parse_args(argv)
    prep = Path(a.prep_root)
    bank = prep / BANK_NAME
    out = bank / OUT_DIRNAME

    print("=" * 96)
    print("R1 - BMC COHORT LOCK: eight federated clients + three external sets")
    print("=" * 96)
    print(f"generated : {now_iso()}")
    print(f"bank      : {bank}  [READ-ONLY]")
    print("scope     : reads the canonical manifests; creates NO split and touches no "
          "canonical file")
    if not bank.is_dir():
        print(f"BLOCKED: bank not found: {bank}")
        return 2

    records: List[dict] = []
    summary: List[dict] = []
    problems: List[str] = []
    todo = ([(cid, folder, name, mod, pid, "CLIENT")
             for cid, folder, name, mod, pid in CLIENTS]
            + [("", folder, name, mod, "", "EXTERNAL")
               for folder, name, mod in EXTERNALS])

    for cid, folder, name, mod, pid, role in todo:
        mp = bank / folder / "manifests" / "canonical_manifest.csv"
        if not mp.is_file():
            problems.append(f"{name}: canonical_manifest.csv not found under {folder}")
            print(f"\n  {name}: MANIFEST NOT FOUND ({folder})")
            continue
        rows = read_rows(mp)
        inc = [r for r in rows if is_included(r)]
        gcol, gunit, gprov = resolve_group(inc)
        groups: Dict[str, List[dict]] = defaultdict(list)
        for r in inc:
            raw = str(r.get(gcol, "") or "").strip() if gcol else ""
            if gunit == "IMAGE" or not raw:
                key = (f"IMAGE::{r.get('canonical_record_id') or r.get('sample_id') or r.get('canonical_image_path')}")
                unit_here = "IMAGE"
            else:
                key = raw if "::" in raw else f"{gunit}::{raw}"
                # the per-record unit follows the same rule as the dataset summary:
                # the claim a prefix makes is its last token, so INBREAST_PATIENT
                # reads PATIENT here too. The key itself is left untouched.
                unit_here = key.split("::", 1)[0].rsplit("_", 1)[-1]
            groups[key].append(r)
            lbl = label_of(r)
            rel_i = str(r.get("canonical_image_path", "") or "").strip()
            rel_m = str(r.get("canonical_mask_path", "") or "").strip()
            records.append({
                "client_id": cid, "client_name": (f"Client {cid}" if role == "CLIENT"
                                                  else "EXTERNAL"),
                "dataset": name, "modality": mod, "paper_dataset_id": pid,
                "role": role,
                "canonical_record_id": r.get("canonical_record_id", ""),
                "sample_id": r.get("sample_id", ""),
                "canonical_image_path": rel_i, "canonical_mask_path": rel_m,
                "canonical_image_abspath": str(bank / folder / rel_i) if rel_i else "",
                "canonical_mask_abspath": str(bank / folder / rel_m) if rel_m else "",
                "label": lbl, "label_binary": (0 if lbl == "BENIGN"
                                               else 1 if lbl == "MALIGNANT" else ""),
                "group_key": key, "group_unit": unit_here,
                "group_provenance": gprov,
                "patient_independent_split_possible": ("YES" if unit_here == "PATIENT"
                                                       else "NO"),
                "source_record_id": (r.get("source_record_id")
                                     or r.get("image_reference")
                                     or r.get("segmentation_filename")
                                     or r.get("image_stem") or r.get("sample_id", "")),
                "source_image_path": r.get("source_image_path", ""),
                "source_mask_path": r.get("source_mask_path", ""),
                "mask_policy_id": r.get("mask_policy_id", ""),
                "image_policy_id": r.get("image_policy_id", ""),
                "bank_exclusion_status": r.get("exclusion_status", ""),
                "notes": "",
            })
        if gunit == "GROUP_LOCK" and groups and max(
                len(v) for v in groups.values()) == 1:
            gunit = "IMAGE"
            gprov = (f"{gprov}; every group holds exactly one image, so the unit is "
                     "effectively the image and no grouping constraint exists")
            for rec in records[-len(inc):]:
                rec["group_unit"] = "IMAGE"
        lab = Counter(label_of(r) for r in inc)
        both = sum(1 for v in groups.values()
                   if len({label_of(x) for x in v}) > 1)
        per = [len(v) for v in groups.values()]
        if lab.get("BENIGN", 0) + lab.get("MALIGNANT", 0) != len(inc):
            problems.append(f"{name}: {len(inc)} record(s) but "
                            f"{lab.get('BENIGN', 0)}+{lab.get('MALIGNANT', 0)} labelled")
        miss = [r for r in inc if not str(r.get("canonical_image_path", "")).strip()
                or not str(r.get("canonical_mask_path", "")).strip()]
        if miss:
            problems.append(f"{name}: {len(miss)} included record(s) without both "
                            "canonical paths")
        summary.append({
            "paper_dataset_id": pid, "client_id": cid, "dataset": name,
            "modality": mod, "role": role, "images": len(inc),
            "benign": lab.get("BENIGN", 0), "malignant": lab.get("MALIGNANT", 0),
            "benign_pct": (round(100.0 * lab.get("BENIGN", 0) / len(inc), 1)
                           if inc else 0),
            "group_unit": gunit, "groups": len(groups),
            "images_per_group_max": max(per) if per else 0,
            "patient_independent_split_possible": ("YES" if gunit == "PATIENT"
                                                   else "NO"),
            "group_provenance": gprov,
            "groups_with_both_labels": both,
            "mask_policy_id": "; ".join(sorted(
                {str(r.get("mask_policy_id", "") or "").strip() for r in inc
                 if str(r.get("mask_policy_id", "") or "").strip()})) or "see manifest",
            "notes": "",
        })
        print(f"\n  {('Client ' + str(cid)) if role == 'CLIENT' else 'EXTERNAL':9s} "
              f"{name:12s} {mod:12s} {len(inc):>5d} image(s)  "
              f"{lab.get('BENIGN', 0):>5d} B / {lab.get('MALIGNANT', 0):>5d} M")
        print(f"      grouping: {gunit:10s} {len(groups):>5d} group(s), "
              f"max {max(per) if per else 0} image(s) per group, "
              f"{both} group(s) carry both labels")
        print(f"      source  : {gprov}")

    cl = [s for s in summary if s["role"] == "CLIENT"]
    ex = [s for s in summary if s["role"] == "EXTERNAL"]
    for bucket, label in ((cl, "FEDERATED CLIENTS"), (ex, "EXTERNAL VALIDATION")):
        if not bucket:
            continue
        print("\n" + "=" * 96)
        print(label)
        print("=" * 96)
        print(f"{'ID':4s} {'dataset':12s} {'modality':12s} {'images':>7s} {'B':>6s} "
              f"{'M':>6s} {'unit':>9s} {'groups':>7s} {'max/grp':>8s} {'pat.indep':>10s}")
        print("-" * 96)
        for s in bucket:
            print(f"{str(s['paper_dataset_id'] or '-'):4s} {s['dataset']:12s} "
                  f"{s['modality']:12s} {s['images']:>7d} {s['benign']:>6d} "
                  f"{s['malignant']:>6d} {s['group_unit']:>9s} {s['groups']:>7d} "
                  f"{s['images_per_group_max']:>8d} "
                  f"{s['patient_independent_split_possible']:>10s}")
        print("-" * 96)
        print(f"{'':4s} {'TOTAL':12s} {'':12s} "
              f"{sum(s['images'] for s in bucket):>7d} "
              f"{sum(s['benign'] for s in bucket):>6d} "
              f"{sum(s['malignant'] for s in bucket):>6d} "
              f"{'':>9s} {sum(s['groups'] for s in bucket):>7d}")

    client_recs = [r for r in records if r["role"] == "CLIENT"]
    ext_recs = [r for r in records if r["role"] == "EXTERNAL"]
    gk = Counter(r["group_key"] for r in client_recs)
    shared = [k for k, v in
              Counter((r["dataset"], r["group_key"]) for r in client_recs).items()
              if False]
    cross = [k for k in gk
             if len({r["dataset"] for r in client_recs if r["group_key"] == k}) > 1]
    units = Counter(s["group_unit"] for s in cl)
    pat_ok = [s["dataset"] for s in cl
              if s["patient_independent_split_possible"] == "YES"]
    pat_no = [s["dataset"] for s in cl
              if s["patient_independent_split_possible"] != "YES"]

    checks = [
        ("every client manifest was found and read",
         len(cl) == len(CLIENTS) and len(ex) == len(EXTERNALS)),
        ("every record carries both a canonical image and a canonical mask path",
         all(r["canonical_image_path"] and r["canonical_mask_path"]
             for r in records)),
        ("every record carries a binary label",
         all(r["label_binary"] in (0, 1) for r in records)),
        ("benign + malignant == images, for every dataset",
         all(s["benign"] + s["malignant"] == s["images"] for s in summary)),
        ("every record carries a group key",
         all(r["group_key"] and "::" in r["group_key"] for r in records)),
        ("no group key is shared between two datasets",
         not cross),
        ("no split, fold or partition column exists in this manifest",
         not [f for f in MANIFEST_FIELDS
              if set(f.lower().split("_")) & {"split", "splits", "fold", "folds",
                                              "train", "val", "test", "partition"}
              and f != "patient_independent_split_possible"]),
        ("no dataset without a patient identifier is reported as patient-independent",
         all(s["patient_independent_split_possible"] == "NO"
             for s in summary if s["group_unit"] != "PATIENT")),
        ("no problem found while reading the bank", not problems),
    ]
    print("\n" + "=" * 96)
    for d, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {d}")
    for p in problems:
        print(f"     {p}")
    failed = [d for d, ok in checks if not ok]

    print(f"\n  grouping units across the eight clients: {dict(units)}")
    print(f"  patient-independent splitting POSSIBLE   : "
          f"{', '.join(pat_ok) if pat_ok else 'none'}")
    print(f"  patient-independent splitting NOT possible: "
          f"{', '.join(pat_no) if pat_no else 'none'}")
    print("\n  This is the honest answer to Editor item 4 and Reviewer 1 comment 3:")
    print("  the split can be patient-independent where a patient identifier exists,")
    print("  and is image-level elsewhere because the public release carries no")
    print("  identifier. R2 will group by whatever key this manifest fixed, and the")
    print("  manuscript must state the unit per dataset rather than claim one rule.")

    write_csv(out / "BMC_COHORT_MANIFEST.csv", MANIFEST_FIELDS, records)
    write_xlsx(out / "BMC_COHORT_MANIFEST.xlsx", MANIFEST_FIELDS, records,
               wide=("canonical_image_abspath", "canonical_mask_abspath",
                     "group_provenance", "mask_policy_id"))
    write_csv(out / "TABLE_3_dataset_summary.csv", SUMMARY_FIELDS, summary)
    write_xlsx(out / "TABLE_3_dataset_summary.xlsx", SUMMARY_FIELDS, summary,
               wide=("group_provenance", "mask_policy_id", "notes"))
    (out / "BMC_COHORT_REPORT.json").write_text(json.dumps({
        "phase": "R1_BMC_COHORT_LOCK", "generated": now_iso(),
        "bank": str(bank),
        "manifest_sha256": sha256_text(json.dumps(records, sort_keys=True,
                                                  default=str)),
        "clients": cl, "external": ex,
        "client_totals": {"images": sum(s["images"] for s in cl),
                          "benign": sum(s["benign"] for s in cl),
                          "malignant": sum(s["malignant"] for s in cl),
                          "groups": sum(s["groups"] for s in cl)},
        "external_totals": {"images": sum(s["images"] for s in ex),
                            "benign": sum(s["benign"] for s in ex),
                            "malignant": sum(s["malignant"] for s in ex),
                            "groups": sum(s["groups"] for s in ex)},
        "grouping_units": dict(units),
        "patient_independent_possible": pat_ok,
        "patient_independent_not_possible": pat_no,
        "splits_created": 0, "canonical_files_touched": 0,
        "source_manifests_touched": 0, "raw_touched": False,
        "checks": {d: ok for d, ok in checks}, "problems": problems,
        "verdict": "BMC_COHORT_LOCKED" if not failed else "BLOCKED",
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"WROTE {(out / 'BMC_COHORT_REPORT.json').name}")
    print("\ncanonical files touched: NONE | source manifests: NONE | RAW: NONE | "
          "splits: 0")
    print("\nFINAL VERDICT:")
    if failed:
        print("BLOCKED")
        for d in failed:
            print(f"    FAILED: {d}")
        return 4
    print("BMC_COHORT_LOCKED")
    print("Next: R2 - leakage-safe partitioning on group_key. Nothing is split yet.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
