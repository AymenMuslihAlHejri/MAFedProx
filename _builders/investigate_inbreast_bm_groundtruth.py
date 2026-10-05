#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
investigate_inbreast_bm_groundtruth.py
======================================

TASK 1 - RECOVER SOURCE-DEFINED BENIGN/MALIGNANT GROUND TRUTH FOR INbreast

Searches ONLY the original full source on H: for a reproducible mapping from the 108
Mass mammograms to a benign/malignant status DEFINED BY THE SOURCE. Read-only: it
opens the workbook, the medical reports, the XML annotations and the documentation
files, and writes one report into the bank's report folder. H: is never modified.

WHAT COUNTS AS A FINDING, AND WHAT DOES NOT
    The INbreast paper states a biopsy was performed on 56 cases, 11 benign and 45
    malignant, the rest being "considered benign" without biopsy. A recoverable ground
    truth therefore means a per-record field or text that states an OUTCOME.

    A referral is not an outcome. A report reading "foram colhidos 4 fragmentos, para
    estudo anatomopatologico" says the biopsy was SENT, not what it found, and a line
    reading "achados imagiologicos muito sugestivos de malignidade - Bi-Rads - 5" is a
    radiological impression restated from BI-RADS. This tool separates the two and
    never counts a referral or a BI-RADS restatement as pathology.

    No benign/malignant value is derived from BI-RADS anywhere in this tool.

Usage:
    python investigate_inbreast_bm_groundtruth.py
    python investigate_inbreast_bm_groundtruth.py --show 6
Requires: Python >= 3.8, pandas, xlrd. pydicom only for the filename-hash linkage.
"""

from __future__ import annotations

import argparse
import json
import plistlib
import re
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_RAW_ROOT = r"H:\BC datasets\Mammo\01 INbreast 2"
DEFAULT_OUT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging\Revision 1"
    r"\dataset preparing\CANONICAL_DATASET_BANK\INbreast_FullSource\reports"
)
WORKBOOK = "INbreast.xls"
DICOM_DIR, XML_DIR, REPORT_DIR = "AllDICOMs", "AllXML", "MedicalReports"
REAL_ROWS, COHORT_COL = 410, "Mass "

# An OUTCOME: a histological entity or an explicit result statement.
OUTCOME_TERMS = {
    "carcinoma": "malignant-indicating histology term",
    "cdi": "carcinoma ductal invasivo",
    "cdis": "carcinoma ductal in situ",
    "adenocarcinoma": "malignant-indicating histology term",
    "invasor": "invasive",
    "invasivo": "invasive",
    "neoplasia maligna": "malignant neoplasm",
    "fibroadenoma": "benign-indicating histology term",
    "quisto": "cyst (benign-indicating)",
    "cisto": "cyst (benign-indicating)",
    "hiperplasia": "hyperplasia",
    "papiloma": "papilloma",
    "fibrose": "fibrosis",
    "resultado": "explicit result statement",
    "histologia": "histology",
    "histologico": "histological",
    "anatomia patologica": "pathology department result",
    "benigno": "explicitly benign",
    "benigna": "explicitly benign",
    "maligno": "explicitly malignant",
    "maligna": "explicitly malignant",
}
# A REFERRAL or an impression: never counted as an outcome.
REFERRAL_TERMS = {
    "para estudo anatomopatologico": "biopsy SENT for pathology, no result",
    "estudo anatomopatologico": "pathology study referenced",
    "microbiopsia": "core biopsy performed",
    "mb-": "core biopsy shorthand",
    "fragmentos": "biopsy fragments taken",
    "sugestivo": "radiological impression",
    "sugestivos": "radiological impression",
    "provavelmente": "radiological impression",
    "suspeitos": "radiological impression",
    "bi-rads": "BI-RADS restatement",
    "birads": "BI-RADS restatement",
}
BM_COLUMN_HINTS = ("biops", "patholog", "histolog", "malign", "benign", "diagnos",
                   "outcome", "result", "cancer", "carcinoma")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c)).lower()


def read_text(p: Path) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    try:
        return p.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def scan_terms(text: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    flat = deaccent(text)
    out = {t: d for t, d in OUTCOME_TERMS.items() if t in flat}
    ref = {t: d for t, d in REFERRAL_TERMS.items() if t in flat}
    return out, ref


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Recover INbreast source-defined B/M.")
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--show", type=int, default=4,
                    help="how many matching report excerpts to print")
    a = ap.parse_args(argv)
    raw = Path(a.raw_root)
    out_dir = Path(a.out_dir)

    try:
        import pandas as pd
    except ImportError:
        sys.stderr.write("pandas (and xlrd for .xls) required\n")
        return 3

    print("=" * 78)
    print("INbreast - SEARCH FOR A SOURCE-DEFINED BENIGN/MALIGNANT GROUND TRUTH")
    print("=" * 78)
    print(f"generated : {now_iso()}")
    print(f"source    : {raw}  [READ-ONLY]")
    print("rule      : BI-RADS is NEVER used to derive benign/malignant here")

    findings: Dict[str, Any] = {"generated": now_iso(), "raw_root": str(raw),
                                "birads_used_for_bm": False}

    # ---- A. the workbook -----------------------------------------------------
    print("\n" + "-" * 78)
    print("A. INbreast.xls")
    wb = raw / WORKBOOK
    if not wb.is_file():
        alt = raw / "INbreast.xlsx"
        wb = alt if alt.is_file() else wb
    if not wb.is_file():
        print("   workbook not found")
        return 2
    raw_df = pd.read_excel(wb)
    df = raw_df.iloc[:REAL_ROWS].copy()
    col = COHORT_COL if COHORT_COL in df.columns else next(
        (c for c in df.columns if c.strip().lower() == "mass"), None)
    mass = df[df[col].astype(str).str.strip().str.upper() == "X"].copy()
    print(f"   columns ({len(df.columns)}): {list(df.columns)}")
    hits = [c for c in df.columns if any(h in deaccent(str(c)) for h in BM_COLUMN_HINTS)]
    print(f"   columns whose NAME suggests biopsy/pathology: {hits or 'NONE'}")
    free_cols = [c for c in ("Findings Notes (in Portuguese)", "Other Notes",
                             "Other Annotations", "Lesion Annotation Status")
                 if c in df.columns]
    wb_outcome_rows = []
    for c in free_cols:
        vals = df[c].dropna().astype(str)
        uniq = Counter(v.strip() for v in vals if v.strip())
        oc = {v: scan_terms(v)[0] for v in uniq if scan_terms(v)[0]}
        print(f"   [{c}] {len(vals)} non-empty, {len(uniq)} distinct; "
              f"values containing an OUTCOME term: {len(oc)}")
        for v, terms in list(oc.items())[:a.show]:
            print(f"        {v[:70]!r} -> {sorted(terms)}")
        for v in oc:
            wb_outcome_rows.append({"column": c, "value": v})
    findings["workbook"] = {
        "path": str(wb), "columns": [str(c) for c in df.columns],
        "bm_named_columns": hits, "free_text_outcome_values": wb_outcome_rows,
        "mass_rows": int(len(mass))}

    # ---- B. the medical reports ----------------------------------------------
    print("\n" + "-" * 78)
    print("B. MedicalReports")
    reps = sorted((raw / REPORT_DIR).glob("*.txt"))
    print(f"   report files: {len(reps)}")
    rep_rows: List[dict] = []
    for p in reps:
        txt = read_text(p)
        oc, rf = scan_terms(txt)
        rep_rows.append({"case_id": p.stem, "file": p.name, "chars": len(txt),
                         "outcome_terms": sorted(oc), "referral_terms": sorted(rf),
                         "text": txt})
    with_outcome = [r for r in rep_rows if r["outcome_terms"]]
    referral_only = [r for r in rep_rows if not r["outcome_terms"] and r["referral_terms"]]
    neither = [r for r in rep_rows if not r["outcome_terms"] and not r["referral_terms"]]
    print(f"   reports containing an OUTCOME term          : {len(with_outcome)}")
    print(f"   reports containing ONLY referral/impression : {len(referral_only)}")
    print(f"   reports containing neither                  : {len(neither)}")
    term_freq = Counter(t for r in rep_rows for t in r["outcome_terms"])
    print(f"   outcome-term frequency: {dict(term_freq.most_common())}")
    for r in with_outcome[:a.show]:
        snippet = " ".join(r["text"].split())
        keep = []
        for t in r["outcome_terms"]:
            m = re.search(rf".{{0,70}}{re.escape(t)}.{{0,70}}", deaccent(snippet))
            if m:
                keep.append(m.group(0))
        print(f"\n      case {r['case_id']}  terms={r['outcome_terms']}")
        for k in keep[:2]:
            print(f"         ...{k}...")

    # ---- C. link to the 108 ---------------------------------------------------
    print("\n" + "-" * 78)
    print("C. linkage from the 108 Mass mammograms to the reports")
    hash_by_num: Dict[str, str] = {}
    for p in (raw / DICOM_DIR).glob("*.dcm"):
        parts = p.name.split("_")
        if len(parts) > 1:
            hash_by_num[parts[0]] = parts[1]
    nums = []
    for v in mass["File Name"].dropna():
        try:
            nums.append(str(int(v)))
        except (TypeError, ValueError):
            nums.append(str(v).strip())
    cohort_hash = {n: hash_by_num.get(n, "") for n in nums}
    missing_hash = [n for n, h in cohort_hash.items() if not h]
    groups = {h for h in cohort_hash.values() if h}
    rep_ids = {r["case_id"] for r in rep_rows}
    outcome_ids = {r["case_id"] for r in with_outcome}
    linked = {n: h for n, h in cohort_hash.items() if h in rep_ids}
    linked_outcome = {n: h for n, h in cohort_hash.items() if h in outcome_ids}
    print(f"   cohort images                      : {len(nums)}")
    print(f"   distinct patient groups (hash)     : {len(groups)}")
    print(f"   images with no DICOM hash          : {len(missing_hash)}")
    print(f"   images whose group has ANY report  : {len(linked)}")
    print(f"   images whose group has an OUTCOME  : {len(linked_outcome)}")
    print(f"   groups with an OUTCOME report      : "
          f"{len({h for h in linked_outcome.values()})} of {len(groups)}")

    # ---- D. XML and documentation --------------------------------------------
    print("\n" + "-" * 78)
    print("D. XML annotations and documentation")
    xml_keys: Counter = Counter()
    xml_scanned = 0
    for n in nums:
        xp = raw / XML_DIR / f"{n}.xml"
        if not xp.is_file():
            continue
        try:
            with xp.open("rb") as fh:
                blob = plistlib.load(fh, fmt=plistlib.FMT_XML)
        except Exception:  # noqa: BLE001
            continue
        xml_scanned += 1
        rois = []
        for im in (blob.get("Images") or []):
            rois.extend(im.get("ROIs") or [])
        for r in rois:
            xml_keys.update(r.keys())
    bm_keys = [k for k in xml_keys if any(h in deaccent(str(k)) for h in BM_COLUMN_HINTS)]
    print(f"   XML scanned: {xml_scanned}; distinct ROI keys: {sorted(xml_keys)}")
    print(f"   ROI keys suggesting pathology: {bm_keys or 'NONE'}")
    docs = [p for p in raw.iterdir()
            if p.is_file() and p.suffix.lower() in (".txt", ".pdf", ".csv", ".doc",
                                                    ".docx", ".rtf")]
    doc_rows = []
    print(f"   documentation files: {[p.name for p in docs]}")
    for p in docs:
        if p.suffix.lower() == ".pdf":
            print(f"      {p.name}: PDF, not parsed here "
                  f"({p.stat().st_size // 1024} KB) - open it if the search below "
                  "finds nothing")
            doc_rows.append({"file": p.name, "parsed": False})
            continue
        txt = read_text(p)
        oc, _rf = scan_terms(txt)
        print(f"      {p.name}: {len(txt)} chars; outcome terms: {sorted(oc) or 'none'}")
        doc_rows.append({"file": p.name, "parsed": True, "outcome_terms": sorted(oc)})

    # ---- E. verdict -------------------------------------------------------------
    print("\n" + "=" * 78)
    recoverable = bool(linked_outcome) and len(linked_outcome) == len(nums)
    partial = bool(linked_outcome) and not recoverable
    print("1. source of B/M information")
    if wb_outcome_rows:
        print(f"   workbook free-text values carrying an outcome term: "
              f"{len(wb_outcome_rows)}")
    if with_outcome:
        print(f"   MedicalReports: {len(with_outcome)} of {len(reps)} report(s) carry "
              "an outcome term")
        print(f"      (a further {len(referral_only)} carry only a biopsy REFERRAL or "
              "a BI-RADS restatement, which is not an outcome)")
    if bm_keys:
        print(f"   XML ROI keys: {bm_keys}")
    if not (wb_outcome_rows or with_outcome or bm_keys):
        print("   NONE FOUND in the workbook, the reports or the XML")
    print("\n2. exact linkage key to the 108 Mass images")
    print("   DICOM filename hash token == MedicalReports/<hash>.txt stem "
          "== REPORT_PATIENT_OR_CASE_ID  (patient level, not image level)")
    print(f"   {len(groups)} group(s) cover the 108 images; "
          f"{len(missing_hash)} image(s) have no hash")
    print("\n3. B/M counts among the 108")
    print("   NOT COMPUTED: an outcome term in a patient report does not assign a "
          "status to a specific mammogram. A patient group here holds several images "
          "and both breasts; the source states no per-image outcome.")
    print("\n4. biopsy-confirmed vs clinically-considered-benign")
    print(f"   reports naming a biopsy REFERRAL: {len(referral_only) + sum(1 for r in with_outcome if r['referral_terms'])}")
    print(f"   reports naming an OUTCOME       : {len(with_outcome)}")
    print("   The paper's 56 biopsies (11 benign / 45 malignant) are reported at CASE "
          "level in the publication and are not distributed with the database.")
    print("\n5. unmatched / ambiguous")
    print(f"   cohort images with no report for their group: "
          f"{len(nums) - len(linked)}")
    print(f"   cohort images whose group has a report but no outcome term: "
          f"{len(linked) - len(linked_outcome)}")
    print("\n6. VERDICT")
    verdict = ("SOURCE_DEFINED_BM_RECOVERABLE_PER_IMAGE" if recoverable else
               "PARTIAL_PATIENT_LEVEL_EVIDENCE_ONLY" if partial else
               "NOT_RECOVERABLE")
    print(f"   {verdict}")
    if verdict != "SOURCE_DEFINED_BM_RECOVERABLE_PER_IMAGE":
        print("   The database does not distribute a per-image benign/malignant "
              "status. Anything recovered above is patient-level free text, which "
              "cannot be assigned to one of a patient's four mammograms without an "
              "assumption the source does not support.")

    findings.update({
        "reports": {"files": len(reps), "with_outcome_term": len(with_outcome),
                    "referral_or_impression_only": len(referral_only),
                    "neither": len(neither),
                    "outcome_term_frequency": dict(term_freq),
                    "cases_with_outcome": sorted(outcome_ids)},
        "linkage": {"key": "DICOM filename hash == report stem == "
                           "REPORT_PATIENT_OR_CASE_ID",
                    "level": "PATIENT_GROUP_NOT_IMAGE",
                    "cohort_images": len(nums), "groups": len(groups),
                    "images_without_hash": missing_hash,
                    "images_group_has_report": len(linked),
                    "images_group_has_outcome": len(linked_outcome)},
        "xml": {"scanned": xml_scanned, "roi_keys": sorted(xml_keys),
                "pathology_keys": bm_keys},
        "documentation": doc_rows,
        "bm_counts_among_108": None,
        "verdict": verdict,
        "raw_modified": False})
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = out_dir / "INBREAST_BM_GROUNDTRUTH_INVESTIGATION.json"
    rep_path.write_text(json.dumps(findings, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nWROTE {rep_path}")
    print("RAW modified: FALSE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
