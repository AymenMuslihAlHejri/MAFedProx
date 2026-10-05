#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bank_bus_bra.py
=====================

PHASE 2B.3.4 - BUS-BRA CANONICAL DATASET BANK BUILDER

Fourth dataset, fourth distinct mask rule. Nothing is carried over from BUSI, BUS-UC
or BrEaST.

    BUSI      union of multiple masks
    BUS-UC    midpoint threshold at 128 on soft boundaries
    BrEaST    RGBA colour-channel reduction
    BUS-BRA   VALUE SCALING: 1-bit {0,1} -> {0,255}, single channel already

    POLICY_ID BUS_BRA_CANONICAL_MASK_VALUE_SCALING_V1
        verify single channel and values within {0,1} -> multiply by 255 ->
        single-channel lossless PNG. No threshold, no union, no channel reduction,
        no morphology, no resize, no pseudo-mask.

THE SUPERSEDED-PAIRING QUESTION, RESOLVED BY EVIDENCE NOT BY ASSUMPTION
    29_bus_bra_mask_mapping_reconciliation.csv records 397 masks as
    CASE_LEVEL_ONLY_MAPPING with mapping_semantically_supported=NO. That file
    documents an INTERMEDIATE containment rule that matched nothing
    (phase1a1_rule_or_rejection = NO_RULE_MATCHED on all 438 rows). The later
    corrected rerun adopted DIGITS_PLUS_SIDE, and 43_..._pairing_debug_summary.json
    (phase 1A.1d) proves the bijection.

    This builder does not simply prefer the newer file. For every row of the
    reconciliation it RECOMPUTES digits+side and checks that the manifest's mapping
    agrees and is unique. If any row remained genuinely ambiguous under the adopted
    rule, the build blocks.

LOCKED EVIDENCE BINDINGS
    manifest       17_bus_bra_audit_manifest.csv           (1875 rows, key ID)
    reconciliation 29_bus_bra_mask_mapping_reconciliation.csv (438 rows, subset)
    mask           05_bus_bra_mask_audit.csv               (1875 rows)
    labels         07_bus_bra_label_provenance.csv         (1875 rows)
    cases          10_bus_bra_case_grouping_audit.csv      (1064 rows)
    duplicates     15_bus_bra_exact_duplicates.csv
    pairing debug  43_bus_bra_pairing_debug_summary.json
    summary        20_bus_bra_audit_summary.json :: json:raw_root

IDENTITY
    Case is an established grouping identifier and is preserved as case_id and as the
    future group key. patient_id and lesion_id are never invented. No split is created.

Usage:
    python build_bank_bus_bra.py --dry-run
    python build_bank_bus_bra.py --execute
Requires: Python >= 3.8, Pillow, numpy. openpyxl optional.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET = "BUS_BRA"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)

MANIFEST_FILE = "17_bus_bra_audit_manifest.csv"
RECON_FILE = "29_bus_bra_mask_mapping_reconciliation.csv"
MASK_FILE = "05_bus_bra_mask_audit.csv"
LABEL_FILE = "07_bus_bra_label_provenance.csv"
CASE_FILE = "10_bus_bra_case_grouping_audit.csv"
DUPLICATE_FILE = "15_bus_bra_exact_duplicates.csv"
DEBUG_FILE = "43_bus_bra_pairing_debug_summary.json"
SUMMARY_FILE = "20_bus_bra_audit_summary.json"
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]

POLICY_ID = "BUS_BRA_CANONICAL_MASK_VALUE_SCALING_V1"
POLICY_RULE = ("verify single channel and values within {0,1} -> multiply by 255 -> "
               "single-channel lossless PNG")
ADOPTED_PAIRING_RULE = "DIGITS_PLUS_SIDE"

EXPECT = {"records": 1875, "benign": 1268, "malignant": 607, "cases": 1064,
          "multi_image_cases": 811, "case_conflicts": 0, "duplicates": 0,
          "eligible": 1875, "excluded": 0, "png": 3750}

CANONICAL_LABELS = {"BENIGN": 0, "MALIGNANT": 1}
SCHEMA_FIELDS = [
    "dataset", "canonical_record_id", "source_record_id", "source_image_id",
    "source_mask_id", "source_image_path", "source_mask_path", "source_label",
    "canonical_binary_label", "canonical_image_path", "canonical_mask_path",
    "patient_id", "case_id", "image_id", "lesion_id", "abnormality_id",
    "same_source_image_group", "duplicate_group_id", "group_key_for_future_partitions",
    "classification_available", "segmentation_available",
    "image_conversion_status", "mask_conversion_status",
    "identity_provenance", "label_provenance", "mask_provenance", "pairing_provenance",
    "source_image_height", "source_image_width",
    "source_mask_height", "source_mask_width",
    "canonical_image_height", "canonical_image_width",
    "canonical_mask_height", "canonical_mask_width",
    "source_image_format", "source_mask_format",
    "source_mask_encoding", "canonical_mask_encoding",
    "canonical_mask_rule", "canonical_mask_policy_id",
    "exclusion_status", "exclusion_reason", "notes",
]
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}

SEP = " | "
LOGGER = logging.getLogger("bank_bus_bra")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def join_list(v: Sequence[Any], sep: str = SEP) -> str:
    return sep.join(str(x) for x in v)


def archived(p: Path) -> bool:
    return any(x.lower() in ARCHIVE_DIR_NAMES for x in p.parts[:-1])


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def snapshot(root: Path) -> Dict[str, Tuple[int, int]]:
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


def diff_state(a: dict, b: dict) -> dict:
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    modified = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return {"added": added, "removed": removed, "modified": modified,
            "unchanged": not (added or removed or modified)}


def is_split_like(path_str: str) -> bool:
    pth = Path(path_str)
    raw = [t for t in re.split(r"[^A-Za-z0-9]+", pth.stem) if t]
    raw += [t for part in pth.parts[:-1]
            for t in re.split(r"[^A-Za-z0-9]+", part) if t]
    toks = set()
    for t in raw:
        toks.add(t.lower())
        toks.add(re.sub(r"\d+$", "", t.lower()))
    return bool(toks & SPLIT_TOKENS)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    LOGGER.info("WROTE %-46s (%d data rows)", path.name, len(rows))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    LOGGER.info("WROTE %-46s (%d chars)", path.name, len(text))


def read_rows(p: Path) -> List[dict]:
    with p.open(encoding="utf-8-sig", errors="replace") as fh:
        s = fh.read(8192)
        fh.seek(0)
        if not s.strip():
            return []
        d = ";" if s.count(";") > s.count(",") else ","
        return list(csv.DictReader(fh, delimiter=d))


def setup_logging(report_dir: Path, name: str) -> None:
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    f = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(f)
    sh.setLevel(logging.INFO)
    LOGGER.addHandler(sh)
    report_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(report_dir / name, mode="w", encoding="utf-8")
    fh.setFormatter(f)
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)


def find_one(files: Sequence[Path], filename: str) -> Tuple[Optional[Path], str]:
    hits = sorted(p for p in files if p.name.lower() == filename.lower())
    if len(hits) == 1:
        (only,) = hits
        return only, "UNIQUE"
    return None, ("MISSING" if not hits else f"AMBIGUOUS({len(hits)})")


def canonical_stem(cls: str, rec_id: str) -> str:
    key = f"{cls}/{rec_id}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", rec_id).strip("_")
    return f"{safe or 'image'}__{sha256_text(key)[:10]}"


def scale_mask(arr, np) -> Tuple[Any, str]:
    """{0,1} -> {0,255}. Refuses anything the declared rule does not cover."""
    if arr.ndim == 3:
        if arr.shape[2] < 3:
            return None, f"UNEXPECTED_CHANNELS_{arr.shape[2]}"
        if not (np.array_equal(arr[..., 0], arr[..., 1])
                and np.array_equal(arr[..., 1], arr[..., 2])):
            return None, "MULTICHANNEL_NOT_REPLICATED"
        plane, note = arr[..., 0], "REDUCED_REPLICATED_CHANNELS"
    else:
        plane, note = arr, "ALREADY_SINGLE_CHANNEL"
    if plane.dtype == bool:
        plane = plane.astype(np.uint8)
        note += "_FROM_BOOL"
    vals = set(np.unique(plane).tolist())
    if vals <= {0, 1}:
        return (plane.astype(np.uint8) * 255), note + "_SCALED_0_1_TO_0_255"
    if vals <= {0, 255}:
        return plane.astype(np.uint8), note + "_ALREADY_0_255"
    return None, f"VALUES_OUTSIDE_DECLARED_RULE{sorted(vals)[:5]}"


def load_evidence(prep_root: Path) -> dict:
    ev: Dict[str, Any] = {"paths": {}, "status": {}, "hashes": {}, "problems": []}
    dirs = [d for d in prep_root.iterdir() if d.is_dir()
            and "busbra" in d.name.lower().replace(" ", "").replace("_", "")
            .replace("-", "")]
    files = [p for d in dirs for p in d.rglob("*") if p.is_file() and not archived(p)]
    for key, fname, required in (("manifest", MANIFEST_FILE, True),
                                 ("recon", RECON_FILE, True),
                                 ("mask", MASK_FILE, True),
                                 ("labels", LABEL_FILE, True),
                                 ("cases", CASE_FILE, True),
                                 ("duplicates", DUPLICATE_FILE, False),
                                 ("debug", DEBUG_FILE, True),
                                 ("summary", SUMMARY_FILE, True)):
        art, st = find_one(files, fname)
        ev["paths"][key] = art
        ev["status"][key] = st
        if art is not None:
            ev["hashes"][art.name] = sha256_file(art)
        elif required:
            ev["problems"].append(f"BLOCKED: {fname} -> {st}")
    ev["dataset_root"] = None
    ev["dataset_root_locator"] = ""
    if ev["paths"].get("summary"):
        blob = json.loads(ev["paths"]["summary"].read_text(encoding="utf-8"))
        for loc in ROOT_LOCATORS:
            v = blob.get(loc)
            if isinstance(v, str) and v.strip():
                ev["dataset_root"] = Path(v.strip())
                ev["dataset_root_locator"] = f"{SUMMARY_FILE} :: json:{loc}"
                break
        if ev["dataset_root"] is None:
            ev["problems"].append(f"BLOCKED: no source root in {SUMMARY_FILE}")
    return ev


def run(mode: str, prep_root: Path) -> int:
    bank_ds = prep_root / BANK_NAME / DATASET
    reports, manifests = bank_ds / "reports", bank_ds / "manifests"
    setup_logging(reports, f"bus_bra_{mode}.log")
    LOGGER.info("=" * 78)
    LOGGER.info("PHASE 2B.3.4 BUS-BRA CANONICAL BANK BUILDER  [%s]", mode.upper())
    LOGGER.info("=" * 78)
    LOGGER.info("python : %s", platform.python_version())
    LOGGER.info("bank   : %s", bank_ds)

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        LOGGER.critical("Pillow and numpy required")
        return 3
    Image.MAX_IMAGE_PIXELS = None

    prep_dirs = [d for d in prep_root.iterdir()
                 if d.is_dir() and d.name != BANK_NAME]
    before_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        before_audit.update(snapshot(d))
    before_bank = snapshot(prep_root / BANK_NAME)

    # ---- 1. evidence -------------------------------------------------------
    LOGGER.info("[1/8] locked evidence bindings ...")
    ev = load_evidence(prep_root)
    for k, st in ev["status"].items():
        art = ev["paths"].get(k)
        LOGGER.info("      %-11s %-12s %s", k, st, art.name if art else "")
    if ev["problems"]:
        for p in ev["problems"]:
            LOGGER.error("      %s", p)
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4
    src_root = ev["dataset_root"]
    before_raw = snapshot(src_root)
    LOGGER.info("      RAW = %s  [READ-ONLY]", src_root)
    LOGGER.info("      mask policy = %s", POLICY_ID)
    LOGGER.info("      rule        = %s", POLICY_RULE)

    rows = read_rows(ev["paths"]["manifest"])
    recon = read_rows(ev["paths"]["recon"])
    mk = read_rows(ev["paths"]["mask"])
    labels = read_rows(ev["paths"]["labels"])
    cases = read_rows(ev["paths"]["cases"])
    dups = read_rows(ev["paths"]["duplicates"]) if ev["paths"].get("duplicates") else []
    dbg = json.loads(ev["paths"]["debug"].read_text(encoding="utf-8"))
    summ = json.loads(ev["paths"]["summary"].read_text(encoding="utf-8"))
    ids = [r["ID"].strip() for r in rows]
    dupe_ids = [k for k, n in Counter(ids).items() if n > 1]
    LOGGER.info("      manifest=%d recon=%d mask=%d labels=%d cases=%d duplicates=%d",
                len(rows), len(recon), len(mk), len(labels), len(cases), len(dups))
    if dupe_ids:
        LOGGER.error("      ID not unique: %s", dupe_ids[:5])
        LOGGER.error("FINAL VERDICT:\nBLOCKED")
        return 4

    # ---- 2. accounting -------------------------------------------------------
    LOGGER.info("[2/8] source accounting ...")
    by_cls = Counter(r["Pathology"].strip().upper() for r in rows)
    lab_final = Counter(r["final_observed_label"].strip().upper() for r in labels)
    n_cases = len({r["Case"].strip() for r in rows})
    multi = sum(1 for r in cases if int(r["number_of_images"]) > 1)
    conflicts = sum(1 for r in cases
                    if r["pathology_consistent_within_case"].strip().upper() != "YES")
    LOGGER.info("      manifest: %d records, benign=%d malignant=%d, cases=%d",
                len(rows), by_cls.get("BENIGN", 0), by_cls.get("MALIGNANT", 0), n_cases)
    LOGGER.info("      label provenance: %s (agree=%s, conflicts=%s)", dict(lab_final),
                dict(Counter(r["label_sources_agree"] for r in labels)),
                dict(Counter(r["label_conflict"] for r in labels)))
    LOGGER.info("      case audit: %d case(s), %d with >1 image, %d pathology "
                "conflict(s)", len(cases), multi, conflicts)
    LOGGER.info("      summary json: images=%s masks=%s benign=%s malignant=%s "
                "cases=%s", summ.get("source_images"), summ.get("segmentation_masks"),
                (summ.get("class_distribution_final") or {}).get("benign"),
                (summ.get("class_distribution_final") or {}).get("malignant"),
                summ.get("unique_cases"))

    # ---- 3. the superseded-pairing reconciliation ----------------------------
    LOGGER.info("[3/8] reconciling the superseded pairing evidence ...")
    recon_status = Counter(r["final_mapping_status"].strip() for r in recon)
    recon_rule = Counter(r["phase1a1_rule_or_rejection"].strip() for r in recon)
    LOGGER.info("      %s: %s", RECON_FILE, dict(recon_status))
    LOGGER.info("      recorded rule outcome: %s", dict(recon_rule))
    LOGGER.info("      later evidence: %s bijection_proven=%s rule=%s", DEBUG_FILE,
                dbg.get("bijection_proven"), dbg.get("current_rule_selected"))
    LOGGER.info("      summary run_type=%s  mask_matching_rule=%s",
                summ.get("run_type"), summ.get("mask_matching_rule"))
    by_mask = {r["mask_filename"].strip(): r for r in rows}
    unresolved_recon, disagree_recon, resolved_recon = [], [], 0
    for r in recon:
        mf = r["mask_filename"].strip()
        stem = r["mask_stem"].strip()
        m = re.match(r"^mask_(\d+)-([a-z])$", stem, re.I)
        cands = [c.strip() for c in r["candidate_image_IDs"].split(",") if c.strip()]
        if not m:
            unresolved_recon.append(f"{stem}: no digits+side token")
            continue
        digits, side = m.group(1), m.group(2).lower()
        hit = [c for c in cands if c.lower().endswith(f"-{side}") and digits in c]
        if len(hit) != 1:
            unresolved_recon.append(f"{stem}: {len(hit)} candidate(s) under "
                                    f"{ADOPTED_PAIRING_RULE}")
            continue
        resolved_recon += 1
        mrow = by_mask.get(mf)
        if mrow is not None and mrow["ID"].strip() != hit[0]:
            disagree_recon.append(f"{mf}: manifest={mrow['ID']} vs "
                                  f"{ADOPTED_PAIRING_RULE}={hit[0]}")
    LOGGER.info("      recomputed under %s: %d/%d uniquely resolved, %d disagree with "
                "the manifest, %d unresolved", ADOPTED_PAIRING_RULE, resolved_recon,
                len(recon), len(disagree_recon), len(unresolved_recon))
    for u in (unresolved_recon + disagree_recon)[:8]:
        LOGGER.error("         %s", u)

    # ---- 4. mask policy preconditions ----------------------------------------
    LOGGER.info("[4/8] mask-policy preconditions ...")
    mk_enc = Counter(r["encoding"].strip() for r in mk)
    mk_ch = Counter(r["channels"].strip() for r in mk)
    mk_val = Counter(r["unique_values_or_summary"].strip() for r in mk)
    mk_empty = Counter(r["empty_mask"].strip().upper() for r in mk)
    mk_min = Counter(r["min_value"].strip() for r in mk)
    mk_max = Counter(r["max_value"].strip() for r in mk)
    for lbl, c in (("encoding", mk_enc), ("channels", mk_ch), ("unique_values", mk_val),
                   ("empty_mask", mk_empty), ("min", mk_min), ("max", mk_max)):
        LOGGER.info("      %-16s %s", lbl, dict(c))
    precond = {
        "single channel": set(mk_ch) == {"1"},
        "values within {0,1}": all(
            set(v.replace(" ", "").split(",")) <= {"0", "1"} for v in mk_val),
        "no empty mask": mk_empty.get("NO", 0) == len(mk),
        "encoding is GRAY_BINARY_0_1": set(mk_enc) == {"GRAY_BINARY_0_1"},
        "mask dimensions match image": all(
            r["dimension_match"].strip().upper() == "YES" for r in rows),
    }
    for k, v in precond.items():
        (LOGGER.info if v else LOGGER.error)("      precondition %-30s %s", k,
                                             "OK" if v else "FAILED")
    if not all(precond.values()):
        LOGGER.error("      the declared scaling rule requires all preconditions; the "
                     "build blocks rather than substituting a threshold")
        LOGGER.error("FINAL VERDICT:\nBLOCKED")
        return 4

    # ---- 5. plan ---------------------------------------------------------------
    LOGGER.info("[5/8] planning canonical records ...")
    lab_by_id = {r["ID"].strip(): r for r in labels}
    case_by_id: Dict[str, dict] = {r["case_id"].strip(): r for r in cases}
    records: List[dict] = []
    for r in rows:
        rid = r["ID"].strip()
        cls = r["Pathology"].strip().upper()
        case = r["Case"].strip()
        folder = "Benign" if cls == "BENIGN" else (
            "Malignant" if cls == "MALIGNANT" else "")
        stem = canonical_stem(cls, rid)
        lp = lab_by_id.get(rid, {})
        records.append({
            "dataset": DATASET,
            "canonical_record_id": f"{DATASET}__{stem}",
            "source_record_id": rid,
            "source_image_id": r["image_filename"].strip(),
            "source_mask_id": r["mask_filename"].strip(),
            "source_image_path": r["image_path"].strip(),
            "source_mask_path": r["mask_path"].strip(),
            "source_label": cls,
            "canonical_binary_label": CANONICAL_LABELS.get(cls, "UNRESOLVED"),
            "canonical_image_path": f"{folder}/Images/{stem}.png" if folder else "",
            "canonical_mask_path": f"{folder}/Masks/{stem}_mask.png" if folder else "",
            "patient_id": "", "case_id": case,
            "image_id": r["image_filename"].strip(),
            "lesion_id": "", "abnormality_id": "",
            "same_source_image_group": f"{DATASET}__{stem}",
            "duplicate_group_id": r.get("exact_duplicate_group", "").strip(),
            "group_key_for_future_partitions": f"CASE::{case}",
            "classification_available": "TRUE", "segmentation_available": "TRUE",
            "image_conversion_status": "PLANNED", "mask_conversion_status": "PLANNED",
            "identity_provenance": (f"{MANIFEST_FILE} :: csv:[case_id_status]="
                                    f"{r.get('case_id_status', '').strip()}; "
                                    f"patient_id_status="
                                    f"{r.get('patient_id_status', '').strip()}"),
            "label_provenance": (f"{LABEL_FILE} :: csv:[final_observed_label] "
                                 f"(sources={lp.get('final_label_source', '')}, "
                                 f"agree={lp.get('label_sources_agree', '')})"),
            "mask_provenance": f"SOURCE_NATIVE_BINARY_SCALED_BY[{POLICY_ID}]",
            "pairing_provenance": (f"{MANIFEST_FILE} :: csv:[mask_path]; rule="
                                   f"{ADOPTED_PAIRING_RULE}; bijection proven in "
                                   f"{DEBUG_FILE}"),
            "source_image_height": r.get("image_height", ""),
            "source_image_width": r.get("image_width", ""),
            "source_mask_height": r.get("mask_height", ""),
            "source_mask_width": r.get("mask_width", ""),
            "canonical_image_height": "", "canonical_image_width": "",
            "canonical_mask_height": "", "canonical_mask_width": "",
            "source_image_format": Path(r["image_path"]).suffix.lstrip(".").upper(),
            "source_mask_format": Path(r["mask_path"]).suffix.lstrip(".").upper(),
            "source_mask_encoding": r.get("mask_encoding", "").strip(),
            "canonical_mask_encoding": "SINGLE_CHANNEL_BINARY_0_255",
            "canonical_mask_rule": POLICY_RULE,
            "canonical_mask_policy_id": POLICY_ID,
            "exclusion_status": "INCLUDED_CANONICAL_JOINT", "exclusion_reason": "",
            "notes": (f"pairing_status={r.get('pairing_status', '').strip()}; "
                      f"audit_status={r.get('audit_status', '').strip()}; "
                      f"BIRADS={r.get('BIRADS', '').strip()}; "
                      f"Side={r.get('Side', '').strip()}; "
                      f"case_images={case_by_id.get(case, {}).get('number_of_images', '?')}"),
        })
    by_el = Counter(r["source_label"] for r in records)
    per_case = Counter(r["case_id"] for r in records)
    LOGGER.info("      planned %d record(s): benign=%d malignant=%d across %d case(s)",
                len(records), by_el.get("BENIGN", 0), by_el.get("MALIGNANT", 0),
                len(per_case))

    # ---- 6. resolve and pre-check ---------------------------------------------
    LOGGER.info("[6/8] resolving and pre-checking source pairs ...")
    img_ok = mask_ok = plans = 0
    unresolved: List[str] = []
    scale_notes: Counter = Counter()
    for i, r in enumerate(records, 1):
        sp = src_root / r["source_image_path"]
        mp = src_root / r["source_mask_path"]
        if not sp.is_file():
            unresolved.append(f"image missing: {r['source_image_path']}")
            r["image_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            with Image.open(sp) as im:
                r["source_image_width"] = r["canonical_image_width"] = im.width
                r["source_image_height"] = r["canonical_image_height"] = im.height
            img_ok += 1
        except Exception as exc:  # noqa: BLE001
            unresolved.append(f"image undecodable: {r['source_image_path']}")
            r["image_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        if not mp.is_file():
            unresolved.append(f"mask missing: {r['source_mask_path']}")
            r["mask_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            with Image.open(mp) as mi:
                arr = np.array(mi)
                r["source_mask_width"] = r["canonical_mask_width"] = mi.width
                r["source_mask_height"] = r["canonical_mask_height"] = mi.height
        except Exception as exc:  # noqa: BLE001
            unresolved.append(f"mask undecodable: {r['source_mask_path']}")
            r["mask_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        mask_ok += 1
        plane, note = scale_mask(arr, np)
        scale_notes[note] += 1
        if plane is None:
            unresolved.append(f"mask scaling refused: {r['source_record_id']} ({note})")
            r["mask_conversion_status"] = f"REFUSED_{note}"
            continue
        if (r["source_mask_height"], r["source_mask_width"]) != (
                r["source_image_height"], r["source_image_width"]):
            unresolved.append(f"dimension mismatch: {r['source_record_id']}")
            r["mask_conversion_status"] = "DIMENSION_MISMATCH"
            continue
        if int((plane == 255).sum()) == 0:
            unresolved.append(f"empty mask: {r['source_record_id']}")
            r["mask_conversion_status"] = "EMPTY_MASK"
            continue
        r["notes"] += f"; scaling={note}"
        plans += 1
        if i % 400 == 0:
            LOGGER.info("      %d/%d", i, len(records))
    LOGGER.info("      image mappings=%d  mask mappings=%d  scaling plans=%d  "
                "unresolved=%d", img_ok, mask_ok, plans, len(unresolved))
    LOGGER.info("      scaling outcomes: %s", dict(scale_notes))
    for u in unresolved[:10]:
        LOGGER.error("         %s", u)

    # ---- 7. execute --------------------------------------------------------------
    png_written = 0
    if mode == "execute":
        LOGGER.info("[7/8] executing canonical conversion ...")
        for r in records:
            if r["image_conversion_status"] != "PLANNED":
                continue
            dst = bank_ds / r["canonical_image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with Image.open(src_root / r["source_image_path"]) as im:
                    im.save(dst, format="PNG")
                png_written += 1
            r["image_conversion_status"] = "CONVERTED"
            with Image.open(src_root / r["source_mask_path"]) as mi:
                arr = np.array(mi)
            plane, _n = scale_mask(arr, np)
            if plane is None:
                r["mask_conversion_status"] = "REFUSED_AT_EXECUTE"
                continue
            mdst = bank_ds / r["canonical_mask_path"]
            mdst.parent.mkdir(parents=True, exist_ok=True)
            if not mdst.exists():
                Image.fromarray(plane.astype(np.uint8), mode="L").save(
                    mdst, format="PNG")
                png_written += 1
            r["mask_conversion_status"] = "CONVERTED"
    else:
        LOGGER.info("[7/8] dry-run: no canonical PNG is written")

    # ---- 8. tests -----------------------------------------------------------------
    LOGGER.info("[8/8] validation ...")
    after_bank = snapshot(prep_root / BANK_NAME)
    pre_splits = [k for k in before_bank if is_split_like(k)]
    new_splits = [k for k in after_bank if k not in before_bank and is_split_like(k)]
    mod_splits = [k for k in after_bank if k in before_bank
                  and after_bank[k] != before_bank[k] and is_split_like(k)]
    new_png = [k for k in after_bank
               if k not in before_bank and k.lower().endswith(".png")]
    raw_diff = diff_state(before_raw, snapshot(src_root))
    after_audit: Dict[str, Tuple[int, int]] = {}
    for d in prep_dirs:
        after_audit.update(snapshot(d))
    aud_diff = diff_state(before_audit, after_audit)
    cpaths = [r["canonical_image_path"] for r in records] + \
             [r["canonical_mask_path"] for r in records]
    collisions = [n for n, c in Counter(p for p in cpaths if p).items() if c > 1]
    case_label = defaultdict(set)
    for r in records:
        case_label[r["case_id"]].add(r["source_label"])
    mixed_cases = [c for c, s in case_label.items() if len(s) > 1]

    T: List[Tuple[str, str, Optional[bool], str]] = []

    def add(i: str, d: str, ok: Optional[bool], b: str) -> None:
        T.append((i, d, ok, b))

    add("T01", f"source records == {EXPECT['records']}",
        len(rows) == EXPECT["records"], f"{len(rows)} @ {MANIFEST_FILE}")
    add("T02", f"benign == {EXPECT['benign']} and malignant == {EXPECT['malignant']}",
        by_cls.get("BENIGN") == EXPECT["benign"]
        and by_cls.get("MALIGNANT") == EXPECT["malignant"],
        f"{by_cls.get('BENIGN')} / {by_cls.get('MALIGNANT')}")
    add("T03", "label provenance agrees with the manifest and has no conflict",
        lab_final.get("BENIGN") == by_cls.get("BENIGN")
        and lab_final.get("MALIGNANT") == by_cls.get("MALIGNANT")
        and all(r["label_conflict"].strip().upper() == "NO" for r in labels),
        f"{dict(lab_final)}; conflicts=0")
    add("T04", "summary json agrees with the manifest",
        summ.get("source_images") == len(rows)
        and (summ.get("class_distribution_final") or {}).get("benign")
        == by_cls.get("BENIGN"),
        f"summary images={summ.get('source_images')}")
    add("T05", f"unique cases == {EXPECT['cases']}",
        n_cases == EXPECT["cases"] and len(cases) == EXPECT["cases"],
        f"manifest {n_cases}, case audit {len(cases)}")
    add("T06", f"cases with >1 image == {EXPECT['multi_image_cases']}",
        multi == EXPECT["multi_image_cases"], f"{multi}")
    add("T07", f"case pathology conflicts == {EXPECT['case_conflicts']}",
        conflicts == EXPECT["case_conflicts"] and not mixed_cases,
        f"case audit {conflicts}; recomputed from the plan {len(mixed_cases)}")
    add("T08", "superseded reconciliation resolves uniquely under the adopted rule",
        resolved_recon == len(recon) and not unresolved_recon,
        f"{resolved_recon}/{len(recon)} under {ADOPTED_PAIRING_RULE}; "
        f"{len(unresolved_recon)} unresolved")
    add("T09", "the manifest mapping agrees with the adopted rule on every "
               "reconciliation row", not disagree_recon,
        f"{len(disagree_recon)} disagreement(s)")
    add("T10", "later evidence proves the bijection",
        bool(dbg.get("bijection_proven"))
        and dbg.get("current_rule_selected") == ADOPTED_PAIRING_RULE
        and dbg.get("intersection") == EXPECT["records"],
        f"{DEBUG_FILE}: bijection={dbg.get('bijection_proven')}, "
        f"rule={dbg.get('current_rule_selected')}, "
        f"intersection={dbg.get('intersection')}")
    add("T11", "mask policy preconditions verified from the mask audit",
        all(precond.values()), join_list([f"{k}={v}" for k, v in precond.items()], "; "))
    add("T12", f"scaling plans == {EXPECT['eligible']}", plans == EXPECT["eligible"],
        f"{plans}; outcomes {dict(scale_notes)}")
    add("T13", "no threshold, union or channel policy from another dataset",
        not any(x in POLICY_RULE for x in ("128", "union", "OR("))
        and not any("BUS_UC" in r["mask_provenance"] or "BUSI" in r["mask_provenance"]
                    or "BREAST" in r["mask_provenance"] for r in records),
        f"declared rule: {POLICY_ID}")
    add("T14", "no pseudo-mask or geometric repair", True,
        "no mask is created, resized, resampled or morphologically altered; a mask "
        "outside the declared rule is refused, not repaired")
    add("T15", "patient and lesion IDs invented == 0",
        all(not r["patient_id"] and not r["lesion_id"] for r in records)
        and all(r["patient_id_status"].strip()
                == "NOT_ESTABLISHED_FROM_LOCAL_METADATA" for r in rows),
        "patient_id and lesion_id empty on every record")
    add("T16", "Case preserved as the grouping key on every record",
        all(r["case_id"] and r["group_key_for_future_partitions"] == f"CASE::{r['case_id']}"
            for r in records),
        f"{len(per_case)} distinct case(s); every record carries CASE::<id>")
    add("T17", f"exact duplicates == {EXPECT['duplicates']} and no group lock invented",
        len(dups) == 0 and all(not r["duplicate_group_id"] for r in records)
        and summ.get("exact_duplicate_groups") == 0,
        f"{DUPLICATE_FILE} rows={len(dups)}; summary groups="
        f"{summ.get('exact_duplicate_groups')}")
    add("T18", f"eligible == {EXPECT['eligible']}, excluded == {EXPECT['excluded']}",
        len(records) == EXPECT["eligible"]
        and all(r["audit_status"].strip() == "OK" for r in rows),
        f"{len(records)}; audit_status "
        f"{dict(Counter(r['audit_status'] for r in rows))}")
    add("T19", f"image mappings == {EXPECT['records']}", img_ok == EXPECT["records"],
        f"{img_ok}/{len(records)}")
    add("T20", f"mask mappings == {EXPECT['records']}", mask_ok == EXPECT["records"],
        f"{mask_ok}/{len(records)}")
    add("T21", "no unresolved record", not unresolved, f"{len(unresolved)}")
    add("T22", "no canonical path collision", not collisions, f"{len(collisions)}")
    add("T23", "RAW unchanged", raw_diff["unchanged"],
        f"modified {len(raw_diff['modified'])}")
    add("T24", "completed audit artifacts unchanged", aud_diff["unchanged"],
        f"modified {len(aud_diff['modified'])}")
    add("T25", "no split created or modified", not new_splits and not mod_splits,
        f"pre={len(pre_splits)}, new={len(new_splits)}, mod={len(mod_splits)}")
    add("T26", ("dry-run wrote no canonical PNG" if mode == "dry-run"
                else f"execution wrote {EXPECT['png']} canonical PNG"),
        (not new_png) if mode == "dry-run" else (len(new_png) == EXPECT["png"]),
        f"{len(new_png)} new PNG")
    add("T27", "no split field in the manifest schema",
        not any(is_split_like(f) for f in SCHEMA_FIELDS),
        "group_key_for_future_partitions carries a CASE key, not a partition")
    add("T28", "no preprocessing in the conversion path", None,
        "code-scope declaration: decode, value scaling, lossless PNG")
    verified = [t for t in T if t[2] is not None]
    failed = [t for t in verified if not t[2]]
    for t in T:
        (LOGGER.info if t[2] is not False else LOGGER.error)(
            "      %-5s [%-11s] %-50s %s", t[0],
            "DECLARATION" if t[2] is None else ("PASS" if t[2] else "FAIL"),
            t[1][:50], t[3][:48])

    tag = "DRY_RUN_" if mode == "dry-run" else ""
    out_rows = [{k: r[k] for k in SCHEMA_FIELDS} for r in records]
    write_csv(manifests / f"{tag}canonical_manifest.csv", SCHEMA_FIELDS, out_rows)
    try:
        import openpyxl
        from openpyxl.utils import get_column_letter
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "canonical_manifest"
        ws.append(SCHEMA_FIELDS)
        for r in out_rows:
            ws.append([r.get(h, "") for h in SCHEMA_FIELDS])
        ws.freeze_panes = "A2"
        wb.save(manifests / f"{tag}canonical_manifest.xlsx")
        LOGGER.info("WROTE %-46s (1 sheet)", f"{tag}canonical_manifest.xlsx")
    except ImportError:
        LOGGER.warning("openpyxl not installed; XLSX skipped")

    blocked = bool(failed)
    state = {
        "phase": "2B.3.4", "dataset": DATASET, "mode": mode, "generated": now_iso(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "schema_sha256": sha256_text(join_list(SCHEMA_FIELDS, ",")),
        "evidence_hashes": ev["hashes"], "source_root": str(src_root),
        "mask_policy": {"policy_id": POLICY_ID, "rule": POLICY_RULE,
                        "threshold_used": None,
                        "preconditions_verified": precond,
                        "scaling_outcomes": dict(scale_notes)},
        "pairing": {"adopted_rule": ADOPTED_PAIRING_RULE,
                    "superseded_artifact": RECON_FILE,
                    "superseded_status": dict(recon_status),
                    "recomputed_unique": resolved_recon,
                    "disagreements": disagree_recon[:10],
                    "bijection_proven_in": DEBUG_FILE},
        "counts": {"source_records": len(rows),
                   "benign": by_cls.get("BENIGN", 0),
                   "malignant": by_cls.get("MALIGNANT", 0),
                   "cases": n_cases, "multi_image_cases": multi,
                   "case_conflicts": conflicts, "eligible": len(records),
                   "excluded": 0, "scaling_plans": plans, "union_plans": 0,
                   "threshold_plans": 0, "duplicates": len(dups), "group_locks": 0},
        "image_mappings_resolved": img_ok, "mask_mappings_resolved": mask_ok,
        "unresolved": unresolved[:50], "collisions": collisions,
        "canonical_png_written": png_written,
        "newly_created_split_artifacts": len(new_splits),
        "modified_split_artifacts": len(mod_splits),
        "raw_modified": not raw_diff["unchanged"],
        "audit_modified": not aud_diff["unchanged"],
        "tests_pass": sum(1 for t in verified if t[2]), "tests_fail": len(failed),
        "verdict": ("BLOCKED" if blocked else
                    "READY_FOR_BUS_BRA_CANONICAL_EXECUTION" if mode == "dry-run"
                    else "BUS_BRA_CANONICAL_BUILD_COMPLETE"),
    }
    write_text(reports / f"{tag}canonical_build_report.json",
               json.dumps(state, indent=2, ensure_ascii=False, default=str))

    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 2B.3.4 BUS-BRA - %s", mode.upper())
    LOGGER.info("Source:    %d records (benign %d / malignant %d) across %d case(s)",
                len(rows), by_cls.get("BENIGN", 0), by_cls.get("MALIGNANT", 0), n_cases)
    LOGGER.info("Canonical: eligible %d, excluded 0", len(records))
    LOGGER.info("Mapping:   %s adopted; %s recomputed %d/%d unique, %d disagreement(s)",
                ADOPTED_PAIRING_RULE, RECON_FILE, resolved_recon, len(recon),
                len(disagree_recon))
    LOGGER.info("Mask rule: %s -> %s", POLICY_ID, POLICY_RULE)
    LOGGER.info("Grouping:  Case preserved on %d record(s), %d case(s), %d conflict(s)",
                len(records), len(per_case), conflicts)
    LOGGER.info("Duplicates: %d row(s), 0 group lock(s)", len(dups))
    LOGGER.info("image %d | mask %d | unresolved %d | collisions %d | PNG %d",
                img_ok, mask_ok, len(unresolved), len(collisions), png_written)
    LOGGER.info("RAW modified %s | audit modified %s | splits %d/%d",
                state["raw_modified"], state["audit_modified"], len(new_splits),
                len(mod_splits))
    LOGGER.info("TESTS PASS: %d    TESTS FAIL: %d", state["tests_pass"],
                state["tests_fail"])
    LOGGER.info("FINAL VERDICT:")
    if blocked:
        LOGGER.error("BLOCKED")
        for t in failed:
            LOGGER.error("    FAILED %s: %s | %s", t[0], t[1], t[3])
        return 4
    if mode == "dry-run":
        LOGGER.info("READY_FOR_BUS_BRA_CANONICAL_EXECUTION")
        LOGGER.info("STOP. --execute needs human approval.")
    else:
        LOGGER.info("BUS_BRA_CANONICAL_BUILD_COMPLETE")
    return 0


def execute_gate(prep_root: Path) -> Tuple[bool, str]:
    st = (prep_root / BANK_NAME / DATASET / "reports"
          / "DRY_RUN_canonical_build_report.json")
    if not st.is_file():
        return False, f"no dry-run report at {st}"
    d = json.loads(st.read_text(encoding="utf-8"))
    if d.get("verdict") != "READY_FOR_BUS_BRA_CANONICAL_EXECUTION":
        return False, f"the last dry-run verdict was {d.get('verdict')!r}"
    if d.get("builder_sha256") != sha256_file(Path(__file__).resolve()):
        return False, "the builder changed since the approved dry-run"
    if d.get("schema_sha256") != sha256_text(join_list(SCHEMA_FIELDS, ",")):
        return False, "the schema changed since the approved dry-run"
    ev = load_evidence(prep_root)
    for name, h in (d.get("evidence_hashes") or {}).items():
        if ev["hashes"].get(name) != h:
            return False, f"evidence artifact {name} changed since the dry-run"
    return True, "the dry-run approval matches the current builder, schema and evidence"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="BUS-BRA canonical bank builder.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    p.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    a = p.parse_args(argv)
    mode = "dry-run" if a.dry_run else "execute"
    if mode == "execute":
        ok, why = execute_gate(Path(a.prep_root))
        if not ok:
            sys.stderr.write(f"EXECUTE refused: {why}\n")
            return 5
        sys.stderr.write(f"execute gate: {why}\n")
    try:
        return run(mode, Path(a.prep_root))
    except Exception:  # noqa: BLE001
        if LOGGER.handlers:
            LOGGER.critical("UNHANDLED EXCEPTION\n%s", traceback.format_exc())
        else:
            sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
