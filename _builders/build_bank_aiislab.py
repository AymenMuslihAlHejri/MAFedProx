#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bank_bluis.py
===================

PHASE 2B.3.11 - AIISLAB CANONICAL DATASET BANK BUILDER

Eleventh dataset. It carries TWO mask policies because the source ships two kinds of
mask file, and each is handled by the rule its own evidence supports.

    MASK A  AIISLAB_CANONICAL_MASK_NATIVE_BINARY_V1        148 masks
        RGB with replicated channels holding only {0,255}. The replication is verified
        on that file's own pixels, one channel is taken, and the values are preserved
        exactly. Nothing is thresholded, because there is nothing to threshold.

    MASK B  AIISLAB_JPEG_MASK_BINARIZATION_V1               18 masks
        source_gray >= 128 -> 255, source_gray < 128 -> 0.

        These 18 are the double-extension files the audit adjudicated as
        JPEG_REENCODED_MASK_WITH_INTERMEDIATE_PIXEL_VALUES. The diagnostic pass
        measured them and the threshold rests on that measurement, not on convention:

            the observed values form two separated bands, 1-31 and 192-254, with ZERO
            pixels in 32-191 on every one of the 18;
            thresholds 64, 128 and 192 produced pixel-identical binary masks on all 18,
            Dice = 1.0, and identical connected-component counts;
            threshold 1 produced up to 365 spurious components and threshold 254 moved
            the boundary by up to 1.6 px, so both were rejected by measurement.

        128 is therefore a fixed midpoint inside an empirically empty intensity gap.
        It was not selected using model performance, and the builder VERIFIES the gap
        is empty on every file before converting: a mask with pixels in 32-191 is
        refused, not thresholded.

        A caveat that belongs in the manuscript: these files were re-encoded through
        JPEG and their original history is not available locally. The recovered mask
        matches what the file contains; it is not claimed to be proven identical to
        what the annotator drew before that re-encoding.

    IMAGE  AIISLAB_CANONICAL_IMAGE_LOSSLESS_V1
        the decoded image is written unchanged to a lossless PNG at native dimensions.

THE 18 ARE ELIGIBLE
    They were recovered by exact string construction, mask_<stem><prior_ext><final_ext>,
    18 of 18 with zero ambiguity and zero orphan masks left over. They are NOT
    joint-task-ineligible and do NOT enter the global exclusion ledger.

IDENTITY
    TOKEN_1 is the patient identifier on the authority of the dataset's first author
    (AIISLAB_AUTHOR_CONFIRMED_DATASET_SEMANTICS.json), which the Phase 1A.1 audit could
    not establish from the files alone. The grouping key is PATIENT::<TOKEN_1> and the
    provenance recorded on every record is AUTHOR_CONFIRMED_DATASET_SEMANTICS.

    Patient 113646 carries Benign on the left breast and Malignant on the right. The
    class difference aligns with laterality, so it is not called a labelling error and
    no label is changed; the four images stay in ONE patient group, which is what
    prevents the leak. Identifier 42574 is kept verbatim, not zero-padded.

Usage:
    python build_bank_aiislab.py --dry-run
    python build_bank_aiislab.py --execute
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

DATASET = "AIISLAB"
DEFAULT_PREP_ROOT = (
    r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging"
    r"\Revision 1\dataset preparing"
)
BANK_NAME = "CANONICAL_DATASET_BANK"
ARCHIVE_DIR_NAMES = ("audit_old",)

PAIRING_FILE = "15_aiislab_phase1a_pairing_manifest.csv"
DOUBLE_FILE = "23_aiislab_phase1a1_double_extension_pairs.csv"
ENCODING_FILE = "24_aiislab_phase1a1_mask_encoding_adjudication.csv"
CROSS_FILE = "25_aiislab_phase1a1_cross_class_group.csv"
SPECIAL_FILE = "26_aiislab_phase1a1_special_identifier_42574.csv"
AUTHOR_FILE = "AIISLAB_AUTHOR_CONFIRMED_DATASET_SEMANTICS.json"
SUMMARY_FILE = "27_aiislab_phase1a1_corrected_units.json"
REPORT_FILE = "28_AIISLAB_PHASE1A1_CORRECTIVE_REPORT.md"
ROOT_LOCATORS = ["raw_root", "source_path", "dataset_root", "source_root"]

NATIVE_POLICY_ID = "AIISLAB_CANONICAL_MASK_NATIVE_BINARY_V1"
NATIVE_POLICY_RULE = ("the source mask is RGB with replicated channels holding only "
                      "{0,255}; the replication is verified on that file's own pixels, "
                      "one channel is taken, and its values are preserved exactly. "
                      "Nothing is thresholded, because there is nothing to threshold")
JPEG_POLICY_ID = "AIISLAB_JPEG_MASK_BINARIZATION_V1"
JPEG_POLICY_RULE = (
    "source_gray >= 128 -> 255, source_gray < 128 -> 0. The 18 affected masks were "
    "re-encoded through JPEG, which leaves sparse edge artefacts. Their observed "
    "values form two separated bands, low (1-31) and high (192-254), with ZERO pixels "
    "in 32-191 on every one of the 18. Thresholds 64, 128 and 192 therefore produced "
    "pixel-identical binary masks on all 18 (Dice = 1.0), so 128 is a fixed midpoint "
    "inside an empirically empty intensity gap, not a threshold selected using model "
    "performance. No morphology, no geometric transformation, no interpolation")
IMAGE_POLICY_ID = "AIISLAB_CANONICAL_IMAGE_LOSSLESS_V1"
IMAGE_POLICY_RULE = ("decoded BMP written unchanged to lossless PNG at native "
                     "dimensions, channels preserved; no resize, crop, ROI, "
                     "normalisation, CLAHE, denoising or augmentation")

EXPECT = {"images": 166, "masks": 166, "pairs": 166, "benign": 66, "malignant": 100,
          "native_binary": 148, "jpeg_reencoded": 18, "patients": 79,
          "excluded": 0, "png": 332, "source_files": 332,
          "empty_gap_lo": 32, "empty_gap_hi": 191}
CROSS_TOKEN = "113646"
SPECIAL_TOKEN = "42574"
CANONICAL_LABELS = {"BENIGN": 0, "MALIGNANT": 1}

SCHEMA_FIELDS = [
    "dataset", "canonical_record_id", "sample_id", "image_stem",
    "TOKEN_1", "TOKEN_2_laterality", "TOKEN_3_view",
    "source_image_path", "source_mask_path",
    "source_label", "canonical_binary_label", "class_folder",
    "canonical_image_path", "canonical_mask_path",
    "patient_id", "patient_id_provenance", "group_key_for_future_partitions",
    "patient_is_cross_class", "case_id", "lesion_id",
    "same_source_image_group",
    "classification_available", "segmentation_available",
    "image_conversion_status", "mask_conversion_status",
    "source_image_height", "source_image_width", "source_image_mode",
    "source_mask_height", "source_mask_width", "source_mask_mode",
    "mask_dimension_verified_here",
    "pairing_status_source", "pairing_rule_source",
    "double_extension_recovered", "embedded_prior_extension", "final_extension",
    "canonical_image_height", "canonical_image_width",
    "canonical_mask_height", "canonical_mask_width",
    "source_image_format", "source_mask_format",
    "source_mask_encoding", "mask_channels_replicated_verified_here",
    "mask_source_unique_value_count", "mask_pixels_in_empty_gap_32_191",
    "mask_rule_applied", "canonical_image_encoding", "canonical_mask_encoding",
    "mask_foreground_pixels", "mask_foreground_fraction",
    "image_policy_id", "image_policy_rule", "mask_policy_id", "mask_policy_rule",
    "identity_provenance", "label_provenance", "mask_provenance",
    "exclusion_status", "exclusion_reason", "notes",
]
SPLIT_TOKENS = {"split", "splits", "fold", "folds", "kfold", "train", "training",
                "trainval", "val", "validation", "test", "testset", "testing",
                "holdout", "partition"}

SEP = " | "
LOGGER = logging.getLogger("bank_aiislab")


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
    LOGGER.info("WROTE %-44s (%d data rows)", path.name, len(rows))


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


def canonical_stem(cls: str, sample_key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_key).strip("_")
    return f"{safe or 'sample'}__{sha256_text(f'{cls}/{sample_key}')[:10]}"


def reduce_channels(arr, np) -> Tuple[Any, str, str]:
    """One plane, with the replication VERIFIED on this file's own pixels."""
    if arr.ndim == 2:
        return arr, "ALREADY_SINGLE_CHANNEL", "NOT_APPLICABLE"
    if arr.ndim == 3 and arr.shape[2] >= 3:
        same = bool(np.array_equal(arr[..., 0], arr[..., 1])
                    and np.array_equal(arr[..., 1], arr[..., 2]))
        if not same:
            return None, "REFUSED_CHANNELS_NOT_REPLICATED", "NO"
        return arr[..., 0], "THREE_IDENTICAL_CHANNELS_REDUCED_TO_ONE", "YES"
    return None, f"REFUSED_UNEXPECTED_SHAPE_{arr.shape}", ""


def native_binary(plane, np) -> Tuple[Any, str]:
    """MASK A: the values must already be {0,255}; they are preserved exactly."""
    vals = set(np.unique(plane).tolist())
    if vals == {0, 255}:
        return plane.astype(np.uint8), "NATIVE_BINARY_PRESERVED_EXACTLY"
    if vals == {0}:
        return None, "EMPTY_MASK"
    return None, f"REFUSED_NOT_NATIVE_BINARY_{sorted(vals)[:6]}"


def jpeg_binarise(plane, np) -> Tuple[Any, str, int]:
    """MASK B: >=128 -> 255. The empty gap the rule rests on is VERIFIED on this
    file before the rule is applied; a mask with pixels in 32-191 is refused, because
    the measurement that justifies 128 would not hold for it."""
    gap = int(((plane >= EXPECT["empty_gap_lo"])
               & (plane <= EXPECT["empty_gap_hi"])).sum())
    if gap:
        return None, f"REFUSED_{gap}_PIXELS_IN_THE_32_191_GAP", gap
    out = (plane >= 128).astype(np.uint8) * 255
    if not out.any():
        return None, "EMPTY_MASK_AFTER_BINARISATION", gap
    return out, "JPEG_BINARISED_GE_128", gap


def load_evidence(prep_root: Path) -> dict:
    ev: Dict[str, Any] = {"paths": {}, "status": {}, "hashes": {}, "problems": []}
    dirs = [d for d in prep_root.iterdir()
            if d.is_dir() and "aiislab" in d.name.lower().replace(" ", "")]
    files = [p for d in dirs for p in d.rglob("*") if p.is_file() and not archived(p)]
    for key, fname, required in (("pairing", PAIRING_FILE, True),
                                 ("double", DOUBLE_FILE, True),
                                 ("encoding", ENCODING_FILE, True),
                                 ("cross", CROSS_FILE, False),
                                 ("special", SPECIAL_FILE, False),
                                 ("summary", SUMMARY_FILE, True),
                                 ("author", AUTHOR_FILE, True),
                                 ("report", REPORT_FILE, False)):
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
            # this dataset's summary carries no raw_root key, which is a fact about
            # the artifact and not an error; --raw-root supplies it and the root that
            # was actually used is printed and recorded
            ev["dataset_root_locator"] = (f"absent from {SUMMARY_FILE}; must be given "
                                          "with --raw-root")
    return ev


def run(mode: str, prep_root: Path, raw_cli: Optional[Path] = None) -> int:
    bank_ds = prep_root / BANK_NAME / DATASET
    reports, manifests = bank_ds / "reports", bank_ds / "manifests"
    setup_logging(reports, f"aiislab_{mode}.log")
    LOGGER.info("=" * 78)
    LOGGER.info("PHASE 2B.3.11 AIISLAB CANONICAL BANK BUILDER  [%s]", mode.upper())
    LOGGER.info("=" * 78)
    LOGGER.info("python : %s", platform.python_version())
    LOGGER.info("bank   : %s", bank_ds)

    try:
        import numpy as np
        from PIL import Image
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

    LOGGER.info("[1/6] locked evidence bindings ...")
    ev = load_evidence(prep_root)
    for k, st in ev["status"].items():
        art = ev["paths"].get(k)
        LOGGER.info("      %-10s %-12s %s", k, st, art.name if art else "")
    if ev["problems"]:
        for p in ev["problems"]:
            LOGGER.error("      %s", p)
        LOGGER.error("FINAL VERDICT:\nBLOCKED")
        return 4
    src_root = raw_cli or ev["dataset_root"]
    if src_root is None:
        LOGGER.error("      BLOCKED: %s carries no source root and --raw-root was "
                     "not given. Nothing was guessed.", SUMMARY_FILE)
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4
    if not src_root.is_dir():
        LOGGER.error("      BLOCKED: the source root does not exist: %s", src_root)
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4
    before_raw = snapshot(src_root)
    summ = json.loads(ev["paths"]["summary"].read_text(encoding="utf-8"))
    LOGGER.info("      RAW    = %s  [READ-ONLY]  (from %s)", src_root,
                "--raw-root" if raw_cli else ev["dataset_root_locator"])
    LOGGER.info("      root evidence = %s", ev["dataset_root_locator"])
    LOGGER.info("      image policy = %s", IMAGE_POLICY_ID)

    pairing = read_rows(ev["paths"]["pairing"])
    doubles = {r["image_stem"].strip(): r
               for r in read_rows(ev["paths"]["double"])}
    enc = {r["image_stem"].strip(): r for r in read_rows(ev["paths"]["encoding"])}
    author = json.loads(ev["paths"]["author"].read_text(encoding="utf-8"))
    images = [r for r in pairing
              if r["pairing_status"].strip() in ("EXACTLY_ONE_MASK", "NO_MASK")]
    orphans = [r for r in pairing
               if r["pairing_status"].strip() == "ORPHAN_MASK_NO_MATCHING_IMAGE"]
    stems = [r["image_stem"].strip() for r in images]
    dupe = [k for k, n in Counter(stems).items() if n > 1]
    if dupe:
        LOGGER.error("      image_stem not unique: %s", dupe[:5])
        LOGGER.error("FINAL VERDICT:")
        LOGGER.error("BLOCKED")
        return 4

    LOGGER.info("[2/6] source accounting ...")
    by_cls = Counter(r["source_class"].strip() for r in images)
    st = Counter(r["pairing_status"].strip() for r in images)
    enc_src = Counter(r["mask_encoding"].strip() or "NONE_IN_PHASE_1A"
                      for r in images)
    dim_src = Counter(r["dimension_agreement"].strip() for r in images)
    cu = summ.get("corrected_units", {})
    LOGGER.info("      pairing manifest %d row(s): %d image(s) + %d orphan mask(s)",
                len(pairing), len(images), len(orphans))
    LOGGER.info("      class: %s | pairing status: %s", dict(by_cls), dict(st))
    LOGGER.info("      corrected units: images=%s masks=%s paired=%s without_mask=%s "
                "orphans=%s resolvable=%s", cu.get("mammogram_images"),
                cu.get("mask_files"), cu.get("paired_image_mask_records_CORRECTED"),
                cu.get("images_without_mask_CORRECTED"), cu.get("orphan_masks"),
                cu.get("canonically_resolvable_pairs"))
    LOGGER.info("      Phase 1A mask encoding: %s", dict(enc_src))
    LOGGER.info("      Phase 1A dimension agreement: %s", dict(dim_src))
    LOGGER.info("      double-extension recoveries: %d (%s)", len(doubles),
                summ.get("double_extension", {}).get("status"))
    LOGGER.info("      encoding adjudication: %s -> %s",
                dict(Counter(r["encoding_from_phase1a"] for r in enc.values())),
                next(iter({r["adjudication"] for r in enc.values()}), ""))
    LOGGER.info("      identity: audit verdict=%s | author says TOKEN1=%s, "
                "grouping unit=%s", summ.get("identity_verdict"),
                (author.get("filename_semantics") or {}).get("TOKEN1"),
                author.get("author_confirmed_grouping_unit"))
    LOGGER.info("      mask policy A = %s  (%d mask(s))", NATIVE_POLICY_ID,
                EXPECT["native_binary"])
    LOGGER.info("      mask policy B = %s  (%d mask(s))", JPEG_POLICY_ID,
                EXPECT["jpeg_reencoded"])

    LOGGER.info("[3/6] planning canonical records ...")
    records: List[dict] = []
    tok_cls: Dict[str, set] = defaultdict(set)
    for r in images:
        tok_cls[r["TOKEN_1"].strip()].add(r["source_class"].strip())
    cross_tokens = sorted(t for t, c in tok_cls.items() if len(c) > 1)
    for r in images:
        stem = r["image_stem"].strip()
        cls = r["source_class"].strip()
        folder = "Benign" if cls == "Benign" else (
            "Malignant" if cls == "Malignant" else "")
        tok = r["TOKEN_1"].strip()
        d = doubles.get(stem, {})
        e = enc.get(stem, {})
        recovered = bool(d)
        # the mask path: from the pairing manifest, or from the adjudicated recovery
        mask_rel = (r["mask_relative_path"].strip()
                    or d.get("recovered_mask_relative_path", "").strip())
        sid = f"{stem}__{sha256_text(cls + '/' + stem)[:8]}"
        rule = JPEG_POLICY_ID if recovered else NATIVE_POLICY_ID
        records.append({
            "dataset": DATASET, "canonical_record_id": f"{DATASET}__{sid}",
            "sample_id": sid, "image_stem": stem,
            "TOKEN_1": tok, "TOKEN_2_laterality": r["TOKEN_2"].strip(),
            "TOKEN_3_view": r["TOKEN_3"].strip(),
            "source_image_path": r["image_relative_path"].strip(),
            "source_mask_path": mask_rel,
            "source_label": cls.upper(),
            "canonical_binary_label": 0 if cls == "Benign" else 1,
            "class_folder": folder,
            "canonical_image_path": f"{folder}/Images/{sid}.png",
            "canonical_mask_path": f"{folder}/Masks/{sid}_mask.png",
            "patient_id": tok,
            "patient_id_provenance": (
                "AUTHOR_CONFIRMED_DATASET_SEMANTICS: the dataset's first author "
                "confirms TOKEN1 = PATIENT_ID, TOKEN2 = LATERALITY, TOKEN3 = VIEW. "
                "The Phase 1A.1 audit could not establish this from the files alone "
                f"({summ.get('identity_verdict')}), so the author confirmation is "
                "what carries it"),
            "group_key_for_future_partitions": f"PATIENT::{tok}",
            "patient_is_cross_class": "YES" if tok in cross_tokens else "NO",
            "case_id": "", "lesion_id": "",
            "same_source_image_group": f"{DATASET}__{sid}",
            "classification_available": "TRUE", "segmentation_available": "TRUE",
            "image_conversion_status": "PLANNED", "mask_conversion_status": "PLANNED",
            "source_image_height": r["image_height"].strip(),
            "source_image_width": r["image_width"].strip(),
            "source_image_mode": "",
            "source_mask_height": r["mask_height"].strip(),
            "source_mask_width": r["mask_width"].strip(),
            "source_mask_mode": r["mask_mode"].strip(),
            "mask_dimension_verified_here": "",
            "pairing_status_source": r["pairing_status"].strip(),
            "pairing_rule_source": r["pairing_rule"].strip(),
            "double_extension_recovered": "YES" if recovered else "NO",
            "embedded_prior_extension": d.get("embedded_prior_extension", ""),
            "final_extension": d.get("final_extension", ""),
            "canonical_image_height": "", "canonical_image_width": "",
            "canonical_mask_height": "", "canonical_mask_width": "",
            "source_image_format": Path(r["image_relative_path"]).suffix.lstrip(
                ".").upper(),
            "source_mask_format": Path(mask_rel).suffix.lstrip(".").upper(),
            "source_mask_encoding": (e.get("encoding_from_phase1a", "")
                                     or r["mask_encoding"].strip()),
            "mask_channels_replicated_verified_here": "",
            "mask_source_unique_value_count": e.get("unique_value_count", "")
                                              or r["mask_unique_value_count"].strip(),
            "mask_pixels_in_empty_gap_32_191": "",
            "mask_rule_applied": "",
            "canonical_image_encoding": "LOSSLESS_PNG_NATIVE",
            "canonical_mask_encoding": "SINGLE_CHANNEL_BINARY_0_255",
            "mask_foreground_pixels": "", "mask_foreground_fraction": "",
            "image_policy_id": IMAGE_POLICY_ID, "image_policy_rule": IMAGE_POLICY_RULE,
            "mask_policy_id": rule,
            "mask_policy_rule": (JPEG_POLICY_RULE if recovered
                                 else NATIVE_POLICY_RULE),
            "identity_provenance": f"{AUTHOR_FILE} :: json:filename_semantics.TOKEN1",
            "label_provenance": (f"{PAIRING_FILE} :: csv:[source_class]; the source "
                                 "class directory, unchanged"),
            "mask_provenance": (f"DOUBLE_EXTENSION_RECOVERED_THEN[{rule}]" if recovered
                                else f"SOURCE_NATIVE_BINARY_BY[{rule}]"),
            "exclusion_status": "INCLUDED_CANONICAL_JOINT", "exclusion_reason": "",
            "notes": (f"phase1a_notes={r['notes'].strip()[:80]}"
                      + (f"; recovered by {d.get('expected_construction', '')}"
                         if recovered else "")),
        })
    by_el = Counter(r["source_label"] for r in records)
    pats = {r["patient_id"] for r in records}
    LOGGER.info("      planned %d: BENIGN=%d MALIGNANT=%d over %d patient(s)",
                len(records), by_el.get("BENIGN", 0), by_el.get("MALIGNANT", 0),
                len(pats))
    LOGGER.info("      mask rule split: native=%d jpeg=%d",
                sum(1 for r in records if r["mask_policy_id"] == NATIVE_POLICY_ID),
                sum(1 for r in records if r["mask_policy_id"] == JPEG_POLICY_ID))
    LOGGER.info("      cross-class patient(s): %s -> %s", cross_tokens or "none",
                {t: sorted(tok_cls[t]) for t in cross_tokens})
    LOGGER.info("      identifier %s: %d image(s), kept verbatim", SPECIAL_TOKEN,
                sum(1 for r in records if r["TOKEN_1"] == SPECIAL_TOKEN))

    LOGGER.info("[4/6] resolving and pre-checking source pairs ...")
    img_ok = mask_ok = plans = 0
    unresolved: List[str] = []
    rule_c: Counter = Counter()
    repl_c: Counter = Counter()
    gap_bad: List[str] = []
    dim_bad: List[str] = []
    for i, r in enumerate(records, 1):
        sp = src_root / r["source_image_path"] if r["source_image_path"] else None
        mp = src_root / r["source_mask_path"] if r["source_mask_path"] else None
        if sp is None or not sp.is_file():
            unresolved.append(f"{r['sample_id']}: image missing "
                              f"({r['source_image_path']})")
            r["image_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            with Image.open(sp) as im:
                r["source_image_mode"] = im.mode
                r["canonical_image_width"] = im.width
                r["canonical_image_height"] = im.height
                ih, iw = im.height, im.width
            img_ok += 1
        except Exception:  # noqa: BLE001
            unresolved.append(f"{r['sample_id']}: image undecodable")
            r["image_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        if mp is None or not mp.is_file():
            unresolved.append(f"{r['sample_id']}: mask missing "
                              f"({r['source_mask_path']})")
            r["mask_conversion_status"] = "UNRESOLVED_SOURCE"
            continue
        try:
            with Image.open(mp) as mi:
                ma = np.array(mi)
                r["source_mask_mode"] = mi.mode
                r["canonical_mask_width"] = mi.width
                r["canonical_mask_height"] = mi.height
                mh, mw = mi.height, mi.width
        except Exception:  # noqa: BLE001
            unresolved.append(f"{r['sample_id']}: mask undecodable")
            r["mask_conversion_status"] = "UNDECODABLE_SOURCE"
            continue
        mask_ok += 1
        same = (mh, mw) == (ih, iw)
        r["mask_dimension_verified_here"] = "MATCH" if same else "MISMATCH"
        if not same:
            dim_bad.append(f"{r['sample_id']}: image {iw}x{ih} vs mask {mw}x{mh}")
            r["mask_conversion_status"] = "DIMENSION_MISMATCH"
            continue
        plane, note, repl = reduce_channels(ma, np)
        r["mask_channels_replicated_verified_here"] = repl
        repl_c[repl] += 1
        if plane is None:
            unresolved.append(f"{r['sample_id']}: {note}")
            r["mask_conversion_status"] = note
            continue
        if r["mask_policy_id"] == JPEG_POLICY_ID:
            out, rnote, gap = jpeg_binarise(plane, np)
            r["mask_pixels_in_empty_gap_32_191"] = gap
            if gap:
                gap_bad.append(f"{r['sample_id']}: {gap} pixel(s) in 32-191")
        else:
            out, rnote = native_binary(plane, np)
            r["mask_pixels_in_empty_gap_32_191"] = int(
                ((plane >= EXPECT["empty_gap_lo"])
                 & (plane <= EXPECT["empty_gap_hi"])).sum())
        rule_c[rnote] += 1
        r["mask_rule_applied"] = rnote
        if out is None:
            unresolved.append(f"{r['sample_id']}: mask refused ({rnote})")
            r["mask_conversion_status"] = f"REFUSED_{rnote}"
            continue
        fg = int((out == 255).sum())
        if fg == 0:
            unresolved.append(f"{r['sample_id']}: empty mask")
            r["mask_conversion_status"] = "EMPTY_MASK"
            continue
        r["mask_foreground_pixels"] = fg
        r["mask_foreground_fraction"] = round(fg / float(mh * mw), 8)
        plans += 1
        if i % 40 == 0:
            LOGGER.info("      %d/%d", i, len(records))
    LOGGER.info("      image=%d mask=%d conversion plans=%d unresolved=%d",
                img_ok, mask_ok, plans, len(unresolved))
    LOGGER.info("      channel replication verified: %s", dict(repl_c))
    LOGGER.info("      mask rules applied: %s", dict(rule_c))
    LOGGER.info("      pixels found in the 32-191 gap: %d record(s)", len(gap_bad))
    for x in (unresolved + dim_bad + gap_bad)[:10]:
        LOGGER.error("         %s", x)

    png_written = 0
    if mode == "execute":
        LOGGER.info("[5/6] executing canonical conversion ...")
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
                ma = np.array(mi)
            plane, _n, _rp = reduce_channels(ma, np)
            if plane is None:
                r["mask_conversion_status"] = "REFUSED_AT_EXECUTE"
                continue
            if r["mask_policy_id"] == JPEG_POLICY_ID:
                out, _rn, _g = jpeg_binarise(plane, np)
            else:
                out, _rn = native_binary(plane, np)
            if out is None:
                r["mask_conversion_status"] = "REFUSED_AT_EXECUTE"
                continue
            mdst = bank_ds / r["canonical_mask_path"]
            mdst.parent.mkdir(parents=True, exist_ok=True)
            if not mdst.exists():
                Image.fromarray(out.astype(np.uint8), mode="L").save(
                    mdst, format="PNG")
                png_written += 1
            with Image.open(mdst) as chk:
                back = np.asarray(chk.convert("L"))
            r["mask_conversion_status"] = ("CONVERTED_VERIFIED_IDENTICAL"
                                           if np.array_equal(back, out)
                                           else "VERIFICATION_FAILED_NOT_IDENTICAL")
    else:
        LOGGER.info("[5/6] dry-run: no canonical PNG is written")

    LOGGER.info("[6/6] validation ...")
    after_bank = snapshot(prep_root / BANK_NAME)
    new_splits = [k for k in after_bank if k not in before_bank and is_split_like(k)]
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
    native = [r for r in records if r["mask_policy_id"] == NATIVE_POLICY_ID]
    jpegs = [r for r in records if r["mask_policy_id"] == JPEG_POLICY_ID]

    T: List[Tuple[str, str, Optional[bool], str]] = []

    def add(i: str, d: str, ok: Optional[bool], b: str) -> None:
        T.append((i, d, ok, b))

    add("T01", f"source images == {EXPECT['images']}",
        len(images) == EXPECT["images"]
        and cu.get("mammogram_images") == EXPECT["images"],
        f"{len(images)} in {PAIRING_FILE}; corrected units say "
        f"{cu.get('mammogram_images')}")
    add("T02", f"source masks == {EXPECT['masks']} and every one is accounted for",
        cu.get("mask_files") == EXPECT["masks"]
        and len(orphans) == EXPECT["jpeg_reencoded"]
        and cu.get("orphan_masks") == EXPECT["jpeg_reencoded"],
        f"{cu.get('mask_files')} mask file(s); {len(orphans)} orphan(s) in Phase 1A, "
        f"all {len(doubles)} recovered in 1A.1")
    add("T03", f"resolved pairs == {EXPECT['pairs']}",
        len(records) == EXPECT["pairs"]
        and cu.get("canonically_resolvable_pairs") == EXPECT["pairs"]
        and all(r["source_mask_path"] for r in records),
        f"{len(records)} planned; {st.get('EXACTLY_ONE_MASK', 0)} direct + "
        f"{len(doubles)} recovered")
    add("T04", f"BENIGN == {EXPECT['benign']} and MALIGNANT == "
               f"{EXPECT['malignant']}",
        by_el.get("BENIGN") == EXPECT["benign"]
        and by_el.get("MALIGNANT") == EXPECT["malignant"],
        f"{by_el.get('BENIGN')} / {by_el.get('MALIGNANT')}")
    add("T05", f"{EXPECT['native_binary']} mask(s) take the native-binary rule and "
               f"{EXPECT['jpeg_reencoded']} take the JPEG rule",
        len(native) == EXPECT["native_binary"]
        and len(jpegs) == EXPECT["jpeg_reencoded"],
        f"native {len(native)}, jpeg {len(jpegs)}")
    add("T06", "the JPEG rule is applied ONLY to the adjudicated double-extension "
               "files",
        {r["image_stem"] for r in jpegs} == set(doubles)
        and all(r["double_extension_recovered"] == "YES" for r in jpegs)
        and all(r["double_extension_recovered"] == "NO" for r in native),
        f"{len(jpegs)} JPEG record(s), all from {DOUBLE_FILE}")
    add("T07", "the native masks were preserved exactly, not thresholded",
        rule_c.get("NATIVE_BINARY_PRESERVED_EXACTLY", 0) == len(native),
        f"{rule_c.get('NATIVE_BINARY_PRESERVED_EXACTLY', 0)}/{len(native)}; "
        f"rules {dict(rule_c)}")
    add("T08", "the 32-191 gap is EMPTY on every JPEG mask, which is what makes 128 a "
               "midpoint rather than a choice",
        not gap_bad
        and all(str(r["mask_pixels_in_empty_gap_32_191"]) == "0" for r in jpegs
                if str(r["mask_pixels_in_empty_gap_32_191"]).isdigit()),
        f"{len(gap_bad)} record(s) with pixels in 32-191")
    add("T09", "channel replication verified per file before any reduction",
        "NO" not in repl_c and sum(repl_c.values()) == len(records),
        f"{dict(repl_c)}")
    add("T10", f"conversion plans == {EXPECT['pairs']}", plans == EXPECT["pairs"],
        f"{plans}; rules {dict(rule_c)}")
    add("T11", "no morphology, no pseudo-mask, no geometric transform, no "
               "performance-based threshold", True,
        "code-scope: decode, verified channel reduction, exact preservation or a "
        "fixed >=128 midpoint inside a verified-empty gap")
    add("T12", "no policy from another dataset",
        not any(pid in r["mask_provenance"] for r in records
                for pid in ("BUSI_CANONICAL_MULTIMASK_UNION_V1",
                            "BUS_UC_CANONICAL_MASK_BINARIZATION_V1",
                            "BUS_UCLM_CANONICAL_MASK_COLOUR_TO_BINARY_V1",
                            "DMID_CANONICAL_MASK_WHITE_FOREGROUND_V1",
                            "CBIS_CANONICAL_MASK_BINARY_V1")),
        f"declared rules: {NATIVE_POLICY_ID} and {JPEG_POLICY_ID}")
    add("T13", f"image mappings == {EXPECT['images']}", img_ok == EXPECT["images"],
        f"{img_ok}/{len(records)}")
    add("T14", f"mask mappings == {EXPECT['masks']}", mask_ok == EXPECT["masks"],
        f"{mask_ok}/{len(records)}")
    add("T15", "mask and image dimensions agree, verified here from the pixels",
        not dim_bad and all(r["mask_dimension_verified_here"] == "MATCH"
                            for r in records),
        f"{len(dim_bad)} mismatch(es); Phase 1A said {dict(dim_src)}")
    add("T16", f"unique patients == {EXPECT['patients']}",
        len(pats) == EXPECT["patients"]
        and cu.get("candidate_grouping_ids_TOKEN_1") == EXPECT["patients"],
        f"{len(pats)} distinct TOKEN_1; corrected units say "
        f"{cu.get('candidate_grouping_ids_TOKEN_1')}")
    add("T17", "patient_id comes from the author confirmation, and every record "
               "carries PATIENT::<TOKEN_1>",
        author.get("author_confirmed_grouping_unit") == "PATIENT_ID"
        and all(r["group_key_for_future_partitions"] == f"PATIENT::{r['TOKEN_1']}"
                and "AUTHOR_CONFIRMED" in r["patient_id_provenance"]
                for r in records),
        f"{author.get('confirmation_source')}; "
        f"{author.get('future_split_requirement')}")
    add("T18", f"the cross-class patient {CROSS_TOKEN} stays in ONE group with its "
               "labels unchanged",
        cross_tokens == [CROSS_TOKEN]
        and len({r["group_key_for_future_partitions"] for r in records
                 if r["TOKEN_1"] == CROSS_TOKEN}) == 1
        and sorted(tok_cls[CROSS_TOKEN]) == ["Benign", "Malignant"],
        f"{CROSS_TOKEN}: {sum(1 for r in records if r['TOKEN_1'] == CROSS_TOKEN)} "
        f"image(s), classes {sorted(tok_cls.get(CROSS_TOKEN, []))}, one group key")
    add("T19", f"identifier {SPECIAL_TOKEN} is kept verbatim, not zero-padded",
        all(r["TOKEN_1"] == SPECIAL_TOKEN for r in records
            if r["TOKEN_1"].lstrip("0") == SPECIAL_TOKEN)
        and sum(1 for r in records if r["TOKEN_1"] == SPECIAL_TOKEN) == 2,
        f"{sum(1 for r in records if r['TOKEN_1'] == SPECIAL_TOKEN)} image(s) with "
        f"TOKEN_1={SPECIAL_TOKEN}")
    add("T20", f"excluded == {EXPECT['excluded']}; the global ledger is NOT modified",
        sum(1 for r in records
            if r["exclusion_status"] != "INCLUDED_CANONICAL_JOINT")
        == EXPECT["excluded"],
        "the 18 JPEG masks are ELIGIBLE and are not joint-task-ineligible")
    add("T21", "no unresolved record", not unresolved, f"{len(unresolved)}")
    add("T22", "no canonical path collision", not collisions, f"{len(collisions)}")
    add("T23", "RAW unchanged", raw_diff["unchanged"],
        f"modified {len(raw_diff['modified'])}, added {len(raw_diff['added'])}")
    add("T24", "completed audit artifacts unchanged", aud_diff["unchanged"],
        f"modified {len(aud_diff['modified'])}")
    add("T25", "no split created", not new_splits, f"{len(new_splits)}")
    add("T26", ("dry-run wrote no canonical PNG" if mode == "dry-run"
                else f"execution wrote {EXPECT['png']} canonical PNG"),
        (not new_png) if mode == "dry-run" else (len(new_png) == EXPECT["png"]),
        f"{len(new_png)} new PNG")
    add("T27", "no split field in the manifest schema",
        not any(is_split_like(f) for f in SCHEMA_FIELDS),
        "group_key_for_future_partitions carries a PATIENT key, not a partition")
    add("T28", "the JPEG masks are re-encoded and their pre-encoding history is not "
               "available locally", None,
        "the recovered mask matches the file's content; it is not claimed identical "
        "to what the annotator drew before the re-encoding")
    verified = [t for t in T if t[2] is not None]
    failed = [t for t in verified if not t[2]]
    for t in T:
        (LOGGER.info if t[2] is not False else LOGGER.error)(
            "      %-5s [%-11s] %-52s %s", t[0],
            "DECLARATION" if t[2] is None else ("PASS" if t[2] else "FAIL"),
            t[1][:52], t[3][:52])

    tag = "DRY_RUN_" if mode == "dry-run" else ""
    out_rows = [{k: r[k] for k in SCHEMA_FIELDS} for r in records]
    write_csv(manifests / f"{tag}canonical_manifest.csv", SCHEMA_FIELDS, out_rows)
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "canonical_manifest"
        ws.append(SCHEMA_FIELDS)
        for c in range(1, len(SCHEMA_FIELDS) + 1):
            ws.cell(1, c).font = Font(bold=True, color="FFFFFF")
            ws.cell(1, c).fill = PatternFill("solid", fgColor="2F5496")
            ws.cell(1, c).alignment = Alignment(horizontal="center", wrap_text=True)
        for r in out_rows:
            ws.append([r.get(h, "") for h in SCHEMA_FIELDS])
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = (f"A1:{get_column_letter(len(SCHEMA_FIELDS))}"
                              f"{len(out_rows) + 1}")
        wide = {"image_policy_rule", "mask_policy_rule", "identity_provenance",
                "label_provenance", "mask_provenance", "notes", "exclusion_reason",
                "near_duplicate_flag_PROVENANCE_ONLY"}
        for i, h in enumerate(SCHEMA_FIELDS, 1):
            ws.column_dimensions[get_column_letter(i)].width = 70 if h in wide else 22
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        wb.save(manifests / f"{tag}canonical_manifest.xlsx")
        LOGGER.info("WROTE %-44s (1 sheet)", f"{tag}canonical_manifest.xlsx")
    except ImportError:
        LOGGER.warning("openpyxl not installed; XLSX skipped")

    blocked = bool(failed)
    state = {
        "phase": "2B.3.11", "dataset": DATASET, "mode": mode, "generated": now_iso(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "schema_sha256": sha256_text(join_list(SCHEMA_FIELDS, ",")),
        "evidence_hashes": ev["hashes"], "source_root": str(src_root),
        "source_root_from": "--raw-root" if raw_cli else ev["dataset_root_locator"],
        "policies": {"image": {"id": IMAGE_POLICY_ID, "rule": IMAGE_POLICY_RULE},
                     "mask_native": {"id": NATIVE_POLICY_ID,
                                     "rule": NATIVE_POLICY_RULE,
                                     "records": len(native)},
                     "mask_jpeg": {"id": JPEG_POLICY_ID, "rule": JPEG_POLICY_RULE,
                                   "records": len(jpegs),
                                   "gap_verified_empty_on_every_file": not gap_bad,
                                   "threshold_selected_using_model_performance":
                                       False}},
        "counts": {"source_images": len(images), "source_masks": cu.get("mask_files"),
                   "resolved_pairs": len(records),
                   "benign": by_el.get("BENIGN", 0),
                   "malignant": by_el.get("MALIGNANT", 0),
                   "native_binary_masks": len(native),
                   "jpeg_binarised_masks": len(jpegs),
                   "conversion_plans": plans, "excluded": 0,
                   "patients": len(pats)},
        "mask_rules_applied": dict(rule_c),
        "channel_replication": dict(repl_c),
        "pixels_in_empty_gap": gap_bad,
        "identity": {"key": "PATIENT::<TOKEN_1>",
                     "provenance": "AUTHOR_CONFIRMED_DATASET_SEMANTICS",
                     "audit_verdict": summ.get("identity_verdict"),
                     "author_confirmation": author,
                     "cross_class_patients": cross_tokens,
                     "special_identifier": SPECIAL_TOKEN},
        "double_extension_recoveries": len(doubles),
        "image_mappings_resolved": img_ok, "mask_mappings_resolved": mask_ok,
        "unresolved": unresolved[:50], "dimension_mismatches": dim_bad,
        "collisions": collisions, "canonical_png_written": png_written,
        "raw_modified": not raw_diff["unchanged"],
        "audit_modified": not aud_diff["unchanged"],
        "exclusion_ledger_modified": False,
        "tests_pass": sum(1 for t in verified if t[2]), "tests_fail": len(failed),
        "verdict": ("BLOCKED" if blocked else
                    "READY_FOR_AIISLAB_CANONICAL_EXECUTION" if mode == "dry-run"
                    else "AIISLAB_CANONICAL_BUILD_COMPLETE"),
    }
    (reports / f"{tag}canonical_build_report.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    LOGGER.info("WROTE %-44s", f"{tag}canonical_build_report.json")

    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 2B.3.11 AIISLAB - %s", mode.upper())
    LOGGER.info("Source:      %d image(s) / %s mask(s); %d recovered by "
                "double-extension", len(images), cu.get("mask_files"), len(doubles))
    LOGGER.info("Pairs:       %d resolved", len(records))
    LOGGER.info("B / M:       %d / %d", by_el.get("BENIGN", 0),
                by_el.get("MALIGNANT", 0))
    LOGGER.info("Mask rules:  %s on %d | %s on %d", NATIVE_POLICY_ID, len(native),
                JPEG_POLICY_ID, len(jpegs))
    LOGGER.info("             applied: %s", dict(rule_c))
    LOGGER.info("             pixels in the 32-191 gap: %d record(s)", len(gap_bad))
    LOGGER.info("Patients:    %d; key PATIENT::<TOKEN_1>, provenance "
                "AUTHOR_CONFIRMED_DATASET_SEMANTICS", len(pats))
    LOGGER.info("             cross-class %s kept in one group; %s kept verbatim",
                cross_tokens or "none", SPECIAL_TOKEN)
    LOGGER.info("Exclusions:  0  (the 18 JPEG masks are ELIGIBLE; ledger untouched)")
    LOGGER.info("RAW modified %s | audit modified %s | PNG %d | splits %d",
                state["raw_modified"], state["audit_modified"], png_written,
                len(new_splits))
    LOGGER.info("TESTS PASS: %d    TESTS FAIL: %d", state["tests_pass"], len(failed))
    LOGGER.info("FINAL VERDICT:")
    if blocked:
        LOGGER.error("BLOCKED")
        for t in failed:
            LOGGER.error("    FAILED %s: %s | %s", t[0], t[1], t[3])
        return 4
    LOGGER.info(state["verdict"])
    if mode == "dry-run":
        LOGGER.info("STOP. --execute needs human approval.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AIISLAB canonical bank builder.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    p.add_argument("--prep-root", default=DEFAULT_PREP_ROOT)
    p.add_argument("--raw-root", default=None,
                   help="AIISLAB source root on H:. The Phase 1A.1 corrected-units "
                        "file carries no raw_root key, so it must be given here.")
    a = p.parse_args(argv)
    mode = "dry-run" if a.dry_run else "execute"
    if mode == "execute":
        st = (Path(a.prep_root) / BANK_NAME / DATASET / "reports"
              / "DRY_RUN_canonical_build_report.json")
        if not st.is_file():
            sys.stderr.write("EXECUTE refused: no dry-run report\n")
            return 5
        d = json.loads(st.read_text(encoding="utf-8"))
        if d.get("verdict") != "READY_FOR_AIISLAB_CANONICAL_EXECUTION":
            sys.stderr.write(f"EXECUTE refused: last verdict {d.get('verdict')!r}\n")
            return 5
        if d.get("builder_sha256") != sha256_file(Path(__file__).resolve()):
            sys.stderr.write("EXECUTE refused: the builder changed since the dry-run\n")
            return 5
        want = d.get("source_root")
        got = str(Path(a.raw_root)) if a.raw_root else None
        if want and got and want != got:
            sys.stderr.write(f"EXECUTE refused: the dry-run verified {want!r} but "
                             f"--raw-root is {got!r}\n")
            return 5
        ev = load_evidence(Path(a.prep_root))
        for name, h in (d.get("evidence_hashes") or {}).items():
            if ev["hashes"].get(name) != h:
                sys.stderr.write(f"EXECUTE refused: {name} changed since the dry-run\n")
                return 5
    try:
        return run(mode, Path(a.prep_root),
                   Path(a.raw_root) if a.raw_root else None)
    except Exception:  # noqa: BLE001
        if LOGGER.handlers:
            LOGGER.critical("UNHANDLED EXCEPTION\n%s", traceback.format_exc())
        else:
            sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
