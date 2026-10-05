#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bmc_eval.py
===========

R4 - PER-RECORD EVALUATOR (R4_EVALUATION_SPEC.md, Section 10.1)

    model outputs -> exactly-once sigmoid -> masks and probabilities
                  -> per-record metrics -> prediction ledger

Scope, and what this module deliberately does NOT do
    It computes per-record quantities only and writes one write-once ledger per
    (run, dataset, split). It does not aggregate across records or clients, does not
    bootstrap, does not compare methods, does not derive tau_val and does not select a
    checkpoint. Those belong to the analysis module (Section 10.2) and the
    validation-only selection component (Section 10.3).

The output contract (Section 1)
    Models return RAW LOGITS. This module applies the logistic sigmoid exactly once.
    A sample is malignant iff p >= tau; a pixel is foreground iff sigma(z) >= 0.5.

Ordering safeguard (Section 10.3)
    A ledger for the internal test or an external dataset is written only when a
    frozen checkpoint-selection record is supplied and agrees with the provenance.

Run the self-tests (synthetic data only; no dataset is read):
    python bmc_eval.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EVAL_SPEC_VERSION = "R4_EVALUATION_SPEC/v1"
TAU_PRIMARY = 0.5          # primary classification operating point (R4-5)
TAU_SEG = 0.5              # fixed segmentation threshold (Section 2.2)
POST_SELECTION_SPLITS = ("test", "EXTERNAL_TEST_ONLY")

LEDGER_FIELDS = [
    # identity
    "record_id", "dataset", "client_id", "split", "group_key", "group_unit",
    "modality", "label",
    # classification
    "cls_logit", "cls_prob", "pred_at_0.5",
    # segmentation
    "seg_threshold", "dice", "iou", "hd95", "gt_fg_px", "pred_fg_px",
    "intersection_px", "pred_empty",
    # provenance
    "run_id", "method", "seed", "selected_round", "selection_criterion_name",
    "selection_criterion_value", "model_sha256", "splits_manifest_sha256",
    "code_sha256", "eval_spec_version",
]
PROVENANCE_KEYS = ("run_id", "method", "seed", "selected_round",
                   "selection_criterion_name", "selection_criterion_value",
                   "model_sha256", "splits_manifest_sha256")


class ContractError(ValueError):
    """An input violates the R4 output contract or an ordering safeguard."""


# ══════════════════════════════════════════════════════════════════════════════
# 1. The sigmoid, applied exactly once
# ══════════════════════════════════════════════════════════════════════════════
def sigmoid(z) -> np.ndarray:
    """Numerically stable logistic sigmoid. The two branches never exponentiate a
    positive number, so no overflow occurs for any finite input."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _log_sigmoid(z) -> np.ndarray:
    """log sigma(z), stable for large |z|; used only by the inverse test."""
    z = np.asarray(z, dtype=np.float64)
    return -np.logaddexp(0.0, -z)


def _probability_like(values: np.ndarray) -> bool:
    """Guard against probabilities passed as logits: sigmoid of values that all lie
    in [0, 1] lands in [0.5, 0.731], the double-sigmoid signature. This is an
    additional guard only; the mathematical tests U1-U8 are the primary check."""
    v = np.asarray(values, dtype=np.float64)
    return v.size >= 20 and bool(np.all((v >= 0.0) & (v <= 1.0)))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-record segmentation metrics
# ══════════════════════════════════════════════════════════════════════════════
def _boundary(mask: np.ndarray) -> np.ndarray:
    """Foreground pixels with at least one 4-neighbour outside the mask. Pixels on
    the image border count as boundary (the outside is treated as background)."""
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask, border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile of the SYMMETRIC boundary-to-boundary Euclidean distances,
    in pixels of the 256 x 256 grid: the distances from every predicted-boundary
    pixel to the ground-truth boundary and from every ground-truth-boundary pixel to
    the predicted boundary are pooled, and the 95th percentile of the pooled set is
    taken (numpy linear interpolation). Undefined (NaN) when either mask is empty.

    Note: this is the pooled-set definition stated in the specification. Some
    libraries instead report the maximum of the two directed 95th percentiles; the
    two coincide only in symmetric cases."""
    from scipy.ndimage import distance_transform_edt
    if not pred.any() or not gt.any():
        return float("nan")
    bp, bg = _boundary(pred), _boundary(gt)
    d_to_gt = distance_transform_edt(~bg)      # distance of every pixel to gt boundary
    d_to_pred = distance_transform_edt(~bp)
    dist = np.concatenate([d_to_gt[bp], d_to_pred[bg]])
    return float(np.percentile(dist, 95))


def segmentation_record(seg_logits: np.ndarray, gt_mask: np.ndarray
                        ) -> Tuple[Dict[str, Any], np.ndarray]:
    """One image: sigmoid once, threshold at 0.5 with >=, and per-record metrics."""
    seg_logits = np.asarray(seg_logits, dtype=np.float64)
    gt = np.asarray(gt_mask).astype(bool)
    if seg_logits.shape != gt.shape:
        raise ContractError(f"segmentation logits {seg_logits.shape} vs mask {gt.shape}")
    if not gt.any():
        raise ContractError("empty ground-truth mask: every cohort mask is non-empty "
                            "(verified in the bank builds), so this input is invalid")
    pred = sigmoid(seg_logits) >= TAU_SEG
    inter = int(np.count_nonzero(pred & gt))
    gt_px = int(np.count_nonzero(gt))
    pr_px = int(np.count_nonzero(pred))
    union = gt_px + pr_px - inter
    empty = pr_px == 0
    rec = {"seg_threshold": TAU_SEG,
           "dice": 2.0 * inter / (gt_px + pr_px),     # gt non-empty, so never 0/0
           "iou": inter / union,
           "hd95": float("nan") if empty else hd95(pred, gt),
           "gt_fg_px": gt_px, "pred_fg_px": pr_px, "intersection_px": inter,
           "pred_empty": int(empty)}
    return rec, pred


def classification_record(cls_logit: float) -> Dict[str, Any]:
    z = float(cls_logit)
    if not math.isfinite(z):
        raise ContractError(f"non-finite classification logit {z}")
    p = float(sigmoid(np.array([z]))[0])
    return {"cls_logit": z, "cls_prob": p, "pred_at_0.5": int(p >= TAU_PRIMARY)}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Ledger: write-once, hashed
# ══════════════════════════════════════════════════════════════════════════════
def _fmt(v: Any) -> str:
    """Floats written with repr(), which round-trips exactly; an undefined value is
    written as an empty field, never imputed."""
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return "" if math.isnan(v) else repr(v)
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    return str(v)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _check_selection_record(split: str, provenance: dict,
                            selection_record: Optional[Path]) -> Optional[str]:
    """Section 10.3 ordering safeguard, enforced at the point of writing."""
    if split not in POST_SELECTION_SPLITS:
        return None
    if selection_record is None or not Path(selection_record).is_file():
        raise ContractError(f"a {split} ledger requires a frozen checkpoint-selection "
                            "record; none was supplied, so selection has not been "
                            "frozen and test/external evaluation is refused")
    rec = json.loads(Path(selection_record).read_text(encoding="utf-8"))
    if rec.get("frozen") is not True:
        raise ContractError("the selection record is not marked frozen")
    if rec.get("selection_split") != "validation":
        raise ContractError("the selection record was not derived from validation data")
    for k in ("selected_round", "selection_criterion_name", "selection_criterion_value"):
        if str(rec.get(k)) != str(provenance.get(k)):
            raise ContractError(f"selection record {k}={rec.get(k)!r} disagrees with "
                                f"the provenance {provenance.get(k)!r}")
    return sha256_file(Path(selection_record))


def write_ledger(path: Path, rows: Sequence[dict], sidecar_extra: dict) -> str:
    """Write once. Refuses to overwrite an existing ledger. Returns the SHA-256."""
    path = Path(path)
    if path.exists() or path.with_suffix(".sha256.json").exists():
        raise ContractError(f"ledger already exists and is write-once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as fh:        # 'x' = never overwrite
        w = csv.writer(fh)
        w.writerow(LEDGER_FIELDS)
        for r in rows:
            w.writerow([_fmt(r[f]) for f in LEDGER_FIELDS])
    digest = sha256_file(path)
    side = {"ledger": path.name, "sha256": digest, "rows": len(rows),
            "fields": LEDGER_FIELDS, "eval_spec_version": EVAL_SPEC_VERSION,
            **sidecar_extra}
    with path.with_suffix(".sha256.json").open("x", encoding="utf-8") as fh:
        json.dump(side, fh, indent=2)
    return digest


def read_ledger(path: Path, verify: bool = True) -> List[dict]:
    path = Path(path)
    if verify:
        side = json.loads(path.with_suffix(".sha256.json").read_text(encoding="utf-8"))
        if sha256_file(path) != side["sha256"]:
            raise ContractError(f"ledger hash mismatch: {path}")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ══════════════════════════════════════════════════════════════════════════════
# 4. One (run, dataset, split)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_split(*, records: Sequence[dict], cls_logits: Sequence[float],
                   seg_logits: np.ndarray, gt_masks: np.ndarray, provenance: dict,
                   out_path: Path, selection_record: Optional[Path] = None,
                   write_masks: Optional[bool] = None) -> dict:
    """Per-record evaluation of one model on one (dataset, split).

    records     : one dict per sample, carrying record_id, dataset, client_id, split,
                  group_key, group_unit, modality, label (e.g. BMCSplitDataset rows)
    cls_logits  : raw classification logits, shape (N,)
    seg_logits  : raw segmentation logits, shape (N, H, W)
    gt_masks    : ground-truth masks on the same grid, shape (N, H, W), values {0,1}
    """
    n = len(records)
    cls_logits = np.asarray(cls_logits, dtype=np.float64)
    seg_logits = np.asarray(seg_logits)
    gt_masks = np.asarray(gt_masks)
    if not (len(cls_logits) == seg_logits.shape[0] == gt_masks.shape[0] == n):
        raise ContractError("records, cls_logits, seg_logits and gt_masks differ in length")
    missing = [k for k in PROVENANCE_KEYS if k not in provenance]
    if missing:
        raise ContractError(f"provenance missing {missing}")
    splits = {r["split"] for r in records}
    datasets = {r["dataset"] for r in records}
    if len(splits) != 1 or len(datasets) != 1:
        raise ContractError(f"one ledger covers one (dataset, split); got {datasets} x {splits}")
    split = splits.pop()
    # record_id must be unique within one (run, dataset, split) ledger. Checked before
    # anything is written, so a duplicate leaves no ledger, sidecar or mask behind.
    ids = [str(r["record_id"]) for r in records]
    if len(set(ids)) != len(ids):
        seen, dup = set(), []
        for x in ids:
            if x in seen and x not in dup:
                dup.append(x)
            seen.add(x)
        raise ContractError(f"record_id is not unique within this ledger: "
                            f"{len(ids) - len(set(ids))} duplicate row(s), e.g. {dup[:3]}")
    if _probability_like(cls_logits):
        raise ContractError("classification inputs all lie in [0, 1]: they look like "
                            "probabilities, not logits (double-sigmoid risk)")
    if _probability_like(seg_logits.reshape(-1)[: max(20, min(seg_logits.size, 100000))]) \
            and _probability_like(seg_logits):
        raise ContractError("segmentation inputs all lie in [0, 1]: they look like "
                            "probabilities, not logits (double-sigmoid risk)")
    sel_hash = _check_selection_record(split, provenance, selection_record)
    if write_masks is None:
        write_masks = split in POST_SELECTION_SPLITS

    csha = code_sha256()
    mask_dir = Path(out_path).with_suffix("")
    mask_dir = mask_dir.parent / (mask_dir.name + "__pred_masks")
    rows: List[dict] = []
    for i, rec in enumerate(records):
        c = classification_record(cls_logits[i])
        s, pred = segmentation_record(seg_logits[i], gt_masks[i])
        row = {k: rec[k] for k in ("record_id", "dataset", "client_id", "split",
                                    "group_key", "group_unit", "modality")}
        row["label"] = int(rec["label"])
        row.update(c)
        row.update(s)
        row.update({k: provenance[k] for k in PROVENANCE_KEYS})
        row["code_sha256"] = csha
        row["eval_spec_version"] = EVAL_SPEC_VERSION
        rows.append(row)
        if write_masks:
            from PIL import Image
            mask_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_"
                           for ch in str(rec["record_id"]))
            Image.fromarray(pred.astype(np.uint8) * 255, mode="L").save(
                mask_dir / f"{safe}.png")
    digest = write_ledger(Path(out_path), rows,
                          {"split": split, "dataset": next(iter(datasets)),
                           "selection_record_sha256": sel_hash,
                           "pred_masks_written": bool(write_masks)})
    return {"ledger": str(out_path), "sha256": digest, "rows": n,
            "masks_written": bool(write_masks)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Self-tests (synthetic; closed-form expected values)
# ══════════════════════════════════════════════════════════════════════════════
def selftest() -> int:
    T: List[Tuple[str, str, bool, str]] = []

    def add(i, d, ok, b=""):
        T.append((i, d, bool(ok), b))

    grid = np.linspace(-30, 30, 6001)
    # U1-U6 ------------------------------------------------------------------
    add("U1", "sigma(0) = 0.5 exactly", sigmoid(np.array([0.0]))[0] == 0.5)
    s3 = sigmoid(np.array([math.log(3.0), -math.log(3.0)]))
    add("U2", "sigma(ln 3) = 0.75 and sigma(-ln 3) = 0.25 to machine precision",
        abs(s3[0] - 0.75) <= 4 * np.finfo(float).eps and abs(s3[1] - 0.25) <= 4 * np.finfo(float).eps,
        f"errors {abs(s3[0]-0.75):.1e}, {abs(s3[1]-0.25):.1e}")
    sym = np.max(np.abs(sigmoid(-grid) - (1.0 - sigmoid(grid))))
    add("U3", "symmetry sigma(-z) = 1 - sigma(z) on [-30, 30]", sym <= 2 * np.finfo(float).eps,
        f"max error {sym:.1e}")
    # U4: logit(p) = log p - log(1 - p), with 1 - sigma(z) = sigma(-z) (U3) so that
    # 1 - p is never formed by subtraction, which would lose precision near p = 1.
    zz = np.linspace(-20, 20, 4001)
    inv = _log_sigmoid(zz) - _log_sigmoid(-zz)
    inv_err = float(np.max(np.abs(inv - zz)))
    add("U4", "inverse: logit(sigma(z)) = z within 1e-9 on [-20, 20]", inv_err <= 1e-9,
        f"max error {inv_err:.1e}")
    sg = sigmoid(np.linspace(-30, 30, 601))
    add("U5", "strict monotonicity", bool(np.all(np.diff(sg) > 0)))
    # Overflow, invalid (NaN) and division faults are raised as errors; any warning
    # under default settings is recorded. IEEE underflow is NOT a fault here:
    # e^(-1000) rounding to 0 is the exact float64 value of sigma(-1000).
    import warnings
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            try:
                ext = sigmoid(np.array([1000.0, -1000.0]))
                ok6 = ext[0] == 1.0 and ext[1] == 0.0 and bool(np.isfinite(ext).all())
            except FloatingPointError:
                ok6 = False
    add("U6", "stability: sigma(+-1000) = 1, 0 with no overflow, NaN or warning",
        ok6 and not wrec, f"{len(wrec)} warning(s); underflow of e^(-1000) to 0 is exact")

    # a synthetic split used by U7, U8, M- and L-tests ----------------------
    rng = np.random.default_rng(7)
    N, H = 24, 64
    known_z = np.array([0.0, 1.0, -1.0, 2.5, -2.5, 4.0, -4.0, 0.3] + list(rng.normal(0, 2, N - 8)))
    gts = np.zeros((N, H, H), np.uint8)
    segz = np.full((N, H, H), -6.0)
    for i in range(N):
        gts[i, 10:30, 10:30] = 1
        segz[i, 10 + (i % 5):30 + (i % 5), 10:30] = 6.0       # shifted prediction
    segz[3] = -6.0                                           # one empty prediction
    recs = [{"record_id": f"R{i:03d}", "dataset": "SYN", "client_id": 0, "split": "validation",
             "group_key": f"PATIENT::{i // 2}", "group_unit": "PATIENT",
             "modality": "ultrasound", "label": int(known_z[i] >= 0)} for i in range(N)]
    prov = {"run_id": "selftest", "method": "SYN", "seed": 0, "selected_round": 7,
            "selection_criterion_name": "joint_val_loss", "selection_criterion_value": 0.123,
            "model_sha256": "0" * 64, "splits_manifest_sha256": "1" * 64}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lp = td / "syn_val.csv"
        evaluate_split(records=recs, cls_logits=known_z, seg_logits=segz, gt_masks=gts,
                       provenance=prov, out_path=lp)
        L = read_ledger(lp)
        probs = np.array([float(r["cls_prob"]) for r in L])
        exp = sigmoid(known_z)
        dbl = sigmoid(sigmoid(known_z))
        diff = ~np.isclose(exp, dbl)
        add("U7", "exactly once: stored p equals sigma(z) and differs from sigma(sigma(z))",
            np.array_equal(probs, exp) and not np.any(np.isclose(probs[diff], dbl[diff])),
            f"{int(diff.sum())}/{N} records where the two differ, all matched sigma(z)")
        recomputed = sigmoid(np.array([float(r["cls_logit"]) for r in L]))
        add("U8", "every ledger probability is recomputable from its stored logit",
            np.array_equal(recomputed, probs), "exact float round-trip")
        # M1 tie rule
        r0 = L[0]
        add("M1", "classification rule p >= tau: logit 0 gives p = 0.5 and predicts malignant",
            float(r0["cls_prob"]) == 0.5 and r0["pred_at_0.5"] == "1")
        # M2 Dice/IoU closed form: GT 20x20, prediction shifted by k rows
        okm2 = True
        for i in (0, 1, 2, 4):
            k = i % 5
            inter = (20 - k) * 20
            d_exp = 2 * inter / 800
            j_exp = inter / (800 - inter)
            okm2 &= math.isclose(float(L[i]["dice"]), d_exp, rel_tol=0, abs_tol=1e-15)
            okm2 &= math.isclose(float(L[i]["iou"]), j_exp, rel_tol=0, abs_tol=1e-15)
        add("M2", "Dice and IoU equal their closed forms for shifted squares", okm2,
            "shift k in {0,1,2,4}: Dice = 2(20-k)20/800, IoU = (20-k)20/(800-(20-k)20)")
        add("M3", "identical masks: Dice = 1, IoU = 1, HD95 = 0",
            L[0]["dice"] == "1.0" and L[0]["iou"] == "1.0" and float(L[0]["hd95"]) == 0.0)
        r3 = L[3]
        add("M4", "empty prediction: Dice = 0, IoU = 0, HD95 undefined, pred_empty = 1",
            r3["dice"] == "0.0" and r3["iou"] == "0.0" and r3["hd95"] == ""
            and r3["pred_empty"] == "1" and r3["pred_fg_px"] == "0")
        m7 = all(math.isclose(float(r["dice"]),
                              2 * int(r["intersection_px"]) / (int(r["gt_fg_px"]) + int(r["pred_fg_px"])),
                              rel_tol=0, abs_tol=1e-15) for r in L)
        add("M7", "Dice recomputes exactly from the stored pixel counts", m7)
        # L1 write-once
        try:
            evaluate_split(records=recs, cls_logits=known_z, seg_logits=segz, gt_masks=gts,
                           provenance=prov, out_path=lp)
            add("L1", "a ledger cannot be overwritten", False)
        except ContractError:
            add("L1", "a ledger cannot be overwritten", True)
        side = json.loads(lp.with_suffix(".sha256.json").read_text())
        add("L2", "the sidecar SHA-256 matches the ledger file", side["sha256"] == sha256_file(lp))
        with lp.open(encoding="utf-8") as fh:
            hdr = next(csv.reader(fh))
        add("L3", "ledger fields exactly as specified, in order", hdr == LEDGER_FIELDS,
            f"{len(hdr)} fields")
        # L5 masks only for test/external
        add("L5a", "no predicted-mask PNGs are written for validation",
            not any(td.glob("syn_val__pred_masks/*.png")))
        # L4 ordering safeguard
        trecs = [dict(r, split="test") for r in recs]
        refused = False
        try:
            evaluate_split(records=trecs, cls_logits=known_z, seg_logits=segz, gt_masks=gts,
                           provenance=prov, out_path=td / "t1.csv")
        except ContractError:
            refused = True
        add("L4a", "a test ledger is refused without a frozen selection record", refused)
        bad = td / "sel_bad.json"
        bad.write_text(json.dumps({"frozen": True, "selection_split": "validation",
                                   "selected_round": 99,
                                   "selection_criterion_name": "joint_val_loss",
                                   "selection_criterion_value": 0.123}))
        refused = False
        try:
            evaluate_split(records=trecs, cls_logits=known_z, seg_logits=segz, gt_masks=gts,
                           provenance=prov, out_path=td / "t2.csv", selection_record=bad)
        except ContractError:
            refused = True
        add("L4b", "a test ledger is refused when the selection record disagrees", refused)
        tst = td / "sel_test.json"
        tst.write_text(json.dumps({"frozen": True, "selection_split": "test",
                                   "selected_round": 7,
                                   "selection_criterion_name": "joint_val_loss",
                                   "selection_criterion_value": 0.123}))
        refused = False
        try:
            evaluate_split(records=trecs, cls_logits=known_z, seg_logits=segz, gt_masks=gts,
                           provenance=prov, out_path=td / "t3.csv", selection_record=tst)
        except ContractError:
            refused = True
        add("L4c", "a selection record derived from test data is refused", refused)
        good = td / "sel_ok.json"
        good.write_text(json.dumps({"frozen": True, "selection_split": "validation",
                                    "selected_round": 7,
                                    "selection_criterion_name": "joint_val_loss",
                                    "selection_criterion_value": 0.123}))
        res = evaluate_split(records=trecs, cls_logits=known_z, seg_logits=segz, gt_masks=gts,
                             provenance=prov, out_path=td / "t4.csv", selection_record=good)
        side4 = json.loads((td / "t4.sha256.json").read_text())
        add("L4d", "with a valid frozen record the test ledger is written and the record hash kept",
            res["rows"] == N and side4["selection_record_sha256"] == sha256_file(good))
        add("L5b", "predicted-mask PNGs are written for test",
            len(list(td.glob("t4__pred_masks/*.png"))) == N)
        # L6: duplicate record_id refused before any output artefact exists. Run on
        # the test split with a valid frozen selection record, so that masks WOULD be
        # written if the guard were absent.
        drecs = [dict(r) for r in trecs]
        drecs[5]["record_id"] = drecs[4]["record_id"]
        l6 = td / "l6.csv"
        raised = False
        try:
            evaluate_split(records=drecs, cls_logits=known_z, seg_logits=segz, gt_masks=gts,
                           provenance=prov, out_path=l6, selection_record=good)
        except ContractError:
            raised = True
        leftover = [x.name for x in (l6, l6.with_suffix(".sha256.json"),
                                     td / "l6__pred_masks") if x.exists()]
        add("L6", "duplicate record_id is refused and no ledger, sidecar or mask is created",
            raised and not leftover,
            "refused; no artefact created" if raised and not leftover
            else f"raised={raised}, artefacts left: {leftover}")
        # G1 guard
        refused = False
        try:
            evaluate_split(records=recs, cls_logits=sigmoid(known_z), seg_logits=segz,
                           gt_masks=gts, provenance=prov, out_path=td / "g1.csv")
        except ContractError:
            refused = True
        add("G1", "probabilities passed as classification logits are refused (additional guard)",
            refused)
    # M5 HD95 closed forms ----------------------------------------------------
    a = np.zeros((64, 64), bool); a[10, 10] = True
    b = np.zeros((64, 64), bool); b[10, 15] = True
    add("M5a", "HD95 of two single pixels 5 px apart = 5", hd95(b, a) == 5.0)
    sq = np.zeros((128, 128), bool); sq[20:60, 20:60] = True
    out = sq.copy(); out[120, 120] = True
    add("M5b", "HD95 ignores one far outlier: square + distant pixel gives 0 (a percentile, not a maximum)",
        hd95(out, sq) == 0.0, f"HD95 = {hd95(out, sq)}")
    c = np.zeros((64, 64), bool); c[20:30, 20:30] = True
    d = np.zeros((64, 64), bool); d[20:30, 23:33] = True
    add("M5c", "HD95 is symmetric in its arguments", hd95(c, d) == hd95(d, c),
        f"{hd95(c, d)} vs {hd95(d, c)}")
    # M6 seg tie
    z = np.full((8, 8), -5.0); z[2, 2] = 0.0
    g = np.zeros((8, 8), np.uint8); g[2, 2] = 1
    s, pmask = segmentation_record(z, g)
    add("M6", "segmentation rule sigma(z) >= 0.5: a pixel with logit 0 is foreground",
        bool(pmask[2, 2]) and s["pred_fg_px"] == 1 and s["dice"] == 1.0)
    # M8 empty GT
    try:
        segmentation_record(np.zeros((8, 8)), np.zeros((8, 8), np.uint8))
        add("M8", "an empty ground-truth mask is refused as a contract violation", False)
    except ContractError:
        add("M8", "an empty ground-truth mask is refused as a contract violation", True)
    # S1 scope
    forbidden = [n for n in dir(sys.modules[__name__])
                 if any(t in n.lower() for t in ("bootstrap", "youden", "aggregate",
                                                  "macro", "micro", "select_checkpoint",
                                                  "tau_val"))]
    add("S1", "scope: no aggregation, bootstrap, tau_val or checkpoint-selection function",
        not forbidden, f"found {forbidden}" if forbidden else "")

    print("=" * 92)
    print(f"bmc_eval.py SELF-TESTS  ({EVAL_SPEC_VERSION})")
    print("=" * 92)
    for i, dsc, ok, b in T:
        print(f"  {i:4s} [{'PASS' if ok else 'FAIL'}]  {dsc}" + (f"\n         {b}" if b else ""))
    fails = [t for t in T if not t[2]]
    print("=" * 92)
    print(f"PASS = {len(T) - len(fails)}   FAIL = {len(fails)}")
    print("BMC_EVAL_VERIFIED" if not fails else "BMC_EVAL_BLOCKED")
    return 0 if not fails else 4


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="R4 per-record evaluator.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        raise SystemExit(1)
