#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bmc_stats.py
============

R4 - ANALYSIS AND STATISTICS MODULE (R4_EVALUATION_SPEC.md, Section 10.2)

Reads FROZEN prediction ledgers written by bmc_eval.py and computes every aggregate
the manuscript will report. It never reads model weights or images and never writes
to a ledger: ledgers are opened read-only, after their SHA-256 has been verified.

    per-client metrics -> aggregation per metric (Section 3)
    tau_val from validation ledgers only (Section 4.2, O3)
    cluster bootstrap on group_key, 2,000 attempted replicates (Sections 9.1-9.2)
    paired cluster bootstrap engine (Section 9.3)
    mean +- SD across a supplied seed set (Section 9.4)

Frozen statistical decisions implemented here:
    * two-sided paired-bootstrap p-value, null-centred with a +1 correction:
          p = [1 + #{ |d_b* - d_hat| >= |d_hat| }] / (B_valid + 1)
      over defined replicates only, so p = 0 is impossible;
    * seed summaries use the sample SD, ddof = 1;
    * BOOTSTRAP_SEED = 20260930, the analysis seed (distinct from the split seed
      20260929), recorded in every bootstrap output.

Still NOT implemented, because R4 leaves it to R5:
    * any confirmatory comparison family or Holm correction (dependency D3);
    * any training seed count or seed values (dependency D4).

Run the self-tests (synthetic ledgers only):
    python bmc_stats.py --selftest
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from bmc_eval import LEDGER_FIELDS, write_ledger  # single source of truth
except ImportError:                                     # pragma: no cover
    LEDGER_FIELDS = None
    write_ledger = None

B_ATTEMPTED = 2000
UNSTABLE_LIMIT = 100                       # n_undefined > 100 -> UNSTABLE_CI (O2)
TAU_PRIMARY = 0.5
INTERNAL_SPLITS = ("validation", "test")
EXTERNAL_SPLIT = "EXTERNAL_TEST_ONLY"
THRESHOLD_METRICS = ("sensitivity", "specificity", "balanced_accuracy", "accuracy", "f1")
SEG_MEAN_METRICS = ("dice", "iou")
SECONDARY_LABEL = "tau_val, validation-derived, secondary"
BOOTSTRAP_SEED = 20260930                  # analysis seed; the split seed is 20260929
SD_DDOF = 1                                # sample SD across training seeds (frozen)


class ContractError(ValueError):
    """An input violates the R4 contract or a data-role rule."""


# ══════════════════════════════════════════════════════════════════════════════
# 1. Frozen ledgers
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Ledger:
    path: str
    dataset: str
    split: str
    run_id: str
    record_id: np.ndarray
    group_key: np.ndarray
    label: np.ndarray
    cls_prob: np.ndarray
    dice: np.ndarray
    iou: np.ndarray
    hd95: np.ndarray                 # NaN where undefined (empty prediction)
    pred_empty: np.ndarray
    sha256: str = ""

    @property
    def n(self) -> int:
        return len(self.record_id)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ledger(path) -> Ledger:
    """Verify the SHA-256 against the sidecar, then read. Read-only."""
    path = Path(path)
    side = json.loads(path.with_suffix(".sha256.json").read_text(encoding="utf-8"))
    digest = _sha256(path)
    if digest != side["sha256"]:
        raise ContractError(f"ledger hash mismatch (tampered or corrupted): {path.name}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        header = rd.fieldnames or []
        rows = list(rd)
    if LEDGER_FIELDS is not None and header != LEDGER_FIELDS:
        raise ContractError(f"{path.name}: fields differ from the ledger specification")
    for need in ("record_id", "dataset", "split", "group_key", "label", "cls_prob",
                 "dice", "iou", "hd95", "pred_empty", "run_id"):
        if need not in header:
            raise ContractError(f"{path.name}: required field {need!r} missing")
    if not rows:
        raise ContractError(f"{path.name}: empty ledger")
    ds, sp, rn = {r["dataset"] for r in rows}, {r["split"] for r in rows}, {r["run_id"] for r in rows}
    if len(ds) != 1 or len(sp) != 1 or len(rn) != 1:
        raise ContractError(f"{path.name}: one ledger must hold one dataset, split and run")
    split = sp.pop()
    if split not in INTERNAL_SPLITS + (EXTERNAL_SPLIT,):
        raise ContractError(f"{path.name}: split {split!r} is not an analysable role")
    rid = np.array([r["record_id"] for r in rows])
    if len(set(rid.tolist())) != len(rid):
        raise ContractError(f"{path.name}: record_id is not unique")
    f = lambda k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
    return Ledger(path=str(path), dataset=ds.pop(), split=split, run_id=rn.pop(),
                  record_id=rid, group_key=np.array([r["group_key"] for r in rows]),
                  label=np.array([int(r["label"]) for r in rows]),
                  cls_prob=f("cls_prob"), dice=f("dice"), iou=f("iou"), hd95=f("hd95"),
                  pred_empty=np.array([int(r["pred_empty"]) for r in rows]),
                  sha256=digest)


def assert_internal(ledgers: Sequence[Ledger], split: str) -> None:
    if split not in INTERNAL_SPLITS:
        raise ContractError(f"internal analysis requires split in {INTERNAL_SPLITS}")
    for L in ledgers:
        if L.split != split:
            raise ContractError(f"{L.dataset}: split {L.split!r} where {split!r} is required")
    if len({L.dataset for L in ledgers}) != len(ledgers):
        raise ContractError("each client may appear once in an internal ledger set")


def assert_external(L: Ledger) -> None:
    if L.split != EXTERNAL_SPLIT:
        raise ContractError(f"{L.dataset}: not an external ledger")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Metrics on one client's rows
# ══════════════════════════════════════════════════════════════════════════════
def confusion(y: np.ndarray, p: np.ndarray, tau: float) -> Tuple[int, int, int, int]:
    """Rule: predicted malignant iff p >= tau."""
    pred = p >= tau
    tp = int(np.sum(pred & (y == 1))); fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0))); fp = int(np.sum(pred & (y == 0)))
    return tp, fn, tn, fp


def metrics_from_counts(tp: int, fn: int, tn: int, fp: int) -> Dict[str, Optional[float]]:
    d = lambda a, b: (a / b) if b else None
    sens, spec = d(tp, tp + fn), d(tn, tn + fp)
    return {"sensitivity": sens, "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2 if sens is not None and spec is not None else None,
            "accuracy": d(tp + tn, tp + tn + fp + fn),
            "f1": d(2 * tp, 2 * tp + fp + fn)}


def roc_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    """Mann-Whitney statistic with average ranks, so tied scores count 1/2."""
    npos, nneg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ps = p[order]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def hd95_summary(h: np.ndarray, empty: np.ndarray) -> Dict[str, Any]:
    ok = h[~np.isnan(h)]
    n = len(h)
    out = {"n_images": n, "n_hd95_defined": int(len(ok)),
           "n_empty_pred": int(np.sum(empty)), "pct_empty_pred": 100.0 * float(np.sum(empty)) / n}
    if len(ok):
        q1, med, q3 = np.percentile(ok, [25, 50, 75])
        out.update(hd95_median=float(med), hd95_q1=float(q1), hd95_q3=float(q3),
                   hd95_iqr=float(q3 - q1))
    else:
        out.update(hd95_median=None, hd95_q1=None, hd95_q3=None, hd95_iqr=None)
    return out


def client_value(L: Ledger, rows: np.ndarray, metric: str, tau: float) -> Optional[float]:
    """One metric for one client on a set of row indices (possibly with repeats)."""
    y, p = L.label[rows], L.cls_prob[rows]
    if metric == "roc_auc":
        return roc_auc(y, p)
    if metric in THRESHOLD_METRICS:
        return metrics_from_counts(*confusion(y, p, tau))[metric]
    if metric in SEG_MEAN_METRICS:
        return float(np.mean(getattr(L, metric)[rows]))
    if metric == "hd95":
        return hd95_summary(L.hd95[rows], L.pred_empty[rows])["hd95_median"]
    raise ContractError(f"unknown metric {metric!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Aggregation, defined per metric (Section 3)
# ══════════════════════════════════════════════════════════════════════════════
def _all(L: Ledger) -> np.ndarray:
    return np.arange(L.n)


def _macro(ledgers, views, metric, tau) -> Optional[float]:
    """Unweighted mean of the per-client values; undefined if any client is."""
    vals = [client_value(L, v, metric, tau) for L, v in zip(ledgers, views)]
    if any(x is None for x in vals):
        return None
    return float(np.mean(vals))


def _micro(ledgers, views, metric, tau) -> Optional[float]:
    if metric == "roc_auc":
        raise ContractError("pooled cross-client ROC-AUC is not defined in R4: it ranks "
                            "scores from clients with different score distributions")
    if metric in THRESHOLD_METRICS:
        c = np.sum([confusion(L.label[v], L.cls_prob[v], tau) for L, v in zip(ledgers, views)], axis=0)
        return metrics_from_counts(*[int(x) for x in c])[metric]
    if metric in SEG_MEAN_METRICS:
        return float(np.mean(np.concatenate([getattr(L, metric)[v] for L, v in zip(ledgers, views)])))
    if metric == "hd95":
        h = np.concatenate([L.hd95[v] for L, v in zip(ledgers, views)])
        e = np.concatenate([L.pred_empty[v] for L, v in zip(ledgers, views)])
        return hd95_summary(h, e)["hd95_median"]
    raise ContractError(f"unknown metric {metric!r}")


def summarise(ledgers: Sequence[Ledger], metric: str, level: str, tau: float = TAU_PRIMARY,
              views: Optional[Sequence[np.ndarray]] = None) -> Optional[float]:
    views = views if views is not None else [_all(L) for L in ledgers]
    if any(L.split == EXTERNAL_SPLIT for L in ledgers) and len(ledgers) > 1:
        raise ContractError("external datasets are summarised separately: no pooling across "
                            "external datasets and no pooling of internal with external")
    if level == "macro":
        return _macro(ledgers, views, metric, tau)
    if level == "micro":
        return _micro(ledgers, views, metric, tau)
    raise ContractError(f"level must be 'macro' or 'micro', not {level!r}")


def per_client_table(ledgers: Sequence[Ledger], tau: float = TAU_PRIMARY) -> List[Dict[str, Any]]:
    out = []
    for L in ledgers:
        a = _all(L)
        row = {"dataset": L.dataset, "split": L.split, "n": L.n, "tau": tau,
               "roc_auc": roc_auc(L.label, L.cls_prob)}
        row.update(metrics_from_counts(*confusion(L.label, L.cls_prob, tau)))
        row.update(dice=float(np.mean(L.dice)), iou=float(np.mean(L.iou)))
        row.update(hd95_summary(L.hd95, L.pred_empty))
        out.append(row)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. tau_val (Section 4.2, O3) - validation ledgers only
# ══════════════════════════════════════════════════════════════════════════════
def tau_val_candidates(p: np.ndarray) -> List[float]:
    """All distinct validation probabilities under p >= tau, plus +inf for the
    all-negative point. The smallest probability already gives all-positive."""
    return sorted(set(float(x) for x in p)) + [math.inf]


def derive_tau_val(val_ledgers: Sequence[Ledger]) -> Dict[str, Any]:
    for L in val_ledgers:
        if L.split != "validation":
            raise ContractError(f"tau_val may be derived from validation ledgers only; "
                                f"{L.dataset} is {L.split!r}")
    y = np.concatenate([L.label for L in val_ledgers])
    p = np.concatenate([L.cls_prob for L in val_ledgers])
    cands = tau_val_candidates(p)
    js = []
    for t in cands:
        m = metrics_from_counts(*confusion(y, p, t))
        js.append(m["sensitivity"] + m["specificity"] - 1.0)
    jmax = max(js)
    best = [t for t, j in zip(cands, js) if j == jmax]
    best.sort(key=lambda t: (abs(t - 0.5), t))       # closest to 0.5, then the lower
    return {"tau_val": best[0], "youden_j": jmax, "n_candidates": len(cands),
            "tied_maximisers": best, "n_validation": int(len(p)),
            "label": SECONDARY_LABEL, "frozen": True}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cluster bootstrap (Sections 9.1-9.2)
# ══════════════════════════════════════════════════════════════════════════════
def _group_rows(L: Ledger) -> List[np.ndarray]:
    keys = sorted(set(L.group_key.tolist()))
    return [np.flatnonzero(L.group_key == k) for k in keys]


def _client_draw(rng: np.random.Generator, group_rows: List[np.ndarray]) -> np.ndarray:
    """Resample GROUPS with replacement; every row of a drawn group comes with it."""
    pick = rng.integers(0, len(group_rows), len(group_rows))
    return np.concatenate([group_rows[i] for i in pick])


def make_resample_plan(ledgers: Sequence[Ledger], seed: int = BOOTSTRAP_SEED, B: int = B_ATTEMPTED
                       ) -> List[List[np.ndarray]]:
    """B replicates; in each, groups are resampled independently within every client.
    The plan is fixed by the seed alone, before any metric is computed."""
    rng = np.random.default_rng(seed)
    grs = [_group_rows(L) for L in ledgers]
    return [[_client_draw(rng, g) for g in grs] for _ in range(B)]


def ci_from_replicates(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    if len(values) != B_ATTEMPTED:
        raise ContractError(f"{len(values)} replicates supplied; exactly {B_ATTEMPTED} "
                            "attempted replicates are required")
    ok = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    n_undef = B_ATTEMPTED - len(ok)
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    return {"ci_low": float(lo), "ci_high": float(hi), "n_valid": int(len(ok)),
            "n_undefined": int(n_undef), "n_attempted": B_ATTEMPTED,
            "flag": "UNSTABLE_CI" if n_undef > UNSTABLE_LIMIT else "OK"}


def bootstrap(ledgers: Sequence[Ledger], stat: Callable[[Sequence[np.ndarray]], Optional[float]],
              seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """stat receives one view per ledger (row indices). A replicate whose statistic
    is undefined is recorded as undefined; it is never redrawn."""
    plan = make_resample_plan(ledgers, seed)
    vals = []
    for view in plan:
        v = stat(view)
        vals.append(v)
    res = ci_from_replicates(vals)
    res["estimate"] = stat([_all(L) for L in ledgers])
    res["seed"] = seed
    res["bootstrap_seed"] = seed
    return res


def centered_p_value(delta_hat: Optional[float], deltas: Sequence[Optional[float]]
                     ) -> Tuple[Optional[float], int]:
    """Two-sided null-centred paired-bootstrap p-value (frozen in R4):
        p = [1 + #{ |d_b* - d_hat| >= |d_hat| }] / (B_valid + 1)
    Only defined replicates enter B_valid; undefined ones are excluded, never
    redrawn. The +1 makes p = 0 impossible: the minimum is 1 / (B_valid + 1)."""
    valid = [d for d in deltas if d is not None and not math.isnan(d)]
    if delta_hat is None:
        return None, len(valid)
    exceed = sum(1 for d in valid if abs(d - delta_hat) >= abs(delta_hat))
    return (1 + exceed) / (len(valid) + 1), len(valid)


def paired_bootstrap(ledgers_a: Sequence[Ledger], ledgers_b: Sequence[Ledger],
                     stat: Callable[[Sequence[Ledger], Sequence[np.ndarray]], Optional[float]],
                     seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Delta = stat(B) - stat(A), both evaluated on the SAME resampled groups."""
    if len(ledgers_a) != len(ledgers_b):
        raise ContractError("paired comparison needs the same ledger set for both methods")
    for La, Lb in zip(ledgers_a, ledgers_b):
        if (La.dataset, La.split) != (Lb.dataset, Lb.split) or \
                not np.array_equal(La.record_id, Lb.record_id) or \
                not np.array_equal(La.group_key, Lb.group_key):
            raise ContractError(f"{La.dataset}: methods are not evaluated on identical "
                                "records and groups, so they cannot be paired")
    plan = make_resample_plan(ledgers_a, seed)
    plan_b = plan                                      # identical resamples for both methods
    deltas = []
    for va, vb in zip(plan, plan_b):
        a, b = stat(ledgers_a, va), stat(ledgers_b, vb)
        deltas.append(None if a is None or b is None else b - a)
    res = ci_from_replicates(deltas)
    ea = stat(ledgers_a, [_all(L) for L in ledgers_a])
    eb = stat(ledgers_b, [_all(L) for L in ledgers_b])
    d_hat = None if ea is None or eb is None else eb - ea
    pv, bvalid = centered_p_value(d_hat, deltas)
    res.update(delta=d_hat, seed=seed, bootstrap_seed=seed, p_value=pv,
               p_value_b_valid=bvalid,
               p_value_definition="two-sided, null-centred, +1 corrected; defined replicates only",
               confirmatory_family="NOT DEFINED (dependency D3)")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 6. Seed summaries (Section 9.4) - the seed set is supplied, never chosen here
# ══════════════════════════════════════════════════════════════════════════════
def seed_summary(values_by_seed: Dict[Any, float]) -> Dict[str, Any]:
    """mean +- SD across whatever seed set is supplied. The SD is the frozen sample
    SD (ddof = SD_DDOF = 1) and cannot be overridden; with one seed it is undefined."""
    v = np.array([float(x) for x in values_by_seed.values()])
    sd = float(np.std(v, ddof=SD_DDOF)) if len(v) > SD_DDOF else None
    return {"n_seeds": int(len(v)), "seeds": list(values_by_seed.keys()),
            "mean": float(np.mean(v)) if len(v) else None, "sd": sd, "ddof": SD_DDOF}


# ══════════════════════════════════════════════════════════════════════════════
# 7. Self-tests
# ══════════════════════════════════════════════════════════════════════════════
def _mk_ledger(td: Path, name: str, dataset: str, split: str, labels, probs, groups,
               dice=None, iou=None, hd95=None, run="run", pred_empty=None) -> Path:
    n = len(labels)
    dice = dice if dice is not None else [0.5] * n
    iou = iou if iou is not None else [0.4] * n
    hd95 = hd95 if hd95 is not None else [3.0] * n
    pe = pred_empty if pred_empty is not None else [int(math.isnan(h)) for h in hd95]
    rows = []
    for i in range(n):
        z = math.log(probs[i] / (1 - probs[i])) if 0 < probs[i] < 1 else (30.0 if probs[i] >= 1 else -30.0)
        rows.append({"record_id": f"{dataset}_{i:04d}", "dataset": dataset, "client_id": 0,
                     "split": split, "group_key": groups[i], "group_unit": "PATIENT",
                     "modality": "ultrasound", "label": int(labels[i]), "cls_logit": z,
                     "cls_prob": float(probs[i]), "pred_at_0.5": int(probs[i] >= 0.5),
                     "seg_threshold": 0.5, "dice": float(dice[i]), "iou": float(iou[i]),
                     "hd95": float(hd95[i]), "gt_fg_px": 100, "pred_fg_px": 0 if pe[i] else 100,
                     "intersection_px": 0, "pred_empty": int(pe[i]), "run_id": run,
                     "method": "SYN", "seed": 0, "selected_round": 1,
                     "selection_criterion_name": "joint_val_loss",
                     "selection_criterion_value": 0.1, "model_sha256": "0" * 64,
                     "splits_manifest_sha256": "1" * 64, "code_sha256": "2" * 64,
                     "eval_spec_version": "R4_EVALUATION_SPEC/v1"})
    p = td / f"{name}.csv"
    write_ledger(p, rows, {"split": split, "dataset": dataset})
    return p


def selftest() -> int:
    if write_ledger is None:
        print("bmc_eval.py must sit beside bmc_stats.py (it writes the synthetic ledgers)")
        return 3
    T: List[Tuple[str, str, bool, str]] = []

    def add(i, d, ok, b=""):
        T.append((i, d, bool(ok), b))

    def raises(fn) -> bool:
        try:
            fn()
            return False
        except ContractError:
            return True

    td = Path(tempfile.mkdtemp())
    # A1 exact confusion metrics: TP=3 FN=1 TN=5 FP=2
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]); p = np.array([.9, .8, .7, .1, .1, .2, .3, .2, .1, .6, .9])
    c = confusion(y, p, 0.5); m = metrics_from_counts(*c)
    add("A1", "exact confusion-matrix metrics (TP=3 FN=1 TN=5 FP=2)",
        c == (3, 1, 5, 2) and m["sensitivity"] == 0.75 and m["specificity"] == 5 / 7
        and m["balanced_accuracy"] == (0.75 + 5 / 7) / 2 and m["accuracy"] == 8 / 11 and m["f1"] == 6 / 9,
        f"sens {m['sensitivity']} spec {m['specificity']:.6f} acc {m['accuracy']:.6f} F1 {m['f1']:.6f}")
    # A2 AUC with ties, closed form 3.5/4, and against brute force with many ties
    auc = roc_auc(np.array([1, 1, 0, 0]), np.array([.8, .5, .5, .2]))
    rng = np.random.default_rng(3)
    yy = rng.integers(0, 2, 300); pp = rng.integers(0, 6, 300) / 5.0
    pos, neg = pp[yy == 1], pp[yy == 0]
    brute = float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))
    add("A2", "ROC-AUC with tied scores: 0.875 closed form, and equals brute-force pairwise",
        auc == 0.875 and abs(roc_auc(yy, pp) - brute) < 1e-12, f"{auc}; |diff| {abs(roc_auc(yy, pp) - brute):.1e}")
    # A3 macro vs micro deliberately different: client A sens 0.8 (n=10), client B 0.2 (n=100)
    yA = [1] * 5 + [0] * 5; pA = [.9] * 4 + [.1] + [.1] * 4 + [.9]
    yB = [1] * 50 + [0] * 50; pB = [.9] * 10 + [.1] * 40 + [.1] * 45 + [.9] * 5
    LA = load_ledger(_mk_ledger(td, "a", "CA", "test", yA, pA, [f"g{i}" for i in range(10)]))
    LB = load_ledger(_mk_ledger(td, "b", "CB", "test", yB, pB, [f"g{i}" for i in range(100)]))
    mac, mic = summarise([LA, LB], "sensitivity", "macro"), summarise([LA, LB], "sensitivity", "micro")
    add("A3", "macro and micro give deliberately different known results",
        mac == 0.5 and mic == 14 / 55, f"macro {mac} (=0.5), micro {mic:.6f} (=14/55)")
    add("A4", "pooled cross-client ROC-AUC is refused",
        raises(lambda: summarise([LA, LB], "roc_auc", "micro")))
    # A5 Dice/IoU aggregation
    LD1 = load_ledger(_mk_ledger(td, "d1", "D1", "test", [1, 0], [.9, .1], ["a", "b"], dice=[.2, .4], iou=[.1, .3]))
    LD2 = load_ledger(_mk_ledger(td, "d2", "D2", "test", [1, 0, 1, 0], [.9, .1, .9, .1], list("cdef"),
                                 dice=[.6, .6, .6, .6], iou=[.5, .5, .5, .5]))
    dm, dmi = summarise([LD1, LD2], "dice", "macro"), summarise([LD1, LD2], "dice", "micro")
    add("A5", "Dice/IoU: per-image -> client mean -> macro; all-image mean as micro",
        math.isclose(dm, (0.3 + 0.6) / 2) and math.isclose(dmi, (0.2 + 0.4 + 2.4) / 6)
        and math.isclose(summarise([LD1, LD2], "iou", "macro"), (0.2 + 0.5) / 2),
        f"Dice macro {dm:.4f} (=0.45), micro {dmi:.4f} (=0.5)")
    # A6 HD95 median / IQR and empty accounting
    LH = load_ledger(_mk_ledger(td, "h", "H", "test", [1, 0, 1, 0, 1], [.9, .1, .9, .1, .9], list("vwxyz"),
                                hd95=[1.0, 2.0, 3.0, 4.0, float("nan")]))
    hs = per_client_table([LH])[0]
    add("A6", "HD95 per-client median and IQR over defined values, with empty count and percentage",
        hs["hd95_median"] == 2.5 and hs["hd95_q1"] == 1.75 and hs["hd95_q3"] == 3.25
        and hs["hd95_iqr"] == 1.5 and hs["n_empty_pred"] == 1 and hs["pct_empty_pred"] == 20.0
        and hs["n_hd95_defined"] == 4,
        f"median {hs['hd95_median']}, IQR {hs['hd95_iqr']}, empty {hs['n_empty_pred']} ({hs['pct_empty_pred']}%)")
    LH2 = load_ledger(_mk_ledger(td, "h2", "H2", "test", [1, 0, 1], [.9, .1, .9], list("pqr"), hd95=[10.0, 20.0, 30.0]))
    hm, hmi = summarise([LH, LH2], "hd95", "macro"), summarise([LH, LH2], "hd95", "micro")
    add("A7", "HD95 cross-client: macro = mean of client medians; micro = median over all defined",
        hm == (2.5 + 20.0) / 2 and hmi == 4.0, f"macro {hm} (=11.25), micro {hmi} (=4.0, median of 1,2,3,4,10,20,30)")
    add("A8", "primary tau = 0.5 with rule p >= tau: p = 0.5 exactly counts as malignant",
        confusion(np.array([1, 0]), np.array([0.5, 0.5]), 0.5) == (1, 0, 0, 1))
    # B tau_val
    LV1 = load_ledger(_mk_ledger(td, "v1", "V1", "validation", [0, 1, 1, 0, 1], [.2, .4, .4, .7, .9], list("abcde")))
    cand = tau_val_candidates(LV1.cls_prob)
    at_min = confusion(LV1.label, LV1.cls_prob, cand[0]); at_inf = confusion(LV1.label, LV1.cls_prob, cand[-1])
    add("B1", "tau_val candidates: every distinct probability plus +inf; min gives all-positive, +inf all-negative",
        cand == [0.2, 0.4, 0.7, 0.9, math.inf] and at_min == (3, 0, 0, 2) and at_inf == (0, 3, 2, 0),
        f"{cand}")
    LV2 = load_ledger(_mk_ledger(td, "v2", "V2", "validation", [0, 0, 1, 1, 1, 0], [.1, .2, .6, .7, .8, .3], list("abcdef")))
    t2 = derive_tau_val([LV2])
    add("B2", "Youden J maximised: perfect separation gives J = 1 at tau = 0.6",
        t2["tau_val"] == 0.6 and t2["youden_j"] == 1.0, f"tau {t2['tau_val']}, J {t2['youden_j']}")
    LV3 = load_ledger(_mk_ledger(td, "v3", "V3", "validation", [0, 1, 0, 1], [.125, .375, .5, .5625], list("abcd")))
    t3 = derive_tau_val([LV3])
    add("B3", "tie on J: the threshold closest to 0.5 wins (0.5625 over 0.375)",
        t3["tau_val"] == 0.5625 and set(t3["tied_maximisers"]) == {0.375, 0.5625}, f"{t3['tau_val']}")
    LV4 = load_ledger(_mk_ledger(td, "v4", "V4", "validation", [0, 1, 0, 1], [.125, .25, .375, .75], list("abcd")))
    t4 = derive_tau_val([LV4])
    add("B4", "exact distance tie (0.25 and 0.75): the lower threshold wins",
        t4["tau_val"] == 0.25 and set(t4["tied_maximisers"]) == {0.25, 0.75}, f"{t4['tau_val']}")
    LT = load_ledger(_mk_ledger(td, "tt", "TT", "test", [0, 1], [.2, .8], ["a", "b"]))
    LE = load_ledger(_mk_ledger(td, "ee", "EE", EXTERNAL_SPLIT, [0, 1], [.2, .8], ["a", "b"]))
    add("B5", "tau_val derivation refuses a test ledger", raises(lambda: derive_tau_val([LV1, LT])))
    add("B6", "tau_val derivation refuses an external ledger", raises(lambda: derive_tau_val([LE])))
    add("B7", "tau_val output is frozen and labelled secondary",
        t2["frozen"] is True and t2["label"] == SECONDARY_LABEL)
    # C cluster bootstrap
    grp = [f"P{i // 3}" for i in range(30)]              # 10 patients x 3 images
    LC = load_ledger(_mk_ledger(td, "c", "CC", "test", [i % 2 for i in range(30)], [.3 + .02 * i for i in range(30)], grp))
    plan = make_resample_plan([LC], seed=11)
    grs = _group_rows(LC)
    intact = True
    for rep in plan[:300]:
        cnt = np.bincount(rep[0], minlength=LC.n)
        for g in grs:
            if len(set(cnt[g].tolist())) != 1:
                intact = False
    add("C1", "resampling unit is group_key: every image of a drawn patient comes together", intact,
        "300 replicates, 10 patients x 3 images")
    add("C2", "each replicate draws as many groups as the client has, not as many images",
        all(len(rep[0]) == 30 and len(rep[0]) % 3 == 0 for rep in plan) and len(plan) == B_ATTEMPTED)
    r = bootstrap([LC], lambda v: client_value(LC, v[0], "dice", 0.5), seed=11)
    add("C3", "exactly 2,000 attempted replicates", r["n_valid"] + r["n_undefined"] == 2000 == r["n_attempted"])
    LU = load_ledger(_mk_ledger(td, "u", "UU", "test", [0, 0, 1, 1], [.2, .3, .7, .8], ["G0", "G0", "G1", "G1"]))
    ru = bootstrap([LU], lambda v: client_value(LU, v[0], "roc_auc", 0.5), seed=5)
    add("C4", "undefined replicates are counted and never redrawn",
        ru["n_undefined"] > 500 and ru["n_valid"] + ru["n_undefined"] == 2000,
        f"n_valid {ru['n_valid']}, n_undefined {ru['n_undefined']}")
    f100 = ci_from_replicates([0.5] * 1900 + [None] * 100)["flag"]
    f101 = ci_from_replicates([0.5] * 1899 + [None] * 101)["flag"]
    add("C5", "UNSTABLE_CI when n_undefined > 100: 100 -> OK, 101 -> UNSTABLE_CI",
        f100 == "OK" and f101 == "UNSTABLE_CI", f"{f100} / {f101}")
    r1 = bootstrap([LC], lambda v: client_value(LC, v[0], "roc_auc", 0.5), seed=11)
    r2 = bootstrap([LC], lambda v: client_value(LC, v[0], "roc_auc", 0.5), seed=11)
    add("C6", "the plan is fixed by the seed: identical seed, identical interval",
        r1["ci_low"] == r2["ci_low"] and r1["ci_high"] == r2["ci_high"])
    add("C7", "fewer than 2,000 supplied replicates is refused", raises(lambda: ci_from_replicates([0.5] * 1999)))
    # D paired bootstrap. Per-record values MUST vary: with a constant metric every
    # resample has the same mean, so pairing would be invisible to the test.
    st = lambda Ls, vs: summarise(Ls, "dice", "macro", views=vs)
    dv = [0.10 + 0.02 * i for i in range(30)]
    LP1 = load_ledger(_mk_ledger(td, "pv1", "PV", "test", [i % 2 for i in range(30)],
                                 [.3 + .02 * i for i in range(30)], grp, dice=dv, run="rA"))
    LP2 = load_ledger(_mk_ledger(td, "pv2", "PV", "test", [i % 2 for i in range(30)],
                                 [.3 + .02 * i for i in range(30)], grp, dice=[d + 0.1 for d in dv], run="rB"))
    d_same = paired_bootstrap([LP1], [LP1], st, seed=7)
    add("D1", "paired engine: identical methods give delta exactly 0 in every replicate "
              "(per-record Dice varies, so unpaired resamples could not give 0)",
        d_same["ci_low"] == 0.0 and d_same["ci_high"] == 0.0 and d_same["delta"] == 0.0 and d_same["n_valid"] == 2000,
        f"CI [{d_same['ci_low']}, {d_same['ci_high']}]")
    d_shift = paired_bootstrap([LP1], [LP2], st, seed=7)
    add("D2", "paired engine: a constant per-record improvement of 0.1 gives an interval of width ~0 at 0.1",
        abs(d_shift["ci_low"] - 0.1) < 1e-9 and abs(d_shift["ci_high"] - 0.1) < 1e-9,
        f"delta {d_shift['delta']:.12f}, CI [{d_shift['ci_low']:.12f}, {d_shift['ci_high']:.12f}]")
    add("D3", "methods on different records are refused as unpairable", raises(lambda: paired_bootstrap([LP1], [LU], st, seed=7)))
    add("D4", "no confirmatory comparison family is defined (dependency D3)",
        "D3" in d_same["confirmatory_family"])
    # P: the frozen two-sided null-centred p-value
    dh = 0.25
    dstar = [0.0, 0.125, 0.25, 0.5, 0.625, 0.75, -0.125, 0.375, 0.875, 0.1875]
    pv, bv = centered_p_value(dh, dstar)
    add("P1", "centred two-sided p-value equals the hand-calculated answer 7/11",
        pv == 7 / 11 and bv == 10,
        "|d*-0.25| >= 0.25 for 6 of 10 replicates: p = (1+6)/(10+1) = 0.636364; "
        "no +1 would give 0.6, sign-proportion 0.4")
    pv0, bv0 = centered_p_value(1.0, [1.0] * 10)
    add("P2", "+1 correction: when no replicate is as extreme, p = 1/(B_valid+1), not 0",
        pv0 == 1 / 11 and bv0 == 10, f"p = {pv0}")
    add("P3", "p is never 0: a degenerate paired interval still gives p = 1/(B_valid+1) > 0",
        d_shift["p_value"] == 1 / (d_shift["p_value_b_valid"] + 1) > 0
        and d_shift["p_value_b_valid"] == 2000, f"p = {d_shift['p_value']:.6g} = 1/2001")
    pvn, bvn = centered_p_value(dh, dstar + [None, None, float("nan")])
    LQ1 = load_ledger(_mk_ledger(td, "q1", "QQ", "test", [0, 0, 1, 1], [.2, .3, .7, .8], ["G0", "G0", "G1", "G1"], run="qA"))
    LQ2 = load_ledger(_mk_ledger(td, "q2", "QQ", "test", [0, 0, 1, 1], [.1, .4, .6, .9], ["G0", "G0", "G1", "G1"], run="qB"))
    au = lambda Ls, vs: summarise(Ls, "roc_auc", "macro", views=vs)
    # an exception inside the paired run is recorded as a FAIL of P4/P5/S1, not allowed
    # to abort the suite and hide the results of every other test
    try:
        dq = paired_bootstrap([LQ1], [LQ2], au)
        dq_err = ""
    except Exception as exc:  # noqa: BLE001
        dq, dq_err = None, f"{type(exc).__name__}: {exc}"
    add("P4", "undefined paired replicates are excluded from B_valid and never redrawn",
        dq is not None and pvn == 7 / 11 and bvn == 10 and dq["p_value_b_valid"] == dq["n_valid"] < 2000
        and dq["n_valid"] + dq["n_undefined"] == 2000 and dq["p_value"] is not None,
        dq_err or f"3 undefined added: B_valid stays 10; paired AUC: n_valid {dq['n_valid']}, "
        f"n_undefined {dq['n_undefined']}, B_valid {dq['p_value_b_valid']}")
    add("P5", "an UNSTABLE_CI flag is retained and reported with the comparison",
        dq is not None and dq["flag"] == "UNSTABLE_CI" and dq["p_value"] is not None,
        dq_err or f"flag {dq['flag']}, p {dq['p_value']:.4f}")
    rb = bootstrap([LC], lambda v: client_value(LC, v[0], "dice", 0.5))
    add("S1", "BOOTSTRAP_SEED = 20260930 is the default and is recorded in bootstrap and paired outputs",
        BOOTSTRAP_SEED == 20260930 and rb["bootstrap_seed"] == 20260930
        and dq is not None and dq["bootstrap_seed"] == 20260930 and BOOTSTRAP_SEED != 20260929,
        f"bootstrap {rb['bootstrap_seed']}, paired {dq['bootstrap_seed'] if dq else 'n/a'}; "
        "split seed 20260929 is distinct")
    # E external
    LE2 = load_ledger(_mk_ledger(td, "e2", "E2", EXTERNAL_SPLIT, [0, 1], [.2, .8], ["a", "b"]))
    add("E1", "external datasets stay separate: no pooling across externals or with internal data",
        raises(lambda: summarise([LE, LE2], "dice", "macro")) and raises(lambda: summarise([LA, LE], "dice", "micro"))
        and summarise([LE], "dice", "macro") == 0.5)
    add("E2", "an external ledger cannot enter an internal analysis set",
        raises(lambda: assert_internal([LA, LE], "test")))
    # F integrity
    tp = _mk_ledger(td, "tamper", "TP", "test", [0, 1], [.2, .8], ["a", "b"])
    txt = tp.read_text(encoding="utf-8").replace("0.8,1", "0.9,1", 1)
    tp.write_text(txt, encoding="utf-8")
    add("F1", "a tampered ledger is detected by its SHA-256", raises(lambda: load_ledger(tp)))
    bad = _mk_ledger(td, "badsplit", "BS", "test", [0, 1], [.2, .8], ["a", "b"])
    rows = list(csv.DictReader(bad.open(encoding="utf-8")))
    for rr in rows:
        rr["split"] = "train"
    bad2 = td / "train_split.csv"
    write_ledger(bad2, [{k: (float(v) if k in ("cls_logit", "cls_prob", "dice", "iou", "hd95",
                                                "seg_threshold", "selection_criterion_value") and v != ""
                             else v) for k, v in rr.items()} for rr in rows], {"split": "train"})
    add("F2", "a split outside the analysable roles is refused", raises(lambda: load_ledger(bad2)))
    before = {q.name: (_sha256(q), q.stat().st_mtime_ns) for q in td.glob("*.csv") if q.name != "tamper.csv"}
    for q in td.glob("*.csv"):
        if q.name not in ("tamper.csv",):
            try:
                L = load_ledger(q)
                per_client_table([L])
            except ContractError:
                pass
    after = {q.name: (_sha256(q), q.stat().st_mtime_ns) for q in td.glob("*.csv") if q.name != "tamper.csv"}
    add("F3", "no ledger is modified by the analysis (hash and mtime unchanged)", before == after,
        f"{len(before)} ledgers")
    # G seeds
    s3 = seed_summary({101: 0.70, 202: 0.72, 303: 0.74})
    add("G1", "frozen sample SD (ddof = 1): 0.70, 0.72, 0.74 give 0.72 +- 0.02 (ddof = 0 would give 0.01633)",
        SD_DDOF == 1 and s3["ddof"] == 1 and math.isclose(s3["mean"], 0.72)
        and math.isclose(s3["sd"], 0.02, rel_tol=0, abs_tol=1e-12),
        f"{s3['mean']:.4f} +- {s3['sd']:.5f}")
    counts = [seed_summary({i: 0.5 + 0.01 * i for i in range(k)})["n_seeds"] for k in (2, 3, 5, 7)]
    try:
        seed_summary({1: 0.5, 2: 0.6}, ddof=0)
        overridable = True
    except TypeError:
        overridable = False
    train_seed_consts = [n for n in dir(sys.modules[__name__]) if "SEED" in n.upper()
                         and not n.startswith("_") and n != "BOOTSTRAP_SEED"
                         and not callable(getattr(sys.modules[__name__], n))]
    add("G2", "any supplied seed set is accepted; no training seed count or values are chosen here; "
              "ddof cannot be overridden",
        counts == [2, 3, 5, 7] and not overridable and not train_seed_consts
        and seed_summary({1: 0.5})["sd"] is None,
        f"accepted n = {counts}; ddof override refused; training seed constants: "
        f"{train_seed_consts or 'none'}")

    print("=" * 96)
    print("bmc_stats.py SELF-TESTS  (R4_EVALUATION_SPEC/v1, Section 10.2)")
    print("=" * 96)
    for i, dsc, ok, b in T:
        print(f"  {i:3s} [{'PASS' if ok else 'FAIL'}]  {dsc}" + (f"\n        {b}" if b else ""))
    fails = [t for t in T if not t[2]]
    print("=" * 96)
    print(f"PASS = {len(T) - len(fails)}   FAIL = {len(fails)}")
    print("BMC_STATS_VERIFIED" if not fails else "BMC_STATS_BLOCKED")
    return 0 if not fails else 4


# ══════════════════════════════════════════════════════════════════════════════
# ADDITIVE (R5.1 decision D3 = O1): multi-seed paired cluster bootstrap.
# Nothing above this block is modified; this function only calls existing ones.
# ══════════════════════════════════════════════════════════════════════════════
MULTI_RUN_SCOPE = ("conditional on the realised trained model instances (one per training seed); "
                   "the interval reflects patient/group resampling and does NOT capture variability "
                   "over possible future training runs")


def paired_bootstrap_multiseed(a_by_seed: Dict[Any, Sequence[Ledger]], b_by_seed: Dict[Any, Sequence[Ledger]],
                               stat: Callable[[Sequence[Ledger], Sequence[np.ndarray]], Optional[float]],
                               seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Delta_bar = (1/S) sum_s [stat(B_s) - stat(A_s)], estimated with ONE shared group_key
    resampling plan per replicate across both methods and all S seeds. The seed average is
    formed inside every replicate; a replicate is undefined if any seed's difference is.
    Frozen centred two-sided p-value with +1 correction (centered_p_value)."""
    seeds = sorted(a_by_seed)
    if not seeds or sorted(b_by_seed) != seeds:
        raise ContractError("both methods must supply ledgers for the same training seeds")
    ref = list(a_by_seed[seeds[0]])
    for s_ in seeds:
        for side in (a_by_seed[s_], b_by_seed[s_]):
            if len(side) != len(ref):
                raise ContractError(f"seed {s_}: ledger count differs")
            for L, R in zip(side, ref):
                if (L.dataset, L.split) != (R.dataset, R.split) or not np.array_equal(L.record_id, R.record_id) \
                        or not np.array_equal(L.group_key, R.group_key):
                    raise ContractError(f"seed {s_}, {L.dataset}: records or groups differ, so the seeds cannot share one plan")
    plan = make_resample_plan(ref, seed)                  # ONE plan: all seeds, both methods
    deltas: List[Optional[float]] = []
    for view in plan:
        per = []
        for s_ in seeds:
            a, b = stat(a_by_seed[s_], view), stat(b_by_seed[s_], view)
            if a is None or b is None:
                per = None
                break
            per.append(b - a)
        deltas.append(None if per is None else float(np.mean(per)))
    res = ci_from_replicates(deltas)
    full = [_all(L) for L in ref]
    per_seed = {}
    for s_ in seeds:
        ea, eb = stat(a_by_seed[s_], full), stat(b_by_seed[s_], full)
        per_seed[s_] = None if ea is None or eb is None else eb - ea
    d_bar = None if any(v is None for v in per_seed.values()) else float(np.mean(list(per_seed.values())))
    pv, bvalid = centered_p_value(d_bar, deltas)
    res.update(delta_bar=d_bar, per_seed_delta=per_seed, seeds=seeds, p_value=pv, p_value_b_valid=bvalid,
               bootstrap_seed=seed, estimand="Delta_bar = mean over training seeds of paired differences (O1)",
               scope=MULTI_RUN_SCOPE,
               seed_variability_descriptive=(seed_summary(per_seed) if d_bar is not None else None),
               confirmatory_family="R5.1: six-test Holm family (applied outside this function)")
    return res


def selftest_multiseed() -> int:
    if write_ledger is None:
        print("bmc_eval.py must sit beside bmc_stats.py"); return 3
    T: List[Tuple[str, str, bool, str]] = []
    add = lambda i, d, ok, b="": T.append((i, d, bool(ok), b))
    td = Path(tempfile.mkdtemp())
    grp = [f"P{i // 3}" for i in range(30)]
    lab = [i % 2 for i in range(30)]
    prob = [.3 + .02 * i for i in range(30)]
    dv = [0.10 + 0.02 * i for i in range(30)]
    e = [0.01 * ((i * 7) % 11 - 5) for i in range(30)]           # varies per record
    mk = lambda name, dice, run: load_ledger(_mk_ledger(td, name, "MS", "test", lab, prob, grp, dice=dice, run=run))
    st = lambda Ls, vs: summarise(Ls, "dice", "macro", views=vs)
    A = {s_: [mk(f"a{s_}", dv, f"A{s_}")] for s_ in (101, 202, 303)}
    # K1 three identical seeds reproduce the single-seed engine exactly
    # per-record differences whose GROUP means vary, so the interval is not degenerate
    Bs = {s_: [mk(f"b{s_}", [d + x + 0.01 for d, x in zip(dv, e)], f"B{s_}")] for s_ in (101, 202, 303)}
    one = paired_bootstrap(A[101], Bs[101], st)
    multi = paired_bootstrap_multiseed({s_: A[101] for s_ in A}, {s_: Bs[101] for s_ in A}, st)
    add("K1", "three identical seeds reproduce the locked single-seed paired engine exactly (non-degenerate case)",
        (one["ci_low"], one["ci_high"], one["p_value"], one["n_valid"]) ==
        (multi["ci_low"], multi["ci_high"], multi["p_value"], multi["n_valid"]) and multi["ci_high"] - multi["ci_low"] > 1e-4,
        f"CI [{multi['ci_low']:.6f}, {multi['ci_high']:.6f}], p {multi['p_value']:.6f}")
    # K2 constant per-seed shifts 0.1, 0.2, 0.3 -> Delta_bar = 0.2 in every replicate
    B2 = {s_: [mk(f"k2{s_}", [d + sh for d in dv], f"K2{s_}")] for s_, sh in ((101, .1), (202, .2), (303, .3))}
    r2 = paired_bootstrap_multiseed(A, B2, st)
    add("K2", "per-seed shifts 0.1/0.2/0.3: Delta_bar = 0.2, degenerate CI at 0.2, p = 1/2001 (+1 correction)",
        abs(r2["delta_bar"] - 0.2) < 1e-12 and abs(r2["ci_low"] - 0.2) < 1e-9 and abs(r2["ci_high"] - 0.2) < 1e-9
        and r2["p_value"] == 1 / 2001, f"Delta_bar {r2['delta_bar']:.12f}, p {r2['p_value']:.6g}")
    # K3 ONE plan shared across seeds: B_1 = d+e, B_2 = d-e, B_3 = d  ->  Delta_bar* = 0 exactly
    B3 = {101: [mk("k3a", [d + x for d, x in zip(dv, e)], "K3a")], 202: [mk("k3b", [d - x for d, x in zip(dv, e)], "K3b")],
          303: [mk("k3c", dv, "K3c")]}
    r3 = paired_bootstrap_multiseed(A, B3, st)
    add("K3", "one group plan is shared across seeds: seed effects +e, -e, 0 cancel in EVERY replicate",
        abs(r3["ci_low"]) < 1e-12 and abs(r3["ci_high"]) < 1e-12 and r3["n_valid"] == 2000,
        f"CI [{r3['ci_low']:.2e}, {r3['ci_high']:.2e}] (independent per-seed plans would spread)")
    # K4 the p-value is the frozen centred p of the seed-averaged replicate distribution
    plan = make_resample_plan(A[101], BOOTSTRAP_SEED)
    manual = [float(np.mean([st(Bs[s_], v) - st(A[s_], v) for s_ in (101, 202, 303)])) for v in plan]
    # zero-centred, seed-specific differences: Delta_bar is small relative to the replicate
    # spread, so p is far from its minimum and differs from any average of per-seed p-values
    B4 = {s_: [mk(f"k4{s_}", [d + 0.02 * (((i * (j + 3)) % 7) - 3) + 0.002 for i, d in enumerate(dv)], f"K4{s_}")]
          for j, s_ in enumerate((101, 202, 303))}
    r4 = paired_bootstrap_multiseed(A, B4, st)
    manual4 = [float(np.mean([st(B4[s_], v) - st(A[s_], v) for s_ in (101, 202, 303)])) for v in plan]
    exp_p, _ = centered_p_value(r4["delta_bar"], manual4)
    per_seed_p = [paired_bootstrap(A[s_], B4[s_], st)["p_value"] for s_ in (101, 202, 303)]
    add("K4", "p equals the centred +1 p-value computed on the seed-averaged replicates (not an average of per-seed p)",
        r4["p_value"] == exp_p and 0.01 < exp_p < 0.99 and abs(float(np.mean(per_seed_p)) - exp_p) > 1e-3,
        f"p {r4['p_value']:.6f} = recomputed {exp_p:.6f}; mean of per-seed p would be {float(np.mean(per_seed_p)):.6f}")
    # K5 a replicate is undefined when ANY seed is undefined, never redrawn
    def st5(Ls, vs):
        if Ls[0].run_id == "K5u" and "P0" not in set(Ls[0].group_key[vs[0]].tolist()):
            return None
        return st(Ls, vs)
    B5 = {101: [mk("k5a", dv, "K5a")], 202: [mk("k5u", dv, "K5u")], 303: [mk("k5c", dv, "K5c")]}
    r5 = paired_bootstrap_multiseed(A, B5, st5)
    exp_undef = sum(1 for v in plan if "P0" not in set(A[101][0].group_key[v[0]].tolist()))
    add("K5", "a replicate is undefined if any seed is undefined; counted, never redrawn; B_valid = n_valid",
        r5["n_undefined"] == exp_undef > 0 and r5["n_valid"] + r5["n_undefined"] == 2000 and r5["p_value_b_valid"] == r5["n_valid"],
        f"n_undefined {r5['n_undefined']} = expected {exp_undef}")
    # K6 refusals
    bad = 0
    for fn in (lambda: paired_bootstrap_multiseed(A, {101: B2[101]}, st),
               lambda: paired_bootstrap_multiseed(A, {s_: [load_ledger(_mk_ledger(td, f"x{s_}", "XX", "test", lab, prob, grp))] for s_ in A}, st)):
        try:
            fn()
        except ContractError:
            bad += 1
    add("K6", "mismatched seed sets and mismatched records are refused", bad == 2)
    # K7 descriptive seed variability uses the frozen sample SD (ddof = 1)
    sv = r2["seed_variability_descriptive"]
    add("K7", "descriptive per-seed variability: mean 0.2, sample SD 0.1 (ddof = 1)",
        abs(sv["mean"] - 0.2) < 1e-12 and abs(sv["sd"] - 0.1) < 1e-12 and sv["ddof"] == 1, f"{sv['mean']:.3f} +- {sv['sd']:.3f}")
    # K8 provenance and scope
    add("K8", "BOOTSTRAP_SEED 20260930 recorded; conditional scope stated in the output",
        r2["bootstrap_seed"] == 20260930 and "does NOT capture variability over possible future training runs" in r2["scope"])
    print("=" * 96); print("bmc_stats.py MULTI-SEED SELF-TESTS (additive, R5.1 D3 = O1)"); print("=" * 96)
    for i, dsc, ok, b in T:
        print(f"  {i:3s} [{'PASS' if ok else 'FAIL'}]  {dsc}" + (f"\n        {b}" if b else ""))
    nf = sum(1 for t in T if not t[2])
    print("=" * 96); print(f"PASS = {len(T) - nf}   FAIL = {nf}"); print("BMC_STATS_MULTISEED_VERIFIED" if not nf else "BMC_STATS_MULTISEED_BLOCKED")
    return 0 if not nf else 4


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="R4 analysis and statistics module.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-multiseed", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.selftest_multiseed:
        return selftest_multiseed()
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
