#!/usr/bin/env python3
"""bmc_select.py - validation-only checkpoint selection (matrix v2, C; completes R4 10.3)
and validation-only FedProx mu selection (F2).

The selector reads ONLY per-evaluation validation criterion values. It refuses any entry
whose split is not 'validation'. argmin; exact ties -> earliest evaluation point.
The frozen record uses the schema bmc_eval.py checks before any test/external ledger.
"""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

CRITERION = "joint_val_loss_client_macro"
MU_GRID = (0.001, 0.01, 0.1, 1.0)


class SelectionError(ValueError):
    pass


def client_macro(per_client_losses: Dict[str, float]) -> float:
    """C3: unweighted mean of the per-client mean validation losses."""
    if not per_client_losses:
        raise SelectionError("no client losses")
    return float(sum(per_client_losses.values()) / len(per_client_losses))


def select(history: Sequence[dict]) -> Tuple[int, float]:
    for h in history:
        if h.get("split") != "validation":
            raise SelectionError(f"selection may read validation only; got split={h.get('split')!r}")
    best_i, best_v = None, None
    for h in history:                                   # strict '<' keeps the earliest tie
        if best_v is None or h["criterion"] < best_v:
            best_i, best_v = int(h["eval_index"]), float(h["criterion"])
    if best_i is None:
        raise SelectionError("empty history")
    return best_i, best_v


def history_sha256(history: Sequence[dict]) -> str:
    return hashlib.sha256(json.dumps(list(history), sort_keys=True).encode()).hexdigest()


def freeze_record(path: Path, *, run_id: str, method: str, seed: int, history: Sequence[dict],
                  best_so_far_index: int, checkpoint_sha256: str) -> dict:
    idx, val = select(history)
    if idx != best_so_far_index:
        raise SelectionError(f"independent selection ({idx}) disagrees with the training loop ({best_so_far_index})")
    rec = {"frozen": True, "selection_split": "validation", "selected_round": idx,
           "selection_criterion_name": CRITERION, "selection_criterion_value": val,
           "run_id": run_id, "method": method, "seed": seed,
           "history_sha256": history_sha256(history), "checkpoint_sha256": checkpoint_sha256,
           "created_at": datetime.now(timezone.utc).isoformat()}
    with Path(path).open("x", encoding="utf-8") as fh:  # write-once
        json.dump(rec, fh, indent=2)
    return rec


def select_mu(criterion_by_mu: Dict[float, float]) -> float:
    """F2: smallest C3 validation criterion over the FedProx grid; tie -> smaller mu."""
    if set(criterion_by_mu) != set(MU_GRID):
        raise SelectionError(f"mu selection needs exactly the grid {MU_GRID}")
    return min(criterion_by_mu, key=lambda m: (criterion_by_mu[m], m))
