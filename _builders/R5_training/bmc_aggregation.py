#!/usr/bin/env python3
"""bmc_aggregation.py - server aggregation (matrix v2, sections E, F, G, A7).

Every floating-point state entry (parameters AND BatchNorm running_mean/running_var)
is aggregated by the rule of its structural group, over that group's contributors:
  global (encoder., cls_head.) : all clients
  us     (us_decoder., us_seg_head.) : ultrasound clients only
  mm     (mm_decoder., mm_seg_head.) : mammography clients only
Integer buffers (num_batches_tracked) are never averaged: the server value is kept.

  fedavg / fedprox : update-aware, n_k-weighted over the group's contributors
  proposed         : us/mm groups -> within-modality weighted mean A_g;
                     global group -> alpha*A_US + (1-alpha)*A_MM;
                     within-group weights n_k-proportional, capped at c by
                     WATER-FILLING (never exceeds c after renormalisation), or uncapped (N2)
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch

from bmc_model import US, MM, group_of, is_excluded_buffer


def waterfill(n: Dict[str, float], cap: float) -> Dict[str, float]:
    if cap * len(n) < 1.0 - 1e-12:
        raise ValueError(f"cap {cap} infeasible for {len(n)} clients")
    tot = float(sum(n.values()))
    w = {k: v / tot for k, v in n.items()}
    fixed: set = set()
    while True:
        over = [k for k in w if k not in fixed and w[k] > cap + 1e-15]
        if not over:
            return w
        for k in over:
            w[k] = cap
            fixed.add(k)
        free = [k for k in w if k not in fixed]
        rem = 1.0 - cap * len(fixed)
        fs = float(sum(n[k] for k in free))
        for k in free:
            w[k] = rem * n[k] / fs


def _proportional(n: Dict[str, float]) -> Dict[str, float]:
    t = float(sum(n.values()))
    return {k: v / t for k, v in n.items()}


def _wmean(states, weights: Dict[str, float], key: str) -> torch.Tensor:
    acc = None
    for k, w in weights.items():
        t = states[k][key].double() * w
        acc = t if acc is None else acc + t
    return acc


def aggregate(method: str, server_state: Dict[str, torch.Tensor],
              client_states: Dict[str, Dict[str, torch.Tensor]],
              client_meta: Dict[str, Tuple[int, int]], *, cap: float = 0.4,
              alpha: float = 0.5, capped: bool = True) -> Dict[str, torch.Tensor]:
    """client_meta[k] = (modality code, n_train). Returns the new server state."""
    if method not in ("fedavg", "fedprox", "proposed"):
        raise ValueError(method)
    us = {k: float(m[1]) for k, m in client_meta.items() if m[0] == US}
    mm = {k: float(m[1]) for k, m in client_meta.items() if m[0] == MM}
    allc = {**us, **mm}
    if method == "proposed":
        wf = (lambda d: waterfill(d, cap)) if capped else _proportional
        w_us, w_mm = wf(us), wf(mm)
    new = {}
    for key, ref in server_state.items():
        if is_excluded_buffer(key, ref):
            new[key] = ref.clone()                       # never averaged
            continue
        g = group_of(key)
        if method in ("fedavg", "fedprox"):
            contrib = allc if g == "global" else (us if g == "us" else mm)
            val = _wmean(client_states, _proportional(contrib), key)
        else:
            if g == "us":
                val = _wmean(client_states, w_us, key)
            elif g == "mm":
                val = _wmean(client_states, w_mm, key)
            else:
                val = alpha * _wmean(client_states, w_us, key) + (1 - alpha) * _wmean(client_states, w_mm, key)
        new[key] = val.to(ref.dtype)
    return new
