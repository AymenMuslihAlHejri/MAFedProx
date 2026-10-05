#!/usr/bin/env python3
"""bmc_schedule.py - exposure-indexed cosine schedule (matrix v2, D6). A pure function
of the evaluation-block index, computed by the server, so no optimiser/scheduler object
can silently disable it (R0 F11). Every method uses the same 50-value sequence."""
from __future__ import annotations
import math

LR0, LR_MIN, T_EPOCH_EQ, BLOCK = 1e-4, 1e-6, 100, 2
N_BLOCKS = T_EPOCH_EQ // BLOCK


def lr_at(block: int, lr0: float = LR0, lr_min: float = LR_MIN, T: int = T_EPOCH_EQ, b: int = BLOCK) -> float:
    if not 0 <= block < T // b:
        raise ValueError(f"block {block} outside 0..{T // b - 1}")
    t = b * block
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * t / T))


def lr_sequence() -> list:
    return [lr_at(k) for k in range(N_BLOCKS)]
