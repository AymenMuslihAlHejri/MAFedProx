#!/usr/bin/env python3
"""bmc_sampling.py - class-balanced TRAINING sampling only (R3; matrix v2 J, H3, D11).

  per client : WeightedRandomSampler, w_i = 1/n_{y_i}, replacement, num_samples = N_k,
               BatchSampler(batch 16, drop_last=True) -> D_k = 16*floor(N_k/16) draws
  centralised: exactly D_k draws from EACH client's own sampler, concatenated,
               shuffled, batched by 16 (sum D_k is a multiple of 16, so batches are full)
  evaluation : never weighted; sequential order.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import torch
from torch.utils.data import BatchSampler, SequentialSampler, WeightedRandomSampler

BATCH = 16


def class_balanced_weights(labels: Sequence[int]) -> torch.Tensor:
    y = torch.as_tensor(list(labels), dtype=torch.long)
    counts = torch.bincount(y, minlength=2).double()
    if (counts == 0).any():
        raise ValueError("a training partition must contain both classes")
    return (1.0 / counts)[y]


def client_train_batches(labels: Sequence[int], generator: torch.Generator) -> List[List[int]]:
    ws = WeightedRandomSampler(class_balanced_weights(labels), num_samples=len(labels),
                               replacement=True, generator=generator)
    return [b for b in BatchSampler(ws, batch_size=BATCH, drop_last=True)]


def draws_per_epoch(n: int) -> int:
    return BATCH * (n // BATCH)


def centralised_train_batches(client_labels: Dict[str, Sequence[int]], generator: torch.Generator
                              ) -> List[List[Tuple[str, int]]]:
    pool: List[Tuple[str, int]] = []
    for k in sorted(client_labels):
        lab = client_labels[k]
        ws = WeightedRandomSampler(class_balanced_weights(lab), num_samples=len(lab),
                                   replacement=True, generator=generator)
        idx = list(ws)[: draws_per_epoch(len(lab))]
        pool.extend((k, i) for i in idx)
    perm = torch.randperm(len(pool), generator=generator).tolist()
    pool = [pool[i] for i in perm]
    return [pool[i:i + BATCH] for i in range(0, len(pool), BATCH)]


def eval_order(n: int) -> List[int]:
    return list(SequentialSampler(range(n)))
