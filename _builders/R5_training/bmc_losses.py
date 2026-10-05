#!/usr/bin/env python3
"""bmc_losses.py - R5 joint loss (matrix v2, section B). One implementation for every method.

  l_cls,i = BCEWithLogits(z_i, y_i)                                  (unweighted)
  l_seg,i = 0.5 * (1 - D_i) + 0.5 * mean_pixels BCEWithLogits(S_i, G_i)
  D_i     = (2 sum sigma(S_i) G_i + eps) / (sum sigma(S_i) + sum G_i + eps),  eps = 1
  l_i     = lambda_seg * l_seg,i + lambda_cls * l_cls,i ;  L = mean_i l_i
  FedProx : L_local = L + (mu/2) * sum_{theta trainable} ||theta - theta_g||^2
All accumulation in float32.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

DICE_EPS = 1.0


def soft_dice_per_image(seg_logits, gt):
    p = torch.sigmoid(seg_logits.float()); g = gt.float()
    dims = tuple(range(1, p.dim()))
    return (2 * (p * g).sum(dims) + DICE_EPS) / (p.sum(dims) + g.sum(dims) + DICE_EPS)


def seg_loss_per_image(seg_logits, gt):
    dims = tuple(range(1, seg_logits.dim()))
    bce = F.binary_cross_entropy_with_logits(seg_logits.float(), gt.float(), reduction="none").mean(dims)
    return 0.5 * (1.0 - soft_dice_per_image(seg_logits, gt)) + 0.5 * bce


def cls_loss_per_image(cls_logits, y):
    return F.binary_cross_entropy_with_logits(cls_logits.float(), y.float(), reduction="none")


def joint_loss_per_image(seg_logits, gt, cls_logits, y, lam_seg: float = 1.0, lam_cls: float = 1.0):
    out = torch.zeros(y.shape[0], dtype=torch.float32, device=y.device)
    if lam_seg:
        out = out + lam_seg * seg_loss_per_image(seg_logits, gt)
    if lam_cls:
        out = out + lam_cls * cls_loss_per_image(cls_logits, y)
    return out


def joint_loss(seg_logits, gt, cls_logits, y, lam_seg: float = 1.0, lam_cls: float = 1.0):
    return joint_loss_per_image(seg_logits, gt, cls_logits, y, lam_seg, lam_cls).mean()


def prox_term(model, ref_params: dict, mu: float):
    """(mu/2) * sum over TRAINABLE parameters; buffers are excluded by construction."""
    if mu == 0:
        return torch.zeros((), dtype=torch.float32)
    tot = None
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        d = ((p.float() - ref_params[n].float()) ** 2).sum()
        tot = d if tot is None else tot + d
    return 0.5 * mu * tot
