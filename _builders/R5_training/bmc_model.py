#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bmc_model.py - R5 architecture (R5 matrix v2, section A)

  * shared ResNet-34 encoder (1 input channel), two modality-specific U-Net decoders
    with segmentation heads, one shared classification head;
  * every output is a RAW LOGIT: no sigmoid exists anywhere in the model (R4 D1);
  * forward(x, modality) routes each sample by its own modality code
    (0 = ultrasound, 1 = mammography), never by the first sample of the batch;
  * parameter groups are structural (module-name prefix), never tensor indices;
  * ImageNet initialisation converts the first convolution explicitly:
    W1 = sum over RGB of W[:, c]  (equivalent to feeding [x, x, x]).
No trained checkpoint of any kind is ever loaded here.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Dict, List, Optional

import torch
import torch.nn as nn

US, MM = 0, 1
DECODER_CHANNELS = (256, 128, 64, 32, 16)
GROUP_PREFIXES = (("encoder.", "global"), ("cls_head.", "global"),
                  ("us_decoder.", "us"), ("us_seg_head.", "us"),
                  ("mm_decoder.", "mm"), ("mm_seg_head.", "mm"))
BN_MOMENTUM = 0.1


def group_of(name: str) -> str:
    """Structural group of a parameter or buffer name. Unmatched names abort."""
    for prefix, g in GROUP_PREFIXES:
        if name.startswith(prefix):
            return g
    raise KeyError(f"{name!r} matches no structural group prefix")


def is_excluded_buffer(name: str, tensor: torch.Tensor) -> bool:
    """Integer buffers (num_batches_tracked) are never averaged."""
    return not torch.is_floating_point(tensor)


def convert_first_conv(w_rgb: torch.Tensor) -> torch.Tensor:
    """Frozen conversion (A2): sum over the RGB input channels."""
    if w_rgb.dim() != 4 or w_rgb.shape[1] != 3:
        raise ValueError(f"expected an RGB first-layer weight [out,3,k,k], got {tuple(w_rgb.shape)}")
    return w_rgb.sum(dim=1, keepdim=True)


def state_sha256(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fetch_imagenet_rgb_encoder_state() -> Dict[str, torch.Tensor]:
    """Public ImageNet ResNet-34 weights, loaded as a 3-channel encoder. The single-
    channel conversion is then done by convert_first_conv, not by the library."""
    import segmentation_models_pytorch as smp
    enc = smp.encoders.get_encoder("resnet34", in_channels=3, depth=5, weights="imagenet")
    return {k: v.detach().clone() for k, v in enc.state_dict().items()}


def decoder_takes_varargs(dec: nn.Module) -> bool:
    """smp's UnetDecoder API changed: older releases define forward(self, *features),
    0.5.x defines forward(self, features: List). Detected from the signature, not the
    version string, so both receive the SAME full encoder feature list."""
    return any(p.kind is inspect.Parameter.VAR_POSITIONAL
               for p in inspect.signature(dec.forward).parameters.values())


def call_decoder(dec: nn.Module, feats: List[torch.Tensor]) -> torch.Tensor:
    return dec(*feats) if decoder_takes_varargs(dec) else dec(feats)


def _seg_head() -> nn.Module:
    return nn.Conv2d(DECODER_CHANNELS[-1], 1, kernel_size=3, padding=1)   # no activation


class BMCModel(nn.Module):
    def __init__(self, rgb_encoder_state: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        import segmentation_models_pytorch as smp
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
        self.encoder = smp.encoders.get_encoder("resnet34", in_channels=1, depth=5, weights=None)
        ch = self.encoder.out_channels
        self.us_decoder = UnetDecoder(encoder_channels=ch, decoder_channels=DECODER_CHANNELS, n_blocks=5)
        self.mm_decoder = UnetDecoder(encoder_channels=ch, decoder_channels=DECODER_CHANNELS, n_blocks=5)
        self.us_seg_head = _seg_head()
        self.mm_seg_head = _seg_head()
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch[-1], 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 1))                                              # raw logit
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = BN_MOMENTUM
        self.pretrained_sha256 = None
        if rgb_encoder_state is not None:
            self.load_rgb_encoder(rgb_encoder_state)

    def load_rgb_encoder(self, rgb: Dict[str, torch.Tensor]) -> None:
        own = self.encoder.state_dict()
        new = {}
        for k, v in own.items():
            if k not in rgb:
                raise KeyError(f"pretrained encoder state lacks {k!r}")
            new[k] = convert_first_conv(rgb[k]) if k == "conv1.weight" else rgb[k].clone()
        self.encoder.load_state_dict(new, strict=True)
        self.pretrained_sha256 = state_sha256(rgb)

    def forward(self, x: torch.Tensor, modality: torch.Tensor):
        if modality.dim() != 1 or modality.numel() != x.shape[0]:
            raise ValueError("modality must be a per-sample vector of length batch")
        if bool((~((modality == US) | (modality == MM))).any()):
            raise ValueError("modality codes must be 0 (ultrasound) or 1 (mammography)")
        feats: List[torch.Tensor] = self.encoder(x)
        cls = self.cls_head(feats[-1]).squeeze(1)
        seg = None
        for code, dec, head in ((US, self.us_decoder, self.us_seg_head),
                                (MM, self.mm_decoder, self.mm_seg_head)):
            idx = (modality == code).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            out = head(call_decoder(dec, [f.index_select(0, idx) for f in feats]))
            if seg is None:
                seg = out.new_empty((x.shape[0],) + tuple(out.shape[1:]))
            seg.index_copy_(0, idx, out)
        return seg, cls


def structural_groups(model: nn.Module) -> Dict[str, str]:
    """Every state entry -> 'global' | 'us' | 'mm' | 'excluded' (integer buffers)."""
    out = {}
    for k, v in model.state_dict().items():
        g = group_of(k)
        out[k] = "excluded" if is_excluded_buffer(k, v) else g
    return out
