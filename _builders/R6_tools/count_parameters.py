#!/usr/bin/env python3
"""count_parameters.py - parameter counts and architecture facts of the FROZEN R5 model.

Place in  _builders\\R6_tools\\  (NOT inside R5_training). It:
  * imports bmc_model.py from ..\\R5_training without writing bytecode there, so the
    frozen folder is not touched;
  * runs on the CPU, uses no study data, downloads nothing, trains nothing;
  * reports parameter counts per structural group, buffer counts, the per-client trainable
    set, decoder block structure, encoder feature shapes and output shapes;
  * prints the SHA-256 of bmc_model.py so the counts can be tied to the run manifests
    (expected 2cce9dd1a79566c1c08d0540df4c9de598e41fe2b374921a5274550ecad345b2);
  * writes the report to parameter_count_report.json beside this script.

Parameter COUNTS do not depend on the weight values, so a randomly initialised model is
used; the ImageNet weights would change values, not counts.

    python count_parameters.py
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

sys.dont_write_bytecode = True                         # leave R5_training untouched
HERE = Path(__file__).resolve().parent
R5 = HERE.parent / "R5_training"
if not (R5 / "bmc_model.py").is_file():
    raise SystemExit(f"bmc_model.py not found in {R5}")
sys.path.insert(0, str(R5))

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from bmc_model import BMCModel, GROUP_PREFIXES, US, MM, decoder_takes_varargs

EXPECTED_MODEL_SHA = "2cce9dd1a79566c1c08d0540df4c9de598e41fe2b374921a5274550ecad345b2"


def n_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def main() -> int:
    model_sha = hashlib.sha256((R5 / "bmc_model.py").read_bytes()).hexdigest()
    torch.manual_seed(0)
    m = BMCModel().eval()                              # counts only; no download
    groups = {
        "encoder (shared)": m.encoder,
        "us_decoder": m.us_decoder, "us_seg_head": m.us_seg_head,
        "mm_decoder": m.mm_decoder, "mm_seg_head": m.mm_seg_head,
        "cls_head (shared)": m.cls_head,
    }
    counts = {k: n_params(v) for k, v in groups.items()}
    total = n_params(m)
    trainable = n_params(m, trainable_only=True)
    float_buffers = sum(b.numel() for b in m.buffers() if torch.is_floating_point(b))
    int_buffers = sum(1 for b in m.buffers() if not torch.is_floating_point(b))
    us_client = counts["encoder (shared)"] + counts["us_decoder"] + counts["us_seg_head"] + counts["cls_head (shared)"]
    mm_client = counts["encoder (shared)"] + counts["mm_decoder"] + counts["mm_seg_head"] + counts["cls_head (shared)"]
    # every parameter must belong to exactly one structural prefix
    unmatched = [n for n, _ in m.named_parameters() if not any(n.startswith(p) for p, _g in GROUP_PREFIXES)]
    # decoder block structure (verifies the smp U-Net decoder actually installed)
    blocks = getattr(m.us_decoder, "blocks", [])
    block_desc = []
    for b in blocks:
        convs = [x for x in b.modules() if isinstance(x, nn.Conv2d)]
        bns = [x for x in b.modules() if isinstance(x, nn.BatchNorm2d)]
        block_desc.append({"conv3x3": sum(1 for c in convs if c.kernel_size == (3, 3)), "batchnorm": len(bns),
                           "out_channels": convs[-1].out_channels if convs else None})
    centre = getattr(m.us_decoder, "center", None)
    act = [n for n, x in m.named_modules() if isinstance(x, (nn.Sigmoid, nn.Softmax, nn.Tanh))]
    with torch.no_grad():
        x = torch.zeros(2, 1, 256, 256)
        feats = m.encoder(x)
        seg, cls = m(x, torch.tensor([US, MM]))
    report = {
        "torch": torch.__version__, "segmentation_models_pytorch": smp.__version__,
        "bmc_model_sha256": model_sha, "bmc_model_matches_frozen_runs": model_sha == EXPECTED_MODEL_SHA,
        "parameters_per_group": counts, "parameters_total": total, "parameters_trainable": trainable,
        "parameters_updated_by_an_ultrasound_client": us_client,
        "parameters_updated_by_a_mammography_client": mm_client,
        "float_buffers_elements": float_buffers, "integer_buffers_count": int_buffers,
        "float32_size_MiB": round(total * 4 / 2 ** 20, 2),
        "unmatched_parameter_names": unmatched,
        "decoder_blocks": block_desc,
        "decoder_center_block": None if centre is None or isinstance(centre, nn.Identity) else type(centre).__name__,
        "decoder_forward_takes_varargs": decoder_takes_varargs(m.us_decoder),
        "encoder_feature_shapes_256": [list(t.shape[1:]) for t in feats],
        "output_shapes": {"seg_logits": list(seg.shape), "cls_logits": list(cls.shape)},
        "output_activation_modules": act,
        "conv1_shape": list(m.encoder.conv1.weight.shape),
        "bn_momenta": sorted({x.momentum for x in m.modules() if isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d))}),
    }
    print("=" * 78)
    print("FROZEN R5 MODEL - PARAMETER COUNT AND ARCHITECTURE FACTS (CPU, no data, no download)")
    print("=" * 78)
    print(f"torch {report['torch']} | smp {report['segmentation_models_pytorch']}")
    print(f"bmc_model.py SHA-256 {model_sha[:16]}...  matches frozen runs: {report['bmc_model_matches_frozen_runs']}")
    print("\nparameters per structural group:")
    for k, v in counts.items():
        print(f"  {k:22s} {v:>12,d}  ({100 * v / total:5.2f}%)")
    print(f"  {'TOTAL':22s} {total:>12,d}   trainable {trainable:,d}   float32 {report['float32_size_MiB']} MiB")
    print(f"\nupdated by one ultrasound client  : {us_client:,d}")
    print(f"updated by one mammography client : {mm_client:,d}")
    print(f"BatchNorm float-buffer elements   : {float_buffers:,d}   integer buffers (never averaged): {int_buffers}")
    print(f"\ndecoder blocks (each): {block_desc}")
    print(f"decoder centre block : {report['decoder_center_block']}   forward(*features): {report['decoder_forward_takes_varargs']}")
    print(f"encoder feature shapes @256: {report['encoder_feature_shapes_256']}")
    print(f"outputs: seg {report['output_shapes']['seg_logits']}  cls {report['output_shapes']['cls_logits']}")
    print(f"output activation modules: {act or 'none'}   conv1 {report['conv1_shape']}   BN momenta {report['bn_momenta']}")
    print(f"unmatched parameter names: {unmatched or 'none'}")
    out = HERE / "parameter_count_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
