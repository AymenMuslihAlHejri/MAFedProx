#!/usr/bin/env python3
"""tests/mutate_r5.py - inject realistic defects into a COPY of R5_training and prove the
intended T-tests catch each one. The original files are never modified."""
from __future__ import annotations
import json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PARENT = HERE.parent
M = [  # (id, description, file, [(old, new), ...], intended tests)
 ("M01", "accidental final sigmoid in the classification head", "bmc_model.py",
  [("nn.Linear(64, 1))                                              # raw logit", "nn.Linear(64, 1), nn.Sigmoid())")], ["T01", "T03"]),
 ("M02", "modality decoders swapped", "bmc_model.py",
  [("for code, dec, head in ((US, self.us_decoder, self.us_seg_head),\n                                (MM, self.mm_decoder, self.mm_seg_head)):",
    "for code, dec, head in ((MM, self.us_decoder, self.us_seg_head),\n                                (US, self.mm_decoder, self.mm_seg_head)):")], ["T10"]),
 ("M03", "first-sample modality routing", "bmc_model.py",
  [("idx = (modality == code).nonzero(as_tuple=True)[0]",
    "idx = torch.arange(x.shape[0], device=x.device) if int(modality[0]) == code else torch.arange(0, device=x.device)")], ["T10", "T11"]),
 ("M04", "cross-modality aggregation (ultrasound group includes all clients)", "bmc_aggregation.py",
  [("us = {k: float(m[1]) for k, m in client_meta.items() if m[0] == US}", "us = {k: float(m[1]) for k, m in client_meta.items()}")], ["T07", "T44"]),
 ("M05", "integer buffers (num_batches_tracked) averaged", "bmc_aggregation.py",
  [("        if is_excluded_buffer(key, ref):\n            new[key] = ref.clone()", "        if False:\n            new[key] = ref.clone()")], ["T09", "T45"]),
 ("M06", "test/external entries accepted by the selector", "bmc_select.py",
  [('        if h.get("split") != "validation":', "        if False:")], ["T15", "T16", "T29"]),
 ("M07", "external/test data accepted for training", "bmc_train.py",
  [("        if self.train_split not in ALLOWED_TRAIN_SPLITS:", "        if False:")], ["T14"]),
 ("M08", "schedule never advances (legacy F11)", "bmc_train.py",
  [("        lr = lr_at(r)\n        server = copy.deepcopy(model.state_dict())", "        lr = lr_at(0)\n        server = copy.deepcopy(model.state_dict())")], ["T21"]),
 ("M09", "optimiser state persists across rounds, contrary to D5", "bmc_train.py",
  [("    history, best, best_idx, best_state = [], None, None, None\n    for r in range(cfg[\"rounds\"]):",
    "    history, best, best_idx, best_state = [], None, None, None\n    opts = {}\n    for r in range(cfg[\"rounds\"]):"),
   ("opt = make_optimizer(model, cfg)                        # reset every round (D5)", "opt = opts.setdefault(c.name, make_optimizer(model, cfg))")], ["T20"]),
 ("M10", "wrong FedProx parameter set (encoder excluded)", "bmc_losses.py",
  [("        if not p.requires_grad:\n            continue", "        if not p.requires_grad or n.startswith(\"encoder.\"):\n            continue")], ["T25"]),
 ("M11", "sample-weighted validation criterion instead of client-macro", "bmc_train.py",
  [("        h = _record(history, r, per, lr, cfg[\"local_epochs\"] * (r + 1))",
    "        h = _record(history, r, per, lr, cfg[\"local_epochs\"] * (r + 1)); h[\"criterion\"] = sum(per[c.name] * len(c.val_items) for c in clients) / sum(len(c.val_items) for c in clients)")], ["T28"]),
 ("M12", "legacy checkpoint loading permitted", "bmc_train.py",
  [('    if "init_checkpoint" in cfg or "resume" in cfg:', "    if False:")], ["T34"]),
 ("M13", "segmentation loss coefficient applied twice", "bmc_losses.py",
  [("out = out + lam_seg * seg_loss_per_image(seg_logits, gt)", "out = out + lam_seg * lam_seg * seg_loss_per_image(seg_logits, gt)")], ["T27"]),
 ("M14", "first-layer conversion averages instead of sums", "bmc_model.py",
  [("return w_rgb.sum(dim=1, keepdim=True)", "return w_rgb.mean(dim=1, keepdim=True)")], ["T42", "T43"]),
 ("M15", "drop_last disabled for training", "bmc_sampling.py",
  [("BatchSampler(ws, batch_size=BATCH, drop_last=True)", "BatchSampler(ws, batch_size=BATCH, drop_last=False)")], ["T46"]),
 ("M16", "legacy single-pass cap (renormalisation breaks the cap)", "bmc_aggregation.py",
  [("    fixed: set = set()\n    while True:",
    "    w = {k: min(v, cap) for k, v in w.items()}; s_ = sum(w.values()); return {k: v / s_ for k, v in w.items()}\n    fixed: set = set()\n    while True:")], ["T40"]),
 ("M17", "cache stores reduced precision (not bit-identical to R3)", "bmc_cache.py",
  [("np.savez(p, image=img, mask=mask)", "np.savez(p, image=img.astype(np.float16), mask=mask)")], ["T41"]),
]


def run(mid, desc, fname, reps, intended):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "b"; shutil.copytree(HERE, root / "R5_training")
        for extra in ("bmc_data.py", "bmc_eval.py"):
            if (PARENT / extra).is_file():
                shutil.copy2(PARENT / extra, root / extra)
        f = root / "R5_training" / fname; s = f.read_text()
        for a, b in reps:
            if s.count(a) != 1:
                return mid, desc, intended, [], "MUTATION TEXT NOT FOUND"
            s = s.replace(a, b)
        f.write_text(s)
        out = subprocess.run([sys.executable, "tests/test_r5.py", "--only", ",".join(intended)],
                             capture_output=True, text=True, cwd=root / "R5_training", timeout=3000)
        failed = re.findall(r"^\s+(T\d+) \[FAIL\]", out.stdout, re.M)
        return mid, desc, intended, failed, "CAUGHT" if set(intended) <= set(failed) else "MISSED"


def main():
    rows = [run(*m) for m in M]
    print(f"{'id':4s} {'mutation':62s} {'intended':16s} {'failed':16s} verdict")
    print("-" * 110)
    for mid, desc, it, fl, v in rows:
        print(f"{mid:4s} {desc:62s} {','.join(it):16s} {','.join(fl) or '-':16s} {v}")
    n = sum(1 for r in rows if r[4] == "CAUGHT")
    print("-" * 110); print(f"mutations caught: {n} of {len(rows)}")
    return 0 if n == len(rows) else 4


if __name__ == "__main__":
    raise SystemExit(main())
