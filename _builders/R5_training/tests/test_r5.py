#!/usr/bin/env python3
"""tests/test_r5.py - pre-R6 self-tests T01-T46 (synthetic data only; no study data,
no training on study images, no test or external data).

    python tests/test_r5.py                 all tests
    python tests/test_r5.py --only T01,T03  a subset (used by the mutation harness)
    python tests/test_r5.py --json out.json machine-readable results
"""
from __future__ import annotations
import argparse, copy, inspect, json, math, re, sys, tempfile, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent            # R5_training/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))                      # _builders/ (bmc_data.py)
for p in (HERE.parent / "R4_evaluation", HERE.parent):
    if (p / "bmc_eval.py").is_file():
        sys.path.insert(0, str(p)); break

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bmc_aggregation as AG
import bmc_config as CF
import bmc_losses as LS
import bmc_model as MD
import bmc_repro as RP
import bmc_sampling as SA
import bmc_schedule as SC
import bmc_select as SE
import bmc_train as TR

torch.set_num_threads(1)
H = 32                                                    # tiny synthetic images


def cfg_small(name="proposed", **over):
    c = copy.deepcopy(CF.build()[name])
    c.update({"mixed_precision": None})
    if c.get("mu") == CF.MU_SELECTED:
        c["mu"] = 0.01
    c.update(over)
    return c


def synth_client(name, code, n_train=16, n_val=4, seed=0, labels=None):
    g = torch.Generator().manual_seed(seed)
    lab = labels or [i % 2 for i in range(n_train)]
    x = torch.randn(len(lab), 1, H, H, generator=g)
    m = (torch.rand(len(lab), 1, H, H, generator=g) > 0.7).float()
    xv = torch.randn(n_val, 1, H, H, generator=g)
    mv = (torch.rand(n_val, 1, H, H, generator=g) > 0.7).float()
    val = [{"image": xv[i], "mask": mv[i], "label": i % 2} for i in range(n_val)]
    return TR.ClientData(name, code, "train", "validation", lab,
                         lambda i: {"image": x[i], "mask": m[i], "label": lab[i]}, val)


def eight_clients():
    return [synth_client(f"C{i}", MD.US if i < 4 else MD.MM, seed=i) for i in range(8)]


def model_eval():
    torch.manual_seed(0)
    return MD.BMCModel().eval()


TESTS = {}


def test(tid):
    def deco(fn):
        TESTS[tid] = fn
        return fn
    return deco


# ---------------------------------------------------------------- A: contract
@test("T01")
def t01():
    m = model_eval()
    last = m.cls_head[-1]
    with torch.no_grad():
        last.bias.fill_(5.0); last.weight.zero_()
        _, c = m(torch.randn(2, 1, H, H), torch.tensor([0, 1]))
    return isinstance(last, nn.Linear) and torch.allclose(c, torch.full_like(c, 5.0)), f"cls logits {c.tolist()} (a sigmoid could not exceed 1)"


@test("T02")
def t02():
    m = model_eval()
    with torch.no_grad():
        for h in (m.us_seg_head, m.mm_seg_head):
            h.weight.zero_(); h.bias.fill_(-7.0)
        s, _ = m(torch.randn(2, 1, H, H), torch.tensor([0, 1]))
    return bool(torch.allclose(s, torch.full_like(s, -7.0))), f"seg logits all {float(s.mean()):.1f} (negative: no sigmoid)"


@test("T03")
def t03():
    m = model_eval()
    act = [n for n, mod in m.named_modules() if isinstance(mod, (nn.Sigmoid, nn.Softmax, nn.Tanh))]
    return not act and isinstance(m.cls_head[-1], nn.Linear) and isinstance(m.us_seg_head, nn.Conv2d), f"output activations found: {act or 'none'}"


@test("T04")
def t04():
    m = model_eval()
    with torch.no_grad():
        s, c = m(torch.randn(2, 1, 256, 256), torch.tensor([0, 1]))
    return tuple(s.shape) == (2, 1, 256, 256) and tuple(c.shape) == (2,), f"seg {tuple(s.shape)}, cls {tuple(c.shape)}"


# ---------------------------------------------------------------- aggregation
def _meta_states():
    m = model_eval()
    base = {k: v.clone() for k, v in m.state_dict().items()}
    meta = {f"U{i}": (MD.US, n) for i, n in enumerate((448, 566, 176, 1313))}
    meta.update({f"M{i}": (MD.MM, n) for i, n in enumerate((74, 173, 1964, 116))})
    states = {}
    for j, k in enumerate(meta):
        states[k] = {kk: (v + (j + 1) if torch.is_floating_point(v) else v + 10 * (j + 1)) for kk, v in base.items()}
    return base, meta, states


@test("T05")
def t05():
    m = model_eval()
    g = MD.structural_groups(m)
    ok = all(v in ("global", "us", "mm", "excluded") for v in g.values())
    ok &= all(g[k] == "us" for k in g if k.startswith("us_") and g[k] != "excluded")
    ok &= all(g[k] == "mm" for k in g if k.startswith("mm_") and g[k] != "excluded")
    try:
        MD.group_of("foo.weight"); raised = False
    except KeyError:
        raised = True
    return ok and raised, f"{len(g)} entries grouped; unmatched name refused: {raised}"


@test("T06")
def t06():
    base, meta, states = _meta_states()
    new = AG.aggregate("fedavg", base, states, meta)
    k = "encoder.conv1.weight"
    n = [m[1] for m in meta.values()]
    expect = base[k] + sum((j + 1) * nj for j, nj in enumerate(n)) / sum(n)
    return torch.allclose(new[k], expect, atol=1e-5), "global = n-weighted mean over all 8"


def _us_key():
    return next(k for k in model_eval().state_dict() if k.startswith("us_decoder.") and k.endswith("weight"))


def _mm_key():
    return next(k for k in model_eval().state_dict() if k.startswith("mm_decoder.") and k.endswith("weight"))


@test("T07")
def t07():
    base, meta, states = _meta_states(); k = _us_key()
    a = AG.aggregate("proposed", base, states, meta)[k]
    for c in meta:
        if meta[c][0] == MD.MM:
            states[c][k] = states[c][k] + 1000.0
    b = AG.aggregate("proposed", base, states, meta)[k]
    b2 = AG.aggregate("fedavg", base, states, meta)[k]
    return torch.equal(a, b) and torch.allclose(b2, AG.aggregate("fedavg", base, _meta_states()[2], meta)[k]), "US params ignore mammography clients"


@test("T08")
def t08():
    base, meta, states = _meta_states(); k = _mm_key()
    a = AG.aggregate("proposed", base, states, meta)[k]
    for c in meta:
        if meta[c][0] == MD.US:
            states[c][k] = states[c][k] + 1000.0
    return torch.equal(a, AG.aggregate("proposed", base, states, meta)[k]), "MM params ignore ultrasound clients"


@test("T09")
def t09():
    base, meta, states = _meta_states()
    new = AG.aggregate("proposed", base, states, meta)
    ex = [k for k, v in base.items() if not torch.is_floating_point(v)]
    return ex and all(torch.equal(new[k], base[k]) for k in ex), f"{len(ex)} integer buffers kept at the server value"


# ---------------------------------------------------------------- routing
@test("T10")
def t10():
    m = model_eval(); x = torch.randn(4, 1, H, H); code = torch.tensor([1, 0, 1, 0])
    with torch.no_grad():
        s, _ = m(x, code)
        ok = all(torch.allclose(s[i:i + 1], m(x[i:i + 1], code[i:i + 1])[0], atol=1e-5) for i in range(4))
        # decoder IDENTITY: ultrasound head fixed at +3, mammography head at -3
        for h, v in ((m.us_seg_head, 3.0), (m.mm_seg_head, -3.0)):
            h.weight.zero_(); h.bias.fill_(v)
        s2, _ = m(x, code)
        ident = all(torch.allclose(s2[i], torch.full_like(s2[i], 3.0 if int(code[i]) == MD.US else -3.0)) for i in range(4))
    return ok and ident, f"per-sample equals single-sample pass: {ok}; each sample through its OWN modality decoder: {ident}"


@test("T11")
def t11():
    m = model_eval(); x = torch.randn(4, 1, H, H); code = torch.tensor([0, 1, 1, 0]); perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        s1, _ = m(x, code); s2, _ = m(x[perm], code[perm])
    return torch.allclose(s1[perm], s2, atol=1e-5), "outputs follow the samples when the first sample's modality changes"


# ---------------------------------------------------------------- data roles
def _manifest_rows():
    import csv
    root = Path(r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging\Revision 1\dataset preparing")
    p = root / "CANONICAL_DATASET_BANK" / "_bmc_splits" / "BMC_SPLITS_MANIFEST.csv"
    if p.is_file():
        return list(csv.DictReader(p.open(encoding="utf-8-sig"))), "real frozen manifest"
    rows = []
    for ds in ("A", "B"):
        for i in range(30):
            sp = ("train", "validation", "test")[i % 3]
            rows.append({"dataset": ds, "split": sp, "canonical_record_id": f"{ds}{i}", "group_key": f"G{ds}{i}", "role": "CLIENT"})
    rows.append({"dataset": "E", "split": "EXTERNAL_TEST_ONLY", "canonical_record_id": "E0", "group_key": "GE0", "role": "EXTERNAL"})
    return rows, "synthetic manifest (the frozen manifest is not on this machine)"


@test("T12")
def t12():
    rows, src = _manifest_rows(); bad = 0
    for ds in {r["dataset"] for r in rows if r["role"] == "CLIENT"}:
        s = {sp: {r["canonical_record_id"] for r in rows if r["dataset"] == ds and r["split"] == sp} for sp in ("train", "validation", "test")}
        bad += len(s["train"] & s["validation"]) + len(s["train"] & s["test"]) + len(s["validation"] & s["test"])
    return bad == 0, f"{src}: {bad} overlapping records"


@test("T13")
def t13():
    rows, src = _manifest_rows(); bad = 0
    for ds in {r["dataset"] for r in rows if r["role"] == "CLIENT"}:
        s = {sp: {r["group_key"] for r in rows if r["dataset"] == ds and r["split"] == sp} for sp in ("train", "validation", "test")}
        bad += len(s["train"] & s["validation"]) + len(s["train"] & s["test"]) + len(s["validation"] & s["test"])
    return bad == 0, f"{src}: {bad} group keys crossing partitions"


def _refuses(fn):
    try:
        fn(); return False
    except (ValueError, RuntimeError, SE.SelectionError):
        return True


@test("T14")
def t14():
    mk = lambda sp: TR.ClientData("x", 0, sp, "validation", [0, 1], lambda i: {}, [])
    return _refuses(lambda: mk("EXTERNAL_TEST_ONLY")) and _refuses(lambda: mk("test")) and _refuses(lambda: mk("validation")), "external, test and validation refused as training data"


@test("T15")
def t15():
    return _refuses(lambda: SE.select([{"eval_index": 0, "split": "test", "criterion": 0.1}])) and \
        _refuses(lambda: TR.ClientData("x", 0, "train", "test", [0, 1], lambda i: {}, [])), "test refused by the selector and as selection data"


@test("T16")
def t16():
    return _refuses(lambda: SE.select([{"eval_index": 0, "split": "EXTERNAL_TEST_ONLY", "criterion": 0.1}])) and \
        _refuses(lambda: TR.ClientData("x", 0, "train", "EXTERNAL_TEST_ONLY", [0, 1], lambda i: {}, [])), "external refused by the selector"


# ---------------------------------------------------------------- sampling
@test("T17")
def t17():
    seen = []
    orig = SA.WeightedRandomSampler

    class Rec(orig):
        def __init__(self, *a, **k):
            seen.append("W"); super().__init__(*a, **k)
    SA.WeightedRandomSampler = Rec
    try:
        SA.client_train_batches([0, 1] * 16, torch.Generator().manual_seed(1))
        n_train = len(seen)
        SA.eval_order(10); n_eval = len(seen) - n_train
    finally:
        SA.WeightedRandomSampler = orig
    return n_train == 1 and n_eval == 0, f"weighted sampler used by training ({n_train}), never by evaluation ({n_eval})"


@test("T18")
def t18():
    w = SA.class_balanced_weights([0, 0, 0, 1])
    return torch.allclose(w, torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0], dtype=w.dtype)), f"weights {w.tolist()} = 1/n_class"


@test("T19")
def t19():
    a, b = SA.eval_order(37), SA.eval_order(37)
    return a == b == list(range(37)), "evaluation order sequential and identical"


# ---------------------------------------------------------------- optimiser/schedule
_RUN = {}


def _two_round_run():
    if "fed" not in _RUN:
        seen = []
        # three clients per modality group: a 0.4 cap is infeasible for a one-client group
        # unequal validation sizes, so sample weighting and client-macro differ
        cl = [synth_client(f"U{i}", MD.US, n_train=32, n_val=2 + i, seed=1 + i) for i in range(3)] + \
             [synth_client(f"M{i}", MD.MM, n_train=32, n_val=5 + 2 * i, seed=11 + i) for i in range(3)]
        c = cfg_small("proposed", rounds=2, local_epochs=2)
        res = TR.run_federated(c, cl, seed=101, on_round=lambda r, k, opt, lr: seen.append(
            (r, k, lr, max(int(s["step"]) for s in opt.state.values()), [g["lr"] for g in opt.param_groups])))
        _RUN["fed"] = (res, seen)
    return _RUN["fed"]


@test("T20")
def t20():
    res, seen = _two_round_run()
    steps = [s[3] for s in seen]                    # 2 batches x 2 epochs = 4 per round
    return all(x == 4 for x in steps), f"AdamW steps at end of each client round: {steps} (persist within a round, reset between rounds)"


@test("T21")
def t21():
    res, seen = _two_round_run()
    ok = all(abs(s[2] - SC.lr_at(s[0])) < 1e-18 and all(abs(g - SC.lr_at(s[0])) < 1e-18 for g in s[4]) for s in seen)
    return ok, f"lr per round {[round(s[2], 10) for s in seen]} = server schedule"


@test("T22")
def t22():
    seq = SC.lr_sequence()
    e49 = 1e-6 + 0.5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * 0.98))
    ok = len(seq) == 50 and seq[0] == 1e-4 and abs(seq[25] - 5.05e-5) < 1e-15 and abs(seq[49] - 1.0976769428e-6) < 1e-15 \
        and abs(seq[49] - e49) < 1e-20 and all(a > b for a, b in zip(seq, seq[1:]))
    return ok, f"lr[0]={seq[0]}, lr[25]={seq[25]:.6e} (5.05e-5), lr[49]={seq[49]:.10e} (1.0976769428e-6), strictly decreasing"


# ---------------------------------------------------------------- FedProx
@test("T23")
def t23():
    m = model_eval(); ref = {n: p.detach().clone() for n, p in m.named_parameters()}
    return LS.prox_term(m, ref, 0.01).item() == 0.0, "identical parameters -> 0"


@test("T24")
def t24():
    m = model_eval(); ref = {n: p.detach().clone() for n, p in m.named_parameters()}
    w = m.cls_head[-1].weight
    with torch.no_grad():
        w.add_(0.5)
    v, expect = LS.prox_term(m, ref, 0.01).item(), 0.5 * 0.01 * w.numel() * 0.25
    return abs(v - expect) <= 1e-6 * expect, f"{v:.9f} = (mu/2)*n*delta^2 = {expect:.9f} (float32, rel. tol. 1e-6)"


@test("T25")
def t25():
    m = model_eval(); ref = {n: p.detach().clone() for n, p in m.named_parameters()}
    bn = next(mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d))
    with torch.no_grad():
        bn.running_mean.add_(9.0)
    zero_after_buffer = float(LS.prox_term(m, ref, 0.01)) == 0.0
    hit = {}
    for prefix in ("encoder.", "us_decoder.", "mm_decoder.", "cls_head."):
        m2 = model_eval(); r2 = {n: p.detach().clone() for n, p in m2.named_parameters()}
        p = next(p for n, p in m2.named_parameters() if n.startswith(prefix))
        with torch.no_grad():
            p.add_(0.1)
        hit[prefix] = float(LS.prox_term(m2, r2, 0.01)) > 0
    return zero_after_buffer and all(hit.values()), f"buffers excluded: {zero_after_buffer}; every trainable group included: {hit}"


# ---------------------------------------------------------------- loss
@test("T26")
def t26():
    seg = torch.zeros(1, 1, 2, 2); gt = torch.tensor([[[[1.0, 0], [0, 0]]]])
    v = float(LS.joint_loss(seg, gt, torch.zeros(1), torch.ones(1)))
    expect = 0.25 + 1.5 * math.log(2.0)
    return abs(v - expect) < 1e-6, f"{v:.6f} = 0.25 + 1.5 ln 2 = {expect:.6f}"


@test("T27")
def t27():
    seg = torch.randn(3, 1, 4, 4); gt = (torch.rand(3, 1, 4, 4) > 0.5).float(); z = torch.randn(3); y = torch.tensor([0., 1, 1])
    s = LS.seg_loss_per_image(seg, gt).mean(); c = LS.cls_loss_per_image(z, y).mean()
    ok = torch.allclose(LS.joint_loss(seg, gt, z, y, 2, 1), 2 * s + c) and torch.allclose(LS.joint_loss(seg, gt, z, y, 1, 0), s) \
        and torch.allclose(LS.joint_loss(seg, gt, z, y, 0, 1), c)
    return bool(ok), "lambda_seg and lambda_cls each applied exactly once"


@test("T28")
def t28():
    macro = SE.client_macro({"A": 1.0, "B": 3.0})
    res, _ = _two_round_run()
    eng = all(abs(h["criterion"] - sum(h["per_client"].values()) / len(h["per_client"])) < 1e-12 for h in res["history"])
    return macro == 2.0 and eng, f"client-macro {macro} (sample-weighted with n=100/10 would be 1.1818); engine uses it: {eng}"


# ---------------------------------------------------------------- selection
@test("T29")
def t29():
    res, _ = _two_round_run()
    return all(h["split"] == "validation" for h in res["history"]) and _refuses(
        lambda: SE.select([{"eval_index": 0, "split": "train", "criterion": 1}])), "selector reads validation entries only"


@test("T30")
def t30():
    i, v = SE.select([{"eval_index": k, "split": "validation", "criterion": c} for k, c in enumerate((3.0, 1.0, 2.0))])
    res, _ = _two_round_run()
    j, _v = SE.select(res["history"])
    return i == 1 and v == 1.0 and j == res["best_index"], f"argmin 1; engine best {res['best_index']} = selector {j}"


@test("T31")
def t31():
    i, _ = SE.select([{"eval_index": k, "split": "validation", "criterion": c} for k, c in enumerate((2.0, 1.0, 1.0))])
    return i == 1, "exact tie -> earliest"


@test("T32")
def t32():
    with tempfile.TemporaryDirectory() as d:
        refused = _refuses(lambda: TR.require_frozen_selection(Path(d)))
        h = [{"eval_index": 0, "split": "validation", "criterion": 1.0}]
        disagree = _refuses(lambda: SE.freeze_record(Path(d) / "x.json", run_id="r", method="m", seed=101,
                                                     history=h, best_so_far_index=1, checkpoint_sha256="0"))
        rec = SE.freeze_record(Path(d) / "selection_record.json", run_id="r", method="m", seed=101,
                               history=h, best_so_far_index=0, checkpoint_sha256="0")
        ok_after = TR.require_frozen_selection(Path(d)).is_file()
        once = _refuses(lambda: SE.freeze_record(Path(d) / "selection_record.json", run_id="r", method="m", seed=101,
                                                 history=h, best_so_far_index=0, checkpoint_sha256="0")) if False else True
        try:
            SE.freeze_record(Path(d) / "selection_record.json", run_id="r", method="m", seed=101,
                             history=h, best_so_far_index=0, checkpoint_sha256="0"); once = False
        except FileExistsError:
            once = True
    ok = refused and disagree and ok_after and once and rec["selection_split"] == "validation" and rec["frozen"] is True
    return ok, "evaluation refused before the record; record write-once; loop/selector disagreement refused"


# ---------------------------------------------------------------- init / seeds
@test("T33")
def t33():
    orig = torch.load
    torch.load = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("file load during init"))
    try:
        a = TR.new_model(cfg_small(), 101); b = TR.new_model(cfg_small(), 202)
    finally:
        torch.load = orig
    k = next(k for k in a.state_dict() if k.startswith("us_decoder.") and k.endswith("weight"))
    rgb = {kk: (torch.randn(64, 3, 7, 7) if kk == "conv1.weight" else
                (torch.randn_like(v) if torch.is_floating_point(v) else v.clone()))
           for kk, v in a.encoder.state_dict().items()}
    c = TR.new_model(cfg_small(), 303, rgb_state=rgb)
    enc_ok = torch.equal(c.encoder.conv1.weight, rgb["conv1.weight"].sum(1, keepdim=True))
    return not torch.equal(a.state_dict()[k], b.state_dict()[k]) and enc_ok, "fresh initialisation; no file loaded; encoder from supplied ImageNet state"


@test("T34")
def t34():
    refused = _refuses(lambda: TR.new_model({**cfg_small(), "init_checkpoint": "best_model.pkl"}, 101))
    # the ONLY permitted load is bmc_run.load_selected_checkpoint: the run's own selected
    # weights, after the selection record is frozen, verified by the record's SHA-256.
    loads = []
    for p in HERE.glob("bmc_*.py"):
        src = p.read_text(encoding="utf-8")
        if p.name == "bmc_run.py":
            src = re.sub(r"def load_selected_checkpoint\(.*?(?=\ndef )", "", src, flags=re.S)
        if re.search(r"torch\.load\(|best_model|load_state_dict\(torch", src):
            loads.append(p.name)
    return refused and not loads, f"init_checkpoint refused; checkpoint loads outside the guarded final-evaluation loader: {loads or 'none'}"


@test("T35")
def t35():
    a, b = TR.new_model(cfg_small(), 101), TR.new_model(cfg_small(), 101)
    return all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()), "same seed -> identical initialisation"


@test("T36")
def t36():
    f = lambda: SA.client_train_batches([0] * 30 + [1] * 18, RP.generator(101, 3))
    g = lambda: SA.centralised_train_batches({"a": [0, 1] * 20, "b": [0] * 10 + [1] * 22}, RP.generator(101, 999))
    return f() == f() and g() == g(), "same seed -> identical sampler order (client and centralised)"


@test("T37")
def t37():
    a, b = TR.new_model(cfg_small(), 101), TR.new_model(cfg_small(), 202)
    s1 = SA.client_train_batches([0, 1] * 24, RP.generator(101, 0)); s2 = SA.client_train_batches([0, 1] * 24, RP.generator(202, 0))
    k = "cls_head.2.weight"
    return not torch.equal(a.state_dict()[k], b.state_dict()[k]) and s1 != s2, "different seeds -> different initialisation and draws"


# ---------------------------------------------------------------- common loss / config
@test("T38")
def t38():
    calls = {"n": 0}; orig = TR.task_loss

    def counting(*a, **k):
        calls["n"] += 1; return orig(*a, **k)
    TR.task_loss = counting
    try:
        seen = {}
        cl = [synth_client(f"U{i}", MD.US, seed=3 + i) for i in range(3)] + \
             [synth_client(f"M{i}", MD.MM, seed=13 + i) for i in range(3)]
        for name in ("proposed", "fedavg", "fedprox"):
            n0 = calls["n"]; TR.run_federated(cfg_small(name, rounds=1, local_epochs=1), cl, 101); seen[name] = calls["n"] - n0
        n0 = calls["n"]; TR.run_centralised(cfg_small("centralised", epochs=2, eval_every=2), cl, 101); seen["centralised"] = calls["n"] - n0
        n0 = calls["n"]; TR.run_local(cfg_small("local", epochs=2, eval_every=2), cl[0], 101, 0); seen["local"] = calls["n"] - n0
    finally:
        TR.task_loss = orig
    return all(v > 0 for v in seen.values()), f"task_loss calls per method: {seen}"


@test("T39")
def t39():
    c = CF.build(); base = c["proposed"]
    allowed = {"fedavg": {"method", "mu", "aggregation"}, "fedprox": {"method", "aggregation"},
               "N1_mu0": {"mu", "ablation"}, "N2_uncapped": {"aggregation", "ablation"},
               "N3_cls_only": {"loss", "ablation"}, "N4_seg_only": {"loss", "ablation"},
               "centralised": {"method", "regime", "rounds", "local_epochs", "optimizer_reset_each_round", "mu",
                               "aggregation", "epochs", "eval_every", "centralised_draws"},
               "local": {"method", "regime", "rounds", "local_epochs", "optimizer_reset_each_round", "mu",
                         "aggregation", "epochs", "eval_every"}}
    bad = {}
    for name, keys in allowed.items():
        diff = {k for k in set(base) | set(c[name]) if base.get(k) != c[name].get(k)}
        if not diff <= keys:
            bad[name] = sorted(diff - keys)
    return not bad, f"differences outside the frozen methodological components: {bad or 'none'}"


# ---------------------------------------------------------------- v2 additions
@test("T40")
def t40():
    us = {"BUSI": 448, "BUS-UC": 566, "BrEaST": 176, "BUS-BRA": 1313}
    mm = {"INbreast": 74, "DMID": 173, "CBIS-DDSM": 1964, "AIISLAB": 116}
    wu, wm = AG.waterfill(us, 0.4), AG.waterfill(mm, 0.4)
    exp = {"BUS-BRA": 0.400, "BUSI": 0.226, "CBIS-DDSM": 0.400, "INbreast": 0.122, "DMID": 0.286, "AIISLAB": 0.192}
    got = {**wu, **wm}
    ok = all(abs(got[k] - v) < 5e-4 for k, v in exp.items()) and max(got.values()) <= 0.4 + 1e-12
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = {i: float(x) for i, x in enumerate(rng.integers(1, 3000, 4))}
        ok &= max(AG.waterfill(n, 0.4).values()) <= 0.4 + 1e-12 and abs(sum(AG.waterfill(n, 0.4).values()) - 1) < 1e-12
    return ok, f"CBIS-DDSM {got['CBIS-DDSM']:.3f}, BUS-BRA {got['BUS-BRA']:.3f}, INbreast {got['INbreast']:.3f}; cap never exceeded in 200 random cases"


@test("T41")
def t41():
    from PIL import Image
    import bmc_cache as CA
    import bmc_data as BD
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); rows = []; rng = np.random.default_rng(1)
        for ds, bits in (("BUSI", 8), ("INbreast", 16)):
            for i in range(3):
                ip, mp = d / f"{ds}{i}.png", d / f"{ds}{i}_m.png"
                if bits == 8:
                    Image.fromarray(rng.integers(0, 256, (70, 50)).astype(np.uint8), "L").save(ip)
                else:
                    Image.fromarray(rng.integers(0, 4096, (70, 50)).astype(np.uint16)).save(ip)
                mk = np.zeros((70, 50), np.uint8); mk[20:40, 10:30] = 255; Image.fromarray(mk, "L").save(mp)
                rows.append({"dataset": ds, "role": "CLIENT", "split": "validation", "modality": "Mammography" if bits == 16 else "Ultrasound",
                             "canonical_record_id": f"{ds}_{i}", "canonical_image_abspath": str(ip),
                             "canonical_mask_abspath": str(mp), "label_binary": str(i % 2), "group_key": f"G{i}"})
        bad_total, eq_items = 0, True
        for ds in ("BUSI", "INbreast"):
            o = BD.BMCSplitDataset(ds, "validation", prep_root=d, manifest_rows=rows, **BD.APPROVED_R3)
            CA.build_cache(o, d / "cache"); bad_total += len(CA.verify_cache(o, d / "cache"))
            c = CA.CachedBMCSplitDataset(ds, "validation", prep_root=d, manifest_rows=rows, cache_dir=d / "cache", **BD.APPROVED_R3)
            for i in range(len(o)):
                a, b = o[i], c[i]
                eq_items &= np.array_equal(a["image"], b["image"]) and np.array_equal(a["mask"], b["mask"])
        f = next((d / "cache" / "INbreast" / "validation").glob("*.npz"))
        with np.load(f) as z:
            img = z["image"].copy(); msk = z["mask"].copy()
        img[0, 0] += 1e-7; np.savez(f, image=img, mask=msk)
        o = BD.BMCSplitDataset("INbreast", "validation", prep_root=d, manifest_rows=rows, **BD.APPROVED_R3)
        caught = len(CA.verify_cache(o, d / "cache")) == 1
    return bad_total == 0 and eq_items and caught, f"0 mismatches over 6 records (8- and 16-bit); cached == R3 outputs: {eq_items}; tampered cache detected: {caught}"


@test("T42")
def t42():
    w = torch.randn(64, 3, 7, 7); w1 = MD.convert_first_conv(w)
    return tuple(w1.shape) == (64, 1, 7, 7) and torch.equal(w1, w[:, 0:1] + w[:, 1:2] + w[:, 2:3]), "W1 == sum of RGB channels exactly"


@test("T43")
def t43():
    w = torch.randn(64, 3, 7, 7); x = torch.randn(2, 1, 40, 40)
    a = F.conv2d(x, MD.convert_first_conv(w), stride=2, padding=3)
    b = F.conv2d(torch.cat([x, x, x], 1), w, stride=2, padding=3)
    return torch.allclose(a, b, atol=1e-4), f"max |diff| {float((a - b).abs().max()):.2e}"


@test("T44")
def t44():
    base, meta, states = _meta_states()
    k = next(k for k, v in base.items() if k.startswith("us_decoder.") and k.endswith("running_mean"))
    a = {m: AG.aggregate(m, base, states, meta)[k] for m in ("proposed", "fedavg")}
    for c in meta:
        if meta[c][0] == MD.MM:
            states[c][k] = states[c][k] + 500.0
    b = {m: AG.aggregate(m, base, states, meta)[k] for m in ("proposed", "fedavg")}
    return all(torch.equal(a[m], b[m]) for m in a), "US-decoder BN buffers unaffected by mammography clients"


@test("T45")
def t45():
    base, meta, states = _meta_states()
    new = AG.aggregate("proposed", base, states, meta)
    nbt = [k for k in base if k.endswith("num_batches_tracked")]
    mom = {mod.momentum for mod in model_eval().modules() if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d))}
    return all(torch.equal(new[k], base[k]) for k in nbt) and mom == {0.1}, f"{len(nbt)} num_batches_tracked never averaged; BN momentum {mom}"


@test("T46")
def t46():
    b33 = SA.client_train_batches([0, 1] * 16 + [0], torch.Generator().manual_seed(0))
    bbra = SA.client_train_batches([0] * 888 + [1] * 425, torch.Generator().manual_seed(0))
    guard = _refuses(lambda: TR.train_step(model_eval(), None, (torch.zeros(1, 1, H, H), torch.zeros(1, 1, H, H),
                                                                 torch.zeros(1), torch.zeros(1, dtype=torch.long)), cfg_small(), "cpu"))
    return all(len(b) == 16 for b in b33 + bbra) and len(bbra) == 82 and guard, f"n=33 -> {len(b33)} batches of 16; BUS-BRA 1313 -> {len(bbra)} batches of 16; batch of 1 refused"


@test("T47")
def t47():
    """Both smp decoder APIs receive the identical full feature list (added after the
    user's environment exposed the older forward(*features) signature)."""
    class ListAPI(nn.Module):
        def forward(self, features):
            return torch.stack([f.float().mean() for f in features])
    class VarAPI(nn.Module):
        def forward(self, *features):
            return torch.stack([f.float().mean() for f in features])
    feats = [torch.randn(2, c, 8, 8) for c in (1, 3, 5)]
    a, b = MD.call_decoder(ListAPI(), feats), MD.call_decoder(VarAPI(), feats)
    real = MD.decoder_takes_varargs(model_eval().us_decoder)
    return torch.equal(a, b) and len(a) == 3, f"list API and *features API give identical inputs; installed smp decoder uses *features: {real}"


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    ids = a.only.split(",") if a.only else list(TESTS)
    res = {}
    for tid in ids:
        try:
            ok, msg = TESTS[tid]()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"EXCEPTION {type(e).__name__}: {e}"
        res[tid] = (bool(ok), msg)
        print(f"  {tid} [{'PASS' if ok else 'FAIL'}]  {msg}", flush=True)
    n_f = sum(1 for v in res.values() if not v[0])
    print(f"PASS = {len(res) - n_f}   FAIL = {n_f}")
    print("R5_TESTS_VERIFIED" if n_f == 0 else "R5_TESTS_BLOCKED")
    if a.json:
        Path(a.json).write_text(json.dumps({k: {"pass": v[0], "detail": v[1]} for k, v in res.items()}, indent=1))
    return 0 if n_f == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
