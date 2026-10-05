#!/usr/bin/env python3
"""tests/test_r6_driver.py - driver-level tests on a SYNTHETIC bank (no study data).

    python tests/test_r6_driver.py
"""
from __future__ import annotations
import copy, csv, json, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
for p in (HERE.parent / "R4_evaluation", HERE.parent):
    if (p / "bmc_eval.py").is_file():
        sys.path.insert(0, str(p)); break
import numpy as np
import torch
from PIL import Image

import bmc_run as RN
import bmc_select as SE
import bmc_train as TR
from bmc_model import US, MM

torch.set_num_threads(1)
H = 32
CLIENTS = [("BUSI", "Ultrasound", 8), ("BUS-UC", "Ultrasound", 8), ("BrEaST", "Ultrasound", 8),
           ("DMID", "Mammography", 8), ("AIISLAB", "Mammography", 8), ("INbreast", "Mammography", 16)]


def synth_bank(root: Path) -> list:
    """Tiny bank with the real directory layout: 6 clients (3 US, 3 MM) + 1 external."""
    rng = np.random.default_rng(0); rows = []
    img_dir = root / "imgs"; img_dir.mkdir(parents=True)
    spec = [(ds, mod, bits, cid, "CLIENT") for cid, (ds, mod, bits) in enumerate(CLIENTS)] + [("BLUIS", "Ultrasound", 8, "", "EXTERNAL")]
    for ds, mod, bits, cid, role in spec:
        splits = [("train", 4), ("validation", 2), ("test", 2)] if role == "CLIENT" else [("EXTERNAL_TEST_ONLY", 2)]
        for sp, n in splits:
            for i in range(n):
                rid = f"{ds}_{sp}_{i}"; ip, mp = img_dir / f"{rid}.png", img_dir / f"{rid}_m.png"
                if bits == 8:
                    Image.fromarray(rng.integers(0, 256, (48, 40)).astype(np.uint8), "L").save(ip)
                else:
                    Image.fromarray(rng.integers(0, 4096, (48, 40)).astype(np.uint16)).save(ip)
                mk = np.zeros((48, 40), np.uint8); mk[10:30, 8:28] = 255; Image.fromarray(mk, "L").save(mp)
                rows.append({"client_id": cid, "dataset": ds, "modality": mod, "role": role, "split": sp,
                             "canonical_record_id": rid, "canonical_image_abspath": str(ip),
                             "canonical_mask_abspath": str(mp), "label": "", "label_binary": str(i % 2),
                             "group_key": f"G_{rid}", "group_unit": "IMAGE", "seed": RN.SPLIT_SEED})
    bank = root / "CANONICAL_DATASET_BANK"
    (bank / "_bmc_splits").mkdir(parents=True); (bank / "_bmc_cohort").mkdir(parents=True)
    with (bank / "_bmc_splits" / "BMC_SPLITS_MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    (bank / "_bmc_splits" / "BMC_SPLITS_REPORT.json").write_text(json.dumps({"seed": RN.SPLIT_SEED, "verdict": "BMC_SPLITS_LOCKED"}))
    (bank / "_bmc_cohort" / "BMC_COHORT_MANIFEST.csv").write_text("synthetic\n")
    return rows


def synth_clients():
    out = []
    for k, (ds, mod, _b) in enumerate(CLIENTS):
        g = torch.Generator().manual_seed(k); lab = [i % 2 for i in range(16)]
        x = torch.randn(16, 1, H, H, generator=g); m = (torch.rand(16, 1, H, H, generator=g) > 0.7).float()
        val = [{"image": torch.randn(1, H, H, generator=g), "mask": (torch.rand(1, H, H, generator=g) > .7).float(), "label": i % 2} for i in range(3)]
        out.append(TR.ClientData(ds, US if mod == "Ultrasound" else MM, "train", "validation", lab,
                                 (lambda i, x=x, m=m, lab=lab: {"image": x[i], "mask": m[i], "label": lab[i]}), val))
    return out



CLIENTS8 = [("BUSI", "Ultrasound", 8), ("BUS-UC", "Ultrasound", 8), ("BrEaST", "Ultrasound", 8), ("BUS-BRA", "Ultrasound", 8),
            ("INbreast", "Mammography", 16), ("DMID", "Mammography", 8), ("CBIS-DDSM", "Mammography", 16), ("AIISLAB", "Mammography", 8)]


def synth_bank8(root: Path) -> list:
    global CLIENTS
    saved = CLIENTS
    CLIENTS = CLIENTS8
    try:
        rows = synth_bank(root)
    finally:
        CLIENTS = saved
    for r in rows:                                        # CBIS-DDSM declares a 65535 range
        if r["dataset"] == "CBIS-DDSM":
            ip = Path(r["canonical_image_abspath"])
            Image.fromarray(np.random.default_rng(3).integers(0, 65536, (48, 40)).astype(np.uint16)).save(ip)
    return rows


def synth_clients8(n_train=16):
    out = []
    for k, (ds, mod, _b) in enumerate(CLIENTS8):
        g = torch.Generator().manual_seed(100 + k); lab = [i % 2 for i in range(n_train)]
        x = torch.randn(n_train, 1, H, H, generator=g); m = (torch.rand(n_train, 1, H, H, generator=g) > 0.7).float()
        val = [{"image": torch.randn(1, H, H, generator=g), "mask": (torch.rand(1, H, H, generator=g) > .7).float(), "label": i % 2} for i in range(3)]
        out.append(TR.ClientData(ds, US if mod == "Ultrasound" else MM, "train", "validation", lab,
                                 (lambda i, x=x, m=m, lab=lab: {"image": x[i], "mask": m[i], "label": lab[i]}), val))
    return out

FAST = {"rounds": 1, "local_epochs": 1, "mixed_precision": None}
T = []


def add(tid, desc, ok, detail=""):
    T.append((tid, desc, bool(ok), detail)); print(f"  {tid} [{'PASS' if ok else 'FAIL'}]  {desc}" + (f"\n        {detail}" if detail else ""), flush=True)


def refuses(fn, exc=(RN.AccessError, RN.DriverError, ValueError, PermissionError, RuntimeError)):
    try:
        fn(); return False
    except exc:
        return True


def grid_records(exp: Path, crit: dict):
    for mu, v in crit.items():
        d = exp / "mu_selection" / f"fedprox_mu_{mu:g}__seed101"; d.mkdir(parents=True)
        (d / "selection_record.json").write_text(json.dumps({"frozen": True, "selection_split": "validation",
                                                             "selection_criterion_value": v}))
        (d / "run_complete.json").write_text("{}")


def main() -> int:
    """Every temporary directory is removed on exit, pass or fail (the suite writes checkpoints)."""
    import shutil
    made = []
    orig_mk = tempfile.mkdtemp

    def tracked_mkdtemp(*a, **k):
        d = orig_mk(*a, **k); made.append(d); return d
    tempfile.mkdtemp = tracked_mkdtemp
    try:
        return _main()
    finally:
        tempfile.mkdtemp = orig_mk
        for d in made:
            shutil.rmtree(d, ignore_errors=True)
        left = [d for d in made if Path(d).exists()]
        print(f"temporary directories removed: {len(made) - len(left)} of {len(made)}")


def _main() -> int:
    td = Path(tempfile.mkdtemp()); rows = synth_bank(td); exp = td / "R6_EXPERIMENTS"
    frozen = RN.frozen_inputs(td)
    # D01 roles cannot be mixed
    g_tr, g_fi, g_fe = RN.AccessGate("training"), RN.AccessGate("final_internal"), RN.AccessGate("final_external")
    ok = all(refuses(lambda s=s: g_tr.check(s)) for s in ("test", "EXTERNAL_TEST_ONLY")) \
        and all(refuses(lambda s=s: g_fi.check(s)) for s in ("train", "validation", "EXTERNAL_TEST_ONLY")) \
        and all(refuses(lambda s=s: g_fe.check(s)) for s in ("train", "validation", "test"))
    add("D01", "train / validation / test / external roles cannot be mixed across modes", ok)
    # D02 mu selection: TRAIN + VAL only; grid configs only; seed 101 only
    g_mu = RN.AccessGate("mu_selection")
    ok = all(refuses(lambda s=s: g_mu.check(s)) for s in ("test", "EXTERNAL_TEST_ONLY")) and not refuses(lambda: (g_mu.check("train"), g_mu.check("validation")))
    ok &= refuses(lambda: RN.train("fedavg", 101, td, "cpu", mode="mu_selection", _clients=synth_clients(), _exp_root=exp, _frozen=frozen))
    ok &= refuses(lambda: RN.train("fedprox_mu_0.01", 101, td, "cpu", mode="training", _clients=synth_clients(), _exp_root=exp, _frozen=frozen))
    ok &= refuses(lambda: RN.train("fedprox_mu_0.01", 202, td, "cpu", mode="mu_selection", _clients=synth_clients(), _exp_root=exp, _frozen=frozen))
    add("D02", "mu selection reaches TRAIN+VAL only, runs only the grid configs, and only on seed 101", ok)
    # D03 selection reads VAL only
    ok = refuses(lambda: SE.select([{"eval_index": 0, "split": "test", "criterion": 1.0}]))
    bad_exp = td / "bad"; grid_records(bad_exp, {0.001: 1.0, 0.01: 0.9, 0.1: 0.8, 1.0: 0.7})
    rp = bad_exp / "mu_selection" / "fedprox_mu_0.1__seed101" / "selection_record.json"
    rp.write_text(json.dumps({"frozen": True, "selection_split": "test", "selection_criterion_value": 0.1}))
    ok &= refuses(lambda: RN.select_mu(bad_exp))
    add("D03", "checkpoint and mu selection accept validation records only", ok)
    # D04 internal test unreachable from training / mu mode (the real data builder requests TRAIN/VAL only)
    import bmc_cache as CA
    seen = []
    orig = CA.CachedBMCSplitDataset.__init__

    def rec_init(self, ds, split, *a, **k):
        seen.append(split); return orig(self, ds, split, *a, **k)
    cache_dir = exp / "_cache" / "deterministic_256"
    cache_rec = RN.build_cache(td, cache_dir)
    CA.CachedBMCSplitDataset.__init__ = rec_init
    try:
        cl = RN.build_training_clients(RN.AccessGate("training"), rows, td, cache_dir)
    finally:
        CA.CachedBMCSplitDataset.__init__ = orig
    add("D04", "the training data builder requests only TRAIN and VALIDATION; test is unreachable", set(seen) == {"train", "validation"} and
        refuses(lambda: RN.build_training_clients(RN.AccessGate("final_internal"), rows, td, cache_dir)), f"splits requested: {sorted(set(seen))}")
    add("D11", "cache built from TRAIN+VAL only and verified bit-identical to R3 before use",
        cache_rec["mismatches"] == 0 and cache_rec["splits"] == ["train", "validation"] and len(cl) == 6 and cl[0].code == US and cl[5].code == MM,
        f"mismatches {cache_rec['mismatches']}; {len(cl)} clients built")
    (cache_dir / "cache_verified.json").write_text(json.dumps({"mismatches": 1}))
    add("D11b", "an unverified or mismatching cache blocks training", refuses(lambda: RN.build_training_clients(RN.AccessGate("training"), rows, td, cache_dir)))
    (cache_dir / "cache_verified.json").write_text(json.dumps({"mismatches": 0}))
    # D06 frozen seeds
    ok = refuses(lambda: RN.train("fedavg", 7, td, "cpu", _clients=synth_clients(), _exp_root=exp, _frozen=frozen))
    rep = td / "CANONICAL_DATASET_BANK" / "_bmc_splits" / "BMC_SPLITS_REPORT.json"; good = rep.read_text()
    rep.write_text(json.dumps({"seed": 1234, "verdict": "BMC_SPLITS_LOCKED"})); ok &= refuses(lambda: RN.frozen_inputs(td)); rep.write_text(good)
    ok &= all(RN.load_config(n)["training_seeds"] == [101, 202, 303] for n in ("proposed", "fedavg", "local"))
    add("D06", "only seeds {101,202,303} train; a manifest not built with SPLIT_SEED 20260929 is refused", ok)
    # D07 unresolved mu blocks FedProx / Proposed / mu-dependent ablations
    blocked = {n: refuses(lambda n=n: RN.train(n, 101, td, "cpu", _clients=synth_clients(), _exp_root=exp, _frozen=frozen, _overrides=FAST))
               for n in ("proposed", "fedprox", "N2_uncapped", "N3_cls_only", "N4_seg_only")}
    fixed = RN.resolve_mu(RN.load_config("fedavg"), exp)
    add("D07", "unresolved mu blocks FedProx, Proposed and the mu-dependent ablations", all(blocked.values()) and fixed["mu"] == 0.0, f"{blocked}")
    grid_records(exp, {0.001: 0.90, 0.01: 0.80, 0.1: 0.80, 1.0: 0.95})
    mrec = RN.select_mu(exp)
    add("D07b", "select-mu freezes the grid minimum from validation records; exact tie -> smaller mu",
        mrec["selected_mu"] == 0.01 and mrec["frozen"] and mrec["selection_split"] == "validation", f"selected mu {mrec['selected_mu']}")
    # D08 + D09: train through the driver, check the manifest, then final evaluation
    run = RN.train("proposed", 101, td, "cpu", _clients=synth_clients(), _exp_root=exp, _frozen=frozen, _overrides=FAST)
    man = json.loads((run / "run_manifest.json").read_text())
    need = {"method", "seed", "mu", "checkpoint_selection", "environment", "frozen_input_sha256", "config_sha256", "split_seed", "bootstrap_seed"}
    fin_keys = {"cohort_manifest", "splits_manifest", "splits_report", "bmc_data", "bmc_eval", "bmc_stats"}
    ok = need <= set(man) and man["mu"]["mu"] == 0.01 and "mu_record_sha256" in man["mu"] and fin_keys <= set(man["frozen_input_sha256"]) \
        and man["checkpoint_selection"]["criterion"] == "joint_val_loss_client_macro" and man["split_seed"] == 20260929
    add("D08", "run manifest records method, seed, mu (+ record hash), selection rule, environment and frozen-input hashes", ok,
        f"{sorted(need & set(man))}")
    add("D08b", "runs are write-once: rerunning the same config and seed is refused",
        refuses(lambda: RN.train("proposed", 101, td, "cpu", _clients=synth_clients(), _exp_root=exp, _frozen=frozen, _overrides=FAST)))
    srec = json.loads((run / "selection_record.json").read_text())
    add("D03b", "the run's selection record is frozen and validation-derived", srec["frozen"] and srec["selection_split"] == "validation")
    # Final evaluation is tested on a CONSTRUCTED run: its contract does not depend on how the
    # checkpoint was trained, and a one-step synthetic model can emit logits that all lie in
    # [0, 1], which the locked R4 double-sigmoid guard (correctly, fail-closed) refuses.
    import bmc_model as MDL
    torch.manual_seed(1); cm = MDL.BMCModel()
    with torch.no_grad():
        for h in (cm.us_seg_head, cm.mm_seg_head):
            h.weight.mul_(0.01); h.bias.fill_(-4.0)             # background-dominated, as after training
        cm.cls_head[-1].bias.fill_(-1.5)                          # classification logits clearly outside [0, 1]
    run = exp / "runs" / "constructed__seed101"
    RN.write_once(run / "run_manifest.json", RN.run_manifest(cfg_name="proposed", cfg=RN.load_config("fedavg"), seed=101,
                  mu_prov={"mu": 0.0, "source": "constructed test run"}, frozen=frozen, device="cpu", overrides={"constructed": True}))
    (run / "checkpoints").mkdir(parents=True); torch.save(cm.state_dict(), run / "checkpoints" / "best.pt")
    h = [{"eval_index": 0, "split": "validation", "criterion": 1.0}]
    srec = SE.freeze_record(run / "selection_record.json", run_id="constructed__seed101", method="fedavg", seed=101,
                            history=h, best_so_far_index=0, checkpoint_sha256=RN.sha256_file(run / "checkpoints" / "best.pt"))
    (run / "run_complete.json").write_text("{}")
    # D05 external unreachable before explicit final-external mode
    ok = refuses(lambda: RN.final_evaluate(run, td, "final_external", "cpu", _rows=rows))
    ints = RN.final_evaluate(run, td, "final_internal", "cpu", _rows=rows)
    no_ext = not any("EXTERNAL" in p for p in ints) and len(ints) == 6
    exts = RN.final_evaluate(run, td, "final_external", "cpu", confirm_external=True, _rows=rows)
    import bmc_stats as ST
    ledgers_ok = all(ST.load_ledger(p).n > 0 for p in ints + exts)
    add("D05", "external data needs explicit --confirm-final-external; final-internal writes only internal-test ledgers",
        ok and no_ext and len(exts) == 1 and ledgers_ok, f"{len(ints)} internal ledgers, {len(exts)} external ledger, all hash-verified")
    fresh = exp / "runs" / "noselect"; fresh.mkdir(parents=True)
    add("D09", "final evaluation is refused without a frozen selection record", refuses(lambda: RN.final_evaluate(fresh, td, "final_internal", "cpu", _rows=rows)))
    ck = run / "checkpoints" / "best.pt"; b = bytearray(ck.read_bytes()); b[-1] ^= 1; ck.write_bytes(bytes(b))
    add("D12", "a checkpoint that does not match the frozen record's SHA-256 is refused", refuses(lambda: RN.load_selected_checkpoint(run, srec)))
    # D13-D15 local-only final evaluation (frozen R5.1 scope)
    lrun = exp / "runs" / "local_constructed__seed101"
    RN.write_once(lrun / "run_manifest.json", RN.run_manifest(cfg_name="local", cfg=RN.load_config("local"), seed=101,
                  mu_prov={"mu": 0.0, "source": "fixed"}, frozen=frozen, device="cpu", overrides={"constructed": True}))
    for ds in RN.client_datasets(rows):
        sub = lrun / "local" / ds; (sub / "checkpoints").mkdir(parents=True)
        torch.save(cm.state_dict(), sub / "checkpoints" / "best.pt")
        SE.freeze_record(sub / "selection_record.json", run_id=f"local__{ds}", method="local", seed=101, history=h,
                         best_so_far_index=0, checkpoint_sha256=RN.sha256_file(sub / "checkpoints" / "best.pt"))
    (lrun / "run_complete.json").write_text("{}")
    lints = RN.final_evaluate(lrun, td, "final_internal", "cpu", _rows=rows)
    per_client = all(ST.load_ledger(pth).dataset == Path(pth).parent.parent.name and ST.load_ledger(pth).split == "test" for pth in lints)
    add("D13", "local-only: each client's own selected model is evaluated on THAT client's internal test only",
        len(lints) == 6 and per_client, f"{len(lints)} ledgers, each in its own client sub-run and on its own test split")
    add("D14", "local-only models are refused on external datasets, even with --confirm-final-external",
        refuses(lambda: RN.final_evaluate(lrun, td, "final_external", "cpu", confirm_external=True, _rows=rows)))
    tampered = lrun / "local" / "DMID" / "checkpoints" / "best.pt"; bb = bytearray(tampered.read_bytes()); bb[-1] ^= 1; tampered.write_bytes(bytes(bb))
    lrun2 = exp / "runs" / "local_missing__seed101"
    RN.write_once(lrun2 / "run_manifest.json", RN.run_manifest(cfg_name="local", cfg=RN.load_config("local"), seed=101,
                  mu_prov={"mu": 0.0, "source": "fixed"}, frozen=frozen, device="cpu", overrides={"constructed": True}))
    (lrun2 / "local" / "BUSI").mkdir(parents=True); (lrun2 / "run_complete.json").write_text("{}")
    import shutil; shutil.rmtree(lrun / "local" / "DMID" / "ledgers", ignore_errors=True)
    for ds in RN.client_datasets(rows):
        shutil.rmtree(lrun / "local" / ds / "ledgers", ignore_errors=True)
    add("D15", "local sub-runs keep the SHA-256 check and the frozen-record requirement",
        refuses(lambda: RN.final_evaluate(lrun, td, "final_internal", "cpu", _rows=rows))
        and refuses(lambda: RN.final_evaluate(lrun2, td, "final_internal", "cpu", _rows=rows)),
        "tampered client checkpoint refused; client sub-run without a frozen record refused")
    # D10 preflight
    add("D10", "preflight refuses without CUDA (no CPU fallback)", (not torch.cuda.is_available() and RN.preflight() == 2) or torch.cuda.is_available())
    import bmc_model as MDL
    enc = MDL.BMCModel().encoder.state_dict()
    fake_rgb = {k: (torch.randn(64, 3, 7, 7) if k == "conv1.weight" else (torch.randn_like(v) if torch.is_floating_point(v) else v.clone()))
                for k, v in enc.items()}                      # stand-in for the ImageNet download (network blocked here)
    r = RN.one_step("cpu", fake_rgb, 16, 64, bf16=False)
    add("D10b", "one_step performs exactly one optimizer step at batch 16 and reports the required fields",
        r["optimizer_steps"] == 1 and r["batch"] == 16 and "single_step_wall_s" in r and "smp_decoder_varargs_path" in r,
        f"loss {r['loss']:.4f}, wall {r['single_step_wall_s']:.2f} s (CPU code-path check only)")
    # D16 production preflight can never silently use a random encoder
    def boom():
        raise OSError("download failed")
    no_rand = refuses(lambda: RN.one_step("cpu", None, 16, 64, bf16=False)) and refuses(lambda: RN.one_step("cpu", {}, 16, 64, bf16=False)) \
        and refuses(lambda: RN._preflight("cpu", lambda: None, 64))
    try:
        RN._preflight("cpu", boom, 64); fetch_fail_propagates = False
    except OSError:
        fetch_fail_propagates = True
    rp = RN._preflight("cpu", lambda: fake_rgb, 64)
    exp_w = fake_rgb["conv1.weight"].sum(1, keepdim=True)
    import inspect
    src = inspect.getsource(RN.preflight)
    add("D16", "production preflight uses fetch_imagenet_rgb_encoder_state + BMCModel(rgb_encoder_state=...); "
               "a missing/failed download is fatal, never replaced by a random encoder",
        no_rand and fetch_fail_propagates and rp["pretrained_sha256"] and len(rp["pretrained_sha256"]) == 64
        and rp["conv1_equals_rgb_sum"] is True and rp["conv1_shape"] == (64, 1, 7, 7)
        and "fetch_imagenet_rgb_encoder_state" in src and "BMCModel()" not in inspect.getsource(RN.one_step),
        f"pretrained_sha256 {rp['pretrained_sha256'][:16]}..., conv1 == RGB sum: {rp['conv1_equals_rgb_sum']}, "
        f"download failure propagates: {fetch_fail_propagates}")
    # ---------------- NON-STUDY smoke (S01-S09)
    import bmc_config as CFG
    g_sm = RN.AccessGate("smoke")
    add("S01", "smoke mode reaches TRAIN and VALIDATION only",
        all(refuses(lambda s_=s_: g_sm.check(s_)) for s_ in ("test", "EXTERNAL_TEST_ONLY")) and not refuses(lambda: (g_sm.check("train"), g_sm.check("validation"))))
    sexp = td / "smoke_exp"; calls = {"fetch": 0}

    def fetch():
        calls["fetch"] += 1; return fake_rgb
    cpu_smoke = lambda **k: RN.smoke(td, "cpu", _clients=synth_clients8(32), _exp_root=sexp, _frozen=frozen, _fetch=fetch, **k)
    orig_bf16 = CFG.BASE["mixed_precision"]
    sd = cpu_smoke()
    rep = json.loads((sd / "smoke_report.json").read_text()); sman = json.loads((sd / "smoke_manifest.json").read_text())
    add("S02", "all 8 clients and both modalities are exercised",
        len(rep["clients"]) == 8 and {c["modality_code"] for c in rep["clients"]} == {0, 1}, f"{[c['name'] for c in rep['clients']]}")
    add("S03", "exactly one round and one local epoch (steps per client = floor(N/16))",
        rep["rounds_run"] == 1 and rep["local_epochs"] == 1 and all(c["steps"] == c["expected_steps"] == 2 for c in rep["clients"])
        and sman["smoke_overrides"] == {"rounds": 1, "local_epochs": 1}, f"steps {[c['steps'] for c in rep['clients']]}")
    add("S04", "batch size stays 16 in every optimizer step",
        rep["batch_size"] == 16 and rep["batch_sizes_seen"] == [16] and rep["optimizer_steps"] == 16, f"batch sizes seen {rep['batch_sizes_seen']}")
    add("S05", "production ImageNet path used; no random-encoder fallback",
        calls["fetch"] == 1 and len(sman["pretrained_sha256"]) == 64 and refuses(lambda: RN.smoke(td, "cpu", _clients=synth_clients8(32), _exp_root=sexp, _frozen=frozen, _fetch=lambda: None)),
        f"pretrained_sha256 {sman['pretrained_sha256'][:16]}...")
    add("S06", "no mu-selection artefact is created or modified",
        not (sexp / "mu_selection").exists() and sman["mu"]["mu"] == 0.0 and "no mu selection" in sman["mu"]["source"])
    add("S07", "no formal run directory is created", not (sexp / "runs").exists() and not (sd / "run_manifest.json").exists())
    add("S08", "outputs marked NON_STUDY_SMOKE, ineligible for results, refused by final evaluation; frozen configs unchanged",
        (sd / RN.SMOKE_MARKER).is_file() and sman["status"] == rep["status"] == "NON_STUDY_SMOKE" and sman["eligible_for_results"] is False
        and refuses(lambda: RN.final_evaluate(sd, td, "final_internal", "cpu", _rows=rows))
        and CFG.build()["N1_mu0"]["rounds"] == 50 and CFG.build()["N1_mu0"]["local_epochs"] == 2 and CFG.BASE["mixed_precision"] == orig_bf16 == "bf16_autocast",
        "bf16 production setting retained in the smoke config; CPU here runs without autocast")
    # S09 real frozen-manifest layout + verified cache + real R3 pipeline, 8 clients (synthetic images)
    td8 = Path(tempfile.mkdtemp()); rows8 = synth_bank8(td8); exp8 = td8 / "R6_EXPERIMENTS"
    RN.build_cache(td8, exp8 / "_cache" / "deterministic_256")
    seen8 = []
    orig8 = CA.CachedBMCSplitDataset.__init__

    def rec8(self, ds, split, *a, **k):
        seen8.append(split); return orig8(self, ds, split, *a, **k)
    CA.CachedBMCSplitDataset.__init__ = rec8
    try:
        sd8 = RN.smoke(td8, "cpu", _exp_root=exp8, _fetch=fetch)
    finally:
        CA.CachedBMCSplitDataset.__init__ = orig8
    rep8 = json.loads((sd8 / "smoke_report.json").read_text())
    add("S09", "through the real manifest + verified cache + R3 pipeline: 8 clients, TRAIN/VAL only, validation after the round",
        set(seen8) == {"train", "validation"} and len(rep8["clients"]) == 8 and rep8["validation"]["split"] == "validation"
        and len(rep8["validation"]["per_client"]) == 8 and not (exp8 / "runs").exists() and not (exp8 / "mu_selection").exists(),
        f"splits requested {sorted(set(seen8))}; validation clients {len(rep8['validation']['per_client'])}")
    # ---------------- pre-registration and interrupted-run quarantine (Q01-Q08)
    qexp = td / "q_exp"

    def mk_incomplete(parent, name, seed=202, cfg="fedavg"):
        d = qexp / parent / name; (d / "checkpoints").mkdir(parents=True)
        RN.write_once(d / "run_manifest.json", {"run_id": name, "config_name": cfg, "method": cfg, "seed": seed,
                                                "mu": {"mu": 0.0, "source": "fixed"}, "frozen_input_sha256": {"x": "y"}})
        (d / "history.json").write_text(json.dumps([{"eval_index": 0, "split": "validation", "criterion": 1.2}]))
        (d / "checkpoints" / "best.pt").write_bytes(bytes(range(256)) * 40)
        return d
    done = mk_incomplete("runs", "qdone__seed101", 101); (done / "run_complete.json").write_text("{}")
    inv_done = RN._inventory(done)
    add("Q01", "a COMPLETED run is refused and left untouched",
        refuses(lambda: RN.quarantine_incomplete(done, "test")) and done.is_dir() and RN._inventory(done) == inv_done)
    inc = mk_incomplete("runs", "qinc__seed202"); inv = RN._inventory(inc)
    rec = RN.quarantine_incomplete(inc, "power loss during round 17")
    dest = Path(rec["destination"])
    add("Q02", "an incomplete run is moved intact to _aborted (never deleted): every file present with identical SHA-256",
        not inc.exists() and dest.is_dir() and dest.parent.name == "_aborted" and rec["file_sha256"] == inv
        and all(RN.sha256_file(dest / k) == v for k, v in inv.items()), f"{len(inv)} files preserved")
    on_disk = json.loads((dest / "ABORT_RECORD.json").read_text())
    need = {"original_path", "original_run_id", "destination", "config_name", "method", "seed", "mu",
            "quarantined_at", "reason", "manifest", "file_sha256", "restart_policy"}
    add("Q03", "provenance record: original path/run ID, destination, config/method, seed, mu, time, reason, manifest and hashes",
        need <= set(on_disk) and on_disk["seed"] == 202 and on_disk["config_name"] == "fedavg" and on_disk["mu"]["mu"] == 0.0
        and on_disk["reason"] == "power loss during round 17" and "no resume" in on_disk["restart_policy"]
        and (dest / RN.ABORT_MARKER).is_file() and (qexp / "_aborted" / "abort_log.jsonl").is_file())
    a1 = mk_incomplete("runs", "qdup__seed303", 303)
    r1 = RN.quarantine_incomplete(a1, "first", _stamp="FIXEDSTAMP")
    a2 = mk_incomplete("runs", "qdup__seed303", 303); inv2 = RN._inventory(a2)
    collide = refuses(lambda: RN.quarantine_incomplete(a2, "second", _stamp="FIXEDSTAMP"))
    r2 = RN.quarantine_incomplete(a2, "second")
    add("Q04", "never overwrites: a colliding destination is refused and the source kept; distinct stamps give distinct destinations",
        collide and r1["destination"] != r2["destination"] and RN._inventory(Path(r1["destination"])) != {} and r2["file_sha256"] == inv2)
    # Q05 exclusion from every downstream input
    good = mk_incomplete("runs", "qgood__seed101", 101); (good / "run_complete.json").write_text("{}")
    pending = mk_incomplete("runs", "qpending__seed202")                 # incomplete, not yet quarantined
    found = [d.name for d in RN.discover_completed_runs(qexp)]
    gexp = td / "g_exp"; grid_records(gexp, {0.001: 0.9, 0.01: 0.8, 0.1: 0.7, 1.0: 0.95})
    (gexp / "mu_selection" / "fedprox_mu_0.1__seed101" / "run_complete.json").unlink()
    incomplete_grid_refused = refuses(lambda: RN.select_mu(gexp))
    RN.quarantine_incomplete(gexp / "mu_selection" / "fedprox_mu_0.1__seed101", "interrupted grid run")
    aborted_grid_refused = refuses(lambda: RN.select_mu(gexp))
    add("Q05", "aborted and incomplete runs are never valid inputs: discovery, mu selection and final evaluation exclude them",
        "qgood__seed101" in found and "qdone__seed101" in found and "qpending__seed202" not in found
        and not any("aborted" in f for f in found) and incomplete_grid_refused and aborted_grid_refused
        and refuses(lambda: RN.final_evaluate(dest, td, "final_internal", "cpu", _rows=rows))
        and refuses(lambda: RN.final_evaluate(pending, td, "final_internal", "cpu", _rows=rows))
        and not (gexp / "mu_selection" / "mu_record.json").exists(), f"discovered: {found}")
    # Q06 restart from round 0 with the same frozen config and seed
    rexp = td / "r_exp"; stale = rexp / "runs" / "fedavg__seed101"; stale.mkdir(parents=True)
    RN.write_once(stale / "run_manifest.json", {"run_id": "fedavg__seed101", "config_name": "fedavg", "method": "fedavg", "seed": 101})
    blocked_before = refuses(lambda: RN.train("fedavg", 101, td, "cpu", _clients=synth_clients(), _exp_root=rexp, _frozen=frozen, _overrides=FAST))
    RN.quarantine_incomplete(stale, "simulated interruption")
    fresh_run = RN.train("fedavg", 101, td, "cpu", _clients=synth_clients(), _exp_root=rexp, _frozen=frozen, _overrides=FAST)
    hist = json.loads((fresh_run / "history.json").read_text())
    add("Q06", "restart policy: blocked while the stale run exists; after quarantine the same config/seed restarts from round 0",
        blocked_before and hist[0]["eval_index"] == 0 and (fresh_run / "run_complete.json").is_file()
        and len(list((rexp / "_aborted").glob("runs__fedavg__seed101__aborted_*"))) == 1)
    odd = td / "elsewhere" / "x__seed101"; odd.mkdir(parents=True)
    smk = qexp / "_smoke" / "smoke__1"; smk.mkdir(parents=True)
    add("Q07", "quarantine refuses non-formal locations (smoke, arbitrary) and requires a reason",
        refuses(lambda: RN.quarantine_incomplete(odd, "x")) and refuses(lambda: RN.quarantine_incomplete(smk, "x"))
        and refuses(lambda: RN.quarantine_incomplete(pending, "   ")) and pending.is_dir())
    # Q08 the pre-registration is frozen into every formal manifest and required
    fi = RN.frozen_inputs(td)
    trained = json.loads((exp / "runs" / "proposed__seed101" / "run_manifest.json").read_text())
    saved = RN.PREREGISTRATION
    RN.PREREGISTRATION = td / "missing.md"
    try:
        missing_refused = refuses(lambda: RN.frozen_inputs(td))
    finally:
        RN.PREREGISTRATION = saved
    txt = saved.read_text(encoding="utf-8")
    add("Q08", "the pre-registration is hashed into every formal run manifest, and formal runs are refused without it",
        fi["preregistration"] == RN.sha256_file(saved) == trained["frozen_input_sha256"]["preregistration"] and missing_refused
        and "0.5182" in txt and "0.4818" in txt and "Capped within-modality weighting is applied to all applicable parameter groups" in txt,
        f"preregistration sha256 {fi['preregistration'][:16]}...")
    nf = sum(1 for t in T if not t[2])
    print(f"PASS = {len(T) - nf}   FAIL = {nf}"); print("R6_DRIVER_VERIFIED" if nf == 0 else "R6_DRIVER_BLOCKED")
    return 0 if nf == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
