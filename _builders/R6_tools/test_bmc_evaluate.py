#!/usr/bin/env python3
"""test_bmc_evaluate.py - synthetic-only tests of bmc_evaluate.py (no real data, no real checkpoints)."""
from __future__ import annotations
import csv, hashlib, json, shutil, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bmc_evaluate as EV                                   # sets the R5/R4 import paths
import numpy as np, torch
from PIL import Image
import bmc_run as RN, bmc_select as SE, bmc_stats as ST, bmc_data as BD
from bmc_model import BMCModel
torch.set_num_threads(1)
CL = [("BUSI", "Ultrasound", 0), ("BUS-UC", "Ultrasound", 1), ("BrEaST", "Ultrasound", 2), ("BUS-BRA", "Ultrasound", 3),
      ("INbreast", "Mammography", 4), ("DMID", "Mammography", 5), ("CBIS-DDSM", "Mammography", 6), ("AIISLAB", "Mammography", 7)]
T = []


def add(tid, desc, ok, detail=""):
    T.append(bool(ok)); print(f"  {tid} [{'PASS' if ok else 'FAIL'}]  {desc}" + (f"\n        {detail}" if detail else ""), flush=True)


def refuses(fn):
    try:
        fn(); return False
    except (EV.EvalError, ST.ContractError, __import__('bmc_eval').ContractError, RuntimeError, FileExistsError):
        return True


def bank(root: Path) -> None:
    rng = np.random.default_rng(0); rows = []; img = root / "imgs"; img.mkdir(parents=True)
    spec = [(d, m, c, "CLIENT", [("train", 4), ("validation", 4), ("test", 4)]) for d, m, c in CL] + \
           [("BLUIS", "Ultrasound", "", "EXTERNAL", [("EXTERNAL_TEST_ONLY", 4)])]
    for ds, mod, cid, role, splits in spec:
        for sp, n in splits:
            for i in range(n):
                rid = f"{ds}_{sp}_{i}"; ip, mp = img / f"{rid}.png", img / f"{rid}_m.png"
                if ds in ("INbreast", "CBIS-DDSM"):
                    Image.fromarray(rng.integers(0, 4096 if ds == "INbreast" else 65536, (48, 40)).astype(np.uint16)).save(ip)
                else:
                    Image.fromarray(rng.integers(0, 256, (48, 40)).astype(np.uint8), "L").save(ip)
                mk = np.zeros((48, 40), np.uint8); mk[10:30, 8:28] = 255; Image.fromarray(mk, "L").save(mp)
                rows.append({"client_id": cid, "dataset": ds, "modality": mod, "role": role, "split": sp,
                             "canonical_record_id": rid, "canonical_image_abspath": str(ip), "canonical_mask_abspath": str(mp),
                             "label": "", "label_binary": str(i % 2), "group_key": f"G_{rid}", "group_unit": "IMAGE", "seed": 20260929})
    b = root / "CANONICAL_DATASET_BANK"; (b / "_bmc_splits").mkdir(parents=True); (b / "_bmc_cohort").mkdir(parents=True)
    with (b / "_bmc_splits" / "BMC_SPLITS_MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    (b / "_bmc_splits" / "BMC_SPLITS_REPORT.json").write_text(json.dumps({"seed": 20260929, "verdict": "BMC_SPLITS_LOCKED"}))
    (b / "_bmc_cohort" / "BMC_COHORT_MANIFEST.csv").write_text("synthetic\n")


def model_state(seed):
    torch.manual_seed(seed); m = BMCModel()
    with torch.no_grad():
        for h in (m.us_seg_head, m.mm_seg_head): h.weight.mul_(0.01); h.bias.fill_(-4.0)
        m.cls_head[-1].bias.fill_(-0.5)
    return m.state_dict()


def selected(dirp: Path, run_id, method, seed, st):
    (dirp / "checkpoints").mkdir(parents=True, exist_ok=True); torch.save(st, dirp / "checkpoints" / "best.pt")
    SE.freeze_record(dirp / "selection_record.json", run_id=run_id, method=method, seed=seed,
                     history=[{"eval_index": 0, "split": "validation", "criterion": 1.0}], best_so_far_index=0,
                     checkpoint_sha256=EV.sha_file(dirp / "checkpoints" / "best.pt"))
    (dirp / "history.json").write_text("[]")


def study(root: Path, ref):
    runs = root / "R6_EXPERIMENTS" / "runs"
    for cfg in ("fedavg", "local"):
        d = runs / f"{cfg}__seed101"; d.mkdir(parents=True)
        RN.write_once(d / "run_manifest.json", RN.run_manifest(cfg_name=cfg, cfg=RN.load_config(cfg), seed=101,
                      mu_prov={"mu": 0.0, "source": "fixed in the frozen config"}, frozen=ref, device="cpu", overrides={}))
        if cfg == "local":
            for k, (ds, _, _) in enumerate(CL):
                selected(d / "local" / ds, f"local__seed101__{ds}", "local", 101, model_state(10 + k))
        else:
            selected(d, "fedavg__seed101", "fedavg", 101, model_state(1))
        (d / "run_complete.json").write_text("{}")


def tree(p: Path):
    return {str(f.relative_to(p)): EV.sha_file(f) for f in sorted(p.rglob("*")) if f.is_file()} if p.exists() else {}


def main() -> int:
    made = []
    try:
        root = Path(tempfile.mkdtemp()); made.append(root)
        bank(root); ref = EV.frozen_inputs_now(root); study(root, ref)
        exp, rows = root / "R6_EXPERIMENTS", RN.read_manifest(root)
        EXP2 = ("fedavg__seed101", "local__seed101"); names = [c[0] for c in CL]
        gate = lambda **k: EV.preflight(exp, root, expected_runs=k.get("runs", EXP2), ref=k.get("ref", ref), client_names=names)
        # E01 gate passes on a complete synthetic study
        tok = gate()
        add("E01", "study gate passes on a complete study; 1 federated + 8 local selected models",
            tok.record["gate"] == "passed" and len(tok.units) == 9, f"{len(tok.units)} model units")
        # E02-E04 fail-closed: the gate raises BEFORE any image is read
        calls = {"n": 0}; orig = BD.BMCSplitDataset.__init__

        def spy(self, *a, **k):
            calls["n"] += 1; return orig(self, *a, **k)
        BD.BMCSplitDataset.__init__ = spy
        try:
            missing = refuses(lambda: gate(runs=EXP2 + ("proposed__seed101",)))
            # a frozen input changed ON DISK after the runs (manifests and reference still agree)
            coh = root / "CANONICAL_DATASET_BANK" / "_bmc_cohort" / "BMC_COHORT_MANIFEST.csv"; cb0 = coh.read_bytes()
            coh.write_bytes(cb0 + b"tampered\n"); changed = refuses(lambda: gate()); coh.write_bytes(cb0)
            # one run's manifest carries a different fingerprint (reference and disk still agree)
            mpth = exp / "runs" / "fedavg__seed101" / "run_manifest.json"; mb0 = mpth.read_text()
            mj = json.loads(mb0); mj["frozen_input_sha256"]["r5:bmc_model.py"] = "0" * 64; mpth.write_text(json.dumps(mj))
            manifest_mismatch = refuses(lambda: gate()); mpth.write_text(mb0)
            m = json.loads((exp / "runs" / "fedavg__seed101" / "run_manifest.json").read_text())
            (exp / "runs" / "fedavg__seed101" / "run_complete.json").unlink()
            incomplete = refuses(lambda: gate())
            (exp / "runs" / "fedavg__seed101" / "run_complete.json").write_text("{}")
        finally:
            BD.BMCSplitDataset.__init__ = orig
        add("E02", "gate CLOSED for a missing run (27/27 not proven), and no image was read", missing and calls["n"] == 0)
        add("E03", "gate CLOSED when a frozen input on disk changed after the runs", changed and calls["n"] == 0)
        add("E03b", "gate CLOSED when run manifests do not carry the reference fingerprints", manifest_mismatch and calls["n"] == 0)
        add("E04", "gate CLOSED for an incomplete run (no run_complete.json)", incomplete and calls["n"] == 0)
        # E05 no token, no evaluation
        add("E05", "evaluation refused without a passed study-gate token",
            refuses(lambda: EV.evaluate_unit(object(), tok.units[0], rows, "cpu")))
        # E06 order: test/external refused before a frozen tau_val artifact
        u = tok.units[0]; model, rec, rec_p = EV.load_unit_model(tok, u, "cpu")
        val = [EV._ledger(tok, u, model, rec, rec_p, rows, ds, "validation", "cpu") for ds in EV.targets(u, rows, "validation")]
        add("E06", "internal-test and external ledgers refused before the tau_val artifact exists",
            refuses(lambda: EV.post_selection_ledgers(tok, u, model, rec, rec_p, rows, "test", val, "cpu"))
            and refuses(lambda: EV.post_selection_ledgers(tok, u, model, rec, rec_p, rows, "EXTERNAL_TEST_ONLY", val, "cpu"))
            and not any((EV.unit_out(tok, u) / "ledgers").glob("*__test.csv")))
        art = EV.freeze_tau(tok, u, rec, val); out = EV.unit_out(tok, u)
        obj = json.loads(art.read_text())
        t_direct = ST.derive_tau_val([ST.load_ledger(p) for p in val])["tau_val"]
        add("E07", "tau_val artifact = frozen R4 derive_tau_val on the validation ledgers; bound to checkpoint and ledgers",
            obj["tau_val"] == t_direct and obj["checkpoint_sha256"] == rec["checkpoint_sha256"]
            and len(obj["validation_ledgers_sha256"]) == 8 and (out / "tau_val.sha256").is_file(), f"tau_val {obj['tau_val']:.4f}")
        # E08 tampering with the artifact, its binding, or a validation ledger closes the gate
        saved = art.read_text()
        art.write_text(saved.replace(str(obj["tau_val"]), "0.123456"))
        t1 = refuses(lambda: EV.require_tau(tok, u, rec, val))
        forged = dict(obj, checkpoint_sha256="0" * 64); art.write_text(json.dumps(forged))
        (out / "tau_val.sha256").write_text(EV.sha_file(art))
        t2 = refuses(lambda: EV.require_tau(tok, u, rec, val))
        art.write_text(saved); (out / "tau_val.sha256").write_text(EV.sha_file(art))
        vb = val[0].read_bytes(); val[0].write_bytes(vb + b"\n")
        t3 = refuses(lambda: EV.require_tau(tok, u, rec, val))
        val[0].write_bytes(vb)
        add("E08", "a tampered tau_val, a tau_val bound to another checkpoint, or a changed validation ledger is refused", t1 and t2 and t3)
        add("E09", "write-once: the tau_val artifact is never re-derived or overwritten",
            refuses(lambda: EV._write_once_json(tok, art, {"x": 1})) and EV.freeze_tau(tok, u, rec, val) == art and art.read_text() == saved)
        add("E10", "writing outside R6_EXPERIMENTS/_evaluation is refused", refuses(lambda: EV._safe(tok, exp / "runs" / "x.json"))
            and refuses(lambda: EV._safe(tok, root / "elsewhere.json")))
        # E11 full evaluation; nothing written into formal runs or R5_training
        before_runs, before_r5 = tree(exp / "runs"), tree(EV.BUILDERS / "R5_training")
        man_path = EV.run(tok, rows, "cpu")
        after_runs, after_r5 = tree(exp / "runs"), tree(EV.BUILDERS / "R5_training")
        outside = [p for p in exp.rglob("*") if p.is_file() and "_evaluation" not in p.parts and "runs" not in p.parts]
        add("E11", "formal run directories and R5_training are byte-identical after evaluation; no file outside _evaluation",
            before_runs == after_runs and before_r5 == after_r5 and not outside)
        M = json.loads(man_path.read_text()); fed = [x for x in M["units"] if x["client"] is None][0]
        loc = [x for x in M["units"] if x["client"]]
        add("E12", "federated model: 8 validation + 8 test + 1 external ledgers, plus its tau_val",
            sum(n.endswith("__validation.csv") for n in fed["ledgers"]) == 8 and sum(n.endswith("__test.csv") for n in fed["ledgers"]) == 8
            and sum(n.endswith("__EXTERNAL_TEST_ONLY.csv") for n in fed["ledgers"]) == 1, f"{len(fed['ledgers'])} ledgers")
        own = all(set(x["ledgers"]) == {f"{x['client']}__validation.csv", f"{x['client']}__test.csv"} for x in loc)
        taus = [json.loads((tok.eval_root / "local__seed101" / "local" / x["client"] / "tau_val.json").read_text()) for x in loc]
        add("E13", "local-only: each client model has its own tau_val from its own validation, its own test, and NO external ledger",
            len(loc) == 8 and own and all(list(t["validation_ledgers_sha256"]) == [f"{t['client']}__validation.csv"] for t in taus))
        lu = [x for x in tok.units if x["local"]][0]
        add("E14", "external evaluation of a local-only model is refused explicitly",
            refuses(lambda: EV.targets(lu, rows, "EXTERNAL_TEST_ONLY")))
        ok_load = all(ST.load_ledger(p).n > 0 for p in (tok.eval_root).rglob("*.csv"))
        add("E15", "every ledger is a frozen R4 ledger that loads with its fingerprint verified", ok_load)
        tree1 = tree(tok.eval_root)
        EV.run(EV.preflight(exp, root, expected_runs=EXP2, ref=ref, client_names=names), rows, "cpu")
        tree2 = tree(tok.eval_root)
        same = all(tree2.get(k) == v for k, v in tree1.items())
        add("E16", "re-running reuses and re-verifies existing ledgers and tau_val; nothing is overwritten", same)
        ck = u["unit_dir"] / "checkpoints" / "best.pt"; cb = ck.read_bytes(); ck.write_bytes(cb + b"x")
        add("E17", "a checkpoint that no longer matches its frozen selection record is refused", refuses(lambda: EV.load_unit_model(tok, u, "cpu")))
        ck.write_bytes(cb)
        add("E18", "real-study expectation is exactly 27 run-equivalents (9 configurations x 3 seeds)",
            len(EV.EXPECTED_RUNS) == 27 and len(set(EV.EXPECTED_RUNS)) == 27)
        # ---------------- R4 G1 guard policy (n = 24 > the guard's 20-value threshold)
        import copy, torch.nn as nn
        fu = [x for x in tok.units if not x["local"]][0]; m0, rec0, rp0 = EV.load_unit_model(tok, fu, "cpu")
        n4 = dict(fu); n4["manifest"] = copy.deepcopy(fu["manifest"]); n4["manifest"]["config_name"] = "N4_seg_only"
        n4["manifest"]["config"]["loss"]["lambda_cls"] = 0.0
        lam1 = dict(n4); lam1["manifest"] = copy.deepcopy(n4["manifest"]); lam1["manifest"]["config"]["loss"]["lambda_cls"] = 1.0
        recs = [{"record_id": f"P{i}", "dataset": "BUSI", "client_id": 0, "split": "test", "group_key": f"G{i}", "group_unit": "IMAGE",
                 "modality": "ultrasound", "label": i % 2} for i in range(24)]
        prov = {"run_id": fu["run_id"], "method": fu["manifest"]["method"], "seed": fu["manifest"]["seed"],
                "selected_round": rec0["selected_round"], "selection_criterion_name": rec0["selection_criterion_name"],
                "selection_criterion_value": rec0["selection_criterion_value"], "model_sha256": rec0["checkpoint_sha256"],
                "splits_manifest_sha256": fu["manifest"]["frozen_input_sha256"]["splits_manifest"]}
        rng = np.random.default_rng(5); cls01 = rng.uniform(0.05, 0.95, 24)
        gts = np.zeros((24, 16, 16), np.uint8); gts[:, 4:12, 4:12] = 1
        seg_ok = np.where(gts == 1, 4.0, -4.0); seg01 = rng.uniform(0.1, 0.9, gts.shape)
        pol = tok.eval_root / "_policy_tests"; pol.mkdir(parents=True, exist_ok=True)
        go = lambda unit, model, name, seg: EV.write_with_guard_policy(tok, unit, model, out=pol / f"{name}__test.csv", records=recs,
                                                                     cls=cls01, seg=seg, gts=gts, prov=prov, rec_p=rp0, split="test")
        add("E19", "non-N4 model with classification logits in [0, 1]: the R4 G1 guard still refuses (no ledger)",
            refuses(lambda: go(fu, m0, "fedavg_like", seg_ok)) and not (pol / "fedavg_like__test.csv").exists())
        out = go(n4, m0, "n4_like", seg_ok); w = json.loads((pol / "n4_like__test.R4_G1_WAIVER.json").read_text())
        L = ST.load_ledger(out)
        add("E20", "N4 (lambda_cls = 0) with raw logits in [0, 1]: ledger written with the GENUINE values + write-once waiver record",
            np.allclose(np.sort(L.cls_prob), np.sort(1 / (1 + np.exp(-cls01)))) and w["ledger_sha256"] == EV.sha_file(out)
            and w["segmentation_guard"].startswith("active") and w["config_name"] == "N4_seg_only" and w["n"] == 24)
        add("E21", "N4: the segmentation guard is NOT waived (segmentation inputs in [0, 1] still refused)",
            refuses(lambda: go(n4, m0, "n4_seg01", seg01)) and not (pol / "n4_seg01__test.csv").exists())
        msig = copy.deepcopy(m0); msig.cls_head = nn.Sequential(msig.cls_head, nn.Sigmoid())
        add("E22", "N4 whose model contains an output Sigmoid: refused (structural proof of raw logits fails)",
            refuses(lambda: go(n4, msig, "n4_sigmoid", seg_ok)) and not EV.output_activation_free(msig) and EV.output_activation_free(m0))
        add("E23", "a manifest named N4 but with lambda_cls = 1: refused (waiver requires the untrained head)",
            refuses(lambda: go(lam1, m0, "n4_lam1", seg_ok)))
        orig_fn = __import__("bmc_eval")._probability_like
        add("E24", "after any waiver the frozen guard function is restored unchanged", orig_fn is EV.__dict__.get("_probability_like", orig_fn)
            and __import__("bmc_eval")._probability_like(np.full(24, 0.5)) is True)
        shutil.rmtree(pol)
    finally:
        for d in made: shutil.rmtree(d, ignore_errors=True)
    nf = T.count(False); print(f"PASS = {len(T) - nf}   FAIL = {nf}"); print("EVALUATION_TOOL_VERIFIED" if nf == 0 else "EVALUATION_TOOL_BLOCKED")
    return 0 if nf == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
