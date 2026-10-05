#!/usr/bin/env python3
"""test_bmc_analyze.py - synthetic-only tests of bmc_analyze.py (synthetic evaluation outputs; no study data)."""
from __future__ import annotations
import csv, json, shutil, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bmc_analyze as AN                                       # sets import paths
import numpy as np
import bmc_eval as EVL, bmc_stats as ST, bmc_select as SE
CLIENTS = [("BUSI", "ultrasound", 0), ("DMID", "mammography", 5)]; EXT = ("BLUIS", "ultrasound")
EFFECT = {"proposed": 2.0, "fedavg": 1.6, "fedprox": 1.6, "local": 0.4, "centralised": 2.0,
          "N1_mu0": 1.8, "N2_uncapped": 1.8, "N3_cls_only": 1.8, "N4_seg_only": 1.8}
T = []


def add(tid, desc, ok, detail=""):
    T.append(bool(ok)); print(f"  {tid} [{'PASS' if ok else 'FAIL'}]  {desc}" + (f"\n        {detail}" if detail else ""), flush=True)


def refuses(fn):
    try:
        fn(); return False
    except (AN.AnalysisError, ST.ContractError, FileExistsError):
        return True


def records(ds, mod, cid, split, n):
    return [{"record_id": f"{ds}_{split}_{i}", "dataset": ds, "client_id": cid, "split": split, "group_key": f"G{ds}{split}{i // 2}",
             "group_unit": "PATIENT", "modality": mod, "label": i % 2} for i in range(n)]


def ledger(path, recs, cfg, seed, rec_p, rng):
    y = np.array([r["label"] for r in recs]); eff = EFFECT[cfg]
    cls = eff * (2 * y - 1) + rng.normal(0, 1.5, len(y))
    gt = np.zeros((len(y), 16, 16), np.uint8); gt[:, 4:12, 4:12] = 1
    seg = np.where(gt == 1, eff, -eff) + rng.normal(0, 2.2, gt.shape)
    prov = {"run_id": f"{cfg}__seed{seed}", "method": cfg, "seed": seed, "selected_round": 0,
            "selection_criterion_name": "joint_val_loss_client_macro", "selection_criterion_value": 1.0,
            "model_sha256": "c" * 64, "splits_manifest_sha256": "s" * 64}
    path.parent.mkdir(parents=True, exist_ok=True)
    EVL.evaluate_split(records=recs, cls_logits=cls, seg_logits=seg, gt_masks=gt, provenance=prov, out_path=path,
                       selection_record=rec_p if recs[0]["split"] in ("test", "EXTERNAL_TEST_ONLY") else None)
    return path


def build(root: Path):
    ev = root / "R6_EXPERIMENTS" / "_evaluation"; recdir = root / "records"; recdir.mkdir(parents=True)
    rec_p = recdir / "sel.json"
    SE.freeze_record(rec_p, run_id="x", method="x", seed=101, history=[{"eval_index": 0, "split": "validation", "criterion": 1.0}],
                     best_so_far_index=0, checkpoint_sha256="c" * 64)
    units = []
    for k, run in enumerate(AN.EXPECTED_RUNS):
        cfg, seed = run.rsplit("__seed", 1); seed = int(seed)
        subunits = [(c, ev / run / "local" / c[0]) for c in CLIENTS] if cfg == "local" else [(None, ev / run)]
        for j, (cl, base) in enumerate(subunits):
            rng = np.random.default_rng(1000 * k + j)
            targets = [cl] if cl else CLIENTS
            val = [ledger(base / "ledgers" / f"{d}__validation.csv", records(d, m, c, "validation", 24), cfg, seed, rec_p, rng) for d, m, c in targets]
            test = [ledger(base / "ledgers" / f"{d}__test.csv", records(d, m, c, "test", 40), cfg, seed, rec_p, rng) for d, m, c in targets]
            ext = [] if cl else [ledger(base / "ledgers" / f"{EXT[0]}__EXTERNAL_TEST_ONLY.csv",
                                        records(EXT[0], EXT[1], "", "EXTERNAL_TEST_ONLY", 30), cfg, seed, rec_p, rng)]
            t = ST.derive_tau_val([ST.load_ledger(p) for p in val])
            art = base / "tau_val.json"
            art.write_text(json.dumps({"frozen": True, "tau_val": t["tau_val"], "derivation": t, "source_split": "validation",
                                       "run_id": run, "client": cl[0] if cl else None, "checkpoint_sha256": "c" * 64,
                                       "validation_ledgers_sha256": {p.name: AN.sha_file(p) for p in val}}, indent=2))
            (base / "tau_val.sha256").write_text(AN.sha_file(art) + "\n")
            units.append({"run_id": run, "client": cl[0] if cl else None, "checkpoint_sha256": "c" * 64, "tau_val": t["tau_val"],
                          "tau_val_artifact_sha256": AN.sha_file(art), "ledgers": {p.name: AN.sha_file(p) for p in val + test + ext}})
    man = ev / "evaluation_manifest__20261004T000000Z.json"
    man.write_text(json.dumps({"study_gate": {"gate": "passed", "runs": list(AN.EXPECTED_RUNS), "n_runs": 27}, "units": units}, indent=2))
    return ev, man


def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def tree(ev):
    return {str(p.relative_to(ev)): AN.sha_file(p) for p in sorted(ev.rglob("*")) if p.is_file() and "analysis" not in p.relative_to(ev).parts}


def main() -> int:
    root = Path(tempfile.mkdtemp())
    try:
        ev, man = build(root)
        before = tree(ev)
        out = AN.analyze(man, ev)
        files = sorted(p.name for p in out.iterdir())
        add("A01", "normal run: 6 tables + analysis_manifest, under _evaluation/analysis/<UTC>/ only",
            files == sorted(["confirmatory_tests.csv", "main_descriptive.csv", "ablations_descriptive.csv", "external_descriptive.csv",
                             "per_client_supplementary.csv", "tau_val_secondary.csv", "analysis_manifest.json"])
            and out.parent == (ev / "analysis").resolve() and tree(ev) == before, f"{len(files)} files")
        conf = read(out / "confirmatory_tests.csv")
        add("A02", "confirmatory family has exactly 6 tests: Proposed vs FedAvg/FedProx/Local-only x ROC-AUC/Dice",
            len(conf) == 6 and {(r["comparison"], r["metric"]) for r in conf} ==
            {(f"Proposed vs {c}", m) for c in ("fedavg", "fedprox", "local") for m in ("roc_auc", "dice")})
        h = AN.holm([0.01, 0.04, 0.03, 0.005, 0.2, 0.5])
        add("A03", "Holm step-down known answer", [round(x["p_holm"], 10) for x in h] == [0.05, 0.12, 0.12, 0.03, 0.4, 0.5]
            and [x["reject"] for x in h] == [True, False, False, True, False, False])
        inp = AN.load_inputs(man, ev)
        stat = lambda Ls, vs: ST.summarise(Ls, "roc_auc", "macro", 0.5, views=vs)
        direct = ST.paired_bootstrap_multiseed({s: AN.test_list(inp, "local", s) for s in AN.SEEDS},
                                               {s: AN.test_list(inp, "proposed", s) for s in AN.SEEDS}, stat)
        row = [r for r in conf if r["comparison"] == "Proposed vs local" and r["metric"] == "roc_auc"][0]
        dh = np.mean([ST.summarise(AN.test_list(inp, "proposed", s), "roc_auc", "macro") - ST.summarise(AN.test_list(inp, "local", s), "roc_auc", "macro") for s in AN.SEEDS])
        add("A04", "each test is the frozen R4 paired multi-seed bootstrap; delta_hat = mean over seeds of (Proposed - comparator)",
            abs(float(row["delta_hat"]) - direct["delta_bar"]) < 1e-12 and abs(float(row["p_raw"]) - direct["p_value"]) < 1e-15
            and abs(float(row["ci_low"]) - direct["ci_low"]) < 1e-12 and abs(direct["delta_bar"] - dh) < 1e-12,
            f"delta {float(row['delta_hat']):+.4f}, CI [{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}], p {float(row['p_raw']):.4g}")
        add("A05", "R4/R5.1 bootstrap contract: 2,000 attempted, valid + undefined = 2,000, seed 20260930",
            all(int(r["B_attempted"]) == 2000 and int(r["B_valid"]) + int(r["B_undefined"]) == 2000 and int(r["bootstrap_seed"]) == 20260930 for r in conf))
        planted = row["reject"] == "True" and float(row["delta_hat"]) > 0 and float(row["ci_low"]) > 0
        add("A06", "a large planted effect (Proposed >> Local-only) is detected and survives Holm", planted,
            f"p_holm {float(row['p_holm']):.4g}")
        main_t = read(out / "main_descriptive.csv")
        r0 = [r for r in main_t if r["configuration"] == "fedavg" and r["metric"] == "dice"][0]
        per = {s: ST.summarise(AN.test_list(inp, "fedavg", s), "dice", "macro") for s in AN.SEEDS}; ss = ST.seed_summary(per)
        add("A07", "main table: per-seed client-macro values and mean +- SD (ddof = 1) equal the frozen R4 functions",
            all(abs(float(r0[f"seed_{s}"]) - per[s]) < 1e-12 for s in AN.SEEDS) and abs(float(r0["mean"]) - ss["mean"]) < 1e-12
            and abs(float(r0["sd"]) - ss["sd"]) < 1e-12 and r0["ddof"] == "1"
            and {r["configuration"] for r in main_t} == set(AN.MAIN))
        ab = read(out / "ablations_descriptive.csv")
        n3 = {r["metric"] for r in ab if r["configuration"] == "N3_cls_only"}; n4 = {r["metric"] for r in ab if r["configuration"] == "N4_seg_only"}
        add("A08", "ablations: N3 classification metrics only, N4 segmentation metrics only",
            n3 == set(AN.CLS_METRICS) and n4 == set(AN.SEG_METRICS))
        ex = read(out / "external_descriptive.csv")
        add("A09", "external: per dataset, never pooled; no Local-only; N3/N4 retained task only",
            {r["dataset"] for r in ex} == {"BLUIS"} and "local" not in {r["configuration"] for r in ex}
            and {r["metric"] for r in ex if r["configuration"] == "N4_seg_only"} == set(AN.SEG_METRICS))
        tv = read(out / "tau_val_secondary.csv")
        lr = [r for r in tv if r["configuration"] == "local" and r["metric"] == "sensitivity"][0]
        man_v = np.mean([ST.client_value(L, ST._all(L), "sensitivity", inp.tau[("local", 101, L.dataset)]) for L in AN.test_list(inp, "local", 101)])
        add("A10", "tau_val secondary: threshold metrics only; local-only uses each client model's own tau_val",
            {r["metric"] for r in tv} <= set(ST.THRESHOLD_METRICS) and abs(float(lr["seed_101"]) - man_v) < 1e-12 and all(r["threshold"] == "tau_val" for r in tv))
        pc = read(out / "per_client_supplementary.csv")
        add("A11", "per-client supplementary table covers every configuration x seed x client", len(pc) == 9 * 3 * 2)
        am = json.loads((out / "analysis_manifest.json").read_text())
        add("A12", "analysis manifest: input hashes, R4 code fingerprints, bootstrap seed, replicate counts, output hashes, scope",
            len(am["input_hashes"]) == 1 + sum(1 for _ in ev.rglob("tau_val.json")) + sum(1 for p in ev.rglob("*.csv") if "analysis" not in p.relative_to(ev).parts)
            and all(AN.sha_file(out / n) == h for n, h in am["output_sha256"].items()) and am["bootstrap_seed"] == 20260930
            and len(am["confirmatory_replicates"]) == 6 and am["holm_family_size"] == 6 and "future training runs" in am["scope"])
        # ---------------- negative / fail-closed (each must stop BEFORE any output is written)
        def closed(mutate, undo):
            n0 = len(list((ev / "analysis").iterdir())); mutate()
            try:
                r = refuses(lambda: AN.analyze(man, ev))
            finally:
                undo()
            return r and len(list((ev / "analysis").iterdir())) == n0
        lp = ev / "proposed__seed101" / "ledgers" / "BUSI__test.csv"; lb = lp.read_bytes()
        add("A13", "a listed ledger that is missing closes the gate (no output written)", closed(lambda: lp.rename(lp.with_suffix(".bak")), lambda: lp.with_suffix(".bak").rename(lp)))
        add("A14", "a ledger altered after evaluation closes the gate", closed(lambda: lp.write_bytes(lb + b"\n"), lambda: lp.write_bytes(lb)))
        stray = ev / "fedavg__seed101" / "ledgers" / "EXTRA__test.csv"
        add("A15", "an unlisted ledger on disk closes the gate", closed(lambda: shutil.copy(lp, stray), lambda: stray.unlink()))
        tp = ev / "fedprox__seed202" / "tau_val.json"; tb = tp.read_bytes()
        add("A16", "an altered tau_val artifact closes the gate", closed(lambda: tp.write_bytes(tb.replace(b'"frozen": true', b'"frozen": true ')), lambda: tp.write_bytes(tb)))
        mb = man.read_bytes(); mj = json.loads(mb); mj["study_gate"]["runs"] = mj["study_gate"]["runs"][:-1]
        add("A17", "an evaluation manifest that does not cover all 27 runs closes the gate", closed(lambda: man.write_text(json.dumps(mj)), lambda: man.write_bytes(mb)))
        other = root / "elsewhere"; other.mkdir(); shutil.copy(man, other / man.name)
        add("A18", "a manifest outside _evaluation is not accepted as an input source", refuses(lambda: AN.analyze(other / man.name, ev)))
        mj2 = json.loads(mb); u = [x for x in mj2["units"] if x["run_id"] == "local__seed101"][0]
        lext = ev / "local__seed101" / "local" / u["client"] / "ledgers" / "BLUIS__EXTERNAL_TEST_ONLY.csv"
        shutil.copy(ev / "fedavg__seed101" / "ledgers" / "BLUIS__EXTERNAL_TEST_ONLY.csv", lext)
        shutil.copy(ev / "fedavg__seed101" / "ledgers" / "BLUIS__EXTERNAL_TEST_ONLY.sha256.json", lext.with_suffix(".sha256.json"))
        u["ledgers"][lext.name] = AN.sha_file(lext)
        add("A19", "a local-only external ledger closes the gate, even if listed",
            closed(lambda: man.write_text(json.dumps(mj2)), lambda: (man.write_bytes(mb), lext.unlink(), lext.with_suffix(".sha256.json").unlink())))
        add("A20", "evaluation artifacts are byte-identical after all analyses (inputs never modified)", tree(ev) == before)
        # ---------------- R4 G1 guard waivers (fast: input gate only)
        wf = ev / "N4_seg_only__seed101" / "ledgers" / "BUSI__test.R4_G1_WAIVER.json"
        wf.write_text(json.dumps({"ledger": "BUSI__test.csv", "config_name": "N4_seg_only"}))
        mw = json.loads(mb); [u.setdefault("r4_guard_waivers", {}).update({wf.name: AN.sha_file(wf)})
                              for u in mw["units"] if u["run_id"] == "N4_seg_only__seed101"]
        man.write_text(json.dumps(mw))
        add("A21", "a listed N4 waiver with a matching fingerprint is accepted", not refuses(lambda: AN.load_inputs(man, ev)))
        wb = wf.read_bytes(); wf.write_bytes(wb + b" ")
        add("A22", "an altered N4 waiver closes the gate", refuses(lambda: AN.load_inputs(man, ev))); wf.write_bytes(wb)
        mx = json.loads(mb); [u.setdefault("r4_guard_waivers", {}).update({wf.name: AN.sha_file(wf)})
                              for u in mx["units"] if u["run_id"] == "fedavg__seed101"]
        fw = ev / "fedavg__seed101" / "ledgers" / wf.name; shutil.copy(wf, fw); man.write_text(json.dumps(mx))
        add("A23", "a waiver on any configuration other than N4 closes the gate", refuses(lambda: AN.load_inputs(man, ev)))
        fw.unlink(); man.write_bytes(mb)
        add("A24", "an unlisted waiver file on disk closes the gate", refuses(lambda: AN.load_inputs(man, ev)))
        wf.unlink()
        fdir = ev / "figures" / "figureX__probe"; fdir.mkdir(parents=True); (fdir / "plotted_values.csv").write_text("a,b\n1,2\n")
        add("A25", "figure outputs under _evaluation/figures are not mistaken for ledgers (gate still verifies every ledger)",
            not refuses(lambda: AN.load_inputs(man, ev)))
        shutil.rmtree(ev / "figures")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    nf = T.count(False); print(f"PASS = {len(T) - nf}   FAIL = {nf}"); print("ANALYSIS_TOOL_VERIFIED" if nf == 0 else "ANALYSIS_TOOL_BLOCKED")
    return 0 if nf == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
