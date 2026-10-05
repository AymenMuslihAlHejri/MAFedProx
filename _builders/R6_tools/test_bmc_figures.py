#!/usr/bin/env python3
"""test_bmc_figures.py - tests of the figure framework (Figure 2) on a COPY of an analysis directory."""
from __future__ import annotations
import csv, hashlib, json, shutil, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bmc_figures as FG
T = []


def add(tid, desc, ok, detail=""):
    T.append(bool(ok)); print(f"  {tid} [{'PASS' if ok else 'FAIL'}]  {desc}" + (f"\n        {detail}" if detail else ""), flush=True)


def refuses(fn):
    try:
        fn(); return False
    except FG.FigureError:
        return True


def tree(p):
    return {str(f.relative_to(p)): FG.sha_file(f) for f in sorted(p.rglob("*")) if f.is_file()}


def main(src: Path) -> int:
    root = Path(tempfile.mkdtemp())
    try:
        ad = root / "R6_EXPERIMENTS" / "_evaluation" / "analysis" / src.name
        shutil.copytree(src, ad); before = tree(ad)
        out = FG.figure2(ad)
        files = sorted(p.name for p in out.iterdir())
        add("F01", "Figure 2 written under _evaluation/figures/<id>__<UTC>/ with PNG, PDF, TIFF, plotted values and manifest",
            out.parent == (root / "R6_EXPERIMENTS" / "_evaluation" / "figures").resolve() and files == sorted(
                ["Figure2_main_model_comparison.png", "Figure2_main_model_comparison.pdf", "Figure2_main_model_comparison.tif",
                 "figure2_plotted_values.csv", "figure_manifest.json"]), ", ".join(files))
        man = json.loads((out / "figure_manifest.json").read_text())
        add("F02", "manifest: analysis-manifest hash, input-table hash, code hash, and the hash of every output",
            man["analysis_manifest_sha256"] == FG.sha_file(ad / "analysis_manifest.json")
            and man["input_tables_sha256"]["main_descriptive.csv"] == FG.sha_file(ad / "main_descriptive.csv")
            and all(FG.sha_file(out / n) == h for n, h in man["output_sha256"].items()) and len(man["output_sha256"]) == 4)
        pv = list(csv.DictReader(open(out / "figure2_plotted_values.csv")))
        md = list(csv.DictReader(open(ad / "main_descriptive.csv")))
        same = all(abs(float(p[k]) - float([m for m in md if m["configuration"] == p["configuration"] and m["metric"] == p["metric"]][0][k])) < 1e-12
                   for p in pv for k in ("seed_101", "seed_202", "seed_303", "mean", "sd"))
        add("F03", "plotted values are exactly the frozen table values (10 rows: 5 methods x 2 metrics)", same and len(pv) == 10)
        add("F04", "analysis inputs are byte-identical after figure generation", tree(ad) == before)
        t_first = tree(out); out2 = FG.figure2(ad)
        add("F05", "a second run writes a new directory and never overwrites the first", out2 != out and tree(out) == t_first)
        tb = (ad / "main_descriptive.csv").read_bytes(); (ad / "main_descriptive.csv").write_bytes(tb + b"\n")
        n_before = len(list((ad.parent.parent / "figures").iterdir()))
        add("F06", "an altered analysis table closes the gate and writes nothing",
            refuses(lambda: FG.figure2(ad)) and len(list((ad.parent.parent / "figures").iterdir())) == n_before)
        (ad / "main_descriptive.csv").write_bytes(tb)
        other = root / "elsewhere" / src.name; shutil.copytree(ad, other)
        add("F07", "an analysis directory outside _evaluation/analysis is refused", refuses(lambda: FG.figure2(other)))
        run = FG.FigureRun(FG.verified_analysis(ad), "probe")
        add("F08", "writing outside the figure directory is refused", refuses(lambda: run.path("../escape.txt")))
        shutil.rmtree(run.dir)
        # ---------------- Figure 3
        o3 = FG.figure3(ad); m3 = json.loads((o3 / "figure_manifest.json").read_text())
        pv3 = list(csv.DictReader(open(o3 / "figure3_plotted_values.csv"))); ct = list(csv.DictReader(open(ad / "confirmatory_tests.csv")))
        match = all(abs(float(p[k]) - float([c for c in ct if c["comparison"] == p["comparison"] and c["metric"] == p["metric"]][0][k])) < 1e-12
                    for p in pv3 for k in ("delta_hat", "ci_low", "ci_high", "p_holm"))
        add("F09", "Figure 3 plots exactly the six confirmatory tests with delta_hat, CI and Holm p equal to the frozen table",
            len(pv3) == 6 and match and {p["comparison"] for p in pv3} == {"Proposed vs fedavg", "Proposed vs fedprox", "Proposed vs local"})
        add("F10", "Figure 3 manifest: input hash, B_valid per test, and every output hash",
            m3["input_tables_sha256"]["confirmatory_tests.csv"] == FG.sha_file(ad / "confirmatory_tests.csv")
            and len(m3["spec"]["B_valid"]) == 6 and all(FG.sha_file(o3 / n) == h for n, h in m3["output_sha256"].items()))
        cb = (ad / "confirmatory_tests.csv").read_bytes(); (ad / "confirmatory_tests.csv").write_bytes(cb + b"\n")
        add("F11", "an altered confirmatory table closes the gate", refuses(lambda: FG.figure3(ad)))
        (ad / "confirmatory_tests.csv").write_bytes(cb)
        add("F12", "analysis inputs are byte-identical after Figure 3", tree(ad) == before)
        lab = open(o3 / "figure3_plotted_values.csv").read()
        # ---------------- Figure 4
        import bmc_stats as ST
        o4 = FG.figure4(ad); m4 = json.loads((o4 / "figure_manifest.json").read_text())
        pv4 = list(csv.DictReader(open(o4 / "figure4_plotted_values.csv"))); pc = list(csv.DictReader(open(ad / "per_client_supplementary.csv")))
        ok = True
        for v in pv4:
            per = {s: float([x for x in pc if x["configuration"] == v["configuration"] and x["dataset"] == v["dataset"] and int(x["seed"]) == s][0][v["metric"]]) for s in (101, 202, 303)}
            ok &= abs(ST.seed_summary(per)["mean"] - float(v["mean"])) < 1e-12 and all(abs(per[s] - float(v[f"seed_{s}"])) < 1e-12 for s in per)
        add("F13", "Figure 4: 80 cells (5 methods x 8 clients x 2 metrics) = frozen seed_summary of the frozen per-client values",
            len(pv4) == 80 and ok and {v["configuration"] for v in pv4} == {"proposed", "fedavg", "fedprox", "local", "centralised"})
        add("F14", "Figure 4: fixed absolute colour scales (AUC 0.5-1, Dice 0-1) and client n recorded; no row normalisation",
            m4["spec"]["colour_scale"] == {"roc_auc": [0.5, 1.0], "dice": [0.0, 1.0]} and all(int(v["n"]) > 0 for v in pv4)
            and "no row normalisation" in m4["spec"]["colour_scale_rule"])
        pb = (ad / "per_client_supplementary.csv").read_bytes(); (ad / "per_client_supplementary.csv").write_bytes(pb + b"\n")
        add("F15", "an altered per-client table closes the gate", refuses(lambda: FG.figure4(ad)))
        (ad / "per_client_supplementary.csv").write_bytes(pb)
        add("F16", "analysis inputs are byte-identical after Figure 4", tree(ad) == before)
        # ---------------- Figure 5
        o5 = FG.figure5(ad); m5 = json.loads((o5 / "figure_manifest.json").read_text())
        pv5 = list(csv.DictReader(open(o5 / "figure5_plotted_values.csv"))); ex = list(csv.DictReader(open(ad / "external_descriptive.csv")))
        okv = all(abs(float(v[k]) - float([e for e in ex if e["configuration"] == v["configuration"] and e["dataset"] == v["dataset"]
                  and e["metric"] == v["metric"] and e["threshold"] == "primary"][0][k])) < 1e-12 for v in pv5 for k in ("seed_101", "seed_202", "seed_303", "mean", "sd"))
        add("F17", "Figure 5: 24 points (4 methods x 3 datasets x 2 metrics) equal to the frozen external table; no Local-only, no N1-N4",
            len(pv5) == 24 and okv and {v["configuration"] for v in pv5} == {"proposed", "fedavg", "fedprox", "centralised"}
            and {v["dataset"] for v in pv5} == {"BLUIS", "BUS-UCLM", "CDD-CESM"})
        add("F18", "Figure 5 manifest: input hash, exclusions, no pooling, and every output hash",
            m5["input_tables_sha256"]["external_descriptive.csv"] == FG.sha_file(ad / "external_descriptive.csv")
            and "none" in m5["spec"]["pooling"] and all(FG.sha_file(o5 / n) == h for n, h in m5["output_sha256"].items()))
        eb = (ad / "external_descriptive.csv").read_bytes(); (ad / "external_descriptive.csv").write_bytes(eb + b"\n")
        add("F19", "an altered external table closes the gate", refuses(lambda: FG.figure5(ad)))
        (ad / "external_descriptive.csv").write_bytes(eb)
        add("F20", "analysis inputs are byte-identical after Figure 5", tree(ad) == before)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    nf = T.count(False); print(f"PASS = {len(T) - nf}   FAIL = {nf}"); print("FIGURE_FRAMEWORK_VERIFIED" if nf == 0 else "FIGURE_FRAMEWORK_BLOCKED")
    return 0 if nf == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
