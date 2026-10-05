#!/usr/bin/env python3
"""test_bmc_figure6_render.py - synthetic end-to-end tests of the Figure 6 renderer (no study data)."""
from __future__ import annotations
import json, shutil, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bmc_evaluate as EV, bmc_analyze as AN, bmc_figure6_select as SEL, bmc_figure6_render as RD, test_bmc_evaluate as TE
import bmc_run as RN
RUNS5 = tuple(f"{c}__seed101" for c in ("proposed", "fedavg", "fedprox", "local", "centralised"))
T = []


def add(tid, desc, ok, detail=""):
    T.append(bool(ok)); print(f"  {tid} [{'PASS' if ok else 'FAIL'}]  {desc}" + (f"\n        {detail}" if detail else ""), flush=True)


def refuses(fn):
    try:
        fn(); return False
    except (RD.RenderError, AN.AnalysisError):
        return True


def tree(p):
    return {str(f.relative_to(p)): EV.sha_file(f) for f in sorted(p.rglob("*")) if f.is_file() and "figures" not in f.relative_to(p).parts}


def main() -> int:
    root = Path(tempfile.mkdtemp())
    try:
        TE.bank(root); ref = EV.frozen_inputs_now(root); runs = root / "R6_EXPERIMENTS" / "runs"
        for k, run in enumerate(RUNS5):
            cfg = run.split("__")[0]; d = runs / run; d.mkdir(parents=True)
            RN.write_once(d / "run_manifest.json", RN.run_manifest(cfg_name=cfg, cfg=RN.load_config(cfg), seed=101,
                          mu_prov={"mu": 0.0, "source": "fixed in the frozen config"}, frozen=ref, device="cpu", overrides={}))
            if cfg == "local":
                for j, (ds, _, _) in enumerate(TE.CL):
                    TE.selected(d / "local" / ds, f"local__seed101__{ds}", "local", 101, TE.model_state(20 + j))
            else:
                TE.selected(d, run, cfg, 101, TE.model_state(k))
            (d / "run_complete.json").write_text("{}")
        exp = root / "R6_EXPERIMENTS"; ev = exp / "_evaluation"; rows = RN.read_manifest(root)
        tok = EV.preflight(exp, root, expected_runs=RUNS5, ref=ref, client_names=[c[0] for c in TE.CL])
        man = EV.run(tok, rows, "cpu")
        sel = SEL.run(man, ev, expected_runs=RUNS5)
        before = tree(ev)
        import torch
        o_load, o_tl = RN._load_selected_model, torch.load
        def boom(*a, **k): raise AssertionError("model loading during rendering")
        RN._load_selected_model = boom; torch.load = boom
        try:
            out = RD.render(man, sel, root, expected_runs=RUNS5); no_infer = True
        except AssertionError:
            no_infer = False; out = None
        finally:
            RN._load_selected_model, torch.load = o_load, o_tl
        add("R01", "renders without loading any model or checkpoint (no new inference)", no_infer)
        files = sorted(p.name for p in out.iterdir())
        add("R02", "outputs: PNG/PDF/TIFF + per-prediction audit + figure manifest under _evaluation/figures",
            files == sorted(["Figure6_qualitative_segmentation.png", "Figure6_qualitative_segmentation.pdf", "Figure6_qualitative_segmentation.tif",
                             "figure6_audit.csv", "figure_manifest.json"]) and out.parent == (ev / "figures").resolve())
        import csv
        au = list(csv.DictReader(open(out / "figure6_audit.csv"))); sr = list(csv.DictReader(open(sel / "figure6_selection.csv")))
        M = json.loads((out / "figure_manifest.json").read_text())
        add("R03", "30 displayed predictions (6 frozen cases x 5 methods): recomputed Dice == frozen ledger Dice exactly; frozen record IDs used",
            len(au) == 30 and all(a["exact_match"] == "True" for a in au) and M["record_ids"] == [r["record_id"] for r in sr]
            and all(a["pred_png_sha256"] and a["r3_input_array_sha256"] and a["source_image_sha256"] for a in au))
        add("R04", "evaluation artifacts byte-identical after rendering", tree(ev) == before)
        a0 = [a for a in au if a["method"] == "Proposed"][0]
        png = Path([v for v in [str(p) for p in ev.rglob(f"{RD.safe_name(a0['record_id'])}.png")] if "proposed__seed101" in v][0])
        pb = png.read_bytes()
        from PIL import Image
        import numpy as np
        arr = np.asarray(Image.open(png)).copy(); arr[0, 0] = 255 - arr[0, 0]; Image.fromarray(arr).save(png)
        n0 = len(list((ev / "figures").iterdir()))
        add("R05", "a single flipped pixel in a saved prediction PNG is refused (Dice or pixel counts differ from the ledger); nothing written",
            refuses(lambda: RD.render(man, sel, root, expected_runs=RUNS5)) and len(list((ev / "figures").iterdir())) == n0)
        png.write_bytes(pb)
        st = sel / "figure6_selection.csv"; sb = st.read_bytes(); st.write_bytes(sb + b"\n")
        add("R06", "an altered frozen selection table is refused", refuses(lambda: RD.render(man, sel, root, expected_runs=RUNS5)))
        st.write_bytes(sb)
        other = root / "elsewhere" / sel.name; shutil.copytree(sel, other)
        add("R07", "a selection directory outside _evaluation/figures is refused", refuses(lambda: RD.render(man, other, root, expected_runs=RUNS5)))
        add("R08", "mask file naming identical to the frozen R4 writer", RD.safe_name("A/B C:1.png") == "A_B_C_1.png")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    nf = T.count(False); print(f"PASS = {len(T) - nf}   FAIL = {nf}"); print("FIGURE6_RENDER_VERIFIED" if nf == 0 else "FIGURE6_RENDER_BLOCKED")
    return 0 if nf == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
