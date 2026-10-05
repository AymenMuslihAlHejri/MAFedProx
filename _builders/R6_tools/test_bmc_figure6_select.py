#!/usr/bin/env python3
"""test_bmc_figure6_select.py - synthetic-only tests of the Figure 6 selection (no study data, no images)."""
from __future__ import annotations
import csv, json, shutil, sys, tempfile, types
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import bmc_figure6_select as SEL, bmc_analyze as AN, test_bmc_analyze as TA
POOLS = {"ultrasound": ["BUSI"], "mammography": ["DMID"]}
T = []


def add(tid, desc, ok, detail=""):
    T.append(bool(ok)); print(f"  {tid} [{'PASS' if ok else 'FAIL'}]  {desc}" + (f"\n        {detail}" if detail else ""), flush=True)


def main() -> int:
    root = Path(tempfile.mkdtemp())
    try:
        ev, man = TA.build(root)
        # S05 guard: any image/mask access during selection must fail loudly
        import PIL.Image, bmc_data as BD
        o_open, o_ds = PIL.Image.open, BD.BMCSplitDataset.__init__
        def boom(*a, **k): raise AssertionError("image access during selection")
        PIL.Image.open = boom; BD.BMCSplitDataset.__init__ = boom
        try:
            out = SEL.run(man, ev, pools=POOLS); no_images = True
        except AssertionError:
            no_images = False
        finally:
            PIL.Image.open, BD.BMCSplitDataset.__init__ = o_open, o_ds
        rows = list(csv.DictReader(open(out / "figure6_selection.csv"))); M = json.loads((out / "figure6_selection_manifest.json").read_text())
        add("S01", "selection writes a write-once table (3 records per modality) and manifest under _evaluation/figures",
            len(rows) == 6 and out.parent == (ev / "figures").resolve() and [r["target_quantile"] for r in rows] == ["25th", "50th", "75th"] * 2)
        add("S02", "nearest-rank known answers: n=40 -> 10/20/30; n=7 -> 2/4/6; n=1 -> 1",
            [SEL.nearest_rank(40, q) for q in SEL.QUANTILES] == [10, 20, 30] and [SEL.nearest_rank(7, q) for q in SEL.QUANTILES] == [2, 4, 6]
            and SEL.nearest_rank(1, 0.5) == 1)
        inp = AN.load_inputs(man, ev)
        ok = True
        for mod, clients in POOLS.items():
            recs = []
            for ds in clients:
                per = {k: dict(zip(inp.ledgers[(k, 101)]["test"][ds].record_id, inp.ledgers[(k, 101)]["test"][ds].dice)) for k, _ in SEL.METHODS}
                recs += [(float(np.median([per[k][r] for k, _ in SEL.METHODS])), str(r)) for r in per["proposed"]]
            recs.sort(); want = [recs[SEL.nearest_rank(len(recs), q) - 1] for q in SEL.QUANTILES]
            got = [(float(r["median_of_five_dice"]), r["record_id"]) for r in rows if r["modality"] == mod]
            ok &= [w[1] for w in want] == [g[1] for g in got] and all(abs(w[0] - g[0]) < 1e-12 for w, g in zip(want, got))
        add("S03", "independent recomputation from the verified ledgers selects the same records and medians", ok)
        F = lambda ids, d: types.SimpleNamespace(record_id=np.array(ids), dice=np.array(d))
        fake = types.SimpleNamespace(ledgers={(k, 101): {"test": {"X": F(["b", "a", "c", "d"], [0.5, 0.5, 0.1, 0.9])}} for k, _ in SEL.METHODS})
        r2, _ = SEL.select(fake, pools={"ultrasound": ["X"]})
        add("S04", "ties broken deterministically by record_id (0.5/'a' ranks before 0.5/'b')",
            [r["record_id"] for r in r2] == ["c", "a", "b"], f"selected {[r['record_id'] for r in r2]}")
        add("S05", "no image or mask is opened during selection (image loaders patched to fail)", no_images)
        lr = [r for r in rows if r["dataset"] == "BUSI"][0]
        lv = dict(zip(inp.ledgers[("local", 101)]["test"]["BUSI"].record_id, inp.ledgers[("local", 101)]["test"]["BUSI"].dice))[lr["record_id"]]
        add("S06", "Local-only Dice comes from the record's own client model", abs(float(lr["dice_Local-only"]) - lv) < 1e-12)
        add("S07", "manifest: evaluation-manifest hash, 10 ledger hashes used (5 methods x 2 clients), pool audit, output hash",
            len(M["ledgers_used_sha256"]) == 10 and M["evaluation_manifest_sha256"] == AN.sha_file(man)
            and M["audit"]["ultrasound"]["pool_size"] == 40 and M["output_sha256"]["figure6_selection.csv"] == AN.sha_file(out / "figure6_selection.csv"))
        lp = ev / "proposed__seed101" / "ledgers" / "BUSI__test.csv"; lb = lp.read_bytes(); lp.write_bytes(lb + b"\n")
        n0 = len(list((ev / "figures").iterdir()))
        try:
            SEL.run(man, ev, pools=POOLS); refused = False
        except (AN.AnalysisError, Exception):
            refused = True
        add("S08", "an altered ledger closes the gate; nothing written", refused and len(list((ev / "figures").iterdir())) == n0)
        lp.write_bytes(lb)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    nf = T.count(False); print(f"PASS = {len(T) - nf}   FAIL = {nf}"); print("FIGURE6_SELECTION_VERIFIED" if nf == 0 else "FIGURE6_SELECTION_BLOCKED")
    return 0 if nf == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
