#!/usr/bin/env python3
"""bmc_figure6_select.py - SELECTION ONLY for Figure 6 (frozen rule, blueprint v2). Opens no image and no mask.

Place in  _builders\\R6_tools\\ . Input gate: the evaluation manifest, verified by bmc_analyze.load_inputs
(every listed ledger, tau_val artifact and waiver re-hashed; unlisted artifacts refused). Reads only the
per-record `dice` column of the internal-test ledgers. Writes only under
R6_EXPERIMENTS\\_evaluation\\figures\\figure6_selection__<UTC>\\ (write-once).

FROZEN RULE
  seed 101; internal test split only; two pools: ultrasound clients and mammography clients;
  record score = median Dice across Proposed, FedAvg, FedProx, Local-only (the record's own client
  model) and Centralised; within each pool, sort ascending by (score, record_id) and take the
  nearest-rank 25th, 50th and 75th percentile records: rank k = ceil(p * N), 1-indexed.

    python bmc_figure6_select.py --manifest "<...\\_evaluation\\evaluation_manifest__*.json>"
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bmc_analyze as AN                                        # input gate (sets the R4/R5 import paths)

SEED = 101
METHODS = [("proposed", "Proposed"), ("fedavg", "FedAvg"), ("fedprox", "FedProx"), ("local", "Local-only"), ("centralised", "Centralised")]
POOLS = {"ultrasound": ["BUSI", "BUS-UC", "BrEaST", "BUS-BRA"], "mammography": ["INbreast", "DMID", "CBIS-DDSM", "AIISLAB"]}
QUANTILES = (0.25, 0.50, 0.75)


class SelectionError(RuntimeError):
    pass


def nearest_rank(n: int, p: float) -> int:
    """1-indexed nearest-rank position: ceil(p * n), at least 1."""
    return max(1, math.ceil(p * n))


def select(inp, pools=POOLS, seed: int = SEED):
    import numpy as np
    rows, audit = [], {}
    for modality, clients in pools.items():
        recs = []
        for ds in clients:
            per = {}
            for key, _ in METHODS:
                L = inp.ledgers[(key, seed)]["test"][ds]
                per[key] = dict(zip(L.record_id.tolist(), L.dice.tolist()))
            ids = set(per["proposed"])
            if any(set(per[k]) != ids for k, _ in METHODS):
                raise SelectionError(f"{ds}: record sets differ across methods")
            for rid in sorted(ids):
                d = {k: per[k][rid] for k, _ in METHODS}
                if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in d.values()):
                    continue                                         # undefined Dice: excluded and counted
                recs.append((float(np.median(list(d.values()))), str(rid), ds, d))
        n_all = sum(len(inp.ledgers[("proposed", seed)]["test"][ds].record_id) for ds in clients)
        recs.sort(key=lambda r: (r[0], r[1]))
        audit[modality] = {"pool_size": len(recs), "records_in_ledgers": n_all, "excluded_undefined_dice": n_all - len(recs),
                           "ranks": {}}
        for q in QUANTILES:
            k = nearest_rank(len(recs), q); score, rid, ds, d = recs[k - 1]
            audit[modality]["ranks"][f"{int(q * 100)}th"] = k
            rows.append({"modality": modality, "target_quantile": f"{int(q * 100)}th", "rank_k": k, "pool_size": len(recs),
                         "dataset": ds, "record_id": rid, "median_of_five_dice": score,
                         **{f"dice_{lab}": d[key] for key, lab in METHODS}})
    return rows, audit


def run(manifest: Path, eval_root: Path, *, expected_runs=AN.EXPECTED_RUNS, pools=POOLS) -> Path:
    inp = AN.load_inputs(manifest, eval_root, expected_runs=expected_runs)     # fail-closed gate first
    rows, audit = select(inp, pools)
    out = (Path(eval_root) / "figures" / f"figure6_selection__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}").resolve()
    if Path(eval_root).resolve() not in out.parents:
        raise SelectionError("refusing to write outside _evaluation/figures")
    out.mkdir(parents=True, exist_ok=False)
    p = out / "figure6_selection.csv"
    with p.open("x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    used = {}
    for modality, clients in pools.items():
        for ds in clients:
            for key, _ in METHODS:
                L = inp.ledgers[(key, SEED)]["test"][ds]; used[str(Path(L.path).relative_to(Path(eval_root).resolve()))] = L.sha256
    man = {"stage": "Figure 6 case SELECTION ONLY (no image or mask opened)",
           "rule": {"seed": SEED, "split": "internal test", "pools": pools,
                    "score": "median Dice across Proposed, FedAvg, FedProx, Local-only (own client model), Centralised",
                    "order": "ascending by (score, record_id)", "quantiles": "nearest-rank: k = ceil(p * N), 1-indexed",
                    "undefined_dice": "records with any undefined Dice are excluded and counted"},
           "evaluation_manifest_sha256": inp.hashes["evaluation_manifest"], "ledgers_used_sha256": used,
           "audit": audit, "code_sha256": AN.sha_file(Path(__file__)),
           "output_sha256": {p.name: AN.sha_file(p)}, "created_at": datetime.now(timezone.utc).isoformat()}
    with (out / "figure6_selection_manifest.json").open("x", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, sort_keys=True, default=str)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--prep-root", default=str(AN.DEFAULT_PREP))
    a = ap.parse_args(argv)
    out = run(Path(a.manifest), Path(a.prep_root) / "R6_EXPERIMENTS" / "_evaluation")
    print("selection written to:", out)
    for r in csv.DictReader((out / "figure6_selection.csv").open(encoding="utf-8")):
        print(f"  {r['modality']:11s} {r['target_quantile']:4s} k={r['rank_k']:>4s}/{r['pool_size']:<5s} {r['dataset']:10s} {r['record_id']:40s} "
              f"median {float(r['median_of_five_dice']):.4f} | " + " ".join(f"{lab} {float(r['dice_' + lab]):.4f}" for _, lab in METHODS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
