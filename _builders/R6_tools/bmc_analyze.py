#!/usr/bin/env python3
"""bmc_analyze.py - fail-closed statistical analysis of the frozen R6 study.

Place in  _builders\\R6_tools\\  beside bmc_evaluate.py and verify_manifests.py.
Writes ONLY under R6_EXPERIMENTS\\_evaluation\\analysis\\<UTC>\\ (write-once files).

INPUT GATE: the evaluation manifest written by bmc_evaluate.py is the ONLY accepted source.
Before any statistic is computed, every ledger and every tau_val artifact it lists is
re-hashed and must match; any listed artifact that is missing or altered, and any ledger
or tau_val artifact on disk that the manifest does not list, closes the gate.

Every metric, aggregation, bootstrap plan, percentile CI, centred +1 p-value, seed summary
and the paired multi-seed bootstrap are the FROZEN R4 functions of bmc_stats.py. The only
procedure implemented here is the Holm step-down adjustment, which R4 deliberately leaves
outside bmc_stats ("applied outside this function").

Inference is conditional on the three realised trained models; between-seed variability
is reported descriptively (per-seed values and mean +- SD, ddof = 1).

    python bmc_analyze.py --manifest <R6_EXPERIMENTS\\_evaluation\\evaluation_manifest__*.json>
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BUILDERS = HERE.parent
for p in (BUILDERS / "R5_training", BUILDERS, BUILDERS / "R4_evaluation", HERE):
    if p.is_dir():
        sys.path.insert(0, str(p))

import bmc_stats as ST                                         # frozen R4 statistics

DEFAULT_PREP = Path(r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging\Revision 1\dataset preparing")
SEEDS = (101, 202, 303)
CONFIGS = ("fedavg", "fedprox", "proposed", "centralised", "local", "N1_mu0", "N2_uncapped", "N3_cls_only", "N4_seg_only")
EXPECTED_RUNS = tuple(f"{c}__seed{s}" for c in CONFIGS for s in SEEDS)
CONFIRMATORY = [(comp, metric) for comp in ("fedavg", "fedprox", "local") for metric in ("roc_auc", "dice")]   # exactly six
MAIN = ("proposed", "fedavg", "fedprox", "local", "centralised")
ABLATIONS = ("N1_mu0", "N2_uncapped", "N3_cls_only", "N4_seg_only")
CLS_METRICS = ("roc_auc",) + ST.THRESHOLD_METRICS
SEG_METRICS = ST.SEG_MEAN_METRICS + ("hd95",)
ALL_METRICS = CLS_METRICS + SEG_METRICS
RETAINED = {"N3_cls_only": CLS_METRICS, "N4_seg_only": SEG_METRICS}
ALPHA = 0.05


class AnalysisError(RuntimeError):
    pass


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metrics_for(config: str) -> Sequence[str]:
    return RETAINED.get(config, ALL_METRICS)


# ---------------------------------------------------------------- input gate
class Inputs:
    """Verified inputs: ledgers[(config, seed)][split] -> ordered list of Ledger; tau[(config, seed[, client])]."""
    def __init__(self):
        self.ledgers: Dict = {}
        self.tau: Dict = {}
        self.hashes: Dict[str, str] = {}
        self.clients: List[str] = []
        self.externals: List[str] = []


def load_inputs(manifest_path: Path, eval_root: Path, *, expected_runs: Sequence[str] = EXPECTED_RUNS) -> Inputs:
    manifest_path, eval_root = Path(manifest_path).resolve(), Path(eval_root).resolve()
    if manifest_path.parent != eval_root or not manifest_path.name.startswith("evaluation_manifest__"):
        raise AnalysisError("the evaluation manifest must be an evaluation_manifest__*.json directly in _evaluation")
    M = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = M.get("study_gate", {})
    if gate.get("gate") != "passed" or sorted(gate.get("runs", [])) != sorted(expected_runs):
        raise AnalysisError(f"evaluation manifest does not cover exactly the expected {len(expected_runs)} runs")
    inp = Inputs(); inp.hashes["evaluation_manifest"] = sha_file(manifest_path)
    listed_ledgers, listed_tau, listed_waivers = set(), set(), set()
    for u in M["units"]:
        run, client = u["run_id"], u["client"]
        cfg, seed = run.rsplit("__seed", 1); seed = int(seed)
        base = eval_root / run / ("local/" + client if client else "")
        # tau_val artifact: listed hash, sidecar hash, recorded value
        art, sig = base / "tau_val.json", base / "tau_val.sha256"
        if not (art.is_file() and sig.is_file()):
            raise AnalysisError(f"{run}{'/' + client if client else ''}: tau_val artifact missing")
        h = sha_file(art)
        if h != u["tau_val_artifact_sha256"] or h != sig.read_text(encoding="utf-8").strip():
            raise AnalysisError(f"{run}{'/' + client if client else ''}: tau_val artifact altered")
        tv = json.loads(art.read_text(encoding="utf-8"))
        if tv.get("frozen") is not True or tv.get("tau_val") != u["tau_val"] or tv.get("checkpoint_sha256") != u["checkpoint_sha256"]:
            raise AnalysisError(f"{run}: tau_val artifact inconsistent with the evaluation manifest")
        inp.tau[(cfg, seed, client)] = float(tv["tau_val"]); listed_tau.add(art.resolve())
        inp.hashes[str(art.relative_to(eval_root))] = h
        for name, digest in u.get("r4_guard_waivers", {}).items():
            wp = base / "ledgers" / name
            if cfg != "N4_seg_only":
                raise AnalysisError(f"{run}: an R4 guard waiver is permitted only for N4_seg_only")
            if not wp.is_file() or sha_file(wp) != digest:
                raise AnalysisError(f"R4 guard waiver missing or altered: {wp}")
            listed_waivers.add(wp.resolve()); inp.hashes[str(wp.relative_to(eval_root))] = digest
        for name, digest in u["ledgers"].items():
            p = base / "ledgers" / name
            if not p.is_file():
                raise AnalysisError(f"listed ledger missing: {p}")
            if sha_file(p) != digest:
                raise AnalysisError(f"ledger altered after evaluation: {p}")
            L = ST.load_ledger(p)                               # frozen loader: re-verifies its own sidecar
            ds, split = name[:-4].rsplit("__", 1)
            if L.dataset != ds or L.split != split:
                raise AnalysisError(f"ledger content does not match its name: {p}")
            inp.ledgers.setdefault((cfg, seed), {}).setdefault(split, {})[ds] = L
            listed_ledgers.add(p.resolve()); inp.hashes[str(p.relative_to(eval_root))] = digest
    # ledgers only ever live in "ledgers" folders; figure/analysis outputs elsewhere are not ledgers
    on_disk_ledgers = {p.resolve() for p in eval_root.rglob("*.csv") if "ledgers" in p.relative_to(eval_root).parts}
    on_disk_tau = {p.resolve() for p in eval_root.rglob("tau_val.json")}
    on_disk_waivers = {p.resolve() for p in eval_root.rglob("*.R4_G1_WAIVER.json")}
    extra = sorted(str(p) for p in (on_disk_ledgers - listed_ledgers) | (on_disk_tau - listed_tau) | (on_disk_waivers - listed_waivers))
    if extra:
        raise AnalysisError(f"artifacts on disk that the evaluation manifest does not list: {extra[:5]}")
    inp.clients = sorted(inp.ledgers[("proposed", SEEDS[0])]["test"])
    inp.externals = sorted(inp.ledgers[("proposed", SEEDS[0])].get(ST.EXTERNAL_SPLIT, {}))
    for (cfg, seed), by in inp.ledgers.items():                 # completeness of what the analysis needs
        if sorted(by.get("test", {})) != inp.clients:
            raise AnalysisError(f"{cfg}__seed{seed}: internal-test ledgers incomplete")
        if cfg != "local" and sorted(by.get(ST.EXTERNAL_SPLIT, {})) != inp.externals:
            raise AnalysisError(f"{cfg}__seed{seed}: external ledgers incomplete")
        if cfg == "local" and by.get(ST.EXTERNAL_SPLIT):
            raise AnalysisError("local-only external ledgers must not exist")
    return inp


def test_list(inp: Inputs, cfg: str, seed: int) -> list:
    return [inp.ledgers[(cfg, seed)]["test"][c] for c in inp.clients]


# ---------------------------------------------------------------- statistics (frozen R4 calls)
def holm(pvals: Sequence[float], alpha: float = ALPHA) -> List[Dict]:
    """Holm step-down: adjusted p_(i) = max_{j<=i} min(1, (m-j+1) p_(j)); reject iff adjusted <= alpha."""
    m = len(pvals); order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return [{"p_holm": adj[i], "reject": adj[i] <= alpha} for i in range(m)]


def confirmatory(inp: Inputs) -> List[Dict]:
    rows = []
    for comp, metric in CONFIRMATORY:
        stat = (lambda m: (lambda Ls, vs: ST.summarise(Ls, m, "macro", ST.TAU_PRIMARY, views=vs)))(metric)
        r = ST.paired_bootstrap_multiseed({s: test_list(inp, comp, s) for s in SEEDS},
                                          {s: test_list(inp, "proposed", s) for s in SEEDS}, stat, seed=ST.BOOTSTRAP_SEED)
        rows.append({"comparison": f"Proposed vs {comp}", "metric": metric, "delta_hat": r["delta_bar"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_raw": r["p_value"],
                     "B_attempted": r["n_attempted"], "B_valid": r["n_valid"], "B_undefined": r["n_undefined"],
                     "ci_flag": r["flag"], "per_seed_delta": r["per_seed_delta"], "bootstrap_seed": r["bootstrap_seed"]})
    for row, h in zip(rows, holm([r["p_raw"] for r in rows])):
        row.update(h)
    return rows


def describe(inp: Inputs, configs: Sequence[str], split: str, dataset: Optional[str] = None, tau_mode: str = "primary") -> List[Dict]:
    out = []
    for cfg in configs:
        for metric in metrics_for(cfg):
            if tau_mode == "tau_val" and metric not in ST.THRESHOLD_METRICS:
                continue
            per = {}
            for s in SEEDS:
                if split == ST.EXTERNAL_SPLIT:
                    Ls = [inp.ledgers[(cfg, s)][split][dataset]]
                else:
                    Ls = test_list(inp, cfg, s)
                if tau_mode == "tau_val" and cfg == "local":   # each client model has its own threshold
                    vals = [ST.client_value(L, ST._all(L), metric, inp.tau[(cfg, s, L.dataset)]) for L in Ls]
                    per[s] = None if any(v is None for v in vals) else float(sum(vals) / len(vals))
                else:
                    tau = inp.tau[(cfg, s, None)] if tau_mode == "tau_val" else ST.TAU_PRIMARY
                    per[s] = ST.summarise(Ls, metric, "macro", tau)
            summ = ST.seed_summary(per) if all(v is not None for v in per.values()) else {"mean": None, "sd": None, "ddof": ST.SD_DDOF}
            out.append({"configuration": cfg, "split": split, "dataset": dataset or "client-macro", "metric": metric,
                        "threshold": tau_mode, **{f"seed_{s}": per[s] for s in SEEDS},
                        "mean": summ["mean"], "sd": summ["sd"], "ddof": summ["ddof"]})
    return out


def per_client(inp: Inputs) -> List[Dict]:
    out = []
    for cfg in MAIN + ABLATIONS:
        for s in SEEDS:
            for r in ST.per_client_table(test_list(inp, cfg, s), ST.TAU_PRIMARY):
                keep = {k: v for k, v in r.items() if k in ("dataset", "split", "n", "tau") or k in metrics_for(cfg)
                        or (k.startswith("hd95") and "hd95" in metrics_for(cfg))}
                out.append({"configuration": cfg, "seed": s, **keep})
    return out


# ---------------------------------------------------------------- outputs (write-once)
def _write_csv(out_dir: Path, name: str, rows: List[Dict]) -> str:
    p = out_dir / name
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    return sha_file(p)


def analyze(manifest_path: Path, eval_root: Path, *, expected_runs: Sequence[str] = EXPECTED_RUNS) -> Path:
    inp = load_inputs(manifest_path, eval_root, expected_runs=expected_runs)        # fail-closed gate first
    out_dir = (Path(eval_root) / "analysis" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if Path(eval_root).resolve() not in out_dir.parents:
        raise AnalysisError("refusing to write outside _evaluation/analysis")
    out_dir.mkdir(parents=True, exist_ok=False)
    conf = confirmatory(inp)
    tables = {
        "confirmatory_tests.csv": conf,
        "main_descriptive.csv": describe(inp, MAIN, "test"),
        "ablations_descriptive.csv": describe(inp, ABLATIONS, "test"),
        "external_descriptive.csv": [r for ds in inp.externals
                                     for r in describe(inp, [c for c in MAIN + ABLATIONS if c != "local"], ST.EXTERNAL_SPLIT, ds)],
        "per_client_supplementary.csv": per_client(inp),
        "tau_val_secondary.csv": describe(inp, MAIN + ABLATIONS, "test", tau_mode="tau_val")
            + [r for ds in inp.externals for r in describe(inp, [c for c in MAIN + ABLATIONS if c != "local"],
                                                             ST.EXTERNAL_SPLIT, ds, tau_mode="tau_val")],
    }
    out_hashes = {name: _write_csv(out_dir, name, rows) for name, rows in tables.items()}
    from bmc_repro import env_versions
    r4 = BUILDERS / "R4_evaluation" if (BUILDERS / "R4_evaluation" / "bmc_stats.py").is_file() else BUILDERS
    manifest = {"input_hashes": inp.hashes, "r4_code_sha256": {"bmc_stats": sha_file(r4 / "bmc_stats.py"), "bmc_eval": sha_file(r4 / "bmc_eval.py")},
                "analysis_code_sha256": sha_file(Path(__file__)), "bootstrap_seed": ST.BOOTSTRAP_SEED, "B_attempted": ST.B_ATTEMPTED,
                "confirmatory_replicates": [{k: r[k] for k in ("comparison", "metric", "B_attempted", "B_valid", "B_undefined", "ci_flag")} for r in conf],
                "family_wise_alpha": ALPHA, "holm_family_size": len(conf), "primary_threshold": ST.TAU_PRIMARY,
                "scope": ST.MULTI_RUN_SCOPE, "outside_confirmatory_family": ["tau_val", "secondary metrics", "micro summaries",
                "per-client analyses", "external validation", "centralised", "ablations"],
                "output_sha256": out_hashes, "environment": env_versions(), "created_at": datetime.now(timezone.utc).isoformat()}
    with (out_dir / "analysis_manifest.json").open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--prep-root", default=str(DEFAULT_PREP))
    a = ap.parse_args(argv)
    out = analyze(Path(a.manifest), Path(a.prep_root) / "R6_EXPERIMENTS" / "_evaluation")
    print("analysis written to:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
