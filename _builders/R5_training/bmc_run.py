#!/usr/bin/env python3
"""bmc_run.py - end-to-end R6 driver (integrates only frozen R1-R5 components).

MODES (each is a separate, explicit invocation)
  preflight        ONE synthetic forward + loss + backward + optimizer step (bf16, batch 16), then exit
  cache            build + verify the R3 deterministic cache for TRAIN and VALIDATION only
  mu-grid-run      one FedProx mu-grid config on seed 101 (TRAIN + VALIDATION only)
  select-mu        freeze mu from the four grid runs' VALIDATION selection records
  train            one frozen config x one frozen training seed (TRAIN + VALIDATION only)
  final-internal   internal TEST ledgers for a run whose selection record is frozen
  final-external   EXTERNAL ledgers; needs --confirm-final-external and a frozen selection record

ACCESS GATE: every data request states its split; the gate of the current mode refuses
any split outside its allow-list, so test/external data cannot be reached in training or
model selection, and external data cannot be reached before final-external mode.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent                     # _builders/R5_training
BUILDERS = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(BUILDERS))
for _p in (BUILDERS / "R4_evaluation", BUILDERS):
    if (_p / "bmc_eval.py").is_file():
        sys.path.insert(0, str(_p)); break

DEFAULT_PREP_ROOT = Path(r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging\Revision 1\dataset preparing")
TRAINING_SEEDS = (101, 202, 303)
SPLIT_SEED, BOOTSTRAP_SEED, MU_GRID_SEED = 20260929, 20260930, 101
MODE_SPLITS = {"cache": {"train", "validation"}, "smoke": {"train", "validation"},
               "mu_selection": {"train", "validation"},
               "training": {"train", "validation"}, "final_internal": {"test"},
               "final_external": {"EXTERNAL_TEST_ONLY"}}
MU_NEEDED = "SELECTED_FROM_FEDPROX_GRID"
# NON-STUDY integration smoke: base config and the ONLY two overrides it may apply
SMOKE_BASE_CONFIG = "N1_mu0"             # proposed aggregation, mu = 0 fixed: needs no mu selection
SMOKE_OVERRIDES = {"rounds": 1, "local_epochs": 1}
SMOKE_SEED = 101
SMOKE_MARKER = "NON_STUDY_SMOKE"
EXPECTED_CLIENTS = 8
PREREGISTRATION = HERE / "R5_PREREGISTERED_INTERPRETATION.md"
ABORT_MARKER = "ABORTED"
FORMAL_PARENTS = ("runs", "mu_selection")
RESTART_POLICY = "restart from round 0 with the same frozen config and seed; no resume"


class AccessError(PermissionError):
    pass


class DriverError(RuntimeError):
    pass


class AccessGate:
    def __init__(self, mode: str):
        if mode not in MODE_SPLITS:
            raise DriverError(f"unknown mode {mode!r}")
        self.mode = mode

    def check(self, split: str) -> None:
        if split not in MODE_SPLITS[self.mode]:
            raise AccessError(f"mode {self.mode!r} may not access split {split!r}")


# ---------------------------------------------------------------- frozen inputs
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def paths(prep_root: Path) -> Dict[str, Path]:
    bank = Path(prep_root) / "CANONICAL_DATASET_BANK"
    return {"cohort_manifest": bank / "_bmc_cohort" / "BMC_COHORT_MANIFEST.csv",
            "splits_manifest": bank / "_bmc_splits" / "BMC_SPLITS_MANIFEST.csv",
            "splits_report": bank / "_bmc_splits" / "BMC_SPLITS_REPORT.json",
            "bmc_data": BUILDERS / "bmc_data.py",
            "bmc_eval": next(p for p in (BUILDERS / "R4_evaluation" / "bmc_eval.py", BUILDERS / "bmc_eval.py") if p.is_file()),
            "bmc_stats": next(p for p in (BUILDERS / "R4_evaluation" / "bmc_stats.py", BUILDERS / "bmc_stats.py") if p.is_file()),
            "experiments": Path(prep_root) / "R6_EXPERIMENTS"}


def frozen_inputs(prep_root: Path) -> Dict[str, str]:
    """SHA-256 of every frozen input, after verifying the R2 split seed and verdict."""
    P = paths(prep_root)
    rep = json.loads(P["splits_report"].read_text(encoding="utf-8"))
    if rep.get("seed") != SPLIT_SEED or rep.get("verdict") != "BMC_SPLITS_LOCKED":
        raise DriverError(f"R2 report seed={rep.get('seed')} verdict={rep.get('verdict')}: not the frozen split")
    if not PREREGISTRATION.is_file():
        raise DriverError(f"the frozen pre-registration {PREREGISTRATION.name} is missing: formal runs refused")
    out = {k: sha256_file(P[k]) for k in ("cohort_manifest", "splits_manifest", "splits_report",
                                         "bmc_data", "bmc_eval", "bmc_stats")}
    out["preregistration"] = sha256_file(PREREGISTRATION)
    for f in sorted(HERE.glob("bmc_*.py")):
        out[f"r5:{f.name}"] = sha256_file(f)
    return out


def read_manifest(prep_root: Path) -> List[dict]:
    import csv
    with paths(prep_root)["splits_manifest"].open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def client_datasets(rows: List[dict]) -> List[str]:
    ds = {r["dataset"]: int(r["client_id"]) for r in rows if r["role"] == "CLIENT"}
    return sorted(ds, key=ds.get)


# ---------------------------------------------------------------- configs / mu
def load_config(name: str) -> dict:
    import bmc_config
    cfgs = bmc_config.build()
    if name not in cfgs:
        raise DriverError(f"unknown config {name!r}")
    return copy.deepcopy(cfgs[name])


def check_seed(cfg: dict, seed: int) -> None:
    allowed = tuple(cfg.get("seeds", TRAINING_SEEDS))
    if cfg.get("training_seeds") != list(TRAINING_SEEDS) or seed not in allowed:
        raise DriverError(f"seed {seed} is not a frozen seed for this config {allowed}")


def resolve_mu(cfg: dict, exp_root: Path) -> Optional[dict]:
    """Returns the mu provenance; raises if a config needs the grid mu and it is not frozen."""
    if cfg.get("mu") != MU_NEEDED:
        return {"mu": cfg.get("mu", 0.0), "source": "fixed in the frozen config"}
    rec_p = Path(exp_root) / "mu_selection" / "mu_record.json"
    if not rec_p.is_file():
        raise DriverError("mu is unresolved: run the FedProx grid and select-mu before FedProx/Proposed training")
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    import bmc_select
    if rec.get("frozen") is not True or rec.get("selection_split") != "validation" \
            or rec.get("selected_mu") not in bmc_select.MU_GRID or rec.get("seed") != MU_GRID_SEED:
        raise DriverError("mu record is not a frozen, validation-derived grid selection")
    cfg["mu"] = rec["selected_mu"]
    return {"mu": rec["selected_mu"], "source": "frozen mu_record.json", "mu_record_sha256": sha256_file(rec_p)}


# ---------------------------------------------------------------- data (TRAIN/VAL)
def build_training_clients(gate: AccessGate, rows: List[dict], prep_root: Path, cache_dir: Path):
    gate.check("train"); gate.check("validation")
    import bmc_data as BD
    from bmc_cache import CachedBMCSplitDataset
    from bmc_train import ClientData
    ver = Path(cache_dir) / "cache_verified.json"
    if not ver.is_file() or json.loads(ver.read_text())["mismatches"] != 0:
        raise DriverError("the deterministic cache has not been verified bit-identical to R3 (run: cache)")
    clients = []
    for ds in client_datasets(rows):
        tr = CachedBMCSplitDataset(ds, "train", prep_root=prep_root, manifest_rows=rows, cache_dir=cache_dir, **BD.APPROVED_R3)
        va = CachedBMCSplitDataset(ds, "validation", prep_root=prep_root, manifest_rows=rows, cache_dir=cache_dir, **BD.APPROVED_R3)
        code = 0 if tr.modality == "ultrasound" else 1
        clients.append(ClientData(ds, code, "train", "validation", [int(v) for v in tr.labels()],
                                  (lambda i, t=tr: t[i]), [va[i] for i in range(len(va))]))
    return clients


def build_cache(prep_root: Path, cache_dir: Path) -> dict:
    gate = AccessGate("cache")
    import bmc_data as BD
    from bmc_cache import build_cache as _build, verify_cache
    rows = read_manifest(prep_root); bad = []
    for ds in client_datasets(rows):
        for sp in ("train", "validation"):
            gate.check(sp)
            o = BD.BMCSplitDataset(ds, sp, prep_root=prep_root, manifest_rows=rows, **BD.APPROVED_R3)
            _build(o, cache_dir); bad += verify_cache(o, cache_dir)
    rec = {"mismatches": len(bad), "failing_records": bad[:20], "splits": ["train", "validation"],
           "verified_at": datetime.now(timezone.utc).isoformat()}
    Path(cache_dir, "cache_verified.json").write_text(json.dumps(rec, indent=2))
    return rec


# ---------------------------------------------------------------- manifests
def write_once(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


def run_manifest(*, cfg_name, cfg, seed, mu_prov, frozen, device, overrides) -> dict:
    from bmc_repro import config_sha256, env_versions
    return {"run_id": f"{cfg_name}__seed{seed}", "config_name": cfg_name, "method": cfg["method"],
            "regime": cfg["regime"], "ablation": cfg.get("ablation"), "seed": seed,
            "split_seed": SPLIT_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "mu": mu_prov,
            "checkpoint_selection": {**cfg["selection"], "retention": "best-so-far on strict improvement"},
            "config": cfg, "config_sha256": config_sha256(cfg), "frozen_input_sha256": frozen,
            "environment": env_versions(), "device": device, "test_overrides": overrides or {},
            "started_at": datetime.now(timezone.utc).isoformat()}


def _save_best(run_dir: Path, res: dict) -> str:
    import torch
    ck = run_dir / "checkpoints" / "best.pt"; ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res["best_state"], ck)
    return sha256_file(ck)


def _finish(run_dir: Path, res: dict, cfg: dict, run_id: str, seed: int) -> dict:
    import bmc_select
    ck_sha = _save_best(run_dir, res)
    write_once(run_dir / "history.json", res["history"])
    return bmc_select.freeze_record(run_dir / "selection_record.json", run_id=run_id, method=cfg["method"], seed=seed,
                                    history=res["history"], best_so_far_index=res["best_index"], checkpoint_sha256=ck_sha)


def train(cfg_name: str, seed: int, prep_root: Path, device: str = "cuda", *, mode: str = "training",
          _overrides: Optional[dict] = None, _clients=None, _exp_root: Optional[Path] = None,
          _frozen: Optional[dict] = None) -> Path:
    """One frozen config x one frozen seed. _overrides/_clients/_frozen exist ONLY for the synthetic
    driver tests; they are recorded in the manifest and are not reachable from the CLI."""
    gate = AccessGate(mode)
    cfg = load_config(cfg_name)
    if mode == "mu_selection" and cfg.get("purpose") != "mu_selection_validation_only":
        raise DriverError("mu-selection mode runs only the FedProx grid configs")
    if mode == "training" and cfg.get("purpose") == "mu_selection_validation_only":
        raise DriverError("grid configs run only in mu-selection mode")
    check_seed(cfg, seed)
    exp = Path(_exp_root) if _exp_root else paths(prep_root)["experiments"]
    mu_prov = resolve_mu(cfg, exp)
    cfg.update(_overrides or {})
    frozen = _frozen if _frozen is not None else frozen_inputs(prep_root)
    clients = _clients if _clients is not None else build_training_clients(
        gate, read_manifest(prep_root), prep_root, exp / "_cache" / "deterministic_256")
    for c in clients:
        gate.check(c.train_split); gate.check(c.val_split)
    sub = "mu_selection" if mode == "mu_selection" else "runs"
    run_dir = exp / sub / f"{cfg_name}__seed{seed}"
    if run_dir.exists():
        raise DriverError(f"run directory exists (runs are write-once): {run_dir}")
    rgb = None
    if not _clients:                                        # production: public ImageNet encoder
        from bmc_model import fetch_imagenet_rgb_encoder_state
        rgb = fetch_imagenet_rgb_encoder_state()
    man = run_manifest(cfg_name=cfg_name, cfg=cfg, seed=seed, mu_prov=mu_prov, frozen=frozen, device=device, overrides=_overrides)
    write_once(run_dir / "run_manifest.json", man)
    import bmc_train as TR
    if cfg["regime"] == "federated":
        res = TR.run_federated(cfg, clients, seed, device, rgb)
        _finish(run_dir, res, cfg, man["run_id"], seed)
    elif cfg["regime"] == "centralised":
        res = TR.run_centralised(cfg, clients, seed, device, rgb)
        _finish(run_dir, res, cfg, man["run_id"], seed)
    else:
        for i, c in enumerate(clients):
            res = TR.run_local(cfg, c, seed, i, device, rgb)
            _finish(run_dir / "local" / c.name, res, cfg, f"{man['run_id']}__{c.name}", seed)
    write_once(run_dir / "run_complete.json", {"finished_at": datetime.now(timezone.utc).isoformat()})
    return run_dir


def select_mu(exp_root: Path) -> dict:
    """Reads ONLY the four grid runs' frozen VALIDATION selection records."""
    import bmc_select
    crit, prov = {}, {}
    for mu in bmc_select.MU_GRID:
        d = Path(exp_root) / "mu_selection" / f"fedprox_mu_{mu:g}__seed{MU_GRID_SEED}"
        if not (d / "run_complete.json").is_file() or (d / ABORT_MARKER).exists():
            raise DriverError(f"grid run {d.name} is missing, incomplete or aborted: mu cannot be selected")
        p = d / "selection_record.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("selection_split") != "validation" or rec.get("frozen") is not True:
            raise DriverError(f"{p}: not a frozen validation selection record")
        crit[mu] = rec["selection_criterion_value"]; prov[f"{mu:g}"] = {"criterion": crit[mu], "record_sha256": sha256_file(p)}
    mu = bmc_select.select_mu(crit)
    rec = {"frozen": True, "selection_split": "validation", "selected_mu": mu, "seed": MU_GRID_SEED,
           "rule": "min client-macro validation loss; tie -> smaller mu", "grid": prov,
           "created_at": datetime.now(timezone.utc).isoformat()}
    write_once(Path(exp_root) / "mu_selection" / "mu_record.json", rec)
    return rec


# ---------------------------------------------------------------- final evaluation
def load_selected_checkpoint(run_dir: Path, rec: dict):
    """The ONLY checkpoint load in R5/R6: this run's own selected weights, after the
    selection record is frozen, verified against the SHA-256 stored in that record."""
    import torch
    ck = Path(run_dir) / "checkpoints" / "best.pt"
    if sha256_file(ck) != rec["checkpoint_sha256"]:
        raise DriverError("checkpoint does not match the frozen selection record")
    return torch.load(ck, map_location="cpu", weights_only=True)


def _evaluate_split_ledgers(model, datasets, split, rows, prep_root, rec, rec_p, man, run_id, out_dir, device) -> List[str]:
    import numpy as np, torch
    import bmc_data as BD, bmc_eval as EV
    out = []
    for ds in datasets:
        o = BD.BMCSplitDataset(ds, split, prep_root=prep_root, manifest_rows=rows, **BD.APPROVED_R3)
        code = 0 if o.modality == "ultrasound" else 1
        items = [o[i] for i in range(len(o))]
        x = torch.stack([torch.as_tensor(it["image"]) for it in items]).to(device)
        with torch.no_grad():
            seg, cls = model(x, torch.full((len(items),), code, dtype=torch.long, device=device))
        recs = [{"record_id": r["canonical_record_id"], "dataset": ds, "client_id": r.get("client_id", ""),
                 "split": split, "group_key": r["group_key"], "group_unit": r.get("group_unit", ""),
                 "modality": o.modality, "label": int(r["label_binary"])} for r in o.rows]
        prov = {"run_id": run_id, "method": man["method"], "seed": man["seed"],
                "selected_round": rec["selected_round"], "selection_criterion_name": rec["selection_criterion_name"],
                "selection_criterion_value": rec["selection_criterion_value"],
                "model_sha256": rec["checkpoint_sha256"], "splits_manifest_sha256": man["frozen_input_sha256"]["splits_manifest"]}
        lp = Path(out_dir) / "ledgers" / f"{ds}__{split}.csv"
        EV.evaluate_split(records=recs, cls_logits=cls.float().cpu().numpy(), seg_logits=seg[:, 0].float().cpu().numpy(),
                          gt_masks=np.stack([it["mask"][0] for it in items]).astype(np.uint8), provenance=prov,
                          out_path=lp, selection_record=rec_p)
        out.append(str(lp))
    return out


def _load_selected_model(sub_dir: Path, device: str):
    from bmc_model import BMCModel
    from bmc_train import require_frozen_selection
    rec_p = require_frozen_selection(sub_dir)                 # ordering safeguard
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    model = BMCModel(); model.load_state_dict(load_selected_checkpoint(sub_dir, rec))   # SHA-verified
    return model.to(device).eval(), rec, rec_p


def final_evaluate(run_dir: Path, prep_root: Path, mode: str, device: str = "cpu", *, confirm_external: bool = False,
                   _rows: Optional[List[dict]] = None) -> List[str]:
    """final_internal: internal TEST ledgers. For local-only runs, each client's own selected
    model is evaluated on THAT client's internal test partition only (frozen R5.1 decision).
    final_external: global/federated and centralised models only; never local-only models."""
    if mode not in ("final_internal", "final_external"):
        raise DriverError(mode)
    if mode == "final_external" and not confirm_external:
        raise AccessError("final-external mode requires explicit --confirm-final-external")
    gate = AccessGate(mode)
    split = "test" if mode == "final_internal" else "EXTERNAL_TEST_ONLY"
    gate.check(split)
    if (Path(run_dir) / SMOKE_MARKER).exists() or "_smoke" in Path(run_dir).parts:
        raise DriverError("NON_STUDY_SMOKE outputs are never eligible for final evaluation or R6 results")
    if (Path(run_dir) / ABORT_MARKER).exists() or "_aborted" in Path(run_dir).parts:
        raise DriverError("aborted runs are never eligible for final evaluation or R6 results")
    if not (Path(run_dir) / "run_complete.json").is_file():
        raise DriverError(f"run {Path(run_dir).name} is incomplete (no run_complete.json): final evaluation refused")
    mp = Path(run_dir) / "run_manifest.json"
    if not mp.is_file():
        raise DriverError(f"no run manifest in {run_dir}: final evaluation refused")
    man = json.loads(mp.read_text(encoding="utf-8"))
    rows = _rows if _rows is not None else read_manifest(prep_root)
    if man["regime"] == "local":
        if mode == "final_external":
            raise DriverError("local-only models are NOT evaluated on external datasets (frozen R5.1 decision): "
                              "no single applicable model exists and none may be chosen arbitrarily")
        out = []
        for ds in client_datasets(rows):
            sub = Path(run_dir) / "local" / ds
            model, rec, rec_p = _load_selected_model(sub, device)
            out += _evaluate_split_ledgers(model, [ds], split, rows, prep_root, rec, rec_p, man,
                                           f"{man['run_id']}__{ds}", sub, device)
        return out
    model, rec, rec_p = _load_selected_model(Path(run_dir), device)
    targets = sorted({r["dataset"] for r in rows if r["split"] == split})
    return _evaluate_split_ledgers(model, targets, split, rows, prep_root, rec, rec_p, man, man["run_id"], run_dir, device)


# ---------------------------------------------------------------- interrupted runs
def _inventory(d: Path) -> Dict[str, str]:
    return {str(f.relative_to(d)).replace("\\", "/"): sha256_file(f) for f in sorted(Path(d).rglob("*")) if f.is_file()}


def quarantine_incomplete(run_dir: Path, reason: str, *, _stamp: Optional[str] = None) -> dict:
    """Move an INCOMPLETE formal run (no run_complete.json) intact to R6_EXPERIMENTS/_aborted/,
    with a provenance record. Never deletes or overwrites; refuses completed runs. The run is
    then restarted from round 0 with the same frozen config and seed (no resume)."""
    import shutil
    run_dir = Path(run_dir).resolve()
    if not reason or not reason.strip():
        raise DriverError("a quarantine reason is required")
    if not run_dir.is_dir():
        raise DriverError(f"not a run directory: {run_dir}")
    if run_dir.parent.name not in FORMAL_PARENTS or "_smoke" in run_dir.parts or "_aborted" in run_dir.parts:
        raise DriverError("quarantine applies only to formal runs under R6_EXPERIMENTS/runs or /mu_selection")
    if (run_dir / "run_complete.json").exists():
        raise DriverError(f"{run_dir.name} is COMPLETE: completed runs are never quarantined")
    exp = run_dir.parent.parent
    stamp = _stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = exp / "_aborted" / f"{run_dir.parent.name}__{run_dir.name}__aborted_{stamp}"
    if dest.exists():
        raise DriverError(f"quarantine destination already exists (never overwritten): {dest}")
    man_p = run_dir / "run_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.is_file() else None
    before = _inventory(run_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(dest))
    after = _inventory(dest)
    if before != after:
        raise DriverError("quarantine altered file contents; the moved directory is left in place for inspection")
    rec = {"original_path": str(run_dir), "original_run_id": (man or {}).get("run_id", run_dir.name),
           "destination": str(dest), "formal_location": run_dir.parent.name,
           "config_name": (man or {}).get("config_name"), "method": (man or {}).get("method"),
           "seed": (man or {}).get("seed"), "mu": (man or {}).get("mu"),
           "quarantined_at": datetime.now(timezone.utc).isoformat(), "reason": reason.strip(),
           "manifest_present": man is not None, "manifest": man, "file_sha256": after,
           "restart_policy": RESTART_POLICY}
    write_once(dest / "ABORT_RECORD.json", rec)
    (dest / ABORT_MARKER).write_text("Interrupted formal run. Never a valid R6 input.\n")
    with (exp / "_aborted" / "abort_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: rec[k] for k in ("original_path", "destination", "config_name", "seed",
                                                  "quarantined_at", "reason")}) + "\n")
    return rec


def discover_completed_runs(exp_root: Path) -> List[Path]:
    """The ONLY valid discovery of formal runs: complete, never smoke, never aborted."""
    out = []
    for parent in FORMAL_PARENTS:
        for d in sorted((Path(exp_root) / parent).glob("*__seed*")):
            if d.is_dir() and (d / "run_complete.json").is_file() and not (d / ABORT_MARKER).exists():
                out.append(d)
    return out


# ---------------------------------------------------------------- NON-STUDY smoke
def smoke(prep_root: Path, device: str = "cuda", *, _clients=None, _exp_root: Optional[Path] = None,
          _frozen: Optional[dict] = None, _fetch=None, _expected_clients: int = EXPECTED_CLIENTS) -> Path:
    """Bounded integration smoke on the REAL manifest, verified cache, R3 pipeline, ImageNet
    initialisation, model, sampler, loss, optimiser and PROPOSED aggregation: exactly one round,
    one local epoch, batch 16, bf16, validation after the round. TRAIN + VALIDATION only.
    Outputs go to R6_EXPERIMENTS/_smoke/, carry the NON_STUDY_SMOKE marker, have no
    run_manifest.json, and are refused by final evaluation. No mu artefact is touched."""
    import bmc_train as TR
    from bmc_model import fetch_imagenet_rgb_encoder_state, state_sha256
    gate = AccessGate("smoke")
    cfg = load_config(SMOKE_BASE_CONFIG)
    if cfg.get("mu") == MU_NEEDED or cfg.get("purpose") == "mu_selection_validation_only":
        raise DriverError("the smoke base config must not require or perform mu selection")
    formal_bs = cfg["batch_size"]
    cfg.update(SMOKE_OVERRIDES)
    if cfg["batch_size"] != 16 or formal_bs != 16:
        raise DriverError("smoke must use the frozen batch size 16")
    exp = Path(_exp_root) if _exp_root else paths(prep_root)["experiments"]
    frozen = _frozen if _frozen is not None else frozen_inputs(prep_root)
    clients = _clients if _clients is not None else build_training_clients(
        gate, read_manifest(prep_root), prep_root, exp / "_cache" / "deterministic_256")
    for c in clients:
        gate.check(c.train_split); gate.check(c.val_split)
    codes = {c.code for c in clients}
    if len(clients) != _expected_clients or codes != {0, 1}:
        raise DriverError(f"smoke must exercise all {_expected_clients} clients and both modalities "
                          f"(got {len(clients)} clients, modality codes {sorted(codes)})")
    rgb = (_fetch or fetch_imagenet_rgb_encoder_state)()
    if not rgb or "conv1.weight" not in rgb:
        raise DriverError("smoke requires the production ImageNet encoder state; no random fallback")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    sdir = exp / "_smoke" / f"smoke__{stamp}"
    sdir.mkdir(parents=True, exist_ok=False)
    (sdir / SMOKE_MARKER).write_text("NON-STUDY integration smoke. Not eligible for any R6 result.\n")
    man = run_manifest(cfg_name=SMOKE_BASE_CONFIG, cfg=cfg, seed=SMOKE_SEED,
                       mu_prov={"mu": cfg["mu"], "source": "fixed in the frozen N1_mu0 config; no mu selection"},
                       frozen=frozen, device=device, overrides=dict(SMOKE_OVERRIDES))
    man.update(status=SMOKE_MARKER, eligible_for_results=False, base_config=SMOKE_BASE_CONFIG,
               smoke_overrides=dict(SMOKE_OVERRIDES), pretrained_sha256=state_sha256(rgb),
               run_id=f"{SMOKE_MARKER}__{stamp}")
    write_once(sdir / "smoke_manifest.json", man)
    steps = {}
    batch_sizes = []
    orig_step = TR.train_step

    def counting_step(model, opt, batch, cfg_, device_, mu=0.0, ref=None):
        batch_sizes.append(int(batch[0].shape[0]))
        return orig_step(model, opt, batch, cfg_, device_, mu, ref)
    TR.train_step = counting_step
    try:
        t0 = time.perf_counter()
        res = TR.run_federated(cfg, clients, SMOKE_SEED, device, rgb,
                               on_round=lambda r, k, opt, lr: steps.__setitem__(
                                   k, max((int(st["step"]) for st in opt.state.values()), default=0)))
        wall = time.perf_counter() - t0
    finally:
        TR.train_step = orig_step
    import bmc_select, torch
    ck = sdir / "checkpoints" / "best.pt"; ck.parent.mkdir(parents=True)
    torch.save(res["best_state"], ck)
    bmc_select.freeze_record(sdir / "smoke_selection_record.json", run_id=man["run_id"], method=cfg["method"],
                             seed=SMOKE_SEED, history=res["history"], best_so_far_index=res["best_index"],
                             checkpoint_sha256=sha256_file(ck))
    report = {"status": SMOKE_MARKER, "eligible_for_results": False, "rounds_run": len(res["history"]),
              "local_epochs": cfg["local_epochs"], "batch_size": cfg["batch_size"],
              "batch_sizes_seen": sorted(set(batch_sizes)), "optimizer_steps": len(batch_sizes),
              "clients": [{"name": c.name, "modality_code": c.code, "n_train": c.n_train,
                           "expected_steps": c.n_train // 16, "steps": steps.get(c.name, 0)} for c in clients],
              "validation": res["history"][-1], "wall_s": wall}
    write_once(sdir / "smoke_report.json", report)
    return sdir


# ---------------------------------------------------------------- preflight
def one_step(device: str, rgb_state, batch: int = 16, size: int = 256, bf16: bool = True) -> dict:
    """Exactly ONE forward + joint loss + backward + optimizer step on synthetic tensors, using
    the PRODUCTION initialisation: BMCModel(rgb_encoder_state=<ImageNet RGB state>). A missing
    state is refused, so the preflight can never silently run with a random encoder."""
    import torch
    from bmc_losses import joint_loss_per_image
    from bmc_model import BMCModel, convert_first_conv, decoder_takes_varargs
    if not rgb_state or "conv1.weight" not in rgb_state:
        raise DriverError("preflight requires the ImageNet RGB encoder state; a random encoder is not permitted")
    torch.manual_seed(0)
    model = BMCModel(rgb_encoder_state=rgb_state)
    if model.pretrained_sha256 is None:
        raise DriverError("pretrained_sha256 is null: the ImageNet initialisation was not applied")
    conv1_ok = bool(torch.equal(model.encoder.conv1.weight.detach().cpu(), convert_first_conv(rgb_state["conv1.weight"]).cpu()))
    conv1_sum_ok = bool(torch.equal(model.encoder.conv1.weight.detach().cpu(),
                                    (rgb_state["conv1.weight"][:, 0:1] + rgb_state["conv1.weight"][:, 1:2] + rgb_state["conv1.weight"][:, 2:3]).cpu()))
    if not (conv1_ok and conv1_sum_ok):
        raise DriverError("single-channel conv1 does not equal the SUM of the ImageNet RGB conv1 channels")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    x = torch.randn(batch, 1, size, size, device=device)
    g = (torch.rand(batch, 1, size, size, device=device) > 0.8).float()
    y = torch.randint(0, 2, (batch,), device=device).float()
    code = torch.tensor([0, 1] * (batch // 2), device=device)          # both decoders exercised
    cuda = device.startswith("cuda")
    mem = lambda: {"allocated_MiB": torch.cuda.memory_allocated() / 2 ** 20, "reserved_MiB": torch.cuda.memory_reserved() / 2 ** 20} if cuda else {}
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    before = mem(); t0 = time.perf_counter()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (cuda and bf16) else torch.autocast("cpu", enabled=False)
    model.train()
    with ctx:
        seg, cls = model(x, code)
    loss = joint_loss_per_image(seg, g, cls, y).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {"device": device, "batch": batch, "image_size": size, "autocast_dtype": str(seg.dtype),
            "pretrained_sha256": model.pretrained_sha256, "conv1_equals_rgb_sum": conv1_ok and conv1_sum_ok,
            "conv1_shape": tuple(model.encoder.conv1.weight.shape),
            "loss_dtype": str(loss.dtype), "loss": float(loss.detach()), "smp_decoder_varargs_path": decoder_takes_varargs(model.us_decoder),
            "memory_before": before, "memory_after": mem(),
            "peak_allocated_MiB": (torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else None,
            "single_step_wall_s": wall, "optimizer_steps": 1}


def _preflight(device: str, fetch_fn, size: int = 256) -> dict:
    """Production path: fetch the ImageNet RGB encoder, build BMCModel(rgb_encoder_state=...),
    run exactly one step. Any fetch failure propagates; there is no random-encoder fallback."""
    rgb = fetch_fn()
    return one_step(device, rgb, 16, size, True)


def preflight() -> int:
    import torch, segmentation_models_pytorch as smp
    from bmc_model import fetch_imagenet_rgb_encoder_state
    if not torch.cuda.is_available():
        print("PREFLIGHT REFUSED: CUDA not available (no CPU fallback)"); return 2
    print("R6 HARDWARE-SAFE PREFLIGHT - synthetic tensors, production ImageNet init, ONE optimizer step, then exit", flush=True)
    print(f"  CUDA device : {torch.cuda.get_device_name(0)}")
    print(f"  torch {torch.__version__} | CUDA {torch.version.cuda} | smp {smp.__version__}", flush=True)
    r = _preflight("cuda", fetch_imagenet_rgb_encoder_state, 256)
    for k, v in r.items():
        print(f"  {k:26s} {v}")
    print("PREFLIGHT COMPLETE - exiting after one step (no warm-up, no loop, no extrapolation, no retry)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["preflight", "cache", "smoke", "mu-grid-run", "select-mu", "train",
                                     "final-internal", "final-external", "quarantine-incomplete"])
    ap.add_argument("--reason")
    ap.add_argument("--config"); ap.add_argument("--seed", type=int); ap.add_argument("--run-dir")
    ap.add_argument("--prep-root", default=str(DEFAULT_PREP_ROOT)); ap.add_argument("--device", default="cuda")
    ap.add_argument("--confirm-final-external", action="store_true")
    a = ap.parse_args(argv); pr = Path(a.prep_root)
    if a.mode == "preflight":
        return preflight()
    if a.mode == "quarantine-incomplete":
        print(json.dumps(quarantine_incomplete(Path(a.run_dir), a.reason or ""), indent=2, default=str)); return 0
    if a.mode == "smoke":
        print(smoke(pr, a.device)); return 0
    if a.mode == "cache":
        print(build_cache(pr, paths(pr)["experiments"] / "_cache" / "deterministic_256")); return 0
    if a.mode == "mu-grid-run":
        print(train(a.config, a.seed, pr, a.device, mode="mu_selection")); return 0
    if a.mode == "select-mu":
        print(select_mu(paths(pr)["experiments"])); return 0
    if a.mode == "train":
        print(train(a.config, a.seed, pr, a.device)); return 0
    m = "final_internal" if a.mode == "final-internal" else "final_external"
    print(final_evaluate(Path(a.run_dir), pr, m, a.device, confirm_external=a.confirm_final_external)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
