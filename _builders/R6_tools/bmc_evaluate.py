#!/usr/bin/env python3
"""bmc_evaluate.py - fail-closed final evaluation of the frozen R6 study.

Place in  _builders\\R6_tools\\  (NOT inside R5_training). It writes ONLY under
R6_EXPERIMENTS\\_evaluation\\ and never writes into R5_training or any formal run directory.

ENFORCED ORDER, per selected model (programmatic, not a convention):
    selected checkpoint (SHA-256 verified against its frozen selection record)
    -> validation ledger(s)
    -> tau_val derived from those validation ledgers only (frozen R4 derive_tau_val)
    -> tau_val artifact written once, hashed, frozen, bound to the checkpoint and ledgers
    -> internal-test ledger(s)       [refused without a valid tau_val artifact]
    -> external ledger(s)            [refused without it; never for local-only models]

FAIL-CLOSED STUDY GATE (real mode): no real validation/test/external image is read until
  * all 27 formal run-equivalents exist and are complete,
  * every run manifest carries the 18 reference fingerprints, and
  * the frozen inputs on disk NOW still match those fingerprints.

Metrics are not defined here: ledgers are written by the frozen R4 bmc_eval.evaluate_split,
tau_val is derived by the frozen R4 bmc_stats.derive_tau_val, and model loading uses the
frozen driver's checkpoint verification (bmc_run._load_selected_model).

    python bmc_evaluate.py            (real study; requires 27/27 complete)
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.dont_write_bytecode = True                              # never write bytecode into R5_training
HERE = Path(__file__).resolve().parent                      # _builders\R6_tools
BUILDERS = HERE.parent
for p in (BUILDERS / "R5_training", BUILDERS, BUILDERS / "R4_evaluation"):
    if p.is_dir():
        sys.path.insert(0, str(p))

DEFAULT_PREP = Path(r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging\Revision 1\dataset preparing")
CONFIGS = ("fedavg", "fedprox", "proposed", "centralised", "local", "N1_mu0", "N2_uncapped", "N3_cls_only", "N4_seg_only")
SEEDS = (101, 202, 303)
EXPECTED_RUNS = tuple(f"{c}__seed{s}" for c in CONFIGS for s in SEEDS)       # 27 formal run-equivalents
POST = ("test", "EXTERNAL_TEST_ONLY")


class EvalError(RuntimeError):
    pass


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reference_fingerprints() -> Dict[str, str]:
    from verify_manifests import REF                        # single source of the 18 reference values
    return dict(REF)


def frozen_inputs_now(prep_root: Path, builders: Path = BUILDERS) -> Dict[str, str]:
    """Fingerprints of the frozen inputs as they are on disk now (same rule as the driver)."""
    bank = Path(prep_root) / "CANONICAL_DATASET_BANK"
    r4 = builders / "R4_evaluation" if (builders / "R4_evaluation" / "bmc_eval.py").is_file() else builders
    paths = {"cohort_manifest": bank / "_bmc_cohort" / "BMC_COHORT_MANIFEST.csv",
             "splits_manifest": bank / "_bmc_splits" / "BMC_SPLITS_MANIFEST.csv",
             "splits_report": bank / "_bmc_splits" / "BMC_SPLITS_REPORT.json",
             "bmc_data": builders / "bmc_data.py", "bmc_eval": r4 / "bmc_eval.py", "bmc_stats": r4 / "bmc_stats.py",
             "preregistration": builders / "R5_training" / "R5_PREREGISTERED_INTERPRETATION.md"}
    for f in sorted((builders / "R5_training").glob("bmc_*.py")):
        paths[f"r5:{f.name}"] = f
    return {k: (sha_file(p) if p.is_file() else "MISSING") for k, p in paths.items()}


# ---------------------------------------------------------------- study gate
class StudyToken:
    """Proof that the fail-closed gate passed; required by every data-reading function."""
    def __init__(self, exp: Path, prep_root: Path, units: list, record: dict):
        self.exp, self.prep_root, self.units, self.record = Path(exp), Path(prep_root), units, record
        self.eval_root = (Path(exp) / "_evaluation").resolve()


def preflight(exp: Path, prep_root: Path, *, expected_runs: Sequence[str] = EXPECTED_RUNS,
              ref: Optional[Dict[str, str]] = None, builders: Path = BUILDERS,
              client_names: Optional[Sequence[str]] = None) -> StudyToken:
    """Reads ONLY manifests, records and file hashes - never images. Raises on any defect."""
    ref = ref if ref is not None else reference_fingerprints()
    problems: List[str] = []
    now = frozen_inputs_now(prep_root, builders)
    changed = sorted(k for k in set(ref) | set(now) if ref.get(k) != now.get(k))
    if changed:
        problems.append(f"frozen inputs on disk differ from the reference: {changed}")
    units = []
    for run_id in expected_runs:
        d = Path(exp) / "runs" / run_id
        cfg_name, seed = run_id.rsplit("__seed", 1)
        mp, cp = d / "run_manifest.json", d / "run_complete.json"
        if not (mp.is_file() and cp.is_file()):
            problems.append(f"{run_id}: missing or incomplete"); continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("frozen_input_sha256") != ref: problems.append(f"{run_id}: manifest fingerprints differ from the reference")
        if m.get("config_name") != cfg_name or str(m.get("seed")) != seed: problems.append(f"{run_id}: manifest identity mismatch")
        if m.get("test_overrides") != {}: problems.append(f"{run_id}: overrides present")
        if (d / "ABORTED").exists(): problems.append(f"{run_id}: marked ABORTED")
        if m.get("regime") == "local":
            subs = sorted(p for p in (d / "local").iterdir() if p.is_dir()) if (d / "local").is_dir() else []
            names = [p.name for p in subs]
            if client_names is not None and sorted(names) != sorted(client_names):
                problems.append(f"{run_id}: local client models {names} != expected {sorted(client_names)}")
            for s in subs:
                if not ((s / "selection_record.json").is_file() and (s / "checkpoints" / "best.pt").is_file()):
                    problems.append(f"{run_id}/{s.name}: selection record or checkpoint missing")
                units.append({"run_id": run_id, "unit_dir": s, "client": s.name, "manifest": m, "local": True})
        else:
            if not ((d / "selection_record.json").is_file() and (d / "checkpoints" / "best.pt").is_file()):
                problems.append(f"{run_id}: selection record or checkpoint missing")
            units.append({"run_id": run_id, "unit_dir": d, "client": None, "manifest": m, "local": False})
    if problems:
        raise EvalError("STUDY GATE CLOSED - no real data was read:\n  - " + "\n  - ".join(problems))
    rec = {"gate": "passed", "runs": list(expected_runs), "n_runs": len(expected_runs), "n_model_units": len(units),
           "frozen_inputs_now": now, "checked_at": datetime.now(timezone.utc).isoformat()}
    return StudyToken(exp, prep_root, units, rec)


# ---------------------------------------------------------------- paths and safety
def _safe(token: StudyToken, path: Path) -> Path:
    p = Path(path).resolve()
    if token.eval_root not in p.parents and p != token.eval_root:
        raise EvalError(f"refusing to write outside {token.eval_root}: {p}")
    return p


def unit_out(token: StudyToken, unit: dict) -> Path:
    base = token.eval_root / unit["run_id"]
    return _safe(token, base / "local" / unit["client"] if unit["local"] else base)


def _write_once_json(token: StudyToken, path: Path, obj: dict) -> str:
    path = _safe(token, path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    return sha_file(path)


# ---------------------------------------------------------------- inference and ledgers
def load_unit_model(token: StudyToken, unit: dict, device: str):
    import bmc_run as RN
    return RN._load_selected_model(unit["unit_dir"], device)    # frozen record required; checkpoint SHA verified


def _infer(model, ds_obj, code: int, device: str, batch: int = 32):
    import numpy as np, torch
    cls, seg, gts = [], [], []
    with torch.no_grad():
        for s in range(0, len(ds_obj), batch):
            items = [ds_obj[i] for i in range(s, min(s + batch, len(ds_obj)))]
            x = torch.stack([torch.as_tensor(it["image"]) for it in items]).to(device)
            sg, cl = model(x, torch.full((len(items),), code, dtype=torch.long, device=device))
            cls.append(cl.float().cpu().numpy()); seg.append(sg[:, 0].float().cpu().numpy())
            gts.append(np.stack([it["mask"][0] for it in items]).astype(np.uint8))
    return np.concatenate(cls), np.concatenate(seg), np.concatenate(gts)


G1_CLS_MSG = "classification inputs all lie in [0, 1]"
WAIVABLE_CONFIG = "N4_seg_only"          # classification head untrained by design (lambda_cls = 0); cls excluded from analysis


def output_activation_free(model) -> bool:
    """Structural evidence that model outputs are raw logits: no output activation anywhere,
    and the classification head ends in a Linear layer."""
    import torch.nn as nn
    bad = (nn.Sigmoid, nn.Softmax, nn.LogSoftmax, nn.Tanh, nn.Hardsigmoid, nn.Softmax2d)
    if any(isinstance(m, bad) for m in model.modules()):
        return False
    return isinstance(list(model.cls_head.children())[-1], nn.Linear)


def write_with_guard_policy(token, unit, model, *, out: Path, records, cls, seg, gts, prov, rec_p, split: str) -> Path:
    """Frozen R4 writer. The R4 G1 classification heuristic ('additional guard only') may be waived
    ONLY for N4 (lambda_cls = 0, classification not analysed) with structural proof of raw logits;
    the waiver is recorded in a write-once file. Every other guard and configuration is untouched."""
    import numpy as np, bmc_eval as EV
    kw = dict(records=records, cls_logits=cls, seg_logits=seg, gt_masks=gts, provenance=prov, out_path=out,
              selection_record=rec_p if split in POST else None)
    try:
        EV.evaluate_split(**kw); return out
    except EV.ContractError as e:
        man = unit["manifest"]
        eligible = (str(e).startswith(G1_CLS_MSG) and man.get("config_name") == WAIVABLE_CONFIG
                    and man["config"]["loss"]["lambda_cls"] == 0.0 and output_activation_free(model))
        if not eligible:
            raise
    target = np.asarray(cls, dtype=np.float64)
    orig = EV._probability_like

    def cls_only_waived(values):                     # waive the classification check only; seg guard unchanged
        v = np.asarray(values, dtype=np.float64)
        if v.shape == target.shape and np.array_equal(v, target):
            return False
        return orig(values)
    EV._probability_like = cls_only_waived
    try:
        EV.evaluate_split(**kw)
    finally:
        EV._probability_like = orig
    _write_once_json(token, out.with_name(out.stem + ".R4_G1_WAIVER.json"), {
        "ledger": out.name, "ledger_sha256": sha_file(out), "guard": "R4 bmc_eval G1 classification heuristic (_probability_like)",
        "reason": "classification head untrained by design (lambda_cls = 0); raw logits happened to lie in [0, 1]",
        "config_name": unit["manifest"]["config_name"], "lambda_cls": unit["manifest"]["config"]["loss"]["lambda_cls"],
        "structural_check": "no output activation modules; classification head ends in nn.Linear",
        "cls_logit_min": float(target.min()), "cls_logit_max": float(target.max()), "n": int(target.size),
        "segmentation_guard": "active (not waived)", "analysis_use": "N4 classification metrics excluded (pre-specified)",
        "created_at": datetime.now(timezone.utc).isoformat()})
    return out


def _ledger(token, unit, model, rec, rec_p, rows, ds: str, split: str, device: str) -> Path:
    """One (dataset, split) ledger via the frozen R4 writer; reused (and re-verified) if it already exists."""
    import bmc_data as BD, bmc_eval as EV, bmc_stats as ST
    out = _safe(token, unit_out(token, unit) / "ledgers" / f"{ds}__{split}.csv")
    if out.exists():
        ST.load_ledger(out)                                   # raises if its sidecar fingerprint does not match
        return out
    o = BD.BMCSplitDataset(ds, split, prep_root=token.prep_root, manifest_rows=rows, **BD.APPROVED_R3)
    code = 0 if o.modality == "ultrasound" else 1
    cls, seg, gts = _infer(model, o, code, device)
    man = unit["manifest"]
    recs = [{"record_id": r["canonical_record_id"], "dataset": ds, "client_id": r.get("client_id", ""), "split": split,
             "group_key": r["group_key"], "group_unit": r.get("group_unit", ""), "modality": o.modality,
             "label": int(r["label_binary"])} for r in o.rows]
    run_label = unit["run_id"] + (f"__{unit['client']}" if unit["local"] else "")
    prov = {"run_id": run_label, "method": man["method"], "seed": man["seed"], "selected_round": rec["selected_round"],
            "selection_criterion_name": rec["selection_criterion_name"],
            "selection_criterion_value": rec["selection_criterion_value"], "model_sha256": rec["checkpoint_sha256"],
            "splits_manifest_sha256": man["frozen_input_sha256"]["splits_manifest"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_with_guard_policy(token, unit, model, out=out, records=recs, cls=cls, seg=seg, gts=gts,
                                   prov=prov, rec_p=rec_p, split=split)


def targets(unit: dict, rows: List[dict], split: str) -> List[str]:
    if unit["local"]:
        if split == "EXTERNAL_TEST_ONLY":
            raise EvalError("local-only models are never evaluated on external datasets (frozen R5.1 decision)")
        return [unit["client"]]
    return sorted({r["dataset"] for r in rows if r["split"] == split})


# ---------------------------------------------------------------- tau_val artifact
def freeze_tau(token, unit, rec, val_ledgers: List[Path]) -> Path:
    import bmc_stats as ST
    out = unit_out(token, unit)
    art = out / "tau_val.json"
    if art.exists():
        require_tau(token, unit, rec, val_ledgers); return art
    t = ST.derive_tau_val([ST.load_ledger(p) for p in val_ledgers])
    obj = {"frozen": True, "tau_val": t["tau_val"], "derivation": t, "source_split": "validation",
           "run_id": unit["run_id"], "client": unit["client"],
           "checkpoint_sha256": rec["checkpoint_sha256"],
           "selection_record_sha256": sha_file(unit["unit_dir"] / "selection_record.json"),
           "validation_ledgers_sha256": {p.name: sha_file(p) for p in val_ledgers},
           "created_at": datetime.now(timezone.utc).isoformat()}
    digest = _write_once_json(token, art, obj)
    _safe(token, out / "tau_val.sha256").write_text(digest + "\n", encoding="utf-8")
    return art


def require_tau(token, unit, rec, val_ledgers: List[Path]) -> float:
    """Gate for every internal-test and external ledger: a valid, frozen, bound tau_val artifact."""
    out = unit_out(token, unit)
    art, sig = out / "tau_val.json", out / "tau_val.sha256"
    if not (art.is_file() and sig.is_file()):
        raise EvalError("no frozen tau_val artifact: internal-test/external ledgers refused")
    if sha_file(art) != sig.read_text(encoding="utf-8").strip():
        raise EvalError("tau_val artifact does not match its frozen fingerprint")
    obj = json.loads(art.read_text(encoding="utf-8"))
    if obj.get("frozen") is not True or obj.get("source_split") != "validation":
        raise EvalError("tau_val artifact is not a frozen validation-derived threshold")
    if obj.get("checkpoint_sha256") != rec["checkpoint_sha256"] or \
            sha_file(unit["unit_dir"] / "checkpoints" / "best.pt") != rec["checkpoint_sha256"]:
        raise EvalError("tau_val artifact is not bound to the selected checkpoint")
    now = {p.name: sha_file(p) for p in val_ledgers}
    if obj.get("validation_ledgers_sha256") != now:
        raise EvalError("validation ledgers differ from those the tau_val artifact was derived from")
    return float(obj["tau_val"])


def post_selection_ledgers(token, unit, model, rec, rec_p, rows, split: str, val_ledgers, device) -> List[Path]:
    if split not in POST:
        raise EvalError(split)
    require_tau(token, unit, rec, val_ledgers)                 # cannot be bypassed by calling this directly
    return [_ledger(token, unit, model, rec, rec_p, rows, ds, split, device) for ds in targets(unit, rows, split)]


# ---------------------------------------------------------------- orchestration
def evaluate_unit(token: StudyToken, unit: dict, rows: List[dict], device: str) -> dict:
    if not isinstance(token, StudyToken) or token.record.get("gate") != "passed":
        raise EvalError("study gate not passed: evaluation refused")
    model, rec, rec_p = load_unit_model(token, unit, device)
    val = [_ledger(token, unit, model, rec, rec_p, rows, ds, "validation", device) for ds in targets(unit, rows, "validation")]
    art = freeze_tau(token, unit, rec, val)
    test = post_selection_ledgers(token, unit, model, rec, rec_p, rows, "test", val, device)
    ext = [] if unit["local"] else post_selection_ledgers(token, unit, model, rec, rec_p, rows, "EXTERNAL_TEST_ONLY", val, device)
    return {"run_id": unit["run_id"], "client": unit["client"], "checkpoint_sha256": rec["checkpoint_sha256"],
            "tau_val": json.loads(art.read_text())["tau_val"], "tau_val_artifact_sha256": sha_file(art),
            "ledgers": {p.name: sha_file(p) for p in val + test + ext},
            "r4_guard_waivers": {w.name: sha_file(w) for w in sorted((unit_out(token, unit) / "ledgers").glob("*.R4_G1_WAIVER.json"))}}


def run(token: StudyToken, rows: List[dict], device: str) -> Path:
    results = [evaluate_unit(token, u, rows, device) for u in token.units]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = token.eval_root / f"evaluation_manifest__{stamp}.json"
    from bmc_repro import env_versions
    _write_once_json(token, path, {"study_gate": token.record, "units": results, "environment": env_versions(),
                                   "order": "checkpoint -> validation -> tau_val (frozen) -> test -> external"})
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-root", default=str(DEFAULT_PREP)); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    prep = Path(a.prep_root); exp = prep / "R6_EXPERIMENTS"
    import bmc_run as RN
    rows = RN.read_manifest(prep)
    token = preflight(exp, prep, client_names=RN.client_datasets(rows))   # fail-closed: raises before any image is read
    print(f"study gate passed: {token.record['n_runs']} runs, {token.record['n_model_units']} selected models")
    print("evaluation manifest:", run(token, rows, a.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
