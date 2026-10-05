#!/usr/bin/env python3
"""verify_manifests.py - verify every formal run manifest, the quarantined runs, and the
integrity of the frozen code currently on disk.

Place in  _builders\\R6_tools\\  (NOT inside R5_training). Read-only: it reads and hashes
files; it modifies nothing, needs no GPU, and can run while training is in progress.

  1. every run in runs\\ : 18 frozen fingerprints, mu and its source, seed, configuration,
     no overrides, completion, wall-clock time
  2. every quarantined run in _aborted\\ : original run, reason, time
  3. the frozen inputs on disk NOW still match the reference fingerprints

    python verify_manifests.py
"""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # _builders\R6_tools
BUILDERS = HERE.parent
PREP = Path(r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging\Revision 1\dataset preparing")
EXP = PREP / "R6_EXPERIMENTS"

# reference: the 18 fingerprints recorded by the first formal run (fedprox_mu_0.001__seed101)
REF = {
 "bmc_data": "99b7d3ac88cb0e2722025c596be27cef629b02f30d9f5bb7cb478284b5a6c116",
 "bmc_eval": "70afefefdb4c7e52cc26273ddd44cb085a0f84aa9d05757f29791d14472d8304",
 "bmc_stats": "90d6d2b366337b46f3dca18adc9530219a01d4a44c5b0e611509d41616073a5a",
 "cohort_manifest": "d7169d17b19078c75bef3347fa56b0db0f3c48f0227a36f693c5432d4d1e7d98",
 "preregistration": "0e1122f555a4c7e0379c20c18024bb159d45c3122e3be47ddf82a42bafc16e10",
 "r5:bmc_aggregation.py": "b8907eb2e56c9a02144adef2b96e73141ddd1e23e443ee237d85432e451db857",
 "r5:bmc_cache.py": "caf4d0205a71cce404cdf6a895601e084af39b8001ce02cf8bc1fb8815771522",
 "r5:bmc_config.py": "c2e7d4e93b75a4354a4e22f82b5bf6bcc493184985f7976d1a47485cdcf002ef",
 "r5:bmc_losses.py": "76334422dfdc42c0f1bbb891e7681ce44d637acc49f31af2b8edf3c7ae68727d",
 "r5:bmc_model.py": "2cce9dd1a79566c1c08d0540df4c9de598e41fe2b374921a5274550ecad345b2",
 "r5:bmc_repro.py": "615d7ef807be3b19493232f6f8ad7df9f488f2fadba4ff3029b2cee00d3c66c4",
 "r5:bmc_run.py": "261d603144d0f70927f6d89734a148164d87d36d0212c1872eb715d0ba0df03c",
 "r5:bmc_sampling.py": "375e1b5e9b9eb48de5e9c2527f1d3bd0c10874225d0583f040b96b2d440959ae",
 "r5:bmc_schedule.py": "49fbea145c2da362638f23d5fd9f9b92605eb8de622f699131591c61b00fd673",
 "r5:bmc_select.py": "eaadda0c79d300e047f6db1fef74ada0176f781a28ff69a1f5ae5ce93731b254",
 "r5:bmc_train.py": "81487da8d40a09b539fc385d96cf36a79b215bedf7de6a3f83665144c7917781",
 "splits_manifest": "b4373dc2e19a8a05835897d6e5d576b5496219fcbbedd718fae08fa106ae81f9",
 "splits_report": "a09beeb0f44a6250c8339c14bf10babf2f4e4727b96cbe1938d808400a6f44de",
}
MU_RECORD_SHA = "4e318c1bbc72bd13684f5da984693e702fd6a7e74df135368eedb66ce2d6cc68"
SEEDS = {101, 202, 303}
NEEDS_GRID_MU = {"proposed", "fedprox", "N2_uncapped", "N3_cls_only", "N4_seg_only"}
FEDERATED = NEEDS_GRID_MU | {"fedavg", "N1_mu0"}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected_problems(m: dict) -> list:
    """Every deviation of a manifest from the frozen protocol; an empty list means PASS."""
    p, c, name = [], m.get("config", {}), m.get("config_name")
    if m.get("frozen_input_sha256") != REF:
        bad = sorted(k for k in set(REF) | set(m.get("frozen_input_sha256", {})) if REF.get(k) != m["frozen_input_sha256"].get(k))
        p.append(f"fingerprints differ: {bad}")
    if m.get("seed") not in SEEDS: p.append(f"seed {m.get('seed')} not frozen")
    if m.get("test_overrides") != {}: p.append(f"overrides present: {m.get('test_overrides')}")
    if "purpose" in c: p.append("grid config used as a formal run")
    mu = m.get("mu", {})
    if name in NEEDS_GRID_MU:
        if not (mu.get("mu") == 0.01 and mu.get("source") == "frozen mu_record.json" and mu.get("mu_record_sha256") == MU_RECORD_SHA):
            p.append(f"mu not read from the frozen record: {mu}")
    elif mu.get("mu") != 0.0:
        p.append(f"mu should be 0, got {mu.get('mu')}")
    if name in FEDERATED and (c.get("rounds"), c.get("local_epochs")) != (50, 2): p.append("not 50 rounds x 2 local epochs")
    if name in ("centralised", "local") and (c.get("epochs"), c.get("eval_every")) != (100, 2): p.append("not 100 epochs / eval every 2")
    lam = (c.get("loss", {}).get("lambda_seg"), c.get("loss", {}).get("lambda_cls"))
    want = {"N3_cls_only": (0.0, 1.0), "N4_seg_only": (1.0, 0.0)}.get(name, (1.0, 1.0))
    if lam != want: p.append(f"loss weights {lam}, expected {want}")
    if name == "N2_uncapped" and c.get("aggregation", {}).get("capped") is not False: p.append("N2 must be uncapped")
    if name in ("proposed", "N1_mu0", "N3_cls_only", "N4_seg_only") and c.get("aggregation", {}).get("capped") is not True: p.append("cap missing")
    return p


def main() -> int:
    n_fail = 0
    print("=" * 96); print("1. FORMAL RUN MANIFESTS"); print("=" * 96)
    runs = sorted(d for d in (EXP / "runs").iterdir() if d.is_dir()) if (EXP / "runs").is_dir() else []
    for d in runs:
        mp, cp = d / "run_manifest.json", d / "run_complete.json"
        if not mp.is_file():
            print(f"  [FAIL] {d.name}: no run_manifest.json"); n_fail += 1; continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        probs = expected_problems(m)
        if not cp.is_file(): probs.append("INCOMPLETE (no run_complete.json)")
        mins = ""
        if cp.is_file():
            t0 = datetime.fromisoformat(m["started_at"]); t1 = datetime.fromisoformat(json.loads(cp.read_text())["finished_at"])
            mins = f"{(t1 - t0).total_seconds() / 60:6.1f} min"
        mu = m.get("mu", {}).get("mu")
        print(f"  [{'PASS' if not probs else 'FAIL'}] {d.name:24s} seed {m.get('seed')} | mu {mu} | {mins}")
        for pr in probs: print(f"         - {pr}")
        n_fail += bool(probs)

    print("\n" + "=" * 96); print("2. QUARANTINED (ABORTED) RUNS"); print("=" * 96)
    ab = sorted((EXP / "_aborted").glob("*/ABORT_RECORD.json")) if (EXP / "_aborted").is_dir() else []
    for a in ab:
        r = json.loads(a.read_text(encoding="utf-8"))
        print(f"  {r.get('original_run_id')} | quarantined {r.get('quarantined_at', '')[:19]} | reason: {r.get('reason')}")
        print(f"     files preserved: {len(r.get('file_sha256', {}))} | restart policy: {r.get('restart_policy')}")
    print(f"  total interrupted formal runs: {len(ab)}")

    print("\n" + "=" * 96); print("3. FROZEN INPUTS ON DISK NOW vs REFERENCE"); print("=" * 96)
    bank = PREP / "CANONICAL_DATASET_BANK"
    paths = {"cohort_manifest": bank / "_bmc_cohort" / "BMC_COHORT_MANIFEST.csv",
             "splits_manifest": bank / "_bmc_splits" / "BMC_SPLITS_MANIFEST.csv",
             "splits_report": bank / "_bmc_splits" / "BMC_SPLITS_REPORT.json",
             "bmc_data": BUILDERS / "bmc_data.py", "bmc_eval": BUILDERS / "R4_evaluation" / "bmc_eval.py",
             "bmc_stats": BUILDERS / "R4_evaluation" / "bmc_stats.py",
             "preregistration": BUILDERS / "R5_training" / "R5_PREREGISTERED_INTERPRETATION.md"}
    for f in sorted((BUILDERS / "R5_training").glob("bmc_*.py")):
        paths[f"r5:{f.name}"] = f
    now = {k: sha_file(p) if p.is_file() else "MISSING" for k, p in paths.items()}
    changed = sorted(k for k in set(REF) | set(now) if REF.get(k) != now.get(k))
    print(f"  {len(REF) - len([k for k in changed if k in REF])} of {len(REF)} frozen inputs unchanged"
          + (f" | CHANGED / EXTRA / MISSING: {changed}" if changed else " | nothing changed since the first formal run"))
    n_fail += bool(changed)

    print("\n" + "=" * 96)
    print(f"runs checked: {len(runs)} | failures: {n_fail}"
          + ("  -> ALL MANIFESTS CONFORM TO THE FROZEN PROTOCOL" if n_fail == 0 else "  -> SEE FAILURES ABOVE"))
    return 0 if n_fail == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
