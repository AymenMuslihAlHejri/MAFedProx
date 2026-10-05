#!/usr/bin/env python3
"""verify_checkpoints.py - verify every formal checkpoint against its frozen selection record.

Place in  _builders\\R6_tools\\  (NOT inside R5_training). Read-only: it hashes files and
compares; it modifies nothing, needs no GPU, and loads no model.

For every formal run (and every local-only client sub-run) it checks that
  * SHA-256(checkpoints/best.pt) == selection_record.json["checkpoint_sha256"]
  * SHA-256(history.json, canonical form) == selection_record.json["history_sha256"]
  * run_complete.json exists in the run directory

    python verify_checkpoints.py
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

RUNS = Path(r"D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging\Revision 1\dataset preparing\R6_EXPERIMENTS\runs")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(sel_dir: Path, label: str) -> bool:
    rec_p = sel_dir / "selection_record.json"
    if not rec_p.is_file():
        print(f"  [MISSING] {label}: no selection_record.json"); return False
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    ck = sel_dir / "checkpoints" / "best.pt"
    ck_ok = ck.is_file() and sha_file(ck) == rec["checkpoint_sha256"]
    hist = json.loads((sel_dir / "history.json").read_text(encoding="utf-8"))
    h_ok = hashlib.sha256(json.dumps(hist, sort_keys=True).encode()).hexdigest() == rec["history_sha256"]
    ok = ck_ok and h_ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:34s} round {rec['selected_round']:2d} | "
          f"checkpoint {'OK' if ck_ok else 'MISMATCH'} | history {'OK' if h_ok else 'MISMATCH'}")
    return ok


def main() -> int:
    if not RUNS.is_dir():
        print(f"runs folder not found: {RUNS}"); return 2
    results = []
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        complete = (run / "run_complete.json").is_file()
        print(f"{run.name}{'' if complete else '   (INCOMPLETE: no run_complete.json)'}")
        local = run / "local"
        if local.is_dir():
            for client in sorted(p for p in local.iterdir() if p.is_dir()):
                results.append(check(client, f"{run.name} / {client.name}") and complete)
        else:
            results.append(check(run, run.name) and complete)
    n_fail = sum(1 for r in results if not r)
    print(f"\nverified {len(results) - n_fail} of {len(results)} selected models"
          + ("  -> ALL CHECKPOINTS MATCH THEIR FROZEN RECORDS" if n_fail == 0 else "  -> SEE FAILURES ABOVE"))
    return 0 if n_fail == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
