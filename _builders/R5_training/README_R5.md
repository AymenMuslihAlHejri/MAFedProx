# R5 implementation — placement and verification commands

Place this folder as `CANONICAL_DATASET_BANK\_builders\R5_training\`, beside
`_builders\bmc_data.py` (R3) and `_builders\R4_evaluation\` (R4). Replace
`R4_evaluation\bmc_stats.py` with the supplied version: it is **additive only**, and its
26 existing functions are byte-identical.

Requires `segmentation_models_pytorch` (tested with 0.5.0) in the `torch` environment.

Run, in order, before any study training:

```
cd _builders\R5_training
python tests\test_r5.py                 # T01-T46
python tests\mutate_r5.py               # 17 injected defects
cd ..\R4_evaluation
python bmc_eval.py --selftest           # locked, expect 30/30
python bmc_stats.py --selftest          # locked, expect 39/39
python bmc_stats.py --selftest-multiseed   # additive, expect 8/8
cd ..\R5_training
python bench_throughput.py              # synthetic tensors only; prints env + projection
```

No command here reads a test or external partition, loads a checkpoint, or trains on
study images.

## R6 driver (`bmc_run.py`)

```
python tests\test_r6_driver.py              # 38 synthetic driver, smoke and quarantine tests
python bmc_run.py preflight                  # ONE synthetic bf16 step at batch 16, then exit
```
python bmc_run.py smoke                      # NON_STUDY_SMOKE: 1 round, 1 local epoch, 8 clients, TRAIN/VAL only
```
Later modes (not to be run until approved): `mu-grid-run`, `select-mu`, `train`,
`final-internal`, `final-external --confirm-final-external`.

## Interrupted formal runs

```
python bmc_run.py quarantine-incomplete --run-dir "<R6_EXPERIMENTS\runs\...>" --reason "<what happened>"
```
Moves an incomplete run (no `run_complete.json`) intact to `R6_EXPERIMENTS\_aborted\` with an
`ABORT_RECORD.json`; completed runs are refused. Then rerun the same config and seed: it restarts
from round 0 (no resume). `R5_PREREGISTERED_INTERPRETATION.md` is frozen and hashed into every
formal run manifest; do not edit it.
