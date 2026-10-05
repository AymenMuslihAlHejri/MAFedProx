# R6 final evaluation — `bmc_evaluate.py`

Place in `_builders\R6_tools\` beside `verify_manifests.py` (required: single source of the 18 reference fingerprints).
Writes **only** under `R6_EXPERIMENTS\_evaluation\`. Never writes into `R5_training` or any formal run directory.

## Enforced order (per selected model)

```
selected checkpoint (SHA-256 = frozen selection record)
  -> validation ledger(s)
  -> tau_val = frozen R4 derive_tau_val(validation ledgers)
  -> tau_val.json written once + tau_val.sha256  (bound to checkpoint, selection record, validation ledgers)
  -> internal-test ledger(s)      refused without a valid tau_val artifact
  -> external ledger(s)           refused without it; never for local-only models
```

## Fail-closed study gate (before any real image is read)

* all 27 formal run-equivalents (`fedavg, fedprox, proposed, centralised, local, N1_mu0, N2_uncapped, N3_cls_only, N4_seg_only` x seeds 101/202/303) exist, carry `run_complete.json`, and are not ABORTED;
* every manifest carries exactly the 18 reference fingerprints, its own config name and seed, and no overrides;
* local-only runs contain exactly the 8 client models, each with a selection record and checkpoint;
* the frozen inputs **on disk now** match the 18 reference fingerprints.

## File tree

```
R6_EXPERIMENTS\_evaluation\
├── <run_id>\                                  (all non-local runs: 24 models)
│   ├── ledgers\<client>__validation.csv (+ .sha256.json)      x 8
│   ├── ledgers\<client>__test.csv (+ .sha256.json, __pred_masks\)  x 8
│   ├── ledgers\<external>__EXTERNAL_TEST_ONLY.csv (+ ...)     x 3
│   ├── tau_val.json
│   └── tau_val.sha256
├── local__seed<S>\local\<client>\             (local-only: 24 client models)
│   ├── ledgers\<client>__validation.csv, <client>__test.csv  (own client only; no external)
│   ├── tau_val.json
│   └── tau_val.sha256
└── evaluation_manifest__<UTC>.json
```

## Schemas

**Ledger** (frozen R4 `bmc_eval.LEDGER_FIELDS`, 29 columns, write-once, sidecar SHA-256):
`record_id, dataset, client_id, split, group_key, group_unit, modality, label, cls_logit, cls_prob, pred_at_0.5,
seg_threshold, dice, iou, hd95, gt_fg_px, pred_fg_px, intersection_px, pred_empty, run_id, method, seed,
selected_round, selection_criterion_name, selection_criterion_value, model_sha256, splits_manifest_sha256,
code_sha256, eval_spec_version`

**tau_val.json**: `frozen, tau_val, derivation {tau_val, youden_j, n_candidates, ...}, source_split = "validation",
run_id, client, checkpoint_sha256, selection_record_sha256, validation_ledgers_sha256 {file: sha256}, created_at`;
`tau_val.sha256` holds the SHA-256 of `tau_val.json`.

**evaluation_manifest__<UTC>.json**: `study_gate {gate, runs[27], n_runs, n_model_units, frozen_inputs_now{18}, checked_at},
units [{run_id, client, checkpoint_sha256, tau_val, tau_val_artifact_sha256, ledgers {file: sha256}}], environment, order`.

## Exact command sequence (after training reaches 27/27)

```
cd "D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging\Revision 1\dataset preparing\CANONICAL_DATASET_BANK\_builders\R6_tools"
python verify_manifests.py          # expect: 27 PASS, 18 of 18 unchanged, 1 interrupted run
python verify_checkpoints.py        # expect: verified 48 of 48 selected models
python test_bmc_evaluate.py         # synthetic only; expect PASS = 19, EVALUATION_TOOL_VERIFIED
python bmc_evaluate.py --device cuda
```

The last command prints `study gate passed: 27 runs, 48 selected models` and the path of the evaluation manifest.
If any precondition fails it stops with `STUDY GATE CLOSED - no real data was read` and the list of defects.
Re-running is safe: existing ledgers and tau_val artifacts are re-verified and reused, never overwritten.

---

# R6 statistical analysis — `bmc_analyze.py`

Place beside `bmc_evaluate.py`. Reads **only** the files listed in one `evaluation_manifest__*.json`
(located directly in `_evaluation\`). Writes **only** under `_evaluation\analysis\<UTC>\`, write-once.

## Input gate (fail-closed, before any statistic)

* the manifest covers exactly the 27 formal runs and records a passed study gate;
* every listed ledger exists, its SHA-256 equals the manifest value, and the frozen R4 loader re-verifies its sidecar;
* every `tau_val.json` matches the manifest hash, its `tau_val.sha256`, and the recorded value and checkpoint;
* no ledger or `tau_val.json` on disk is missing from the manifest;
* internal-test ledgers complete for every model; external ledgers complete for every non-local model; none for local-only.

## Outputs (`_evaluation\analysis\<UTC>\`)

| File | Content |
|---|---|
| `confirmatory_tests.csv` | exactly 6 tests: Proposed vs FedAvg / FedProx / Local-only x ROC-AUC / Dice; `delta_hat` (Proposed - comparator), `ci_low`, `ci_high` (2.5th/97.5th percentiles), `p_raw` (centred +1), `p_holm`, `reject` (family-wise alpha 0.05), `B_attempted`, `B_valid`, `B_undefined`, `ci_flag`, `per_seed_delta`, `bootstrap_seed` |
| `main_descriptive.csv` | Proposed, FedAvg, FedProx, Local-only, Centralised: client-macro per seed + mean, SD (ddof 1), tau = 0.5 |
| `ablations_descriptive.csv` | N1-N4; N3 classification metrics only, N4 segmentation metrics only |
| `external_descriptive.csv` | BLUIS, BUS-UCLM, CDD-CESM separately; no Local-only; N3/N4 retained task only |
| `per_client_supplementary.csv` | frozen R4 `per_client_table` per configuration x seed |
| `tau_val_secondary.csv` | threshold metrics at each model's tau_val (local-only: each client model's own), internal and external, descriptive |
| `analysis_manifest.json` | input hashes, R4 code fingerprints, analysis code hash, bootstrap seed, attempted/valid/undefined replicates per test, Holm family size, scope statement, output hashes, environment |

All metrics, macro aggregation, resampling plan, percentile CI, centred p-value, seed summary and the paired multi-seed
bootstrap are the frozen `bmc_stats.py` functions. Only the Holm step-down adjustment is implemented in the tool.

## Exact commands (after `bmc_evaluate.py` has finished)

```
cd "D:\PhD\002 FL Mul BC\01 ready\Springer\BMC Med Imaging\Revision 1\dataset preparing\CANONICAL_DATASET_BANK\_builders\R6_tools"
python test_bmc_analyze.py                      # synthetic only; expect PASS = 20, ANALYSIS_TOOL_VERIFIED
dir "..\..\..\R6_EXPERIMENTS\_evaluation\evaluation_manifest__*.json"
python bmc_analyze.py --manifest "<full path of that evaluation_manifest__*.json>"
```

---

## R4 G1 guard policy (added after the real run stopped on N4_seg_only__seed101)

The frozen R4 heuristic `_probability_like` ("additional guard only") refuses any ledger whose >= 20 classification
inputs all lie in [0, 1]. N4 (lambda_cls = 0) has an untrained classification head whose genuine raw logits can
fall entirely in [0, 1]. `bmc_evaluate.py` waives **only the classification check**, **only** when:
the guard raised on classification; the manifest is `N4_seg_only` with `lambda_cls = 0`; and the loaded model has
no output activation and a classification head ending in `nn.Linear`. The genuine logits are written unchanged;
the segmentation guard stays active; a write-once `<ledger>.R4_G1_WAIVER.json` records the justification and is
listed (with SHA-256) in the evaluation manifest. `bmc_analyze.py` accepts waivers only for N4, verifies their
hashes, and refuses unlisted ones. N4 classification metrics are excluded from all analyses (pre-specified).
`bmc_eval.py` is not modified. Tests: `test_bmc_evaluate.py` E19-E24, `test_bmc_analyze.py` A21-A24.

---

## Gate scope correction (bmc_analyze.py) and Figure 6 selection

`bmc_analyze.load_inputs` originally treated every CSV under `_evaluation\` (except `analysis\`) as a potential
unlisted ledger, so figure outputs under `_evaluation\figures\` would have closed the gate. Unlisted-ledger
discovery is now restricted to `ledgers\` folders, the only place ledgers are written. Every listed ledger is
still re-hashed; an unlisted ledger inside any `ledgers\` folder still closes the gate (A15); figure outputs do
not (A25). No statistic is affected. The existing analysis (`analysis\20261004T184158374207Z`) was produced
by the previous code version (hash recorded in its manifest) and does not need to be re-run.

`bmc_figure6_select.py` applies the frozen Figure 6 selection rule to the verified internal-test ledgers of
seed 101 (Dice column only) and writes `_evaluation\figures\figure6_selection__<UTC>\` with a write-once
selection table and manifest. It opens no image and no mask (test S05).
