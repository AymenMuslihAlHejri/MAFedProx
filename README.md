# MAFedProx — revision code (BMC Medical Imaging)

Code for *Modality-Aware Federated Aggregation for Joint Breast Lesion Segmentation and
Classification Across Heterogeneous Ultrasound and Mammography Clients*.

This release contains the code used for the revised manuscript. Code from the original
submission is not part of this release and does not produce any reported result.

## Layout
The original layout of the working directory is preserved, so that relative imports and
paths behave exactly as in the reported runs:

```
_builders/                 canonical dataset bank: builders, verifiers, curation steps,
                           cohort freezing and group-aware splitting
_builders/R4_evaluation/   frozen evaluation protocol (metrics, statistics)
_builders/R5_training/     federated / local / centralised training, model selection,
                           frozen configurations (configs/) and tests (tests/)
_builders/R6_tools/        manifest and checkpoint verification, final evaluation,
                           statistical analysis, Figures 2-6, tests
```

## Data
The eleven public datasets are not redistributed. Obtain them from the sources listed in the
manuscript's Data Availability statement. The builders expect the original releases and
write the canonical bank one level above `_builders/`
(i.e. `CANONICAL_DATASET_BANK/_builders` inside the bank folder).

## Execution order
1. Bank: `build_bank_<dataset>.py`, then `verify_bank_<dataset>.py`, for the eleven datasets
   (curation steps and diagnostics in `_builders/`).
2. Cohort and splits: `build_bmc_cohort.py`, then `build_bmc_splits.py` (split seed 20260929).
3. Training: `R5_training/bmc_run.py` with the frozen configurations in `R5_training/configs/`
   (seeds 101, 202, 303; mu grid configurations were used only for mu selection).
4. Verification: `R6_tools/verify_manifests.py`, `R6_tools/verify_checkpoints.py`.
5. Final evaluation and analysis: `R6_tools/bmc_evaluate.py`, then `R6_tools/bmc_analyze.py`
   (bootstrap seed 20260930).
6. Figures: `R6_tools/bmc_figures.py`, `R6_tools/bmc_figure6_select.py`,
   `R6_tools/bmc_figure6_render.py`.

## Provenance
`FILE_MANIFEST.csv` lists the SHA-256 of every file. Files marked `bound` match the
fingerprints recorded in the frozen run, evaluation and analysis manifests.
`R5_training/R5_PREREGISTERED_INTERPRETATION.md` is the internal analysis protocol frozen
before formal evaluation; "preregistered" in its file name refers to this internal freezing,
not to registration in a public registry. It is kept byte-identical because its SHA-256 is
the protocol fingerprint.

`R6_tools/bmc_analyze.py` differs from the version fingerprinted in the frozen analysis
manifest only by a later fix that restricts an artifact-listing check to ledger folders.
Re-running it on the frozen evaluation outputs reproduced all six analysis output files
byte-identically (SHA-256 equal to the frozen manifest); see
`provenance/analysis_reproduction_report.txt`.

## Environment
See `requirements.txt` (Python 3.10.19, CUDA 12.8). Bitwise GPU reproducibility is not claimed.

Release prepared: 2026-10-06
