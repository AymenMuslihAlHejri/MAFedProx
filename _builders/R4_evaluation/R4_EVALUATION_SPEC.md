# R4 — Evaluation Protocol Specification

**Status.** All decisions APPROVED, except **R4-8, which is PROVISIONAL** pending R5
(dependency D2). The open items O1–O3 are resolved (Section 13). **No evaluation code
exists yet.**

**Scope.** Defines how every model in the revision is evaluated: federated,
centralised, local-only and ablation models alike. The same components score every
method, so each is evaluated by identical code on identical samples with identical
thresholds (Editor item 2).

**Inputs.** The frozen split manifest (R2), the verified loader (R3), and model
outputs as defined in Section 1.

---

## 1. Model-output contract (R4-4)

1.1 Every model returns **raw logits**: one scalar classification logit per sample,
and one segmentation logit per pixel on the 256 × 256 output grid. No model applies
a sigmoid to its outputs.

1.2 The evaluator applies the logistic sigmoid **exactly once**:
p = σ(z) = 1 / (1 + e^(−z)), computed in the numerically stable form
σ(z) = e^z / (1 + e^z) for z < 0 and 1 / (1 + e^(−z)) for z ≥ 0.

1.3 **Decision rule.** A sample is predicted malignant **iff p ≥ τ**. A pixel is
predicted foreground **iff σ(z) ≥ τ_seg**. The same `≥` rule is used at every
threshold, primary and secondary.

1.4 **Consequence for R5.** The legacy classification head ends in `nn.Sigmoid()`.
The R5 architecture must remove it, or the contract is violated (dependency D1,
Section 11).

1.5 **Deterministic unit tests** verify the transformation mathematically, not by
detecting the historical [0.50, 0.73] signature:

| Test | Property verified |
|---|---|
| U1 | σ(0) = 0.5 exactly |
| U2 | σ(ln 3) = 0.75 and σ(−ln 3) = 0.25, to machine precision |
| U3 | symmetry: σ(−z) = 1 − σ(z) for z on a grid over [−30, 30] |
| U4 | inverse: logit(σ(z)) = z to within 1e−9 for z ∈ [−20, 20] |
| U5 | strict monotonicity over the grid |
| U6 | stability: σ(±1000) returns 1 and 0 without overflow, NaN or warning |
| U7 | exactly-once: for known logits z_i, the evaluator's stored probability equals σ(z_i) and differs from σ(σ(z_i)) wherever the two differ |
| U8 | every ledger probability is recomputable from its stored logit (Section 7) |

The historical-signature check is retained only as an additional guard, never as the
primary test.

---

## 2. Metrics

### 2.1 Classification (R4-1)
Positive class = malignant. From the confusion counts TP, FN, TN, FP at a stated
threshold, under the rule p ≥ τ:

| Metric | Definition |
|---|---|
| Sensitivity | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Balanced accuracy | (Sensitivity + Specificity) / 2 |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| F1 | 2TP / (2TP + FP + FN) |
| ROC-AUC | threshold-free; Mann–Whitney rank statistic on σ(z), ties counted ½ |

ROC-AUC is the primary classification metric. A metric whose denominator is zero is
reported as undefined, never imputed.

### 2.2 Segmentation (R4-2, R4-3, O1)
Computed per image in the **256 × 256 model-output space**, against the ground-truth
mask resized by nearest neighbour (R3). The predicted mask is σ(segmentation
logit) ≥ τ_seg, with **τ_seg = 0.5 fixed** and not tuned.

| Metric | Role | Definition |
|---|---|---|
| Dice | **primary** | 2\|P ∩ G\| / (\|P\| + \|G\|) |
| IoU | secondary | \|P ∩ G\| / \|P ∪ G\| |
| HD95 | secondary | 95th percentile of the symmetric boundary-to-boundary Euclidean distances, in pixels of the 256 × 256 grid |

HD95 is reported in pixels, not millimetres: pixel spacing is not available
uniformly across datasets, and direct resizing does not preserve it.

Pixel-level sensitivity and specificity are **not** part of the principal
segmentation reporting.

**Empty predictions (O1).** All ground-truth masks in the cohort are non-empty
(verified in the bank builds), so the both-empty case does not arise. When the
prediction is empty against a non-empty ground truth:
- Dice = 0 and IoU = 0;
- **HD95 is undefined** and is excluded from every HD95 summary;
- the **count and percentage of empty predictions** are reported alongside every
  HD95 summary, per client and in every cross-client summary.

---

## 3. Aggregation, defined per metric

Per-client results are **mandatory** in every results table. Cross-client summaries
never replace them.

| Metric | Within a client | Primary cross-client summary | Secondary summary |
|---|---|---|---|
| ROC-AUC | computed once over all of the client's samples | **client-macro**: unweighted mean of the eight client AUCs | **none**: a pooled AUC ranks scores from clients with different score distributions against each other and does not summarise per-client discrimination |
| Sensitivity, Specificity, Balanced accuracy, Accuracy, F1 | computed from the client's confusion matrix, never from batch averages | **client-macro** | **micro**: the same formula applied to the confusion matrix summed over clients |
| Dice, IoU | per image, then the mean over the client's images | **client-macro** | **micro**: the mean over all images of all clients |
| HD95 | per image; per client the **median and IQR** over images where HD95 is defined, with the count and percentage of empty predictions | **client-macro**: unweighted mean of the eight client medians | **micro**: median and IQR over all images of all clients where HD95 is defined |

Client-macro is primary because the federated question is how well the model serves
each institution; a micro summary is dominated by the largest client (CBIS-DDSM holds
422 of the 1,041 internal test images).

External datasets are summarised separately (Section 8).

---

## 4. Classification operating points

4.1 **Primary: τ = 0.5 (R4-5).** Fixed before any training. Every
threshold-dependent classification result in the main text is reported at τ = 0.5.

4.2 **Secondary: τ_val (R4-6, R4-7, O3).** A single global threshold derived from
the pooled internal **validation** predictions of the selected checkpoint:
- **Candidates:** all distinct validation probabilities under the rule p ≥ τ,
  together with **τ = +∞** to represent the all-negative operating point. The
  smallest observed probability already represents the all-positive operating point,
  so no further candidate is needed.
- **Criterion:** maximise Youden's J = Sensitivity + Specificity − 1.
- **Ties:** if several candidates attain the maximum J, choose the one closest to
  0.5; if an exact distance tie remains, choose the **lower** threshold.
- It is then frozen and applied unchanged to the internal test and external datasets.

4.3 τ_val is a **clearly labelled secondary sensitivity analysis**. It never replaces
the τ = 0.5 results, and every table or figure using it is labelled
"τ_val, validation-derived, secondary".

4.4 No client-specific or dataset-specific threshold is derived.

4.5 The legacy value τ = 0.566 is not carried forward; it was an artefact of a double
sigmoid (R0, F3).

---

## 5. Checkpoint selection (R4-8) — PROVISIONAL

5.1 **Criterion:** minimisation of a **pre-specified joint validation loss**, the
same joint segmentation–classification loss used for training.

5.2 **PROVISIONAL.** The exact formulation of the joint loss, its component weights,
and its aggregation across clients must be **frozen in R5, before any training**.
Until then, R4-8 is not implementable (dependency D2, Section 11).

5.3 The rule, once frozen: evaluate the validation loss at every round (or epoch, for
non-federated models); select the round with the minimum; break ties with the
earliest round.

5.4 **Test data never participate in checkpoint selection**, for any method.

5.5 The identical rule applies to every method: the centralised model is selected on
the same validation partitions with the same loss, and each local-only model on its
own client's validation partition.

5.6 The composite S = ½·Dice + ½·AUC is **not** adopted.

---

## 6. Data-role isolation

| Partition | Permitted use |
|---|---|
| Train | model fitting |
| Validation | checkpoint selection (Section 5) and the secondary τ_val (Section 4.2); nothing else |
| Internal test | final evaluation only, once, after selection is frozen |
| External (BUS-UCLM, BLUIS, CDD-CESM) | final external evaluation only, after selection is frozen |

---

## 7. Prediction-level ledger (R4-9)

One file per (run, dataset, split), with **one row per record**. Ledgers are
**write-once**: each is hashed (SHA-256) when written and is never modified
afterwards. Anything derived later is written by the analysis module to its own
outputs, not back into a ledger.

| Field group | Fields |
|---|---|
| identity | `record_id`, `dataset`, `client_id`, `split`, `group_key`, `group_unit`, `modality`, `label` |
| classification | `cls_logit` (raw, full precision), `cls_prob` = σ(`cls_logit`), `pred_at_0.5` |
| segmentation | `seg_threshold` (= 0.5), `dice`, `iou`, `hd95` (undefined when `pred_empty`), `gt_fg_px`, `pred_fg_px`, `intersection_px`, `pred_empty` |
| provenance | `run_id`, `method`, `seed`, `selected_round`, `selection_criterion_name`, `selection_criterion_value`, `model_sha256`, `splits_manifest_sha256`, `code_sha256`, `eval_spec_version` |

- `intersection_px`, `gt_fg_px` and `pred_fg_px` allow Dice and IoU to be recomputed
  exactly without the model.
- `cls_prob` must be recomputable from `cls_logit` (U8).
- τ_val and predictions at τ_val are **not** stored in the ledger. They are produced
  by the analysis module from `cls_prob` and the frozen τ_val (Section 10.2).
- Predicted masks are stored as PNG for the internal test and external datasets only.

---

## 8. External evaluation

- Same metrics and definitions as Section 2.
- Classification at τ = 0.5 (primary) and at the frozen τ_val (secondary).
- **No recalibration, no threshold tuning, no checkpoint selection, and no use in any
  development decision.**
- Each external dataset is reported separately. **No metric pools internal with
  external data**, and no summary across the three external datasets is pre-specified.
- Each dataset is evaluated through the decoder of its own modality: ultrasound for
  BLUIS and BUS-UCLM, mammography for CDD-CESM.

---

## 9. Uncertainty and comparisons

9.1 **Confidence intervals (R4-10).** Cluster bootstrap, **2,000 attempted
replicates**, 95% percentile intervals, fixed bootstrap seed. The resampling unit is
the `group_key` from the R2 manifest, the strongest identifier established in R1/R2:
patient for INbreast, CBIS-DDSM, AIISLAB, BUS-UCLM and CDD-CESM; case for BUS-BRA;
image for BUSI, BUS-UC, BrEaST, DMID and BLUIS.
- Per-client interval: groups resampled within the client.
- Macro and micro intervals: groups resampled **independently within each client**,
  each client's metric recomputed, and the summary recomputed. A macro replicate is
  undefined if any client's metric is undefined in that replicate.
- External: groups resampled within each dataset.

Resampling individual images is not used where a group spans several images, because
it treats correlated images as independent and narrows the intervals spuriously. For
the same reason, DeLong's test and image-level Wilcoxon tests are not used.

9.2 **Undefined replicates (O2).** A replicate in which a metric is undefined (for
example, a single-class resample, for which ROC-AUC, sensitivity or specificity is
undefined) is **excluded for that metric only**; the same replicate remains valid
for every metric that is defined in it.
- **Never redraw** a replicate until the metric becomes defined.
- For every interval, record **`n_valid`** and **`n_undefined`**, with
  `n_valid + n_undefined = 2,000`.
- Flag the interval **`UNSTABLE_CI`** when more than 5% of the 2,000 attempted
  replicates are undefined, that is, when `n_undefined > 100`.

9.3 **Comparisons (R4-11).** Paired cluster bootstrap: both methods are evaluated on
the **same** resampled groups in each replicate, giving a distribution of Δ; report Δ
with its 95% interval and a two-sided bootstrap p-value. The O2 rules apply to Δ.
- The **confirmatory comparison family is not frozen yet**. It must be pre-specified
  after R5 defines the final baselines and **before any R6 result is inspected**, and
  Holm's correction is applied within that family only (dependency D3).
- **Ablations are descriptive.** They are reported with intervals and are not part of
  the confirmatory family.

9.4 **Seed variability (R4-12).** Every summary is reported as mean ± SD across
independent training seeds. The number of seeds is frozen in R5, before training
(dependency D4).

---

## 10. Implementation architecture

Three components, each with a single responsibility. None of them performs another's
task.

### 10.1 `bmc_eval.py` — per-record evaluation
**model outputs → exactly-once sigmoid → masks and probabilities → per-record
metrics → prediction ledger.**
- Input: raw logits from one model on one (dataset, split).
- Output: one write-once ledger (Section 7) and, for test and external, predicted
  mask PNGs.
- Computes per-record quantities only: `cls_prob`, `pred_at_0.5`, `dice`, `iou`,
  `hd95`, the pixel counts and `pred_empty`.
- **Does not** aggregate, bootstrap, compare methods, derive τ_val or select a
  checkpoint.
- Ships with U1–U8 and tests on synthetic outputs whose per-record metric values are
  known in closed form.

### 10.2 Analysis and statistics module — reads frozen ledgers only
- Input: frozen ledgers, each verified against its recorded SHA-256 before use.
- Computes: aggregate metrics per Section 3; τ_val per Section 4.2 and predictions at
  τ_val; cluster-bootstrap intervals per Sections 9.1–9.2; paired comparisons per
  Section 9.3; seed summaries per Section 9.4.
- **τ_val derivation receives validation ledgers only**, and asserts that every row
  it reads has `split = validation`.
- Never reads model weights or raw images, and never writes to a ledger.

### 10.3 Checkpoint-selection component — validation only
- Input: the per-round validation criterion values produced during training.
- Output: the selected round, its criterion value, and a selection record that is
  hashed and frozen.
- **Never inspects test or external ledgers.** It accepts inputs only when every
  entry carries `split = validation`, and refuses any other split.
- **Ordering safeguard:** internal-test and external ledgers are produced only for
  the selected checkpoint, and only **after** the selection record is frozen. No
  test or external ledger can exist when selection runs.
- Implementable only after R5 freezes the joint loss (D2).

---

## 11. Dependencies on R5 (must close before training)

| ID | Dependency | Blocks |
|---|---|---|
| **D1** | Model heads output raw logits; the legacy final `nn.Sigmoid()` is removed | R4-4 contract |
| **D2** | Exact joint-loss formulation, component weights, and cross-client aggregation of the validation loss, frozen before training | **R4-8 (PROVISIONAL)** and the selection component (10.3) |
| **D3** | Final baselines defined, then the confirmatory comparison family pre-specified before any R6 result is inspected | R4-11 Holm family |
| **D4** | Number of independent training seeds frozen before training | R4-12 |

---

## 12. Constraints common to all components

- One set of components, used unchanged by every method.
- Classification metrics from confusion counts over whole partitions; segmentation
  metrics per image. No batch-level averaging anywhere.
- No component reads a test or external partition during model development or
  selection.

---

## 13. Resolved items

| ID | Decision (APPROVED) |
|---|---|
| **O1** | Empty prediction against non-empty ground truth: Dice = 0, IoU = 0, HD95 undefined and excluded from HD95 summaries. Where defined, HD95 is summarised per client by median and IQR. The count and percentage of empty predictions are reported alongside every HD95 summary. |
| **O2** | Undefined replicates are excluded for the affected metric only; `n_valid` and `n_undefined` are recorded; the interval is flagged `UNSTABLE_CI` when more than 5% of the 2,000 attempted replicates are undefined. Replicates are never redrawn. |
| **O3** | τ_val from the pooled internal validation predictions: candidates are all distinct validation probabilities under the rule p ≥ τ, together with τ = +∞ for the all-negative operating point (the smallest observed probability already gives the all-positive point); maximise Youden's J; ties go to the threshold closest to 0.5, and an exact distance tie to the lower threshold. Secondary only. |

---

## 14. Decision register

| ID | Status | Final decision |
|---|---|---|
| R4-1 | APPROVED | ROC-AUC, Sensitivity, Specificity, Balanced accuracy, Accuracy, F1 |
| R4-2 | APPROVED | Dice primary; IoU, HD95 secondary; no pixel-level Sens/Spec in principal reporting |
| R4-3 | APPROVED | evaluation in the 256 × 256 model-output space |
| R4-4 | APPROVED | raw logits; sigmoid applied exactly once by the evaluator; deterministic tests U1–U8 |
| R4-5 | APPROVED | primary τ = 0.5, rule p ≥ τ |
| R4-6 | APPROVED | τ_val by Youden on validation, secondary sensitivity analysis only |
| R4-7 | APPROVED | one global τ_val |
| R4-8 | **PROVISIONAL** | minimum pre-specified joint validation loss; blocked by D2 |
| Aggregation | APPROVED | per-client mandatory; client-macro primary; micro secondary where appropriate; defined per metric (Section 3) |
| R4-9 | APPROVED | write-once prediction ledger including raw logits, segmentation threshold and selection-criterion value |
| R4-10 | APPROVED | cluster bootstrap, 2,000 attempted replicates, 95% percentile, R1/R2 grouping unit |
| R4-11 | APPROVED (family deferred) | paired cluster bootstrap; confirmatory family blocked by D3; ablations descriptive |
| R4-12 | APPROVED (count deferred) | mean ± SD across seeds; count blocked by D4 |
| External | APPROVED | fully separate: no recalibration, tuning, selection, or pooling with internal data |
| O1 | APPROVED | Section 13 |
| O2 | APPROVED | Section 13 |
| O3 | APPROVED (amended) | Section 13 |
| Architecture | APPROVED | three components: `bmc_eval.py`, analysis module, validation-only selection (Section 10) |
