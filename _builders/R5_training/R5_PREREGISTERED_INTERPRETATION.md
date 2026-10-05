# R5 — Pre-registered Interpretation of the R6 Comparisons

**Status: FROZEN before any formal R6 result was generated.**
This file must not be edited. Its SHA-256 is recorded in `frozen_input_sha256` of every
formal run manifest, and the driver refuses formal runs if the file is missing, so every
R6 result carries proof that this interpretation existed before that result.

Frozen at: 2026-09-30, after the NON_STUDY_SMOKE integration gate and before the FedProx
μ grid. No μ-grid, training, internal-test or external result existed at that time.

---

## 1. Common architecture (not a contribution of any single method)

Every federated method uses the same architecture with **modality-specific decoders**.
Every federated aggregation is **update-aware**: each modality-specific decoder (including
its segmentation head) is aggregated only over clients of its corresponding modality. Shared
encoder and classification-head parameters are aggregated over the clients that update them,
with the Proposed method additionally applying modality-group aggregation as specified below. Decoder preservation is therefore a **fairness rule shared by FedAvg, FedProx
and the Proposed method**, not a component that distinguishes the Proposed method.

## 2. Meaning of each comparison

| Comparison | What it isolates |
|---|---|
| **Proposed vs FedProx** | the effect of **capped modality-group aggregation** under the common update-aware, modality-specific-decoder architecture. It is **not** a test of all four components of the legacy MAA. |
| **Proposed vs N2** | the **within-modality cap** ρ = 0.4 (water-filled). |
| **N2 vs FedProx** | **only** the modality-level fusion of the shared encoder and classification head: 0.5 / 0.5 in N2 versus sample-size modality totals in FedProx (2,503 / 4,830 = **0.5182** and 2,327 / 4,830 = **0.4818** for the frozen training partitions). The within-modality uncapped weights and the decoder aggregation are otherwise the same. |
| **Proposed vs N1** | the **proximal term** (N1 keeps the Proposed aggregation with μ = 0). |
| **N3 / N4** | the **classification** and **segmentation** task objectives respectively (λ_seg = 0 and λ_cls = 0). |

## 3. Pre-specified expectation

Because N2 and FedProx differ only in a modality-level fusion weight of 0.5 versus 0.5182,
**similar performance of N2 and FedProx is a pre-specified expectation, not a post-hoc
interpretation.** A similar result is not evidence that modality balancing matters or does
not matter in general; it reflects that the two modality groups have nearly equal training
sizes in this cohort.

For reference, the within-modality weights that N2 and FedProx share (uncapped,
n-proportional) are:
- ultrasound: BUSI 0.1790, BUS-UC 0.2261, BrEaST 0.0703, BUS-BRA 0.5246;
- mammography: INbreast 0.0318, DMID 0.0743, CBIS-DDSM 0.8440, AIISLAB 0.0498.

The water-filled cap in the Proposed method changes these to at most 0.400 per client, with
its largest effect on CBIS-DDSM in the mammography group (0.8440 → 0.4000) and on BUS-BRA in
the ultrasound group (0.5246 → 0.4000).

## 4. Approved contribution wording

> **Capped within-modality weighting is applied to all applicable parameter groups: directly
> to each modality-specific decoder and, through modality-level fusion, to the shared encoder
> and classification head.**

The short name MAA may be retained only if it is defined in the revised manuscript with this
meaning. Modality balancing through α = 0.5 must not be presented as an important effect in
this cohort, because α = 0.5 is close to the sample-proportional value 0.5182.

## 5. Scope reminders (unchanged from the frozen protocol)

- Confirmatory inference: the six-test Holm family of R5.1 (Proposed vs FedAvg, FedProx,
  Local-only; client-macro Dice and client-macro ROC-AUC), conditional seed-averaged (O1).
- Ablations N1–N4 are descriptive and are not part of the confirmatory family.
- No privacy guarantee beyond raw-data locality in a single-machine simulation of virtual
  federated clients.
