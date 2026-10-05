# Architecture Fact Card — frozen R5 model (basis for Section 3.5)

**Source of every fact: the frozen code that the formal runs fingerprinted.** The files
below are byte-identical to the SHA-256 values recorded in `frozen_input_sha256` of every
formal run manifest (verified against `fedprox_mu_0.001__seed101`):

| File | SHA-256 (first 12) |
|---|---|
| `bmc_model.py` | `2cce9dd1a795` |
| `bmc_losses.py` | `76334422dfdc` |
| `bmc_config.py` | `c2e7d4e93b75` |
| `bmc_train.py` | `81487da8d40a` |

Line numbers refer to those exact files. Items marked **[target]** must be confirmed by
running `count_parameters.py` on the training machine (smp 0.3.3); values shown for them
were measured in the development environment (smp 0.5.0).

---

## 1. Input

| Fact | Value | Evidence |
|---|---|---|
| Channels | 1 (greyscale) | `bmc_model.py` L90 `in_channels=1` |
| Spatial size | 256 × 256 | `bmc_config.py` L11 `"image_size": 256`; R3 direct resize |
| Intensity | R3 declared-range scaling, then (x − 0.5)/0.5 | R3 (locked), Section 3.3 |

## 2. Shared encoder

| Fact | Value | Evidence |
|---|---|---|
| Backbone | ResNet-34 | `bmc_model.py` L90 `get_encoder("resnet34", …)` |
| Depth | 5 downsampling stages | L90 `depth=5` |
| Feature maps at 256 × 256 | (1, 256²), (64, 128²), (64, 64²), (128, 32²), (256, 16²), **(512, 8²)** | measured forward pass |
| Deepest feature (classification input) | 512 × 8 × 8 | L124 `self.cls_head(feats[-1])` |
| Parameters | 21,278,400 **[target]** | `count_parameters.py` |

## 3. Modality-specific segmentation pathways

| Fact | Value | Evidence |
|---|---|---|
| Number of decoders | 2 (ultrasound, mammography), independent parameters | L92–93 |
| Implementation | smp `UnetDecoder`, constructed directly | L92–93 |
| Decoder channels | (256, 128, 64, 32, 16), 5 blocks | L26, L92–93 `n_blocks=5` |
| Block structure | nearest-neighbour upsampling, concatenation with the encoder skip, two 3 × 3 conv–BatchNorm–ReLU | smp source; `count_parameters.py` reports 2 conv3x3 + 2 BN per block **[target]** |
| Centre block | none | smp default; reported by the script **[target]** |
| Segmentation head | Conv2d 16 → 1, kernel 3 × 3, padding 1, **no activation** | L81–82 |
| Output | raw segmentation logits, 1 × 256 × 256 | L131; T02 |
| Parameters per decoder | 3,151,552 + head 145 **[target]** | `count_parameters.py` |
| smp API compatibility | `forward(*features)` (0.3.3) and `forward(features)` (0.5) both receive the identical full feature list | L77 `call_decoder`; T47 |

## 4. Per-sample routing

| Fact | Value | Evidence |
|---|---|---|
| Rule | each sample is routed by its own modality code (0 = US, 1 = MM); never by the first sample of the batch | L118–131 (`index_select(0, idx)` per modality, `index_copy_` back) |
| Invalid codes | refused | L122 |
| Consequence | mixed-modality batches (centralised reference) are routed correctly | T10, T11 |

## 5. Shared classification head

| Fact | Value | Evidence |
|---|---|---|
| Layers | GAP → Flatten → Linear(512, 256) → BatchNorm1d(256) → ReLU → Dropout(0.4) → Linear(256, 64) → ReLU → Dropout(0.2) → Linear(64, 1) | L97–100 |
| Output | **raw logit**; no sigmoid anywhere in the model | L100; T01, T03 |
| Parameters | 148,353 **[target]** | `count_parameters.py` |

## 6. Normalisation

| Fact | Value | Evidence |
|---|---|---|
| BatchNorm momentum | 0.1 in every BatchNorm layer | L30, L103; `bmc_config.py` L22 |

## 7. Initialisation

| Fact | Value | Evidence |
|---|---|---|
| Encoder | public ImageNet ResNet-34 weights, loaded as a 3-channel encoder | L61–65 |
| First-layer conversion | **W₁ = Σ_{c∈{R,G,B}} W[:, c, :, :]**, explicit (not a library default) | L46–50, L114; T42, T43 |
| Encoder BatchNorm statistics | ImageNet values retained at initialisation | L108–114 (all encoder state copied) |
| Decoders, segmentation heads, classification head | **PyTorch default initialisation**, drawn under the training seed. smp's own initialiser runs only via `SegmentationModel.initialize()`, which this model never calls | L92–100; L117–121 in `bmc_train.py` (`seed_everything(seed)` then `BMCModel(...)`) |
| Legacy checkpoints | never loaded; `init_checkpoint`/`resume` refused | `bmc_train.py` L118; T33, T34 |
| Pretrained-state fingerprint | `ea3c2233…2172` (recorded per run) | run manifests, preflight |

## 8. Loss (one implementation for every method)

Notation: zᵢ classification logit, yᵢ ∈ {0,1}, Sᵢ segmentation logits, Gᵢ mask, P pixels.
Define ℓ_BCE(z, y) = −[y log σ(z) + (1 − y) log(1 − σ(z))], computed stably from logits.

| Fact | Value | Evidence |
|---|---|---|
| Classification | ℓ_cls,i = ℓ_BCE(zᵢ, yᵢ), **unweighted** | `bmc_losses.py` L30–31 |
| Soft Dice (per image) | Dᵢ = (2 Σ_p σ(S_ip) G_ip + ε) / (Σ_p σ(S_ip) + Σ_p G_ip + ε), **ε = 1** | L15, L18–21 |
| Segmentation | ℓ_seg,i = ½(1 − Dᵢ) + ½ · (1/P) Σ_p ℓ_BCE(S_ip, G_ip) | L24–27 |
| Joint (per image) | ℓᵢ = λ_seg ℓ_seg,i + λ_cls ℓ_cls,i | L34–39 |
| Weights | **λ_seg = λ_cls = 1**, except ablation **N3 (λ_seg = 0)** and **N4 (λ_cls = 0)** | `bmc_config.py` L18, L49, L51 |
| Batch reduction | arithmetic mean over the batch | L43–44 |
| Precision | all loss terms in float32 (inputs cast with `.float()`); forward in bf16 autocast on CUDA | L19, L26, L31; `bmc_config.py` L21 |
| Class imbalance | handled only by the training sampler, not by loss weights | Section 3.7 |
| Entry point | `task_loss` (single function used by every method) | `bmc_train.py` L29; T38 |
| FedProx term (separate) | (μ/2) Σ_{θ trainable} ‖θ − θ_g‖²; buffers excluded | L47–57; T23–T25 |

## 9. Sizes [target]

| Quantity | Development value (smp 0.5.0) |
|---|---|
| Total parameters | **27,730,147** (all trainable), 105.78 MiB in float32 |
| Encoder | 21,278,400 (76.73%) |
| Each decoder + head | 3,151,697 (11.37%) |
| Classification head | 148,353 (0.53%) |
| Updated by one client per round | 24,578,450 (encoder + own decoder + own head + classification head) |
| BatchNorm float-buffer elements | 21,504; integer buffers: 57 (never averaged) |

**These counts must be re-measured on the training machine** (smp 0.3.3) before they enter
the manuscript; any difference between smp versions would appear here.

---

## Notes for writing Section 3.5

- Cite the decoder as *the U-Net design [Ronneberger] as implemented in
  segmentation_models_pytorch v0.3.3* — the original U-Net used transposed convolutions and
  no BatchNorm.
- Write the classification input as the **deepest** feature map F_i^(5), not the hierarchy.
- State the λ exception for N3/N4.
- Initialisation of the decoders and heads is **PyTorch default**, not smp's initialiser.
