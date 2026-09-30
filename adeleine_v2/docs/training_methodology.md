# Training Methodology

Status: implementation snapshot for 2026-09-30. This document describes the current long-running FLUX.2 Klein LoRA experiment and the code that produces it.

## Objective

Adeleine v2 learns one colorization model that always receives line art and may additionally receive Atari color hints, a reference image, or text. The task mixture preserves useful line-only behavior while teaching each optional condition independently and jointly.

The current experiment specifically tests whether split, strongly deformed self-references can teach palette and attribute transfer without teaching the model to paste reference structure.

## Dataset and split

Training reads image-backed OpenNiji Parquet shards from all locally cached split repositories selected by `--openniji_repo_id all --openniji_parquet_pattern data/*.parquet`. Each record stores embedded image bytes plus `prompt`, `style`, and URL metadata.

At startup, the loader enumerates every Parquet row into an `OpenNijiParquetRecord`. It then creates a deterministic 1,000-record holdout using NumPy `default_rng(seed=3170)` and removes those records from training. The holdout record locations are serialized to `holdout_1000.jsonl` so later evaluation can load the exact images without reconstructing the split.

Important split properties:

- The split is by individual record, not prompt group or character identity.
- Same-prompt or visually similar images can exist on both sides.
- Current reference training uses self-derived references, so it does not select holdout records as training references.
- Evaluation uses the manifest rather than the mutable dataset order.

## Per-sample construction

For every selected record:

1. Decode embedded image bytes with OpenCV.
2. Resize directly to 512 x 512.
3. Generate one line-art variant from available XDoG or cached methods.
4. Generate Atari dot or line hints from target colors and the line art.
5. Build a strongly deformed self-reference and carry its foreground mask through the same geometry.
6. Split the deformed reference into foreground and inpainted background layers.
7. Build the text caption from OpenNiji prompt and style.
8. Sample one explicit task and remove modalities not used by that task.

Dataset decode errors are skipped by trying up to 32 subsequent records. This keeps training alive but means an index does not always map to the same sample when corrupt data is encountered.

## Task mixture

The active run uses the following normalized mixture:

| Task | Probability | Conditions kept | Mode |
| --- | ---: | --- | --- |
| `line` | 0.10 | line | render |
| `line_atari` | 0.25 | line, Atari | render |
| `line_reference` | 0.25 | line, reference | render |
| `line_text` | 0.10 | line, text | diverse |
| `line_reference_atari` | 0.20 | line, reference, Atari | render |
| `all` | 0.10 | line, reference, Atari, text | render |

This is task-level modality dropout, not independent Bernoulli dropout. It guarantees meaningful coverage for specific condition combinations and prevents accidental near-zero line-only probability.

## Line-art augmentation

The default method pool is XDoG, cached SketchKeras pencil, cached digital, cached anime line art, and XDoG/pencil blend. Only methods with available assets participate. The active run supplies `sketch_root`; availability of other cached roots determines whether their methods are included.

All line outputs are normalized to dark lines on white, then receive legacy morphology and dark-line color variation. Density checks reject morphology that erases or overfills the line art.

## Atari hint methodology

The active generator uses `mixed` mode with `atari_dot_prob=0.75`; the remaining probability selects line hints.

- Dot hints sample target-colored patches and return a separate binary mask.
- Line hints copy target-colored horizontal, vertical, or diagonal strokes.
- The trainer forms `spatial_atari = lineart * (1-mask) + atari_rgb * mask`.
- `fused_masked` sends line, fused Atari, and an RGB-expanded mask as condition images.
- `hint_to_output` rewrites only fused Atari tokens to the target output T-plane. Line and reference tokens keep separate T-planes.

The explicit mask preserves white hints, which would otherwise be indistinguishable from absent hints.

## Reference methodology

The active reference policy is `deformed_self` with `strong` deformation. The original target provides reference colors, but the transformation removes exact spatial correspondence:

- 50% horizontal flip;
- rotation up to 25 degrees;
- scale from 0.75 to 1.25;
- translation up to 18% of image size;
- perspective jitter from 3% to 9%;
- elastic amplitude of 4%;
- contrast, brightness, and Gaussian noise;
- optional blur, resolution degradation, and rectangular occlusion.

RGB and mask share flip, affine, elastic, perspective, blur, and resolution transforms. This is essential: reflected or warped foreground must remain labeled foreground or it leaks into `ref_bg`.

SkyTNT masks are looked up by digest in `openniji/reference_masks/skytnt_512`. The active fallback is `skip`, so missing masks do not trigger segmentation during training. With a valid mask:

- `ref_fg` is foreground composited on white;
- `ref_bg` removes mask alpha above 0.1, dilates the hole by roughly 1% of image size, and inpaints it;
- foreground coverage under 2% produces background only;
- coverage over 98% produces foreground only.

The active `reference_background_source=self` derives background from the deformed reference itself. `other` exists but is not enabled.

## Text conditioning

Text is present only for `line_text` and `all`. The prompt is OpenNiji prompt plus optional style, followed by mode and active-condition metadata. Empty text tasks send an empty prompt. FLUX.2's frozen Qwen text encoder produces prompt embeddings with maximum sequence length 128.

Reference tags and WD projected tokens are implemented but disabled in the active `reference_conditioning=split` profile.

## Model and trainable parameters

The base is FLUX.2 Klein 4B loaded in bfloat16. The VAE, text encoder, and base transformer are frozen. PEFT LoRA is trainable on attention projection modules:

```text
to_q, to_k, to_v, to_out.0,
add_q_proj, add_k_proj, add_v_proj, to_add_out
```

The active adapter is rank 16 with alpha 16 and zero dropout. It was initialized by resuming the LoRA from `flux2_klein_openniji_spatial_hint_from_base_holdout1k_512_ddp2/lora`; this reference run is therefore continued specialization, not training from the untouched base model.

## Flow-matching objective

Let `x_0` be packed clean target latents, `epsilon` Gaussian noise, and `sigma ~ Uniform(0,1)` independently per sample. Training constructs:

```text
x_sigma = (1 - sigma) * x_0 + sigma * epsilon
target_flow = epsilon - x_0
```

The noisy target tokens are concatenated with frozen-VAE condition tokens. The transformer receives the combined image IDs, text context, and `sigma` as timestep. Its output is sliced back to the target-token length. The loss is:

```text
MSE(predicted_target_flow.float(), target_flow.float())
```

No loss is applied to condition tokens.

## Optimization and distributed execution

| Setting | Active value |
| --- | --- |
| Optimizer | `torch.optim.AdamW` with PyTorch defaults except LR |
| Learning rate | `1e-4` |
| LR schedule | None |
| Precision | bfloat16 model and latents; float32 loss |
| Gradient clipping | Global norm 1.0 |
| Per-rank batch | 1 |
| World size | 2 |
| Effective global batch | 2 samples per optimizer step |
| Gradient accumulation | None |
| Target steps | 100,000 in this run's local step counter |
| DDP | One full pipeline per GPU; LoRA transformer wrapped in DDP |

`NCCL_P2P_DISABLE=1` is required on the current host. DDP barriers are used after checkpoint and sample intervals so rank 1 waits while rank 0 serializes and renders validation images.

## Checkpoints and inline validation

Rank 0 saves PEFT adapters under `checkpoints/step_NNNNNN` every 10,000 steps and also at step 1 and the final step. Each complete adapter directory must contain at least `adapter_config.json` and `adapter_model.safetensors`.

At the same interval, rank 0 renders four fixed validation examples using task `all`, mixed Atari hints, 12 inference steps, guidance 3.0, and deterministic sample seeds derived from step and row.

The inline sheet is qualitative. Different-image reference evaluation is a separate process described in [validation.md](validation.md).

## Reproducibility limitations

- The trainer seeds Torch but does not globally seed NumPy before stochastic dataset augmentation. Exact augmentation replay is not guaranteed.
- Checkpoints contain LoRA weights, not optimizer state, LR state, current step, sampler epoch, or RNG state.
- Resuming from a LoRA starts a new optimizer and a new local step counter.
- Checkpoint names are based on that local counter. Resuming into the same output directory can overwrite earlier names.
- No immutable experiment config is saved until normal training completion writes `smoke_metrics.txt`.
- The loader uses private diffusers pipeline methods, so dependency upgrades can change behavior.

These limitations must be considered before interpreting a resumed run as mathematically continuous.
