# Validation Methodology and Results

Status: results available as of 2026-09-30.

## Validation layers

There are two distinct validation paths:

1. Inline checkpoint sheets render four fixed held-out samples with the configured validation task. They catch obvious conditioning regressions but do not quantify reference copying.
2. `evaluate_reference_transfer.py` renders deterministic, different-image reference pairs and computes per-sample transfer, line adherence, and copy metrics.

Do not use training self-references to claim reference-transfer performance. The cross-reference evaluator deliberately pairs line art from holdout index `i` with the target image at `(i + ref_offset) % N`.

## Cross-reference cases

| Case | Inputs |
| --- | --- |
| `hint` | Line art plus target-derived dot hints; no reference |
| `cross_ref` | Line art plus a different holdout image as reference |
| `cross_ref_hint` | Line art, target-derived hints, and the different reference |

Reference images pass through the same split builder used by inference, without training deformation. Using a new nonzero offset changes all reference identities while preserving the evaluated line-art examples.

## Metrics

- `hint_mae`: mean absolute color error inside active hint pixels; lower is better.
- `target_mae`: full-image RGB MAE to ground truth. It is not a perceptual metric and reference-driven alternatives can score poorly.
- `line_recall`: fraction of line strokes with a generated edge nearby; higher is better.
- `edge_precision`: fraction of generated edges near line art; lower values indicate unsupported structure.
- `copy_struct_ref` and `copy_struct_ref_bg`: grayscale SSIM structure terms against full reference and background.
- `copy_struct_baseline` and `copy_struct_bg_baseline`: the same scores between the honest target and reference.
- `copy_edge_ref` and `copy_edge_baseline`: generated/reference edge overlap and honest target/reference edge overlap.
- `ref_color_dist`: Lab a/b histogram Bhattacharyya distance on reference foreground; lower suggests stronger color transfer.
- `copy_alarm`: 1 when any copy score exceeds its per-sample baseline by `copy_margin`, currently 0.10.

Aggregate copy scores must be interpreted as deltas from baselines. A low raw score is not sufficient, and one alarm does not imply every condition copied structure.

## Completed results

### Step 20,000, reference offset 57

| Case | Hint MAE | Target MAE | Line recall | Edge precision | Ref color dist | Copy alarms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hint | 28.31 | 30.28 | 0.472 | 0.991 | n/a | n/a |
| cross_ref | n/a | 64.15 | 0.490 | 0.974 | 0.524 | 1/16 |
| cross_ref_hint | 35.53 | 41.61 | 0.482 | 0.970 | 0.579 | 1/16 |

Artifacts: `eval_ref_transfer_step_020000_cross_refs/` in the active run directory.

### Step 40,000, new reference offset 173

| Case | Hint MAE | Target MAE | Line recall | Edge precision | Ref color dist | Copy alarms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hint | 30.75 | 32.89 | 0.493 | 0.993 | n/a | n/a |
| cross_ref | n/a | 65.69 | 0.514 | 0.988 | 0.527 | 3/16 |
| cross_ref_hint | 44.00 | 48.59 | 0.534 | 0.986 | 0.568 | 1/16 |

Artifacts: `eval_ref_transfer_step_040000_cross_refs_offset173/` in the active run directory.

The two evaluations change both checkpoint and reference identity, so they are not a controlled checkpoint-only comparison. Step 40k shows stronger line metrics but worse hint MAE and more reference-only alarm outliers. Inspect per-sample grids before drawing a training trend conclusion.

## Reproduction command

Set `RUN`, choose a completed checkpoint, and choose an offset not used by prior evaluation:

```bash
RUN=/data/shasegawa/adeleine/outputs/flux2_klein_openniji_spatial_ref_split_fixed_strongdeform_fulldata_from_spatial327956_holdout1k_512_ddp2
STEP=step_050000
OFFSET=311

CUDA_VISIBLE_DEVICES=2 python -m adeleine_v2.evaluate_reference_transfer \
  --lora_dirs "$RUN/checkpoints/$STEP" \
  --holdout_manifest "$RUN/holdout_1000.jsonl" \
  --output_dir "$RUN/eval_ref_transfer_${STEP}_cross_refs_offset${OFFSET}" \
  --hf_home /data/shasegawa/adeleine/huggingface \
  --local_files_only --device cuda \
  --examples 16 --ref_offset "$OFFSET" \
  --sketch_root /data/shasegawa/adeleine/openniji/sketchkeras \
  --image_size 512 --inference_steps 12 \
  --spatial_hint_mode fused_masked \
  --spatial_condition_id_mode hint_to_output \
  --reference_conditioning split \
  --reference_condition_mode split \
  --reference_mask_root /data/shasegawa/adeleine/openniji/reference_masks/skytnt_512 \
  --reference_mask_fallback skip \
  --max_condition_images 6
```

Use GPU 2 or 3 only while the active trainer occupies GPUs 0 and 1. Verify with `nvidia-smi` first.

## Evaluation gaps

- Sample count 16 is useful for regression detection but too small for a stable population estimate.
- Offsets are deterministic, not stratified by palette, character count, or semantic similarity.
- No perceptual target metric such as LPIPS is recorded.
- No explicit palette-transfer success threshold is defined.
- Copy alarms are heuristic and can fire on naturally similar composition.
- There is no human preference or condition-faithfulness annotation set.
- The evaluator defaults to deterministic XDoG line art, while training samples multiple line methods.

A stronger release evaluation should freeze several offsets, increase sample count, report confidence intervals, and separate similar-reference from deliberately mismatched-reference cohorts.
