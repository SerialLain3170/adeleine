# Adeleine v2 Takeover Runbook

Operational snapshot: 2026-09-30, America/Los_Angeles.

## First read

The repository is not clean. The Adeleine v2 implementation, reference evaluator, mask tools, architecture document, and generated assets include uncommitted or untracked work. Do not run broad cleanup, reset, checkout, or mass formatting. The recorded Git HEAD is `4d5b5d9` (`add adeleine v2`), but the active experiment depends on working-tree code newer than that commit.

Read [architecture.md](architecture.md), [training_methodology.md](training_methodology.md), and [validation.md](validation.md) before altering the active run.

## Active experiment

Run directory:

```text
/data/shasegawa/adeleine/outputs/flux2_klein_openniji_spatial_ref_split_fixed_strongdeform_fulldata_from_spatial327956_holdout1k_512_ddp2
```

Log:

```text
/data/shasegawa/adeleine/logs/flux2_klein_openniji_spatial_ref_split_fixed_strongdeform_fulldata_from_spatial327956_holdout1k_512_ddp2.log
```

As of this snapshot:

- target is 100,000 local optimizer steps;
- latest complete checkpoint is `step_050000`;
- latest inline sheet is `samples/step_050000.png`;
- checkpoint 50k was written at 2026-09-30 10:55 PDT;
- the trainer has been running since 2026-09-28 14:20 PDT;
- torchrun parent PID was 1505827, with workers 1507557 and 1507558;
- GPUs 0 and 1 are reserved for training;
- GPUs 2 and 3 were free;
- run directory size was approximately 196 MB;
- Hugging Face cache size was approximately 1.0 TB.

PIDs and GPU state are ephemeral. Re-query them rather than assuming these values remain current.

## Exact active launch

The active environment includes:

```bash
NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1
HF_HOME=/data/shasegawa/
```

The active command is equivalent to:

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node 2 -m adeleine_v2.smoke_train_flux_klein \
  --openniji_repo_id all \
  --openniji_parquet_pattern 'data/*.parquet' \
  --hf_home /data/shasegawa/adeleine/huggingface \
  --local_files_only \
  --sketch_root /data/shasegawa/adeleine/openniji/sketchkeras \
  --resume_lora /data/shasegawa/adeleine/outputs/flux2_klein_openniji_spatial_hint_from_base_holdout1k_512_ddp2/lora \
  --output_dir /data/shasegawa/adeleine/outputs/flux2_klein_openniji_spatial_ref_split_fixed_strongdeform_fulldata_from_spatial327956_holdout1k_512_ddp2 \
  --steps 100000 \
  --save_lora --checkpoint_every 10000 \
  --sample_every 10000 --sample_inference_steps 12 \
  --holdout_records 1000 \
  --holdout_manifest /data/shasegawa/adeleine/outputs/flux2_klein_openniji_spatial_ref_split_fixed_strongdeform_fulldata_from_spatial327956_holdout1k_512_ddp2/holdout_1000.jsonl \
  --validation_task all --validation_atari_mode mixed \
  --task_weights line=0.10,line_atari=0.25,line_reference=0.25,line_text=0.10,line_reference_atari=0.20,all=0.10 \
  --atari_mode mixed --atari_dot_prob 0.75 \
  --spatial_hint_mode fused_masked \
  --spatial_condition_id_mode hint_to_output \
  --reference_policy deformed_self \
  --reference_deform_strength strong \
  --reference_background_source self \
  --reference_conditioning split \
  --reference_condition_mode split \
  --reference_mask_root /data/shasegawa/adeleine/openniji/reference_masks/skytnt_512 \
  --reference_mask_fallback skip \
  --max_condition_images 6
```

The filename says `ddp2`; this means two processes and GPUs. The adapter at step 50k is verified rank 16 and alpha 16 even though the current CLI default is rank 8, because `PeftModel.from_pretrained()` loads the inherited adapter configuration.

## Monitoring checklist

Run these without changing process state:

```bash
ps -eo pid,ppid,etime,%cpu,%mem,args | rg 'torchrun|smoke_train_flux_klein'
nvidia-smi
tail -n 80 /data/shasegawa/adeleine/logs/flux2_klein_openniji_spatial_ref_split_fixed_strongdeform_fulldata_from_spatial327956_holdout1k_512_ddp2.log
```

Check complete checkpoints with:

```bash
RUN=/data/shasegawa/adeleine/outputs/flux2_klein_openniji_spatial_ref_split_fixed_strongdeform_fulldata_from_spatial327956_holdout1k_512_ddp2
find "$RUN/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V
test -s "$RUN/checkpoints/step_050000/adapter_model.safetensors"
test -s "$RUN/checkpoints/step_050000/adapter_config.json"
```

At each 10k boundary, rank 0 saves the adapter and then renders the sample sheet. GPU 1 can show 0% utilization while it waits at the DDP barrier; this is expected for a short interval. Investigate if both workers remain idle, the sample file does not advance, and the log does not change for substantially longer than prior sample generation.

## Do not do these during the run

- Do not start another trainer on GPUs 0 or 1.
- Do not edit trainer or dataset code and assume the running process picks it up; Python modules are already loaded.
- Do not modify or remove the mask cache, model cache, holdout manifest, inherited LoRA, or current run directory.
- Do not kill one worker independently; terminate through the torchrun parent if shutdown is unavoidable.
- Do not assume a checkpoint includes optimizer or RNG state.

## Interruption and recovery

There is no signal-triggered emergency checkpoint. If possible, wait for the next complete 10k checkpoint before stopping. A controlled stop targets the torchrun parent, but verify the PID first:

```bash
kill -TERM <torchrun-parent-pid>
```

Recovery is weight-only continuation. The trainer does not restore optimizer, step, sampler epoch, or RNG state. It also restarts checkpoint numbering at step 1. Therefore, do not resume into the existing output directory.

Use a new directory and train only the remaining number of local steps. Example after a stop at 50k:

```bash
OLD=/data/shasegawa/adeleine/outputs/flux2_klein_openniji_spatial_ref_split_fixed_strongdeform_fulldata_from_spatial327956_holdout1k_512_ddp2
NEW=/data/shasegawa/adeleine/outputs/flux2_klein_openniji_spatial_ref_split_fixed_strongdeform_resume_from_050000_holdout1k_512_ddp2
```

Re-run the active launch with these substitutions:

```text
--resume_lora $OLD/checkpoints/step_050000
--output_dir $NEW
--steps 50000
--holdout_manifest $NEW/holdout_1000.jsonl
```

Keep seed, dataset selection, and `--holdout_records 1000` unchanged so the deterministic holdout selection is reproduced. Compare the new manifest with the old one before accepting the continuation:

```bash
cmp "$OLD/holdout_1000.jsonl" "$NEW/holdout_1000.jsonl"
```

The resumed checkpoints will represent continuation steps, not absolute original-run steps. Record the mapping in the new directory name and notes.

## Validation state

Completed quantitative evaluations:

```text
step_020000, offset 57:
  eval_ref_transfer_step_020000_cross_refs/

step_040000, offset 173:
  eval_ref_transfer_step_040000_cross_refs_offset173/
```

Step 50k has an inline sheet but no completed different-image quantitative evaluation at this snapshot. Use a new offset such as 311 to avoid reusing earlier references. The exact command is in [validation.md](validation.md).

The main concerns from current results are:

- reference-only copy alarms increased from 1/16 at step 20k offset 57 to 3/16 at step 40k offset 173;
- reference+hint alarms were 1/16 in both runs;
- hint MAE worsened at step 40k, especially when reference conditioning was active;
- because checkpoint and reference pairs both changed, more offsets are required before calling this a training trend.

## Data and cache dependencies

| Purpose | Path |
| --- | --- |
| FLUX/OpenNiji Hugging Face cache | `/data/shasegawa/adeleine/huggingface` |
| SketchKeras cache | `/data/shasegawa/adeleine/openniji/sketchkeras` |
| SkyTNT mask cache | `/data/shasegawa/adeleine/openniji/reference_masks/skytnt_512` |
| Inherited spatial LoRA | `/data/shasegawa/adeleine/outputs/flux2_klein_openniji_spatial_hint_from_base_holdout1k_512_ddp2/lora` |
| Active run | Path listed above |
| Active log | Path listed above |

Training is local-files-only. Missing cache files will fail model or dataset discovery rather than download replacements.

## Known technical debt and risks

1. The production trainer is named `smoke_train_flux_klein.py`.
2. No structured experiment config is written at launch; `smoke_metrics.txt` is written only after successful completion.
3. Checkpoints omit optimizer, step, and RNG state and are not exact-resume checkpoints.
4. NumPy augmentation RNG is not globally seeded by the trainer.
5. DataLoader uses `num_workers=0` and repeatedly opens Parquet row groups; input performance is not optimized.
6. DDP uses `init_sync=False`; all ranks currently load the same adapter, but this assumption should remain explicit.
7. The trainer and inference ID policy rely on private diffusers APIs and monkey-patching.
8. The holdout is record-level rather than prompt-group-level.
9. The current working tree has no clean committed experiment revision.
10. Automated tests for the end-to-end reference path are limited; most confidence comes from smoke paths and generated artifacts.

## Recommended next actions

1. Let the active run reach the next checkpoint; do not disrupt it for documentation changes.
2. Run different-reference evaluation on step 50k or the newest checkpoint with offset 311.
3. Add launch-time `config.json`, Git diff and revision capture, and environment or package snapshots.
4. Add exact-resume state: optimizer, step, epoch, sampler, Torch RNG, NumPy RNG, and scaler if mixed precision changes.
5. Add a `--start_step` or absolute checkpoint-step argument before attempting continuation in the same run namespace.
6. Freeze a multi-offset reference benchmark to separate checkpoint progress from reference-pair variance.
7. Commit the current reference-conditioning implementation and docs as a reviewed unit after inspecting the dirty tree.

## Completion criteria for this experiment

Before declaring the run successful:

- verify `step_100000` adapter files and final sample sheet;
- run different-image evaluation on at least three fixed offsets;
- compare copy-alarm rate, hint MAE, line metrics, and reference color distance across checkpoints;
- visually inspect every alarm row;
- preserve the launch command, final package versions, holdout manifest, summaries, and representative grids;
- only then select and copy a final adapter into a stable model-release directory.
