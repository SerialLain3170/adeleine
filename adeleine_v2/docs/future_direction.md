# Future Direction

Status: proposed roadmap as of 2026-10-01. This document builds on the current [architecture](architecture.md), [training methodology](training_methodology.md), and [validation protocol](validation.md).

## Product Goal

Adeleine should become a dependable line-art colorization system in which an artist can independently control:

- local colors with dot and line hints;
- character, clothing, and object palettes with foreground references;
- scenery palette, lighting, and atmosphere with background references;
- semantic attributes with text; and
- output variation with seed and condition-strength controls.

The system should preserve the input line structure and composition. References should transfer intended appearance without copying their pose, layout, identity, or incidental background content.

## Current Baseline

The current FLUX.2 Klein LoRA pipeline already combines several useful capabilities:

- line art supplied through native image context;
- fused Atari hints in the output image plane;
- separate foreground and background references derived from SkyTNT masks;
- strongly deformed self-references to discourage spatial copying;
- a mixed task curriculum covering line-only, hint, text, and reference conditions; and
- different-image reference evaluation on a fixed holdout.

The main limitations are experimental reproducibility, benchmark size, imperfect foreground/background intent separation, weakly specified behavior when modalities conflict, square-only training, and dependence on private Diffusers interfaces.

## Guiding Principles

1. Measure controllability separately from visual quality.
2. Treat preservation of the line drawing as a hard requirement.
3. Evaluate foreground and background transfer independently.
4. Require an ablation before accepting architectural complexity.
5. Establish reproducible training and evaluation before scaling data or compute.
6. Keep training, offline validation, and web inference condition semantics identical.

## Priority 0: Reliable Experimentation

### Reproducible launch records

Every run should save a machine-readable manifest containing:

- full arguments and resolved defaults;
- Git revision and a hash or copy of the working-tree diff;
- Python, PyTorch, CUDA, Diffusers, and Transformers versions;
- base-model revision and LoRA initialization source;
- dataset shard list, holdout identity, and split hash; and
- parent checkpoint and global step for resumed runs.

### Exact resume

Checkpoint recovery should restore the LoRA, optimizer, learning-rate scheduler, global step, data-sampler state, and Torch, NumPy, and Python random states. A resumed run must match an uninterrupted control run for a short deterministic test.

### Compatibility boundary

Private Diffusers calls should be isolated behind a small compatibility module with pinned versions and startup assertions. This reduces silent breakage while preserving the current implementation until a stable public API is available.

### Focused automated tests

Add tests for task sampling and dropout probabilities, synchronized reference deformation, mask split thresholds, condition ordering and image limits, spatial time-plane construction, checkpoint resume, and parity between validation and web inference.

## Priority 1: Fixed Control Benchmark

The benchmark should be versioned and immutable within an experiment series. It should include:

- multiple line-extraction methods and line densities;
- sparse and dense dot hints;
- short and long line hints;
- simple and rich-background references;
- foreground-only and background-only references;
- visually similar and deliberately mismatched references;
- text prompts specifying hair, eye, clothing, and background attributes; and
- agreeing and conflicting combinations of hints, references, and text.

Use fixed reference offsets and publish the exact sample identifiers. Report the existing metrics together with:

- foreground and background Lab palette distance, computed separately;
- region-level hint accuracy;
- contour and edge preservation;
- text-attribute agreement;
- reference-content leakage or layout-copy alarms;
- LPIPS or an equivalent perceptual distance; and
- latency and peak memory.

Blind human evaluation should score line preservation, reference fidelity, prompt fidelity, artifact rate, and overall preference as separate questions. A single preference score cannot reveal which control failed.

## Priority 2: Reference Control

### Independent foreground and background controls

Train and evaluate foreground-only, background-only, and joint-reference tasks explicitly. Supplying a character reference must not alter unrelated scenery, and supplying a background reference must not recolor the character.

### Condition strength

Expose reference strength in training and inference. Train with randomized strengths and reference dropout, then verify that measured palette transfer changes monotonically without destabilizing structure.

### Palette-focused representations

Test compact color features such as Lab histograms, dominant colors, or learned palette tokens. These may transfer color more directly while reducing unwanted pose and layout copying.

Semantic embeddings from WD, DINO, or CLIP should only be added after the fixed benchmark exists. Compare at least:

1. current full-image reference;
2. split foreground/background reference;
3. explicit palette features only;
4. semantic features only; and
5. palette and semantic features combined.

### Richer reference policies

Expand training beyond deformed self-reference with sibling images, palette-similar unrelated images, deliberately mismatched images, and foreground/background references drawn from different sources. Each policy needs recorded provenance so results remain interpretable.

### Background transfer objective

Introduce an explicit background palette and lighting objective, evaluated only within background masks. Pair it with copy detection and strong geometric augmentation so rich-background references transfer atmosphere without reproducing their scene layout.

## Priority 3: Hint and Text Fidelity

Generate region-aware Atari hints that cover small semantic areas such as eyes, hair streaks, accessories, and clothing trim. Stratify evaluation by hint length, area, and distance from edges.

Define and train a consistent precedence policy for conflicts:

1. local Atari hint;
2. explicit text attribute;
3. reference appearance; and
4. model prior.

Text supervision should contain explicit attribute statements rather than captions alone. Conflict cases should be first-class benchmark items, not incidental failures.

## Priority 4: Data Quality and Coverage

- Add aspect-ratio buckets so portraits, full-body images, and wide scenes are not forced through square crops.
- Diversify line art across extraction methods, thicknesses, opacity, scan noise, and hand-drawn inputs.
- Audit SkyTNT masks, record confidence, and handle multiple characters and occlusions explicitly.
- Add dedicated flat-color data if animation-cel output is a product target; aesthetic illustrations alone will not guarantee clean region fills.
- Track dataset and license provenance for every shard used in a release candidate.

## Priority 5: Architecture Experiments

Architecture changes should follow benchmark evidence rather than precede it.

1. Replace private time-ID patching with trained condition-type embeddings.
2. Compare native image context with a ControlNet-style or lightweight spatial adapter for line structure and Atari hints.
3. Test a reference adapter that consumes palette and semantic features rather than raw reference tokens alone.
4. Ablate LoRA scope and rank before unfreezing more of the base model.

Each experiment must report quality, controllability, memory, latency, and regression on existing tasks.

## Priority 6: Product and Inference

The inference surface should eventually provide:

- separate foreground and background reference slots;
- enable toggles and strength controls per condition;
- an in-canvas dot and line hint editor;
- crop, mask, and condition previews;
- seed locking and controlled variation;
- import and export of the complete generation configuration;
- embedding caches for unchanged conditions;
- queue progress and cancellation; and
- output metadata sufficient to reproduce an image.

Quantization, CPU offload, batching, and concurrency limits should be evaluated against the same output-quality gates as the full-precision path.

## Milestones

| Milestone | Deliverable | Exit gate |
| --- | --- | --- |
| M0: Reproducible runs | Launch manifest, exact resume, compatibility checks | Resumed short run matches uninterrupted control |
| M1: Frozen benchmark | Versioned cases, metrics, and human-evaluation form | Repeated evaluation is deterministic and attributable |
| M2: Reference separation | Independent foreground/background controls | Target-region transfer improves without measurable cross-region leakage |
| M3: Hint and text fidelity | Region-aware hints and conflict curriculum | Precedence tests pass without line-preservation regression |
| M4: Architecture ablations | Adapter and embedding comparisons | Added complexity beats the baseline on predefined gates |
| M5: Product candidate | Complete inference controls and metadata | Offline and web outputs agree for identical configurations |
| M6: Release candidate | Licensed data record, model card, performance report | Quality, reproducibility, latency, and memory targets are met |

## Immediate Experiment Queue

1. Evaluate the latest checkpoint with three fixed different-image reference offsets, including rich-background cases.
2. Record the exact checkpoint step, launch configuration, dataset split, and reference identities with every result.
3. Add foreground-only and background-only benchmark passes and report region-separated palette metrics.
4. Compare full-image reference conditioning with the current split-reference path.
5. Add an explicit background-palette task and measure both transfer and scene-copy alarms.
6. Train randomized condition strengths and dropout, then test monotonic control.
7. Decide whether semantic or spatial adapters are justified only after these results are available.

## Decision Gates

A change should enter the default pipeline only when it:

- improves its target metric with repeated-seed or statistical support;
- does not materially degrade line preservation;
- does not increase reference identity, pose, or layout copying;
- stays within defined memory and latency budgets;
- behaves consistently in training, offline validation, and web inference; and
- has a reproducible artifact and a documented ablation.

This ordering keeps the next phase focused: first make results repeatable, then make reference transfer measurable, then add model complexity only where the evidence identifies a specific limitation.
