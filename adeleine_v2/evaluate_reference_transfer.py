"""Hold-out evaluation for hint following and reference transfer, with a copy detector.

For each checkpoint and each fixed hold-out example this renders three cases:

- ``hint``: line art + dot Atari hints from the example itself (target is known).
- ``cross_ref``: line art + a reference taken from a *different* hold-out image, prepared exactly as the
  web server does (raw image -> ReferenceConditioningBuilder split). No deformation, no target pixels.
- ``cross_ref_hint``: both.

Metrics per sample (all on 0..255 RGB at ``--image_size``):

- ``hint_mae``: mean abs error between generated and hint colors inside the hint mask.
- ``target_mae``: mean abs error to the ground-truth color image.
- ``line_recall``: fraction of line-art stroke pixels that have a generated edge within ``--edge_tolerance`` px.
- ``edge_precision``: fraction of generated edges lying within ``--edge_tolerance`` px of a line-art stroke.
  Low precision means the model drew structure that is not in the line art (e.g. pasted reference content).
- ``copy_struct_ref`` / ``copy_struct_ref_bg``: structure term of grayscale SSIM (luminance term dropped, so
  transferring the reference's tone/palette does not count) between generated and the full reference / the
  ref_bg condition image. ``copy_struct_baseline`` / ``copy_struct_bg_baseline`` are the same scores for the
  ground-truth target: what an honest colorization of unrelated content scores for this pair.
- ``copy_edge_ref``: fraction of generated edges within tolerance of reference edges (layout copying), with
  ``copy_edge_baseline`` the same fraction for the target's edges.
- ``ref_color_dist``: Bhattacharyya distance between Lab a/b histograms of generated and reference
  foreground (lower = colors transferred).

A sample is flagged ``copy_alarm`` when any copy score exceeds its per-sample baseline by ``--copy_margin``.
Absolute ``edge_precision`` is not used for the alarm: dense line art puts nearly every pixel near a stroke.
Generated images are saved under ``<output_dir>/<checkpoint>/<case>/``; ``--reuse_generated`` recomputes metrics
from them and only loads the model for missing images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from .adapters import batch_to_tensors
from .conditions import ColorizationCondition, ColorizationMode, ModalityDropout, TaskSampler
from .dataset import UnifiedCollator
from .openniji import OpenNijiParquetColorizationDataset, OpenNijiParquetRecord
from .smoke_train_flux_klein import (
    apply_records,
    build_wd_projector,
    build_condition_images,
    build_prompt_text,
    component_dtype,
    configure_atari,
    generate_sample,
    labeled_grid,
    stack_grids,
    tensor_to_pil,
)


CASES = ("hint", "cross_ref", "cross_ref_hint")
METRICS = (
    "hint_mae",
    "target_mae",
    "line_recall",
    "edge_precision",
    "copy_struct_ref",
    "copy_struct_ref_bg",
    "copy_struct_baseline",
    "copy_struct_bg_baseline",
    "copy_edge_ref",
    "copy_edge_baseline",
    "ref_color_dist",
    "copy_alarm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora_dirs", type=Path, nargs="+", required=True, help="One or more LoRA checkpoint directories to compare")
    parser.add_argument("--holdout_manifest", type=Path, required=True, help="holdout_*.jsonl written by smoke_train_flux_klein.py")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--ref_offset", type=int, default=57, help="Reference for example i is hold-out example (i + ref_offset) %% N")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--line_methods", nargs="+", default=["xdog"], help="Deterministic line extraction used for evaluation")
    parser.add_argument("--sketch_root", type=Path)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--inference_steps", type=int, default=12)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--max_condition_images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=43170)
    parser.add_argument("--spatial_hint_mode", choices=["separate", "fused", "fused_masked"], default="fused_masked")
    parser.add_argument("--spatial_condition_id_mode", choices=["default", "hint_to_output", "line_hint_to_output"], default="hint_to_output")
    parser.add_argument("--reference_condition_mode", choices=["full", "foreground", "background", "split", "split_full"], default="split")
    parser.add_argument("--reference_conditioning", choices=["none", "split", "split_tags", "split_wd", "split_wd_tags"], default="split")
    parser.add_argument("--reference_mask_root", type=Path)
    parser.add_argument("--reference_mask_fallback", choices=["skytnt", "grabcut", "ellipse", "whole", "skip"], default="grabcut")
    parser.add_argument("--reference_tag_cache_root", type=Path)
    parser.add_argument("--skytnt_repo", type=Path)
    parser.add_argument("--skytnt_ckpt", type=Path)
    parser.add_argument("--wd_tagger_model", type=Path)
    parser.add_argument("--wd_tagger_labels", type=Path)
    parser.add_argument("--reference_wd_max_refs", type=int, default=2)
    parser.add_argument("--reference_wd_tokens_per_ref", type=int, default=16)
    parser.add_argument("--reference_wd_embed_dim", type=int, default=768)
    parser.add_argument("--edge_tolerance", type=int, default=3)
    parser.add_argument("--copy_margin", type=float, default=0.10)
    parser.add_argument("--cell_size", type=int, default=256)
    parser.add_argument("--reuse_generated", action="store_true", help="Reuse generated images already saved in --output_dir")
    args = parser.parse_args()
    # Names expected by smoke_train_flux_klein.generate_sample.
    args.sample_inference_steps = args.inference_steps
    args.sample_guidance_scale = args.guidance_scale
    args.reference_wd_context = args.reference_conditioning in {"split_wd", "split_wd_tags"}
    args.append_reference_tags = args.reference_conditioning in {"split_tags", "split_wd_tags"}
    return args


# ----------------------------------------------------------------------------- metrics


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)


def _edges(rgb: np.ndarray) -> np.ndarray:
    gray = cv.GaussianBlur(_gray(rgb), (0, 0), 1.0)
    return cv.Canny(gray, 60, 140) > 0


def _line_strokes(lineart_rgb: np.ndarray) -> np.ndarray:
    return _gray(lineart_rgb) < 128


def _near(mask: np.ndarray, tolerance: int) -> np.ndarray:
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    return cv.dilate(mask.astype(np.uint8), kernel) > 0


def _fraction(hit: np.ndarray, total: np.ndarray) -> float:
    n = int(total.sum())
    return float((hit & total).sum()) / n if n else float("nan")


def structure_similarity(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    """Contrast-structure term of SSIM: local correlation of grayscale detail, independent of brightness."""
    a = _gray(a_rgb).astype(np.float64)
    b = _gray(b_rgb).astype(np.float64)
    c2 = (0.03 * 255) ** 2
    blur = lambda x: cv.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    return float(((2 * cov + c2) / (var_a + var_b + c2)).mean())


def ab_hist_distance(a_rgb: np.ndarray, b_rgb: np.ndarray, b_mask: np.ndarray | None = None) -> float:
    def hist(rgb, mask):
        lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB)
        m = None if mask is None else (mask > 127).astype(np.uint8)
        h = cv.calcHist([lab], [1, 2], m, [32, 32], [0, 256, 0, 256])
        return cv.normalize(h, h).astype(np.float32)

    return float(cv.compareHist(hist(a_rgb, None), hist(b_rgb, b_mask), cv.HISTCMP_BHATTACHARYYA))


def compute_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    lineart: np.ndarray,
    hint_rgb: np.ndarray | None,
    hint_mask: np.ndarray | None,
    reference: np.ndarray | None,
    ref_bg: np.ndarray | None,
    ref_mask: np.ndarray | None,
    tolerance: int,
    copy_margin: float,
) -> dict[str, float]:
    nan = float("nan")
    out = {key: nan for key in METRICS}
    out["target_mae"] = float(np.abs(generated.astype(np.float32) - target.astype(np.float32)).mean())
    if hint_rgb is not None and hint_mask is not None:
        m = hint_mask[..., 0] > 127
        if m.any():
            out["hint_mae"] = float(np.abs(generated[m].astype(np.float32) - hint_rgb[m].astype(np.float32)).mean())

    gen_edges = _edges(generated)
    strokes = _line_strokes(lineart)
    out["line_recall"] = _fraction(_near(gen_edges, tolerance), strokes)
    out["edge_precision"] = _fraction(_near(strokes, tolerance), gen_edges)

    if reference is not None:
        ref_edges_near = _near(_edges(reference), tolerance)
        out["copy_struct_ref"] = structure_similarity(generated, reference)
        out["copy_struct_baseline"] = structure_similarity(target, reference)
        out["copy_edge_ref"] = _fraction(ref_edges_near, gen_edges)
        out["copy_edge_baseline"] = _fraction(ref_edges_near, _edges(target))
        out["ref_color_dist"] = ab_hist_distance(generated, reference, ref_mask)
        excess = [out["copy_struct_ref"] - out["copy_struct_baseline"], out["copy_edge_ref"] - out["copy_edge_baseline"]]
        if ref_bg is not None:
            out["copy_struct_ref_bg"] = structure_similarity(generated, ref_bg)
            out["copy_struct_bg_baseline"] = structure_similarity(target, ref_bg)
            excess.append(out["copy_struct_ref_bg"] - out["copy_struct_bg_baseline"])
        out["copy_alarm"] = float(any(np.isfinite(e) and e > copy_margin for e in excess))
    return out


# ----------------------------------------------------------------------------- data


def load_holdout_records(path: Path) -> list[OpenNijiParquetRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "parquet_path" not in obj:
            raise ValueError(f"{path} has no parquet_path fields; only parquet-backed hold-out manifests are supported")
        records.append(
            OpenNijiParquetRecord(
                parquet_path=Path(obj["parquet_path"]),
                row_group=int(obj["row_group"]),
                row_in_group=int(obj["row_in_group"]),
                prompt=obj.get("prompt", ""),
                style=obj.get("style", ""),
                url=obj.get("url", ""),
                group_key=obj.get("group_key", ""),
            )
        )
    return records


def build_dataset(args: argparse.Namespace, records: list[OpenNijiParquetRecord]) -> OpenNijiParquetColorizationDataset:
    parquet_root = records[0].parquet_path.parent.parent
    dataset = OpenNijiParquetColorizationDataset(
        parquet_root=parquet_root,
        hf_home=args.hf_home,
        sketch_root=args.sketch_root,
        image_size=args.image_size,
        max_records=1,
        line_methods=tuple(args.line_methods),
        dropout=ModalityDropout(TaskSampler(weights={"all": 1.0})),
        reference_policy="none",
        reference_conditioning=args.reference_conditioning,
        reference_tag_cache_root=args.reference_tag_cache_root,
        reference_mask_root=args.reference_mask_root,
        reference_mask_fallback=args.reference_mask_fallback,
        skytnt_repo=args.skytnt_repo,
        skytnt_ckpt=args.skytnt_ckpt,
        skytnt_device=args.device if args.device.startswith("cuda") else "cpu",
        wd_tagger_model=args.wd_tagger_model,
        wd_tagger_labels=args.wd_tagger_labels,
    )
    apply_records(dataset, records)
    dataset.lineart.config.morphology_prob = 0.0
    dataset.lineart.config.color_variant_prob = 0.0
    configure_atari(dataset, "dot", 1.0)
    return dataset


def base_condition(dataset: OpenNijiParquetColorizationDataset, index: int, seed: int) -> ColorizationCondition:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return dataset[index]
    finally:
        np.random.set_state(state)


def case_condition(dataset, base: ColorizationCondition, reference: np.ndarray | None, case: str) -> ColorizationCondition:
    keep_hint = case in {"hint", "cross_ref_hint"}
    refs = [reference] if case in {"cross_ref", "cross_ref_hint"} and reference is not None else []
    ref_cond = dataset.reference_conditioner.build(refs)
    return ColorizationCondition(
        lineart=base.lineart,
        target=base.target,
        atari_rgb=base.atari_rgb if keep_hint else None,
        atari_mask=base.atari_mask if keep_hint else None,
        references=refs,
        reference_foregrounds=ref_cond.foregrounds,
        reference_backgrounds=ref_cond.backgrounds,
        reference_masks=ref_cond.masks,
        reference_tags=ref_cond.tags,
        reference_wd_indices=ref_cond.wd_indices,
        reference_wd_scores=ref_cond.wd_scores,
        text="",
        mode=ColorizationMode.RENDER,
        metadata={**base.metadata, "task": case},
    )


# ----------------------------------------------------------------------------- model


def load_pipeline(args: argparse.Namespace):
    from diffusers import Flux2KleinPipeline

    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id, cache_dir=str(args.hf_home / "hub"), torch_dtype=torch.bfloat16, local_files_only=args.local_files_only
    )
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def attach_lora(pipe, base_transformer, lora_dir: Path):
    from peft import PeftModel

    pipe.transformer = PeftModel.from_pretrained(base_transformer, lora_dir)
    pipe.transformer.eval()
    return pipe.transformer


def detach_lora(pipe):
    # Restores the untouched base transformer so the next checkpoint starts clean.
    pipe.transformer = pipe.transformer.unload()
    return pipe.transformer


def load_wd_projector(args: argparse.Namespace, pipe, lora_dir: Path):
    if not args.reference_wd_context:
        return None
    projector = build_wd_projector(
        pipe,
        args.wd_tagger_labels,
        args.reference_wd_embed_dim,
        args.reference_wd_max_refs,
        args.reference_wd_tokens_per_ref,
        args.device,
        component_dtype(pipe.transformer),
        state_dir=lora_dir,
    )
    return projector.eval()


# ----------------------------------------------------------------------------- main


def as_uint8(image) -> np.ndarray:
    return np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.uint8)


def nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_holdout_records(args.holdout_manifest)
    dataset = build_dataset(args, records)
    collator = UnifiedCollator()

    n = len(records)
    count = min(args.examples, n)
    example_indices = [int(round(x)) for x in np.linspace(0, n - 1, count)] if count > 1 else [0]
    bases = {i: base_condition(dataset, i, args.seed + i) for i in example_indices}
    references = {i: base_condition(dataset, (i + args.ref_offset) % n, args.seed + 100_000 + i).target for i in example_indices}

    pipe = None
    base_transformer = None
    per_sample_rows: list[dict] = []
    summary_rows: list[dict] = []
    for lora_dir in args.lora_dirs:
        name = lora_dir.name if lora_dir.name != "lora" else lora_dir.parent.name
        lora_attached = False
        wd_projector = None
        ckpt_dir = args.output_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for case in args.cases:
            grid_rows = []
            case_rows = []
            for i in example_indices:
                base = bases[i]
                reference = references[i]
                condition = case_condition(dataset, base, reference, case)
                batch = collator([condition])
                tensors = batch_to_tensors(batch, device="cpu")
                cond_images, labels, _ = build_condition_images(
                    tensors, False, args.max_condition_images, args.spatial_hint_mode, args.reference_condition_mode
                )
                prompt = build_prompt_text(
                    tensors.text,
                    batch.mode,
                    batch.presence,
                    False,
                    reference_tags=tensors.reference_tags,
                    append_reference_tags=args.append_reference_tags,
                )
                image_path = ckpt_dir / case / f"{i:04d}.png"
                if args.reuse_generated and image_path.exists():
                    generated = Image.open(image_path).convert("RGB")
                else:
                    if pipe is None:
                        pipe = load_pipeline(args)
                        base_transformer = pipe.transformer
                    if not lora_attached:
                        attach_lora(pipe, base_transformer, lora_dir)
                        wd_projector = load_wd_projector(args, pipe, lora_dir)
                        lora_attached = True
                    generated = generate_sample(pipe, tensors, cond_images, labels, prompt, args, args.seed + i, wd_projector)
                    image_path.parent.mkdir(exist_ok=True)
                    generated.save(image_path)
                has_ref = bool(condition.references)
                metrics = compute_metrics(
                    as_uint8(generated),
                    base.target,
                    base.lineart,
                    condition.atari_rgb,
                    condition.atari_mask,
                    reference if has_ref else None,
                    condition.reference_backgrounds[0] if condition.reference_backgrounds else None,
                    condition.reference_masks[0] if condition.reference_masks else None,
                    args.edge_tolerance,
                    args.copy_margin,
                )
                row = {"checkpoint": name, "case": case, "example": i, "reference_example": (i + args.ref_offset) % n if has_ref else "", "conditions": "+".join(labels), **metrics}
                case_rows.append(row)
                items = [("target", Image.fromarray(base.target)), ("lineart", Image.fromarray(base.lineart))]
                if has_ref:
                    items.append(("reference", Image.fromarray(reference)))
                items += [(label, tensor_to_pil(img)) for label, img in zip(labels[1:], cond_images[1:])]
                caption = f"generated struct_ref={metrics['copy_struct_ref'] - metrics['copy_struct_baseline']:+.2f} edge_ref={metrics['copy_edge_ref'] - metrics['copy_edge_baseline']:+.2f}"
                if metrics["copy_alarm"] == 1.0:
                    caption += " COPY"
                items.append((caption, generated))
                grid_rows.append(labeled_grid(items, cell=args.cell_size))
            stack_grids(grid_rows).save(ckpt_dir / f"{case}.png")
            per_sample_rows.extend(case_rows)
            summary = {"checkpoint": name, "case": case, "n": len(case_rows)}
            summary.update({key: nanmean([r[key] for r in case_rows]) for key in METRICS})
            summary_rows.append(summary)
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()), flush=True)
        if lora_attached:
            detach_lora(pipe)

    write_tsv(args.output_dir / "per_sample.tsv", per_sample_rows)
    write_tsv(args.output_dir / "summary.tsv", summary_rows)
    (args.output_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    print(args.output_dir / "summary.tsv")


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.5f}" if isinstance(v, float) else v for k, v in row.items()})


if __name__ == "__main__":
    main()
