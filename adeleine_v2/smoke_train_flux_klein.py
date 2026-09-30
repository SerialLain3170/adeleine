from __future__ import annotations

import argparse
import copy
from datetime import timedelta
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any

import csv
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image, ImageDraw
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .adapters import batch_to_tensors
from .atari import AtariHintConfig, AtariHintGenerator
from .conditions import ModalityDropout, TaskSampler
from .dataset import UnifiedCollator
from .flux_klein import FluxKleinColorizer, FluxKleinConfig
from .openniji import OpenNijiParquetColorizationDataset


class WDReferenceTokenProjector(torch.nn.Module):
    """Projects frozen WD tagger score tokens into FLUX text-context tokens."""

    def __init__(self, num_tags: int, output_dim: int, embed_dim: int = 768, max_refs: int = 4, max_tokens_per_ref: int = 16):
        super().__init__()
        self.max_refs = max_refs
        self.max_tokens_per_ref = max_tokens_per_ref
        self.tag_embed = torch.nn.Embedding(max(num_tags, 1), embed_dim)
        self.ref_type_embed = torch.nn.Embedding(max_refs, embed_dim)
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, output_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(output_dim, output_dim),
        )

    def forward(self, indices_by_batch: list[list[torch.Tensor]], scores_by_batch: list[list[torch.Tensor]], batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        rows: list[torch.Tensor] = []
        max_len = 0
        for batch_index in range(batch_size):
            ref_indices = indices_by_batch[batch_index] if batch_index < len(indices_by_batch) else []
            ref_scores = scores_by_batch[batch_index] if batch_index < len(scores_by_batch) else []
            tokens: list[torch.Tensor] = []
            for ref_index, (indices, scores) in enumerate(zip(ref_indices[: self.max_refs], ref_scores[: self.max_refs])):
                if indices.numel() == 0 or scores.numel() == 0:
                    continue
                take = min(self.max_tokens_per_ref, int(indices.numel()), int(scores.numel()))
                idx = indices[:take].to(device=device, dtype=torch.long).clamp(min=0, max=self.tag_embed.num_embeddings - 1)
                weight = scores[:take].to(device=device, dtype=dtype).clamp(0.0, 1.0).unsqueeze(-1)
                emb = self.tag_embed(idx).to(dtype=dtype) * weight
                emb = emb + self.ref_type_embed.weight[ref_index].to(device=device, dtype=dtype).unsqueeze(0)
                tokens.append(self.proj(emb))
            if tokens:
                row = torch.cat(tokens, dim=0)
            else:
                row = torch.empty((0, self.proj[-1].out_features), device=device, dtype=dtype)
            rows.append(row)
            max_len = max(max_len, row.shape[0])
        if max_len == 0:
            return torch.empty((batch_size, 0, self.proj[-1].out_features), device=device, dtype=dtype)
        padded = []
        for row in rows:
            if row.shape[0] < max_len:
                pad = torch.zeros((max_len - row.shape[0], row.shape[1]), device=device, dtype=dtype)
                row = torch.cat([row, pad], dim=0)
            padded.append(row)
        return torch.stack(padded, dim=0)


WD_PROJECTOR_FILE = "wd_reference_projector.pt"


def build_wd_projector(
    pipe,
    labels_path: Path | None,
    embed_dim: int,
    max_refs: int,
    tokens_per_ref: int,
    device: torch.device | str,
    dtype: torch.dtype,
    state_dir: Path | None = None,
) -> "WDReferenceTokenProjector":
    """Create the WD context projector sized to the text encoder, restoring weights from state_dir if present."""
    with torch.no_grad():
        probe_embeds, _ = pipe.encode_prompt(prompt=[""], device=component_device(pipe.text_encoder), max_sequence_length=16)
    projector = WDReferenceTokenProjector(
        num_tags=count_wd_labels(labels_path),
        output_dim=int(probe_embeds.shape[-1]),
        embed_dim=embed_dim,
        max_refs=max_refs,
        max_tokens_per_ref=tokens_per_ref,
    )
    state_path = state_dir / WD_PROJECTOR_FILE if state_dir is not None else None
    if state_path is not None and state_path.exists():
        projector.load_state_dict(torch.load(state_path, map_location="cpu"))
        print(f"loaded WD projector from {state_path}", flush=True)
    elif state_dir is not None:
        print(f"warning: {state_path} not found; WD projector starts from random init", flush=True)
    return projector.to(device=device, dtype=dtype)


def count_wd_labels(path: Path | None) -> int:
    if path is None or not path.exists():
        return 1
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            return sum(1 for _ in reader)
        f.seek(0)
        return sum(1 for line in f if line.strip())


def append_prompt_context(prompt_embeds: torch.Tensor, text_ids: torch.Tensor, extra_tokens: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if extra_tokens is None or extra_tokens.numel() == 0:
        return prompt_embeds, text_ids
    extra_tokens = extra_tokens.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    prompt_embeds = torch.cat([prompt_embeds, extra_tokens], dim=1)
    if text_ids.ndim == 2:
        extra_ids = torch.zeros((extra_tokens.shape[1], text_ids.shape[-1]), device=text_ids.device, dtype=text_ids.dtype)
        text_ids = torch.cat([text_ids, extra_ids], dim=0)
    elif text_ids.ndim == 3:
        extra_ids = torch.zeros((text_ids.shape[0], extra_tokens.shape[1], text_ids.shape[-1]), device=text_ids.device, dtype=text_ids.dtype)
        text_ids = torch.cat([text_ids, extra_ids], dim=1)
    return prompt_embeds, text_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-train FLUX.2 Klein LoRA on the first OpenNiji shard")
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--openniji_repo_id", default="ShoukanLabs/OpenNiji-0_32237")
    parser.add_argument("--openniji_parquet_root", type=Path)
    parser.add_argument("--openniji_parquet_pattern", default="data/train-00000*.parquet", help="Default reads only the first shard per repo (smoke tests); use 'data/*.parquet' for full training")
    parser.add_argument("--sketch_root", type=Path)
    parser.add_argument("--digital_root", type=Path)
    parser.add_argument("--anime_line_root", type=Path)
    parser.add_argument("--reference_policy", choices=["self", "deformed_self", "self_deformed", "sibling", "mixed", "none"], default="deformed_self")
    parser.add_argument("--reference_deform_strength", choices=["mild", "strong"], default="mild", help="Geometric deformation of self references; 'strong' adds flips, larger affine range and an elastic warp")
    parser.add_argument("--reference_background_source", choices=["self", "other"], default="self", help="ref_bg for self references: background split from the deformed reference, or from a random other record")
    parser.add_argument("--reference_conditioning", choices=["none", "split", "split_tags", "split_wd", "split_wd_tags"], default="none", help="Build foreground/background reference layers plus optional WD semantic tokens/tags in the dataset")
    parser.add_argument("--reference_condition_mode", choices=["full", "foreground", "background", "split", "split_full"], default="full", help="Which reference condition images are passed to FLUX")
    parser.add_argument("--reference_tag_cache_root", type=Path, help="Directory for cached reference tag JSON files")
    parser.add_argument("--reference_mask_root", type=Path, help="Directory of digest-named foreground mask PNG files for reference images")
    parser.add_argument("--reference_mask_fallback", choices=["skytnt", "grabcut", "ellipse", "whole", "skip"], default="grabcut", help="Fallback when a reference foreground mask is missing")
    parser.add_argument("--skytnt_repo", type=Path, help="Local clone of SkyTNT/anime-segmentation")
    parser.add_argument("--skytnt_model_id", default="skytnt/anime-seg")
    parser.add_argument("--skytnt_ckpt", type=Path)
    parser.add_argument("--skytnt_net", default="isnet_is")
    parser.add_argument("--skytnt_image_size", type=int, default=1024)
    parser.add_argument("--skytnt_device", help="Device for on-demand SkyTNT masks; defaults to this rank's --device")
    parser.add_argument("--skytnt_fp32", action="store_true")
    parser.add_argument("--skytnt_local_files_only", action="store_true")
    parser.add_argument("--no_reference_cache_generated_masks", action="store_true")
    parser.add_argument("--wd_tagger_model", type=Path, help="Optional ONNX WD tagger model path for reference anime tags")
    parser.add_argument("--wd_tagger_labels", type=Path, help="Optional WD tagger CSV label path")
    parser.add_argument("--wd_tagger_threshold", type=float, default=0.35)
    parser.add_argument("--wd_tagger_character_threshold", type=float, default=0.85)
    parser.add_argument("--wd_tagger_max_tokens", type=int, default=32)
    parser.add_argument("--reference_wd_context", action="store_true", help="Project cached WD tagger score tokens into extra FLUX text-context tokens")
    parser.add_argument("--reference_wd_max_refs", type=int, default=2)
    parser.add_argument("--reference_wd_tokens_per_ref", type=int, default=16)
    parser.add_argument("--reference_wd_embed_dim", type=int, default=768)
    parser.add_argument("--reference_tag_max", type=int, default=24)
    parser.add_argument("--append_reference_tags", action="store_true", help="Append reference tags to the FLUX text prompt when reference conditioning is active")
    parser.add_argument("--reference_tag_prefix", default="reference attributes")
    parser.add_argument("--output_dir", type=Path, default=Path("/data/shasegawa/adeleine/outputs/flux2_klein_smoke_lora"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3170)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--save_lora", action="store_true")
    parser.add_argument("--resume_lora", type=Path, help="Optional existing LoRA adapter directory to continue training from")
    parser.add_argument("--full_conditions", action="store_true", help="Force line+Atari+reference+text conditions and pass them to FLUX.2 image/text conditioning")
    parser.add_argument("--max_condition_images", type=int, default=6, help="Maximum FLUX.2 image conditions per sample, including lineart/Atari/mask/references")
    parser.add_argument(
        "--spatial_hint_mode",
        choices=["separate", "fused", "fused_masked"],
        default="separate",
        help=(
            "How Atari hints are passed to FLUX image conditioning. "
            "'separate' keeps legacy line/atari/mask images; 'fused' overlays hint colors onto the line-art plane; "
            "'fused_masked' also keeps the binary mask image."
        ),
    )
    parser.add_argument(
        "--spatial_condition_id_mode",
        choices=["default", "hint_to_output", "line_hint_to_output"],
        default="default",
        help=(
            "Optionally rewrite condition image positional T-coordinates. "
            "Use hint_to_output with fused/fused_masked Atari hints to place spatial Atari tokens on the output latent plane."
        ),
    )
    parser.add_argument("--sample_every", type=int, default=0, help="Save a colorization sample every N optimizer steps; 0 disables sampling")
    parser.add_argument("--checkpoint_every", type=int, default=0, help="Save LoRA checkpoints every N optimizer steps; 0 disables interval checkpointing")
    parser.add_argument("--sample_inference_steps", type=int, default=8)
    parser.add_argument("--sample_guidance_scale", type=float, default=3.0)
    parser.add_argument("--sample_seed", type=int, default=43170)
    parser.add_argument("--sample_cell_size", type=int, default=512, help="Cell size used in saved validation contact sheets")
    parser.add_argument("--validation_examples", type=int, default=4, help="Number of fixed validation examples to render when saving samples")
    parser.add_argument("--validation_task", choices=["line", "line_atari", "line_reference", "line_text", "line_reference_atari", "all"], default="line_atari")
    parser.add_argument("--holdout_records", type=int, default=0, help="Number of records to reserve from training for validation/test")
    parser.add_argument("--holdout_manifest", type=Path, help="Where to save held-out record metadata as JSONL")
    parser.add_argument("--task_weights", default="line=0.10,line_atari=0.45,line_reference=0.15,line_text=0.10,line_reference_atari=0.15,all=0.05")
    parser.add_argument("--atari_mode", choices=["line", "dot", "mixed"], default="mixed")
    parser.add_argument("--atari_dot_prob", type=float, default=0.75, help="Probability of dot hints when --atari_mode=mixed")
    parser.add_argument("--validation_atari_mode", choices=["line", "dot", "mixed"], default="dot")
    return parser.parse_args()


def freeze(module: Any) -> None:
    if module is None:
        return
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def component_device(module: Any) -> torch.device:
    return next(module.parameters()).device


def component_dtype(module: Any) -> torch.dtype:
    return next(module.parameters()).dtype


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training currently requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Rank 0 renders validation samples while the other ranks wait at a barrier; allow for slow sampling.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}"
    return True, rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()



def debug_log(rank: int, message: str) -> None:
    print(f"[rank{rank}] {message}", flush=True)


def build_spatial_hint_image(lineart: torch.Tensor, atari_rgb: torch.Tensor, atari_mask: torch.Tensor) -> torch.Tensor:
    mask = atari_mask.clamp(0.0, 1.0).to(dtype=lineart.dtype)
    return lineart * (1.0 - mask) + atari_rgb * mask


def _append_reference_images(
    images: list[torch.Tensor],
    labels: list[str],
    refs: list[torch.Tensor],
    label: str,
    max_condition_images: int,
) -> None:
    for ref in refs:
        if len(images) >= max_condition_images:
            return
        images.append(ref.unsqueeze(0))
        labels.append(label)


def build_condition_images(
    tensors,
    full_conditions: bool,
    max_condition_images: int,
    spatial_hint_mode: str = "separate",
    reference_condition_mode: str = "full",
) -> tuple[list[torch.Tensor], list[str], str]:
    images: list[torch.Tensor] = [tensors.lineart]
    labels = ["line"]
    if tensors.atari_rgb is not None:
        if spatial_hint_mode in {"fused", "fused_masked"} and tensors.atari_mask is not None:
            images.append(build_spatial_hint_image(tensors.lineart, tensors.atari_rgb, tensors.atari_mask))
            labels.append("spatial_atari")
        else:
            images.append(tensors.atari_rgb)
            labels.append("atari")
    if tensors.atari_mask is not None and spatial_hint_mode in {"separate", "fused_masked"}:
        mask_rgb = tensors.atari_mask.repeat(1, 3, 1, 1) * 2.0 - 1.0
        images.append(mask_rgb)
        labels.append("mask")

    full_refs = tensors.reference_images[0]
    fg_refs = tensors.reference_foregrounds[0] if tensors.reference_foregrounds else []
    bg_refs = tensors.reference_backgrounds[0] if tensors.reference_backgrounds else []
    if reference_condition_mode == "foreground":
        _append_reference_images(images, labels, fg_refs or full_refs, "ref_fg", max_condition_images)
    elif reference_condition_mode == "background":
        _append_reference_images(images, labels, bg_refs or full_refs, "ref_bg", max_condition_images)
    elif reference_condition_mode in {"split", "split_full"}:
        # A reference may have only one layer (no character found, or all character); use the full image
        # only when it has neither (no mask available).
        if fg_refs or bg_refs:
            _append_reference_images(images, labels, fg_refs, "ref_fg", max_condition_images)
            _append_reference_images(images, labels, bg_refs, "ref_bg", max_condition_images)
            if reference_condition_mode == "split_full":
                _append_reference_images(images, labels, full_refs, "ref", max_condition_images)
        else:
            _append_reference_images(images, labels, full_refs, "ref", max_condition_images)
    else:
        _append_reference_images(images, labels, full_refs, "ref", max_condition_images)

    images = images[:max_condition_images]
    labels = labels[:max_condition_images]
    return images, labels, "+".join(labels)


_warned_spatial_ids: set[str] = set()


def align_spatial_condition_ids(
    image_latent_ids: torch.Tensor,
    condition_labels: list[str],
    spatial_condition_id_mode: str = "default",
) -> torch.Tensor:
    if spatial_condition_id_mode == "default" or not condition_labels:
        return image_latent_ids
    tokens_per_image, remainder = divmod(image_latent_ids.shape[1], len(condition_labels))
    if remainder != 0:
        if _warned_spatial_ids:
            return image_latent_ids
        _warned_spatial_ids.add(spatial_condition_id_mode)
        print(
            f"warning: {spatial_condition_id_mode} skipped: {image_latent_ids.shape[1]} condition tokens are not divisible "
            f"by {len(condition_labels)} images ({'+'.join(condition_labels)}); condition images must share one size",
            flush=True,
        )
        return image_latent_ids
    if spatial_condition_id_mode == "hint_to_output":
        aligned_labels = {"spatial_atari"}
    elif spatial_condition_id_mode == "line_hint_to_output":
        aligned_labels = {"line", "spatial_atari", "atari"}
    else:
        return image_latent_ids
    out = image_latent_ids.clone()
    for index, label in enumerate(condition_labels):
        if label not in aligned_labels:
            continue
        start = index * tokens_per_image
        end = start + tokens_per_image
        out[:, start:end, 0] = 0
    return out


@contextmanager
def condition_id_policy(pipe, condition_labels: list[str], spatial_condition_id_mode: str = "default"):
    if spatial_condition_id_mode == "default":
        yield
        return
    original = pipe._prepare_image_ids

    def patched_prepare_image_ids(image_latents, scale: int = 10):
        ids = original(image_latents, scale=scale)
        return align_spatial_condition_ids(ids, condition_labels, spatial_condition_id_mode)

    pipe._prepare_image_ids = patched_prepare_image_ids
    try:
        yield
    finally:
        pipe._prepare_image_ids = original




def should_save_interval(step: int, total_steps: int, interval: int) -> bool:
    if interval <= 0:
        return False
    current = step + 1
    return current == 1 or current % interval == 0 or current == total_steps


def save_lora_checkpoint(colorizer: FluxKleinColorizer, args: argparse.Namespace, step: int, wd_projector: torch.nn.Module | None = None) -> Path:
    checkpoint_dir = args.output_dir / "checkpoints" / f"step_{step + 1:06d}"
    colorizer.save_lora(checkpoint_dir)
    if wd_projector is not None:
        module = wd_projector.module if hasattr(wd_projector, "module") else wd_projector
        torch.save(module.state_dict(), checkpoint_dir / WD_PROJECTOR_FILE)
    return checkpoint_dir

def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    image = image.detach().float().cpu().clamp(-1, 1)
    array = ((image.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def labeled_grid(items: list[tuple[str, Image.Image]], cell: int = 512, label_h: int = 28) -> Image.Image:
    sheet = Image.new("RGB", (cell * len(items), cell + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        draw.text((index * cell + 6, 5), label, fill=(0, 0, 0))
        sheet.paste(image.resize((cell, cell), Image.Resampling.LANCZOS), (index * cell, label_h))
    return sheet


def stack_grids(rows: list[Image.Image]) -> Image.Image:
    if not rows:
        return Image.new("RGB", (1, 1), "white")
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


def generate_sample(
    pipe,
    tensors,
    condition_images_cpu,
    condition_labels: list[str],
    prompt_text: list[str],
    args: argparse.Namespace,
    seed: int,
    wd_projector: torch.nn.Module | None = None,
) -> Image.Image:
    generator = torch.Generator(device=args.device).manual_seed(seed)
    condition_pils = [tensor_to_pil(img) for img in condition_images_cpu]
    spatial_condition_id_mode = getattr(args, "spatial_condition_id_mode", "default")
    prompt_kwargs = {"prompt": prompt_text[0]}
    if wd_projector is not None and getattr(args, "reference_wd_context", False):
        text_device = component_device(pipe.text_encoder)
        transformer_device = component_device(pipe.transformer)
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt_text,
                device=text_device,
                max_sequence_length=args.max_sequence_length,
            )
            prompt_embeds = prompt_embeds.to(device=transformer_device, dtype=component_dtype(pipe.transformer))
            extra = wd_projector(
                tensors.reference_wd_indices,
                tensors.reference_wd_scores,
                batch_size=1,
                device=transformer_device,
                dtype=prompt_embeds.dtype,
            )
            prompt_embeds, _ = append_prompt_context(prompt_embeds, text_ids.to(transformer_device), extra)
        prompt_kwargs = {"prompt_embeds": prompt_embeds}
    with torch.inference_mode(), condition_id_policy(pipe, condition_labels, spatial_condition_id_mode):
        result = pipe(
            image=condition_pils,
            **prompt_kwargs,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.sample_inference_steps,
            guidance_scale=args.sample_guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        ).images[0]
    return result.convert("RGB")


def sample_grid_row(
    pipe,
    tensors,
    condition_images_cpu,
    condition_labels: list[str],
    prompt_text: list[str],
    args: argparse.Namespace,
    seed: int,
    row_index: int,
    wd_projector: torch.nn.Module | None = None,
) -> Image.Image:
    generated = generate_sample(pipe, tensors, condition_images_cpu, condition_labels, prompt_text, args, seed, wd_projector)
    items = [(f"target {row_index}", tensor_to_pil(tensors.target)), ("lineart", tensor_to_pil(tensors.lineart))]
    for label, image in zip(condition_labels[1:], condition_images_cpu[1:]):
        items.append((label, tensor_to_pil(image)))
    items.append(("generated", generated))
    return labeled_grid(items, cell=args.sample_cell_size)


def build_validation_indices(dataset_len: int, count: int) -> list[int]:
    count = max(0, min(count, dataset_len))
    if count == 0:
        return []
    if count == 1:
        return [0]
    return [int(round(x)) for x in np.linspace(0, dataset_len - 1, count)]


def parse_task_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, raw = item.split("=", 1)
        weights[key.strip()] = float(raw)
    required = {"line", "line_atari", "line_reference", "line_text", "line_reference_atari", "all"}
    missing = required - set(weights)
    if missing:
        raise ValueError(f"Missing task weights: {sorted(missing)}")
    return weights


def configure_atari(dataset, mode: str, dot_prob: float) -> None:
    dataset.atari = AtariHintGenerator(AtariHintConfig(mode=mode, mixed_dot_prob=float(np.clip(dot_prob, 0.0, 1.0))))


def split_holdout_records(records: list, holdout_records: int, seed: int) -> tuple[list, list, list[int], list[int]]:
    if holdout_records <= 0:
        return records, [], list(range(len(records))), []
    holdout_records = min(holdout_records, max(0, len(records) - 1))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    holdout_set = set(int(x) for x in indices[:holdout_records])
    train_indices = [i for i in range(len(records)) if i not in holdout_set]
    holdout_indices = [i for i in range(len(records)) if i in holdout_set]
    return [records[i] for i in train_indices], [records[i] for i in holdout_indices], train_indices, holdout_indices


def apply_records(dataset, records: list) -> None:
    dataset.records = records
    dataset.groups = dataset._build_groups(records)


def record_to_json(index: int, record) -> str:
    payload = {"index": index}
    for name in ["url", "prompt", "style", "group_key", "parquet_path", "row_group", "row_in_group"]:
        value = getattr(record, name, None)
        if value is not None:
            payload[name] = str(value)
    return json.dumps(payload, ensure_ascii=False)


def make_validation_dataset(dataset, holdout_records: list, args: argparse.Namespace):
    validation_dataset = copy.copy(dataset)
    apply_records(validation_dataset, holdout_records if holdout_records else dataset.records)
    validation_dataset.dropout = ModalityDropout(TaskSampler(weights={args.validation_task: 1.0}))
    configure_atari(validation_dataset, args.validation_atari_mode, args.atari_dot_prob)
    return validation_dataset


def save_validation_samples(pipe, dataset, collator: UnifiedCollator, validation_indices: list[int], args: argparse.Namespace, step: int, wd_projector: torch.nn.Module | None = None) -> None:
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    was_training = pipe.transformer.training
    pipe.transformer.eval()
    rows = []
    for row_index, dataset_index in enumerate(validation_indices):
        condition = dataset[dataset_index]
        batch = collator([condition])
        tensors = batch_to_tensors(batch, device="cpu")
        condition_images_cpu, condition_labels, _ = build_condition_images(tensors, args.full_conditions, args.max_condition_images, args.spatial_hint_mode, args.reference_condition_mode)
        prompt_text = build_prompt_text(
            tensors.text,
            batch.mode,
            batch.presence,
            args.full_conditions,
            reference_tags=tensors.reference_tags,
            append_reference_tags=args.append_reference_tags,
            reference_tag_prefix=args.reference_tag_prefix,
        )
        rows.append(sample_grid_row(pipe, tensors, condition_images_cpu, condition_labels, prompt_text, args, args.sample_seed + step * 1000 + row_index, row_index, wd_projector))
    stack_grids(rows).save(sample_dir / f"step_{step + 1:06d}.png")
    if was_training:
        pipe.transformer.train()


def _presence_value(presence: dict, key: str, index: int) -> bool:
    value = presence.get(key)
    if value is None:
        return False
    return bool(value[index])


def build_prompt_text(
    texts: list[str],
    modes: list[str],
    presence: dict,
    full_conditions: bool,
    reference_tags: list[list[str]] | None = None,
    append_reference_tags: bool = False,
    reference_tag_prefix: str = "reference attributes",
) -> list[str]:
    out = []
    for index, text in enumerate(texts):
        has_text = full_conditions or _presence_value(presence, "text", index)
        has_reference = full_conditions or _presence_value(presence, "reference", index)
        tags = []
        if append_reference_tags and has_reference and reference_tags is not None and index < len(reference_tags):
            tags = [tag for tag in reference_tags[index] if tag]
        if not has_text and not tags:
            out.append("")
            continue
        base = text if has_text and text else "anime illustration colorization"
        active = []
        for key in ["lineart", "atari", "reference", "reference_foreground", "reference_background", "reference_tags", "text", "flat", "diverse"]:
            if full_conditions or _presence_value(presence, key, index):
                active.append(key)
        parts = [base]
        if tags:
            parts.append(f"{reference_tag_prefix}: {', '.join(tags)}")
        parts.append(f"mode: {modes[index]}; conditions: {', '.join(active)}")
        out.append("; ".join(part for part in parts if part))
    return out

def main() -> None:
    args = parse_args()
    if args.reference_conditioning == "split_tags":
        args.append_reference_tags = True
    if args.reference_conditioning in {"split_wd", "split_wd_tags"}:
        args.reference_wd_context = True
    if args.reference_conditioning == "split_wd_tags":
        args.append_reference_tags = True
    distributed, rank, local_rank, world_size = setup_distributed(args)
    main_process = is_main_process(rank)
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    torch.manual_seed(args.seed)

    task_weights = {"all": 1.0} if args.full_conditions else parse_task_weights(args.task_weights)
    dropout = ModalityDropout(TaskSampler(weights=task_weights))
    dataset = OpenNijiParquetColorizationDataset(
        parquet_root=args.openniji_parquet_root,
        repo_id=args.openniji_repo_id,
        hf_home=args.hf_home,
        parquet_pattern=args.openniji_parquet_pattern,
        sketch_root=args.sketch_root,
        digital_root=args.digital_root,
        anime_line_root=args.anime_line_root,
        image_size=args.image_size,
        max_records=args.max_records,
        dropout=dropout,
        reference_policy=args.reference_policy,
        reference_deform_strength=args.reference_deform_strength,
        reference_background_source=args.reference_background_source,
        reference_conditioning=args.reference_conditioning,
        reference_tag_cache_root=args.reference_tag_cache_root,
        reference_mask_root=args.reference_mask_root,
        reference_mask_fallback=args.reference_mask_fallback,
        skytnt_repo=args.skytnt_repo,
        skytnt_model_id=args.skytnt_model_id,
        skytnt_ckpt=args.skytnt_ckpt,
        skytnt_net=args.skytnt_net,
        skytnt_image_size=args.skytnt_image_size,
        skytnt_device=args.skytnt_device or args.device,
        skytnt_fp32=args.skytnt_fp32,
        skytnt_local_files_only=args.skytnt_local_files_only,
        reference_cache_generated_masks=not args.no_reference_cache_generated_masks,
        wd_tagger_model=args.wd_tagger_model,
        wd_tagger_labels=args.wd_tagger_labels,
        wd_tagger_threshold=args.wd_tagger_threshold,
        wd_tagger_character_threshold=args.wd_tagger_character_threshold,
        wd_tagger_max_tokens=args.wd_tagger_max_tokens,
        reference_tag_max=args.reference_tag_max,
    )
    if main_process:
        print(
            f"dataset: repo_id={args.openniji_repo_id} parquet_pattern={args.openniji_parquet_pattern} "
            f"parquet_files={len(dataset.parquet_paths)} records={len(dataset.records)}",
            flush=True,
        )
    configure_atari(dataset, args.atari_mode, args.atari_dot_prob)
    train_records, holdout_records, train_record_indices, holdout_record_indices = split_holdout_records(dataset.records, args.holdout_records, args.seed)
    apply_records(dataset, train_records)
    validation_dataset = make_validation_dataset(dataset, holdout_records, args)
    if args.holdout_manifest is not None and holdout_records and main_process:
        args.holdout_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.holdout_manifest.write_text("\n".join(record_to_json(i, r) for i, r in zip(holdout_record_indices, holdout_records)) + "\n")
    collator = UnifiedCollator()
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if distributed else None
    loader = DataLoader(dataset, batch_size=1, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=0, collate_fn=collator, drop_last=True)
    validation_indices = build_validation_indices(len(validation_dataset), args.validation_examples)

    # Load directly so we can pass cache/offload options without changing the generic wrapper API.
    from diffusers import Flux2KleinPipeline

    dtype = torch.bfloat16
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id,
        cache_dir=str(args.hf_home / "hub"),
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    debug_log(rank, f"pipeline_loaded moving_to_device={args.device}")
    pipe.to(args.device)
    debug_log(rank, "pipeline_on_device")
    pipe.set_progress_bar_config(disable=True)

    debug_log(rank, "freezing_vae")
    freeze(pipe.vae)
    debug_log(rank, "freezing_text_encoder")
    freeze(pipe.text_encoder)
    debug_log(rank, "freezing_transformer")
    freeze(pipe.transformer)
    debug_log(rank, "frozen_all")

    wd_projector: torch.nn.Module | None = None
    forward_wd_projector: torch.nn.Module | None = None

    colorizer = FluxKleinColorizer(
        FluxKleinConfig(
            model_id=args.model_id,
            torch_dtype="bfloat16",
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_rank,
            device=args.device,
        )
    )
    colorizer.pipeline = pipe
    colorizer.transformer = pipe.transformer
    if args.resume_lora is not None:
        debug_log(rank, f"loading_lora resume={args.resume_lora}")
        from peft import PeftModel

        transformer = PeftModel.from_pretrained(pipe.transformer, args.resume_lora, is_trainable=True)
        pipe.transformer = transformer
        colorizer.transformer = transformer
        debug_log(rank, "loaded_lora")
    else:
        debug_log(rank, "prepare_lora_start")
        transformer = colorizer.prepare_lora()
        debug_log(rank, "prepare_lora_done")
    transformer.train()
    debug_log(rank, "transformer_train_mode")

    if args.reference_wd_context:
        wd_projector = build_wd_projector(
            pipe,
            args.wd_tagger_labels,
            args.reference_wd_embed_dim,
            args.reference_wd_max_refs,
            args.reference_wd_tokens_per_ref,
            args.device,
            component_dtype(transformer),
            state_dir=args.resume_lora,
        )
        wd_projector.train()
        debug_log(rank, "wd_projector_ready")
        forward_wd_projector = (
            DDP(
                wd_projector,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
            if distributed
            else wd_projector
        )

    debug_log(rank, "count_trainable_start")
    trainable, total = colorizer.trainable_parameter_count(transformer)
    trainable_tensors = sum(1 for p in transformer.parameters() if p.requires_grad)
    print(f"[rank{rank}] lora_ready trainable_tensors={trainable_tensors} trainable_parameters={trainable}", flush=True)
    debug_log(rank, "ddp_wrap_start")
    forward_transformer = (
        DDP(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            init_sync=False,
        )
        if distributed
        else transformer
    )
    debug_log(rank, "ddp_wrap_done")

    optim_params = [p for p in transformer.parameters() if p.requires_grad]
    if wd_projector is not None:
        optim_params.extend(p for p in wd_projector.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr)

    device = component_device(transformer)
    vae_device = component_device(pipe.vae)
    text_device = component_device(pipe.text_encoder)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    losses = []
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    iterator = iter(loader)
    total_steps = len(loader) if args.steps <= 0 else args.steps
    pbar = tqdm(range(total_steps), desc="flux2-smoke-train", disable=not main_process)
    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)

        tensors = batch_to_tensors(batch, device="cpu")
        target = tensors.target.to(device=vae_device, dtype=component_dtype(pipe.vae))
        condition_images_cpu, condition_labels, condition_summary = build_condition_images(tensors, args.full_conditions, args.max_condition_images, args.spatial_hint_mode, args.reference_condition_mode)
        if bool(batch.presence.get("text", [False])[0]):
            condition_summary = f"{condition_summary}+text"
        condition_images = [img.to(device=vae_device, dtype=component_dtype(pipe.vae)) for img in condition_images_cpu]
        prompt_text = build_prompt_text(
            tensors.text,
            batch.mode,
            batch.presence,
            args.full_conditions,
            reference_tags=tensors.reference_tags,
            append_reference_tags=args.append_reference_tags,
            reference_tag_prefix=args.reference_tag_prefix,
        )

        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt_text,
                device=text_device,
                max_sequence_length=args.max_sequence_length,
            )
            prompt_embeds = prompt_embeds.to(device=device, dtype=component_dtype(transformer))
            text_ids = text_ids.to(device=device)

            target_latents = pipe._encode_vae_image(target, generator=None)
            target_latents = target_latents.to(device=device, dtype=component_dtype(transformer))
            latent_ids = pipe._prepare_latent_ids(target_latents).to(device=device)
            clean = pipe._pack_latents(target_latents)

            image_latents, image_latent_ids = pipe.prepare_image_latents(
                images=condition_images,
                batch_size=1,
                generator=None,
                device=device,
                dtype=component_dtype(pipe.vae),
            )
            image_latents = image_latents.to(device=device, dtype=component_dtype(transformer))
            image_latent_ids = image_latent_ids.to(device=device)
            image_latent_ids = align_spatial_condition_ids(image_latent_ids, condition_labels, args.spatial_condition_id_mode)

        if forward_wd_projector is not None:
            wd_tokens = forward_wd_projector(
                tensors.reference_wd_indices,
                tensors.reference_wd_scores,
                batch_size=prompt_embeds.shape[0],
                device=device,
                dtype=prompt_embeds.dtype,
            )
            prompt_embeds, text_ids = append_prompt_context(prompt_embeds, text_ids, wd_tokens)
            if wd_tokens is not None and wd_tokens.numel() > 0:
                condition_summary = f"{condition_summary}+wd"

        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
        sigma = torch.rand((clean.shape[0], 1, 1), generator=generator, device=device, dtype=clean.dtype)
        noisy = (1.0 - sigma) * clean + sigma * noise
        target_flow = noise - clean

        hidden_states = torch.cat([noisy, image_latents], dim=1)
        img_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
        timestep = sigma.flatten().to(dtype=clean.dtype)

        optimizer.zero_grad(set_to_none=True)
        with transformer.cache_context("cond"):
            pred = forward_transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
        pred = pred[:, : clean.shape[1], :]
        loss = F.mse_loss(pred.float(), target_flow.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
        optimizer.step()

        checkpoint_now = should_save_interval(step, total_steps, args.checkpoint_every)
        sample_now = should_save_interval(step, total_steps, args.sample_every)
        if main_process and checkpoint_now:
            save_lora_checkpoint(colorizer, args, step, wd_projector)
        if main_process and (checkpoint_now or sample_now):
            save_validation_samples(pipe, validation_dataset, collator, validation_indices, args, step, wd_projector)
        if distributed and (checkpoint_now or sample_now):
            dist.barrier()

        value = float(loss.detach().cpu())
        if main_process:
            losses.append(value)
            pbar.set_postfix(loss=f"{value:.5f}", grad_norm=f"{float(grad_norm):.3f}", cond=condition_summary)

    if not main_process:
        cleanup_distributed(distributed)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "smoke_metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"model_id={args.model_id}",
                f"image_size={args.image_size}",
                f"steps={total_steps}",
                f"openniji_repo_id={args.openniji_repo_id}",
                f"openniji_parquet_pattern={args.openniji_parquet_pattern}",
                f"parquet_files={len(dataset.parquet_paths)}",
                f"dataset_train_records={len(dataset)}",
                f"dataset_holdout_records={len(holdout_records)}",
                f"distributed={distributed}",
                f"world_size={world_size}",
                f"holdout_manifest={args.holdout_manifest or ''}",
                f"task_weights={task_weights}",
                f"atari_mode={args.atari_mode}",
                f"atari_dot_prob={args.atari_dot_prob}",
                f"validation_task={args.validation_task}",
                f"validation_atari_mode={args.validation_atari_mode}",
                f"sample_cell_size={args.sample_cell_size}",
                f"resume_lora={args.resume_lora or ''}",
                f"full_conditions={args.full_conditions}",
                f"max_condition_images={args.max_condition_images}",
                f"spatial_hint_mode={args.spatial_hint_mode}",
                f"spatial_condition_id_mode={args.spatial_condition_id_mode}",
                f"reference_policy={args.reference_policy}",
                f"reference_deform_strength={args.reference_deform_strength}",
                f"reference_background_source={args.reference_background_source}",
                f"reference_conditioning={args.reference_conditioning}",
                f"reference_condition_mode={args.reference_condition_mode}",
                f"reference_tag_cache_root={args.reference_tag_cache_root or ''}",
                f"reference_mask_root={args.reference_mask_root or ''}",
                f"reference_mask_fallback={args.reference_mask_fallback}",
                f"skytnt_repo={args.skytnt_repo or ''}",
                f"skytnt_model_id={args.skytnt_model_id}",
                f"skytnt_ckpt={args.skytnt_ckpt or ''}",
                f"skytnt_net={args.skytnt_net}",
                f"skytnt_image_size={args.skytnt_image_size}",
                f"skytnt_device={args.skytnt_device or args.device}",
                f"skytnt_fp32={args.skytnt_fp32}",
                f"skytnt_local_files_only={args.skytnt_local_files_only}",
                f"reference_cache_generated_masks={not args.no_reference_cache_generated_masks}",
                f"wd_tagger_model={args.wd_tagger_model or ''}",
                f"wd_tagger_labels={args.wd_tagger_labels or ''}",
                f"wd_tagger_threshold={args.wd_tagger_threshold}",
                f"wd_tagger_character_threshold={args.wd_tagger_character_threshold}",
                f"wd_tagger_max_tokens={args.wd_tagger_max_tokens}",
                f"reference_wd_context={args.reference_wd_context}",
                f"reference_wd_max_refs={args.reference_wd_max_refs}",
                f"reference_wd_tokens_per_ref={args.reference_wd_tokens_per_ref}",
                f"reference_wd_embed_dim={args.reference_wd_embed_dim}",
                f"reference_tag_max={args.reference_tag_max}",
                f"append_reference_tags={args.append_reference_tags}",
                f"sample_every={args.sample_every}",
                f"checkpoint_every={args.checkpoint_every}",
                f"sample_inference_steps={args.sample_inference_steps}",
                f"validation_examples={args.validation_examples}",
                "validation_indices=" + ",".join(str(x) for x in validation_indices),
                f"trainable_parameters={trainable}",
                f"total_parameters={total}",
                "losses=" + ",".join(f"{x:.8f}" for x in losses),
            ]
        )
        + "\n"
    )
    if args.save_lora:
        colorizer.save_lora(args.output_dir / "lora")
        if wd_projector is not None:
            torch.save(wd_projector.state_dict(), args.output_dir / "lora" / WD_PROJECTOR_FILE)
    print({"metrics": str(metrics_path), "losses": losses, "trainable_parameters": trainable, "total_parameters": total})
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
