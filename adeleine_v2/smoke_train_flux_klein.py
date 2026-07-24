from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image, ImageDraw
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adapters import batch_to_tensors
from .atari import AtariHintConfig, AtariHintGenerator
from .conditions import ModalityDropout, TaskSampler
from .dataset import UnifiedCollator
from .flux_klein import FluxKleinColorizer, FluxKleinConfig
from .openniji import OpenNijiParquetColorizationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-train FLUX.2 Klein LoRA on the first OpenNiji shard")
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--openniji_repo_id", default="ShoukanLabs/OpenNiji-0_32237")
    parser.add_argument("--openniji_parquet_root", type=Path)
    parser.add_argument("--openniji_parquet_pattern", default="data/train-00000*.parquet")
    parser.add_argument("--sketch_root", type=Path)
    parser.add_argument("--digital_root", type=Path)
    parser.add_argument("--anime_line_root", type=Path)
    parser.add_argument("--reference_policy", choices=["self", "deformed_self", "self_deformed", "sibling", "mixed", "none"], default="deformed_self")
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



def build_condition_images(tensors, full_conditions: bool, max_condition_images: int) -> tuple[list[torch.Tensor], list[str], str]:
    images: list[torch.Tensor] = [tensors.lineart]
    labels = ["line"]
    if tensors.atari_rgb is not None:
        images.append(tensors.atari_rgb)
        labels.append("atari")
    if tensors.atari_mask is not None:
        mask_rgb = tensors.atari_mask.repeat(1, 3, 1, 1) * 2.0 - 1.0
        images.append(mask_rgb)
        labels.append("mask")
    for ref in tensors.reference_images[0]:
        images.append(ref.unsqueeze(0))
        labels.append("ref")
        if len(images) >= max_condition_images:
            break
    images = images[:max_condition_images]
    labels = labels[:max_condition_images]
    return images, labels, "+".join(labels)




def should_save_interval(step: int, total_steps: int, interval: int) -> bool:
    if interval <= 0:
        return False
    current = step + 1
    return current == 1 or current % interval == 0 or current == total_steps


def save_lora_checkpoint(colorizer: FluxKleinColorizer, args: argparse.Namespace, step: int) -> Path:
    checkpoint_dir = args.output_dir / "checkpoints" / f"step_{step + 1:06d}"
    colorizer.save_lora(checkpoint_dir)
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


def sample_grid_row(pipe, tensors, condition_images_cpu, condition_labels: list[str], prompt_text: list[str], args: argparse.Namespace, seed: int, row_index: int) -> Image.Image:
    generator = torch.Generator(device=args.device).manual_seed(seed)
    condition_pils = [tensor_to_pil(img) for img in condition_images_cpu]
    with torch.inference_mode():
        result = pipe(
            image=condition_pils,
            prompt=prompt_text[0],
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.sample_inference_steps,
            guidance_scale=args.sample_guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        ).images[0]
    items = [(f"target {row_index}", tensor_to_pil(tensors.target)), ("lineart", tensor_to_pil(tensors.lineart))]
    for label, image in zip(condition_labels[1:], condition_pils[1:]):
        items.append((label, image))
    items.append(("generated", result.convert("RGB")))
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


def save_validation_samples(pipe, dataset, collator: UnifiedCollator, validation_indices: list[int], args: argparse.Namespace, step: int) -> None:
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    was_training = pipe.transformer.training
    pipe.transformer.eval()
    rows = []
    for row_index, dataset_index in enumerate(validation_indices):
        condition = dataset[dataset_index]
        batch = collator([condition])
        tensors = batch_to_tensors(batch, device="cpu")
        condition_images_cpu, condition_labels, _ = build_condition_images(tensors, args.full_conditions, args.max_condition_images)
        prompt_text = build_prompt_text(tensors.text, batch.mode, batch.presence, args.full_conditions)
        rows.append(sample_grid_row(pipe, tensors, condition_images_cpu, condition_labels, prompt_text, args, args.sample_seed + step * 1000 + row_index, row_index))
    stack_grids(rows).save(sample_dir / f"step_{step + 1:06d}.png")
    if was_training:
        pipe.transformer.train()


def build_prompt_text(texts: list[str], modes: list[str], presence: dict, full_conditions: bool) -> list[str]:
    out = []
    for index, text in enumerate(texts):
        has_text = full_conditions or bool(presence.get("text", [False])[index])
        if not has_text:
            out.append("")
            continue
        active = []
        for key in ["lineart", "atari", "reference", "text", "flat", "diverse"]:
            value = presence.get(key)
            if value is not None and bool(value[index]):
                active.append(key)
        suffix = f" mode: {modes[index]}; conditions: {', '.join(active)}"
        out.append((text or "anime illustration colorization") + suffix)
    return out

def main() -> None:
    args = parse_args()
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
    )
    configure_atari(dataset, args.atari_mode, args.atari_dot_prob)
    train_records, holdout_records, train_record_indices, holdout_record_indices = split_holdout_records(dataset.records, args.holdout_records, args.seed)
    apply_records(dataset, train_records)
    validation_dataset = make_validation_dataset(dataset, holdout_records, args)
    if args.holdout_manifest is not None and holdout_records:
        args.holdout_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.holdout_manifest.write_text("\n".join(record_to_json(i, r) for i, r in zip(holdout_record_indices, holdout_records)) + "\n")
    collator = UnifiedCollator()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collator, drop_last=True)
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
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    freeze(pipe.vae)
    freeze(pipe.text_encoder)
    freeze(pipe.transformer)

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
        from peft import PeftModel

        transformer = PeftModel.from_pretrained(pipe.transformer, args.resume_lora, is_trainable=True)
        pipe.transformer = transformer
        colorizer.transformer = transformer
    else:
        transformer = colorizer.prepare_lora()
    transformer.train()

    trainable, total = colorizer.trainable_parameter_count(transformer)
    optimizer = torch.optim.AdamW((p for p in transformer.parameters() if p.requires_grad), lr=args.lr)

    device = component_device(transformer)
    vae_device = component_device(pipe.vae)
    text_device = component_device(pipe.text_encoder)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    losses = []
    iterator = iter(loader)
    total_steps = len(dataset) if args.steps <= 0 else args.steps
    pbar = tqdm(range(total_steps), desc="flux2-smoke-train")
    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        tensors = batch_to_tensors(batch, device="cpu")
        target = tensors.target.to(device=vae_device, dtype=component_dtype(pipe.vae))
        condition_images_cpu, condition_labels, condition_summary = build_condition_images(tensors, args.full_conditions, args.max_condition_images)
        if bool(batch.presence.get("text", [False])[0]):
            condition_summary = f"{condition_summary}+text"
        condition_images = [img.to(device=vae_device, dtype=component_dtype(pipe.vae)) for img in condition_images_cpu]
        prompt_text = build_prompt_text(tensors.text, batch.mode, batch.presence, args.full_conditions)

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

        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
        sigma = torch.rand((clean.shape[0], 1, 1), generator=generator, device=device, dtype=clean.dtype)
        noisy = (1.0 - sigma) * clean + sigma * noise
        target_flow = noise - clean

        hidden_states = torch.cat([noisy, image_latents], dim=1)
        img_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
        timestep = sigma.flatten().to(dtype=clean.dtype)

        optimizer.zero_grad(set_to_none=True)
        with transformer.cache_context("cond"):
            pred = transformer(
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
        grad_norm = torch.nn.utils.clip_grad_norm_((p for p in transformer.parameters() if p.requires_grad), 1.0)
        optimizer.step()

        checkpoint_now = should_save_interval(step, total_steps, args.checkpoint_every)
        sample_now = should_save_interval(step, total_steps, args.sample_every)
        if checkpoint_now:
            save_lora_checkpoint(colorizer, args, step)
        if checkpoint_now or sample_now:
            save_validation_samples(pipe, validation_dataset, collator, validation_indices, args, step)

        value = float(loss.detach().cpu())
        losses.append(value)
        pbar.set_postfix(loss=f"{value:.5f}", grad_norm=f"{float(grad_norm):.3f}", cond=condition_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "smoke_metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"model_id={args.model_id}",
                f"image_size={args.image_size}",
                f"steps={total_steps}",
                f"dataset_train_records={len(dataset)}",
                f"dataset_holdout_records={len(holdout_records)}",
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
                f"reference_policy={args.reference_policy}",
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
    print({"metrics": str(metrics_path), "losses": losses, "trainable_parameters": trainable, "total_parameters": total})


if __name__ == "__main__":
    main()
