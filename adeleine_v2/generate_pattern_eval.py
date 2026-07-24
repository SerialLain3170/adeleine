from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageDraw

from .adapters import batch_to_tensors
from .conditions import ColorizationCondition, ColorizationMode, ModalityDropout, TaskSampler
from .dataset import UnifiedCollator
from .openniji import OpenNijiParquetColorizationDataset
from .smoke_train_flux_klein import build_condition_images, build_prompt_text, tensor_to_pil


PATTERNS = [
    "line",
    "line_atari",
    "line_reference",
    "line_text",
    "line_reference_atari",
    "all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final Adeleine v2 pattern evaluation grids")
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--lora_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--openniji_repo_id", default="ShoukanLabs/OpenNiji-0_32237")
    parser.add_argument("--openniji_parquet_pattern", default="data/*.parquet")
    parser.add_argument("--sketch_root", type=Path)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--ref_offset", type=int, default=137)
    parser.add_argument("--inference_steps", type=int, default=12)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=913170)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def freeze_random_lineart(dataset: OpenNijiParquetColorizationDataset) -> None:
    dataset.lineart.config.morphology_prob = 0.0
    dataset.lineart.config.color_variant_prob = 0.0


def make_condition(base: ColorizationCondition, ref: np.ndarray | None, pattern: str) -> ColorizationCondition:
    keep_atari = pattern in {"line_atari", "line_reference_atari", "all"}
    keep_ref = pattern in {"line_reference", "line_reference_atari", "all"}
    keep_text = pattern in {"line_text", "all"}
    return ColorizationCondition(
        lineart=base.lineart,
        target=base.target,
        atari_rgb=base.atari_rgb if keep_atari else None,
        atari_mask=base.atari_mask if keep_atari else None,
        references=[ref] if keep_ref and ref is not None else [],
        text=base.text if keep_text else "",
        mode=ColorizationMode.DIVERSE if pattern == "line_text" else ColorizationMode.RENDER,
        metadata={**base.metadata, "task": pattern},
    )


def blank_like(target: np.ndarray) -> Image.Image:
    return Image.fromarray(np.full_like(target, 255, dtype=np.uint8))


def mask_preview(mask_tensor: torch.Tensor | None, target: np.ndarray) -> Image.Image:
    if mask_tensor is None:
        return blank_like(target)
    mask_rgb = mask_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
    return tensor_to_pil(mask_rgb)


def load_pipeline(args: argparse.Namespace):
    from diffusers import Flux2KleinPipeline
    from peft import PeftModel

    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_dir)
    pipe.to(args.device)
    pipe.transformer.eval()
    return pipe


def run_one(pipe, tensors, condition_images_cpu, prompt_text: list[str], args: argparse.Namespace, seed: int) -> Image.Image:
    condition_pils = [tensor_to_pil(img) for img in condition_images_cpu]
    generator = torch.Generator(device=args.device).manual_seed(seed)
    with torch.inference_mode():
        image = pipe(
            image=condition_pils,
            prompt=prompt_text[0],
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        ).images[0]
    return image.convert("RGB")


def draw_cell(sheet: Image.Image, image: Image.Image, col: int, row: int, cell: int, label_h: int, label: str | None = None) -> None:
    draw = ImageDraw.Draw(sheet)
    x = col * cell
    y = row * (cell + label_h)
    if label is not None:
        draw.text((x + 6, y + 6), label, fill=(0, 0, 0))
    sheet.paste(image.resize((cell, cell), Image.Resampling.LANCZOS), (x, y + label_h))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    line_methods: tuple[str, ...] = ("pencil",) if args.sketch_root is not None else ("xdog",)
    dataset = OpenNijiParquetColorizationDataset(
        repo_id=args.openniji_repo_id,
        hf_home=args.hf_home,
        parquet_pattern=args.openniji_parquet_pattern,
        sketch_root=args.sketch_root,
        image_size=args.image_size,
        max_records=max(args.start_index + args.examples + args.ref_offset + 1, 512),
        line_methods=line_methods,
        dropout=ModalityDropout(TaskSampler(weights={"all": 1.0})),
        reference_policy="deformed_self",
    )
    freeze_random_lineart(dataset)
    collator = UnifiedCollator()
    pipe = load_pipeline(args)

    cell = 192
    label_h = 28
    columns = ["pattern", "target", "lineart", "atari", "mask", "reference", "generated"]
    rows = len(PATTERNS) * args.examples + 1
    sheet = Image.new("RGB", (len(columns) * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(columns):
        draw.text((col * cell + 6, 6), label, fill=(0, 0, 0))

    metadata = ["row\tpattern\tindex\tref_index\tprompt"]
    row = 1
    for pattern_index, pattern in enumerate(PATTERNS):
        for example_index in range(args.examples):
            index = args.start_index + example_index
            ref_index = (index + args.ref_offset + pattern_index * 17) % len(dataset)
            base = dataset[index]
            ref_item = dataset[ref_index]
            ref = ref_item.target
            condition = make_condition(base, ref, pattern)
            batch = collator([condition])
            tensors = batch_to_tensors(batch, device="cpu")
            condition_images_cpu, condition_labels, _ = build_condition_images(tensors, False, 6)
            prompt_text = build_prompt_text(tensors.text, batch.mode, batch.presence, False)
            generated = run_one(pipe, tensors, condition_images_cpu, prompt_text, args, args.seed + row)

            draw_cell(sheet, Image.new("RGB", (cell, cell), "white"), 0, row, cell, label_h, f"{pattern}\nex {example_index}")
            draw_cell(sheet, tensor_to_pil(tensors.target), 1, row, cell, label_h)
            draw_cell(sheet, tensor_to_pil(tensors.lineart), 2, row, cell, label_h)
            draw_cell(sheet, tensor_to_pil(tensors.atari_rgb) if tensors.atari_rgb is not None else blank_like(base.target), 3, row, cell, label_h)
            draw_cell(sheet, mask_preview(tensors.atari_mask, base.target), 4, row, cell, label_h)
            ref_image = tensor_to_pil(tensors.reference_images[0][0].unsqueeze(0)) if tensors.reference_images and tensors.reference_images[0] else blank_like(base.target)
            draw_cell(sheet, ref_image, 5, row, cell, label_h)
            draw_cell(sheet, generated, 6, row, cell, label_h)
            metadata.append(f"{row}\t{pattern}\t{index}\t{ref_index}\t{prompt_text[0].replace(chr(9), ' ')[:240]}")
            row += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    args.output.with_suffix(".tsv").write_text("\n".join(metadata) + "\n")
    print(args.output)
    print(args.output.with_suffix(".tsv"))


if __name__ == "__main__":
    main()
