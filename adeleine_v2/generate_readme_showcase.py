from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .adapters import batch_to_tensors
from .atari import AtariHintConfig, AtariHintGenerator
from .conditions import ColorizationCondition, ColorizationMode, ModalityDropout, TaskSampler
from .dataset import UnifiedCollator
from .openniji import (
    OpenNijiParquetColorizationDataset,
    OpenNijiParquetRecord,
    deform_reference_rgb,
    prompt_group_key,
)
from .smoke_train_flux_klein import build_condition_images, build_prompt_text, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Adeleine README showcase from holdout samples")
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--lora_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--holdout_manifest",
        type=Path,
        default=Path("/data/shasegawa/adeleine/outputs/flux2_klein_openniji_full_epoch_atari_focus_holdout1k_512/holdout_1000.jsonl"),
    )
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--sketch_root", type=Path, default=Path("/data/shasegawa/adeleine/openniji/sketchkeras"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--sample", type=int, default=5, help="Holdout manifest row used as the fixed line-art sample")
    parser.add_argument("--ref_samples", default="1,7,19", help="Comma-separated holdout manifest rows used as alternate references")
    parser.add_argument("--inference_steps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--max_condition_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=803170)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--candidate_sheet", action="store_true", help="Only export target/lineart candidates")
    parser.add_argument("--candidate_count", type=int, default=36)
    return parser.parse_args()


def manifest_records(path: Path) -> list[OpenNijiParquetRecord]:
    records: list[OpenNijiParquetRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt", "")
            records.append(
                OpenNijiParquetRecord(
                    parquet_path=Path(item["parquet_path"]),
                    row_group=int(item["row_group"]),
                    row_in_group=int(item["row_in_group"]),
                    prompt=prompt,
                    style=item.get("style", ""),
                    url=item.get("url", ""),
                    group_key=item.get("group_key") or prompt_group_key(prompt),
                )
            )
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def make_dataset(args: argparse.Namespace, records: list[OpenNijiParquetRecord]) -> OpenNijiParquetColorizationDataset:
    dataset = OpenNijiParquetColorizationDataset(
        repo_id="all",
        hf_home=args.hf_home,
        parquet_pattern="data/*.parquet",
        sketch_root=args.sketch_root,
        image_size=args.image_size,
        max_records=1,
        line_methods=("pencil",),
        dropout=ModalityDropout(TaskSampler(weights={"all": 1.0})),
        reference_policy="deformed_self",
    )
    dataset.records = records
    dataset.groups = dataset._build_groups(records)
    dataset.lineart.config.morphology_prob = 0.0
    dataset.lineart.config.color_variant_prob = 0.0
    return dataset


def load_pipeline(args: argparse.Namespace):
    from diffusers import Flux2KleinPipeline
    from peft import PeftModel

    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id,
        cache_dir=str(args.hf_home / "hub"),
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_dir)
    pipe.to(args.device)
    pipe.transformer.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def render_condition(pipe, condition: ColorizationCondition, args: argparse.Namespace, seed: int) -> tuple[Image.Image, list[str], str]:
    collator = UnifiedCollator()
    batch = collator([condition])
    tensors = batch_to_tensors(batch, device="cpu")
    condition_images_cpu, labels, _ = build_condition_images(tensors, False, args.max_condition_images)
    prompt_text = build_prompt_text(tensors.text, batch.mode, batch.presence, False)
    condition_pils = [tensor_to_pil(img) for img in condition_images_cpu]
    generator = torch.Generator(device=args.device).manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            image=condition_pils,
            prompt=prompt_text[0],
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        ).images[0]
    return result.convert("RGB"), labels, prompt_text[0]


def make_condition(
    base: ColorizationCondition,
    *,
    text: str = "",
    atari_rgb: np.ndarray | None = None,
    atari_mask: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    name: str,
) -> ColorizationCondition:
    return ColorizationCondition(
        lineart=base.lineart,
        target=base.target,
        atari_rgb=atari_rgb,
        atari_mask=atari_mask,
        references=[reference] if reference is not None else [],
        text=text,
        mode=ColorizationMode.RENDER,
        metadata={**base.metadata, "showcase": name},
    )


def mask_preview(mask: np.ndarray | None, size: int) -> Image.Image:
    if mask is None:
        return Image.new("RGB", (size, size), "white")
    m = mask.squeeze()
    rgb = np.full((m.shape[0], m.shape[1], 3), 255, dtype=np.uint8)
    rgb[m > 0] = 0
    return Image.fromarray(rgb)


def fit_text(draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.ImageFont) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        wrapped = textwrap.wrap(paragraph, width=42) or [""]
        lines.extend(wrapped)
    out: list[str] = []
    for line in lines:
        while draw.textlength(line, font=font) > width and len(line) > 8:
            line = line[:-1]
        out.append(line)
    return out


def framed_tile(
    title: str,
    subtitle: str,
    image: Image.Image,
    *,
    tile_w: int,
    tile_h: int,
    image_h: int,
    bg: tuple[int, int, int],
    accent: tuple[int, int, int],
) -> Image.Image:
    tile = Image.new("RGB", (tile_w, tile_h), bg)
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, tile_w - 1, tile_h - 1), outline=(225, 225, 225), width=1)
    draw.rectangle((0, 0, tile_w - 1, 6), fill=accent)
    draw.text((14, 14), title, fill=(22, 22, 28), font=font)
    if subtitle:
        y = 32
        for line in fit_text(draw, subtitle, tile_w - 28, font)[:2]:
            draw.text((14, y), line, fill=(80, 80, 88), font=font)
            y += 13
    image_size = min(tile_w - 28, image_h)
    resized = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    x = (tile_w - image_size) // 2
    y = tile_h - image_size - 14
    tile.paste(resized, (x, y))
    return tile


def make_showcase_sheet(
    base: ColorizationCondition,
    variants: list[dict],
    prompt: str,
    output: Path,
    image_size: int,
) -> None:
    tile_w = 286
    tile_h = 374
    image_h = 286
    gap = 18
    margin = 32
    header_h = 126
    cols = 4
    rows = 3
    width = margin * 2 + cols * tile_w + (cols - 1) * gap
    height = header_h + margin + rows * tile_h + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), (252, 250, 246))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, header_h), fill=(255, 238, 225))
    draw.rectangle((0, header_h - 8, width, header_h), fill=(255, 111, 97))
    draw.text((margin, 28), "Adeleine v2 line-art colorization", fill=(24, 24, 30), font=font)
    draw.text((margin, 52), "Same SketchKeras line art, different optional conditions", fill=(64, 64, 72), font=font)
    prompt_text = textwrap.shorten(prompt.replace("\n", " "), width=150, placeholder="...")
    draw.text((margin, 78), f"Holdout prompt: {prompt_text}", fill=(92, 92, 100), font=font)

    base_tiles = [
        {
            "title": "target",
            "subtitle": "held-out image",
            "image": Image.fromarray(base.target),
            "accent": (255, 111, 97),
        },
        {
            "title": "SketchKeras line art",
            "subtitle": "fixed input for every result",
            "image": Image.fromarray(base.lineart),
            "accent": (40, 40, 46),
        },
        {
            "title": "dot Atari",
            "subtitle": "sparse color dots",
            "image": Image.fromarray(variants[2]["atari"]),
            "accent": (255, 177, 66),
        },
        {
            "title": "line Atari",
            "subtitle": "colored line hints",
            "image": Image.fromarray(variants[3]["atari"]),
            "accent": (91, 192, 190),
        },
    ]
    all_tiles = base_tiles + variants
    for idx, item in enumerate(all_tiles[: cols * rows]):
        col = idx % cols
        row = idx // cols
        x = margin + col * (tile_w + gap)
        y = header_h + margin + row * (tile_h + gap)
        tile = framed_tile(
            item["title"],
            item.get("subtitle", ""),
            item["image"],
            tile_w=tile_w,
            tile_h=tile_h,
            image_h=image_h,
            bg=(255, 255, 255),
            accent=item.get("accent", (145, 106, 255)),
        )
        sheet.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def candidate_sheet(dataset: OpenNijiParquetColorizationDataset, args: argparse.Namespace) -> Path:
    count = min(args.candidate_count, len(dataset))
    cell = 160
    label_h = 24
    cols = 6
    rows = int(np.ceil(count / cols))
    sheet = Image.new("RGB", (cols * cell * 2, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(count):
        item = dataset[i]
        x = (i % cols) * cell * 2
        y = (i // cols) * (cell + label_h)
        draw.text((x + 4, y + 5), f"{i}", fill=(0, 0, 0))
        sheet.paste(Image.fromarray(item.target).resize((cell, cell), Image.Resampling.LANCZOS), (x, y + label_h))
        sheet.paste(Image.fromarray(item.lineart).resize((cell, cell), Image.Resampling.LANCZOS), (x + cell, y + label_h))
    out = args.output_dir / "holdout_candidates.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def save_tile(path: Path, image: Image.Image | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(path)
    else:
        image.save(path)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = manifest_records(args.holdout_manifest)
    dataset = make_dataset(args, records)
    if args.candidate_sheet:
        print(candidate_sheet(dataset, args))
        return

    base = dataset[args.sample % len(dataset)]
    prompt = base.text or "anime illustration colorization"
    ref_indices = [int(x.strip()) % len(dataset) for x in args.ref_samples.split(",") if x.strip()]
    ref_items = [dataset[i] for i in ref_indices]

    np.random.seed(args.seed + 10)
    dot_gen = AtariHintGenerator(
        AtariHintConfig(mode="dot", dot_min_hints=70, dot_max_hints=80, dot_max_patch_size=15, dot_uniform=True)
    )
    dot_rgb, dot_mask = dot_gen(base.target, base.lineart)
    np.random.seed(args.seed + 20)
    line_gen = AtariHintGenerator(
        AtariHintConfig(mode="line", line_min_hints=14, line_max_hints=22, line_min_length=24, line_max_length=72)
    )
    line_rgb, line_mask = line_gen(base.target, base.lineart)
    np.random.seed(args.seed + 30)
    self_ref = deform_reference_rgb(base.target.copy())

    variants = [
        {
            "name": "line",
            "title": "result: line only",
            "subtitle": "no hint, no text, no reference",
            "condition": make_condition(base, name="line"),
            "accent": (72, 78, 255),
        },
        {
            "name": "text",
            "title": "result: + text",
            "subtitle": "prompt conditions color and context",
            "condition": make_condition(base, text=prompt, name="text"),
            "accent": (255, 111, 97),
        },
        {
            "name": "dot_atari",
            "title": "result: + dot Atari",
            "subtitle": "sparse dot color hints",
            "condition": make_condition(base, atari_rgb=dot_rgb, atari_mask=dot_mask, name="dot_atari"),
            "atari": dot_rgb,
            "mask": dot_mask,
            "accent": (255, 177, 66),
        },
        {
            "name": "line_atari",
            "title": "result: + line Atari",
            "subtitle": "colored line hints",
            "condition": make_condition(base, atari_rgb=line_rgb, atari_mask=line_mask, name="line_atari"),
            "atari": line_rgb,
            "mask": line_mask,
            "accent": (91, 192, 190),
        },
        {
            "name": "self_reference",
            "title": "result: + self ref",
            "subtitle": "deformed original reference",
            "condition": make_condition(base, reference=self_ref, name="self_reference"),
            "reference": self_ref,
            "accent": (116, 210, 144),
        },
    ]
    for ref_i, ref_item in zip(ref_indices[:2], ref_items[:2]):
        variants.append(
            {
                "name": f"reference_{ref_i}",
                "title": f"result: + ref #{ref_i}",
                "subtitle": "different holdout reference",
                "condition": make_condition(base, reference=ref_item.target, name=f"reference_{ref_i}"),
                "reference": ref_item.target,
                "accent": (145, 106, 255),
            }
        )
    variants.append(
        {
            "name": "all",
            "title": "result: all conditions",
            "subtitle": "text + dot Atari + reference",
            "condition": make_condition(base, text=prompt, atari_rgb=dot_rgb, atari_mask=dot_mask, reference=self_ref, name="all"),
            "reference": self_ref,
            "atari": dot_rgb,
            "mask": dot_mask,
            "accent": (255, 90, 160),
        }
    )

    pipe = load_pipeline(args)
    metadata = {
        "sample": args.sample,
        "ref_samples": ref_indices,
        "lora_dir": str(args.lora_dir),
        "prompt": prompt,
        "seed": args.seed,
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "variants": [],
    }

    save_tile(args.output_dir / "target.png", base.target)
    save_tile(args.output_dir / "lineart_sketchkeras.png", base.lineart)
    save_tile(args.output_dir / "dot_atari.png", dot_rgb)
    save_tile(args.output_dir / "dot_mask.png", mask_preview(dot_mask, args.image_size))
    save_tile(args.output_dir / "line_atari.png", line_rgb)
    save_tile(args.output_dir / "line_mask.png", mask_preview(line_mask, args.image_size))
    save_tile(args.output_dir / "reference_self_deformed.png", self_ref)
    for ref_i, ref_item in zip(ref_indices, ref_items):
        save_tile(args.output_dir / f"reference_{ref_i}.png", ref_item.target)

    for index, variant in enumerate(variants):
        generated, labels, prompt_text = render_condition(pipe, variant["condition"], args, args.seed + index)
        variant["image"] = generated
        out = args.output_dir / f"generated_{index:02d}_{variant['name']}.png"
        generated.save(out)
        metadata["variants"].append(
            {
                "name": variant["name"],
                "condition_labels": labels,
                "prompt": prompt_text,
                "path": str(out),
            }
        )

    sheet_path = args.output_dir / "readme_showcase_step080000.png"
    make_showcase_sheet(base, variants, prompt, sheet_path, args.image_size)
    (args.output_dir / "readme_showcase_step080000.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(sheet_path)
    print(args.output_dir / "readme_showcase_step080000.json")


if __name__ == "__main__":
    main()
