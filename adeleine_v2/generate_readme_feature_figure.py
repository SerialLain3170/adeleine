from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .adapters import batch_to_tensors
from .atari import AtariHintConfig, AtariHintGenerator
from .conditions import ColorizationCondition, ColorizationMode, ModalityDropout, TaskSampler
from .dataset import UnifiedCollator
from .evaluate_reference_transfer import compute_metrics
from .generate_readme_showcase import load_pipeline, manifest_records
from .openniji import OpenNijiParquetColorizationDataset
from .smoke_train_flux_klein import (
    build_condition_images,
    build_prompt_text,
    condition_id_policy,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact latest-checkpoint README feature figure")
    parser.add_argument("--lora_dir", type=Path, required=True)
    parser.add_argument("--holdout_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", default="readme_latest_feature.png")
    parser.add_argument("--background_output_name", default="readme_reference_background_check.png")
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--sketch_root", type=Path, default=Path("/data/shasegawa/adeleine/openniji/sketchkeras"))
    parser.add_argument(
        "--reference_mask_root",
        type=Path,
        default=Path("/data/shasegawa/adeleine/openniji/reference_masks/skytnt_512"),
    )
    parser.add_argument("--sample", type=int, default=33)
    parser.add_argument("--ref_samples", default="25,1,24,6")
    parser.add_argument("--background_ref_sample", type=int, default=16)
    parser.add_argument(
        "--text_prompt",
        default="anime girl with vivid violet hair, emerald green eyes, midnight blue dress, gold flower ornaments, clean cel shading",
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--inference_steps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=770001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def make_dataset(args: argparse.Namespace, records) -> OpenNijiParquetColorizationDataset:
    dataset = OpenNijiParquetColorizationDataset(
        repo_id="all",
        hf_home=args.hf_home,
        parquet_pattern="data/*.parquet",
        sketch_root=args.sketch_root,
        image_size=args.image_size,
        max_records=1,
        line_methods=("pencil",),
        dropout=ModalityDropout(TaskSampler(weights={"line": 1.0})),
        reference_policy="none",
        reference_conditioning="split",
        reference_mask_root=args.reference_mask_root,
        reference_mask_fallback="skip",
    )
    dataset.records = records
    dataset.groups = dataset._build_groups(records)
    dataset.lineart.config.morphology_prob = 0.0
    dataset.lineart.config.color_variant_prob = 0.0
    return dataset


def condition(
    dataset: OpenNijiParquetColorizationDataset,
    base: ColorizationCondition,
    *,
    text: str = "",
    atari_rgb: np.ndarray | None = None,
    atari_mask: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    name: str,
) -> ColorizationCondition:
    refs = [reference] if reference is not None else []
    ref_cond = dataset.reference_conditioner.build(refs)
    return ColorizationCondition(
        lineart=base.lineart,
        target=base.target,
        atari_rgb=atari_rgb,
        atari_mask=atari_mask,
        references=refs,
        reference_foregrounds=ref_cond.foregrounds,
        reference_backgrounds=ref_cond.backgrounds,
        reference_masks=ref_cond.masks,
        text=text,
        mode=ColorizationMode.RENDER,
        metadata={**base.metadata, "showcase": name},
    )


def render(pipe, item: ColorizationCondition, args: argparse.Namespace) -> tuple[Image.Image, list[str]]:
    batch = UnifiedCollator()([item])
    tensors = batch_to_tensors(batch, device="cpu")
    images, labels, _ = build_condition_images(
        tensors,
        False,
        6,
        spatial_hint_mode="fused_masked",
        reference_condition_mode="split",
    )
    prompt = build_prompt_text(tensors.text, batch.mode, batch.presence, False)[0]
    condition_pils = [tensor_to_pil(image) for image in images]
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.inference_mode(), condition_id_policy(pipe, labels, "hint_to_output"):
        generated = pipe(
            image=condition_pils,
            prompt=prompt,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=128,
        ).images[0]
    return generated.convert("RGB"), labels


def text_card(prompt: str, size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size, 62), fill=(236, 91, 143))
    draw.text((24, 17), "TEXT PROMPT", fill=(255, 255, 255), font=font(24, bold=True))
    y = 100
    for line in textwrap.wrap(prompt.upper(), width=19):
        draw.text((26, y), line, fill=(25, 28, 38), font=font(28, bold=True))
        y += 42
    return image


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    ratio = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (round(image.width * ratio), round(image.height * ratio)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def compose(
    lineart: Image.Image,
    variants: list[dict],
    checkpoint: str,
    output: Path,
) -> None:
    width, height = 2620, 1120
    canvas = Image.new("RGB", (width, height), (243, 245, 248))
    draw = ImageDraw.Draw(canvas)

    ink = (25, 28, 38)
    muted = (91, 98, 112)
    white = (255, 255, 255)
    coral = (244, 92, 92)
    accents = [(87, 101, 242), (236, 91, 143), (239, 164, 55), (32, 167, 153), (203, 68, 135), (226, 112, 56), (38, 155, 149), (235, 121, 51)]

    draw.rectangle((0, 0, width, 162), fill=ink)
    draw.rectangle((0, 154, width, 162), fill=coral)
    draw.text((54, 30), "ADELEINE v2", fill=white, font=font(48, bold=True))
    draw.text((54, 92), "ONE DRAWING. MANY COLOR STORIES.", fill=(207, 213, 225), font=font(24, bold=True))
    badge = f"FLUX.2 Klein / LoRA / {checkpoint.replace('_', ' ')}"
    badge_font = font(19, bold=True)
    bbox = draw.textbbox((0, 0), badge, font=badge_font)
    badge_w = bbox[2] - bbox[0] + 38
    draw.rounded_rectangle((width - badge_w - 54, 51, width - 54, 101), radius=6, fill=(48, 53, 68))
    draw.text((width - badge_w - 35, 65), badge, fill=white, font=badge_font)

    left_x, top = 54, 210
    line_size = 520
    draw.text((left_x, top - 34), "FIXED LINE ART", fill=ink, font=font(22, bold=True))
    draw.rectangle((left_x - 3, top - 3, left_x + line_size + 3, top + line_size + 3), fill=ink)
    canvas.paste(cover(lineart.convert("RGB"), (line_size, line_size)), (left_x, top))
    draw.text((left_x, top + line_size + 28), "Structure stays fixed.", fill=ink, font=font(25, bold=True))
    draw.text((left_x, top + line_size + 69), "Hints and references steer the color.", fill=muted, font=font(19))
    draw.text((left_x, top + line_size + 106), "Same seed across every result.", fill=muted, font=font(17))

    grid_x = 628
    card_w, card_h = 470, 414
    col_gap, row_gap = 22, 34
    for index, variant in enumerate(variants):
        col, row = index % 4, index // 4
        x = grid_x + col * (card_w + col_gap)
        y = top + row * (card_h + row_gap)
        accent = accents[index]

        draw.rectangle((x, y, x + card_w, y + card_h), fill=white)
        draw.rectangle((x, y, x + card_w, y + 7), fill=accent)
        draw.text((x + 22, y + 22), variant["title"], fill=ink, font=font(22, bold=True))
        draw.text((x + 22, y + 55), variant["subtitle"], fill=muted, font=font(15))

        inset_x, inset_y, inset_size = x + 22, y + 112, 102
        draw.rectangle((inset_x - 2, inset_y - 2, inset_x + inset_size + 2, inset_y + inset_size + 2), fill=(220, 224, 232))
        canvas.paste(cover(variant["input"].convert("RGB"), (inset_size, inset_size)), (inset_x, inset_y))
        draw.text((inset_x, inset_y + inset_size + 13), variant["input_label"], fill=muted, font=font(14, bold=True))
        draw.text((x + 138, y + 150), ">", fill=accent, font=font(32, bold=True))

        result_x, result_y, result_size = x + 180, y + 114, 266
        draw.rectangle((result_x - 2, result_y - 2, result_x + result_size + 2, result_y + result_size + 2), fill=ink)
        canvas.paste(cover(variant["result"], (result_size, result_size)), (result_x, result_y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def compose_background_check(
    lineart: Image.Image,
    reference: Image.Image,
    foreground: Image.Image,
    background: Image.Image,
    generated: Image.Image,
    metrics: dict[str, float],
    checkpoint: str,
    output: Path,
) -> None:
    width, height = 2200, 620
    canvas = Image.new("RGB", (width, height), (243, 245, 248))
    draw = ImageDraw.Draw(canvas)
    ink = (25, 28, 38)
    muted = (91, 98, 112)
    white = (255, 255, 255)
    teal = (32, 167, 153)

    draw.rectangle((0, 0, width, 126), fill=ink)
    draw.rectangle((0, 118, width, 126), fill=teal)
    draw.text((42, 24), "REFERENCE BACKGROUND CHECK", fill=white, font=font(38, bold=True))
    draw.text(
        (42, 75),
        "Character + detailed interior, split into the exact foreground and background conditions used by FLUX.",
        fill=(207, 213, 225),
        font=font(18),
    )
    badge = f"{checkpoint.replace('_', ' ')} / split reference"
    badge_font = font(17, bold=True)
    badge_box = draw.textbbox((0, 0), badge, font=badge_font)
    badge_w = badge_box[2] - badge_box[0] + 34
    draw.rounded_rectangle((width - badge_w - 42, 38, width - 42, 82), radius=6, fill=(48, 53, 68))
    draw.text((width - badge_w - 25, 51), badge, fill=white, font=badge_font)

    panels = [
        ("FIXED LINE ART", "structure input", lineart, (87, 101, 242)),
        ("FULL REFERENCE", "character + rich interior", reference, (236, 91, 143)),
        ("REFERENCE FG", "masked character layer", foreground, (203, 68, 135)),
        ("REFERENCE BG", "foreground removed + inpainted", background, teal),
        ("GENERATED", "reference-guided colorization", generated, (239, 164, 55)),
    ]
    margin, gap, panel_w = 42, 20, 407
    image_size = 340
    y = 154
    for index, (title, subtitle, image, accent) in enumerate(panels):
        x = margin + index * (panel_w + gap)
        draw.rectangle((x, y, x + panel_w, y + 405), fill=white)
        draw.rectangle((x, y, x + panel_w, y + 6), fill=accent)
        draw.text((x + 16, y + 16), title, fill=ink, font=font(20, bold=True))
        draw.text((x + 16, y + 45), subtitle, fill=muted, font=font(14))
        canvas.paste(cover(image.convert("RGB"), (image_size, image_size)), (x + 33, y + 72))

    struct_delta = metrics["copy_struct_ref_bg"] - metrics["copy_struct_bg_baseline"]
    edge_delta = metrics["copy_edge_ref"] - metrics["copy_edge_baseline"]
    alarm = "yes" if metrics["copy_alarm"] == 1.0 else "no"
    footer = f"background structure delta {struct_delta:+.3f}  /  reference edge delta {edge_delta:+.3f}  /  copy alarm {alarm}"
    draw.text((42, 582), footer, fill=muted, font=font(15, bold=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = manifest_records(args.holdout_manifest)
    dataset = make_dataset(args, records)
    base = dataset[args.sample % len(dataset)]
    ref_indices = [int(value.strip()) % len(dataset) for value in args.ref_samples.split(",") if value.strip()]
    if len(ref_indices) < 4:
        raise ValueError("--ref_samples must contain at least four indices")
    refs = [dataset[index].target for index in ref_indices[:4]]

    np.random.seed(args.seed + 1)
    dot_rgb, dot_mask = AtariHintGenerator(
        AtariHintConfig(
            mode="dot",
            dot_min_hints=58,
            dot_max_hints=68,
            dot_max_patch_size=18,
            dot_uniform=True,
        )
    )(base.target, base.lineart)
    np.random.seed(args.seed + 2)
    line_rgb, line_mask = AtariHintGenerator(
        AtariHintConfig(
            mode="line",
            line_min_hints=16,
            line_max_hints=22,
            line_min_length=28,
            line_max_length=76,
        )
    )(base.target, base.lineart)

    blank = Image.new("RGB", (args.image_size, args.image_size), "white")
    rows = [
        {
            "name": "line_only",
            "title": "LINE ONLY",
            "subtitle": "No optional condition",
            "input": blank,
            "input_label": "NONE",
            "condition": condition(dataset, base, name="line_only"),
        },
        {
            "name": "text",
            "title": "TEXT PROMPT",
            "subtitle": "Violet hair and emerald eyes",
            "input": text_card(args.text_prompt, args.image_size),
            "input_label": "TEXT",
            "condition": condition(dataset, base, text=args.text_prompt, name="text"),
        },
        {
            "name": "dot_atari",
            "title": "ATARI DOT HINTS",
            "subtitle": "Sparse target colors",
            "input": Image.fromarray(dot_rgb),
            "input_label": "DOT HINTS",
            "condition": condition(
                dataset,
                base,
                atari_rgb=dot_rgb,
                atari_mask=dot_mask,
                name="dot_atari",
            ),
        },
        {
            "name": "line_atari",
            "title": "ATARI LINE HINTS",
            "subtitle": "Colored strokes on the drawing",
            "input": Image.fromarray(line_rgb),
            "input_label": "LINE HINTS",
            "condition": condition(
                dataset,
                base,
                atari_rgb=line_rgb,
                atari_mask=line_mask,
                name="line_atari",
            ),
        },
    ]
    for palette, ref_index, reference in zip("ABCD", ref_indices[:4], refs):
        rows.append(
            {
                "name": f"reference_{ref_index}",
                "title": f"REFERENCE PALETTE {palette}",
                "subtitle": "Different image, same line art",
                "input": Image.fromarray(reference),
                "input_label": f"REFERENCE {ref_index}",
                "condition": condition(dataset, base, reference=reference, name=f"reference_{ref_index}"),
            }
        )

    background_ref_index = args.background_ref_sample % len(dataset)
    background_reference = dataset[background_ref_index].target
    background_condition = condition(
        dataset,
        base,
        reference=background_reference,
        name=f"background_reference_{background_ref_index}",
    )

    pipe = load_pipeline(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": args.lora_dir.name,
        "lora_dir": str(args.lora_dir),
        "sample": args.sample,
        "ref_samples": ref_indices[:4],
        "seed": args.seed,
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "text_prompt": args.text_prompt,
        "spatial_hint_mode": "fused_masked",
        "spatial_condition_id_mode": "hint_to_output",
        "reference_condition_mode": "split",
        "variants": [],
    }
    Image.fromarray(base.lineart).save(args.output_dir / "lineart.png")
    Image.fromarray(dot_rgb).save(args.output_dir / "dot_atari.png")
    Image.fromarray(line_rgb).save(args.output_dir / "line_atari.png")
    for ref_index, reference in zip(ref_indices[:4], refs):
        Image.fromarray(reference).save(args.output_dir / f"reference_{ref_index}.png")

    for row in rows:
        generated, labels = render(pipe, row["condition"], args)
        row["result"] = generated
        generated_path = args.output_dir / f"generated_{row['name']}.png"
        generated.save(generated_path)
        metadata["variants"].append(
            {
                "name": row["name"],
                "condition_labels": labels,
                "path": str(generated_path),
            }
        )

    background_generated, background_labels = render(pipe, background_condition, args)
    background_generated_path = args.output_dir / f"generated_background_reference_{background_ref_index}.png"
    background_generated.save(background_generated_path)
    foreground_array = (
        background_condition.reference_foregrounds[0]
        if background_condition.reference_foregrounds
        else background_reference
    )
    background_array = (
        background_condition.reference_backgrounds[0]
        if background_condition.reference_backgrounds
        else background_reference
    )
    mask_array = background_condition.reference_masks[0] if background_condition.reference_masks else None
    Image.fromarray(background_reference).save(args.output_dir / f"background_reference_{background_ref_index}.png")
    Image.fromarray(foreground_array).save(args.output_dir / f"background_reference_{background_ref_index}_fg.png")
    Image.fromarray(background_array).save(args.output_dir / f"background_reference_{background_ref_index}_bg.png")
    background_metrics = compute_metrics(
        np.asarray(background_generated, dtype=np.uint8),
        base.target,
        base.lineart,
        None,
        None,
        background_reference,
        background_array,
        mask_array,
        3,
        0.10,
    )
    background_output = args.output_dir / args.background_output_name
    compose_background_check(
        Image.fromarray(base.lineart),
        Image.fromarray(background_reference),
        Image.fromarray(foreground_array),
        Image.fromarray(background_array),
        background_generated,
        background_metrics,
        args.lora_dir.name,
        background_output,
    )
    metadata["background_check"] = {
        "reference_sample": background_ref_index,
        "condition_labels": background_labels,
        "generated_path": str(background_generated_path),
        "figure_path": str(background_output),
        "metrics": background_metrics,
    }

    output = args.output_dir / args.output_name
    compose(Image.fromarray(base.lineart), rows, args.lora_dir.name, output)
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(background_output)
    print(metadata_path)


if __name__ == "__main__":
    main()
