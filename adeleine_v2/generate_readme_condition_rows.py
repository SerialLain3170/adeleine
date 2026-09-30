from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .atari import AtariHintConfig, AtariHintGenerator
from .conditions import ColorizationCondition, ColorizationMode
from .generate_readme_showcase import load_pipeline, make_dataset, manifest_records, render_condition


DEFAULT_TEXT_PROMPTS = (
    "pastel anime girl portrait, glossy blue eyes, soft pink hair, white frilled dress, "
    "blue lace ribbon, flower background, clean cel shading, vibrant polished anime illustration"
    "|gothic lolita anime girl, short silver hair, sapphire eyes, black and purple frilled dress, "
    "neon flower background, high contrast, crisp studio anime coloring"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README condition rows with visible reference inputs")
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
    parser.add_argument("--sample", type=int, default=33)
    parser.add_argument("--ref_samples", default="24,10")
    parser.add_argument("--text_prompts", default=DEFAULT_TEXT_PROMPTS)
    parser.add_argument("--inference_steps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--max_condition_images", type=int, default=4)
    parser.add_argument("--spatial_hint_mode", choices=["separate", "fused", "fused_masked"], default="separate")
    parser.add_argument("--spatial_condition_id_mode", choices=["default", "hint_to_output", "line_hint_to_output"], default="default")
    parser.add_argument("--output_basename", default="readme_showcase_condition_rows")
    parser.add_argument("--text_count", type=int, default=2)
    parser.add_argument("--hide_intro_row", action="store_true", help="Do not include the target/fixed-line-art intro row")
    parser.add_argument("--hide_line_only", action="store_true", help="Do not include the line-only baseline row")
    parser.add_argument("--seed", type=int, default=883170)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def color_condition(
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


def text_card(title: str, body: str, size: int, accent: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (size, size), (255, 252, 248))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size - 1, size - 1), outline=(224, 224, 224), width=2)
    draw.rectangle((0, 0, size - 1, 14), fill=accent)
    draw.text((26, 30), title, fill=(24, 24, 30), font=font)
    y = 68
    for line in textwrap.wrap(body, width=43)[:12]:
        draw.text((26, y), line, fill=(64, 64, 72), font=font)
        y += 18
    return image


def blank_condition(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size - 1, size - 1), outline=(224, 224, 224), width=2)
    draw.text((26, 30), "no optional input", fill=(64, 64, 72), font=font)
    return image


def save_image(path: Path, image: Image.Image | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(path)
    else:
        image.save(path)


def make_rows_sheet(base: ColorizationCondition, rows: list[dict], output: Path, include_intro_row: bool = True) -> None:
    cell = 300
    label_h = 36
    row_gap = 20
    header_h = 132
    margin = 34
    gap = 20
    row_h = label_h + cell + row_gap
    cols = 3
    width = margin * 2 + cols * cell + (cols - 1) * gap
    intro_rows = 1 if include_intro_row else 0
    height = header_h + margin + (len(rows) + intro_rows) * row_h
    sheet = Image.new("RGB", (width, height), (252, 250, 246))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, header_h), fill=(255, 238, 225))
    draw.rectangle((0, header_h - 8, width, header_h), fill=(255, 111, 97))
    draw.text((margin, 28), "Adeleine v2: same line art, different conditions", fill=(24, 24, 30), font=font)
    draw.text((margin, 54), "SketchKeras line art is fixed. Only text, Atari, or reference input changes.", fill=(64, 64, 72), font=font)
    draw.text((margin, 82), "Each row: fixed line art / active optional input / generated result", fill=(92, 92, 100), font=font)

    def paste_cell(image: Image.Image, col: int, row: int, title: str, accent: tuple[int, int, int]) -> None:
        x = margin + col * (cell + gap)
        y = header_h + margin + row * row_h
        draw.rectangle((x, y, x + cell - 1, y + label_h - 1), fill=(255, 255, 255), outline=(225, 225, 225))
        draw.rectangle((x, y, x + cell - 1, y + 5), fill=accent)
        draw.text((x + 10, y + 12), title, fill=(24, 24, 30), font=font)
        draw.rectangle((x, y + label_h, x + cell - 1, y + label_h + cell - 1), fill=(255, 255, 255), outline=(225, 225, 225))
        sheet.paste(image.resize((cell, cell), Image.Resampling.LANCZOS), (x, y + label_h))

    lineart = Image.fromarray(base.lineart)
    if include_intro_row:
        paste_cell(Image.fromarray(base.target), 0, 0, "holdout target", (255, 111, 97))
        paste_cell(lineart, 1, 0, "fixed SketchKeras line art", (40, 40, 46))
        paste_cell(lineart, 2, 0, "line art reused below", (40, 40, 46))
    for idx, row in enumerate(rows, start=intro_rows):
        accent = row.get("accent", (145, 106, 255))
        paste_cell(lineart, 0, idx, "same line art", (40, 40, 46))
        paste_cell(row["condition_image"], 1, idx, row["condition_title"], accent)
        paste_cell(row["result"], 2, idx, row["result_title"], accent)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = manifest_records(args.holdout_manifest)
    dataset = make_dataset(args, records)
    base = dataset[args.sample % len(dataset)]
    text_prompts = [item.strip() for item in args.text_prompts.split("|") if item.strip()]
    ref_indices = [int(item.strip()) % len(dataset) for item in args.ref_samples.split(",") if item.strip()]
    ref_items = [dataset[index] for index in ref_indices]

    np.random.seed(args.seed + 10)
    dot_rgb, dot_mask = AtariHintGenerator(
        AtariHintConfig(mode="dot", dot_min_hints=80, dot_max_hints=90, dot_max_patch_size=16, dot_uniform=True)
    )(base.target, base.lineart)
    np.random.seed(args.seed + 15)
    sparse_dot_rgb, sparse_dot_mask = AtariHintGenerator(
        AtariHintConfig(mode="dot", dot_min_hints=18, dot_max_hints=24, dot_max_patch_size=36, dot_uniform=True)
    )(base.target, base.lineart)
    np.random.seed(args.seed + 20)
    line_rgb, line_mask = AtariHintGenerator(
        AtariHintConfig(mode="line", line_min_hints=16, line_max_hints=24, line_min_length=28, line_max_length=80)
    )(base.target, base.lineart)
    rows: list[dict] = []
    if not args.hide_line_only:
        rows.append(
            {
                "name": "line_only",
                "condition": color_condition(base, name="line_only"),
                "condition_image": blank_condition(args.image_size),
                "condition_title": "no optional input",
                "result_title": "line-only result",
                "accent": (72, 78, 255),
            }
        )
    text_accents = [(255, 111, 97), (255, 90, 160), (111, 120, 255), (91, 192, 190)]
    for i, prompt in enumerate(text_prompts[: args.text_count], start=1):
        accent = text_accents[(i - 1) % len(text_accents)]
        rows.append(
            {
                "name": f"text_{i}",
                "condition": color_condition(base, text=prompt, name=f"text_{i}"),
                "condition_image": text_card(f"text prompt {i}", prompt, args.image_size, accent),
                "condition_title": f"text prompt {i}",
                "result_title": f"text result {i}",
                "accent": accent,
            }
        )
    rows.extend(
        [
            {
                "name": "dot_atari",
                "condition": color_condition(base, atari_rgb=dot_rgb, atari_mask=dot_mask, name="dot_atari"),
                "condition_image": Image.fromarray(dot_rgb),
                "condition_title": "dot Atari input",
                "result_title": "dot Atari result",
                "accent": (255, 177, 66),
            },
            {
                "name": "large_sparse_dot_atari",
                "condition": color_condition(base, atari_rgb=sparse_dot_rgb, atari_mask=sparse_dot_mask, name="large_sparse_dot_atari"),
                "condition_image": Image.fromarray(sparse_dot_rgb),
                "condition_title": "large sparse dot Atari",
                "result_title": "large sparse dot result",
                "accent": (255, 132, 75),
            },
            {
                "name": "line_atari",
                "condition": color_condition(base, atari_rgb=line_rgb, atari_mask=line_mask, name="line_atari"),
                "condition_image": Image.fromarray(line_rgb),
                "condition_title": "line Atari input",
                "result_title": "line Atari result",
                "accent": (91, 192, 190),
            },
        ]
    )
    for ref_index, ref_item in zip(ref_indices, ref_items):
        rows.append(
            {
                "name": f"reference_{ref_index}",
                "condition": color_condition(base, reference=ref_item.target, name=f"reference_{ref_index}"),
                "condition_image": Image.fromarray(ref_item.target),
                "condition_title": f"reference #{ref_index}",
                "result_title": f"reference #{ref_index} result",
                "accent": (145, 106, 255),
            }
        )
    save_image(args.output_dir / "target.png", base.target)
    save_image(args.output_dir / "lineart_sketchkeras.png", base.lineart)
    save_image(args.output_dir / "dot_atari.png", dot_rgb)
    save_image(args.output_dir / "large_sparse_dot_atari.png", sparse_dot_rgb)
    save_image(args.output_dir / "line_atari.png", line_rgb)
    for ref_index, ref_item in zip(ref_indices, ref_items):
        save_image(args.output_dir / f"reference_{ref_index}.png", ref_item.target)

    pipe = load_pipeline(args)
    metadata = {
        "sample": args.sample,
        "ref_samples": ref_indices,
        "lora_dir": str(args.lora_dir),
        "seed": args.seed,
        "spatial_hint_mode": args.spatial_hint_mode,
        "spatial_condition_id_mode": args.spatial_condition_id_mode,
        "text_prompts": text_prompts,
        "text_count": args.text_count,
        "hide_intro_row": args.hide_intro_row,
        "hide_line_only": args.hide_line_only,
        "rows": [],
    }
    for i, row in enumerate(rows):
        result, labels, prompt_text = render_condition(pipe, row["condition"], args, args.seed + i)
        row["result"] = result
        out = args.output_dir / f"generated_{i:02d}_{row['name']}.png"
        result.save(out)
        metadata["rows"].append({"name": row["name"], "condition_labels": labels, "prompt": prompt_text, "path": str(out)})

    sheet = args.output_dir / f"{args.output_basename}.png"
    make_rows_sheet(base, rows, sheet, include_intro_row=not args.hide_intro_row)
    meta_path = args.output_dir / f"{args.output_basename}.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(sheet)
    print(meta_path)


if __name__ == "__main__":
    main()
