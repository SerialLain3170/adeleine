from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .augmentations import AtariHintConfig, LegacyAtariHintGenerator
from .conditions import ModalityDropout, TaskSampler
from .openniji import OpenNijiParquetColorizationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Adeleine v2 line/Atari/reference examples")
    parser.add_argument("--output", type=Path, default=Path("/data/shasegawa/adeleine/outputs/examples/openniji_line_atari_reference_examples.png"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_records", type=int, default=24)
    parser.add_argument("--sketch_root", type=Path)
    parser.add_argument("--digital_root", type=Path)
    parser.add_argument("--anime_line_root", type=Path)
    parser.add_argument("--reference_policy", choices=["self", "deformed_self", "self_deformed", "sibling", "mixed", "none"], default="self")
    return parser.parse_args()


def thumb(array: np.ndarray, size: int = 220) -> Image.Image:
    return Image.fromarray(array).resize((size, size), Image.Resampling.LANCZOS)


def mask_preview(mask: np.ndarray) -> np.ndarray:
    mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
    return np.repeat(mask_2d[:, :, None], 3, axis=2).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if args.anime_line_root is not None:
        line_methods = ("lineart_anime",)
    elif args.sketch_root is not None:
        line_methods = ("pencil",)
    else:
        line_methods = ("xdog", "canny")

    dataset = OpenNijiParquetColorizationDataset(
        image_size=args.image_size,
        max_records=args.max_records,
        sketch_root=args.sketch_root,
        digital_root=args.digital_root,
        anime_line_root=args.anime_line_root,
        line_methods=line_methods,
        dropout=ModalityDropout(TaskSampler(weights={"all": 1.0})),
        reference_policy=args.reference_policy,
    )
    dataset.lineart.config.morphology_prob = 0.0
    dataset.lineart.config.color_variant_prob = 0.0
    dotter = LegacyAtariHintGenerator(AtariHintConfig(mode="dot"))
    liner = LegacyAtariHintGenerator(AtariHintConfig(mode="line"))

    samples = []
    meta_rows = ["sample	policy	input_sha	reference_sha	same_url	input_url	reference_url"]
    for idx in range(min(4, len(dataset))):
        item = dataset[idx]
        dot_rgb, dot_mask = dotter(item.target, item.lineart)
        line_rgb, line_mask = liner(item.target, item.lineart)
        input_url = item.metadata.get("url", "")
        ref_urls = item.metadata.get("reference_urls", [])
        ref_policy = item.metadata.get("reference_policy_actual", item.metadata.get("reference_policy", ""))
        ref_url = ref_urls[0] if ref_urls else ""
        ref = item.references[0] if item.references else np.full_like(item.target, 255)
        same_ref = bool(ref_url and ref_url == input_url)
        input_sha = hashlib.sha256(input_url.encode("utf-8")).hexdigest()[:10] if input_url else "none"
        ref_sha = hashlib.sha256(ref_url.encode("utf-8")).hexdigest()[:10] if ref_url else "none"
        meta_rows.append(f"{idx}\t{ref_policy}\t{input_sha}\t{ref_sha}\t{same_ref}\t{input_url}\t{ref_url}")
        samples.append([
            item.target,
            item.lineart,
            dot_rgb,
            mask_preview(dot_mask),
            line_rgb,
            mask_preview(line_mask),
            ref,
            np.full_like(item.target, 255 if not same_ref else 180),
        ])

    labels = ["target", "lineart", "dot_atari", "dot_mask", "line_atari", "line_mask", "reference", "ref_meta"]
    cell = 220
    label_h = 28
    sheet = Image.new("RGB", (len(labels) * cell, label_h + len(samples) * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(labels):
        draw.text((c * cell + 6, 8), label, fill=(0, 0, 0))
    for r, row in enumerate(samples):
        y = label_h + r * (cell + label_h)
        draw.text((6, y + 8), f"sample {r}", fill=(0, 0, 0))
        for c, array in enumerate(row):
            sheet.paste(thumb(array, cell), (c * cell, y + label_h))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    metadata_path = args.output.with_suffix(".tsv")
    metadata_path.write_text("\n".join(meta_rows) + "\n")
    print(args.output)
    print(metadata_path)


if __name__ == "__main__":
    main()
