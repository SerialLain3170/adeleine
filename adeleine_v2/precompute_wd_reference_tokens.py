from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

from .openniji import OpenNijiParquetColorizationDataset
from .reference_conditioning import ReferenceConditioningBuilder, ReferenceConditioningConfig, split_reference_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute WD tagger semantic tokens for Adeleine v2 reference foreground/background layers")
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--openniji_repo_id", default="all")
    parser.add_argument("--openniji_parquet_root", type=Path)
    parser.add_argument("--openniji_parquet_pattern", default="data/*.parquet")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--tag_cache_root", type=Path, required=True)
    parser.add_argument("--mask_root", type=Path, required=True)
    parser.add_argument("--mask_fallback", choices=["skytnt", "grabcut", "ellipse", "whole", "skip"], default="skip")
    parser.add_argument("--wd_tagger_model", type=Path, required=True)
    parser.add_argument("--wd_tagger_labels", type=Path, required=True)
    parser.add_argument("--wd_tagger_threshold", type=float, default=0.35)
    parser.add_argument("--wd_tagger_character_threshold", type=float, default=0.85)
    parser.add_argument("--wd_tagger_max_tokens", type=int, default=32)
    parser.add_argument("--reference_tag_max", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    args.tag_cache_root.mkdir(parents=True, exist_ok=True)

    dataset = OpenNijiParquetColorizationDataset(
        parquet_root=args.openniji_parquet_root,
        repo_id=args.openniji_repo_id,
        hf_home=args.hf_home,
        parquet_pattern=args.openniji_parquet_pattern,
        image_size=args.image_size,
        max_records=args.max_records,
        reference_policy="none",
        reference_conditioning="none",
    )
    builder = ReferenceConditioningBuilder(
        ReferenceConditioningConfig(
            mode="split_wd",
            tag_cache_root=args.tag_cache_root,
            mask_root=args.mask_root,
            mask_fallback=args.mask_fallback,
            wd_tagger_model=args.wd_tagger_model,
            wd_tagger_labels=args.wd_tagger_labels,
            wd_tagger_threshold=args.wd_tagger_threshold,
            wd_tagger_character_threshold=args.wd_tagger_character_threshold,
            wd_tagger_max_tokens=args.wd_tagger_max_tokens,
            max_tags=args.reference_tag_max,
        )
    )

    if args.manifest_path is not None:
        manifest_path = args.manifest_path
    elif args.start == 0 and args.stride == 1:
        manifest_path = args.tag_cache_root / "manifest.jsonl"
    else:
        manifest_path = args.tag_cache_root / f"manifest_start{args.start}_stride{args.stride}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0
    records = dataset.records[args.start :: max(1, args.stride)]
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for record in tqdm(records, desc="wd-reference-tokens"):
            try:
                row = dataset._read_record(record)
                color_bgr = dataset._decode_image(row["image"]["bytes"])
                color_bgr = dataset._resize_square(color_bgr)
                color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)
                mask = builder.reference_mask(color_rgb)
                if mask is None:
                    skipped += 1
                    continue
                fg, bg, _ = split_reference_layers(color_rgb, mask)
                layer_payloads = []
                for name, layer in (("foreground", fg), ("background", bg)):
                    cache_path = builder._cache_path(layer)
                    if cache_path is not None and cache_path.exists() and not args.overwrite:
                        skipped += 1
                        continue
                    result = builder.wd_result(layer, mask)
                    layer_payloads.append({
                        "layer": name,
                        "tags": result.tags,
                        "num_tokens": int(result.indices.shape[0]),
                    })
                    done += 1
                if layer_payloads:
                    manifest.write(
                        json.dumps(
                            {
                                "url": record.url,
                                "prompt": record.prompt,
                                "style": record.style,
                                "parquet_path": str(record.parquet_path),
                                "row_group": record.row_group,
                                "row_in_group": record.row_in_group,
                                "layers": layer_payloads,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception as exc:
                failed += 1
                tqdm.write(f"failed: {record.url} {exc}")
    print(json.dumps({"done_layers": done, "skipped_layers": skipped, "failed_records": failed, "tag_cache_root": str(args.tag_cache_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
