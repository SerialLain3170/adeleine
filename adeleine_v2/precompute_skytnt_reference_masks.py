from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

from .openniji import OpenNijiParquetColorizationDataset
from .reference_conditioning import reference_digest
from .skytnt_segmentation import SkyTNTMaskExtractor, SkyTNTMaskExtractorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute SkyTNT anime foreground masks for Adeleine v2 references")
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--openniji_repo_id", default="all")
    parser.add_argument("--openniji_parquet_root", type=Path)
    parser.add_argument("--openniji_parquet_pattern", default="data/*.parquet")
    parser.add_argument("--output_dir", type=Path, default=Path("/data/shasegawa/adeleine/openniji/reference_masks/skytnt_512"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest_path", type=Path, help="Optional per-worker manifest path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skytnt_repo", type=Path, help="Local clone of https://github.com/SkyTNT/anime-segmentation")
    parser.add_argument("--skytnt_model_id", default="skytnt/anime-seg")
    parser.add_argument("--skytnt_ckpt", type=Path)
    parser.add_argument("--skytnt_net", default="isnet_is")
    parser.add_argument("--skytnt_image_size", type=int, default=1024)
    parser.add_argument("--skytnt_fp32", action="store_true")
    parser.add_argument("--skytnt_local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    extractor = SkyTNTMaskExtractor(
        SkyTNTMaskExtractorConfig(
            repo=args.skytnt_repo,
            model_id=args.skytnt_model_id,
            ckpt=args.skytnt_ckpt,
            net=args.skytnt_net,
            image_size=args.skytnt_image_size,
            device=args.device,
            fp32=args.skytnt_fp32,
            local_files_only=args.skytnt_local_files_only,
        )
    )

    if args.manifest_path is not None:
        manifest_path = args.manifest_path
    elif args.start == 0 and args.stride == 1:
        manifest_path = args.output_dir / "manifest.jsonl"
    else:
        manifest_path = args.output_dir / f"manifest_start{args.start}_stride{args.stride}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    skipped = 0
    failed = 0
    records = dataset.records[args.start :: max(1, args.stride)]
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for record in tqdm(records, desc="skytnt-reference-masks"):
            try:
                row = dataset._read_record(record)
                color_bgr = dataset._decode_image(row["image"]["bytes"])
                color_bgr = dataset._resize_square(color_bgr)
                color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)
                digest = reference_digest(color_rgb)
                out_path = args.output_dir / f"{digest}.png"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                mask = extractor.mask(color_rgb)
                cv.imwrite(str(out_path), mask[..., 0])
                manifest.write(
                    json.dumps(
                        {
                            "digest": digest,
                            "mask": str(out_path),
                            "url": record.url,
                            "prompt": record.prompt,
                            "style": record.style,
                            "parquet_path": str(record.parquet_path),
                            "row_group": record.row_group,
                            "row_in_group": record.row_in_group,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                done += 1
            except Exception as exc:
                failed += 1
                tqdm.write(f"failed: {record.url} {exc}")
    print(json.dumps({"done": done, "skipped": skipped, "failed": failed, "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
