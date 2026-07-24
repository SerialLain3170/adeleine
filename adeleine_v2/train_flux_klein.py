from __future__ import annotations

import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader

from .adapters import AdeleineConditionAdapter, batch_to_tensors
from .dataset import UnifiedCollator, UnifiedColorizationDataset
from .flat import FlatOutputConfig
from .flux_klein import FluxKleinColorizer, FluxKleinConfig
from .openniji import OpenNijiColorizationDataset, OpenNijiParquetColorizationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adeleine v2 FLUX.2 Klein adapter training scaffold")
    parser.add_argument("--dataset", choices=["folder", "openniji"], default="openniji")
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--sketch_root", type=Path)
    parser.add_argument("--digital_root", type=Path)
    parser.add_argument("--anime_line_root", type=Path)
    parser.add_argument("--flat_root", type=Path)
    parser.add_argument("--reference_root", type=Path)
    parser.add_argument("--reference_policy", choices=["self", "deformed_self", "self_deformed", "sibling", "mixed", "none"], default="self")
    parser.add_argument("--caption_file", type=Path)
    parser.add_argument("--openniji_source", choices=["parquet", "jsonl"], default="parquet")
    parser.add_argument("--openniji_repo_id", default="ShoukanLabs/OpenNiji-0_32237")
    parser.add_argument("--openniji_parquet_root", type=Path)
    parser.add_argument("--openniji_parquet_pattern", default="data/*.parquet")
    parser.add_argument("--openniji_jsonl", type=Path)
    parser.add_argument("--openniji_image_cache", type=Path, default=Path("/data/shasegawa/adeleine/openniji/images"))
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--no_download_images", action="store_true")
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--extension", default=".jpg")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adapter_check", action="store_true")
    parser.add_argument("--prepare_lora", action="store_true")
    parser.add_argument("--lora_output_dir", type=Path)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    if args.dataset == "openniji":
        if args.openniji_source == "parquet":
            return OpenNijiParquetColorizationDataset(
                parquet_root=args.openniji_parquet_root,
                repo_id=args.openniji_repo_id,
                hf_home=args.hf_home,
                parquet_pattern=args.openniji_parquet_pattern,
                sketch_root=args.sketch_root,
                digital_root=args.digital_root,
                anime_line_root=args.anime_line_root,
                image_size=args.image_size,
                max_records=args.max_records,
                reference_policy=args.reference_policy,
            )
        return OpenNijiColorizationDataset(
            jsonl_path=args.openniji_jsonl,
            image_cache=args.openniji_image_cache,
            hf_home=args.hf_home,
            sketch_root=args.sketch_root,
            digital_root=args.digital_root,
            anime_line_root=args.anime_line_root,
            image_size=args.image_size,
            max_records=args.max_records,
            download=not args.no_download_images,
            reference_policy=args.reference_policy,
        )

    if args.data_root is None:
        raise ValueError("--data_root is required when --dataset folder")
    return UnifiedColorizationDataset(
        data_root=args.data_root,
        sketch_root=args.sketch_root,
        digital_root=args.digital_root,
        anime_line_root=args.anime_line_root,
        flat_root=args.flat_root,
        reference_root=args.reference_root,
        caption_file=args.caption_file,
        extension=args.extension,
        image_size=args.image_size,
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=UnifiedCollator(),
        drop_last=True,
    )
    model = FluxKleinColorizer(FluxKleinConfig(model_id=args.model_id, lora_rank=args.lora_rank, lora_alpha=args.lora_rank))
    flat = FlatOutputConfig(enabled=args.flat_root is not None)

    batch = next(iter(loader))
    payload = model.build_condition_payload(batch)
    print("Adeleine v2 batch ready")
    print({key: (None if value is None else type(value).__name__) for key, value in payload.items()})
    print("flat_enabled:", flat.enabled)

    if args.adapter_check:
        tensors = batch_to_tensors(batch, device="cpu")
        adapter = AdeleineConditionAdapter(hidden_size=args.hidden_size)
        outputs = adapter(tensors)
        print("adapter_tokens:", {key: tuple(value.shape) for key, value in outputs.items()})

    if args.dry_run:
        return

    model.load_pipeline()
    if args.prepare_lora:
        transformer = model.prepare_lora()
        trainable, total = model.trainable_parameter_count(transformer)
        print("lora_parameters:", {"trainable": trainable, "total": total})
        if args.lora_output_dir is not None:
            model.save_lora(args.lora_output_dir)
            print("saved_lora:", str(args.lora_output_dir))
        return

    raise NotImplementedError(
        "FLUX.2 Klein loaded successfully. Use --prepare_lora to initialize LoRA, "
        "then connect the payload to the upstream FLUX.2 training loss/optimizer."
    )


if __name__ == "__main__":
    main()
