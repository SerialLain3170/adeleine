from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator, Optional

import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

from .openniji import OpenNijiParquetColorizationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and cache SketchKeras-style pencil line art")
    parser.add_argument("--input_dir", type=Path, help="Folder of RGB images. Filenames are preserved in output_dir.")
    parser.add_argument("--extension", default=".png")
    parser.add_argument("--openniji", action="store_true", help="Read images from OpenNiji Parquet instead of input_dir.")
    parser.add_argument("--openniji_parquet_root", type=Path)
    parser.add_argument("--openniji_repo_id", default="all")
    parser.add_argument("--openniji_parquet_pattern", default="data/train-00000*.parquet")
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--output_dir", type=Path, default=Path("/data/shasegawa/adeleine/openniji/sketchkeras"))
    parser.add_argument("--model_path", type=Path, default=Path("/data/shasegawa/adeleine/models/sketchkeras/mod.h5"), help="Keras .h5/.keras SketchKeras model path.")
    parser.add_argument("--download_model", action="store_true", help="Download lllyasviel/sketchKeras mod.h5 when model_path is missing.")
    parser.add_argument("--model_url", default="https://github.com/lllyasviel/sketchKeras/releases/download/0.1/mod.h5")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def maybe_download_model(path: Path, url: str) -> None:
    if path.exists():
        return
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with tmp.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(path)


def load_model(path: Path, download: bool = False, url: str = ""):
    if download:
        maybe_download_model(path, url)
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Install tensorflow to run SketchKeras extraction") from exc
    return tf.keras.models.load_model(path, compile=False)


def resize_preserve_aspect(rgb: np.ndarray, image_size: int) -> tuple[np.ndarray, int, int]:
    height, width = rgb.shape[:2]
    if width > height:
        new_width = image_size
        new_height = int(image_size / float(width) * float(height))
    else:
        new_width = int(image_size / float(height) * float(width))
        new_height = image_size
    image = cv.resize(rgb, (new_width, new_height), interpolation=cv.INTER_AREA)
    return image, new_width, new_height


def get_light_map_single(channel: np.ndarray) -> np.ndarray:
    gray = channel.astype(np.float32)
    blur = cv.GaussianBlur(gray, (0, 0), 3)
    high_pass = gray.astype(np.int32) - blur.astype(np.int32)
    return high_pass.astype(np.float32) / 128.0


def preprocess_rgb(rgb: np.ndarray, image_size: int) -> tuple[np.ndarray, int, int]:
    image, new_width, new_height = resize_preserve_aspect(rgb, image_size)
    chw = image.transpose((2, 0, 1))
    light_map = np.zeros(chw.shape, dtype=np.float32)
    for channel in range(3):
        light_map[channel] = get_light_map_single(chw[channel])
    denom = float(np.max(light_map))
    if denom > 1e-8:
        light_map = light_map / denom
    padded = np.zeros((3, image_size, image_size, 1), dtype=np.float32)
    padded[:, :new_height, :new_width, 0] = light_map
    return padded, new_width, new_height


def active_image(line_prob: np.ndarray, threshold: float | None = None) -> np.ndarray:
    mat = line_prob.astype(np.float32).copy()
    if threshold is not None:
        mat[mat < threshold] = 0.0
    mat = -mat + 1.0
    mat = mat * 255.0
    mat = np.clip(mat, 0, 255).astype(np.uint8)
    return ndimage.median_filter(mat, 1)


def postprocess_line(pred: np.ndarray, new_width: int, new_height: int, variant: str = "pured") -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 4:
        raise ValueError(f"SketchKeras prediction must be NHWC, got {pred.shape}")
    channels = pred[:3, :new_height, :new_width, 0]
    line_prob = np.amax(channels, axis=0)
    threshold = 0.18 if variant == "pured" else 0.10 if variant == "enhanced" else None
    line = active_image(line_prob, threshold=threshold)
    return cv.cvtColor(line, cv.COLOR_GRAY2RGB)


def iter_folder(input_dir: Path, extension: str) -> Iterator[tuple[str, np.ndarray]]:
    for path in sorted(input_dir.glob(f"**/*{extension}")):
        bgr = cv.imread(str(path), cv.IMREAD_COLOR)
        if bgr is None:
            continue
        yield path.name, cv.cvtColor(bgr, cv.COLOR_BGR2RGB)


def iter_openniji(args: argparse.Namespace) -> Iterator[tuple[str, np.ndarray]]:
    dataset = OpenNijiParquetColorizationDataset(
        parquet_root=args.openniji_parquet_root,
        repo_id=args.openniji_repo_id,
        parquet_pattern=args.openniji_parquet_pattern,
        image_size=args.image_size,
        max_records=args.max_records,
        line_methods=("xdog",),
        reference_policy="none",
    )
    skipped = 0
    for record in dataset.records:
        try:
            row = dataset._read_record(record)
            image_bytes = row.get("image", {}).get("bytes")
            if not image_bytes:
                skipped += 1
                continue
            bgr = dataset._decode_image(image_bytes)
        except Exception as exc:
            skipped += 1
            if skipped <= 10 or skipped % 100 == 0:
                print({"skipped": skipped, "url": record.url, "error": str(exc)}, flush=True)
            continue
        rgb = cv.cvtColor(dataset._resize_square(bgr), cv.COLOR_BGR2RGB)
        digest = hashlib.sha256(record.url.encode("utf-8")).hexdigest()
        yield f"{digest}.png", rgb


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_path, download=args.download_model, url=args.model_url)
    samples = iter_openniji(args) if args.openniji else iter_folder(args.input_dir, args.extension)
    for name, rgb in tqdm(samples):
        out_path = args.output_dir / name
        if out_path.exists():
            continue
        model_input, new_width, new_height = preprocess_rgb(rgb, args.image_size)
        pred = model.predict(model_input, verbose=0)
        line = postprocess_line(pred, new_width, new_height)
        Image.fromarray(line).save(out_path)
    print({"output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
