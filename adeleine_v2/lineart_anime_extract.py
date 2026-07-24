from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator

import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm

from .openniji import OpenNijiParquetColorizationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ControlNet/Annotators anime lineart cache")
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--extension", default=".png")
    parser.add_argument("--openniji", action="store_true")
    parser.add_argument("--openniji_parquet_root", type=Path)
    parser.add_argument("--openniji_parquet_pattern", default="data/train-00000*.parquet")
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--output_dir", type=Path, default=Path("/data/shasegawa/adeleine/openniji/lineart_anime"))
    parser.add_argument("--detect_resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_detector(device: str):
    try:
        import importlib.util
        import site
        import sys
        import types
    except ImportError as exc:
        raise ImportError("Python importlib/site are required to load LineartAnimeDetector") from exc

    roots = []
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        root = Path(base) / "controlnet_aux"
        if root.exists():
            roots.append(root)
    if not roots:
        raise ImportError("Install controlnet-aux to use LineartAnimeDetector")
    root = roots[0]

    pkg = sys.modules.get("controlnet_aux")
    if pkg is None or not hasattr(pkg, "__path__"):
        pkg = types.ModuleType("controlnet_aux")
        pkg.__path__ = [str(root)]
        sys.modules["controlnet_aux"] = pkg

    util_name = "controlnet_aux.util"
    if util_name not in sys.modules:
        util_spec = importlib.util.spec_from_file_location(util_name, root / "util.py")
        if util_spec is None or util_spec.loader is None:
            raise ImportError(f"Cannot load {root / 'util.py'}")
        util_module = importlib.util.module_from_spec(util_spec)
        sys.modules[util_name] = util_module
        util_spec.loader.exec_module(util_module)

    module_name = "controlnet_aux.lineart_anime"
    module_path = root / "lineart_anime" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
        submodule_search_locations=[str(module_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    detector = module.LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    if hasattr(detector, "model") and hasattr(detector.model, "to"):
        detector.model.to(device)
    return detector


def ensure_black_on_white(rgb: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    if float(np.mean(gray)) < 127.0:
        gray = 255 - gray
    return cv.cvtColor(gray, cv.COLOR_GRAY2RGB)


def iter_folder(input_dir: Path, extension: str) -> Iterator[tuple[str, np.ndarray]]:
    for path in sorted(input_dir.glob(f"**/*{extension}")):
        image = Image.open(path).convert("RGB")
        yield path.name, np.asarray(image)


def iter_openniji(args: argparse.Namespace) -> Iterator[tuple[str, np.ndarray]]:
    dataset = OpenNijiParquetColorizationDataset(
        parquet_root=args.openniji_parquet_root,
        parquet_pattern=args.openniji_parquet_pattern,
        image_size=args.detect_resolution,
        max_records=args.max_records,
    )
    for record in dataset.records:
        row = dataset._read_record(record)
        bgr = dataset._decode_image(row["image"]["bytes"])
        bgr = dataset._resize_square(bgr)
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        digest = hashlib.sha256(record.url.encode("utf-8")).hexdigest()
        yield f"{digest}.png", rgb


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = load_detector(args.device)
    samples = iter_openniji(args) if args.openniji else iter_folder(args.input_dir, args.extension)
    for name, rgb in tqdm(samples):
        out_path = args.output_dir / name
        if out_path.exists():
            continue
        pil = Image.fromarray(rgb)
        line = detector(pil, detect_resolution=args.detect_resolution, image_resolution=args.detect_resolution, output_type="pil")
        line_rgb = ensure_black_on_white(np.asarray(line.convert("RGB")))
        Image.fromarray(line_rgb).save(out_path)
    print({"output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
