from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2 as cv
import numpy as np


@dataclass
class ReferencePack:
    images: List[np.ndarray]
    patch_ids: List[int]
    positions: List[tuple[int, int]]


class ReferenceSelector:
    """Packs references with Cobra-style localized reusable positions.

    This does not implement Cobra's sparse attention itself. It prepares
    references so a DiT/VLM adapter can reuse a small positional coordinate
    range while still receiving multiple reference images per line-art patch.
    """

    def __init__(self, max_refs: int = 8, refs_per_patch: int = 2, image_size: int = 512):
        self.max_refs = max_refs
        self.refs_per_patch = refs_per_patch
        self.image_size = image_size

    def from_paths(self, lineart_rgb: np.ndarray, reference_paths: Sequence[Path]) -> ReferencePack:
        refs = [self._read_rgb(path) for path in reference_paths[: self.max_refs]]
        return self.pack(lineart_rgb, refs)

    def pack(self, lineart_rgb: np.ndarray, references: Sequence[np.ndarray]) -> ReferencePack:
        if not references:
            return ReferencePack(images=[], patch_ids=[], positions=[])

        line_features = self._quadrant_features(lineart_rgb)
        ref_features = np.asarray([self._feature(ref) for ref in references])
        selected: List[np.ndarray] = []
        patch_ids: List[int] = []
        positions: List[tuple[int, int]] = []

        for patch_id, feat in enumerate(line_features):
            scores = ref_features @ feat
            order = np.argsort(-scores)[: self.refs_per_patch]
            for ref_index in order:
                selected.append(self._resize_square(references[int(ref_index)]))
                patch_ids.append(patch_id)
                positions.append(self._reusable_position(patch_id))
                if len(selected) >= self.max_refs:
                    return ReferencePack(selected, patch_ids, positions)

        return ReferencePack(selected, patch_ids, positions)

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        img = cv.imread(str(path), cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

    def _resize_square(self, img: np.ndarray) -> np.ndarray:
        return cv.resize(img, (self.image_size, self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _feature(img: np.ndarray) -> np.ndarray:
        small = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
        hist = []
        for ch in range(3):
            h = cv.calcHist([small], [ch], None, [16], [0, 256]).flatten()
            hist.append(h)
        feat = np.concatenate(hist).astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / max(norm, 1e-6)

    def _quadrant_features(self, lineart_rgb: np.ndarray) -> np.ndarray:
        h, w = lineart_rgb.shape[:2]
        patches = [
            lineart_rgb[: h // 2, : w // 2],
            lineart_rgb[: h // 2, w // 2 :],
            lineart_rgb[h // 2 :, : w // 2],
            lineart_rgb[h // 2 :, w // 2 :],
        ]
        return np.asarray([self._feature(255 - patch) for patch in patches])

    @staticmethod
    def _reusable_position(patch_id: int) -> tuple[int, int]:
        return [(0, 0), (1, 0), (0, 1), (1, 1)][patch_id]


def grouped_reference_paths(image_path: Path, reference_root: Path, extensions: Iterable[str] = (".jpg", ".png", ".jpeg")) -> List[Path]:
    group = reference_root / image_path.stem
    if not group.exists():
        return []
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(group.glob(f"*{ext}"))
    return sorted(paths)
