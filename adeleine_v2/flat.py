from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np


@dataclass
class FlatOutputConfig:
    enabled: bool = True
    region_variance_weight: float = 0.25
    palette_weight: float = 0.10
    dacon_command: Optional[str] = None


class FlatPostprocess:
    """Small paint-bucket-ready fallback while DACoN integration is external."""

    def __init__(self, colors: int = 32):
        self.colors = colors

    def __call__(self, image_rgb: np.ndarray, lineart_rgb: np.ndarray) -> np.ndarray:
        flat = self._quantize(image_rgb)
        line_mask = np.mean(lineart_rgb, axis=2) < 180
        flat[line_mask] = lineart_rgb[line_mask]
        return flat

    def _quantize(self, image_rgb: np.ndarray) -> np.ndarray:
        pixels = image_rgb.reshape((-1, 3)).astype(np.float32)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 24, 0.5)
        _, labels, centers = cv.kmeans(pixels, self.colors, None, criteria, 1, cv.KMEANS_PP_CENTERS)
        centers = np.clip(centers, 0, 255).astype(np.uint8)
        return centers[labels.flatten()].reshape(image_rgb.shape)


def dacon_command_available(command: Optional[str]) -> bool:
    return bool(command and Path(command.split()[0]).exists())
