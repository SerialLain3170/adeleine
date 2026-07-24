from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple

import cv2 as cv
import numpy as np

LineMethod = Literal["xdog", "pencil", "digital", "lineart_anime", "canny", "blend"]
AtariMode = Literal["line", "dot", "mixed"]


@dataclass
class LineArtPaths:
    pencil_dir: Optional[Path] = None
    digital_dir: Optional[Path] = None
    anime_dir: Optional[Path] = None


@dataclass
class LineArtConfig:
    methods: tuple[LineMethod, ...] = ("xdog", "pencil", "digital", "lineart_anime", "blend")
    min_dark_ratio: float = 0.015
    max_dark_ratio: float = 0.55
    blend: float = 0.5
    morphology_prob: float = 0.5
    color_variant_prob: float = 0.5
    line_color_max: int = 30
    line_threshold: int = 200


@dataclass
class AtariHintConfig:
    mode: AtariMode = "mixed"
    line_min_hints: int = 8
    line_max_hints: int = 20
    dot_min_hints: int = 55
    dot_max_hints: int = 60
    dot_max_patch_size: int = 13
    dot_uniform: bool = True
    mixed_dot_prob: float = 0.5
    line_region_aware: bool = True
    line_min_length: int = 12
    line_max_length: int = 56
    line_min_thickness: int = 2
    line_max_thickness: int = 5
    max_region_variance: float = 0.015
    line_white_threshold: int = 190


def available_line_methods(methods: Iterable[str], paths: LineArtPaths) -> list[str]:
    ok = []
    for method in methods:
        if method == "pencil" and paths.pencil_dir is None:
            continue
        if method == "digital" and paths.digital_dir is None:
            continue
        if method == "lineart_anime" and paths.anime_dir is None:
            continue
        if method == "blend" and paths.pencil_dir is None:
            continue
        ok.append(method)
    if not ok:
        raise ValueError("No usable line-art methods are available")
    return ok


class LegacyLineArtAugmentor:
    """Line augmentation matching the repository README and legacy processors."""

    def __init__(self, config: LineArtConfig | None = None, paths: LineArtPaths | None = None):
        self.config = config or LineArtConfig()
        self.paths = paths or LineArtPaths()
        self.methods = available_line_methods(self.config.methods, self.paths)

    def __call__(self, image_path: Path, color_bgr: Optional[np.ndarray] = None) -> np.ndarray:
        method = str(np.random.choice(self.methods))
        if color_bgr is None:
            color_bgr = cv.imread(str(image_path), cv.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(image_path)

        if method == "xdog":
            line = self._xdog(color_bgr)
        elif method == "pencil":
            line = self._read_precomputed_or_fallback(image_path, self.paths.pencil_dir, color_bgr)
        elif method == "digital":
            line = self._read_precomputed_or_fallback(image_path, self.paths.digital_dir, color_bgr)
        elif method == "lineart_anime":
            line = self._read_precomputed_or_fallback(image_path, self.paths.anime_dir, color_bgr)
        elif method == "canny":
            line = self._canny(color_bgr)
        elif method == "blend":
            xdog = self._xdog(color_bgr)
            pencil = self._read_precomputed_or_fallback(image_path, self.paths.pencil_dir, color_bgr)
            pencil = self._add_intensity(pencil, 1.4)
            xdog_blur = cv.GaussianBlur(xdog, (5, 5), 1)
            xdog_blur = cv.addWeighted(xdog_blur, 0.75, xdog, 0.25, 0)
            line = cv.addWeighted(xdog_blur, self.config.blend, pencil, 1.0 - self.config.blend, 0)
            line = self._add_intensity(line, 1.0 / 1.5)
        else:
            raise ValueError(f"Unknown line-art method: {method}")

        line = self._ensure_black_on_white(line)
        if not self._passes_density_gate(line):
            line = self._canny(color_bgr)
        line = self._binarize_black_on_white(line)
        line = self._maybe_morphology(line)
        line = self._maybe_color_variant(line)
        return self._ensure_black_on_white(line)

    @staticmethod
    def _xdog(color_bgr: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(color_bgr, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sigma = float(np.random.choice([0.3, 0.4, 0.5]))
        sigma_large = sigma * 4.5
        g_small = cv.GaussianBlur(gray, (0, 0), sigma)
        g_large = cv.GaussianBlur(gray, (0, 0), sigma_large)
        sharp = (1.0 + 19.0) * g_small - 19.0 * g_large
        response = gray * sharp
        line = np.ones_like(response, dtype=np.float32)
        dark = response < 0.01
        line[dark] = 1.0 + np.tanh(1e9 * (response[dark] - 0.01))
        line = np.clip(line * 255.0, 0, 255).astype(np.uint8)
        return cv.cvtColor(line, cv.COLOR_GRAY2BGR)

    @staticmethod
    def _canny(color_bgr: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(color_bgr, cv.COLOR_BGR2GRAY)
        gray = cv.bilateralFilter(gray, 5, 35, 35)
        edges = cv.Canny(gray, 60, 160)
        line = 255 - edges
        return cv.cvtColor(line, cv.COLOR_GRAY2BGR)

    @staticmethod
    def _read_precomputed(image_path: Path, root: Optional[Path]) -> np.ndarray:
        if root is None:
            raise ValueError(f"Precomputed line directory is required for {image_path.name}")
        path = root / image_path.name
        if not path.exists():
            raise FileNotFoundError(path)
        line = cv.imread(str(path), cv.IMREAD_COLOR)
        if line is None:
            raise FileNotFoundError(path)
        return line

    def _read_precomputed_or_fallback(self, image_path: Path, root: Optional[Path], color_bgr: np.ndarray) -> np.ndarray:
        try:
            return self._read_precomputed(image_path, root)
        except (FileNotFoundError, ValueError):
            return self._xdog(color_bgr)

    @staticmethod
    def _add_intensity(img: np.ndarray, intensity: float) -> np.ndarray:
        img_f = img.astype(np.float32)
        const = 255.0 ** (1.0 - intensity)
        return np.clip(const * (img_f**intensity), 0, 255).astype(np.uint8)

    @staticmethod
    def _ensure_black_on_white(line: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(line, cv.COLOR_BGR2GRAY) if line.ndim == 3 else line
        if float(np.mean(gray)) < 127.0:
            line = 255 - line
        return line

    def _binarize_black_on_white(self, line: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(line, cv.COLOR_BGR2GRAY) if line.ndim == 3 else line
        out = np.full_like(gray, 255, dtype=np.uint8)
        out[gray < self.config.line_threshold] = 0
        return cv.cvtColor(out, cv.COLOR_GRAY2BGR)

    def _passes_density_gate(self, line: np.ndarray) -> bool:
        gray = cv.cvtColor(line, cv.COLOR_BGR2GRAY) if line.ndim == 3 else line
        dark_ratio = float(np.mean(gray < 180))
        return self.config.min_dark_ratio <= dark_ratio <= self.config.max_dark_ratio

    def _maybe_morphology(self, line: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.config.morphology_prob:
            return line
        kernel = np.ones((5, 5), dtype=np.uint8)
        candidate = cv.erode(line, kernel, iterations=1) if np.random.randint(2) else cv.dilate(line, kernel, iterations=1)
        return candidate if self._passes_density_gate(candidate) else line

    def _maybe_color_variant(self, line: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.config.color_variant_prob:
            return line
        out = line.copy()
        value = int(np.random.randint(self.config.line_color_max + 1))
        out[out < self.config.line_threshold] = value
        return out


class LegacyAtariHintGenerator:
    """Atari hint creation refactored from legacy `atari_whitebox` and `atari_userhint_v2`."""

    def __init__(self, config: AtariHintConfig | None = None):
        self.config = config or AtariHintConfig()

    def __call__(self, color_rgb: np.ndarray, lineart_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mode = self.config.mode
        if mode == "mixed":
            mode = "dot" if np.random.random() < self.config.mixed_dot_prob else "line"
        if mode == "line":
            return self._line_hint(color_rgb, lineart_rgb)
        if mode == "dot":
            return self._dot_hint(color_rgb, lineart_rgb)
        raise ValueError(f"Unknown Atari hint mode: {mode}")

    def _line_hint(self, color_rgb: np.ndarray, lineart_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        size = color_rgb.shape[0]
        hint = lineart_rgb.copy()
        alpha = np.zeros((color_rgb.shape[0], color_rgb.shape[1], 1), dtype=np.uint8)
        repeat = int(np.random.randint(self.config.line_min_hints, self.config.line_max_hints + 1))
        for _ in range(repeat):
            if self.config.line_region_aware:
                hint, alpha = self._draw_region_line(hint, color_rgb, lineart_rgb, alpha)
            else:
                hint, alpha = self._draw_line(hint, color_rgb, alpha, size)
        return hint, alpha

    def _draw_region_line(self, hint: np.ndarray, color: np.ndarray, lineart: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = color.shape[:2]
        y, x = self._sample_region_point(color, lineart)
        length = int(np.random.randint(self.config.line_min_length, self.config.line_max_length + 1))
        thickness = int(np.random.randint(self.config.line_min_thickness, self.config.line_max_thickness + 1))
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        dx = int(np.cos(angle) * length / 2)
        dy = int(np.sin(angle) * length / 2)
        p0 = (int(np.clip(x - dx, 0, w - 1)), int(np.clip(y - dy, 0, h - 1)))
        p1 = (int(np.clip(x + dx, 0, w - 1)), int(np.clip(y + dy, 0, h - 1)))
        local = color[max(0, y - 2) : min(h, y + 3), max(0, x - 2) : min(w, x + 3)]
        rgb = np.mean(local, axis=(0, 1)).astype(np.uint8).tolist()
        cv.line(hint, p0, p1, rgb, thickness=thickness, lineType=cv.LINE_AA)
        cv.line(alpha, p0, p1, 255, thickness=thickness, lineType=cv.LINE_AA)
        return hint, alpha

    def _sample_region_point(self, color: np.ndarray, lineart: np.ndarray) -> tuple[int, int]:
        h, w = color.shape[:2]
        patch = 9
        radius = patch // 2
        line_gray = cv.cvtColor(lineart, cv.COLOR_RGB2GRAY) if lineart.ndim == 3 else lineart
        for _ in range(256):
            y = int(np.random.randint(radius, max(radius + 1, h - radius)))
            x = int(np.random.randint(radius, max(radius + 1, w - radius)))
            c = color[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float32) / 255.0
            l = line_gray[y - radius : y + radius + 1, x - radius : x + radius + 1]
            if float(np.var(c)) <= self.config.max_region_variance and float(np.mean(l)) >= self.config.line_white_threshold:
                return y, x
        return int(np.random.randint(0, h)), int(np.random.randint(0, w))

    @staticmethod
    def _draw_line(hint: np.ndarray, color: np.ndarray, alpha: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        choice = str(np.random.choice(["width", "height", "diag"]))
        if choice == "width":
            rnd_height = int(np.random.randint(4, 8))
            rnd_width = int(np.random.randint(4, 64))
            rnd1 = int(np.random.randint(size - rnd_height))
            rnd2 = int(np.random.randint(size - rnd_width))
            hint[rnd1 : rnd1 + rnd_height, rnd2 : rnd2 + rnd_width] = color[rnd1 : rnd1 + rnd_height, rnd2 : rnd2 + rnd_width]
            alpha[rnd1 : rnd1 + rnd_height, rnd2 : rnd2 + rnd_width] = 255
        elif choice == "height":
            rnd_height = int(np.random.randint(4, 64))
            rnd_width = int(np.random.randint(4, 8))
            rnd1 = int(np.random.randint(size - rnd_height))
            rnd2 = int(np.random.randint(size - rnd_width))
            hint[rnd1 : rnd1 + rnd_height, rnd2 : rnd2 + rnd_width] = color[rnd1 : rnd1 + rnd_height, rnd2 : rnd2 + rnd_width]
            alpha[rnd1 : rnd1 + rnd_height, rnd2 : rnd2 + rnd_width] = 255
        else:
            rnd_height = int(np.random.randint(4, 8))
            rnd_width = int(np.random.randint(4, 64))
            rnd1 = int(np.random.randint(size - rnd_height - rnd_width - 1))
            rnd2 = int(np.random.randint(size - rnd_width))
            for index in range(rnd_width):
                hint[rnd1 + index : rnd1 + rnd_height + index, rnd2 + index] = color[
                    rnd1 + index : rnd1 + rnd_height + index, rnd2 + index
                ]
                alpha[rnd1 + index : rnd1 + rnd_height + index, rnd2 + index] = 255
        return hint, alpha

    def _dot_hint(self, color_rgb: np.ndarray, lineart_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = color_rgb.shape[:2]
        hint = np.full_like(lineart_rgb, 255, dtype=np.uint8)
        alpha = np.zeros((h, w, 1), dtype=np.uint8)
        repeat = int(np.random.randint(self.config.dot_min_hints, self.config.dot_max_hints + 1))
        for _ in range(repeat):
            hint, alpha = self._draw_dot(hint, color_rgb, alpha)
        return hint, alpha

    def _draw_dot(self, hint: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = color.shape[:2]
        patch_size = int(np.random.randint(1, self.config.dot_max_patch_size))
        if self.config.dot_uniform:
            rnd_h = int(np.random.randint(0, max(1, h - patch_size)))
            rnd_w = int(np.random.randint(0, max(1, w - patch_size)))
        else:
            rnd_h = int(np.random.normal(loc=h // 2, scale=max(1, h // 4)))
            rnd_w = int(np.random.normal(loc=w // 2, scale=max(1, w // 4)))
            rnd_h = min(max(rnd_h, 0), max(0, h - patch_size - 1))
            rnd_w = min(max(rnd_w, 0), max(0, w - patch_size - 1))
        patch = color[rnd_h : rnd_h + patch_size, rnd_w : rnd_w + patch_size]
        patch = np.mean(patch, axis=(0, 1), keepdims=True).astype(np.uint8)
        hint[rnd_h : rnd_h + patch_size, rnd_w : rnd_w + patch_size] = np.tile(patch, (patch_size, patch_size, 1))
        alpha[rnd_h : rnd_h + patch_size, rnd_w : rnd_w + patch_size] = 255
        return hint, alpha
