from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image


@dataclass
class ReferenceConditioningConfig:
    mode: str = "none"
    tag_cache_root: Optional[Path] = None
    mask_root: Optional[Path] = None
    mask_fallback: str = "grabcut"
    skytnt_repo: Optional[Path] = None
    skytnt_model_id: str = "skytnt/anime-seg"
    skytnt_ckpt: Optional[Path] = None
    skytnt_net: str = "isnet_is"
    skytnt_image_size: int = 1024
    skytnt_device: str = "cuda:0"
    skytnt_fp32: bool = False
    skytnt_local_files_only: bool = False
    cache_generated_masks: bool = True
    wd_tagger_model: Optional[Path] = None
    wd_tagger_labels: Optional[Path] = None
    wd_tagger_threshold: float = 0.35
    wd_tagger_character_threshold: float = 0.85
    wd_tagger_max_tokens: int = 32
    max_tags: int = 24
    foreground_threshold: float = 0.5
    # Masks covering less than min are "no character found": the whole reference is the background layer.
    # Masks covering more than max are "all character": the whole reference is the foreground layer.
    # Splitting either would give a blank layer plus a near-copy of the image.
    min_foreground_coverage: float = 0.02
    max_foreground_coverage: float = 0.98


@dataclass
class ReferenceConditioningResult:
    foregrounds: list[np.ndarray]
    backgrounds: list[np.ndarray]
    masks: list[np.ndarray]
    tags: list[str]
    wd_indices: list[np.ndarray]
    wd_scores: list[np.ndarray]


@dataclass
class WDTaggerResult:
    tags: list[str]
    indices: np.ndarray
    scores: np.ndarray


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 4:
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = arr[..., :3].astype(np.float32)
        arr = rgb * alpha + 255.0 * (1.0 - alpha)
    else:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def reference_digest(image: np.ndarray) -> str:
    arr = _as_uint8_rgb(image)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _fallback_foreground_mask(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.52
    rx = max(width * 0.42, 1.0)
    ry = max(height * 0.48, 1.0)
    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = (dist <= 1.0).astype(np.float32)
    return mask


def estimate_foreground_mask(image: np.ndarray) -> np.ndarray:
    rgb = _as_uint8_rgb(image)
    h, w = rgb.shape[:2]
    try:
        import cv2

        # GrabCut with a central rectangle gives a useful foreground prior for
        # character-centric anime references while still allowing scenery-only
        # references to fall back to a soft central mask.
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        rect = (
            max(1, int(w * 0.06)),
            max(1, int(h * 0.03)),
            max(2, int(w * 0.88)),
            max(2, int(h * 0.92)),
        )
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD).astype(np.float32)
        ratio = float(fg.mean())
        if ratio < 0.03 or ratio > 0.95:
            fg = _fallback_foreground_mask(h, w)
        kernel = np.ones((5, 5), np.uint8)
        fg8 = (fg * 255).astype(np.uint8)
        fg8 = cv2.morphologyEx(fg8, cv2.MORPH_OPEN, kernel)
        fg8 = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, kernel)
        fg = cv2.GaussianBlur(fg8, (0, 0), 1.2).astype(np.float32) / 255.0
        return np.clip(fg, 0.0, 1.0)
    except Exception:
        return _fallback_foreground_mask(h, w)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Return a uint8 HxWx1 mask in 0..255 from a 0..1 or 0..255, 2D or 3D mask."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask_f = arr.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f = mask_f / 255.0
    return np.rint(np.clip(mask_f, 0.0, 1.0)[..., None] * 255.0).astype(np.uint8)


def mask_coverage(mask: np.ndarray) -> float:
    return float((normalize_mask(mask) > 127).mean())


def split_reference_layers(image: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = _as_uint8_rgb(image)
    if mask is None:
        mask_f = estimate_foreground_mask(rgb)
    else:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask_f = mask_arr.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f = mask_f / 255.0
        mask_f = np.clip(mask_f, 0.0, 1.0)
    m = mask_f[..., None]
    white = np.full_like(rgb, 255)
    fg = np.clip(rgb.astype(np.float32) * m + white.astype(np.float32) * (1.0 - m), 0, 255).astype(np.uint8)
    # Soft SkyTNT edges carry foreground colour down to low alpha; remove them too so no halo survives inpainting.
    hole = (mask_f > 0.1).astype(np.uint8) * 255
    hole_area = int(hole.sum())
    if 0 < hole_area < hole.size * 255:
        try:
            import cv2

            # Build a background-looking reference: use the mask to remove the
            # foreground, then fill the removed area instead of showing the mask
            # as an explicit white silhouette.
            radius = max(2, round(min(rgb.shape[:2]) * 0.01))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            inpaint_mask = cv2.dilate(hole, kernel, iterations=1)
            bg = cv2.inpaint(rgb, inpaint_mask, 3, cv2.INPAINT_TELEA)
            bg = np.asarray(bg, dtype=np.uint8)
        except Exception:
            bg_pixels = rgb[mask_f < 0.2]
            if bg_pixels.size:
                fill = bg_pixels.reshape(-1, 3).mean(axis=0)
            else:
                fill = np.array([255.0, 255.0, 255.0], dtype=np.float32)
            bg_f = rgb.astype(np.float32) * (1.0 - m) + fill.reshape(1, 1, 3) * m
            bg = np.clip(bg_f, 0, 255).astype(np.uint8)
    else:
        bg = rgb.copy()
    return fg, bg, (mask_f[..., None] * 255.0).astype(np.uint8)


def _read_label_rows(path: Path) -> tuple[list[str], list[int]]:
    with path.open("r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames:
            name_key = "name" if "name" in reader.fieldnames else reader.fieldnames[0]
            category_key = "category" if "category" in reader.fieldnames else None
            names: list[str] = []
            categories: list[int] = []
            for row in reader:
                name = str(row.get(name_key, "")).strip()
                if not name:
                    continue
                names.append(name)
                raw_category = row.get(category_key, 0) if category_key else 0
                try:
                    categories.append(int(raw_category))
                except Exception:
                    categories.append(0)
            return names, categories
        f.seek(0)
        names = [line.strip().split(",")[0] for line in f if line.strip()]
        return names, [0] * len(names)


class WDTaggerONNX:
    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        threshold: float = 0.35,
        max_tags: int = 24,
        character_threshold: float = 0.85,
        max_tokens: int = 32,
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.output_name = self.session.get_outputs()[0].name
        self.labels, self.categories = _read_label_rows(labels_path)
        self.threshold = threshold
        self.character_threshold = character_threshold
        self.max_tags = max_tags
        self.max_tokens = max_tokens
        self.rating_categories = {9}
        self.character_categories = {4}

    def _input_size(self) -> int:
        shape = list(self.input.shape)
        numeric = [int(x) for x in shape if isinstance(x, int)]
        for value in numeric:
            if value not in (1, 3) and value > 16:
                return value
        return 448

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        rgb = _as_uint8_rgb(image)
        size = self._input_size()
        h, w = rgb.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), 255, dtype=np.uint8)
        top = (side - h) // 2
        left = (side - w) // 2
        canvas[top : top + h, left : left + w] = rgb
        arr = np.asarray(Image.fromarray(canvas).resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)
        # SmilingWolf WD ONNX exports expect BGR channel order and 0..255 float inputs.
        return np.ascontiguousarray(arr[:, :, ::-1])

    def predict(self, image: np.ndarray) -> WDTaggerResult:
        arr = self._preprocess(image)
        shape = list(self.input.shape)
        if len(shape) == 4 and shape[-1] == 3:
            batch = arr[None, ...]
        else:
            batch = np.transpose(arr, (2, 0, 1))[None, ...]
        scores = self.session.run([self.output_name], {self.input.name: batch.astype(np.float32)})[0]
        scores = np.asarray(scores).reshape(-1)
        order = np.argsort(scores)[::-1]
        tags: list[str] = []
        indices: list[int] = []
        values: list[float] = []
        for idx in order:
            if idx >= len(self.labels):
                continue
            category = self.categories[idx] if idx < len(self.categories) else 0
            if category in self.rating_categories:
                continue
            score = float(scores[idx])
            threshold = self.character_threshold if category in self.character_categories else self.threshold
            if score < threshold:
                continue
            tag = self.labels[idx].replace("_", " ").strip()
            if tag:
                if len(tags) < self.max_tags:
                    tags.append(tag)
                indices.append(int(idx))
                values.append(score)
            if len(indices) >= self.max_tokens:
                break
        return WDTaggerResult(
            tags=tags,
            indices=np.asarray(indices, dtype=np.int64),
            scores=np.asarray(values, dtype=np.float32),
        )

    def __call__(self, image: np.ndarray) -> list[str]:
        return self.predict(image).tags


def _color_name(pixels: np.ndarray) -> Optional[str]:
    if pixels.size == 0:
        return None
    px = pixels.reshape(-1, 3).astype(np.float32) / 255.0
    if len(px) > 4096:
        px = px[np.linspace(0, len(px) - 1, 4096).astype(np.int64)]
    mean = px.mean(axis=0)
    mx = float(mean.max())
    mn = float(mean.min())
    sat = mx - mn
    val = mx
    if val < 0.16:
        return "black"
    if sat < 0.08:
        if val > 0.82:
            return "white"
        if val > 0.48:
            return "gray"
        return "dark gray"
    r, g, b = mean
    hue = np.degrees(np.arctan2(np.sqrt(3.0) * (g - b), 2.0 * r - g - b)) % 360.0
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 38:
        return "orange"
    if hue < 68:
        return "yellow"
    if hue < 155:
        return "green"
    if hue < 195:
        return "cyan"
    if hue < 250:
        return "blue"
    if hue < 292:
        return "purple"
    if hue < 345:
        return "pink"
    return None


def estimate_reference_tags(image: np.ndarray, foreground_mask: Optional[np.ndarray] = None, max_tags: int = 24) -> list[str]:
    rgb = _as_uint8_rgb(image)
    h, _w = rgb.shape[:2]
    if foreground_mask is None:
        mask = estimate_foreground_mask(rgb)
    else:
        mask = np.asarray(foreground_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
    fg = mask > 0.45
    bg = mask < 0.20
    upper = np.zeros_like(fg)
    upper[: int(h * 0.52), :] = True
    lower = np.zeros_like(fg)
    lower[int(h * 0.42) :, :] = True

    tags: list[str] = []
    for label, selector in (
        ("hair", fg & upper),
        ("clothing", fg & lower),
        ("foreground", fg),
        ("background", bg),
    ):
        color = _color_name(rgb[selector])
        if color:
            tags.append(f"{color} {label}")
    if bg.any():
        bg_val = float(rgb[bg].mean()) / 255.0
        if bg_val > 0.78:
            tags.append("bright background")
        elif bg_val < 0.28:
            tags.append("dark background")
    seen = set()
    unique: list[str] = []
    for tag in tags:
        if tag not in seen:
            unique.append(tag)
            seen.add(tag)
        if len(unique) >= max_tags:
            break
    return unique


class ReferenceConditioningBuilder:
    def __init__(self, config: ReferenceConditioningConfig) -> None:
        self.config = config
        self.config.mode = self.config.mode or "none"
        if self.config.tag_cache_root is not None:
            self.config.tag_cache_root.mkdir(parents=True, exist_ok=True)
        self._skytnt = None
        self._wd_tagger = None
        if self.config.wd_tagger_model and self.config.wd_tagger_labels:
            model = Path(self.config.wd_tagger_model)
            labels = Path(self.config.wd_tagger_labels)
            if model.exists() and labels.exists():
                self._wd_tagger = WDTaggerONNX(
                    model,
                    labels,
                    self.config.wd_tagger_threshold,
                    self.config.max_tags,
                    self.config.wd_tagger_character_threshold,
                    self.config.wd_tagger_max_tokens,
                )

    @property
    def enabled(self) -> bool:
        return self.config.mode in {"split", "split_tags", "split_wd", "split_wd_tags"}

    @property
    def wants_tags(self) -> bool:
        return self.config.mode in {"split_tags", "split_wd_tags"}

    @property
    def wants_wd_tokens(self) -> bool:
        return self.config.mode in {"split_wd", "split_wd_tags"}

    def _mask_path(self, image: np.ndarray) -> Optional[Path]:
        if self.config.mask_root is None:
            return None
        return Path(self.config.mask_root) / f"{reference_digest(image)}.png"

    def _load_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        path = self._mask_path(image)
        if path is None or not path.exists():
            return None
        try:
            gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        except Exception:
            return None
        h, w = _as_uint8_rgb(image).shape[:2]
        if gray.shape[:2] != (h, w):
            gray = np.asarray(Image.fromarray(gray).resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)
        return gray[..., None]

    def _save_mask(self, image: np.ndarray, mask: np.ndarray) -> None:
        path = self._mask_path(image)
        if path is None or not self.config.cache_generated_masks:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        Image.fromarray(arr.astype(np.uint8)).save(path)

    def _skytnt_mask(self, image: np.ndarray) -> np.ndarray:
        if self._skytnt is None:
            from .skytnt_segmentation import SkyTNTMaskExtractor, SkyTNTMaskExtractorConfig

            self._skytnt = SkyTNTMaskExtractor(
                SkyTNTMaskExtractorConfig(
                    repo=self.config.skytnt_repo,
                    model_id=self.config.skytnt_model_id,
                    ckpt=self.config.skytnt_ckpt,
                    net=self.config.skytnt_net,
                    image_size=self.config.skytnt_image_size,
                    device=self.config.skytnt_device,
                    fp32=self.config.skytnt_fp32,
                    local_files_only=self.config.skytnt_local_files_only,
                )
            )
        mask = self._skytnt.mask(_as_uint8_rgb(image))
        self._save_mask(image, mask)
        return mask

    def _fallback_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        rgb = _as_uint8_rgb(image)
        h, w = rgb.shape[:2]
        mode = (self.config.mask_fallback or "grabcut").lower()
        if mode == "skip":
            return None
        if mode == "skytnt":
            return self._skytnt_mask(rgb)
        if mode == "ellipse":
            return (_fallback_foreground_mask(h, w)[..., None] * 255.0).astype(np.uint8)
        if mode == "whole":
            return np.full((h, w, 1), 255, dtype=np.uint8)
        return (estimate_foreground_mask(rgb)[..., None] * 255.0).astype(np.uint8)

    def reference_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        mask = self._load_mask(image)
        if mask is not None:
            return mask
        return self._fallback_mask(image)

    def _cache_path(self, image: np.ndarray) -> Optional[Path]:
        if self.config.tag_cache_root is None:
            return None
        digest = reference_digest(image)
        return self.config.tag_cache_root / f"{digest}.json"

    def _load_wd_result(self, image: np.ndarray) -> Optional[WDTaggerResult]:
        path = self._cache_path(image)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if self.wants_wd_tokens and ("wd_indices" not in data or "wd_scores" not in data):
                    return None
                tags = data.get("tags", [])
                indices = np.asarray(data.get("wd_indices", []), dtype=np.int64)
                scores = np.asarray(data.get("wd_scores", []), dtype=np.float32)
            else:
                if self.wants_wd_tokens:
                    return None
                tags = data if isinstance(data, list) else []
                indices = np.zeros((0,), dtype=np.int64)
                scores = np.zeros((0,), dtype=np.float32)
            return WDTaggerResult([str(tag) for tag in tags][: self.config.max_tags], indices, scores)
        except Exception:
            return None

    def _save_wd_result(self, image: np.ndarray, result: WDTaggerResult, source: str) -> None:
        path = self._cache_path(image)
        if path is None:
            return
        payload = {
            "source": source,
            "tags": list(result.tags)[: self.config.max_tags],
            "wd_indices": result.indices.astype(np.int64).tolist(),
            "wd_scores": [float(x) for x in result.scores.astype(np.float32).tolist()],
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def wd_result(self, image: np.ndarray, foreground_mask: Optional[np.ndarray] = None) -> WDTaggerResult:
        cached = self._load_wd_result(image)
        if cached is not None:
            return cached
        source = "heuristic"
        if self._wd_tagger is not None:
            try:
                result = self._wd_tagger.predict(image)
                source = "wd_tagger_onnx"
            except Exception:
                tags = estimate_reference_tags(image, foreground_mask, self.config.max_tags)
                result = WDTaggerResult(tags, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32))
        else:
            tags = estimate_reference_tags(image, foreground_mask, self.config.max_tags)
            result = WDTaggerResult(tags, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32))
        self._save_wd_result(image, result, source)
        return result

    def tag(self, image: np.ndarray, foreground_mask: Optional[np.ndarray] = None) -> list[str]:
        if not self.wants_tags:
            return []
        return self.wd_result(image, foreground_mask).tags

    def split_layers(self, image: np.ndarray, mask: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Foreground/background condition layers for one reference, or (None, None) without a mask."""
        if mask is None:
            return None, None
        coverage = mask_coverage(mask)
        if coverage < self.config.min_foreground_coverage:
            return None, _as_uint8_rgb(image)
        if coverage > self.config.max_foreground_coverage:
            return _as_uint8_rgb(image), None
        fg, bg, _ = split_reference_layers(image, mask)
        return fg, bg

    def wd_results(self, image: np.ndarray, mask: Optional[np.ndarray]) -> list[WDTaggerResult]:
        # Match the official flow's separated reference contexts: feed foreground
        # and background semantics as distinct learned conditioning tokens.
        fg, bg = self.split_layers(image, mask)
        layers = [layer for layer in (fg, bg) if layer is not None] or [image]
        return [self.wd_result(layer, mask) for layer in layers]

    def build(
        self,
        references: Iterable[np.ndarray],
        reference_masks: Optional[Iterable[Optional[np.ndarray]]] = None,
        reference_tags: Optional[Iterable[Sequence[str]]] = None,
        reference_wd_indices: Optional[Iterable[Optional[Sequence[np.ndarray]]]] = None,
        reference_wd_scores: Optional[Iterable[Optional[Sequence[np.ndarray]]]] = None,
        reference_foregrounds: Optional[Iterable[Optional[np.ndarray]]] = None,
        reference_backgrounds: Optional[Iterable[Optional[np.ndarray]]] = None,
    ) -> ReferenceConditioningResult:
        foregrounds: list[np.ndarray] = []
        backgrounds: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        tags: list[str] = []
        wd_indices: list[np.ndarray] = []
        wd_scores: list[np.ndarray] = []
        if not self.enabled:
            return ReferenceConditioningResult(foregrounds, backgrounds, masks, tags, wd_indices, wd_scores)

        def at(items, index):
            return items[index] if index < len(items) else None

        mask_list = list(reference_masks) if reference_masks is not None else []
        tag_list = list(reference_tags) if reference_tags is not None else []
        wd_index_list = list(reference_wd_indices) if reference_wd_indices is not None else []
        wd_score_list = list(reference_wd_scores) if reference_wd_scores is not None else []
        fg_layer_list = list(reference_foregrounds) if reference_foregrounds is not None else []
        bg_layer_list = list(reference_backgrounds) if reference_backgrounds is not None else []
        for index, ref in enumerate(references):
            mask_input = mask_list[index] if index < len(mask_list) else self.reference_mask(ref)
            fg: Optional[np.ndarray] = None
            bg: Optional[np.ndarray] = None
            mask: Optional[np.ndarray] = None
            if mask_input is not None:
                mask = normalize_mask(mask_input)
                masks.append(mask)
                fg, bg = self.split_layers(ref, mask)
            explicit_fg = at(fg_layer_list, index)
            explicit_bg = at(bg_layer_list, index)
            if explicit_fg is not None:
                fg = explicit_fg
            if explicit_bg is not None:
                bg = explicit_bg
            if fg is not None:
                foregrounds.append(_as_uint8_rgb(fg))
            if bg is not None:
                backgrounds.append(_as_uint8_rgb(bg))

            if self.wants_tags:
                supplied_tags = at(tag_list, index)
                tags.extend(list(supplied_tags) if supplied_tags is not None else self.tag(ref, mask))
            if self.wants_wd_tokens:
                supplied_wd_indices = at(wd_index_list, index)
                supplied_wd_scores = at(wd_score_list, index)
                if supplied_wd_indices is not None and supplied_wd_scores is not None:
                    wd_indices.extend(np.asarray(item, dtype=np.int64) for item in supplied_wd_indices)
                    wd_scores.extend(np.asarray(item, dtype=np.float32) for item in supplied_wd_scores)
                else:
                    for result in self.wd_results(ref, mask):
                        wd_indices.append(result.indices)
                        wd_scores.append(result.scores)
        seen = set()
        unique_tags: list[str] = []
        for tag in tags:
            if tag not in seen:
                unique_tags.append(tag)
                seen.add(tag)
            if len(unique_tags) >= self.config.max_tags:
                break
        return ReferenceConditioningResult(foregrounds, backgrounds, masks, unique_tags, wd_indices, wd_scores)
