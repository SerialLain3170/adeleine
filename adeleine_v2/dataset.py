from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from .atari import AtariHintGenerator
from .conditions import ColorizationBatch, ColorizationCondition, ModalityDropout, stack_presence
from .lineart import LineArtAugmentor, LineArtPaths, available_methods
from .reference import ReferenceSelector, grouped_reference_paths


class UnifiedColorizationDataset(Dataset):
    """Dataset for line-only, Atari, reference, text, render/diverse/flat tasks."""

    def __init__(
        self,
        data_root: Path,
        sketch_root: Optional[Path] = None,
        digital_root: Optional[Path] = None,
        anime_line_root: Optional[Path] = None,
        flat_root: Optional[Path] = None,
        reference_root: Optional[Path] = None,
        caption_file: Optional[Path] = None,
        extension: str = ".jpg",
        image_size: int = 512,
        line_methods: Sequence[str] = ("xdog", "pencil", "digital", "lineart_anime", "blend"),
        dropout: Optional[ModalityDropout] = None,
    ):
        self.data_root = data_root
        self.paths = sorted(data_root.glob(f"**/*{extension}"))
        self.flat_root = flat_root
        self.reference_root = reference_root
        self.image_size = image_size
        line_paths = LineArtPaths(pencil_dir=sketch_root, digital_dir=digital_root, anime_dir=anime_line_root)
        self.lineart = LineArtAugmentor(available_methods(list(line_methods), line_paths), line_paths)
        self.atari = AtariHintGenerator()
        self.references = ReferenceSelector(image_size=image_size)
        self.dropout = dropout or ModalityDropout()
        self.captions = self._read_captions(caption_file)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> ColorizationCondition:
        image_path = self.paths[index]
        color_bgr = cv.imread(str(image_path), cv.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(image_path)
        color_bgr = self._resize_square(color_bgr)
        color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)

        line_bgr = self.lineart(image_path, color_bgr=color_bgr)
        line_bgr = self._resize_square(line_bgr)
        line_rgb = cv.cvtColor(line_bgr, cv.COLOR_BGR2RGB)
        atari_rgb, atari_mask = self.atari(color_rgb, line_rgb)

        refs = []
        if self.reference_root is not None:
            paths = grouped_reference_paths(image_path, self.reference_root)
            refs = self.references.from_paths(line_rgb, paths).images

        target = self._flat_target(image_path)
        if target is None:
            target = color_rgb

        condition = ColorizationCondition(
            lineart=line_rgb,
            target=target,
            atari_rgb=atari_rgb,
            atari_mask=atari_mask,
            references=refs,
            text=self.captions.get(image_path.name, ""),
            metadata={"image_path": str(image_path)},
        )
        return self.dropout(condition)

    def _resize_square(self, img: np.ndarray) -> np.ndarray:
        return cv.resize(img, (self.image_size, self.image_size), interpolation=cv.INTER_AREA)

    def _flat_target(self, image_path: Path) -> Optional[np.ndarray]:
        if self.flat_root is None:
            return None
        flat_path = self.flat_root / image_path.name
        flat = cv.imread(str(flat_path), cv.IMREAD_COLOR)
        if flat is None:
            return None
        flat = self._resize_square(flat)
        return cv.cvtColor(flat, cv.COLOR_BGR2RGB)

    @staticmethod
    def _read_captions(path: Optional[Path]) -> Dict[str, str]:
        if path is None:
            return {}
        captions: Dict[str, str] = {}
        for line in path.read_text().splitlines():
            if not line.strip() or "\t" not in line:
                continue
            name, caption = line.split("\t", 1)
            captions[name] = caption
        return captions


class UnifiedCollator:
    def __call__(self, batch: Sequence[ColorizationCondition]) -> ColorizationBatch:
        return ColorizationBatch(
            lineart=np.stack([item.lineart for item in batch]),
            target=np.stack([item.target for item in batch]) if batch[0].target is not None else None,
            atari_rgb=self._stack_optional(batch, "atari_rgb"),
            atari_mask=self._stack_optional(batch, "atari_mask"),
            references=[item.references for item in batch],
            text=[item.text for item in batch],
            mode=[item.mode.value for item in batch],
            presence=stack_presence(batch),
            metadata=[item.metadata for item in batch],
        )

    @staticmethod
    def _stack_optional(batch: Sequence[ColorizationCondition], name: str) -> Optional[np.ndarray]:
        values = [getattr(item, name) for item in batch]
        template = next((value for value in values if value is not None), None)
        if template is None:
            return None
        filled = [value if value is not None else np.zeros_like(template) for value in values]
        return np.stack(filled)
