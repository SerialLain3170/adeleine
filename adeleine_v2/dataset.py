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
from .reference_conditioning import ReferenceConditioningBuilder, ReferenceConditioningConfig


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
        reference_conditioning: str = "none",
        reference_tag_cache_root: Optional[Path] = None,
        reference_mask_root: Optional[Path] = None,
        reference_mask_fallback: str = "grabcut",
        skytnt_repo: Optional[Path] = None,
        skytnt_model_id: str = "skytnt/anime-seg",
        skytnt_ckpt: Optional[Path] = None,
        skytnt_net: str = "isnet_is",
        skytnt_image_size: int = 1024,
        skytnt_device: str = "cuda:0",
        skytnt_fp32: bool = False,
        skytnt_local_files_only: bool = False,
        reference_cache_generated_masks: bool = True,
        wd_tagger_model: Optional[Path] = None,
        wd_tagger_labels: Optional[Path] = None,
        wd_tagger_threshold: float = 0.35,
        wd_tagger_character_threshold: float = 0.85,
        wd_tagger_max_tokens: int = 32,
        reference_tag_max: int = 24,
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
        self.reference_conditioner = ReferenceConditioningBuilder(
            ReferenceConditioningConfig(
                mode=reference_conditioning,
                tag_cache_root=reference_tag_cache_root,
                mask_root=reference_mask_root,
                mask_fallback=reference_mask_fallback,
                skytnt_repo=skytnt_repo,
                skytnt_model_id=skytnt_model_id,
                skytnt_ckpt=skytnt_ckpt,
                skytnt_net=skytnt_net,
                skytnt_image_size=skytnt_image_size,
                skytnt_device=skytnt_device,
                skytnt_fp32=skytnt_fp32,
                skytnt_local_files_only=skytnt_local_files_only,
                cache_generated_masks=reference_cache_generated_masks,
                wd_tagger_model=wd_tagger_model,
                wd_tagger_labels=wd_tagger_labels,
                wd_tagger_threshold=wd_tagger_threshold,
                wd_tagger_character_threshold=wd_tagger_character_threshold,
                wd_tagger_max_tokens=wd_tagger_max_tokens,
                max_tags=reference_tag_max,
            )
        )
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

        ref_cond = self.reference_conditioner.build(refs)
        target = self._flat_target(image_path)
        if target is None:
            target = color_rgb

        condition = ColorizationCondition(
            lineart=line_rgb,
            target=target,
            atari_rgb=atari_rgb,
            atari_mask=atari_mask,
            references=refs,
            reference_foregrounds=ref_cond.foregrounds,
            reference_backgrounds=ref_cond.backgrounds,
            reference_masks=ref_cond.masks,
            reference_tags=ref_cond.tags,
            reference_wd_indices=ref_cond.wd_indices,
            reference_wd_scores=ref_cond.wd_scores,
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
            reference_foregrounds=[item.reference_foregrounds for item in batch],
            reference_backgrounds=[item.reference_backgrounds for item in batch],
            reference_masks=[item.reference_masks for item in batch],
            reference_tags=[item.reference_tags for item in batch],
            reference_wd_indices=[item.reference_wd_indices for item in batch],
            reference_wd_scores=[item.reference_wd_scores for item in batch],
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
