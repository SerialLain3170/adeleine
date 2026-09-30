from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .conditions import ColorizationBatch


@dataclass
class ConditionTensors:
    lineart: torch.Tensor
    target: Optional[torch.Tensor]
    atari_rgb: Optional[torch.Tensor]
    atari_mask: Optional[torch.Tensor]
    reference_images: List[List[torch.Tensor]]
    reference_foregrounds: List[List[torch.Tensor]]
    reference_backgrounds: List[List[torch.Tensor]]
    reference_masks: List[List[torch.Tensor]]
    reference_tags: List[List[str]]
    reference_wd_indices: List[List[torch.Tensor]]
    reference_wd_scores: List[List[torch.Tensor]]
    presence: Dict[str, torch.Tensor]
    mode_ids: torch.Tensor
    text: List[str]


MODE_TO_ID = {"render": 0, "flat": 1, "diverse": 2}


def image_array_to_tensor(array: np.ndarray, normalize: bool = True) -> torch.Tensor:
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    if normalize:
        tensor = tensor / 127.5 - 1.0
    return tensor


def mask_array_to_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    return tensor / 255.0


def batch_to_tensors(batch: ColorizationBatch, device: torch.device | str = "cpu") -> ConditionTensors:
    lineart = image_array_to_tensor(batch.lineart).to(device)
    target = image_array_to_tensor(batch.target).to(device) if batch.target is not None else None
    atari_rgb = image_array_to_tensor(batch.atari_rgb).to(device) if batch.atari_rgb is not None else None
    atari_mask = mask_array_to_tensor(batch.atari_mask).to(device) if batch.atari_mask is not None else None
    def tensorize_images(items: List[List[np.ndarray]], normalize: bool = True) -> List[List[torch.Tensor]]:
        out: List[List[torch.Tensor]] = []
        for images in items:
            out.append([image_array_to_tensor(image[None], normalize=normalize).squeeze(0).to(device) for image in images])
        return out

    def tensorize_masks(items: List[List[np.ndarray]]) -> List[List[torch.Tensor]]:
        out: List[List[torch.Tensor]] = []
        for masks in items:
            out.append([mask_array_to_tensor(mask[None]).squeeze(0).to(device) for mask in masks])
        return out

    references = tensorize_images(batch.references)
    reference_foregrounds = tensorize_images(batch.reference_foregrounds)
    reference_backgrounds = tensorize_images(batch.reference_backgrounds)
    reference_masks = tensorize_masks(batch.reference_masks)

    def tensorize_wd(items: List[List[np.ndarray]], dtype: torch.dtype) -> List[List[torch.Tensor]]:
        out: List[List[torch.Tensor]] = []
        for refs in items:
            out.append([torch.as_tensor(ref, dtype=dtype, device=device) for ref in refs])
        return out

    reference_wd_indices = tensorize_wd(batch.reference_wd_indices, torch.long)
    reference_wd_scores = tensorize_wd(batch.reference_wd_scores, torch.float32)
    presence = {key: torch.from_numpy(value.astype(np.float32)).to(device) for key, value in batch.presence.items()}
    mode_ids = torch.tensor([MODE_TO_ID.get(mode, 0) for mode in batch.mode], dtype=torch.long, device=device)
    return ConditionTensors(
        lineart,
        target,
        atari_rgb,
        atari_mask,
        references,
        reference_foregrounds,
        reference_backgrounds,
        reference_masks,
        batch.reference_tags,
        reference_wd_indices,
        reference_wd_scores,
        presence,
        mode_ids,
        batch.text,
    )


class SpatialConditionAdapter(nn.Module):
    """Encodes line art plus optional Atari RGB/mask into spatial tokens."""

    def __init__(self, hidden_size: int = 1024, patch_size: int = 16, in_channels: int = 7):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        lineart: torch.Tensor,
        atari_rgb: Optional[torch.Tensor] = None,
        atari_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if atari_rgb is None:
            atari_rgb = torch.zeros_like(lineart)
        if atari_mask is None:
            atari_mask = torch.zeros(lineart.shape[0], 1, lineart.shape[2], lineart.shape[3], device=lineart.device, dtype=lineart.dtype)
        x = torch.cat([lineart, atari_rgb, atari_mask], dim=1)
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(tokens)


class ReferenceImageAdapter(nn.Module):
    """Small reference encoder placeholder before replacing with VLM/DINO tokens."""

    def __init__(self, hidden_size: int = 1024, max_refs: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_refs = max_refs
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.ref_index = nn.Embedding(max_refs, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, reference_images: List[List[torch.Tensor]], batch_size: int, device: torch.device | str) -> torch.Tensor:
        tokens = []
        for batch_index in range(batch_size):
            refs = reference_images[batch_index][: self.max_refs]
            if not refs:
                tokens.append(torch.zeros(self.max_refs, self.hidden_size, device=device))
                continue
            encoded = []
            for ref_index, ref in enumerate(refs):
                feat = self.encoder(ref.unsqueeze(0)).mean(dim=(2, 3)).squeeze(0)
                feat = feat + self.ref_index.weight[ref_index]
                encoded.append(feat)
            while len(encoded) < self.max_refs:
                encoded.append(torch.zeros(self.hidden_size, device=device))
            tokens.append(torch.stack(encoded, dim=0))
        return self.norm(torch.stack(tokens, dim=0))


class ModePresenceEmbedding(nn.Module):
    """Embeds mode plus explicit modality-presence bits."""

    def __init__(self, hidden_size: int = 1024, presence_keys: Optional[List[str]] = None):
        super().__init__()
        self.presence_keys = presence_keys or ["lineart", "atari", "reference", "text", "flat", "diverse"]
        self.mode = nn.Embedding(len(MODE_TO_ID), hidden_size)
        self.presence = nn.Linear(len(self.presence_keys), hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, mode_ids: torch.Tensor, presence: Dict[str, torch.Tensor]) -> torch.Tensor:
        bits = torch.stack([presence.get(key, torch.zeros_like(mode_ids, dtype=torch.float32)).float() for key in self.presence_keys], dim=1)
        return self.norm(self.mode(mode_ids) + self.presence(bits))


class AdeleineConditionAdapter(nn.Module):
    """Combines spatial, reference, and mode/presence condition tokens."""

    def __init__(self, hidden_size: int = 1024, patch_size: int = 16, max_refs: int = 8):
        super().__init__()
        self.spatial = SpatialConditionAdapter(hidden_size=hidden_size, patch_size=patch_size)
        self.reference = ReferenceImageAdapter(hidden_size=hidden_size, max_refs=max_refs)
        self.mode_presence = ModePresenceEmbedding(hidden_size=hidden_size)

    def forward(self, batch: ConditionTensors) -> Dict[str, torch.Tensor]:
        spatial = self.spatial(batch.lineart, batch.atari_rgb, batch.atari_mask)
        refs = self.reference(batch.reference_images, batch.lineart.shape[0], batch.lineart.device)
        global_token = self.mode_presence(batch.mode_ids, batch.presence).unsqueeze(1)
        return {
            "spatial_tokens": spatial,
            "reference_tokens": refs,
            "global_tokens": global_token,
            "all_tokens": torch.cat([global_token, refs, spatial], dim=1),
        }
