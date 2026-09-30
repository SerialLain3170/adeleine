from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SkyTNTMaskExtractorConfig:
    repo: Optional[Path] = None
    model_id: str = "skytnt/anime-seg"
    ckpt: Optional[Path] = None
    net: str = "isnet_is"
    image_size: int = 1024
    device: str = "cuda:0"
    fp32: bool = False
    local_files_only: bool = False


class SkyTNTMaskExtractor:
    """Thin wrapper around SkyTNT/anime-segmentation inference.

    The external repository owns the model definitions. This wrapper keeps the
    dependency optional and lets Adeleine use either a local checkpoint or the
    Hugging Face model-hub checkpoint exposed by AnimeSegmentation.
    """

    def __init__(self, config: SkyTNTMaskExtractorConfig) -> None:
        self.config = config
        self._model = None

    def _import_model_class(self):
        if self.config.repo is not None:
            repo = Path(self.config.repo).expanduser().resolve()
            if not repo.exists():
                raise FileNotFoundError(f"SkyTNT repository not found: {repo}")
            repo_str = str(repo)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
        try:
            from train import AnimeSegmentation
        except Exception as exc:
            raise ImportError(
                "Could not import SkyTNT AnimeSegmentation. Clone "
                "https://github.com/SkyTNT/anime-segmentation and pass "
                "--skytnt_repo, or install its dependencies in this environment."
            ) from exc
        return AnimeSegmentation

    def load(self):
        if self._model is not None:
            return self._model
        import torch

        AnimeSegmentation = self._import_model_class()
        if self.config.ckpt is not None:
            model = AnimeSegmentation.try_load(
                self.config.net,
                str(self.config.ckpt),
                self.config.device,
                img_size=self.config.image_size,
            )
        else:
            kwargs = {"net_name": self.config.net, "img_size": self.config.image_size}
            try:
                model = AnimeSegmentation.from_pretrained(
                    self.config.model_id,
                    map_location=self.config.device,
                    local_files_only=self.config.local_files_only,
                    **kwargs,
                )
            except TypeError:
                model = AnimeSegmentation.from_pretrained(self.config.model_id, **kwargs)
        device = torch.device(self.config.device)
        model.eval()
        model.to(device)
        self._model = model
        return model

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        from torch.cuda import amp

        model = self.load()
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError("SkyTNTMaskExtractor expects an RGB image")
        image = image[..., :3]
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        input_img = (image / 255.0).astype(np.float32)
        h0, w0 = input_img.shape[:2]
        size = int(self.config.image_size)
        if h0 > w0:
            h, w = size, int(size * w0 / h0)
        else:
            h, w = int(size * h0 / w0), size
        ph, pw = size - h, size - w
        img_input = np.zeros((size, size, 3), dtype=np.float32)
        img_input[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w] = cv2.resize(input_img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))[None]
        tensor = torch.from_numpy(img_input).float().to(model.device)
        with torch.no_grad():
            if not self.config.fp32 and str(model.device).startswith("cuda"):
                with amp.autocast():
                    pred = model(tensor)
                pred = pred.to(dtype=torch.float32)
            else:
                pred = model(tensor)
            pred_np = pred.detach().cpu().numpy()[0]
        pred_np = np.transpose(pred_np, (1, 2, 0))
        pred_np = pred_np[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
        pred_np = cv2.resize(pred_np, (w0, h0))
        if pred_np.ndim == 2:
            pred_np = pred_np[..., None]
        return np.clip(pred_np[..., :1] * 255.0, 0, 255).astype(np.uint8)
