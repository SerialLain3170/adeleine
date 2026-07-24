from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .conditions import ColorizationBatch


@dataclass
class FluxKleinConfig:
    model_id: str = "black-forest-labs/FLUX.2-klein-base-4B"
    revision: Optional[str] = None
    torch_dtype: str = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lineart_control_rank: int = 16
    train_text: bool = False
    device: str = "cuda"


class FluxKleinColorizer:
    """Adapter boundary for FLUX.2 Klein.

    `diffusers` now exposes explicit Flux2 Klein classes, but full model
    fine-tuning still depends on the exact upstream trainer. This class keeps
    Adeleine-specific conditions separate from vendor APIs and prepares the
    transformer for LoRA when possible.
    """

    def __init__(self, config: FluxKleinConfig):
        self.config = config
        self.pipeline = None
        self.transformer = None

    def load_pipeline(self) -> Any:
        try:
            import torch
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise ImportError("Install diffusers and torch to load FLUX.2 Klein") from exc

        dtype = getattr(torch, self.config.torch_dtype)
        self.pipeline = Flux2KleinPipeline.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            torch_dtype=dtype,
        )
        if hasattr(self.pipeline, "to"):
            self.pipeline.to(self.config.device)
        self.transformer = getattr(self.pipeline, "transformer", None)
        return self.pipeline

    def load_transformer(self) -> Any:
        try:
            import torch
            from diffusers import Flux2Transformer2DModel
        except ImportError as exc:
            raise ImportError("Install diffusers and torch to load the FLUX.2 transformer") from exc

        dtype = getattr(torch, self.config.torch_dtype)
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            self.config.model_id,
            subfolder="transformer",
            revision=self.config.revision,
            torch_dtype=dtype,
        )
        if hasattr(self.transformer, "to"):
            self.transformer.to(self.config.device)
        return self.transformer

    def prepare_lora(self, target_modules: Optional[List[str]] = None) -> Any:
        if self.transformer is None:
            if self.pipeline is None:
                self.load_pipeline()
            self.transformer = getattr(self.pipeline, "transformer", None)
        if self.transformer is None:
            raise RuntimeError("Could not find a transformer module on the FLUX.2 Klein pipeline")

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("Install peft to prepare LoRA training") from exc

        modules = target_modules or self.default_lora_targets(self.transformer)
        if not modules:
            raise RuntimeError("No LoRA target modules were found on the transformer")

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=modules,
            bias="none",
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        if self.pipeline is not None and hasattr(self.pipeline, "transformer"):
            self.pipeline.transformer = self.transformer
        return self.transformer

    @staticmethod
    def default_lora_targets(module: Any) -> List[str]:
        suffixes = {"to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"}
        found = set()
        for name, child in module.named_modules():
            if child.__class__.__name__ != "Linear":
                continue
            for suffix in suffixes:
                if name.endswith(suffix):
                    found.add(suffix)
        return sorted(found)

    @staticmethod
    def trainable_parameter_count(module: Any) -> tuple[int, int]:
        trainable = 0
        total = 0
        for param in module.parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
        return trainable, total

    def save_lora(self, output_dir: Path) -> None:
        if self.transformer is None:
            raise RuntimeError("No transformer has been prepared")
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.transformer, "save_pretrained"):
            self.transformer.save_pretrained(output_dir)
        else:
            raise RuntimeError("Prepared transformer does not expose save_pretrained")

    def build_condition_payload(self, batch: ColorizationBatch) -> Dict[str, Any]:
        """Maps Adeleine conditions to a multimodal editing payload."""
        return {
            "lineart": batch.lineart,
            "atari_rgb": batch.atari_rgb,
            "atari_mask": batch.atari_mask,
            "references": batch.references,
            "text": batch.text,
            "mode": batch.mode,
            "presence": batch.presence,
        }

    @staticmethod
    def iter_payload_images(payload: Dict[str, Any]) -> Iterable[Any]:
        yield payload["lineart"]
        if payload.get("atari_rgb") is not None:
            yield payload["atari_rgb"]
        if payload.get("atari_mask") is not None:
            yield payload["atari_mask"]
        for refs in payload.get("references", []):
            for ref in refs:
                yield ref
