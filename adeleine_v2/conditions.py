from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np


class ColorizationMode(str, Enum):
    RENDER = "render"
    FLAT = "flat"
    DIVERSE = "diverse"


@dataclass
class ColorizationCondition:
    """Unified condition object for Adeleine v2.

    Images are uint8 RGB arrays in HWC order. `lineart` is required; all other
    modalities may be dropped independently by the dataset sampler.
    """

    lineart: np.ndarray
    target: Optional[np.ndarray] = None
    atari_rgb: Optional[np.ndarray] = None
    atari_mask: Optional[np.ndarray] = None
    references: List[np.ndarray] = field(default_factory=list)
    text: str = ""
    mode: ColorizationMode = ColorizationMode.RENDER
    metadata: Dict[str, object] = field(default_factory=dict)

    def presence(self) -> Dict[str, bool]:
        return {
            "lineart": True,
            "atari": self.atari_rgb is not None and self.atari_mask is not None,
            "reference": len(self.references) > 0,
            "text": bool(self.text),
            "target": self.target is not None,
            "flat": self.mode == ColorizationMode.FLAT,
            "diverse": self.mode == ColorizationMode.DIVERSE,
        }


@dataclass
class ColorizationBatch:
    lineart: np.ndarray
    target: Optional[np.ndarray]
    atari_rgb: Optional[np.ndarray]
    atari_mask: Optional[np.ndarray]
    references: List[List[np.ndarray]]
    text: List[str]
    mode: List[str]
    presence: Dict[str, np.ndarray]
    metadata: List[Dict[str, object]]


class TaskSampler:
    """Samples explicit training tasks instead of independent tiny probabilities."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        modes: Optional[Dict[str, ColorizationMode]] = None,
    ):
        self.weights = weights or {
            "line": 0.20,
            "line_atari": 0.20,
            "line_reference": 0.20,
            "line_text": 0.15,
            "line_reference_atari": 0.15,
            "all": 0.10,
        }
        self.modes = modes or {
            "line": ColorizationMode.RENDER,
            "line_atari": ColorizationMode.RENDER,
            "line_reference": ColorizationMode.RENDER,
            "line_text": ColorizationMode.DIVERSE,
            "line_reference_atari": ColorizationMode.RENDER,
            "all": ColorizationMode.RENDER,
        }
        keys = list(self.weights)
        total = sum(self.weights.values())
        self._keys = keys
        self._prob = np.array([self.weights[k] / total for k in keys])

    def sample(self) -> str:
        return str(np.random.choice(self._keys, p=self._prob))

    def mode_for(self, task: str) -> ColorizationMode:
        return self.modes.get(task, ColorizationMode.RENDER)


class ModalityDropout:
    """Applies task-level modality selection to a condition."""

    def __init__(self, task_sampler: Optional[TaskSampler] = None):
        self.task_sampler = task_sampler or TaskSampler()

    def __call__(self, condition: ColorizationCondition) -> ColorizationCondition:
        task = self.task_sampler.sample()
        keep_atari = task in {"line_atari", "line_reference_atari", "all"}
        keep_reference = task in {"line_reference", "line_reference_atari", "all"}
        keep_text = task in {"line_text", "all"}

        return ColorizationCondition(
            lineart=condition.lineart,
            target=condition.target,
            atari_rgb=condition.atari_rgb if keep_atari else None,
            atari_mask=condition.atari_mask if keep_atari else None,
            references=condition.references if keep_reference else [],
            text=condition.text if keep_text else "",
            mode=self.task_sampler.mode_for(task),
            metadata={**condition.metadata, "task": task},
        )


def stack_presence(conditions: Sequence[ColorizationCondition]) -> Dict[str, np.ndarray]:
    keys = conditions[0].presence().keys()
    return {
        key: np.asarray([item.presence()[key] for item in conditions], dtype=np.bool_)
        for key in keys
    }
