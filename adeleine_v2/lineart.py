from __future__ import annotations

from typing import Iterable

from .augmentations import (
    LegacyLineArtAugmentor,
    LineArtConfig,
    LineArtPaths,
    available_line_methods as available_methods,
)


class LineArtAugmentor(LegacyLineArtAugmentor):
    def __init__(self, methods: Iterable[str] | LineArtConfig = ("xdog", "pencil", "digital", "lineart_anime", "blend"), paths: LineArtPaths | None = None, **kwargs):
        if isinstance(methods, LineArtConfig):
            config = methods
        else:
            config = LineArtConfig(methods=tuple(methods), **kwargs)
        super().__init__(config=config, paths=paths)


__all__ = ["LineArtAugmentor", "LineArtConfig", "LineArtPaths", "available_methods"]
