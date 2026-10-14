"""Repeatable, full-resolution image streams at the ingestion boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ImageStream:
    """Yield RGB uint8 frames without keeping their pixels in memory.

    The factory must return a fresh iterator on each call. The cache contains
    only derived scalar sequences, so separate analyzers can share one decode.
    """

    frame_factory: Callable[[], Iterator[np.ndarray]]
    n_frames: int
    spatial_shape: tuple[int, int] | None = None
    metric_cache: dict[tuple, Any] = field(default_factory=dict, repr=False, compare=False)
    validated: bool = field(default=False, repr=False, compare=False)

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        frames = iter(self.frame_factory())
        count = 0
        shape = None
        try:
            for frame in frames:
                if count >= self.n_frames:
                    raise ValueError(f"Image stream frame count exceeds expected {self.n_frames}")
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    raise ValueError("Image streams must yield RGB uint8 frames")
                if shape is not None and frame.shape != shape:
                    raise ValueError("Image stream frame dimensions changed within an episode")
                shape = frame.shape
                self.spatial_shape = frame.shape[:2]
                count += 1
                yield frame
        finally:
            close = getattr(frames, "close", None)
            if close is not None:
                close()
        if count != self.n_frames:
            raise ValueError(
                f"Image stream frame count mismatch: expected={self.n_frames}, decoded={count}"
            )
        self.validated = True

    def validate(self) -> None:
        """Validate even an unused camera, releasing each frame immediately."""
        if not self.validated:
            for _ in self:
                pass
