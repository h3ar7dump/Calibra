"""Camera metrics shared by array-backed and streaming integrity checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibra.schema.episode import Episode
from calibra.schema.images import ImageStream

_VISUAL_TOKENS = ("camera", "image", "rgb", "depth", "visual")


def iter_image_sources(ep: Episode, tokens: tuple[str, ...] = _VISUAL_TOKENS):
    """Keep camera selection stable, including before and after decoding."""
    for key in ep.observations:
        if any(token in key.lower() for token in tokens):
            images = ep.observations[key]
            if isinstance(images, np.ndarray) and images.ndim in (3, 4):
                yield key, images
    for key in sorted(ep.image_streams):
        if key not in ep.observations:
            yield key, ep.image_streams[key]


def find_image_source(
    ep: Episode, min_frames: int = 2, min_spatial_size: int = 0
) -> np.ndarray | ImageStream | None:
    for _, images in iter_image_sources(ep):
        if len(images) < min_frames:
            continue
        shape = images.spatial_shape if isinstance(images, ImageStream) else images.shape[1:3]
        if isinstance(images, ImageStream) and shape is None and min_spatial_size:
            measure_image_stream(images)
            shape = images.spatial_shape
        if shape is not None and min(shape) < min_spatial_size:
            continue
        return images
    return None


def frame_laplacian_variance(frame: np.ndarray) -> float:
    gray = frame.mean(axis=-1).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4 * gray[1:-1, 1:-1]
    return float(lap.var())


def _repeated_pair(
    previous: np.ndarray, current: np.ndarray, tolerance: float, min_changed_fraction: float
) -> bool:
    # Preserve repeated_transitions' arithmetic and pixel/channel semantics.
    diff = np.abs(current.astype(np.float32) - previous.astype(np.float32)) * 1.0
    changed = (diff > tolerance).any(axis=-1)
    return bool(changed.mean() < min_changed_fraction)


def _visual_pair(previous: np.ndarray, current: np.ndarray) -> float:
    diff = np.abs(current[:, :, 1].astype(np.int16) - previous[:, :, 1].astype(np.int16))
    return float(diff.mean())


@dataclass
class ImageMetrics:
    repeated: np.ndarray
    visual_activity: np.ndarray
    sharpness: np.ndarray
    mean_sharpness: float | None = None


def measure_image_stream(
    stream: ImageStream,
    pixel_tolerance: float = 2.0,
    min_changed_fraction: float = 5e-4,
    *,
    all_sharpness: bool = False,
) -> ImageMetrics:
    """Decode once, retaining adjacent frames and compact metric arrays only.

    Sampling uses episode-wide frame indices, including across file boundaries.
    Absolute blur optionally uses every frame and the original summation order.
    """
    key = (pixel_tolerance, min_changed_fraction, all_sharpness)
    cached = stream.metric_cache.get(key)
    if cached is None and not all_sharpness:
        cached = stream.metric_cache.get((pixel_tolerance, min_changed_fraction, True))
    if cached is not None:
        return cached

    total = len(stream)
    step = max(1, total // 30) if total > 30 else 1
    metrics = ImageMetrics(
        repeated=np.empty(max(0, total - 1), dtype=bool),
        visual_activity=np.empty(max(0, total - 1), dtype=np.float32),
        sharpness=np.empty(len(range(0, total, step)), dtype=np.float32),
    )
    previous = None
    sharpness_sum = 0.0
    count = 0
    frames = iter(stream)
    try:
        for index, frame in enumerate(frames):
            if previous is not None:
                metrics.repeated[index - 1] = _repeated_pair(
                    previous, frame, pixel_tolerance, min_changed_fraction
                )
                metrics.visual_activity[index - 1] = _visual_pair(previous, frame)
            if all_sharpness or index % step == 0:
                sharpness = frame_laplacian_variance(frame)
                if index % step == 0:
                    metrics.sharpness[index // step] = sharpness
                if all_sharpness:
                    if not np.isfinite(sharpness):
                        raise ValueError("non-finite camera sharpness in image stream")
                    sharpness_sum += sharpness
            previous = frame
            count += 1
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
    if all_sharpness and count:
        metrics.mean_sharpness = sharpness_sum / count
    stream.metric_cache[key] = metrics
    return metrics
