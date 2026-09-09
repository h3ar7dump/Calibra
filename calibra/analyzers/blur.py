"""
Blur Analyzer.

Detects episodes whose camera frames are anomalously blurry relative to the
rest of the dataset. Uses Laplacian variance — the standard, dependency-free
sharpness proxy (a low-variance Laplacian response means little high-frequency
detail, i.e. a blurry or out-of-focus frame; a high-variance response means
sharp edges).

By default this is an IQR-outlier check, not a fixed absolute threshold: Laplacian
variance's scale depends on camera resolution, exposure, and scene content,
so there's no universal "blurry" cutoff that generalizes across datasets —
the same pitfall `TaskStructureAnalyzer._check_short_episodes` avoids for
episode length by comparing episodes against the dataset's own distribution
rather than a magic number.

For a batch containing exactly one episode, supply sharpness_threshold to
compare its mean Laplacian variance across all frames with a calibrated cutoff.
Multi-episode batches always use the original IQR test, ignoring that cutoff.
Both paths report blurry_episode_fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.analyzers.duplicate_frame import _find_image_obs
from calibra.analyzers.task_structure import _threshold_level_upper
from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import AnalyzerResult, ObservedValue, RiskFlag, RiskLevel

_BLUR_WARNING = 0.05  # 5% of episodes are blur outliers
_BLUR_CRITICAL = 0.15  # 15%


def _frame_laplacian_variance(frame: np.ndarray) -> float:
    gray = frame.mean(axis=-1).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4 * gray[1:-1, 1:-1]
    return float(lap.var())


def compute_laplacian_variance(images: np.ndarray) -> np.ndarray:
    """
    Per-frame Laplacian variance — a standard, dependency-free blur metric.

    Parameters
    ----------
    images : np.ndarray, shape (T, H, W, C) or (T, H, W)

    Returns
    -------
    np.ndarray, shape (T,)
        Variance of the discrete Laplacian response per frame. Lower values
        indicate less high-frequency detail (blurrier frames).
    """
    if images.ndim not in (3, 4):
        raise ValueError(f"images must be (T, H, W) or (T, H, W, C), got shape {images.shape}.")
    if len(images) == 0:
        return np.array([], dtype=np.float32)

    T = len(images)
    step = max(1, T // 30) if T > 30 else 1
    indices = list(range(0, T, step))
    variances = np.empty(len(indices), dtype=np.float32)

    for i, t in enumerate(indices):
        variances[i] = _frame_laplacian_variance(images[t])

    return variances


def _episode_mean_sharpness(ep: Episode) -> Optional[float]:
    cached = getattr(ep, "_cached_sharpness", None)
    if cached is not None:
        return float(cached)

    images = _find_image_obs(ep)
    if images is None:
        return None
    val = float(np.mean(compute_laplacian_variance(images)))
    try:
        setattr(ep, "_cached_sharpness", val)
    except Exception:
        pass
    return val


@dataclass
class BlurAnalyzer(Analyzer):
    """
    Flags episodes whose camera frames are blur outliers relative to the
    rest of the dataset (IQR outlier on mean per-episode Laplacian variance).

    Parameters
    ----------
    warning, critical : blurry-episode-fraction thresholds for risk level.
    sharpness_threshold : optional positive, finite Laplacian variance cutoff.
                          Used only when the batch contains exactly one episode:
                          average every frame's sharpness and compare that mean
                          with the cutoff. Ignored for multi-episode IQR checks.
                          Calibrate for the input camera, resolution, and pixel
                          scale; images are not normalized.
    """

    requires = frozenset({"images"})
    version = "4"

    warning: float = _BLUR_WARNING
    critical: float = _BLUR_CRITICAL
    sharpness_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sharpness_threshold is not None:
            if not np.isfinite(self.sharpness_threshold) or self.sharpness_threshold <= 0:
                raise ValueError("sharpness_threshold must be finite and positive")
            if not 0 < self.warning <= self.critical <= 1:
                raise ValueError(
                    "absolute blur thresholds must satisfy 0 < warning <= critical <= 1"
                )

    @property
    def name(self) -> str:
        return "blur"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        if batch.n_episodes == 1:
            if self.sharpness_threshold is None:
                return AnalyzerResult(
                    analyzer_name=self.name,
                    flags=[
                        RiskFlag(
                            level=RiskLevel.INFO,
                            metric="blurry_episode_fraction",
                            observed=ObservedValue(value=None),
                            interpretation="Single-episode blur detection needs a sharpness cutoff.",
                            implication="Set sharpness_threshold on BlurAnalyzer to run it.",
                        )
                    ],
                    raw_metrics={"skipped": "sharpness_threshold required for single episode"},
                )
            return self._analyze_absolute(batch, self.sharpness_threshold)

        ep_values: list[Optional[float]] = [_episode_mean_sharpness(ep) for ep in batch.episodes]
        checked = [(ep, v) for ep, v in zip(batch.episodes, ep_values) if v is not None]

        if not checked:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="blurry_episode_fraction",
                        observed=ObservedValue(value=None),
                        interpretation="No decodable camera frames found for this dataset.",
                        implication="Blur detection was skipped.",
                    )
                ],
                raw_metrics={"skipped": "no image observations"},
            )

        if len(checked) < 4:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="blurry_episode_fraction",
                        observed=ObservedValue(value=None),
                        interpretation="Too few episodes with image data for IQR outlier detection.",
                        implication="Need at least 4 episodes with camera frames.",
                    )
                ],
                raw_metrics={"skipped": "too few episodes with image data"},
            )

        sharpness = np.array([v for _, v in checked])
        q1, q3 = np.percentile(sharpness, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr

        outlier_mask = sharpness < lower_fence
        frac = float(np.mean(outlier_mask))
        outlier_ids = [
            ep.metadata.episode_id
            for (ep, _), is_outlier in zip(checked, outlier_mask)
            if is_outlier
        ]
        raw = {
            "blurry_episode_fraction": frac,
            "lower_fence_laplacian_var": float(lower_fence),
            "q1": float(q1),
            "q3": float(q3),
            "n_episodes_checked": len(checked),
            "outlier_episode_ids": outlier_ids[:20],
        }

        level = _threshold_level_upper(frac, self.warning, self.critical)
        if level == RiskLevel.OK or frac == 0.0:
            flag = RiskFlag(
                level=RiskLevel.OK,
                metric="blurry_episode_fraction",
                observed=ObservedValue(value=frac, unit="fraction"),
                threshold=self.warning,
                interpretation="No episodes are anomalously blurry relative to the rest of the dataset.",
                implication="No blur risk detected.",
            )
        else:
            flag = RiskFlag(
                level=level,
                metric="blurry_episode_fraction",
                observed=ObservedValue(value=frac, unit="fraction"),
                threshold=self.warning,
                interpretation=(
                    f"{frac:.1%} of episodes ({len(outlier_ids)}/{len(checked)}) have "
                    f"camera frames markedly blurrier than the rest of the dataset "
                    f"(mean Laplacian variance below the IQR lower fence). "
                    f"IDs: {outlier_ids[:5]}{'...' if len(outlier_ids) > 5 else ''}"
                ),
                implication=(
                    "Blurry frames (motion blur, defocus, or a dirty/misconfigured lens) "
                    "give the policy a degraded or misleading visual observation for that "
                    "episode. Inspect the flagged episodes before training — this is a "
                    "relative comparison within this dataset, not an absolute sharpness "
                    "standard, so a genuinely blurry whole dataset won't self-flag."
                ),
                affected_fraction=frac,
            )

        return AnalyzerResult(analyzer_name=self.name, flags=[flag], raw_metrics=raw)

    def _analyze_absolute(self, batch: EpisodeBatch, sharpness_threshold: float) -> AnalyzerResult:
        """Classify each episode by mean sharpness, independent of other episodes."""
        n_frames = 0
        n_checked = 0
        outlier_ids: list[str] = []
        episode_ids: list[str] = []
        episode_values: list[Optional[float]] = []
        for ep in batch.episodes:
            episode_ids.append(ep.metadata.episode_id)
            images = _find_image_obs(ep, min_frames=1)
            if images is None:
                episode_values.append(None)
                continue
            total_sharpness = 0.0
            for frame in images:
                sharpness = _frame_laplacian_variance(frame)
                if not np.isfinite(sharpness):
                    raise ValueError(
                        f"non-finite camera sharpness in episode {ep.metadata.episode_id}"
                    )
                total_sharpness += sharpness
            mean_sharpness = total_sharpness / len(images)
            episode_values.append(mean_sharpness)
            if mean_sharpness < sharpness_threshold:
                outlier_ids.append(ep.metadata.episode_id)
            n_frames += len(images)
            n_checked += 1

        fraction = len(outlier_ids) / n_checked if n_checked else None
        raw = {
            "mode": "absolute",
            "sharpness_threshold": sharpness_threshold,
            "warning": self.warning,
            "critical": self.critical,
            "blurry_episode_fraction": fraction,
            "n_frames_checked": n_frames,
            "n_blurry_episodes": len(outlier_ids),
            "n_episodes_checked": n_checked,
            "outlier_episode_ids": outlier_ids[:20],
            "episode_ids": episode_ids,
            "episode_values": episode_values,
        }
        if fraction is None:
            raw["skipped"] = "no image observations"
            flag = RiskFlag(
                level=RiskLevel.INFO,
                metric="blurry_episode_fraction",
                observed=ObservedValue(value=None),
                interpretation="No decodable camera frames found for this dataset.",
                implication="Blur detection was skipped.",
            )
        else:
            flag = RiskFlag(
                level=_threshold_level_upper(fraction, self.warning, self.critical),
                metric="blurry_episode_fraction",
                observed=ObservedValue(value=fraction, unit="fraction"),
                threshold=self.warning,
                interpretation=(
                    f"{len(outlier_ids)} of {n_checked} episodes ({fraction:.1%}) have mean "
                    f"Laplacian variance below {sharpness_threshold:g} "
                    f"({n_frames} camera frames checked)."
                ),
                implication=(
                    "Episodes below the configured mean sharpness cutoff may be blurred or lack "
                    "texture. Interpret this result using a cutoff calibrated for the "
                    "camera, resolution, scene content, and input pixel scale."
                ),
                affected_fraction=fraction,
            )
        return AnalyzerResult(analyzer_name=self.name, flags=[flag], raw_metrics=raw)
