"""Tests for the Blur Analyzer."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from calibra.analyzers.blur import BlurAnalyzer, compute_laplacian_variance
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel

# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_ep(episode_id: str = "ep_0", n_steps: int = 10, image_mode: str = "sharp") -> Episode:
    # zlib.crc32 (not the builtin hash()) so the seed — and thus the test's
    # pass/fail outcome — doesn't depend on PYTHONHASHSEED, which Python
    # randomizes per process for str.
    rng = np.random.default_rng(zlib.crc32(episode_id.encode()) % (2**31))
    ts = np.arange(n_steps, dtype=np.float64) * 0.05
    obs: dict = {"proprio": rng.random((n_steps, 8)).astype(np.float32)}

    if image_mode == "sharp":
        # High-frequency noise: lots of local pixel-to-pixel variation.
        obs["camera_rgb"] = rng.integers(0, 255, (n_steps, 32, 32, 3), dtype=np.uint8)
    elif image_mode == "blurry":
        # Smooth low-frequency gradient: minimal local variation.
        x = np.linspace(0, 1, 32)
        gradient = np.outer(x, x)
        frame = (gradient * 255).astype(np.uint8)
        frame = np.stack([frame] * 3, axis=-1)
        obs["camera_rgb"] = np.tile(frame, (n_steps, 1, 1, 1))
    # "none" → no image key

    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=ts,
        observations=obs,
        actions=rng.random((n_steps, 6)).astype(np.float32),
    )


def _make_batch(episodes: list[Episode]) -> EpisodeBatch:
    return EpisodeBatch(
        episodes=episodes, dataset_name="blur_test", format="hdf5", source_path="/dummy/path.h5"
    )


# ── compute_laplacian_variance ──────────────────────────────────────────────


class TestComputeLaplacianVariance:
    def test_sharp_has_higher_variance_than_blurry(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        blurry = np.full((5, 32, 32, 3), 128, dtype=np.uint8)
        sharp_var = compute_laplacian_variance(sharp).mean()
        blurry_var = compute_laplacian_variance(blurry).mean()
        assert sharp_var > blurry_var

    def test_output_shape(self):
        images = np.zeros((7, 16, 16, 3), dtype=np.uint8)
        result = compute_laplacian_variance(images)
        assert result.shape == (7,)

    def test_grayscale_input(self):
        images = np.zeros((7, 16, 16), dtype=np.uint8)
        result = compute_laplacian_variance(images)
        assert result.shape == (7,)


# ── BlurAnalyzer ─────────────────────────────────────────────────────────────


class TestBlurAnalyzer:
    @pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_sharpness_threshold(self, threshold):
        with pytest.raises(ValueError, match="sharpness_threshold"):
            BlurAnalyzer(sharpness_threshold=threshold)

    @pytest.mark.parametrize(
        "warning, critical",
        [(0, 0.15), (-0.1, 0.15), (0.2, 0.1), (0.05, 1.1), (0.05, float("nan"))],
    )
    def test_rejects_invalid_absolute_fraction_thresholds(self, warning, critical):
        with pytest.raises(ValueError, match="warning.*critical"):
            BlurAnalyzer(sharpness_threshold=100.0, warning=warning, critical=critical)

    def test_single_blurry_episode_uses_existing_metric(self):
        episode = _make_ep(n_steps=20, image_mode="blurry")

        result = BlurAnalyzer(sharpness_threshold=100.0).analyze(_make_batch([episode]))

        flag = result.flags[0]
        assert flag.metric == "blurry_episode_fraction"
        assert flag.observed.value == 1.0
        assert flag.level == RiskLevel.CRITICAL
        assert result.raw_metrics["blurry_episode_fraction"] == 1.0
        assert result.raw_metrics["outlier_episode_ids"] == ["ep_0"]
        assert "blurry_frame_fraction" not in result.raw_metrics
        assert result.raw_metrics["n_frames_checked"] == 20

    @pytest.mark.parametrize(
        "n_blurry, expected_level",
        [
            (0, RiskLevel.OK),
            (1, RiskLevel.WARNING),
            (3, RiskLevel.CRITICAL),
            (20, RiskLevel.OK),  # A uniformly blurry batch has no relative outliers.
        ],
    )
    def test_multiple_episodes_use_iqr_even_with_absolute_threshold(self, n_blurry, expected_level):
        episodes = [
            _make_ep(f"ep_{i}", image_mode="blurry" if i < n_blurry else "sharp") for i in range(20)
        ]
        batch = _make_batch(episodes)
        result = BlurAnalyzer(sharpness_threshold=100.0).analyze(batch)
        baseline = BlurAnalyzer().analyze(batch)
        assert result.model_dump() == baseline.model_dump()
        assert result.flags[0].level == expected_level
        assert result.flags[0].observed.value == pytest.approx(
            0 if n_blurry == 20 else n_blurry / 20
        )

    def test_custom_fraction_thresholds_change_severity(self):
        episodes = [_make_ep("blur", image_mode="blurry")]
        episodes.extend(_make_ep(f"sharp_{i}") for i in range(4))
        analyzer = BlurAnalyzer(sharpness_threshold=100.0, warning=0.2, critical=0.5)
        result = analyzer.analyze(_make_batch(episodes))
        assert result.flags[0].level == RiskLevel.WARNING
        assert result.flags[0].threshold == 0.2

    def test_absolute_checks_all_frames_in_long_episode(self):
        episode = _make_ep(n_steps=120)
        checkerboard = (np.indices((4, 4)).sum(axis=0) % 2).astype(np.float32)
        episode.observations["camera_rgb"] = np.tile(checkerboard, (120, 1, 1))
        # Historical batch sampling takes every fourth frame, missing these.
        episode.observations["camera_rgb"][1::4] = 0
        result = BlurAnalyzer(sharpness_threshold=13.0).analyze(_make_batch([episode]))
        assert result.flags[0].observed.value == 1.0
        assert result.raw_metrics["episode_values"] == [12.0]
        assert result.raw_metrics["n_frames_checked"] == 120

    @pytest.mark.parametrize("n_episodes, n_images", [(2, 2), (3, 3), (4, 1)])
    def test_multiple_episodes_with_insufficient_images_do_not_fall_back_to_absolute(
        self, n_episodes, n_images
    ):
        episodes = [
            _make_ep(f"ep_{i}", image_mode="blurry" if i < n_images else "none")
            for i in range(n_episodes)
        ]
        batch = _make_batch(episodes)
        result = BlurAnalyzer(sharpness_threshold=100.0).analyze(batch)
        assert result.model_dump() == BlurAnalyzer().analyze(batch).model_dump()
        assert result.flags[0].level == RiskLevel.INFO
        assert result.flags[0].observed.value is None

    def test_single_episode_without_threshold_requests_configuration(self):
        result = BlurAnalyzer().analyze(_make_batch([_make_ep(image_mode="blurry")]))
        assert result.flags[0].metric == "blurry_episode_fraction"
        assert result.flags[0].level == RiskLevel.INFO
        assert result.flags[0].observed.value is None
        assert "sharpness_threshold" in result.flags[0].implication

    def test_absolute_no_images_returns_info(self):
        result = BlurAnalyzer(sharpness_threshold=100.0).analyze(
            _make_batch([_make_ep(image_mode="none")])
        )
        assert result.flags[0].level == RiskLevel.INFO
        assert result.flags[0].metric == "blurry_episode_fraction"
        assert result.flags[0].observed.value is None

    def test_absolute_accepts_single_frame_episode(self):
        episode = _make_ep(n_steps=1, image_mode="blurry")
        result = BlurAnalyzer(sharpness_threshold=100.0).analyze(_make_batch([episode]))
        assert result.flags[0].observed.value == 1.0
        assert result.flags[0].level == RiskLevel.CRITICAL

    def test_absolute_uses_input_pixel_scale_and_strict_cutoff(self):
        episode = _make_ep(n_steps=2)
        # A unit checkerboard has Laplacian values -4/+4, so variance is exactly 16.
        checkerboard = (np.indices((4, 4)).sum(axis=0) % 2).astype(np.float32)
        episode.observations["camera_rgb"] = np.stack([checkerboard, checkerboard])
        batch = _make_batch([episode])
        assert BlurAnalyzer(sharpness_threshold=16.0).analyze(batch).flags[0].observed.value == 0.0
        assert BlurAnalyzer(sharpness_threshold=17.0).analyze(batch).flags[0].observed.value == 1.0
        # Re-running with a new cutoff must not reuse a previous classification.
        assert BlurAnalyzer(sharpness_threshold=16.0).analyze(batch).flags[0].observed.value == 0.0

    def test_absolute_rejects_nonfinite_images_instead_of_passing(self):
        episode = _make_ep(n_steps=2)
        episode.observations["camera_rgb"] = np.full((2, 4, 4), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite.*ep_0"):
            BlurAnalyzer(sharpness_threshold=100.0).analyze(_make_batch([episode]))

    def test_outlier_blurry_episode_flagged(self):
        episodes = [_make_ep(f"ep_{i}", image_mode="sharp") for i in range(6)]
        episodes.append(_make_ep("ep_blurry", image_mode="blurry"))
        batch = _make_batch(episodes)
        result = BlurAnalyzer().analyze(batch)
        flag = result.flags[0]
        assert flag.metric == "blurry_episode_fraction"
        assert flag.level in (RiskLevel.WARNING, RiskLevel.CRITICAL)
        assert "ep_blurry" in result.raw_metrics["outlier_episode_ids"]

    def test_uniformly_sharp_dataset_ok(self):
        episodes = [_make_ep(f"ep_{i}", image_mode="sharp") for i in range(6)]
        batch = _make_batch(episodes)
        result = BlurAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.OK

    def test_too_few_episodes_returns_info(self):
        episodes = [_make_ep(f"ep_{i}", image_mode="sharp") for i in range(3)]
        batch = _make_batch(episodes)
        result = BlurAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.INFO
        assert result.raw_metrics["skipped"] == "too few episodes with image data"

    def test_no_image_data_returns_info(self):
        episodes = [_make_ep(f"ep_{i}", image_mode="none") for i in range(6)]
        batch = _make_batch(episodes)
        result = BlurAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.INFO
        assert result.raw_metrics["skipped"] == "no image observations"

    def test_empty_batch_returns_no_flags(self):
        batch = _make_batch([])
        result = BlurAnalyzer().analyze(batch)
        assert result.flags == []

    def test_requires_images_capability(self):
        assert BlurAnalyzer.requires == frozenset({"images"})
