"""Streaming camera metrics preserve array results without retaining pixels."""

import weakref

import numpy as np
import pytest

from calibra.analyzers.blur import BlurAnalyzer, compute_laplacian_variance
from calibra.analyzers.camera_freeze import CameraFreezeAnalyzer
from calibra.analyzers.camera_view import CameraViewMismatchAnalyzer
from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer, repeated_transitions
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.images import ImageStream
from calibra.temporal.drift import compute_visual_activity
from calibra.vision import measure_image_stream


@pytest.mark.parametrize("length", [17, 65, 97])
def test_stream_metrics_match_arrays_and_release_frames(length):
    images = np.random.default_rng(4).integers(0, 256, (length, 12, 16, 3), dtype=np.uint8)
    images[5:12] = images[4]  # A freeze crossing the simulated segment boundary.
    references = []
    calls = []

    def frames():
        calls.append(True)
        for start, stop in ((0, 8), (8, length)):
            for image in images[start:stop]:
                frame = image.copy()
                references.append(weakref.ref(frame))
                yield frame
                # Only the previous frame may survive the next yield.
                assert sum(ref() is not None for ref in references) <= 2

    stream = ImageStream(frames, length)
    metrics = measure_image_stream(stream)
    np.testing.assert_array_equal(metrics.repeated, repeated_transitions(images))
    np.testing.assert_array_equal(metrics.visual_activity, compute_visual_activity(images))
    np.testing.assert_array_equal(metrics.sharpness, compute_laplacian_variance(images))
    assert all(ref() is None for ref in references)
    assert measure_image_stream(stream) is metrics
    assert len(calls) == 1


def make_batch(images, *, streaming):
    episodes = []
    for index in range(4):
        episode = Episode(
            EpisodeMetadata(str(index)),
            np.arange(len(images)) / 30,
            {} if streaming else {"camera_left": images},
            np.random.default_rng(index).normal(size=(len(images), 3)),
        )
        if streaming:
            episode.image_streams["camera_left"] = ImageStream(lambda: iter(images), len(images))
        episodes.append(episode)
    return EpisodeBatch(episodes, "stream_test", "lerobot", "unused")


def test_analyzer_reports_match_materialized_images():
    images = np.random.default_rng(0).integers(0, 256, (65, 12, 16, 3), dtype=np.uint8)
    images[5:12] = images[4]
    arrays = make_batch(images, streaming=False)
    streams = make_batch(images, streaming=True)
    assert "images" in streams.capabilities
    for analyzer in (
        DuplicateFrameAnalyzer(),
        CameraFreezeAnalyzer(),
        BlurAnalyzer(),
        CameraViewMismatchAnalyzer(),
    ):
        assert analyzer.analyze(streams).model_dump() == analyzer.analyze(arrays).model_dump()


def test_absolute_blur_preserves_all_frame_mean():
    images = np.random.default_rng(0).integers(0, 256, (65, 12, 16, 3), dtype=np.uint8)
    arrays = make_batch(images, streaming=False)
    streams = make_batch(images, streaming=True)
    arrays.episodes = arrays.episodes[:1]
    streams.episodes = streams.episodes[:1]
    analyzer = BlurAnalyzer(sharpness_threshold=100)
    assert analyzer.analyze(streams).model_dump() == analyzer.analyze(arrays).model_dump()


def test_temporal_drift_matches_materialized_images():
    images = np.random.default_rng(0).integers(0, 256, (65, 12, 16, 3), dtype=np.uint8)
    arrays = make_batch(images, streaming=False)
    streams = make_batch(images, streaming=True)
    for original, streamed in zip(arrays.episodes, streams.episodes):
        velocity = np.random.default_rng(3).normal(size=(len(images), 3))
        original.observations["joint_velocity"] = velocity
        streamed.observations["joint_velocity"] = velocity
    analyzer = TemporalAnalyzer()
    assert analyzer.analyze(streams).model_dump() == analyzer.analyze(arrays).model_dump()


def test_stream_rejects_frame_count_mismatch():
    stream = ImageStream(lambda: iter([np.zeros((12, 16, 3), dtype=np.uint8)]), 2)
    with pytest.raises(ValueError, match="frame count"):
        measure_image_stream(stream)


@pytest.mark.parametrize("tolerance,fraction", [(0.0, 5e-4), (2.0, 0.01), (7.5, 0.5)])
def test_stream_repeat_settings_match_array_computation(tolerance, fraction):
    images = np.random.default_rng(2).integers(0, 20, (9, 12, 16, 3), dtype=np.uint8)
    images[3:5] = images[2]
    stream = ImageStream(lambda: iter(images), len(images))
    metrics = measure_image_stream(stream, tolerance, fraction)
    np.testing.assert_array_equal(
        metrics.repeated, repeated_transitions(images, tolerance, fraction)
    )


def test_stream_closes_decoder_on_invalid_frame_without_caching():
    closed = []

    def frames():
        try:
            yield np.zeros((12, 16, 3), dtype=np.float32)
        finally:
            closed.append(True)

    stream = ImageStream(frames, 1)
    with pytest.raises(ValueError, match="RGB uint8"):
        measure_image_stream(stream)
    assert closed == [True]
    assert not stream.validated
    assert stream.metric_cache == {}


def test_temporal_stream_preserves_minimum_spatial_dimensions():
    images = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    arrays = make_batch(images, streaming=False)
    streams = make_batch(images, streaming=True)
    for original, streamed in zip(arrays.episodes, streams.episodes):
        velocity = np.random.default_rng(2).normal(size=(len(images), 3))
        original.observations["joint_velocity"] = velocity
        streamed.observations["joint_velocity"] = velocity
    analyzer = TemporalAnalyzer()
    assert analyzer.analyze(streams).model_dump() == analyzer.analyze(arrays).model_dump()


def test_camera_view_skips_insufficient_episodes_without_decoding():
    images = np.zeros((10, 12, 16, 3), dtype=np.uint8)
    batch = make_batch(images, streaming=True)
    batch.episodes = batch.episodes[:1]

    def unexpected_decode():
        raise AssertionError("insufficient episodes must not decode video")

    batch.episodes[0].image_streams["camera_left"] = ImageStream(unexpected_decode, 10)
    result = CameraViewMismatchAnalyzer().analyze(batch)
    assert result.flags[0].observed.value is None
