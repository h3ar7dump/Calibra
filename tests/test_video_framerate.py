"""Tests for VideoFramerateAnalyzer (calibra/analyzers/video_framerate.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calibra.analyzers.video_framerate import VideoFramerateAnalyzer, probe_video_fps
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel

# ── Helper to create synthetic video file ────────────────────────────────────


def _write_test_video(path: Path, n_frames: int = 30, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.random.default_rng(0).integers(0, 255, (n_frames, 16, 16, 3), dtype=np.uint8)
    try:
        import av

        container = av.open(str(path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for img in frames:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return
    except ImportError:
        pass

    try:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(path), fourcc, float(fps), (16, 16))
        for img in frames:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            vw.write(bgr)
        vw.release()
        return
    except ImportError:
        pass

    pytest.skip("Neither PyAV nor OpenCV is available for writing test videos.")


def _make_batch_with_video(video_path: Path, nominal_fps: float = 30.0, n_steps: int = 30) -> EpisodeBatch:
    dt = 1.0 / nominal_fps
    timestamps = np.arange(n_steps, dtype=np.float64) * dt
    actions = np.zeros((n_steps, 6), dtype=np.float32)
    ep = Episode(
        metadata=EpisodeMetadata(episode_id="ep_0"),
        timestamps=timestamps,
        observations={"camera_head": np.zeros((n_steps, 16, 16, 3), dtype=np.uint8)},
        actions=actions,
    )
    return EpisodeBatch(
        episodes=[ep],
        dataset_name="test_video_ds",
        format="lerobot",
        source_path=str(video_path.parent),
        extra={"fps": nominal_fps, "video_paths": [str(video_path)]},
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestVideoFramerateAnalyzer:
    def test_probe_video_fps(self, tmp_path):
        v_path = tmp_path / "test_30fps.mp4"
        _write_test_video(v_path, n_frames=30, fps=30)
        fps = probe_video_fps(v_path)
        assert fps is not None
        assert abs(fps - 30.0) < 0.5

    def test_matching_fps_reports_ok(self, tmp_path):
        v_path = tmp_path / "test_30fps.mp4"
        _write_test_video(v_path, n_frames=30, fps=30)
        batch = _make_batch_with_video(v_path, nominal_fps=30.0)

        analyzer = VideoFramerateAnalyzer()
        result = analyzer.analyze(batch)

        assert result.flags
        mismatch_flag = next((f for f in result.flags if f.metric == "video_fps_mismatch"), None)
        assert mismatch_flag is not None
        assert mismatch_flag.level == RiskLevel.OK

    def test_mismatched_fps_reports_critical(self, tmp_path):
        # Video is 15fps, but dataset nominal fps is 30fps -> 50% mismatch
        v_path = tmp_path / "test_15fps.mp4"
        _write_test_video(v_path, n_frames=15, fps=15)
        batch = _make_batch_with_video(v_path, nominal_fps=30.0)

        analyzer = VideoFramerateAnalyzer(max_mismatch_fraction=0.05)
        result = analyzer.analyze(batch)

        assert result.flags
        mismatch_flag = next((f for f in result.flags if f.metric == "video_fps_mismatch"), None)
        assert mismatch_flag is not None
        assert mismatch_flag.level == RiskLevel.CRITICAL
        assert "50.0%" in mismatch_flag.interpretation or "15" in mismatch_flag.interpretation

    def test_float_tolerance_tolerance_ok(self, tmp_path):
        # Video is 30fps, dataset is 29.97 (or vice versa)
        v_path = tmp_path / "test_30fps.mp4"
        _write_test_video(v_path, n_frames=30, fps=30)
        batch = _make_batch_with_video(v_path, nominal_fps=29.97)

        analyzer = VideoFramerateAnalyzer()
        result = analyzer.analyze(batch)

        mismatch_flag = next((f for f in result.flags if f.metric == "video_fps_mismatch"), None)
        assert mismatch_flag is not None
        assert mismatch_flag.level == RiskLevel.OK

    def test_no_videos_returns_info(self):
        ep = Episode(
            metadata=EpisodeMetadata(episode_id="ep_0"),
            timestamps=np.arange(10, dtype=np.float64) * 0.1,
            observations={"state": np.zeros((10, 4), dtype=np.float32)},
            actions=np.zeros((10, 6), dtype=np.float32),
        )
        batch = EpisodeBatch(
            episodes=[ep],
            dataset_name="empty",
            format="hdf5",
            source_path="/empty",
        )
        analyzer = VideoFramerateAnalyzer()
        result = analyzer.analyze(batch)
        assert result.flags
        assert result.flags[0].level == RiskLevel.INFO

    def test_pipeline_integrates_video_framerate(self, tmp_path):
        from calibra.pipeline import Pipeline

        v_path = tmp_path / "test_30fps.mp4"
        _write_test_video(v_path, n_frames=30, fps=30)
        batch = _make_batch_with_video(v_path, nominal_fps=30.0)

        pipeline = Pipeline(analyzers=[VideoFramerateAnalyzer()])
        report = pipeline.run(batch)
        assert any(r.analyzer_name == "video_framerate" for r in report.analyzer_results)

