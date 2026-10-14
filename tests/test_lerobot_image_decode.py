"""Tests for opt-in LeRobot v1/v2/v3 image & video decoding (calibra/ingestion/adapters/lerobot.py)."""

from __future__ import annotations

import io
import json
import weakref
from pathlib import Path

import numpy as np
import pytest

from calibra.analyzers.blur import BlurAnalyzer
from calibra.analyzers.camera_freeze import CameraFreezeAnalyzer
from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer
from calibra.ingestion.adapters.lerobot import (
    LeRobotReader,
    VideoDatasetResolver,
    _decode_image_column,
    _decode_video_frames,
)
from calibra.schema.report import RiskLevel

datasets = pytest.importorskip("datasets")
pyarrow = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

# ── fixtures & video writer helper ───────────────────────────────────────────


def _write_test_video(path: Path, frames: np.ndarray, fps: int = 30) -> None:
    """Write an array of uint8 RGB frames to an MP4 video file using PyAV or OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import av

        container = av.open(str(path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
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
        h, w = frames.shape[1], frames.shape[2]
        vw = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
        for img in frames:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            vw.write(bgr)
        vw.release()
        return
    except ImportError:
        pass

    pytest.skip("Neither PyAV nor OpenCV is available for writing test videos.")


def _make_v1_dataset(tmp_path, n_episodes: int = 6, n_steps: int = 10, freeze_ep: int = 0):
    """Build a small local v1 (HuggingFace `datasets`-saved) LeRobot dataset."""
    from datasets import Dataset, Features, Sequence, Value
    from datasets import Image as HFImage

    rng = np.random.default_rng(0)
    episode_index, frame_index, timestamp, action, images = [], [], [], [], []
    for ep in range(n_episodes):
        frames = rng.integers(0, 255, (n_steps, 16, 16, 3), dtype=np.uint8)
        if ep == freeze_ep:
            frames[:8] = frames[0]
        for step in range(n_steps):
            episode_index.append(ep)
            frame_index.append(step)
            timestamp.append(step * 0.05)
            action.append(rng.uniform(-1, 1, 6).astype(np.float32).tolist())
            images.append(frames[step])

    ds = Dataset.from_dict(
        {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "action": action,
            "observation.images.top": images,
        },
        features=Features(
            {
                "episode_index": Value("int64"),
                "frame_index": Value("int64"),
                "timestamp": Value("float32"),
                "action": Sequence(Value("float32"), length=6),
                "observation.images.top": HFImage(),
            }
        ),
    )
    path = tmp_path / "v1_dataset"
    ds.save_to_disk(str(path))
    return str(path)


def _make_v2_dataset(tmp_path, n_episodes: int = 6, n_steps: int = 10, freeze_ep: int = 0):
    """
    Build a small local v2 LeRobot dataset (meta/info.json + Parquet shards + per-episode videos).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "v2_dataset"
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v2.0",
        "fps": 30.0,
        "features": {
            "action": {"dtype": "float32", "shape": [6]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "observation.images.top": {
                "dtype": "video",
                "shape": [3, 16, 16],
                "names": ["channels", "height", "width"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(root / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    rng = np.random.default_rng(0)
    episode_index, frame_index, timestamp, actions, states = [], [], [], [], []

    for ep in range(n_episodes):
        frames = rng.integers(0, 255, (n_steps, 16, 16, 3), dtype=np.uint8)
        if ep == freeze_ep:
            frames[:8] = frames[0]
        # Write per-episode video
        video_path = (
            root / "videos" / "chunk-000" / "observation.images.top" / f"episode_{ep:06d}.mp4"
        )
        _write_test_video(video_path, frames, fps=30)

        for step in range(n_steps):
            episode_index.append(ep)
            frame_index.append(step)
            timestamp.append(step * 0.05)
            actions.append(rng.uniform(-1, 1, 6).astype(np.float32).tolist())
            states.append(rng.uniform(-1, 1, 6).astype(np.float32).tolist())

    table = pa.Table.from_pydict(
        {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "action": actions,
            "observation.state": states,
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")
    return str(root)


def _make_v3_dataset(tmp_path, n_episodes: int = 6, n_steps: int = 10, freeze_ep: int = 0):
    """
    Build a small local v3 LeRobot dataset (meta/info.json + episodes.jsonl + chunk-concatenated video).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "v3_dataset"
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "fps": 30.0,
        "features": {
            "action": {"dtype": "float32", "shape": [6]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "observation.images.top": {
                "dtype": "video",
                "shape": [3, 16, 16],
                "names": ["channels", "height", "width"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(root / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    rng = np.random.default_rng(0)
    all_frames = []
    episode_index, frame_index, timestamp, actions, states = [], [], [], [], []
    episodes_meta = []

    curr_frame_idx = 0
    for ep in range(n_episodes):
        frames = rng.integers(0, 255, (n_steps, 16, 16, 3), dtype=np.uint8)
        if ep == freeze_ep:
            frames[:8] = frames[0]
        all_frames.append(frames)

        episodes_meta.append(
            {
                "episode_index": ep,
                "length": n_steps,
                "videos/observation.images.top/from_frame": curr_frame_idx,
                "videos/observation.images.top/to_frame": curr_frame_idx + n_steps,
            }
        )
        curr_frame_idx += n_steps

        for step in range(n_steps):
            episode_index.append(ep)
            frame_index.append(step)
            timestamp.append(step * 0.05)
            actions.append(rng.uniform(-1, 1, 6).astype(np.float32).tolist())
            states.append(rng.uniform(-1, 1, 6).astype(np.float32).tolist())

    # Write chunk-concatenated video
    concatenated_video = np.concatenate(all_frames, axis=0)
    video_path = root / "videos" / "chunk-000" / "observation.images.top.mp4"
    _write_test_video(video_path, concatenated_video, fps=30)

    # Write episodes.jsonl
    with open(root / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for item in episodes_meta:
            f.write(json.dumps(item) + "\n")

    table = pa.Table.from_pydict(
        {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "action": actions,
            "observation.state": states,
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")
    return str(root)


# ── _decode_image_column & _decode_video_frames ─────────────────────────────


class TestDecodeImageColumn:
    def test_decodes_valid_png_bytes(self):
        from PIL import Image as PILImage

        img = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cells = [{"bytes": buf.getvalue(), "path": None}]
        result = _decode_image_column(cells)
        assert result is not None
        assert result.shape == (1, 4, 4, 3)
        assert result.dtype == np.uint8

    def test_non_image_cells_return_none(self):
        assert _decode_image_column([{"not": "an image"}]) is None
        assert _decode_image_column([42, 43]) is None


class TestVideoDatasetResolver:
    def test_resolve_v2_and_v3_paths(self, tmp_path):
        v2_path = Path(_make_v2_dataset(tmp_path))
        with open(v2_path / "meta" / "info.json", encoding="utf-8") as f:
            info_v2 = json.load(f)
        resolver_v2 = VideoDatasetResolver(v2_path, info_v2)
        v_path, s_f, n_f = resolver_v2.resolve_video_slice("observation.images.top", 0, ep_len=10)
        assert v_path is not None
        assert v_path.exists()
        assert s_f == 0
        assert n_f == 10

        v3_path = Path(_make_v3_dataset(tmp_path))
        with open(v3_path / "meta" / "info.json", encoding="utf-8") as f:
            info_v3 = json.load(f)
        resolver_v3 = VideoDatasetResolver(v3_path, info_v3)
        v_path3, s_f3, n_f3 = resolver_v3.resolve_video_slice(
            "observation.images.top", 1, ep_len=10
        )
        assert v_path3 is not None
        assert v_path3.exists()
        assert s_f3 == 10
        assert n_f3 == 10


class TestDecodeVideoFrames:
    def test_decode_video_slice(self, tmp_path):
        frames = np.random.default_rng(0).integers(0, 255, (20, 16, 16, 3), dtype=np.uint8)
        video_path = tmp_path / "sample.mp4"
        _write_test_video(video_path, frames, fps=30)

        # Full decode
        full = _decode_video_frames(video_path)
        assert full is not None
        assert full.shape == (20, 16, 16, 3)
        assert full.dtype == np.uint8

        # Slice decode
        sliced = _decode_video_frames(video_path, start_frame=5, num_frames=10)
        assert sliced is not None
        assert sliced.shape == (10, 16, 16, 3)
        assert sliced.dtype == np.uint8

    def test_nonexistent_video_returns_none(self, tmp_path):
        assert _decode_video_frames(tmp_path / "does_not_exist.mp4") is None


# ── LeRobot v1 Image Decoding ────────────────────────────────────────────────


class TestLeRobotV1ImageDecoding:
    def test_default_excludes_images(self, tmp_path):
        path = _make_v1_dataset(tmp_path)
        batch = LeRobotReader().read(path)
        obs_keys = set(batch.episodes[0].observations.keys())
        assert not any("image" in k.lower() or "camera" in k.lower() for k in obs_keys)
        assert "images" not in batch.capabilities

    def test_decode_images_true_includes_images(self, tmp_path):
        path = _make_v1_dataset(tmp_path)
        batch = LeRobotReader(decode_images=True).read(path)
        obs_keys = set(batch.episodes[0].observations.keys())
        image_keys = [k for k in obs_keys if "camera" in k.lower() or "image" in k.lower()]
        assert image_keys, f"expected an image observation key, got {obs_keys}"
        arr = batch.episodes[0].observations[image_keys[0]]
        assert arr.ndim == 4  # (T, H, W, C)
        assert arr.dtype == np.uint8
        assert "images" in batch.capabilities

    def test_end_to_end_image_analyzers_find_real_signal(self, tmp_path):
        path = _make_v1_dataset(tmp_path, freeze_ep=0)
        batch = LeRobotReader(decode_images=True).read(path)

        dup_result = DuplicateFrameAnalyzer().analyze(batch)
        freeze_result = CameraFreezeAnalyzer(min_freeze_run=5).analyze(batch)
        blur_result = BlurAnalyzer().analyze(batch)

        for result in (dup_result, freeze_result, blur_result):
            assert result.flags, f"{result.analyzer_name} produced no flags"
            assert result.flags[0].level != RiskLevel.INFO, (
                f"{result.analyzer_name} returned INFO — found no image data to work with"
            )
        assert freeze_result.flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)


# ── LeRobot v2 Video Decoding ────────────────────────────────────────────────


class TestLeRobotV2VideoDecoding:
    def test_v2_default_excludes_images(self, tmp_path):
        path = _make_v2_dataset(tmp_path)
        batch = LeRobotReader().read(path)
        obs_keys = set(batch.episodes[0].observations.keys())
        assert not any("image" in k.lower() or "camera" in k.lower() for k in obs_keys)
        assert "images" not in batch.capabilities

    def test_v2_decode_images_true(self, tmp_path):
        path = _make_v2_dataset(tmp_path)
        batch = LeRobotReader(decode_images=True).read(path)
        obs_keys = set(batch.episodes[0].observations.keys())
        image_keys = [k for k in obs_keys if "camera" in k.lower() or "image" in k.lower()]
        assert image_keys, f"expected an image observation key, got {obs_keys}"
        arr = batch.episodes[0].observations[image_keys[0]]
        assert arr.ndim == 4  # (T, H, W, C)
        assert arr.shape[0] == 10
        assert arr.dtype == np.uint8
        assert "images" in batch.capabilities

    def test_v2_end_to_end_analyzers(self, tmp_path):
        path = _make_v2_dataset(tmp_path, freeze_ep=0)
        batch = LeRobotReader(decode_images=True).read(path)

        dup_result = DuplicateFrameAnalyzer().analyze(batch)
        freeze_result = CameraFreezeAnalyzer(min_freeze_run=5).analyze(batch)
        blur_result = BlurAnalyzer().analyze(batch)

        for result in (dup_result, freeze_result, blur_result):
            assert result.flags, f"{result.analyzer_name} produced no flags"
            assert result.flags[0].level != RiskLevel.INFO
        assert freeze_result.flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)


# ── LeRobot v3 Video Decoding ────────────────────────────────────────────────


class TestLeRobotV3VideoDecoding:
    @pytest.mark.parametrize("version", [2, 3])
    @pytest.mark.parametrize("reader_path", ["duckdb", "pyarrow"])
    def test_streaming_reports_match_arrays_without_materializing_pixels(
        self, tmp_path, monkeypatch, version, reader_path
    ):
        from calibra.analyzers.camera_view import CameraViewMismatchAnalyzer
        from calibra.ingestion.adapters import lerobot

        path = (_make_v2_dataset if version == 2 else _make_v3_dataset)(tmp_path, freeze_ep=0)
        arrays = LeRobotReader(decode_images=True).read(path)
        analyzers = [
            DuplicateFrameAnalyzer(),
            CameraFreezeAnalyzer(),
            BlurAnalyzer(),
            CameraViewMismatchAnalyzer(),
        ]
        expected = [analyzer.analyze(arrays).model_dump() for analyzer in analyzers]
        calls = []
        decode = lerobot._iter_video_frames

        def track_decode(video_path, start_frame=0, num_frames=None):
            calls.append((str(video_path), start_frame, num_frames))
            yield from decode(video_path, start_frame, num_frames)

        def unexpected_materialization(*args):
            raise AssertionError("streaming QC must not build an episode image array")

        monkeypatch.setattr(lerobot, "_iter_video_frames", track_decode)
        monkeypatch.setattr(lerobot, "_decode_video_frames", unexpected_materialization)
        reader = LeRobotReader(decode_images=True, stream_images=True)
        if reader_path == "pyarrow":

            def missing_duckdb(*args):
                raise ImportError("DuckDB unavailable")

            monkeypatch.setattr(reader, "_build_duckdb_conn", missing_duckdb)
        batch = reader.read(path)
        assert calls == []
        for analyzer, result in zip(analyzers, expected):
            assert analyzer.analyze(batch).model_dump() == result
        assert len(calls) == batch.n_episodes  # Scalar caches survive episode eviction.
        for episode in batch.episodes:
            assert episode.image_streams
            assert not any("camera" in key for key in episode.observations)

    @pytest.mark.parametrize("reader_path", ["duckdb", "pyarrow", "stream"])
    @pytest.mark.parametrize("task_format", ["parquet", "jsonl"])
    def test_multitask_video_reading_preserves_lazy_decoding(
        self, tmp_path, monkeypatch, reader_path, task_format
    ):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from calibra.ingestion.adapters import lerobot

        path = _make_v3_dataset(tmp_path)
        root = Path(path)
        data_path = root / "data" / "chunk-000" / "episode_000000.parquet"
        data = pq.read_table(data_path)
        task_indices = [int(ep_id) % 2 for ep_id in data["episode_index"].to_pylist()]
        pq.write_table(data.append_column("task_index", pa.array(task_indices)), data_path)
        tasks = ["pick up the red block", "open the drawer"]
        if task_format == "parquet":
            pq.write_table(
                pa.table({"task_index": [0, 1], "task": tasks}), root / "meta" / "tasks.parquet"
            )
        else:
            (root / "meta" / "tasks.jsonl").write_text(
                "".join(
                    json.dumps({"task_index": i, "task": task}) + "\n"
                    for i, task in enumerate(tasks)
                )
            )

        decoded_slices = []
        decode = lerobot._decode_video_frames

        def track_decode(video_path, start_frame=0, num_frames=None):
            decoded_slices.append((start_frame, num_frames))
            return decode(video_path, start_frame, num_frames)

        monkeypatch.setattr(lerobot, "_decode_video_frames", track_decode)
        reader = LeRobotReader(decode_images=True)
        if reader_path == "pyarrow":

            def missing_duckdb(*args):
                raise ImportError("DuckDB is unavailable")

            monkeypatch.setattr(reader, "_build_duckdb_conn", missing_duckdb)
        if reader_path == "stream":
            episodes = reader.iter_episodes_lazy(path)
        else:
            batch = reader.read(path)
            assert batch.extra["fps"] == 30.0
            episodes = iter(batch.episodes)
        assert decoded_slices == []

        previous_frames = None
        descriptions = []
        for ep in episodes:
            if previous_frames is not None:
                assert previous_frames() is None, "Previous episode pixels stayed in memory"
            descriptions.append(ep.metadata.task_description)
            camera_key = next(key for key in ep.observations if "camera" in key)
            frames = ep.observations[camera_key]
            assert frames.shape == (10, 16, 16, 3)
            previous_frames = weakref.ref(frames)
            del frames
        assert descriptions == tasks * 3
        assert decoded_slices == [(i * 10, 10) for i in range(6)]

    def test_v3_default_excludes_images(self, tmp_path):
        path = _make_v3_dataset(tmp_path)
        batch = LeRobotReader().read(path)
        obs_keys = set(batch.episodes[0].observations.keys())
        assert not any("image" in k.lower() or "camera" in k.lower() for k in obs_keys)
        assert "images" not in batch.capabilities

    def test_v3_decode_images_with_episodes_jsonl(self, tmp_path):
        path = _make_v3_dataset(tmp_path)
        batch = LeRobotReader(decode_images=True).read(path)
        obs_keys = set(batch.episodes[0].observations.keys())
        image_keys = [k for k in obs_keys if "camera" in k.lower() or "image" in k.lower()]
        assert image_keys, f"expected an image observation key, got {obs_keys}"
        arr = batch.episodes[0].observations[image_keys[0]]
        assert arr.ndim == 4  # (T, H, W, C)
        assert arr.shape[0] == 10
        assert arr.dtype == np.uint8
        assert "images" in batch.capabilities

    def test_v3_lazy_iter_decode_images(self, tmp_path):
        path = _make_v3_dataset(tmp_path)
        reader = LeRobotReader(decode_images=True)
        episodes = list(reader.iter_episodes_lazy(path))
        assert len(episodes) == 6
        for ep in episodes:
            image_keys = [
                k for k in ep.observations if "camera" in k.lower() or "image" in k.lower()
            ]
            assert image_keys
            assert ep.observations[image_keys[0]].shape == (10, 16, 16, 3)

    def test_v3_end_to_end_analyzers(self, tmp_path):
        path = _make_v3_dataset(tmp_path, freeze_ep=0)
        batch = LeRobotReader(decode_images=True).read(path)

        dup_result = DuplicateFrameAnalyzer().analyze(batch)
        freeze_result = CameraFreezeAnalyzer(min_freeze_run=5).analyze(batch)
        blur_result = BlurAnalyzer().analyze(batch)

        for result in (dup_result, freeze_result, blur_result):
            assert result.flags, f"{result.analyzer_name} produced no flags"
            assert result.flags[0].level != RiskLevel.INFO
        assert freeze_result.flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)

    def test_progress_callback_invocations(self, tmp_path):
        calls = []

        def callback(current: int, total: int, message: str) -> None:
            calls.append((current, total, message))

        path = _make_v2_dataset(tmp_path)
        reader = LeRobotReader(decode_images=True, progress_callback=callback)
        batch = reader.read(path)
        episodes = list(batch.episodes)
        assert len(episodes) == 6
        assert len(calls) > 0
        # Final call should report current == total == 6
        assert calls[-1][0] == 6
        assert calls[-1][1] == 6
