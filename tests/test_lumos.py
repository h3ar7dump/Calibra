"""Lumos reader contract: recordings, native clocks, segments and CLI integration."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from calibra.ingestion.adapters.lumos import LumosReader
from calibra.ingestion.registry import detect_reader, load
from calibra.integrity import run_integrity
from calibra.pipeline import Pipeline
from calibra.score import compute_score

ORIGIN = 1_790_580_978_188_457


def make_episode(root: Path, name: str = "EGO_test", indices=(1,)) -> Path:
    episode = root / name
    episode.mkdir()
    streams = {
        "color_left": {"enabled": True, "fps": 30, "segments": []},
        "color_right": {"enabled": True, "fps": 30, "segments": []},
        "imu": {"enabled": True, "segments": []},
        "audio": {
            "enabled": True,
            "sample_rate_hz": 48000,
            "channels": 1,
            "bits_per_sample": 16,
            "segments": [],
        },
    }
    for index in indices:
        part = episode / f"part{index:04d}"
        part.mkdir()
        start = ORIGIN + (index - 1) * 60_000_000
        for camera in ("left", "right"):
            prefix = f"{name}_camera_{camera}_part{index:04d}"
            media = part / f"{prefix}.mp4"
            media.touch()  # Header/pixels are only accessed with decode_images=True.
            pts = part / f"{prefix}_pts.csv"
            offset = 1_000_000 + (100 if camera == "right" else 0)
            pts.write_text(
                "timestamp_us\n"
                + "\n".join(str(start + offset + i * 33_333) for i in range(3))
                + "\n"
            )
            streams[f"color_{camera}"]["segments"].append(
                {
                    "index": index,
                    "file": media.relative_to(episode).as_posix(),
                    "pts_file": pts.relative_to(episode).as_posix(),
                    "frame_count": 3,
                    "start_time_sec": 0,
                    "end_time_sec": 0,
                }
            )
        imu = part / f"{name}_imu_part{index:04d}.csv"
        imu.write_text(
            "timestamp_us,x,y,z,type\n"
            + "".join(
                f"{start + 900_000 + i * 1000},1,2,3,accel\n"
                f"{start + 900_000 + i * 1000},4,5,6,gyro\n"
                for i in range(4)
            )
        )
        streams["imu"]["segments"].append(
            {
                "index": index,
                "file": imu.relative_to(episode).as_posix(),
                "frame_count": 4,
            }
        )
        audio = part / f"{name}_audio_part{index:04d}.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(np.array([0, 16384, -32768, 32767] * 1024, dtype="<i2").tobytes())
        pts = part / f"{audio.stem}_pts.csv"
        pts.write_text(f"timestamp_us\n{start}\n{start + 42667}\n")
        streams["audio"]["segments"].append(
            {
                "index": index,
                "file": audio.relative_to(episode).as_posix(),
                "pts_file": pts.relative_to(episode).as_posix(),
                "frame_count": 2,
            }
        )
    calibration = episode / f"{name}_calibration_camera.yaml"
    calibration.write_text("calibration_info:\n  format_version: 1.0\n")
    metadata = episode / f"meta_{name}.json"
    metadata.write_text(
        json.dumps(
            {
                "device": {"name": "EGO"},
                "timebase": {"unit": "microsecond"},
                "files": {"calibration_camera": calibration.name},
                "streams": streams,
            }
        )
    )
    return episode


def edit_metadata(episode: Path, edit) -> None:
    path = next(episode.glob("meta_*.json"))
    metadata = json.loads(path.read_text())
    edit(metadata)
    path.write_text(json.dumps(metadata))


@pytest.mark.parametrize("entry", ["episode", "metadata", "dataset"])
def test_loads_recording_through_registry(tmp_path, entry):
    episode = make_episode(tmp_path)
    path = {"episode": episode, "metadata": next(episode.glob("meta_*.json")), "dataset": tmp_path}[
        entry
    ]
    assert detect_reader(str(path)) is LumosReader
    batch = load(str(path))
    assert batch.format == "lumos"
    assert batch.n_episodes == 1
    assert batch.n_samples == 3
    ep = batch.episodes[0]
    assert ep.metadata.episode_id == "EGO_test"
    assert ep.metadata.extra["timestamp_origin_us"] == ORIGIN
    np.testing.assert_allclose(ep.timestamps, [1, 1.033333, 1.066666], atol=1e-10)
    np.testing.assert_allclose(
        ep.obs_timestamps["camera_right"], [1.0001, 1.033433, 1.066766], atol=1e-10
    )
    np.testing.assert_allclose(ep.obs_timestamps["imu_accel"], [0.9, 0.901, 0.902, 0.903])
    np.testing.assert_array_equal(ep.observations["imu_accel"], [[1, 2, 3]] * 4)
    np.testing.assert_array_equal(ep.observations["imu_gyro"], [[4, 5, 6]] * 4)
    assert ep.actions.shape == (0, 0)
    assert "actions" not in batch.capabilities
    assert "images" not in batch.capabilities
    assert len(batch.extra["video_paths"]) == 2
    assert ep.metadata.extra["audio_segments"][0]["n_samples"] == 4096
    assert Path(ep.metadata.extra["calibration_paths"]["calibration_camera"]).is_file()


def test_reads_dataset_in_episode_order(tmp_path):
    make_episode(tmp_path, "EGO_b")
    make_episode(tmp_path, "EGO_a")
    batch = LumosReader().read(str(tmp_path))
    assert [ep.metadata.episode_id for ep in batch.episodes] == ["EGO_a", "EGO_b"]
    assert batch.n_samples == 6


def test_orders_parts_and_preserves_gaps_on_shared_clock(tmp_path):
    episode = make_episode(tmp_path, indices=(2, 1))
    ep = LumosReader().read(str(episode)).episodes[0]
    assert ep.n_steps == 6
    np.testing.assert_allclose(ep.timestamps, [1, 1.033333, 1.066666, 61, 61.033333, 61.066666])
    np.testing.assert_allclose(ep.obs_timestamps["audio_pts"], [0, 0.042667, 60, 60.042667])
    assert ep.observations["imu_accel"].shape == (8, 3)
    assert [s["index"] for s in ep.metadata.extra["segments"]["color_left"]] == [1, 2]


def test_decodes_audio_and_retains_block_clock(tmp_path):
    episode = make_episode(tmp_path, indices=(1, 2))
    ep = LumosReader(decode_audio=True).read(str(episode)).episodes[0]
    assert ep.observations["audio"].shape == (8192, 1)
    np.testing.assert_allclose(ep.observations["audio"][:3, 0], [0, 0.5, -1])
    np.testing.assert_allclose(ep.obs_timestamps["audio"][[0, 1, 4096]], [0, 1 / 48000, 60])
    assert len(ep.obs_timestamps["audio_pts"]) == 4


def test_uses_imu_clock_when_video_and_audio_are_disabled(tmp_path):
    episode = make_episode(tmp_path)

    def disable(metadata):
        for name in ("audio", "color_left", "color_right"):
            metadata["streams"][name]["enabled"] = False

    edit_metadata(episode, disable)
    batch = load(str(episode))
    ep = batch.episodes[0]
    assert ep.metadata.extra["master_clock"] == "imu_accel"
    np.testing.assert_allclose(ep.timestamps, [0, 0.001, 0.002, 0.003])
    assert not batch.extra["video_paths"]


def test_supports_legacy_single_file_stream_entries(tmp_path):
    episode = make_episode(tmp_path)

    def flatten(metadata):
        for stream in metadata["streams"].values():
            stream.update(stream.pop("segments")[0])

    edit_metadata(episode, flatten)
    assert load(str(episode)).n_samples == 3


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing_file", "Missing Lumos file"),
        ("count", "frame_count mismatch"),
        ("index", "disagrees"),
        ("duplicate_index", "Duplicate"),
        ("unit", "microseconds"),
        ("path_escape", "escapes episode directory"),
    ],
)
def test_rejects_inconsistent_metadata(tmp_path, problem, match):
    episode = make_episode(tmp_path)

    def corrupt(metadata):
        segment = metadata["streams"]["color_left"]["segments"][0]
        if problem == "missing_file":
            (episode / segment["file"]).unlink()
        elif problem == "count":
            segment["frame_count"] = 100
        elif problem == "index":
            segment["index"] = 2
        elif problem == "duplicate_index":
            metadata["streams"]["color_left"]["segments"].append(dict(segment))
        elif problem == "unit":
            metadata["timebase"]["unit"] = "nanosecond"
        else:
            segment["file"] = "../outside.mp4"

    edit_metadata(episode, corrupt)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        LumosReader().read(str(episode))


def test_rejects_overlapping_parts(tmp_path):
    episode = make_episode(tmp_path, indices=(1, 2))
    pts = next((episode / "part0002").glob("*camera_left*_pts.csv"))
    pts.write_text(f"timestamp_us\n{ORIGIN}\n{ORIGIN + 33333}\n{ORIGIN + 66666}\n")
    with pytest.raises(ValueError, match="Overlapping"):
        LumosReader().read(str(episode))


@pytest.mark.parametrize("decode_audio", [True, False])
def test_audio_sample_overlap_after_last_block_pts(tmp_path, decode_audio):
    episode = make_episode(tmp_path, indices=(1, 2))
    pts = next((episode / "part0002").glob("*audio*_pts.csv"))
    # 60 ms is after the final 42.667 ms block PTS but inside the first WAV's
    # 85.333 ms sample span. Checking only block clocks would miss the overlap.
    pts.write_text(f"timestamp_us\n{ORIGIN + 60000}\n{ORIGIN + 102667}\n")
    reader = LumosReader(decode_audio=decode_audio)
    if decode_audio:
        with pytest.raises(ValueError, match="Overlapping WAV sample clocks"):
            reader.read(str(episode))
    else:
        batch = reader.read(str(episode))
        warning = batch.extra["ingestion_warnings"][0]
        assert warning["overlap_us"] == pytest.approx(25312.5)
        assert warning["episode_id"] == "EGO_test"
        assert warning["previous_segment_index"] == 1
        assert warning["segment_index"] == 2
        assert warning["level"] == "WARNING"
        assert "audio" not in batch.episodes[0].observations
        np.testing.assert_allclose(
            batch.episodes[0].obs_timestamps["audio_pts"], [0, 0.042667, 0.06, 0.102667]
        )


def test_nominal_audio_overlap_with_disjoint_device_metadata(tmp_path):
    episode = make_episode(tmp_path, indices=(1, 2))
    audio = next((episode / "part0001").glob("*.wav"))
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(np.zeros(4800, dtype="<i2").tobytes())
    pts = next((episode / "part0002").glob("*audio*_pts.csv"))
    pts.write_text(f"timestamp_us\n{ORIGIN + 99760}\n{ORIGIN + 142427}\n")

    def add_device_times(metadata):
        first, second = metadata["streams"]["audio"]["segments"]
        first.update(start_timestamp_us=str(ORIGIN), end_timestamp_us=str(ORIGIN + 99739))
        second.update(start_timestamp_us=str(ORIGIN + 99760))

    edit_metadata(episode, add_device_times)
    metadata_file = next(episode.glob("meta_*.json"))
    original = metadata_file.read_bytes()
    batch = LumosReader().read(str(episode))
    warning = batch.extra["ingestion_warnings"][0]
    assert warning["metric"] == "wav_sample_clock_overlap"
    assert warning["overlap_us"] == pytest.approx(219.1666666667)
    assert metadata_file.read_bytes() == original
    assert batch.episodes[0].metadata.extra["audio_segments"][0]["n_samples"] == 4800
    assert batch.episodes[0].metadata.extra["ingestion_warnings"] == [warning]


@pytest.mark.parametrize("decode_audio", [False, True])
def test_audio_block_overlap_remains_a_structural_error(tmp_path, decode_audio):
    episode = make_episode(tmp_path, indices=(1, 2))
    pts = next((episode / "part0002").glob("*audio*_pts.csv"))
    pts.write_text(f"timestamp_us\n{ORIGIN + 40000}\n{ORIGIN + 82667}\n")
    with pytest.raises(ValueError, match="non-increasing audio_pts"):
        LumosReader(decode_audio=decode_audio).read(str(episode))


def test_ingestion_warnings_survive_pipeline_serialization(tmp_path):
    episode = make_episode(tmp_path, indices=(1, 2))
    pts = next((episode / "part0002").glob("*audio*_pts.csv"))
    pts.write_text(f"timestamp_us\n{ORIGIN + 60000}\n{ORIGIN + 102667}\n")
    from calibra.schema.report import DiagnosticReport

    report = Pipeline(analyzers=[]).analyze_path(str(episode))
    restored = DiagnosticReport.model_validate_json(report.model_dump_json())
    assert restored.ingestion_warnings == report.ingestion_warnings
    assert restored.ingestion_warnings[0]["overlap_us"] == pytest.approx(25312.5)
    assert restored.analyzer_results == []
    assert "wav_sample_clock_overlap" in report.summary()


def test_clean_audio_has_no_ingestion_warnings(tmp_path):
    episode = make_episode(tmp_path, indices=(1, 2))
    batch = LumosReader().read(str(episode))
    assert batch.extra["ingestion_warnings"] == []
    assert all(ep.metadata.extra["ingestion_warnings"] == [] for ep in batch.episodes)
    assert Pipeline(analyzers=[]).run(batch).ingestion_warnings == []


def test_reader_subclasses_can_return_episodes_without_warning_metadata(tmp_path):
    class ExistingReader(LumosReader):
        def _read_episode(self, file):
            episode = super()._read_episode(file)
            episode.metadata.extra.pop("ingestion_warnings")
            return episode

    episode = make_episode(tmp_path)
    assert ExistingReader().read(str(episode)).extra["ingestion_warnings"] == []


@pytest.mark.parametrize("as_json", [False, True])
def test_integrity_exposes_audio_ingestion_warnings(tmp_path, capsys, as_json):
    episode = make_episode(tmp_path, indices=(1, 2))
    pts = next((episode / "part0002").glob("*audio*_pts.csv"))
    pts.write_text(f"timestamp_us\n{ORIGIN + 60000}\n{ORIGIN + 102667}\n")
    # Keep the master clock uniform so the audio warning is the only new
    # finding, rather than also triggering the fixture's 60-second video gap.
    for camera in ("left", "right"):
        pts = next((episode / "part0002").glob(f"*camera_{camera}*_pts.csv"))
        start = ORIGIN + 1_000_000 + 99999 + (100 if camera == "right" else 0)
        pts.write_text("timestamp_us\n" + "".join(f"{start + i * 33333}\n" for i in range(3)))
    args = [str(episode), "--format", "lumos"] + (["--json"] if as_json else [])
    with pytest.raises(SystemExit) as result:
        run_integrity(args)
    assert result.value.code == 0
    output = capsys.readouterr().out
    if as_json:
        assert json.loads(output)["ingestion_warnings"][0]["overlap_us"] > 0
    else:
        assert "wav_sample_clock_overlap" in output


@pytest.mark.parametrize("value", ["nan", "inf", "bad"])
def test_rejects_invalid_imu_values(tmp_path, value):
    episode = make_episode(tmp_path)
    imu = next((episode / "part0001").glob("*imu*.csv"))
    imu.write_text(imu.read_text().replace(",1,2,3,accel", f",{value},2,3,accel"))
    with pytest.raises(ValueError, match="IMU values"):
        LumosReader().read(str(episode))


def test_does_not_claim_unrelated_json_directories(tmp_path):
    (tmp_path / "meta_other.json").write_text('{"streams": {"other": {}}}')
    assert not LumosReader.can_read(str(tmp_path))
    assert not LumosReader.can_read(str(tmp_path / "missing"))


def test_pipeline_skips_action_diagnostics_and_score(tmp_path):
    episode = make_episode(tmp_path)
    report = Pipeline(world_model=True).run(load(str(episode)))
    assert {
        "control_smoothness",
        "phase_balance",
        "influence",
        "ssl_embed",
        "world_model_consistency",
    } <= set(report.skipped_analyzers)
    assert not any("action_dropout" in flag.metric for flag in report.flags)
    with pytest.raises(ValueError, match="requires robot action data"):
        compute_score(report)


def test_retains_episode_length_checks_without_actions(tmp_path):
    for name in ("EGO_a", "EGO_b", "EGO_c", "EGO_d"):
        make_episode(tmp_path, name)
    report = Pipeline().run(load(str(tmp_path)))
    coverage = next(r for r in report.analyzer_results if r.analyzer_name == "coverage_entropy")
    assert coverage.raw_metrics["episode_lengths"]["mean_steps"] == 3
    assert (
        next(f for f in coverage.flags if f.metric == "action_entropy_bits_per_dim").observed.value
        is None
    )
    structure = next(r for r in report.analyzer_results if r.analyzer_name == "task_structure")
    assert structure.raw_metrics["contact_density"]["skipped"] == "no action data"
    assert any(f.metric == "short_episode_fraction" for f in structure.flags)


def test_disabled_camera_files_are_excluded_from_video_checks(tmp_path):
    episode = make_episode(tmp_path)
    for path in episode.glob("part*/*.mp4"):
        write_video(path)

    def disable(metadata):
        for name in ("color_left", "color_right"):
            metadata["streams"][name]["enabled"] = False

    edit_metadata(episode, disable)
    report = Pipeline().run(load(str(episode)))
    fps_flag = next(f for f in report.flags if f.metric == "video_fps_mismatch")
    assert fps_flag.observed.value is None


def test_integrity_cli_supports_forced_format(tmp_path, capsys):
    episode = make_episode(tmp_path)
    with pytest.raises(SystemExit) as result:
        run_integrity([str(episode), "--format", "lumos", "--json"])
    assert result.value.code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["format"] == "lumos"
    assert {"check": "control_smoothness", "reason": "requires: actions"} in output["not_evaluated"]


def write_video(path: Path, count=3, width=32, height=24, constant=False):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(
                np.full((height, width, 3), 0 if constant else i * 50, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_decodes_all_camera_frames_with_memory_limit(tmp_path):
    episode = make_episode(tmp_path, indices=(1, 2))
    for path in episode.glob("part*/*.mp4"):
        write_video(path)
    batch = LumosReader(decode_images=True, max_image_size=16).read(str(episode))
    assert "images" in batch.capabilities
    for camera in ("camera_left", "camera_right"):
        assert batch.episodes[0].observations[camera].shape == (6, 12, 16, 3)


def test_rejects_video_pts_count_mismatch(tmp_path):
    episode = make_episode(tmp_path)
    for path in episode.glob("part*/*.mp4"):
        write_video(path, count=2)
    with pytest.raises(ValueError, match="video/PTS frame mismatch"):
        LumosReader(decode_images=True).read(str(episode))


def test_streaming_lumos_preserves_resolution_and_segment_boundaries(tmp_path, monkeypatch):
    from calibra.analyzers.camera_freeze import CameraFreezeAnalyzer
    from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer

    episode = make_episode(tmp_path, indices=(1, 2))
    for path in episode.glob("part*/*.mp4"):
        write_video(path, width=320, height=260, constant=True)
    arrays = LumosReader(decode_images=True).read(str(episode))
    assert arrays.episodes[0].observations["camera_left"].shape == (6, 260, 320, 3)
    calls = []
    reader = LumosReader(decode_images=True, stream_images=True)
    decode = reader._iter_video

    def track_decode(path, expected):
        calls.append(path)
        for frame in decode(path, expected):
            assert frame.shape == (260, 320, 3)
            yield frame

    def unexpected_materialization(*args):
        raise AssertionError("streaming Lumos QC must not concatenate video frames")

    monkeypatch.setattr(reader, "_iter_video", track_decode)
    monkeypatch.setattr(reader, "_read_video", unexpected_materialization)
    streams = reader.read(str(episode))
    assert calls == []
    assert "images" in streams.capabilities
    assert not any("camera" in key for key in streams.episodes[0].observations)
    for analyzer in (DuplicateFrameAnalyzer(), CameraFreezeAnalyzer()):
        assert analyzer.analyze(streams).model_dump() == analyzer.analyze(arrays).model_dump()
    assert CameraFreezeAnalyzer().analyze(streams).raw_metrics["freeze_episodes"] == [
        {"episode_id": "EGO_test", "run_length": 5}
    ]
    assert len(calls) == 2  # One camera, both segments, shared by the two analyzers.


def test_streaming_lumos_rejects_each_segment_frame_count_mismatch(tmp_path):
    from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer

    episode = make_episode(tmp_path, indices=(1, 2))
    for path in episode.glob("part*/*.mp4"):
        write_video(path, count=2 if path.parent.name == "part0001" else 4)
    batch = LumosReader(decode_images=True, stream_images=True).read(str(episode))
    with pytest.raises(ValueError, match="video/PTS frame mismatch"):
        DuplicateFrameAnalyzer().analyze(batch)


def test_integrity_validates_camera_not_selected_by_metrics(tmp_path, capsys):
    episode = make_episode(tmp_path)
    for path in episode.glob("part*/*.mp4"):
        write_video(path, count=2 if "camera_right" in path.name else 3)
    with pytest.raises(SystemExit) as result:
        run_integrity([str(episode), "--decode-images", "--json"])
    assert result.value.code == 1
    assert "video/PTS frame mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("forced", [True, False])
def test_integrity_cli_decodes_lumos_without_forcing_lerobot(tmp_path, capsys, monkeypatch, forced):
    episode = make_episode(tmp_path)
    for path in episode.glob("part*/*.mp4"):
        write_video(path)

    def unexpected_materialization(*args):
        raise AssertionError("integrity must stream Lumos camera frames")

    monkeypatch.setattr(LumosReader, "_read_video", unexpected_materialization)
    args = [str(episode), "--decode-images", "--json"]
    if forced:
        args += ["--format", "lumos"]
    with pytest.raises(SystemExit) as result:
        run_integrity(args)
    assert result.value.code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["format"] == "lumos"
    assert not any(item["reason"] == "requires: images" for item in output["not_evaluated"])
