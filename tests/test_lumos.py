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
def test_rejects_audio_sample_overlap_after_last_block_pts(tmp_path, decode_audio):
    episode = make_episode(tmp_path, indices=(1, 2))
    pts = next((episode / "part0002").glob("*audio*_pts.csv"))
    # 60 ms is after the final 42.667 ms block PTS but inside the first WAV's
    # 85.333 ms sample span. Checking only block clocks would miss the overlap.
    pts.write_text(f"timestamp_us\n{ORIGIN + 60000}\n{ORIGIN + 102667}\n")
    with pytest.raises(ValueError, match="Overlapping WAV sample clocks"):
        LumosReader(decode_audio=decode_audio).read(str(episode))


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


def write_video(path: Path, count=3):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), i * 50, dtype=np.uint8), format="rgb24"
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


@pytest.mark.parametrize("forced", [True, False])
def test_integrity_cli_decodes_lumos_without_forcing_lerobot(tmp_path, capsys, forced):
    episode = make_episode(tmp_path)
    for path in episode.glob("part*/*.mp4"):
        write_video(path)
    args = [str(episode), "--decode-images", "--json"]
    if forced:
        args += ["--format", "lumos"]
    with pytest.raises(SystemExit) as result:
        run_integrity(args)
    assert result.value.code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["format"] == "lumos"
    assert not any(item["reason"] == "requires: images" for item in output["not_evaluated"])
