"""Read Lumos EGO recordings: metadata, segmented stereo video, IMU and audio.

One metadata file describes one episode, regardless of the number of partXXXX
directories. Device timestamps are retained on independent sensor clocks,
relative to a single episode origin. Video and audio decoding are opt-in.
Camera/IMU calibration files are preserved as references without applying
unverified units, coordinate transforms or time-offset conventions.
"""

from __future__ import annotations

import csv
import json
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

_STREAM_KEYS = {
    "color_left": "camera_left",
    "color_right": "camera_right",
    "imu": "imu",
    "audio": "audio",
}


def _metadata_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.name.startswith("meta_") and path.suffix == ".json" else []
    if not path.is_dir():
        return []
    direct = sorted(path.glob("meta_*.json"))
    return direct or sorted(path.glob("*/meta_*.json"))


def _read_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as source:
        data = json.load(source)
    if not isinstance(data, dict) or not isinstance(data.get("streams"), dict):
        raise ValueError(f"Not a Lumos metadata file: '{path}'")
    if not _STREAM_KEYS.keys() & data["streams"].keys():
        raise ValueError(f"No supported Lumos streams in '{path}'")
    if data.get("timebase", {}).get("unit", "microsecond") != "microsecond":
        raise ValueError(f"Lumos timestamps must use microseconds: '{path}'")
    return data


def _referenced_file(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Lumos file reference escapes episode directory: '{value}'")
    if not path.is_file():
        raise FileNotFoundError(f"Missing Lumos file: '{path}'")
    return path


def _csv_rows(path: Path, columns: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not columns <= set(reader.fieldnames or []):
            raise ValueError(f"Lumos CSV '{path}' requires columns {sorted(columns)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty Lumos CSV: '{path}'")
    return rows


def _timestamps(rows: list[dict[str, str]], path: Path) -> NDArray[np.int64]:
    try:
        result = np.array([int(row["timestamp_us"]) for row in rows], dtype=np.int64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid timestamp_us in '{path}'") from exc
    if np.any(np.diff(result) <= 0):
        raise ValueError(f"Non-increasing Lumos timestamps in '{path}'")
    return result


def _segments(stream: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    # Legacy single-file entries have the same shape as a segment, minus index.
    segments = stream.get("segments")
    if segments is None:
        segments = [dict(stream, index=1)] if stream.get("file") else []
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"Enabled Lumos stream has no segments in '{root}'")
    try:
        ordered = sorted(segments, key=lambda segment: int(segment["index"]))
        indices = [int(segment["index"]) for segment in ordered]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Lumos segment index in '{root}'") from exc
    if indices[0] < 1 or len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate or non-positive Lumos segment index in '{root}'")
    for segment, index in zip(ordered, indices):
        for field in ("file", "pts_file"):
            if field not in segment:
                continue
            path = _referenced_file(root, segment[field])
            for component in (path.parent.name, path.stem):
                match = re.search(r"(?:^|_)part(\d+)(?:_|$)", component)
                if match and int(match[1]) != index:
                    raise ValueError(f"Lumos segment index {index} disagrees with '{path}'")
        if "file" not in segment:
            raise ValueError(f"Missing file in Lumos segment {index} in '{root}'")
    return ordered


def _check_count(segment: dict[str, Any], actual: int, path: Path) -> None:
    if "frame_count" in segment and int(segment["frame_count"]) != actual:
        raise ValueError(
            f"Lumos frame_count mismatch in '{path}': "
            f"metadata={segment['frame_count']}, actual={actual}"
        )


@register
class LumosReader(DatasetReader):
    """Read an episode directory, meta_*.json, or a directory of episodes.

    ``observations`` contains native-rate imu_accel / imu_gyro arrays. With
    decode_images=True, camera_left / camera_right contain RGB uint8 frames
    resized to at most max_image_size pixels per side (None keeps full size).
    decode_audio=True adds normalized PCM samples and a derived sample clock;
    audio_pts always retains the original block clock in obs_timestamps.
    Actions are empty because these recordings contain no robot commands.
    """

    def __init__(
        self,
        decode_images: bool = False,
        decode_audio: bool = False,
        max_image_size: int | None = 256,
    ) -> None:
        if max_image_size is not None and max_image_size < 1:
            raise ValueError("max_image_size must be positive or None")
        self.decode_images = decode_images
        self.decode_audio = decode_audio
        self.max_image_size = max_image_size

    @property
    def format_name(self) -> str:
        return "lumos"

    @classmethod
    def can_read(cls, path: str) -> bool:
        for file in _metadata_files(Path(path)):
            try:
                _read_metadata(file)
                return True
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        return False

    def read(self, path: str) -> EpisodeBatch:
        root = Path(path).resolve()
        files = _metadata_files(root)
        if not files:
            raise ValueError(f"No Lumos meta_*.json files found at '{path}'")
        if len({file.parent for file in files}) != len(files):
            raise ValueError(f"Multiple Lumos metadata files for one episode at '{path}'")
        episodes = [self._read_episode(file) for file in files]
        fps = {ep.metadata.extra["fps"] for ep in episodes if ep.metadata.extra.get("fps")}
        return EpisodeBatch(
            episodes=episodes,
            dataset_name=root.parent.name if root.is_file() else root.name,
            format=self.format_name,
            source_path=str(root),
            extra={
                "video_paths": [p for ep in episodes for p in ep.metadata.extra["video_paths"]],
                **({"fps": fps.pop()} if len(fps) == 1 else {}),
            },
        )

    def _read_episode(self, file: Path) -> Episode:
        metadata = _read_metadata(file)
        root = file.parent
        observations: dict[str, NDArray[Any]] = {}
        clocks: dict[str, NDArray[np.int64]] = {}
        video_paths: dict[str, list[str]] = {}
        audio_segments: list[dict[str, Any]] = []
        stream_segments: dict[str, list[dict[str, Any]]] = {}
        audio_segments_clock: list[tuple[int, int, int]] = []
        fps: float | None = None
        for name, key in _STREAM_KEYS.items():
            stream = metadata["streams"].get(name)
            if not stream or not stream.get("enabled", True):
                continue
            segments = _segments(stream, root)
            stream_segments[name] = segments
            if name == "imu":
                self._read_imu(stream, segments, root, observations, clocks)
            else:
                timestamps = []
                images = []
                samples = []
                for segment in segments:
                    media = _referenced_file(root, segment["file"])
                    if not segment.get("pts_file"):
                        raise ValueError(f"Missing pts_file for '{media}'")
                    pts = _referenced_file(root, segment["pts_file"])
                    ts = _timestamps(_csv_rows(pts, {"timestamp_us"}), pts)
                    _check_count(segment, len(ts), pts)
                    timestamps.append(ts)
                    if name == "audio":
                        info, audio = self._read_audio(media, stream)
                        audio_segments.append(dict(info, file=str(media), index=segment["index"]))
                        audio_segments_clock.append(
                            (int(ts[0]), info["n_samples"], info["sample_rate_hz"])
                        )
                        if audio is not None:
                            samples.append(audio)
                    else:
                        video_paths.setdefault(key, []).append(str(media))
                        if self.decode_images:
                            images.append(self._read_video(media, len(ts)))
                clock_key = "audio_pts" if name == "audio" else key
                clocks[clock_key] = self._join_clock(timestamps, file, clock_key)
                if images:
                    observations[key] = np.concatenate(images)
                if samples:
                    observations["audio"] = np.concatenate(samples)
                if name == "color_left" or (name == "color_right" and fps is None):
                    fps = float(stream["fps"]) if stream.get("fps") else None
        if not clocks:
            raise ValueError(f"No enabled sensor data in '{file}'")
        # WAV headers reveal sample spans even when decoding is disabled. A
        # following block PTS can be increasing while still overlapping the
        # previous file's tail. Compare integer deltas to avoid epoch rounding.
        for previous, current in zip(audio_segments_clock, audio_segments_clock[1:]):
            start, count, rate = previous
            if (current[0] - start) * rate <= (count - 1) * 1_000_000:
                raise ValueError(f"Overlapping WAV sample clocks in '{file}'")
        origin = min(int(ts[0]) for ts in clocks.values())
        obs_timestamps = {
            key: (ts - origin).astype(np.float64) * 1e-6 for key, ts in clocks.items()
        }
        if "audio" in observations:
            obs_timestamps["audio"] = np.concatenate(
                [
                    (start - origin) * 1e-6 + np.arange(count, dtype=np.float64) / rate
                    for start, count, rate in audio_segments_clock
                ]
            )
        master_key = next(
            key
            for key in ("camera_left", "camera_right", "imu_accel", "imu_gyro", "audio_pts")
            if key in clocks
        )
        calibration_paths = {
            key: str(_referenced_file(root, value))
            for key, value in metadata.get("files", {}).items()
            if key.startswith("calibration_")
        }
        return Episode(
            metadata=EpisodeMetadata(
                episode_id=file.stem.removeprefix("meta_"),
                source_file=str(file),
                extra={
                    "lumos": metadata,
                    "timestamp_origin_us": origin,
                    "master_clock": master_key,
                    "segments": stream_segments,
                    "video_paths": [p for paths in video_paths.values() for p in paths],
                    "video_paths_by_camera": video_paths,
                    "audio_segments": audio_segments,
                    "calibration_paths": calibration_paths,
                    "fps": fps,
                },
            ),
            timestamps=obs_timestamps[master_key],
            observations=observations,
            obs_timestamps=obs_timestamps,
            actions=np.empty((0, 0), dtype=np.float32),
        )

    @staticmethod
    def _join_clock(parts: list[NDArray[np.int64]], file: Path, key: str) -> NDArray[np.int64]:
        joined = np.concatenate(parts)
        if np.any(np.diff(joined) <= 0):
            raise ValueError(f"Overlapping or non-increasing {key} parts in '{file}'")
        return joined

    def _read_imu(
        self,
        stream: dict[str, Any],
        segments: list[dict[str, Any]],
        root: Path,
        observations: dict[str, NDArray[Any]],
        clocks: dict[str, NDArray[np.int64]],
    ) -> None:
        values: dict[str, list[NDArray[np.float32]]] = {"accel": [], "gyro": []}
        times: dict[str, list[NDArray[np.int64]]] = {"accel": [], "gyro": []}
        for segment in segments:
            path = _referenced_file(root, segment["file"])
            rows = _csv_rows(path, {"timestamp_us", "x", "y", "z", "type"})
            if any(row["type"] not in values for row in rows):
                raise ValueError(f"Unknown IMU type in '{path}'")
            counts = []
            for kind in values:
                if not stream.get(f"{kind}_enabled", True):
                    continue
                sensor_rows = [row for row in rows if row["type"] == kind]
                if not sensor_rows:
                    raise ValueError(f"Missing {kind} samples in '{path}'")
                ts = _timestamps(sensor_rows, path)
                try:
                    xyz = np.array(
                        [[float(row[c]) for c in ("x", "y", "z")] for row in sensor_rows],
                        dtype=np.float32,
                    )
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid IMU values in '{path}'") from exc
                if not np.isfinite(xyz).all():
                    raise ValueError(f"Non-finite IMU values in '{path}'")
                values[kind].append(xyz)
                times[kind].append(ts)
                counts.append(len(ts))
            # Observed firmware counts sample ticks, not interleaved CSV rows.
            _check_count(segment, max(counts, default=0), path)
        for kind in values:
            if values[kind]:
                key = f"imu_{kind}"
                observations[key] = np.concatenate(values[kind])
                clocks[key] = self._join_clock(times[kind], root, key)

    def _read_audio(
        self, path: Path, stream: dict[str, Any]
    ) -> tuple[dict[str, int], NDArray[np.float32] | None]:
        with wave.open(str(path), "rb") as audio:
            info = {
                "sample_rate_hz": audio.getframerate(),
                "channels": audio.getnchannels(),
                "bits_per_sample": audio.getsampwidth() * 8,
                "n_samples": audio.getnframes(),
            }
            for field in ("sample_rate_hz", "channels", "bits_per_sample"):
                if field in stream and int(stream[field]) != info[field]:
                    raise ValueError(f"WAV {field} disagrees with Lumos metadata in '{path}'")
            if not self.decode_audio:
                return info, None
            if info["bits_per_sample"] != 16:
                raise ValueError(f"Lumos audio decoding requires 16-bit PCM: '{path}'")
            raw = audio.readframes(info["n_samples"])
            if len(raw) != info["n_samples"] * info["channels"] * 2:
                raise ValueError(f"Truncated Lumos WAV: '{path}'")
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            return info, samples.reshape(-1, info["channels"])

    def _read_video(self, path: Path, expected: int) -> NDArray[np.uint8]:
        try:
            import av
        except ImportError:
            raise ImportError(
                "PyAV is required to decode Lumos videos. "
                "Install it with: pip install 'calibra-robotics[lumos]'"
            ) from None
        frames = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if self.max_image_size is not None:
                    scale = min(1.0, self.max_image_size / max(frame.width, frame.height))
                    frame = frame.reformat(
                        width=max(1, round(frame.width * scale)),
                        height=max(1, round(frame.height * scale)),
                    )
                frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) != expected:
            raise ValueError(
                f"Lumos video/PTS frame mismatch in '{path}': "
                f"decoded={len(frames)}, timestamps={expected}"
            )
        return np.stack(frames)
