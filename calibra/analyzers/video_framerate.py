"""
Video Framerate Analyzer.

Verifies that video stream encoding framerates (FPS) match the dataset's
nominal control/recording frequency, and checks for variable framerate (VFR)
timing jitter in video streams.

A mismatch between physical video FPS and nominal control loop FPS (e.g. 15fps
video paired with 30Hz actions) corrupts the temporal scale of visual
observations, causing imitation policies to misestimate object and robot
velocities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import AnalyzerResult, ObservedValue, RiskFlag, RiskLevel

_MAX_FPS_MISMATCH_FRACTION = 0.05  # >5% discrepancy between video and nominal FPS is CRITICAL
_MAX_VFR_JITTER_CV = 0.05  # >5% variation in frame presentation intervals is WARNING


def probe_video_fps(video_path: str | Path) -> Optional[float]:
    """
    Extract the container/stream framerate from a video file without decoding pixel buffers.

    Returns FPS as float, or None if the video file cannot be probed.
    """
    p = str(video_path)
    # 1. Try PyAV (fastest, reads stream metadata directly)
    try:
        import av

        with av.open(p) as container:
            if container.streams.video:
                stream = container.streams.video[0]
                avg_rate = getattr(stream, "average_rate", None)
                if avg_rate is not None and float(avg_rate) > 0:
                    return float(avg_rate)
                r_rate = getattr(stream, "r_frame_rate", None) or getattr(stream, "framerate", None)
                if r_rate is not None and float(r_rate) > 0:
                    return float(r_rate)
                if stream.duration and stream.frames and float(stream.duration) > 0:
                    tb = float(stream.time_base) if stream.time_base else 1.0
                    duration_s = float(stream.duration) * tb
                    if duration_s > 0:
                        return float(stream.frames) / duration_s
    except Exception:
        pass

    # 2. Try OpenCV fallback
    try:
        import cv2

        cap = cv2.VideoCapture(p)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if fps > 0:
            return fps
    except Exception:
        pass

    return None


def probe_video_pts_jitter(video_path: str | Path) -> Optional[float]:
    """
    Extract packet/frame presentation timestamps (PTS) via demuxing to compute
    frame-interval CV (coefficient of variation). Returns CV or None.
    """
    try:
        import av

        pts_list: list[float] = []
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            tb = float(stream.time_base) if stream.time_base else 1.0
            # Read first up to 100 packets to quickly measure timing jitter without decoding
            count = 0
            for packet in container.demux(stream):
                if packet.pts is not None:
                    pts_list.append(float(packet.pts) * tb)
                    count += 1
                    if count >= 100:
                        break
        if len(pts_list) >= 4:
            pts_arr = np.sort(np.asarray(pts_list))
            dts = np.diff(pts_arr)
            pos_dts = dts[dts > 1e-5]
            if len(pos_dts) >= 3 and np.mean(pos_dts) > 0:
                return float(np.std(pos_dts) / np.mean(pos_dts))
    except Exception:
        pass

    return None


def _find_video_paths(batch: EpisodeBatch) -> list[Path]:
    """Discover referenced video files from batch metadata or standard directory layouts."""
    paths: list[Path] = []
    seen: set[str] = set()

    # 1. From batch.extra
    batch_paths = batch.extra.get("video_paths", [])
    if isinstance(batch_paths, (list, tuple)):
        for bp in batch_paths:
            if bp and str(bp) not in seen:
                p = Path(bp)
                if p.is_file():
                    paths.append(p)
                    seen.add(str(bp))
        # An explicit list is authoritative, including an empty list when all
        # video streams are disabled. Do not rediscover excluded media on disk.
        if "video_paths" in batch.extra:
            return paths

    # 2. From episode / metadata extra if present
    episodes = list(batch.episodes)
    for ep in episodes:
        ep_extra = getattr(ep, "extra", {}) or getattr(ep.metadata, "extra", {})
        if isinstance(ep_extra, dict):
            ep_vpaths = ep_extra.get("video_paths", {})
            if isinstance(ep_vpaths, dict):
                for vp in ep_vpaths.values():
                    if vp and str(vp) not in seen:
                        p = Path(vp)
                        if p.is_file():
                            paths.append(p)
                            seen.add(str(vp))
            elif isinstance(ep_vpaths, (list, tuple)):
                for vp in ep_vpaths:
                    if vp and str(vp) not in seen:
                        p = Path(vp)
                        if p.is_file():
                            paths.append(p)
                            seen.add(str(vp))

    # 3. From batch.source_path / "videos" filesystem
    if not paths and batch.source_path:
        root = Path(batch.source_path)
        video_dir = root / "videos" if (root / "videos").exists() else root
        if video_dir.exists():
            for ext in (".mp4", ".mkv", ".avi", ".webm"):
                for vp in sorted(video_dir.glob(f"**/*{ext}")):
                    if vp.is_file() and str(vp) not in seen:
                        paths.append(vp)
                        seen.add(str(vp))
                        if len(paths) >= 20:  # Sample up to 20 representative video files
                            break

    return paths


def _infer_nominal_fps(batch: EpisodeBatch) -> Optional[float]:
    """Determine nominal control frequency from batch metadata or step timestamps."""
    # 1. Metadata
    if "fps" in batch.extra and batch.extra["fps"] is not None:
        try:
            return float(batch.extra["fps"])
        except (ValueError, TypeError):
            pass

    # 2. Timestamp delta median
    all_dts: list[float] = []
    episodes = list(batch.episodes)
    for ep in episodes:
        if ep.timestamps is not None and len(ep.timestamps) >= 2:
            diffs = np.diff(ep.timestamps)
            pos = diffs[diffs > 1e-5]
            if len(pos) > 0:
                all_dts.append(float(np.median(pos)))
    if all_dts:
        med_dt = float(np.median(all_dts))
        if med_dt > 0:
            return float(round(1.0 / med_dt, 3))

    return None


@dataclass
class VideoFramerateAnalyzer(Analyzer):
    """
    Checks that video streams have consistent, non-jittery framerates matching
    the dataset's nominal recording frequency.

    Parameters
    ----------
    max_mismatch_fraction : relative deviation (|video_fps - nominal_fps| / nominal_fps)
                            above which a CRITICAL block-level flag is emitted.
    max_vfr_jitter_cv     : frame interval CV above which a WARNING is emitted.
    """

    requires = frozenset()  # Can run without full pixel decode if video files exist

    max_mismatch_fraction: float = _MAX_FPS_MISMATCH_FRACTION
    max_vfr_jitter_cv: float = _MAX_VFR_JITTER_CV

    @property
    def name(self) -> str:
        return "video_framerate"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        video_files = _find_video_paths(batch)
        if not video_files:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="video_fps_mismatch",
                        observed=ObservedValue(value=None, unit="fps"),
                        interpretation="No video files found to probe for framerate.",
                        implication="Video framerate checks were skipped.",
                    )
                ],
            )

        nominal_fps = _infer_nominal_fps(batch)
        if nominal_fps is None or nominal_fps <= 0:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="video_fps_mismatch",
                        observed=ObservedValue(value=None, unit="fps"),
                        interpretation="Could not infer nominal dataset FPS from timestamps or metadata.",
                        implication="Framerate mismatch check was skipped.",
                    )
                ],
            )

        # Probe video files
        measured_fps_list: list[float] = []
        jitter_cv_list: list[float] = []

        for vf in video_files:
            fps = probe_video_fps(vf)
            if fps is not None and fps > 0:
                measured_fps_list.append(fps)
            jitter_cv = probe_video_pts_jitter(vf)
            if jitter_cv is not None:
                jitter_cv_list.append(jitter_cv)

        if not measured_fps_list:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="video_fps_mismatch",
                        observed=ObservedValue(value=None, unit="fps"),
                        interpretation="Video decoding library (PyAV or OpenCV) not available to probe video header.",
                        implication="Install PyAV with `pip install av` to enable video framerate checks.",
                    )
                ],
            )

        med_measured_fps = float(np.median(measured_fps_list))
        rel_diff = abs(med_measured_fps - nominal_fps) / nominal_fps

        flags: list[RiskFlag] = []

        # 1. FPS Mismatch Flag
        if rel_diff > self.max_mismatch_fraction:
            flags.append(
                RiskFlag(
                    level=RiskLevel.CRITICAL,
                    metric="video_fps_mismatch",
                    observed=ObservedValue(
                        value=round(rel_diff * 100, 2),
                        unit="%",
                    ),
                    interpretation=(
                        f"Video file FPS ({med_measured_fps:.1f} fps) deviates by "
                        f"{rel_diff * 100:.1f}% from dataset nominal FPS ({nominal_fps:.1f} fps)."
                    ),
                    implication=(
                        "Video time scale does not match action and state recording frequency. "
                        "Training vision-based policies on time-dilated or compressed video "
                        "causes severe dynamics estimation errors and erratic robot rollouts."
                    ),
                )
            )
        elif rel_diff > 0.01:
            flags.append(
                RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="video_fps_mismatch",
                    observed=ObservedValue(
                        value=round(rel_diff * 100, 2),
                        unit="%",
                    ),
                    interpretation=(
                        f"Minor framerate difference: video is {med_measured_fps:.2f} fps, "
                        f"nominal is {nominal_fps:.2f} fps ({rel_diff * 100:.1f}% difference)."
                    ),
                    implication="Likely a minor container encoding precision difference (e.g. 29.97 vs 30fps).",
                )
            )
        else:
            flags.append(
                RiskFlag(
                    level=RiskLevel.OK,
                    metric="video_fps_mismatch",
                    observed=ObservedValue(
                        value=round(rel_diff * 100, 2),
                        unit="%",
                    ),
                    interpretation=(
                        f"Video framerate ({med_measured_fps:.1f} fps) matches nominal "
                        f"recording rate ({nominal_fps:.1f} fps)."
                    ),
                    implication="Visual stream temporal scale is aligned.",
                )
            )

        # 2. VFR / Timing Jitter Flag
        if jitter_cv_list:
            med_jitter = float(np.median(jitter_cv_list))
            if med_jitter > self.max_vfr_jitter_cv:
                flags.append(
                    RiskFlag(
                        level=RiskLevel.WARNING,
                        metric="video_vfr_jitter_cv",
                        observed=ObservedValue(
                            value=round(med_jitter, 3),
                            unit="cv",
                        ),
                        interpretation=(
                            f"Video streams show variable framerate (VFR) jitter (CV = {med_jitter:.3f})."
                        ),
                        implication=(
                            "Inconsistent inter-frame intervals can cause frame drops and sync jitter "
                            "during model training."
                        ),
                    )
                )
            else:
                flags.append(
                    RiskFlag(
                        level=RiskLevel.OK,
                        metric="video_vfr_jitter_cv",
                        observed=ObservedValue(
                            value=round(med_jitter, 3),
                            unit="cv",
                        ),
                        interpretation="Video stream inter-frame timing (PTS) is stable and constant (CFR).",
                        implication="No variable-framerate timing drift detected.",
                    )
                )

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            raw_metrics={
                "video_fps_measured": med_measured_fps,
                "video_fps_nominal": nominal_fps,
                "video_fps_mismatch_pct": round(rel_diff * 100, 2),
            },
        )
