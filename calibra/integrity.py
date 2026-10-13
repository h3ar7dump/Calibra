"""
calibra integrity — "Can I trust this dataset?"

Runs a fixed, cheap set of diagnostics that answer the first question
practitioners ask about a new dataset before anything about quality,
diversity, or optimization: are timestamps consistent, are episodes
complete, is the camera stream actually updating, are the recorded
motions themselves jittery/jerky. Groups findings into Critical /
Warnings / Passed rather than leading with a single score — the score
is still computed, but demoted to a summary line, not the headline
result.

Usage:
    calibra integrity /data/my_demos.h5
    calibra integrity /data/my_demos.h5 --format hdf5
    calibra integrity /data/my_demos.h5 --json
    calibra integrity lerobot/pusht --decode-images   # v1 LeRobot only, see below

Exit codes:
    0  No block-level CRITICAL findings (or --strict not passed).
    1  One or more block-level CRITICAL findings, or (with --strict) any
       CRITICAL finding at all, including context-dependent motion-review
       ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from calibra.analyzers.base import Analyzer
from calibra.analyzers.blur import BlurAnalyzer
from calibra.analyzers.calibration_drift import CalibrationDriftAnalyzer
from calibra.analyzers.camera_freeze import CameraFreezeAnalyzer
from calibra.analyzers.camera_view import CameraViewMismatchAnalyzer
from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.task_structure import TaskStructureAnalyzer
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.analyzers.video_framerate import VideoFramerateAnalyzer
from calibra.pipeline import Pipeline
from calibra.progress import ProgressBar
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import DiagnosticReport, RiskFlag, RiskLevel

# Metric-name whitelist, not analyzer selection — TaskStructureAnalyzer also
# emits trajectory_diversity/contact_density/grasp_events_per_episode, and
# ControlSmoothnessAnalyzer also emits action_state_divergence/
# motion_collection_signature, which belong to the Quality layer, not
# Integrity. Only the three raw jerkiness metrics below ("are the recorded
# motions physically jittery") count as an Integrity-layer trust question —
# tracking error and collection-method signature are Quality-layer questions.
_INTEGRITY_METRICS = frozenset(
    {
        "timestamp_jitter_cv",
        "timestamp_dropout_rate",
        "action_obs_misalignment",
        "camera_physics_drift",
        "action_dropout_rate",
        "short_episode_fraction",
        "duplicate_frame_rate",
        "camera_freeze_events",
        "blurry_episode_fraction",
        "video_fps_mismatch",
        "video_vfr_jitter_cv",
        "camera_view_name_mismatch",
        "ldlj",
        "jerk_spike_rate",
        "velocity_discontinuity_rate",
        "joint_offset_max_abs",
    }
)

# Metrics that are context-dependent review signals rather than objective
# acquisition/format/sync failures. A CRITICAL here means "statistically
# unusual, worth a human look" (e.g. a scripted planner's jerk-spike rate
# is *supposed* to be high) — not "the recording is broken." These never
# fail CI by default; --strict opts back into the old "any CRITICAL fails"
# behavior. See _suggested_action(). joint_offset_max_abs is included even
# though CalibrationDriftAnalyzer never emits CRITICAL today (see that
# module's docstring) — future-proofs this list if thresholds are validated
# and the cap is lifted later.
_MOTION_REVIEW_METRICS = frozenset(
    {
        "ldlj",
        "jerk_spike_rate",
        "velocity_discontinuity_rate",
        "joint_offset_max_abs",
    }
)

_ICONS = {
    RiskLevel.CRITICAL: "❌",
    RiskLevel.WARNING: "⚠️ ",
    RiskLevel.OK: "✅",
}

# Credit given per level toward the demoted Integrity Score.
_LEVEL_CREDIT = {RiskLevel.OK: 1.0, RiskLevel.WARNING: 0.5, RiskLevel.CRITICAL: 0.0}


def _is_integrity_flag(f: RiskFlag) -> bool:
    return f.metric in _INTEGRITY_METRICS or f.metric.startswith("camera_lag_std[")


def _integrity_flags(report: DiagnosticReport) -> list[RiskFlag]:
    return [f for f in report.flags if _is_integrity_flag(f)]


def _integrity_score(flags: list[RiskFlag]) -> tuple[int, str]:
    scored = [f for f in flags if f.level in _LEVEL_CREDIT]
    if not scored:
        return 100, "Healthy"
    score = round(100 * sum(_LEVEL_CREDIT[f.level] for f in scored) / len(scored))
    status = "Healthy" if score >= 90 else "Warning" if score >= 70 else "Critical"
    return score, status


def _suggested_action(f: RiskFlag, policy: Optional[dict[str, str]] = None) -> str:
    """
    informational (OK) / inspect (WARNING, or CRITICAL motion-review) / block
    (CRITICAL, objective acquisition/format/sync/completeness failure).

    `policy` (see calibra/policy.py) overrides the block/inspect split for
    CRITICAL findings on the metrics it names; OK and WARNING are never
    affected by policy — a WARNING never fails CI regardless of configuration.
    """
    if f.level == RiskLevel.OK:
        return "informational"
    if f.level == RiskLevel.WARNING:
        return "inspect"
    if policy and f.metric in policy:
        return policy[f.metric]
    return "inspect" if f.metric in _MOTION_REVIEW_METRICS else "block"


def _ci_result(
    flags: list[RiskFlag], strict: bool, policy: Optional[dict[str, str]] = None
) -> tuple[str, Optional[str]]:
    """Returns (ci_result, ci_reason). ci_reason is None when Passed."""
    critical = [f for f in flags if f.level == RiskLevel.CRITICAL]
    blocking = [f for f in critical if _suggested_action(f, policy) == "block"]
    if blocking:
        names = ", ".join(f.metric for f in blocking)
        return "Failed", f"{len(blocking)} CRITICAL block-level finding(s): {names}"
    if strict and critical:
        names = ", ".join(f.metric for f in critical)
        return "Failed", f"{len(critical)} CRITICAL finding(s) (--strict): {names}"
    return "Passed", None


def _not_evaluated(
    report: DiagnosticReport, batch: EpisodeBatch, analyzers: list[Analyzer]
) -> list[dict]:
    by_name = {a.name: a for a in analyzers}
    result = []
    for name in report.skipped_analyzers:
        analyzer = by_name.get(name)
        missing = sorted(analyzer.requires - batch.capabilities) if analyzer else []
        reason = f"requires: {', '.join(missing)}" if missing else "requirements not met"
        result.append({"check": name, "reason": reason})
    return result


def run_integrity(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra integrity",
        description='Check "can I trust this dataset?" before anything else.',
    )
    p.add_argument("path", help="Path or Hub ID of the dataset to check")
    p.add_argument(
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap", "lumos"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Print the report as JSON instead of human-readable text",
    )
    p.add_argument(
        "--decode-images",
        action="store_true",
        help=(
            "Decode camera frames for LeRobot or Lumos "
            "datasets so duplicate_frame_rate/camera_freeze_events/blurry_episode_fraction "
            "can run. Increases load time and memory use."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail CI (exit 1) on ANY CRITICAL finding, including context-dependent "
            "motion-review ones (ldlj, jerk_spike_rate, velocity_discontinuity_rate). "
            "Default only fails on objective acquisition/format/sync/completeness "
            "failures; see 'suggested_action' on each finding. Mutually exclusive "
            "with --policy."
        ),
    )
    p.add_argument(
        "--policy",
        metavar="FILE",
        help=(
            "JSON file mapping metric name -> 'block'|'inspect', overriding the "
            "default block/inspect split for CRITICAL findings on the metrics it "
            "names (see docs/integrity.md 'CI Policy Files'). Mutually exclusive "
            "with --strict."
        ),
    )
    from calibra.dataset_profiles import add_profile_argument

    add_profile_argument(p)
    args = p.parse_args(argv)

    if args.strict and args.policy:
        print("error: --strict and --policy are mutually exclusive", file=sys.stderr)
        sys.exit(2)

    policy: Optional[dict[str, str]] = None
    if args.policy:
        from calibra.policy import load_policy

        try:
            policy = load_policy(args.policy)
        except Exception as exc:
            print(f"error loading policy file: {exc}", file=sys.stderr)
            sys.exit(2)
        unknown = sorted(set(policy) - _INTEGRITY_METRICS)
        if unknown:
            print(
                f"warning: policy file references unknown metric(s), ignored: {', '.join(unknown)}",
                file=sys.stderr,
            )

    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://") :]

    reader = None
    decode_pbar = None
    if args.decode_images:
        from calibra.__main__ import _get_reader
        from calibra.ingestion.adapters.lerobot import LeRobotReader
        from calibra.ingestion.adapters.lumos import LumosReader
        from calibra.ingestion.registry import detect_reader

        decode_pbar = ProgressBar(description="Decoding frames", stream=sys.stderr)

        def decode_callback(current: int, total: int, message: str) -> None:
            if decode_pbar is not None:
                decode_pbar.update(current, total=total, message=message)

        try:
            reader_type = (
                type(_get_reader(args.format)) if args.format else detect_reader(dataset_path)
            )
            if reader_type is LumosReader:
                reader = LumosReader(decode_images=True)
            elif reader_type is LeRobotReader:
                reader = LeRobotReader(decode_images=True, progress_callback=decode_callback)
            else:
                raise ValueError("--decode-images supports LeRobot and Lumos datasets")
        except Exception as exc:
            decode_pbar.close()
            print(f"error loading dataset: {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.format:
        from calibra.__main__ import _get_reader

        reader = _get_reader(args.format)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading {dataset_path!r} ...")
    try:
        from calibra.ingestion.registry import load

        batch = load(dataset_path, reader=reader)
    except Exception as exc:
        print(f"error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if decode_pbar is not None:
            decode_pbar.close()

    log(f"  {batch.n_episodes} episodes  ·  {batch.n_samples} steps")
    log("Running integrity checks ...")
    try:
        analyzers: list[Analyzer] = [
            TemporalAnalyzer(),
            TaskStructureAnalyzer(),
            DuplicateFrameAnalyzer(),
            CameraFreezeAnalyzer(),
            BlurAnalyzer(),
            VideoFramerateAnalyzer(),
            CameraViewMismatchAnalyzer(),
            ControlSmoothnessAnalyzer(),
            CalibrationDriftAnalyzer(),
        ]
        check_pbar = ProgressBar(
            total=len(analyzers), description="Integrity checks", stream=sys.stderr
        )

        def check_callback(current: int, total: int, message: str) -> None:
            check_pbar.update(current, total=total, message=message)

        try:
            report = Pipeline(analyzers=analyzers, profile=args.profile).run(
                batch, progress_callback=check_callback
            )
        finally:
            check_pbar.close()
    except Exception as exc:
        print(f"error running pipeline: {exc}", file=sys.stderr)
        sys.exit(1)

    flags = _integrity_flags(report)
    critical = [f for f in flags if f.level == RiskLevel.CRITICAL]
    warnings = [f for f in flags if f.level == RiskLevel.WARNING]
    passed = [f for f in flags if f.level == RiskLevel.OK]
    score, status = _integrity_score(flags)
    ci_result, ci_reason = _ci_result(flags, args.strict, policy)
    not_evaluated = _not_evaluated(report, batch, analyzers)

    if args.json:
        result = {
            "dataset_name": report.dataset_name,
            "source_path": report.source_path,
            "format": report.format,
            "n_episodes": report.n_episodes,
            "critical": [_flag_to_dict(f, policy) for f in critical],
            "warnings": [_flag_to_dict(f, policy) for f in warnings],
            "passed": [_flag_to_dict(f, policy) for f in passed],
            "not_evaluated": not_evaluated,
            "integrity_score": score,
            "status": status,
            "ci_result": ci_result,
            "ci_reason": ci_reason,
            "policy_path": args.policy,
        }
        if report.ingestion_warnings:
            result["ingestion_warnings"] = report.ingestion_warnings
        print(json.dumps(result, indent=2))
    else:
        print(
            render(
                report,
                critical,
                warnings,
                passed,
                not_evaluated,
                score,
                status,
                ci_result,
                ci_reason,
                policy,
            )
        )

    sys.exit(1 if ci_result == "Failed" else 0)


def _flag_to_dict(f: RiskFlag, policy: Optional[dict[str, str]] = None) -> dict:
    return {
        "metric": f.metric,
        "interpretation": f.interpretation,
        "implication": f.implication,
        "observed_value": f.observed.value,
        "suggested_action": _suggested_action(f, policy),
    }


def render(
    report: DiagnosticReport,
    critical: list[RiskFlag],
    warnings: list[RiskFlag],
    passed: list[RiskFlag],
    not_evaluated: list[dict],
    score: int,
    status: str,
    ci_result: str,
    ci_reason: Optional[str],
    policy: Optional[dict[str, str]] = None,
) -> str:
    lines = [
        "─── Dataset Integrity " + "─" * 36,
        f"{report.dataset_name} · {report.n_episodes} episodes",
        "",
    ]

    for label, group in (("Critical", critical), ("Warnings", warnings), ("Passed", passed)):
        lines.append(f"{label} ({len(group)})")
        for f in group:
            icon = _ICONS[f.level]
            tag = (
                f"  [{_suggested_action(f, policy).upper()}]"
                if f.level == RiskLevel.CRITICAL
                else ""
            )
            lines.append(f"  {icon} {f.metric}: {f.interpretation}{tag}")
            if f.level != RiskLevel.OK:
                lines.append(f"      {f.implication}")
        lines.append("")

    if report.ingestion_warnings:
        lines.append(f"Ingestion Warnings ({len(report.ingestion_warnings)})")
        for warning in report.ingestion_warnings:
            lines.append(f"  ⚠️ {warning['metric']}: {warning['interpretation']}")
            lines.append(f"      {warning['implication']}")
        lines.append("")

    if not_evaluated:
        lines.append(f"Not Evaluated ({len(not_evaluated)})")
        for item in not_evaluated:
            lines.append(f"  ⬚ {item['check']}: {item['reason']}")
        lines.append("")

    lines.append(f"Integrity Score: {score}/100  ·  Status: {status}")
    lines.append(f"CI result: {ci_result}" + (f"  ·  Reason: {ci_reason}" if ci_reason else ""))
    lines.append("─" * 58)
    return "\n".join(lines)
