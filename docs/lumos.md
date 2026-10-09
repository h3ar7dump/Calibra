# Lumos EGO recordings

`LumosReader` reads a single episode directory, its `meta_*.json` file, or a
dataset directory whose immediate children are episodes. Each metadata file
produces one `Episode`, including all its `partXXXX` segments.

```bash
calibra integrity datasets/lumos --format lumos --json
calibra integrity datasets/lumos/EGO_AZEC86400G8_20260928_153613
pip install 'calibra-robotics[lumos]'
calibra integrity datasets/lumos --decode-images
calibra datasets/lumos --format lumos --json
```

## Segments and file references

An episode has root-level metadata and camera/IMU calibration YAML files.
Each `part0001`, `part0002`, etc. contains the streams recorded during that
segment: stereo MP4 files with companion PTS CSV files, an interleaved IMU
CSV, and a WAV file with its own PTS CSV.

The reader uses `streams.<name>.segments` as the authoritative file list,
sorts its integer `index` values, and verifies referenced files and part
numbers. Single-file `streams.<name>.file` entries are also accepted when
`segments` is absent. Disabled streams are skipped. The reader does not infer
data files from directory contents or invent timestamps for missing parts.

Metadata advertises a nominal 60-second segment duration. Actual timestamps
determine continuity: gaps are retained; repeated timestamps, backwards time
and overlapping timestamp ranges raise descriptive errors. Local samples
contain only `part0001`; multi-part behavior is tested with synthetic recordings.

PTS row counts must match segment `frame_count`. For IMU, that count refers to
sample ticks (the larger of accel/gyro counts), rather than the total number of
interleaved rows. Decoded video frame counts must match the corresponding PTS
CSV. WAV rate, channel count and bit depth must match the stream metadata.

## Normalized episode

All CSV timestamps are parsed as integer microseconds. One common origin is
subtracted before converting to floating-point seconds, retaining relative
offsets between streams and parts. The origin is the earliest enabled sensor
timestamp, not the directory name or `recording.start_time`.

The master clock prefers left camera, then right camera, accelerometer,
gyroscope, and audio block PTS. `n_steps` counts entries on this master clock.
Each other modality retains its own clock and sample count:

| Key | Content |
| --- | --- |
| `observations["imu_accel"]` | Native-rate XYZ acceleration, float32 |
| `observations["imu_gyro"]` | Native-rate XYZ angular velocity, float32 |
| `observations["camera_left/right"]` | RGB uint8 frames, only when decoding images |
| `observations["audio"]` | Normalized 16-bit PCM, `(samples, channels)`, only when decoding audio |
| `obs_timestamps["camera_left/right"]` | Original per-frame device clock |
| `obs_timestamps["imu_accel/gyro"]` | Independent per-sensor device clock |
| `obs_timestamps["audio_pts"]` | Original block PTS; not a per-sample clock |
| `obs_timestamps["audio"]` | Derived clock from each WAV start PTS and sample rate, when decoding audio |
| `actions` | Empty `(0, 0)` array |

`EpisodeMetadata.extra` preserves the original metadata (`lumos`), integer
`timestamp_origin_us`, `master_clock`, sorted `segments`, `video_paths`,
`video_paths_by_camera`, WAV header information (`audio_segments`), and
`calibration_paths`. Calibration units, transforms and time offsets are
preserved without applying corrections. IMU physical units are not inferred.

## Optional decoding

```python
from calibra.ingestion.adapters.lumos import LumosReader

batch = LumosReader().read("datasets/lumos")
decoded = LumosReader(
    decode_images=True,
    decode_audio=True,
    max_image_size=256,
).read("datasets/lumos")
```

Base loading needs only the core dependencies: JSON, CSV and WAV header
reading use the Python standard library. Video decoding requires PyAV via
`calibra-robotics[lumos]`. Audio decoding currently supports 16-bit PCM.

Image decoding retains every frame and resizes each to a maximum side length
of 256 pixels by default. Use `max_image_size=None` for full resolution.
Episodes and decoded arrays are loaded into memory; read individual episodes
for large collections. Image-quality thresholds depend on resolution, and
the current camera quality analyzers use the first matching camera stream.

## Diagnostic scope

Timestamp jitter/dropout checks use the master clock. With decoded images,
the existing camera checks can run. Video framerate checks use the referenced
media and nominal camera FPS. These are acquisition diagnostics, not proof of
training readiness or calibration accuracy.

Robot-action analyzers are skipped and appear in `not_evaluated` /
`skipped_analyzers`. Mixed analyzers retain episode-length checks and skip
their action metrics. Calibra Score rejects reports missing action-based motion
analysis, preventing sensor-only recordings from receiving full motion credit.
Action-dependent quality, pruning and training recommendations are outside
this reader's supported workflow.
