"""
LeRobot adapter.

LeRobot stores datasets as HuggingFace datasets (Parquet shards) with a
standard schema:

  observation.images.<camera>   (T,) — encoded frames or paths
  observation.state             (T, state_dim)
  action                        (T, action_dim)
  timestamp                     (T,)
  episode_index                 (T,)   — integer episode ID per step
  frame_index                   (T,)   — step index within episode

Supports:
  - HuggingFace Hub repo IDs:  "lerobot/pusht"
  - HuggingFace Hub URIs:      "hf://lerobot/pusht"
  - Local disk (v1):  directory with metadata.json or dataset_dict.json
  - Local disk (v2):  directory with meta/info.json + Parquet shards
  - Local disk (v3):  directory with meta/info.json (codebase_version "v3.0")
                      + Parquet shards; same fast path as v2
                      (fast path: DuckDB reads Parquet directly without
                       loading image columns into RAM)

Image feature columns are skipped by default for performance. They can be
decoded instead via `LeRobotReader(decode_images=True)` — opt-in, since it
changes load time and memory characteristics.
  - v1 datasets: Decodes HuggingFace `Image`-feature columns into (T, H, W, C) arrays.
  - v2/v3 datasets: Decodes per-episode or chunk-concatenated `.mp4` video files
    referenced in `videos/` into (T, H, W, C) uint8 RGB arrays.

Dependencies:
  pip install 'calibra[lerobot]'  (datasets, pyarrow, duckdb, pillow, av/opencv-python)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.normalization import normalize_obs_keys

if TYPE_CHECKING:
    import datasets as _hf_datasets
    import pandas as _pd


def _require_datasets() -> "_hf_datasets":
    try:
        import datasets

        return datasets
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for the LeRobot adapter.\n"
            "Install it with: pip install 'calibra[lerobot]'"
        ) from None


def _require_duckdb():
    try:
        import duckdb

        return duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required for fast local LeRobot Parquet scanning.\n"
            "Install it with: pip install 'calibra[lerobot]'"
        ) from None


def _require_pillow():
    try:
        from PIL import Image

        return Image
    except ImportError:
        raise ImportError(
            "Pillow is required to decode LeRobot image columns.\n"
            "Install it with: pip install 'calibra[lerobot]'"
        ) from None


def _require_video_decoder():
    """
    Check and return an available video decoding backend ('av', 'cv2', or 'imageio').
    """
    try:
        import av

        return "av", av
    except ImportError:
        pass
    try:
        import cv2

        return "cv2", cv2
    except ImportError:
        pass
    try:
        import imageio.v3 as iio

        return "imageio", iio
    except ImportError:
        pass
    raise ImportError(
        "A video decoding library (PyAV or OpenCV) is required to decode LeRobot v2/v3 videos.\n"
        "Install it with: pip install av  or  pip install opencv-python"
    )


def _is_hub_uri(path: str) -> bool:
    """True for 'hf://lerobot/pusht' style URIs."""
    return path.startswith("hf://")


def _strip_hf_prefix(path: str) -> str:
    """Remove 'hf://' prefix if present, returning the bare repo ID."""
    return path[len("hf://") :] if _is_hub_uri(path) else path


def _is_hub_id(path: str) -> bool:
    """
    True for strings like "lerobot/pusht" that are Hub repo IDs rather than
    local filesystem paths. Heuristic: path doesn't exist locally, contains
    exactly one "/" with non-empty parts on both sides, and no filesystem
    indicators (backslash, drive letter, known extensions).
    """
    if _is_hub_uri(path):
        return True
    p = Path(path)
    if p.exists():
        return False
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        return False
    return not any(c in path for c in ("\\", ":", ".parquet", ".h5", ".hdf5", ".json"))


@register
class LeRobotReader(DatasetReader):
    """
    Reads LeRobot-format HuggingFace datasets from Hub or local disk.

    Parameters
    ----------
    decode_images : if True, decode HuggingFace `Image`-feature columns (v1)
                     or video files in `videos/` (v2/v3) into (T, H, W, C) uint8
                     arrays instead of excluding them. Opt-in because it increases
                     load time and memory use.
    """

    def __init__(
        self,
        decode_images: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self.decode_images = decode_images
        self.progress_callback = progress_callback

    @property
    def format_name(self) -> str:
        return "lerobot"

    @classmethod
    def can_read(cls, path: str) -> bool:
        if _is_hub_id(path):
            return True
        p = Path(_strip_hf_prefix(path))
        return p.is_dir() and (
            (p / "metadata.json").exists()
            or (p / "dataset_dict.json").exists()
            or (p / "meta" / "info.json").exists()
            or any(p.glob("*.parquet"))
            or any(p.glob("data/*.parquet"))
        )

    def read(self, path: str) -> EpisodeBatch:
        bare = _strip_hf_prefix(path)

        if _is_hub_id(path):
            hf = _require_datasets()
            ds, dataset_name, task = self._load_hub(hf, bare)
            scalar_cols = self._scalar_columns(hf, ds)
            ds = ds.select_columns(scalar_cols)
            df = ds.to_pandas()
            episodes = self._episodes_from_df(
                df, task, path, progress_callback=self.progress_callback
            )
        else:
            p = Path(bare)
            if (p / "meta" / "info.json").exists():
                # v2/v3 format (meta/info.json + Parquet shards): DuckDB fast path, pyarrow fallback
                try:
                    return self._read_local_v2_duckdb(p, path)
                except ImportError:
                    return self._read_local_v2_pyarrow(p, path)
            else:
                hf = _require_datasets()
                ds, dataset_name, task = self._load_local(hf, p)
                scalar_cols = self._scalar_columns(hf, ds)
                ds = ds.select_columns(scalar_cols)
                df = ds.to_pandas()
                episodes = self._episodes_from_df(
                    df, task, path, progress_callback=self.progress_callback
                )
                return EpisodeBatch(
                    episodes=episodes,
                    dataset_name=p.name,
                    format=self.format_name,
                    source_path=path,
                )

        dataset_name = bare.split("/")[-1]
        return EpisodeBatch(
            episodes=episodes,
            dataset_name=dataset_name,
            format=self.format_name,
            source_path=path,
        )

    # ── DuckDB v2 fast path ──────────────────────────────────────────────────

    def _read_local_v2_duckdb(self, p: Path, source: str) -> EpisodeBatch:
        """
        Read a v2/v3 LeRobot dataset (meta/info.json + Parquet shards) using DuckDB.

        Advantages over the HuggingFace datasets path:
          - Reads Parquet natively; no Python object conversion overhead.
          - Image/video columns are excluded via a SQL projection before any data
            leaves the Parquet pages — they never enter RAM.
          - Global aggregate queries (episode count, action bounds) run in
            under a second on multi-terabyte datasets via DuckDB's push-down.
          - For large datasets (e.g. DROID with 76k episodes), use
            iter_episodes_lazy() to query one episode at a time.
        """
        conn = self._build_duckdb_conn(p)
        task = _read_task_v2(p)

        df = conn.execute("SELECT * FROM dataset").df()
        conn.close()

        video_resolver = None
        if self.decode_images:
            info_path = p / "meta" / "info.json"
            if info_path.exists():
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                video_resolver = VideoDatasetResolver(p, info)

        episodes = self._episodes_from_df(
            df, task, source, video_resolver=video_resolver, progress_callback=self.progress_callback
        )
        return EpisodeBatch(
            episodes=episodes,
            dataset_name=p.name,
            format=self.format_name,
            source_path=source,
        )

    def _read_local_v2_pyarrow(self, p: Path, source: str) -> EpisodeBatch:
        """
        Fallback reader for v2/v3 LeRobot datasets when DuckDB is not installed.

        Uses pyarrow.parquet with column projection to exclude image columns,
        so image bytes never enter RAM even without DuckDB.
        """
        import pyarrow.parquet as pq

        info_path = p / "meta" / "info.json"
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        image_cols = _image_columns_from_info(info)
        parquet_files = sorted(p.glob("data/**/*.parquet"))
        if not parquet_files:
            parquet_files = sorted(p.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No Parquet files found in {p}")

        # Determine scalar columns from first file's schema
        schema = pq.read_schema(str(parquet_files[0]))
        scalar_cols = [c for c in schema.names if c not in image_cols]

        tables = [pq.read_table(str(f), columns=scalar_cols) for f in parquet_files]

        import pyarrow as pa

        combined = pa.concat_tables(tables)
        df = combined.to_pandas()
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        task = _read_task_v2(p)
        video_resolver = None
        if self.decode_images:
            video_resolver = VideoDatasetResolver(p, info)

        episodes = self._episodes_from_df(
            df, task, source, video_resolver=video_resolver, progress_callback=self.progress_callback
        )
        return EpisodeBatch(
            episodes=episodes,
            dataset_name=p.name,
            format=self.format_name,
            source_path=source,
        )

    def iter_episodes_lazy(self, path: str):
        """
        Yield Episode objects one at a time without loading the full dataset.

        Use this for multi-terabyte datasets where loading everything into
        RAM is not feasible (e.g. DROID, BridgeData V2 full splits).
        Each episode is fetched with a single SQL WHERE clause, so only that
        episode's rows are transferred from Parquet into Python memory.

        Parameters
        ----------
        path : local v2/v3 dataset directory (must have meta/info.json).
               Hub IDs and v1 formats are not supported by this method.
        """
        p = Path(_strip_hf_prefix(path))
        info_path = p / "meta" / "info.json"
        if not info_path.exists():
            raise ValueError(
                f"iter_episodes_lazy requires a v2 local dataset (meta/info.json). "
                f"'{path}' does not appear to be v2 format."
            )
        conn = self._build_duckdb_conn(p)
        task = _read_task_v2(p)

        video_resolver = None
        if self.decode_images:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            video_resolver = VideoDatasetResolver(p, info)

        episode_ids: list[int] = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT episode_index FROM dataset ORDER BY episode_index"
            ).fetchall()
        ]

        for ep_id in episode_ids:
            df = conn.execute(
                f"SELECT * FROM dataset WHERE episode_index = {ep_id} ORDER BY frame_index"
            ).df()
            yield self._episode_from_group(df, ep_id, task, path, video_resolver=video_resolver)

        conn.close()

    def _build_duckdb_conn(self, p: Path):
        """
        Create a DuckDB in-memory connection with a 'dataset' view over all
        Parquet shards in `p`, projecting out image columns.
        """
        duckdb = _require_duckdb()
        conn = duckdb.connect(":memory:")

        info_path = p / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        image_cols = _image_columns_from_info(info)

        parquet_files = sorted(p.glob("data/**/*.parquet"))
        if not parquet_files:
            parquet_files = sorted(p.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No Parquet files found in {p}")

        file_list_sql = ", ".join(f"'{f}'" for f in parquet_files)
        conn.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet([{file_list_sql}])")

        all_cols: list[str] = [row[0] for row in conn.execute("DESCRIBE raw").fetchall()]
        scalar_cols = [c for c in all_cols if c not in image_cols]
        col_sql = ", ".join(f'"{c}"' for c in scalar_cols)
        conn.execute(
            f"CREATE VIEW dataset AS SELECT {col_sql} FROM raw ORDER BY episode_index, frame_index"
        )
        return conn

    # ── loading helpers (HF hub / v1 disk) ───────────────────────────────────

    @staticmethod
    def _load_hub(
        hf: "_hf_datasets", path: str
    ) -> tuple["_hf_datasets.Dataset", str, Optional[str]]:
        dataset_name = path.split("/")[-1]
        try:
            ds = hf.load_dataset(path, split="train")
        except Exception:
            dd = hf.load_dataset(path)
            split = next(iter(dd))
            ds = dd[split]
        return ds, dataset_name, None

    @staticmethod
    def _load_local(
        hf: "_hf_datasets", p: Path
    ) -> tuple["_hf_datasets.Dataset", str, Optional[str]]:
        ds = hf.load_from_disk(str(p))
        task = _read_task_v1(p)
        if hasattr(ds, "keys"):
            split = next(iter(ds))
            ds = ds[split]
        return ds, p.name, task

    # ── column filtering ─────────────────────────────────────────────────────

    def _scalar_columns(self, hf: "_hf_datasets", ds: "_hf_datasets.Dataset") -> list[str]:
        """
        Return columns to load. HuggingFace Image features are excluded unless
        `self.decode_images` is set, in which case they're kept for decoding
        in `_episode_from_group`.
        """
        try:
            from datasets import Image as HFImage

            if self.decode_images:
                return list(ds.features.keys())
            return [col for col, feat in ds.features.items() if not isinstance(feat, HFImage)]
        except ImportError:
            if self.decode_images:
                return list(ds.column_names)
            return [col for col in ds.column_names if "image" not in col.lower()]

    # ── episode construction ─────────────────────────────────────────────────

    @staticmethod
    def _episodes_from_df(
        df: "_pd.DataFrame",
        task: Optional[str],
        source: str,
        video_resolver: Optional[VideoDatasetResolver] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[Episode]:
        """Split a full-dataset DataFrame into per-episode Episode objects."""
        episode_col = "episode_index"
        if episode_col not in df.columns:
            raise ValueError(
                f"Expected column '{episode_col}' in LeRobot dataset.\n"
                f"Available columns: {list(df.columns)}"
            )
        episodes: list[Episode] = []
        cum_frame_offset = 0
        groups = list(df.groupby(episode_col, sort=True))
        total_episodes = len(groups)
        for idx, (ep_id, group) in enumerate(groups):
            if progress_callback:
                progress_callback(idx, total_episodes, f"Episode {ep_id}")
            episodes.append(
                LeRobotReader._episode_from_group(
                    group,
                    int(ep_id),
                    task,
                    source,
                    video_resolver=video_resolver,
                    fallback_start_frame=cum_frame_offset,
                )
            )
            cum_frame_offset += len(group)
            if progress_callback:
                progress_callback(idx + 1, total_episodes, f"Episode {ep_id}")
        return episodes

    @staticmethod
    def _episode_from_group(
        group: "_pd.DataFrame",
        ep_id: int,
        task: Optional[str],
        source: str,
        video_resolver: Optional[VideoDatasetResolver] = None,
        fallback_start_frame: int = 0,
    ) -> Episode:
        if "frame_index" in group.columns:
            group = group.sort_values("frame_index")

        timestamps = group["timestamp"].to_numpy(dtype=np.float64)
        actions = np.array(group["action"].tolist(), dtype=np.float32)

        raw_obs: dict[str, np.ndarray] = {}
        for col in group.columns:
            if col.startswith("observation."):
                key = col.removeprefix("observation.")
                try:
                    raw_obs[key] = np.array(group[col].tolist(), dtype=np.float32)
                except (ValueError, TypeError):
                    decoded = _decode_image_column(group[col].tolist())
                    if decoded is not None:
                        raw_obs[key] = decoded

        # Decode video columns for v2/v3 datasets when requested
        if video_resolver is not None:
            for vcol in video_resolver.video_cols:
                key = vcol.removeprefix("observation.")
                if key not in raw_obs:
                    v_path, s_frame, n_frames = video_resolver.resolve_video_slice(
                        vcol,
                        ep_id,
                        ep_len=len(timestamps),
                        fallback_start_frame=fallback_start_frame,
                    )
                    if v_path is not None:
                        decoded = _decode_video_frames(
                            v_path, start_frame=s_frame, num_frames=n_frames
                        )
                        if decoded is not None:
                            raw_obs[key] = decoded

        obs = normalize_obs_keys(raw_obs)

        return Episode(
            metadata=EpisodeMetadata(
                episode_id=str(ep_id),
                task_description=task,
                source_file=source,
            ),
            timestamps=timestamps,
            observations=obs,
            actions=actions,
        )


# ── image & video decoding ───────────────────────────────────────────────────


def _decode_image_column(cells: list) -> Optional[np.ndarray]:
    """
    Decode a HuggingFace Image-feature column into a (T, H, W, C) uint8 array.

    `Dataset.to_pandas()` yields `{"bytes": <encoded>, "path": <str|None>}`
    dicts per cell (verified empirically — it does NOT eagerly decode to
    pixels the way `Dataset.__getitem__` does). Handles that shape, plus an
    already-decoded PIL.Image cell defensively. Returns None (rather than
    raising) if the cells aren't image data, so callers can fall back to
    dropping the column exactly as they did before this column existed.
    """
    try:
        PILImage = _require_pillow()
    except ImportError:
        return None

    frames = []
    for cell in cells:
        try:
            if isinstance(cell, dict):
                data = cell.get("bytes")
                if data is None and cell.get("path"):
                    with open(cell["path"], "rb") as f:
                        data = f.read()
                if data is None:
                    return None
                img = PILImage.open(io.BytesIO(data)).convert("RGB")
            elif isinstance(cell, PILImage.Image):
                img = cell.convert("RGB")
            else:
                return None
        except Exception:
            return None
        frames.append(np.array(img, dtype=np.uint8))

    return np.stack(frames)


def _decode_video_frames(
    video_path: Path,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Decode a range of frames from a video file into a (T, H, W, 3) uint8 RGB array.
    """
    if not video_path.exists():
        return None

    try:
        backend_name, backend = _require_video_decoder()
    except ImportError:
        return None

    if backend_name == "av":
        try:
            container = backend.open(str(video_path))
            if not container.streams.video:
                container.close()
                return None
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            frames = []
            frame_idx = 0
            end_frame = (start_frame + num_frames) if num_frames is not None else None

            for frame in container.decode(stream):
                if frame_idx >= start_frame:
                    if end_frame is not None and frame_idx >= end_frame:
                        break
                    rgb = frame.to_ndarray(format="rgb24")
                    frames.append(rgb)
                frame_idx += 1
            container.close()
            if not frames:
                return None
            return np.stack(frames)
        except Exception:
            return None

    elif backend_name == "cv2":
        try:
            cap = backend.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            if start_frame > 0:
                cap.set(backend.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            count = 0
            while True:
                if num_frames is not None and count >= num_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = backend.cvtColor(frame, backend.COLOR_BGR2RGB)
                frames.append(rgb)
                count += 1
            cap.release()
            if not frames:
                return None
            return np.stack(frames)
        except Exception:
            return None

    elif backend_name == "imageio":
        try:
            if num_frames is not None:
                imgs = backend.imread(str(video_path), index=slice(start_frame, start_frame + num_frames))
            else:
                imgs = backend.imread(str(video_path))
            return np.asarray(imgs, dtype=np.uint8)
        except Exception:
            return None

    return None


class VideoDatasetResolver:
    """
    Locates and indexes video files and episode frame slices for LeRobot v2 and v3 datasets.
    """

    def __init__(self, root: Path, info: dict) -> None:
        self.root = root
        self.info = info
        self.codebase_version = str(info.get("codebase_version", "v2.0")).lower()
        self.fps = float(info.get("fps", 30.0))
        self.video_cols = _image_columns_from_info(info)
        self.episodes_meta = self._load_episodes_meta()
        self._cumulative_offsets = self._build_cumulative_offsets()

    def _load_episodes_meta(self) -> dict[int, dict]:
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            result = {}
            with open(episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    ep_id = record.get("episode_index")
                    if ep_id is not None:
                        result[int(ep_id)] = record
            return result

        # Also support LeRobot v3 Parquet meta: meta/episodes/**/*.parquet or meta/episodes.parquet
        pq_files = sorted(self.root.glob("meta/episodes/**/*.parquet"))
        if not pq_files:
            pq_files = sorted(self.root.glob("meta/episodes.parquet"))
            if not pq_files:
                pq_files = sorted(self.root.glob("meta/episodes/*.parquet"))
        if pq_files:
            try:
                import pyarrow.parquet as pq

                result = {}
                for pf in pq_files:
                    df = pq.read_table(str(pf)).to_pandas()
                    for _, row in df.iterrows():
                        ep_id = row.get("episode_index")
                        if ep_id is not None:
                            result[int(ep_id)] = row.to_dict()
                return result
            except Exception:
                pass

        return {}

    def _build_cumulative_offsets(self) -> dict[int, tuple[int, int]]:
        offsets = {}
        curr = 0
        for ep_id in sorted(self.episodes_meta.keys()):
            meta = self.episodes_meta[ep_id]
            length = meta.get("length", 0)
            offsets[ep_id] = (curr, length)
            curr += length
        return offsets

    def resolve_video_slice(
        self,
        col: str,
        ep_id: int,
        ep_len: Optional[int] = None,
        fallback_start_frame: int = 0,
    ) -> tuple[Optional[Path], int, Optional[int]]:
        """
        Returns (video_file_path, start_frame, num_frames) for the given column and episode.
        """
        col_names = [col]
        for prefix in ("observation.images.", "observation.", "images."):
            if col.startswith(prefix):
                col_names.append(col.removeprefix(prefix))
        col_names = list(dict.fromkeys(col_names))

        exts = (".mp4", ".mkv", ".avi", ".webm")
        video_root = self.root / "videos"
        if not video_root.exists():
            return None, 0, ep_len

        meta = self.episodes_meta.get(ep_id, {})
        video_path_tpl = self.info.get("video_path")

        # 0. Try info.json video_path template with episode metadata indices
        if video_path_tpl:
            for cname in col_names:
                chunk_idx = (
                    meta.get(f"videos/{col}/chunk_index")
                    or meta.get(f"videos/{cname}/chunk_index")
                    or meta.get("data/chunk_index")
                    or meta.get("chunk_index")
                    or 0
                )
                file_idx = (
                    meta.get(f"videos/{col}/file_index")
                    or meta.get(f"videos/{cname}/file_index")
                    or meta.get("data/file_index")
                    or meta.get("file_index")
                    or 0
                )
                try:
                    rel_path = video_path_tpl.format(
                        video_key=cname,
                        chunk_index=int(chunk_idx),
                        file_index=int(file_idx),
                    )
                    cand = self.root / rel_path
                    if cand.is_file():
                        start_f, length = self._get_v3_slice(
                            col, ep_id, ep_len, fallback_start_frame
                        )
                        return cand, start_f, length
                except (KeyError, ValueError):
                    pass

        search_dirs = [video_root] + sorted(video_root.glob("chunk-*"))

        # 1. Try v2 layout: per-episode video file
        for sdir in search_dirs:
            for cname in col_names:
                col_dir = sdir / cname
                if col_dir.is_dir():
                    for ext in exts:
                        candidates = [
                            col_dir / f"episode_{ep_id:06d}{ext}",
                            col_dir / f"episode_{ep_id}{ext}",
                            col_dir / f"{ep_id:06d}{ext}",
                            col_dir / f"{ep_id}{ext}",
                        ]
                        for cand in candidates:
                            if cand.is_file():
                                return cand, 0, ep_len

        # 2. Try v3 chunk layout: recursive videos/{cname}/**/*.mp4 or videos/chunk-*/**/*.mp4
        for sdir in search_dirs:
            for cname in col_names:
                col_dir = sdir / cname
                if col_dir.is_dir():
                    for ext in exts:
                        for cand in sorted(col_dir.glob(f"**/*{ext}")):
                            if cand.is_file() and "episode" not in cand.name:
                                start_f, length = self._get_v3_slice(
                                    col, ep_id, ep_len, fallback_start_frame
                                )
                                return cand, start_f, length

        # 3. Direct videos/{cname}.mp4 or videos/chunk-*/{cname}.mp4
        for sdir in search_dirs:
            for cname in col_names:
                for ext in exts:
                    cand = sdir / f"{cname}{ext}"
                    if cand.is_file():
                        start_f, length = self._get_v3_slice(
                            col, ep_id, ep_len, fallback_start_frame
                        )
                        return cand, start_f, length

        return None, 0, ep_len

    def _get_v3_slice(
        self, col: str, ep_id: int, ep_len: Optional[int], fallback_start: int
    ) -> tuple[int, Optional[int]]:
        meta = self.episodes_meta.get(ep_id, {})
        col_names = [col]
        for prefix in ("observation.images.", "observation.", "images."):
            if col.startswith(prefix):
                col_names.append(col.removeprefix(prefix))

        for c in col_names:
            for k in (f"videos/{c}/from_frame", f"videos.{c}.from_frame", "from_frame"):
                if k in meta and meta[k] is not None:
                    start_f = int(meta[k])
                    to_f = int(meta.get(k.replace("from_frame", "to_frame"), start_f + (ep_len or 0)))
                    return start_f, max(0, to_f - start_f)

            for k in (
                f"videos/{c}/from_timestamp",
                f"videos.{c}.from_timestamp",
                "from_timestamp",
            ):
                if k in meta and meta[k] is not None:
                    from_ts = float(meta[k])
                    to_ts = float(meta.get(k.replace("from_timestamp", "to_timestamp"), from_ts))
                    start_f = int(round(from_ts * self.fps))
                    num_f = int(round((to_ts - from_ts) * self.fps))
                    return start_f, num_f or ep_len

        if ep_id in self._cumulative_offsets:
            start_f, length = self._cumulative_offsets[ep_id]
            return start_f, length or ep_len

        return fallback_start, ep_len


# ── metadata readers ─────────────────────────────────────────────────────────


def _image_columns_from_info(info: dict) -> set[str]:
    """
    Return the set of column names that carry image/video data in a v2 info.json.

    LeRobot v2 info.json has a top-level "features" dict.  Each entry whose
    dtype is "image" or "video" — or whose nested "info" dict has an "encoding"
    key — is an image column.  We also fall back to a name-heuristic for
    datasets that deviate from the spec.
    """
    image_cols: set[str] = set()
    for col_name, feat in info.get("features", {}).items():
        if not isinstance(feat, dict):
            continue
        dtype = feat.get("dtype", "")
        if dtype in ("image", "video"):
            image_cols.add(col_name)
            continue
        if "encoding" in feat.get("info", {}):
            image_cols.add(col_name)
            continue
        # name heuristic: fall back for non-standard schemas
        if any(tok in col_name.lower() for tok in ("image", "video", "camera", "rgb", "depth")):
            image_cols.add(col_name)
    return image_cols


def _read_task_v1(p: Path) -> Optional[str]:
    meta = p / "metadata.json"
    if meta.exists():
        with open(meta) as f:
            return json.load(f).get("task_description")
    return None


def _read_task_v2(p: Path) -> Optional[str]:
    info = p / "meta" / "info.json"
    if info.exists():
        with open(info) as f:
            data = json.load(f)
            return data.get("task_description") or data.get("description")
    tasks = p / "meta" / "tasks.jsonl"
    if tasks.exists():
        with open(tasks) as f:
            first = f.readline()
            if first:
                return json.loads(first).get("task")
    return None
