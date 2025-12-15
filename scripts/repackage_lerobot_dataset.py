#!/usr/bin/env python
"""
Convert an existing LeRobot dataset that already lives under
`~/.cache/huggingface/lerobot/{repo_id}` into a new LeRobot dataset with
OpenPI-friendly feature names (`image`, `wrist_image`, `state`, `actions`).

This script is useful when you download a dataset such as
`libero_plus_lerobot` from the Hugging Face Hub and want to:
    * rename the visual keys to the names expected by OpenPI
    * optionally downsample the videos (e.g. 20 FPS -> 10 FPS) by skipping frames
    * store the resulting visual data as png images instead of mp4 videos

Example:
    uv run scripts/repackage_lerobot_dataset.py \\
        --source-repo libero_plus_lerobot \\
        --target-repo libero_plus_pi \\
        --target-fps 10 \\
        --overwrite

Notes:
    * You need `torch`, `tyro`, and the `lerobot` python package installed.
    * The input dataset must already be in LeRobot format (v2.1 or v3.0).
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import List

import numpy as np
import tyro
from tqdm import tqdm

from lerobot.common.datasets.image_writer import AsyncImageWriter
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import torch


TARGET_FEATURES = {
    "image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],
    },
}

SOURCE_TO_TARGET_FEATURE_MAP = {
    "image": "observation.images.front",
    "wrist_image": "observation.images.wrist",
    "state": "observation.state",
    "actions": "actions",
}


@dataclasses.dataclass
class ConvertConfig:
    # Required identifiers
    source_repo: str = "libero_plus_lerobot"
    target_repo: str = "libero_plus_wo_video"

    # Optional filesystem overrides
    source_root: Path | None = None
    target_root: Path | None = None

    # Temporal controls
    target_fps: int | None = 10
    frame_stride: int | None = None

    # Episode selection
    start_episode: int = 0
    end_episode: int | None = None
    limit_episodes: int | None = None

    # Misc
    robot_type: str = "panda"
    overwrite: bool = False
    copy_readme: bool = True
    image_writer_threads: int = 8
    image_writer_processes: int = 0
    video_backend: str | None = None
    log_every: int = 50
    max_total_frames: int | None = None  # e.g. set to 10 for a tiny preview dataset


def _resolve_root(user_path: Path | None) -> Path:
    return Path(user_path) if user_path is not None else HF_LEROBOT_HOME


def _compute_stride_and_fps(
    source_fps: int, target_fps: int | None, frame_stride: int | None
) -> tuple[int, int]:
    """
    Decide how many frames to skip when iterating over the source dataset and
    return (stride, resulting_fps).
    """
    if frame_stride is not None:
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be > 0, received {frame_stride}.")
        if source_fps % frame_stride != 0:
            raise ValueError(
                f"frame_stride={frame_stride} must divide source fps ({source_fps})."
            )
        return frame_stride, source_fps // frame_stride

    if target_fps is None:
        return 1, source_fps

    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, received {target_fps}.")
    if target_fps > source_fps:
        raise ValueError(
            f"target_fps ({target_fps}) cannot exceed source fps ({source_fps})."
        )
    if source_fps % target_fps != 0:
        raise ValueError(
            f"source fps ({source_fps}) must be divisible by target_fps ({target_fps}). "
            "Use --frame-stride instead if you need a non-divisible ratio."
        )
    stride = source_fps // target_fps
    return stride, target_fps


def _select_episode_indices(
    total_episodes: int, config: ConvertConfig
) -> List[int]:
    start = max(0, config.start_episode)
    end = (
        min(total_episodes, config.end_episode)
        if config.end_episode is not None
        else total_episodes
    )
    if start >= end:
        raise ValueError(
            f"start_episode ({config.start_episode}) must be < end_episode ({config.end_episode})."
        )

    indices = list(range(start, end))
    if config.limit_episodes is not None:
        indices = indices[: config.limit_episodes]
    return indices


def _ensure_target_does_not_exist(target_path: Path, overwrite: bool) -> None:
    if target_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target dataset {target_path} already exists. "
                "Pass --overwrite to delete it."
            )
        shutil.rmtree(target_path)


def _validate_source_features(source_dataset: LeRobotDataset) -> None:
    missing = [
        source_key
        for source_key in SOURCE_TO_TARGET_FEATURE_MAP.values()
        if source_key not in source_dataset.features
    ]
    if missing:
        raise KeyError(
            "Source dataset is missing required features: "
            + ", ".join(missing)
        )


def _copy_readme_if_present(
    source_path: Path, target_path: Path, enabled: bool
) -> None:
    if not enabled:
        return
    readme = source_path / "README.md"
    if readme.exists():
        shutil.copy(readme, target_path / "README.md")


def _tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _convert_image(image) -> np.ndarray:
    img = _tensor_to_numpy(image)
    if img.ndim == 3 and img.shape[0] in {1, 3}:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    return img


def _convert_vector(value, dtype: np.dtype = np.float32) -> np.ndarray:
    return _tensor_to_numpy(value).astype(dtype)


def _episode_span(ep_meta: dict, fallback_start: int) -> tuple[int, int]:
    """Return (dataset_from_index, dataset_to_index) for both v3 and v2.1 metadata."""
    if "dataset_from_index" in ep_meta and "dataset_to_index" in ep_meta:
        return ep_meta["dataset_from_index"], ep_meta["dataset_to_index"]

    # v2.1 metadata only has per-episode length
    if "length" not in ep_meta:
        raise KeyError(
            "Episode metadata lacks both dataset indices and a length field, "
            "cannot derive frame span."
        )
    length = ep_meta["length"]
    return fallback_start, fallback_start + length


def convert(config: ConvertConfig) -> None:
    source_root = _resolve_root(config.source_root)
    target_root = _resolve_root(config.target_root)

    source_path = source_root / config.source_repo
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source dataset not found at {source_path}. "
            "Make sure it has been downloaded beforehand."
        )

    print(f"Loading source dataset from {source_path} ...")

    source_dataset = LeRobotDataset(
        repo_id=config.source_repo,
        #root=source_root,
        video_backend=config.video_backend,
    )
    _validate_source_features(source_dataset)

    stride, target_fps = _compute_stride_and_fps(
        source_dataset.fps, config.target_fps, config.frame_stride
    )
    print(
        f"Converting {config.source_repo} ({source_dataset.fps} FPS) -> "
        f"{config.target_repo} ({target_fps} FPS) using stride={stride}."
    )

    target_path = target_root / config.target_repo
    _ensure_target_does_not_exist(target_path, config.overwrite)

    print(f"Creating target dataset under {target_path} ...")
    target_dataset = LeRobotDataset.create(
        repo_id=config.target_repo,
        fps=target_fps,
        robot_type=config.robot_type,
        features=TARGET_FEATURES,
        use_videos=False,
    )

    if config.image_writer_threads > 0:
        target_dataset.image_writer = AsyncImageWriter(
            num_processes=config.image_writer_processes,
            num_threads=config.image_writer_threads,
        )

    selected_episodes = _select_episode_indices(
        source_dataset.num_episodes, config
    )
    print(f"Selected {len(selected_episodes)} / {source_dataset.num_episodes} episodes.")

    total_frames_written = 0
    stop_conversion = False
    fallback_offset = 0
    for local_ep_idx, source_ep_idx in enumerate(
        tqdm(selected_episodes, desc="Converting episodes")
    ):
        ep_meta = source_dataset.meta.episodes[source_ep_idx]
        start_idx, end_idx = _episode_span(ep_meta, fallback_offset)
        fallback_offset = end_idx

        frames_this_episode = 0
        for global_idx in range(start_idx, end_idx, stride):
            if (
                config.max_total_frames is not None
                and total_frames_written >= config.max_total_frames
            ):
                stop_conversion = True
                break
            sample = source_dataset[global_idx]
            frame = {
                "image": _convert_image(sample[SOURCE_TO_TARGET_FEATURE_MAP["image"]]),
                "wrist_image": _convert_image(
                    sample[SOURCE_TO_TARGET_FEATURE_MAP["wrist_image"]]
                ),
                "state": _convert_vector(sample[SOURCE_TO_TARGET_FEATURE_MAP["state"]]),
                "actions": _convert_vector(
                    sample[SOURCE_TO_TARGET_FEATURE_MAP["actions"]]
                ),
                "task": sample["task"],
            }
            target_dataset.add_frame(frame)
            frames_this_episode += 1
            total_frames_written += 1

        if frames_this_episode > 0:
            target_dataset.save_episode()

        if stop_conversion:
            break

        if (
            config.log_every > 0
            and (local_ep_idx + 1) % config.log_every == 0
        ):
            print(
                f"[{local_ep_idx + 1}/{len(selected_episodes)}] "
                f"Converted {frames_this_episode} frames in episode "
                f"{source_ep_idx} (total {total_frames_written})."
            )
        if stop_conversion:
            break

    _copy_readme_if_present(source_path, target_path, config.copy_readme)

    print(
        f"Done. Wrote {len(selected_episodes)} episodes / "
        f"{total_frames_written} frames to {target_path}."
    )


def main() -> None:
    config = tyro.cli(ConvertConfig)
    convert(config)


if __name__ == "__main__":
    main()

