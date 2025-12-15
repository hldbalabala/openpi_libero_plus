"""Utility functions for LIBERO evaluation."""
# Set MuJoCo rendering backend to EGL for headless environments
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
root_dir = (project_root / "third_party/LIBERO-plus").resolve()
sys.path.insert(0, str(root_dir))

import collections
import gc
import h5py
import json
import logging
import math
import multiprocessing as mp
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Union

import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import SegmentationRenderEnv
from openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def find_latest_output_folder(task_suite_name: str) -> pathlib.Path:
    """Find the latest output folder for the given task suite."""
    base_dir = pathlib.Path("data") / task_suite_name
    if not base_dir.exists():
        raise ValueError(f"Base directory {base_dir} does not exist. Cannot resume.")

    # Find all subdirectories (folders with timestamp names)
    folders = [d for d in base_dir.iterdir() if d.is_dir()]
    if not folders:
        raise ValueError(f"No existing folders found in {base_dir}. Cannot resume.")

    # Sort by modification time (newest first)
    folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_folder = folders[0]
    logging.info(f"Found latest folder: {latest_folder}")
    return latest_folder


def create_safe_filename(task_description: str, task_id: int, episode_idx: int, suffix: str, max_length: int = 240) -> str:
    """Create a safe filename that fits within filesystem limits.
    
    Most filesystems limit individual path components to 255 characters.
    This function ensures the filename stays within that limit.
    
    Args:
        task_description: The task description text
        task_id: Task ID for uniqueness
        episode_idx: Episode index for uniqueness
        suffix: "success" or "failure"
        max_length: Maximum length for the full filename (default 240 to leave margin)
    
    Returns:
        A safe filename string
    """
    # Create base task segment
    task_segment = task_description.replace(" ", "_")
    
    # Try simple format first: "rollout_" + task_segment + "_" + suffix + ".mp4"
    simple_filename = f"rollout_{task_segment}_{suffix}.mp4"
    
    # If it fits, use it
    if len(simple_filename) <= max_length:
        return simple_filename
    
    # Otherwise, truncate and add task_id and episode_idx for uniqueness
    # Format: "rollout_" + truncated_task + "_task{task_id}_ep{episode_idx}_" + suffix + ".mp4"
    task_ep_suffix = f"_task{task_id}_ep{episode_idx}_"
    prefix = "rollout_"
    suffix_part = f"{suffix}.mp4"
    
    # Calculate available space for task segment
    fixed_parts_len = len(prefix) + len(task_ep_suffix) + len(suffix_part)
    available_for_task = max_length - fixed_parts_len
    
    # Ensure we have at least some space for the task segment
    if available_for_task < 10:
        available_for_task = 10
    
    # Truncate task_segment to fit
    truncated_task = task_segment[:available_for_task]
    
    return f"{prefix}{truncated_task}{task_ep_suffix}{suffix_part}"


def load_progress_from_statistics(video_out_path: pathlib.Path, num_trials_per_task: int):
    """Load progress from category_statistics.txt file.
    
    Since tasks are executed sequentially, we can calculate the number of completed tasks
    from the total number of episodes: completed_tasks = total_episodes / num_trials_per_task
    """
    completed_tasks_count = 0
    total_episodes = 0
    total_successes = 0
    category_stats = defaultdict(lambda: {'successes': 0, 'total': 0})
    
    if not video_out_path.exists():
        return completed_tasks_count, total_episodes, total_successes, category_stats
    
    # Load statistics from category_statistics.txt if exists
    stats_file_path = video_out_path / "category_statistics.txt"
    if stats_file_path.exists():
        try:
            with open(stats_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Parse total line: "Total: 1431/1650=86.73%"
                total_match = re.search(r'Total:\s*(\d+)/(\d+)=', content)
                if total_match:
                    total_successes = int(total_match.group(1))
                    total_episodes = int(total_match.group(2))
                    # Calculate completed tasks: tasks are executed sequentially
                    completed_tasks_count = total_episodes // num_trials_per_task
                    logging.info(f"Loaded statistics: {total_successes} successes out of {total_episodes} episodes")
                    logging.info(f"Completed tasks: {completed_tasks_count} (calculated from {total_episodes} episodes / {num_trials_per_task} trials per task)")
                
                # Parse category statistics: "Category: successes/total=rate%"
                for line in content.split('\n'):
                    # Match lines like "Background Textures: 247/248=99.60%"
                    category_match = re.search(r'^([^:]+):\s*(\d+)/(\d+)=', line)
                    if category_match:
                        category = category_match.group(1).strip()
                        successes = int(category_match.group(2))
                        total = int(category_match.group(3))
                        category_stats[category] = {'successes': successes, 'total': total}
        except Exception as e:
            logging.warning(f"Failed to parse statistics file: {e}")
    
    return completed_tasks_count, total_episodes, total_successes, category_stats


def write_category_stats(
    stats_file_path: pathlib.Path,
    total_successes: int,
    total_episodes: int,
    category_stats: dict,
) -> None:
    """Persist category statistics to disk."""
    with open(stats_file_path, 'w', encoding='utf-8') as f:
        f.write("Category Statistics (Success数/总任务数=成功率)\n")
        f.write("=" * 60 + "\n")
        total_rate = (total_successes / total_episodes * 100) if total_episodes > 0 else 0.0
        f.write(f"Total: {total_successes}/{total_episodes}={total_rate:.2f}%\n")
        f.write("-" * 60 + "\n")
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            success_rate = (stats['successes'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
            f.write(f"{category}: {stats['successes']}/{stats['total']}={success_rate:.2f}%\n")
        f.write("=" * 60 + "\n")


def get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = SegmentationRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def stack_state(obs: dict) -> np.ndarray:
    """Stack robot state from observation."""
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(np.array(obs["robot0_eef_quat"]).copy()),
            obs["robot0_gripper_qpos"],
        )
    )


def episode_group_name(root: h5py.File, task_id: int, episode_idx: int) -> str:
    """Generate unique group name for episode in HDF5 file."""
    base = f"task{task_id}_episode{episode_idx}"
    if base not in root:
        return base
    suffix = 1
    while f"{base}_dup{suffix}" in root:
        suffix += 1
    return f"{base}_dup{suffix}"


def save_replay_episode(
    h5_file: h5py.File,
    group_name: str,
    images: List[np.ndarray],
    wrist_images: List[np.ndarray],
    img_segs: Union[List[np.ndarray], List[Dict[str, np.ndarray]]],
    states: List[np.ndarray],
    actions: List[np.ndarray],
    task_description: str,
    task_id: int,
    episode_idx: int,
):
    """Save replay episode data to HDF5 file.
    
    Args:
        img_segs: List of segmentation images. Can be:
            - List[np.ndarray]: Single segmentation per frame (normal mode)
            - List[Dict[str, np.ndarray]]: Multiple segmentations per frame (get_each_seg mode)
                Dictionary keys are object names (e.g., "robot", "cabinet"), values are seg images.
    """
    group = h5_file.create_group(group_name)
    group.attrs["task_id"] = task_id
    group.attrs["task_description"] = task_description
    group.attrs["episode_idx"] = episode_idx
    group.create_dataset("images", data=np.stack(images), compression="gzip")
    group.create_dataset("wrist_images", data=np.stack(wrist_images), compression="gzip")
    
    # Handle img_segs: can be List[np.ndarray] or List[Dict[str, np.ndarray]]
    if img_segs and isinstance(img_segs[0], dict):
        # get_each_seg mode: save each object's segmentation separately
        img_seg_group = group.create_group("img_segs")
        # Get all unique object names across all frames
        all_obj_names = set()
        for frame_seg in img_segs:
            all_obj_names.update(frame_seg.keys())
        
        # Create a dataset for each object
        for obj_name in sorted(all_obj_names):
            obj_segs = []
            for frame_seg in img_segs:
                if obj_name in frame_seg:
                    obj_segs.append(frame_seg[obj_name])
                else:
                    # If object not present in this frame, use zeros
                    obj_segs.append(np.zeros_like(img_segs[0][list(img_segs[0].keys())[0]]))
            img_seg_group.create_dataset(obj_name, data=np.stack(obj_segs), compression="gzip")
    else:
        # Normal mode: single segmentation per frame
        group.create_dataset("img_segs", data=np.stack(img_segs), compression="gzip")
    
    group.create_dataset("states", data=np.stack(states), compression="gzip")
    group.create_dataset("actions", data=np.stack(actions), compression="gzip")


def write_debug_video(video_path: pathlib.Path, frames: List[np.ndarray], fps: int = 10) -> None:
    """Write debug video with concatenated frames."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def merge_replay_files(
    video_out_path: pathlib.Path,
    worker_hdf5_files: set,
    num_workers: int,
) -> None:
    """Merge worker HDF5 files and jsonl files into final files.
    
    Args:
        video_out_path: Output directory path
        worker_hdf5_files: Set of worker HDF5 file paths
        num_workers: Number of parallel workers
    """
    final_hdf5_path = video_out_path / "replay_success.hdf5"
    final_jsonl_path = video_out_path / "successful_actions.jsonl"
    
    # Merge HDF5 files
    if worker_hdf5_files:
        logging.info(f"Merging HDF5 files: {worker_hdf5_files}")
        with h5py.File(final_hdf5_path, "w") as final_h5:
            for worker_hdf5_file in sorted(worker_hdf5_files):
                worker_path = pathlib.Path(worker_hdf5_file)
                if not worker_path.exists():
                    logging.warning(f"Worker HDF5 file not found: {worker_path}")
                    continue
                
                logging.info(f"Reading from worker file: {worker_path}")
                with h5py.File(worker_path, "r") as worker_h5:
                    # Copy all groups from worker file
                    for group_name in worker_h5.keys():
                        # Handle duplicate group names by checking if it exists in final file
                        final_group_name = group_name
                        if final_group_name in final_h5:
                            # Group name already exists, create unique name
                            suffix = 1
                            while f"{group_name}_merged{suffix}" in final_h5:
                                suffix += 1
                            final_group_name = f"{group_name}_merged{suffix}"
                        
                        # Copy group
                        worker_h5.copy(worker_h5[group_name], final_h5, name=final_group_name)
                        logging.debug(f"Copied group: {group_name} -> {final_group_name}")
        
        logging.info(f"Merged HDF5 file saved to: {final_hdf5_path}")
        
        # Clean up worker HDF5 files
        for worker_hdf5_file in worker_hdf5_files:
            worker_path = pathlib.Path(worker_hdf5_file)
            if worker_path.exists():
                try:
                    worker_path.unlink()
                    logging.debug(f"Removed worker HDF5 file: {worker_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove worker HDF5 file {worker_path}: {e}")
    
    # Merge jsonl files
    # Find all worker jsonl files by pattern matching (worker IDs may not be consecutive)
    worker_jsonl_files = []
    for jsonl_file in video_out_path.glob("successful_actions_worker*.jsonl"):
        worker_jsonl_files.append(jsonl_file)
    
    if worker_jsonl_files:
        logging.info(f"Merging {len(worker_jsonl_files)} jsonl files...")
        with open(final_jsonl_path, 'w', encoding='utf-8') as final_jsonl:
            for worker_jsonl_path in sorted(worker_jsonl_files):
                logging.info(f"Reading from worker jsonl: {worker_jsonl_path}")
                with open(worker_jsonl_path, 'r', encoding='utf-8') as worker_jsonl:
                    for line in worker_jsonl:
                        if line.strip():  # Skip empty lines
                            final_jsonl.write(line)
        
        logging.info(f"Merged jsonl file saved to: {final_jsonl_path}")
        
        # Clean up worker jsonl files
        for worker_jsonl_path in worker_jsonl_files:
            try:
                worker_jsonl_path.unlink()
                logging.debug(f"Removed worker jsonl file: {worker_jsonl_path}")
            except Exception as e:
                logging.warning(f"Failed to remove worker jsonl file {worker_jsonl_path}: {e}")

