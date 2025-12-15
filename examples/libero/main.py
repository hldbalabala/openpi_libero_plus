"""Main entry point for LIBERO evaluation."""
# Set MuJoCo rendering backend to EGL for headless environments
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ["PYTHONPATH"] = "/home/ubuntu/Desktop/hld/openpi/third_party/LIBERO-plus"
root_dir = (project_root / "third_party/LIBERO-plus").resolve()
sys.path.insert(0, str(root_dir))

import collections
import dataclasses
import datetime
import json
import logging
import multiprocessing as mp
import pathlib
from typing import Dict, List

import numpy as np
import tqdm
import tyro
from libero.libero import benchmark

from examples.libero.task_runner import run_single_task
from examples.libero.utils import (
    find_latest_output_folder,
    load_progress_from_statistics,
    merge_replay_files,
    write_category_stats,
)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = f"data/{task_suite_name}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"  # Path to save videos
    resume: bool = False  # If True, automatically find and resume from the latest folder for this task suite
    replay_mode: bool = True  # If True, save successful episodes to HDF5
    get_each_seg: bool = False  # If True, save segmentation for each object separately (robot + objects in task description)
    max_retries: int = 20  # Maximum number of retries for failed tasks in replay_mode (0 = no retries)
    seed: int = 7  # Random Seed (for reproducibility)
    parallel_workers: int = 8  # Number of parallel workers for evaluation (use >1 for multiprocessing)


def eval_libero(args: Args) -> None:
    if args.parallel_workers < 1:
        raise ValueError("parallel_workers must be >= 1")
    if args.parallel_workers > 1:
        mp.set_start_method("spawn", force=True)

    skip_categories: List[str] = [
        # "Background Textures",
        # "Camera Viewpoints",
        # "Language Instructions",
        # "Light Conditions",
        # "Objects add",
        # "Objects move",
        # "Robot Initial States",
        # "Sensor Noise",
    ]

    np.random.seed(args.seed)

    if args.resume:
        try:
            latest_folder = find_latest_output_folder(args.task_suite_name)
            args.video_out_path = str(latest_folder)
            logging.info(f"Resume enabled: using latest folder {args.video_out_path}")
        except (ValueError, Exception) as e:
            logging.error(f"Failed to find latest folder for resume: {e}")
            logging.info("Continuing with specified video_out_path instead")
            args.resume = False

    benchmark_dict = benchmark.get_benchmark_dict()
    print("benchmark_dict.keys():", benchmark_dict.keys())
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    classification_path = "third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
    with open(classification_path, 'r') as f:
        classification_data = json.load(f)

    task_to_category = {}
    if args.task_suite_name in classification_data:
        for task_info in classification_data[args.task_suite_name]:
            category = task_info['category']
            if category == "Objects Layout":
                task_name = task_info.get('name', '').lower()
                if "add" in task_name:
                    category = "Objects add"
                elif "move" in task_name:
                    category = "Objects move"
            task_to_category[task_info['id'] - 1] = category

    if skip_categories:
        logging.info(f"Skipping categories: {skip_categories}")

    video_out_path = pathlib.Path(args.video_out_path)
    completed_tasks_count, total_episodes, total_successes, category_stats = load_progress_from_statistics(
        video_out_path, args.num_trials_per_task
    )

    if completed_tasks_count > 0:
        logging.info(
            f"Resuming from existing progress: {completed_tasks_count} tasks completed, "
            f"{total_episodes} episodes, {total_successes} successes"
        )

    stats_file_path = pathlib.Path(args.video_out_path) / "category_statistics.txt"

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 400
    elif args.task_suite_name == "libero_10" or args.task_suite_name == "libero_mix":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    skipped_tasks = 0
    pending_tasks = collections.deque()
    for task_id in range(num_tasks_in_suite):
        if task_id < completed_tasks_count:
            logging.info(f"Skipping task {task_id} (already completed, {completed_tasks_count} tasks done)")
            skipped_tasks += 1
            continue
        if task_id in task_to_category and skip_categories:
            category = task_to_category[task_id]
            if category in skip_categories:
                # logging.info(f"Skipping task {task_id} (category: {category})")
                skipped_tasks += 1
                continue
        pending_tasks.append(task_id)

    progress_bar = tqdm.tqdm(total=len(pending_tasks), desc="Evaluating tasks")
    with open("/home/ubuntu/Desktop/hld/openpi/libero_goal_language_mapping.json", "r") as f:
        language_mapping = json.load(f)

    def _handle_result(res: dict):
        nonlocal total_episodes, total_successes
        total_episodes += res["total_episodes"]
        total_successes += res["total_successes"]
        for category, updates in res["category_updates"].items():
            category_stats[category]['total'] += updates['total']
            category_stats[category]['successes'] += updates['successes']
        write_category_stats(stats_file_path, total_successes, total_episodes, category_stats)
        logging.info(
            f"Task {res['task_id']} success rate: {float(res['task_successes']) / float(res['task_episodes']) if res['task_episodes'] else 0.0}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes) if total_episodes else 0.0}"
        )

    worker_hdf5_files = set()  # Track worker HDF5 files for merging
    task_retry_count = {}  # Track retry count for each task
    
    if args.parallel_workers > 1:
        ctx = mp.get_context("spawn")
        while pending_tasks:
            batch = list(pending_tasks)
            pending_tasks.clear()
            with ctx.Pool(processes=args.parallel_workers) as pool:
                async_results = [
                    pool.apply_async(
                        run_single_task,
                        (task_id, args, task_to_category, language_mapping, max_steps),
                    )
                    for task_id in batch
                ]
                for ar in async_results:
                    res = ar.get()
                    task_id = res["task_id"]
                    _handle_result(res)
                    # Track worker HDF5 files for merging
                    if res.get("worker_hdf5_path"):
                        worker_hdf5_files.add(res["worker_hdf5_path"])
                    
                    # Handle retries for failed tasks in replay_mode
                    if res["task_successes"] == 0 and args.replay_mode and args.max_retries > 0:
                        retry_count = task_retry_count.get(task_id, 0)
                        if retry_count < args.max_retries:
                            task_retry_count[task_id] = retry_count + 1
                            pending_tasks.append(task_id)
                            logging.info(f"Task {task_id} failed (attempt {retry_count + 1}/{args.max_retries + 1}). Re-queueing for retry.")
                        else:
                            logging.info(f"Task {task_id} failed after {args.max_retries + 1} attempts. Giving up.")
                            progress_bar.update(1)
                    else:
                        progress_bar.update(1)
    else:
        while pending_tasks:
            task_id = pending_tasks.popleft()
            res = run_single_task(task_id, args, task_to_category, language_mapping, max_steps)
            _handle_result(res)
            
            # Handle retries for failed tasks in replay_mode
            if res["task_successes"] == 0 and args.replay_mode and args.max_retries > 0:
                retry_count = task_retry_count.get(task_id, 0)
                if retry_count < args.max_retries:
                    task_retry_count[task_id] = retry_count + 1
                    pending_tasks.append(task_id)
                    logging.info(f"Task {task_id} failed (attempt {retry_count + 1}/{args.max_retries + 1}). Re-queueing for retry.")
                else:
                    logging.info(f"Task {task_id} failed after {args.max_retries + 1} attempts. Giving up.")
                    progress_bar.update(1)
            else:
                progress_bar.update(1)

    progress_bar.close()
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes) if total_episodes else 0.0}")
    logging.info(f"Total episodes: {total_episodes}")
    if skipped_tasks > 0:
        logging.info(f"Skipped tasks: {skipped_tasks}")

    write_category_stats(stats_file_path, total_successes, total_episodes, category_stats)
    logging.info(f"Category statistics saved to {stats_file_path}")
    
    # Merge worker HDF5 and jsonl files if using parallel workers with replay_mode
    if args.replay_mode and args.parallel_workers > 1 and worker_hdf5_files:
        logging.info(f"Merging {len(worker_hdf5_files)} worker HDF5 files...")
        merge_replay_files(video_out_path, worker_hdf5_files, args.parallel_workers)
        logging.info("Worker files merged successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
