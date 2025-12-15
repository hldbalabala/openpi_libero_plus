#!/usr/bin/env python3
"""
Organize failed task videos by category.

This script:
1. Loads task classification from JSON file
2. Scans for failed task videos (ending with _failure.mp4)
3. Extracts task name from video filename
4. Matches task name to category
5. Creates 7 category folders and copies corresponding failed videos
"""

import json
import pathlib
import shutil
import logging
from collections import defaultdict
from typing import Dict, Optional
import tyro
import dataclasses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclasses.dataclass
class Args:
    """Arguments for organizing failed tasks."""
    video_dir: str = "data/libero_goal/20251115_120833" # Directory containing the video files
    task_suite_name: str = "libero_goal"  # Task suite name
    output_base_dir: str = "data/libero_goal_failed_tasks_by_category"  # Base directory for output folders
    classification_path: str = "third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"


def load_task_classification(classification_path: str, task_suite_name: str) -> Dict[str, str]:
    """Load task classification mapping: task_name -> category.
    
    Returns a dictionary mapping task names (normalized) to categories.
    Also returns a list of all task names for partial matching.
    """
    with open(classification_path, 'r', encoding='utf-8') as f:
        classification_data = json.load(f)
    
    # Build mapping from task name to category
    # Store both original name (with underscores) and normalized name (with spaces)
    task_name_to_category = {}
    task_names_list = []  # Store all task names for partial matching
    
    if task_suite_name in classification_data:
        for task_info in classification_data[task_suite_name]:
            task_name = task_info['name']
            category = task_info['category']
            # Store original name
            task_name_to_category[task_name] = category
            # Store normalized name (spaces instead of underscores)
            normalized_name = task_name.replace('_', ' ')
            task_name_to_category[normalized_name] = category
            # Store for partial matching
            task_names_list.append((task_name, normalized_name, category))
    
    return task_name_to_category, task_names_list


def extract_task_name_from_filename(filename: str) -> Optional[str]:
    """Extract task name from video filename.
    
    Format: rollout_{task_description}_failure.mp4
    Or: rollout_{task_description}_success.mp4
    
    Returns: task_description (with underscores, not converted to spaces)
    """
    if not filename.startswith("rollout_"):
        return None
    
    # Remove .mp4 extension
    stem = pathlib.Path(filename).stem
    
    # Remove "rollout_" prefix
    task_segment = stem[8:]
    
    # Remove "_failure" or "_success" suffix
    if task_segment.endswith("_failure"):
        task_segment = task_segment[:-8]
    elif task_segment.endswith("_success"):
        task_segment = task_segment[:-8]
    else:
        return None
    
    # Keep underscores, don't convert to spaces
    task_name = task_segment
    
    return task_name


def match_task_to_category(task_name: str, task_name_to_category: Dict[str, str]) -> Optional[str]:
    """Match task name to category using exact match only.
    
    Task name is in underscore format, matches directly.
    """
    # Direct match (task_name already has underscores)
    if task_name in task_name_to_category:
        return task_name_to_category[task_name]
    
    return None


def organize_failed_tasks(args: Args):
    """Main function to organize failed tasks by category."""
    video_dir = pathlib.Path(args.video_dir)
    if not video_dir.exists():
        raise ValueError(f"Video directory {video_dir} does not exist")
    
    # Load task classification
    logging.info(f"Loading task classification for {args.task_suite_name}...")
    task_name_to_category, _ = load_task_classification(args.classification_path, args.task_suite_name)
    logging.info(f"Loaded {len(task_name_to_category)} task name mappings")
    
    # Find all failed video files
    logging.info(f"Scanning for failed videos in {video_dir}...")
    failed_videos = list(video_dir.glob("*_failure.mp4"))
    logging.info(f"Found {len(failed_videos)} failed videos")
    
    # Group videos by category
    category_to_videos = defaultdict(list)
    unmatched_videos = []
    
    for video_file in failed_videos:
        # Extract task name from filename
        task_name = extract_task_name_from_filename(video_file.name)
        if not task_name:
            logging.warning(f"Could not extract task name from {video_file.name}")
            unmatched_videos.append(video_file)
            continue
        
        # Match to category
        category = match_task_to_category(task_name, task_name_to_category)
        if not category:
            logging.warning(f"Could not match task '{task_name}' to any category for {video_file.name}, putting in Language Instructions")
            # Put unmatched videos in Language Instructions category
            category_to_videos["Language Instructions"].append((video_file, task_name))
            continue
        
        category_to_videos[category].append((video_file, task_name))
    
    # Handle unmatched videos (those that couldn't extract task name)
    if unmatched_videos:
        logging.info(f"Putting {len(unmatched_videos)} videos with unextractable task names into Language Instructions")
        for video_file in unmatched_videos:
            category_to_videos["Language Instructions"].append((video_file, None))
    
    logging.info(f"Total organized videos: {sum(len(videos) for videos in category_to_videos.values())}")
    
    # Create output directory structure
    output_base = pathlib.Path(args.output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create category folders and copy videos
    for category, videos in category_to_videos.items():
        # Create safe folder name (replace spaces with underscores)
        category_folder = category.replace(" ", "_")
        category_dir = output_base / category_folder
        category_dir.mkdir(exist_ok=True)
        
        logging.info(f"Category: {category} - {len(videos)} failed videos")
        
        # Copy videos to category folder
        for video_file, task_name in videos:
            dest_file = category_dir / video_file.name
            shutil.copy2(video_file, dest_file)
            logging.debug(f"Copied {video_file.name} to {category_folder}/")
    
    # Create summary file
    summary_file = output_base / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Failed Tasks by Category Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total failed videos: {len(failed_videos)}\n")
        f.write(f"Total organized videos: {sum(len(videos) for videos in category_to_videos.values())}\n\n")
        f.write("Category Breakdown:\n")
        f.write("-" * 60 + "\n")
        for category in sorted(category_to_videos.keys()):
            count = len(category_to_videos[category])
            f.write(f"{category}: {count} videos\n")
    
    logging.info(f"Summary saved to {summary_file}")
    logging.info(f"Organized videos into {len(category_to_videos)} category folders in {output_base}")


if __name__ == "__main__":
    tyro.cli(organize_failed_tasks)
