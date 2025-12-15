"""Task runner for evaluating a single LIBERO task."""
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
import gc
import h5py
import json
import logging
import multiprocessing as mp
import pathlib
from typing import Dict, List

import imageio
import numpy as np
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from examples.libero.utils import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    create_safe_filename,
    episode_group_name,
    get_libero_env,
    quat2axisangle,
    save_replay_episode,
    stack_state,
    write_debug_video,
)


def run_single_task(
    task_id: int,
    args,
    task_to_category: Dict[int, str],
    language_mapping: Dict[str, str],
    max_steps: int,
) -> dict:
    """Run evaluation for a single task. Returns per-task statistics."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    task_description = task.language
    task_description_new = language_mapping.get(task_description, task_description)
    with open("tools/libero_goal_action_mapping.json", "r") as f:
        action_mapping = json.load(f)
    task_description_new = action_mapping.get(task_description_new, task_description_new)

    env, _ = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    # Data accumulators
    task_episodes, task_successes = 0, 0
    total_episodes = 0
    total_successes = 0
    category_updates = collections.defaultdict(lambda: {'successes': 0, 'total': 0})

    stats_file_path = pathlib.Path(args.video_out_path) / "category_statistics.txt"
    success_actions_path = pathlib.Path(args.video_out_path) / "successful_actions.jsonl"

    # Initialize HDF5 / replay (per process to avoid sharing)
    # Each worker uses its own HDF5 file when parallel_workers > 1
    h5_file = None
    replay_video_dir = None
    worker_hdf5_path = None
    if args.replay_mode:
        replay_video_dir = pathlib.Path(args.video_out_path) / "replay_videos"
        replay_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Use process-specific HDF5 file when parallel_workers > 1
        if args.parallel_workers > 1:
            # Extract process number from process name (e.g., "SpawnPoolWorker-1" -> "1")
            process_name = mp.current_process().name
            worker_id = process_name.split("-")[-1] if "-" in process_name else "0"
            worker_hdf5_path = pathlib.Path(args.video_out_path) / f"replay_success_worker{worker_id}.hdf5"
        else:
            worker_hdf5_path = pathlib.Path(args.video_out_path) / "replay_success.hdf5"
        
        h5_file = h5py.File(worker_hdf5_path, "a")
        logging.info(f"[Worker {mp.current_process().name}] 重放模式已启用: HDF5文件={worker_hdf5_path}, 视频目录={replay_video_dir}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for episode_idx in range(args.num_trials_per_task):
        logging.info(f"[Worker {mp.current_process().name}] Task {task_id}: {task_description}")

        env.reset()
        task_episodes += 1
        total_episodes += 1

        t = 0
        replay_images = []
        done = False
        img = None
        wrist_img = None
        action_plan = collections.deque()
        executed_actions = []

        images: List[np.ndarray] = []
        wrist_images: List[np.ndarray] = []
        img_segs: List[np.ndarray] = []  # Will be List[Dict[str, np.ndarray]] in get_each_seg mode
        states: List[np.ndarray] = []
        debug_frames: List[np.ndarray] = []

        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    if t % 3 == 0:
                        del obs
                        obs = None
                    t += 1
                    continue

                if obs is None:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

                agentview = obs["agentview_image"][::-1, ::-1]
                wristview = obs["robot0_eye_in_hand_image"][::-1, ::-1]

                if args.replay_mode and hasattr(env, "get_segmentation_of_interest"):
                    agentview_mask = env.get_segmentation_of_interest(obs["agentview_segmentation_instance"])
                    agentview_mask = agentview_mask[::-1, ::-1]
                    agentview_seg = (agentview * agentview_mask).astype(agentview.dtype)
                    wristview_mask = env.get_segmentation_of_interest(obs["robot0_eye_in_hand_segmentation_instance"])
                    wristview_mask = wristview_mask[::-1, ::-1]
                    wristview_seg = (wristview * wristview_mask).astype(wristview.dtype)
                else:
                    agentview_seg = agentview.copy()
                    wristview_seg = wristview.copy()

                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(agentview, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wristview, args.resize_size, args.resize_size)
                )
                del agentview, wristview

                replay_images.append(img.copy())

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description_new),
                    }

                    action_chunk = client.infer(element)["actions"]
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])
                    del element, action_chunk

                action = action_plan.popleft()
                action_list = action.tolist()
                executed_actions.append(action_list)

                obs, reward, done, info = env.step(action_list)

                if args.replay_mode:
                    input_obj_list = ["robot", "cabinet", "bowl", "plate", "stove", "cream cheese", "rack", "wine bottle"]
                    agentview_post = obs["agentview_image"][::-1, ::-1]
                    wristview_post = obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    
                    if args.get_each_seg:
                        # In get_each_seg mode, save segmentation for each object separately
                        img_seg_dict = {}
                        obj_seg_raw = {}  # Store raw segmentation for debug video
                        
                        # Always save robot segmentation
                        robot_agentview_seg = env.get_segmentation_input_obj(obs["agentview_segmentation_instance"], "robot")
                        robot_agentview_seg = robot_agentview_seg[::-1, ::-1]
                        robot_seg = (agentview_post * robot_agentview_seg).astype(agentview_post.dtype)
                        robot_seg_processed = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(robot_seg, args.resize_size, args.resize_size)
                        )
                        img_seg_dict["robot"] = robot_seg_processed
                        obj_seg_raw["robot"] = robot_seg
                        
                        # Save segmentation for objects that appear in task description
                        for obj in input_obj_list:
                            if obj != "robot" and obj in task_description_new:
                                obj_agentview_seg = env.get_segmentation_input_obj(obs["agentview_segmentation_instance"], obj)
                                obj_agentview_seg = obj_agentview_seg[::-1, ::-1]
                                obj_seg = (agentview_post * obj_agentview_seg).astype(agentview_post.dtype)
                                obj_seg_processed = image_tools.convert_to_uint8(
                                    image_tools.resize_with_pad(obj_seg, args.resize_size, args.resize_size)
                                )
                                img_seg_dict[obj] = obj_seg_processed
                                obj_seg_raw[obj] = obj_seg
                        
                        img_segs.append(img_seg_dict)
                        
                        # For debug video, use the last object in input_obj_list that appears in task_description_new
                        debug_obj_seg = None
                        for obj in reversed(input_obj_list):
                            if obj != "robot" and obj in task_description_new and obj in obj_seg_raw:
                                debug_obj_seg = obj_seg_raw[obj]
                                break
                        
                        # If no matching object found, fall back to robot segmentation
                        if debug_obj_seg is None:
                            agentview_seg = robot_seg
                        else:
                            agentview_seg = debug_obj_seg
                        
                        wristview_seg = wristview_post.copy()  # Use original wrist view for debug
                    else:
                        # Original behavior: use segmentation of interest
                        if hasattr(env, "get_segmentation_of_interest"):
                            agentview_mask = env.get_segmentation_of_interest(obs["agentview_segmentation_instance"])
                            agentview_mask = agentview_mask[::-1, ::-1]
                            agentview_seg = (agentview_post * agentview_mask).astype(agentview_post.dtype)
                            wristview_mask = env.get_segmentation_of_interest(obs["robot0_eye_in_hand_segmentation_instance"])
                            wristview_mask = wristview_mask[::-1, ::-1]
                            wristview_seg = (wristview_post * wristview_mask).astype(wristview_post.dtype)
                        else:
                            agentview_seg = agentview_post.copy()
                            wristview_seg = wristview_post.copy()
                        
                        img_seg = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(agentview_seg, args.resize_size, args.resize_size)
                        )
                        img_segs.append(img_seg)

                    img_post = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(agentview_post, args.resize_size, args.resize_size)
                    )
                    wrist_img_post = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wristview_post, args.resize_size, args.resize_size)
                    )
                    
                    if not args.get_each_seg:
                        wrist_img_seg = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wristview_seg, args.resize_size, args.resize_size)
                        )
                        combined = np.concatenate((img_seg, img_post, wrist_img_seg, wrist_img_post), axis=1)
                    else:
                        # For debug video in get_each_seg mode, use the selected object segmentation
                        debug_seg_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(agentview_seg, args.resize_size, args.resize_size)
                        )
                        combined = np.concatenate((debug_seg_img, img_post, wrist_img_post, wrist_img_post), axis=1)
                    
                    state = stack_state(obs)

                    images.append(img_post.copy())
                    wrist_images.append(wrist_img_post.copy())
                    states.append(state.astype(np.float32))
                    debug_frames.append(combined)

                    del agentview_post, wristview_post
                    if not args.get_each_seg:
                        del agentview_seg, wristview_seg

                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            except Exception as e:
                logging.error(f"[Worker {mp.current_process().name}] Caught exception: {e}")
                break

        if task_id in task_to_category:
            category = task_to_category[task_id]
            category_updates[category]['total'] += 1
            if done:
                category_updates[category]['successes'] += 1

        suffix = "success" if done else "failure"
        safe_filename = create_safe_filename(task_description, task_id, episode_idx, suffix)
        video_path = pathlib.Path(args.video_out_path) / "origin_videos" / safe_filename
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(video_path, fps=10) as writer:
            for img_frame in replay_images:
                if isinstance(img_frame, np.ndarray):
                    writer.append_data(img_frame)
                else:
                    writer.append_data(np.asarray(img_frame))

        del replay_images
        replay_images = None

        if done and args.replay_mode and h5_file is not None:
            group_name = episode_group_name(h5_file, task_id, episode_idx)
            save_replay_episode(
                h5_file,
                group_name,
                images,
                wrist_images,
                img_segs,
                states,
                [np.asarray(action, dtype=np.float32) for action in executed_actions],
                task_description_new,
                task_id,
                episode_idx,
            )

            safe_name = create_safe_filename(task_description, task_id, episode_idx, "debug")
            video_path = replay_video_dir / safe_name.replace(".mp4", "_concat.mp4")
            write_debug_video(video_path, debug_frames)
            logging.info("[Worker %s] 成功episode数据已保存: HDF5组=%s, 视频=%s", mp.current_process().name, group_name, video_path)

            log_entry = {
                "task_id": task_id,
                "episode_idx": episode_idx,
                "task_description": task_description,
                "actions": executed_actions,
            }
            # Use process-specific jsonl file when parallel_workers > 1
            if args.parallel_workers > 1:
                process_name = mp.current_process().name
                worker_id = process_name.split("-")[-1] if "-" in process_name else "0"
                worker_actions_path = pathlib.Path(args.video_out_path) / f"successful_actions_worker{worker_id}.jsonl"
            else:
                worker_actions_path = success_actions_path
            with open(worker_actions_path, 'a', encoding='utf-8') as log_file:
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        elif args.replay_mode:
            logging.debug("[Worker %s] Episode失败，丢弃收集的数据: task_id=%s episode=%s", mp.current_process().name, task_id, episode_idx)

        try:
            del obs, img, wrist_img, action_plan
        except NameError:
            pass
        action_plan = None
        obs = None
        img = None
        wrist_img = None

        if args.replay_mode:
            images.clear()
            wrist_images.clear()
            img_segs.clear()
            states.clear()
            debug_frames.clear()

        if total_episodes % 10 == 0:
            gc.collect()

    try:
        if hasattr(env, 'close'):
            env.close()
        elif hasattr(env, 'env') and hasattr(env.env, 'close'):
            env.env.close()
    except Exception as e:
        logging.warning(f"[Worker {mp.current_process().name}] Error closing environment: {e}")

    if h5_file is not None:
        h5_file.close()

    # Return only lightweight stats, including worker HDF5 path for merging
    return {
        "task_id": task_id,
        "task_episodes": task_episodes,
        "task_successes": task_successes,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "category_updates": {k: v for k, v in category_updates.items()},
        "worker_hdf5_path": str(worker_hdf5_path) if worker_hdf5_path else None,
    }

