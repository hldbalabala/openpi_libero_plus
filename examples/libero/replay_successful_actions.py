"""
根据successful_actions.jsonl中的记录，在LIBERO环境中复现动作，
若episode成功则将图像/手腕图像/状态/动作/任务信息写入HDF5，
并生成agentview+wristview横向拼接的debug视频。
"""

import os
import sys
import json
import logging
import dataclasses
import pathlib
from typing import List, Tuple

import h5py
import imageio
import numpy as np
import tyro

from pathlib import Path

# ---------------------------------------------------------------------------
# 确保LIBERO依赖的路径和渲染后端设定，与examples/libero/main.py保持一致
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTHONPATH", "/home/ubuntu/Desktop/hld/openpi/third_party/LIBERO-plus")
root_dir = (Path(__file__).parent.parent.parent / "third_party" / "LIBERO-plus").resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))



from libero.libero import benchmark
from openpi_client import image_tools

from main import (  # pylint: disable=wrong-import-position
    LIBERO_DUMMY_ACTION,
    _create_safe_filename,
    _get_libero_env,
    _quat2axisangle,
)


@dataclasses.dataclass
class ReplayArgs:
    jsonl_path: str = "/home/ubuntu/Desktop/hld/openpi/data/libero_goal/20251126_171218/successful_actions.jsonl"
    hdf5_path: str = "/home/ubuntu/Desktop/hld/openpi/data/libero_goal/20251126_171218/replay_success.hdf5"
    video_dir: str = "/home/ubuntu/Desktop/hld/openpi/data/libero_goal/20251126_171218/replay_videos"
    task_suite_name: str = "libero_goal"
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: int = 400
    overwrite_hdf5: bool = True
    seed: int = 7
    num_retries: int = 5


def _determine_max_steps(task_suite_name: str) -> int:
    mapping = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 400,
        "libero_10": 520,
        "libero_mix": 520,
        "libero_90": 400,
    }
    try:
        return mapping[task_suite_name]
    except KeyError as exc:
        raise ValueError(f"未知task suite: {task_suite_name}") from exc


def _load_jsonl(path: pathlib.Path) -> List[dict]:
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _prepare_hdf5(path: pathlib.Path, overwrite: bool) -> h5py.File:
    mode = "w" if overwrite else "a"
    return h5py.File(path, mode)


def _episode_group_name(root: h5py.File, task_id: int, episode_idx: int) -> str:
    base = f"task{task_id}_episode{episode_idx}"
    if base not in root:
        return base
    suffix = 1
    while f"{base}_dup{suffix}" in root:
        suffix += 1
    return f"{base}_dup{suffix}"


def _stack_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(np.array(obs["robot0_eef_quat"]).copy()),
            obs["robot0_gripper_qpos"],
        )
    )


def _process_images(obs: dict, env, resize_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    agentview = obs["agentview_image"][::-1, ::-1]
    wristview = obs["robot0_eye_in_hand_image"][::-1, ::-1]

    if hasattr(env, "get_segmentation_of_interest"):
        agentview_mask = env.get_segmentation_of_interest(obs["agentview_segmentation_instance"])
        agentview_mask = agentview_mask[::-1, ::-1]
        agentview_seg = (agentview * agentview_mask).astype(agentview.dtype)
        wristview_mask = env.get_segmentation_of_interest(obs["robot0_eye_in_hand_segmentation_instance"])
        wristview_mask = wristview_mask[::-1, ::-1]
        wristview_seg = (wristview * wristview_mask).astype(wristview.dtype)

    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agentview, resize_size, resize_size)
    )
    img_seg = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agentview_seg, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wristview, resize_size, resize_size)
    )
    wrist_img_seg = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wristview_seg, resize_size, resize_size)
    )
    combined = np.concatenate((img_seg, img, wrist_img_seg, wrist_img), axis=1)
    return img, wrist_img,img_seg, combined


def _save_episode(
    h5_file: h5py.File,
    group_name: str,
    images: List[np.ndarray],
    wrist_images: List[np.ndarray],
    img_segs: List[np.ndarray],
    states: List[np.ndarray],
    actions: List[np.ndarray],
    task_description: str,
    task_id: int,
    episode_idx: int,
):
    group = h5_file.create_group(group_name)
    group.attrs["task_id"] = task_id
    group.attrs["task_description"] = task_description
    group.attrs["episode_idx"] = episode_idx
    group.create_dataset("images", data=np.stack(images), compression="gzip")
    group.create_dataset("wrist_images", data=np.stack(wrist_images), compression="gzip")
    group.create_dataset("img_segs", data=np.stack(img_segs), compression="gzip")
    group.create_dataset("states", data=np.stack(states), compression="gzip")
    group.create_dataset("actions", data=np.stack(actions), compression="gzip")


def _write_debug_video(video_path: pathlib.Path, frames: List[np.ndarray], fps: int = 10) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def replay_success(args: ReplayArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    jsonl_path = pathlib.Path(args.jsonl_path)
    hdf5_path = pathlib.Path(args.hdf5_path)
    video_dir = pathlib.Path(args.video_dir)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到jsonl文件: {jsonl_path}")

    if args.max_steps <= 0:
        args.max_steps = _determine_max_steps(args.task_suite_name)

    entries = _load_jsonl(jsonl_path)
    logging.info("加载成功轨迹 %d 条", len(entries))

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    success_count = 0
    with _prepare_hdf5(hdf5_path, args.overwrite_hdf5) as h5_file:
        for entry in entries:
            task_id = entry["task_id"]
            episode_idx = entry.get("episode_idx", 0)
            actions = entry["actions"]
            task = task_suite.get_task(task_id)
            max_attempts = max(1, args.num_retries + 1)
            task_description = ""
            success = False
            for attempt_idx in range(1, max_attempts + 1):
                env = None
                try:
                    env, task_description = _get_libero_env(task, args.resize_size, seed=args.seed)
                    logging.info(
                        "重放task_id=%s episode=%s (%s) 尝试 %d/%d",
                        task_id,
                        episode_idx,
                        task_description,
                        attempt_idx,
                        max_attempts,
                    )

                    obs = env.reset()
                    for _ in range(args.num_steps_wait):
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                    images: List[np.ndarray] = []
                    wrist_images: List[np.ndarray] = []
                    img_segs: List[np.ndarray] = []
                    states: List[np.ndarray] = []
                    executed_actions: List[np.ndarray] = []
                    debug_frames: List[np.ndarray] = []
                    done = False

                    for step_idx, action in enumerate(actions):
                        obs, reward, done, info = env.step(action)
                        img, wrist_img,img_seg, combined = _process_images(obs, env, args.resize_size)
                        state = _stack_state(obs)

                        images.append(img)
                        wrist_images.append(wrist_img)
                        img_segs.append(img_seg)
                        states.append(state.astype(np.float32))
                        executed_actions.append(np.asarray(action, dtype=np.float32))
                        debug_frames.append(combined)

                        if done or step_idx >= args.max_steps:
                            break

                    if done:
                        success_count += 1
                        success = True
                        group_name = _episode_group_name(h5_file, task_id, episode_idx)
                        _save_episode(
                            h5_file,
                            group_name,
                            images,
                            wrist_images,
                            img_segs,
                            states,
                            executed_actions,
                            task_description,
                            task_id,
                            episode_idx,
                        )
                        safe_name = _create_safe_filename(task_description, task_id, episode_idx, "debug")
                        video_path = video_dir / safe_name.replace(".mp4", "_concat.mp4")
                        _write_debug_video(video_path, debug_frames)
                        logging.info("成功: 数据写入%s, 视频写入%s", group_name, video_path)
                        break

                    logging.warning(
                        "重放失败: task_id=%s episode=%s 尝试 %d/%d",
                        task_id,
                        episode_idx,
                        attempt_idx,
                        max_attempts,
                    )
                finally:
                    try:
                        if env is not None:
                            if hasattr(env, "close"):
                                env.close()
                            elif hasattr(env, "env") and hasattr(env.env, "close"):
                                env.env.close()
                    except Exception as exc:  # pragma: no cover
                        logging.warning("关闭env失败: %s", exc)
            if not success:
                logging.error(
                    "重放失败: task_id=%s episode=%s 已达最大尝试次数 %d",
                    task_id,
                    episode_idx,
                    max_attempts,
                )

    logging.info("完成重放，总成功 %d/%d", success_count, len(entries))


if __name__ == "__main__":
    args = tyro.cli(ReplayArgs)
    replay_success(args)

