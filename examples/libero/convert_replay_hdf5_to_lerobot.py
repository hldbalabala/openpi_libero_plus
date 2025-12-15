"""
将 `examples/libero/replay_successful_actions.py` 生成的 HDF5 文件转换为 LeRobot 数据集。

示例：
uv run examples/libero/convert_replay_hdf5_to_lerobot.py \
    --hdf5-path /home/ubuntu/Desktop/hld/openpi/data/libero_goal/20251119_134716/replay_success.hdf5 \
    --repo-name libero_replay_pi
"""

import dataclasses
import shutil
from pathlib import Path
from typing import Dict, Iterable

import h5py
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


@dataclasses.dataclass
class ConvertReplayArgs:
    hdf5_path: str = "/home/ubuntu/Desktop/hld/openpi/data/libero_goal/20251128_170352/replay_success.hdf5"
    repo_name: str = "libero_replay_obj"
    robot_type: str = "panda"
    fps: int = 10
    overwrite_repo: bool = True
    push_to_hub: bool = False
    hub_private: bool = False
    image_size: int = 224


def _lerobot_features(image_size: int) -> Dict[str, Dict]:
    return {
        "image": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "img_seg": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
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


def _iter_episode_groups(h5_file: h5py.File) -> Iterable[h5py.Group]:
    for name in sorted(h5_file.keys()):
        group = h5_file[name]
        if not isinstance(group, h5py.Group):
            continue
        required = ["images", "wrist_images", "img_segs", "states", "actions"]
        if any(key not in group for key in required):
            continue
        yield group


def convert_replay_hdf5(args: ConvertReplayArgs) -> None:
    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5文件不存在: {hdf5_path}")

    output_path = HF_LEROBOT_HOME / args.repo_name
    if output_path.exists() and args.overwrite_repo:
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        robot_type=args.robot_type,
        fps=args.fps,
        features=_lerobot_features(args.image_size),
        image_writer_threads=8,
        image_writer_processes=4,
    )

    with h5py.File(hdf5_path, "r") as h5_file:
        for group in _iter_episode_groups(h5_file):
            task_desc = group.attrs.get("task_description", "")
            images = group["images"]
            wrist_images = group["wrist_images"]
            img_segs = group["img_segs"]
            states = group["states"]
            actions = group["actions"]

            for idx in range(len(images)):
                dataset.add_frame(
                    {
                        "image": images[idx],
                        "wrist_image": wrist_images[idx],
                        "img_seg": img_segs[idx],
                        "state": states[idx],
                        "actions": actions[idx],
                        "task": task_desc,
                    }
                )
            dataset.save_episode()

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "replay"],
            private=args.hub_private,
            push_videos=False,
            license="apache-2.0",
        )


if __name__ == "__main__":
    convert_replay_hdf5(tyro.cli(ConvertReplayArgs))

