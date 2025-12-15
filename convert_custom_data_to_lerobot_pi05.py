"""
将Aloha hdf5数据转换为LeRobot数据集v2.0格式的脚本。

使用示例: python convert_custom_data_to_lerobot_pi05.py --raw-dir ~/Downloads/hdf5_data --repo-id only_fork
"""

import dataclasses  # 导入数据类装饰器，用于创建不可变的数据结构
from pathlib import Path  # 导入路径处理模块，提供面向对象的文件系统路径操作
import shutil  # 导入高级文件操作模块，用于文件和目录的复制、移动等操作
from typing import Literal  # 导入字面量类型提示，用于限制变量只能取特定的字符串值

import h5py  # 导入HDF5文件处理库，用于读写HDF5格式的数据文件
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # 导入LeRobot数据集的主目录路径
print("HF_LEROBOT_HOME: ", HF_LEROBOT_HOME)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # 导入LeRobot数据集类
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw  # 注释掉的原始数据下载函数
import numpy as np  # 导入NumPy数值计算库
import torch  # 导入PyTorch深度学习框架
import tqdm  # 导入进度条库，用于显示处理进度
import tyro  # 导入命令行参数解析库
import os  # 导入操作系统接口模块


@dataclasses.dataclass(frozen=True)  # 使用数据类装饰器创建不可变的数据配置类
class DatasetConfig:  # 定义数据集配置类
    use_videos: bool = True  # 是否使用视频模式，默认为True
    tolerance_s: float = 0.0001  # 时间容差（秒），用于时间同步，默认为0.0001秒
    image_writer_processes: int = 10  # 图像写入进程数，默认为10个进程
    image_writer_threads: int = 5  # 图像写入线程数，默认为5个线程
    video_backend: str | None = None  # 视频后端，可选参数，默认为None
    verbose: bool = False  # 是否打印详细信息，默认为False


DEFAULT_DATASET_CONFIG = DatasetConfig()  # 创建默认的数据集配置实例


def create_empty_dataset(  # 定义创建空数据集的函数
    repo_id: str,  # 仓库ID，用于标识数据集
    robot_type: str,  # 机器人类型
    mode: Literal["video", "image"] = "video",  # 数据模式，视频或图像，默认为视频
    *,  # 强制关键字参数分隔符
    has_velocity: bool = False,  # 是否包含速度数据，默认为False
    has_effort: bool = False,  # 是否包含力矩数据，默认为False
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,  # 数据集配置，使用默认配置
) -> LeRobotDataset:  # 返回LeRobot数据集对象
    motors = [  # 定义机器人关节名称列表
        "right_waist",  # 右腰关节
        "right_shoulder",  # 右肩关节
        "right_elbow",  # 右肘关节
        "right_forearm_roll",  # 右前臂滚转关节
        "right_wrist_angle",  # 右腕角度关节
        "right_wrist_rotate",  # 右腕旋转关节
        "right_gripper",  # 右夹爪
        "left_waist",  # 左腰关节
        "left_shoulder",  # 左肩关节
        "left_elbow",  # 左肘关节
        "left_forearm_roll",  # 左前臂滚转关节
        "left_wrist_angle",  # 左腕角度关节
        "left_wrist_rotate",  # 左腕旋转关节
        "left_gripper",  # 左夹爪
    ]
    cameras = [  # 定义相机名称列表
        "cam_high",  # 高位相机
        # "cam_low",  # 低位相机（已注释）
        "cam_left_wrist",  # 左腕相机
        "cam_right_wrist",  # 右腕相机
    ]

    features = {  # 定义数据集特征字典
        "observation.state": {  # 观察状态特征
            "dtype": "float32",  # 数据类型为32位浮点数
            "shape": (len(motors),),  # 形状为关节数量的一维数组
            "names": [  # 特征名称列表
                motors,  # 使用关节名称列表
            ],
        },
        "action": {  # 动作特征
            "dtype": "float32",  # 数据类型为32位浮点数
            "shape": (len(motors),),  # 形状为关节数量的一维数组
            "names": [  # 特征名称列表
                motors,  # 使用关节名称列表
            ],
        },
    }

    if has_velocity:  # 如果包含速度数据
        features["observation.velocity"] = {  # 添加速度观察特征
            "dtype": "float32",  # 数据类型为32位浮点数
            "shape": (len(motors),),  # 形状为关节数量的一维数组
            "names": [  # 特征名称列表
                motors,  # 使用关节名称列表
            ],
        }

    if has_effort:  # 如果包含力矩数据
        features["observation.effort"] = {  # 添加力矩观察特征
            "dtype": "float32",  # 数据类型为32位浮点数
            "shape": (len(motors),),  # 形状为关节数量的一维数组
            "names": [  # 特征名称列表
                motors,  # 使用关节名称列表
            ],
        }

    for cam in cameras:  # 遍历所有相机
        features[f"observation.images.{cam}"] = {  # 为每个相机添加图像观察特征
            "dtype": mode,  # 数据类型为指定的模式（视频或图像）
            "shape": (3, 480, 640),  # 图像形状：3个通道，480像素高，640像素宽
            "names": [  # 特征名称列表
                "channels",  # 通道维度
                "height",  # 高度维度
                "width",  # 宽度维度
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():  # 如果数据集目录已存在
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)  # 删除现有目录

    return LeRobotDataset.create(  # 创建并返回LeRobot数据集
        repo_id=repo_id,  # 仓库ID
        fps=50,  # 帧率设置为50fps
        robot_type=robot_type,  # 机器人类型
        features=features,  # 特征定义
        use_videos=dataset_config.use_videos,  # 是否使用视频模式
        tolerance_s=dataset_config.tolerance_s,  # 时间容差
        image_writer_processes=dataset_config.image_writer_processes,  # 图像写入进程数
        image_writer_threads=dataset_config.image_writer_threads,  # 图像写入线程数
        video_backend=dataset_config.video_backend,  # 视频后端
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:  # 定义获取相机列表的函数
    with h5py.File(hdf5_files[0], "r") as ep:  # 打开第一个HDF5文件进行读取
        # 新格式：从 /camera/color/ 获取相机列表
        if "/camera/color" in ep:  # 如果存在新的相机结构
            return [key for key in ep["/camera/color"].keys() if "depth" not in key]  # 返回不包含"depth"的图像键列表
        else:  # 兼容旧格式
            return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # 返回不包含"depth"的图像键列表


def has_velocity(hdf5_files: list[Path]) -> bool:  # 定义检查是否包含速度数据的函数
    with h5py.File(hdf5_files[0], "r") as ep:  # 打开第一个HDF5文件进行读取
        # 新格式：检查 /arm/jointStateVelocity/ 是否存在
        if "/arm/jointStateVelocity" in ep:  # 如果存在新的速度结构
            return True  # 返回True
        else:  # 兼容旧格式
            return "/observations/qvel" in ep  # 检查是否存在速度观察数据


def has_effort(hdf5_files: list[Path]) -> bool:  # 定义检查是否包含力矩数据的函数
    with h5py.File(hdf5_files[0], "r") as ep:  # 打开第一个HDF5文件进行读取
        # 新格式：检查 /arm/jointStateEffort/ 是否存在
        if "/arm/jointStateEffort" in ep:  # 如果存在新的力矩结构
            return True  # 返回True
        else:  # 兼容旧格式
            return "/observations/effort" in ep  # 检查是否存在力矩观察数据


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str], verbose: bool = False) -> dict[str, np.ndarray]:  # 定义按相机加载原始图像的函数
    imgs_per_cam = {}  # 初始化每个相机的图像字典
    
    # 检查是否为新格式
    is_new_format = "/camera/color" in ep  # 检查是否存在新的相机结构
    if verbose:
        print(f"🔍 检测到格式: {'新格式' if is_new_format else '旧格式'}")
    
    for camera_key in cameras:  # 遍历所有相机
        if verbose:
            print(f"\n📷 处理相机: {camera_key}")
        
        if is_new_format:  # 如果是新格式
            # 新格式：直接使用相机名称映射
            camera_mapping = {
                'cam_high': 'front',  # 高位相机对应front
                'cam_left_wrist': 'left',  # 左腕相机对应left
                'cam_right_wrist': 'right',  # 右腕相机对应right
            }
            camera = camera_mapping.get(camera_key, camera_key)  # 获取映射后的相机名称
            camera_path = f"/camera/color/{camera}"  # 构建新格式的相机路径
        else:  # 兼容旧格式
            if camera_key == 'cam_high':  # 如果是高位相机
                camera = 'camera_front'  # 映射为前置相机
            elif camera_key == 'cam_left_wrist':  # 如果是左腕相机
                camera = 'camera_left'  # 映射为左相机
            elif camera_key == 'cam_right_wrist':  # 如果是右腕相机
                camera = 'camera_right'  # 映射为右相机
            camera_path = f"/observations/rgb_images/{camera}"  # 构建旧格式的相机路径
        
        if verbose:
            print(f"  📂 相机路径: {camera_path}")
        
        if camera_path not in ep:  # 如果相机路径不存在
            if verbose:
                print(f"  ❌ 警告: 相机路径不存在 {camera_path}")
                # 列出可用的相机路径
                if is_new_format and "/camera/color" in ep:
                    available_cameras = list(ep["/camera/color"].keys())
                    print(f"  📋 可用的相机: {available_cameras}")
                elif not is_new_format and "/observations/images" in ep:
                    available_cameras = list(ep["/observations/images"].keys())
                    print(f"  📋 可用的相机: {available_cameras}")
            continue  # 跳过这个相机
        
        # 检查数据形状和类型
        dataset = ep[camera_path]
        if verbose:
            print(f"  📊 数据形状: {dataset.shape}")
            print(f"  🏷️ 数据类型: {dataset.dtype}")
            print(f"  💾 数据大小: {dataset.size * dataset.dtype.itemsize / 1024 / 1024:.2f} MB")
        
        uncompressed = dataset.ndim == 4  # 检查图像是否为未压缩格式（4维）
        if verbose:
            print(f"  🔄 压缩状态: {'未压缩' if uncompressed else '压缩'}")

        if uncompressed:  # 如果是未压缩图像
            if verbose:
                print(f"  ✅ 加载未压缩图像: {camera_path}")
            # load all images in RAM  # 将所有图像加载到内存中
            imgs_array = dataset[:]  # 直接读取所有图像数据
            if verbose:
                print(f"  📈 加载后形状: {imgs_array.shape}")
                print(f"  🎨 图像范围: [{imgs_array.min()}, {imgs_array.max()}]")
        else:  # 如果是压缩图像
            if verbose:
                print(f"  ✅ 加载压缩图像: {camera_path}")
            import cv2  # 导入OpenCV库

            # load one compressed image after the other in RAM and uncompress  # 逐个加载压缩图像到内存并解压缩
            imgs_array = []  # 初始化图像数组列表
            if verbose:
                print(f"  🔄 开始解压缩 {len(dataset)} 张图像...")
            for i, data in enumerate(dataset):  # 遍历压缩图像数据
                if verbose and i % 50 == 0:  # 每50张图像打印一次进度
                    print(f"    处理进度: {i}/{len(dataset)}")
                # imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))  # 注释掉的BGR到RGB转换
                decoded_img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)  # 解码压缩图像数据
                if decoded_img is not None:
                    imgs_array.append(decoded_img)  # 添加到列表
                else:
                    if verbose:
                        print(f"    ⚠️ 警告: 第{i}张图像解码失败")
            imgs_array = np.array(imgs_array)  # 将列表转换为NumPy数组
            if verbose:
                print(f"  📈 解压缩后形状: {imgs_array.shape}")
                print(f"  🎨 图像范围: [{imgs_array.min()}, {imgs_array.max()}]")

        imgs_per_cam[camera_key] = imgs_array  # 将图像数组存储到对应相机键下
        if verbose:
            print(f"  ✅ 成功加载 {len(imgs_array)} 张图像到 {camera_key}")
    
    if verbose:
        print(f"\n📊 图像加载总结:")
        for cam_key, img_array in imgs_per_cam.items():
            print(f"  {cam_key}: {img_array.shape} ({img_array.size * img_array.dtype.itemsize / 1024 / 1024:.2f} MB)")
    
    return imgs_per_cam  # 返回每个相机的图像字典


def calculate_quantiles(hdf5_files: list[Path], verbose: bool = False) -> dict:  # 定义计算分位数的函数
    """计算所有HDF5文件中第7和14维的q01和q99分位数"""
    if verbose:
        print("🔢 开始计算数据分位数...")
    
    all_master_data = []  # 存储所有主控端数据
    all_puppet_data = []  # 存储所有从控端数据
    
    for hdf5_file in hdf5_files:  # 遍历所有HDF5文件
        if verbose:
            print(f"  📁 处理文件: {hdf5_file.name}")
        
        try:
            with h5py.File(hdf5_file, "r") as ep:  # 打开HDF5文件
                # 检查是否为新格式
                is_new_format = "/arm/jointStatePosition" in ep
                
                if is_new_format:  # 如果是新格式
                    # 新格式：从 /arm/jointStatePosition/ 读取关节位置数据
                    master_left = ep["/arm/jointStatePosition/masterLeft"][:]  # 读取左臂主控关节位置数据
                    master_right = ep["/arm/jointStatePosition/masterRight"][:]  # 读取右臂主控关节位置数据
                    puppet_left = ep["/arm/jointStatePosition/puppetLeft"][:]  # 读取左臂关节位置数据
                    puppet_right = ep["/arm/jointStatePosition/puppetRight"][:]  # 读取右臂关节位置数据
                    
                    # 合并左右臂数据 [right_arm, left_arm]
                    master_data = np.concatenate([master_right, master_left], axis=1)  # 合并主控端数据
                    puppet_data = np.concatenate([puppet_right, puppet_left], axis=1)  # 合并从控端数据
                else:  # 兼容旧格式
                    # 旧格式：从 /puppet/arm_joint_position 和 /master/arm_joint_position 读取数据
                    master_data = ep["/master/arm_joint_position"][:]  # 读取主控端关节位置数据
                    puppet_data = ep["/puppet/arm_joint_position"][:]  # 读取从控端关节位置数据
                
                all_master_data.append(master_data)  # 添加到主控端数据列表
                all_puppet_data.append(puppet_data)  # 添加到从控端数据列表
                
                if verbose:
                    print(f"    ✅ 加载数据: master {master_data.shape}, puppet {puppet_data.shape}")
                    
        except Exception as e:
            if verbose:
                print(f"    ❌ 处理文件失败: {e}")
            continue  # 跳过有问题的文件
    
    if not all_master_data or not all_puppet_data:  # 如果没有数据
        if verbose:
            print("  ⚠️ 警告: 没有找到有效数据，使用默认分位数")
        # 返回默认分位数
        return {
            "master": {
                "q01": [-0.00091, -0.00098],
                "q99": [0.05964, 0.04851]
            },
            "puppet": {
                "q01": [-0.0009800000116229057, -0.0008399999933317304],
                "q99": [0.05914999917149544, 0.047529999166727066]
            }
        }
    
    # 合并所有数据
    all_master_data = np.concatenate(all_master_data, axis=0)  # 合并所有主控端数据
    all_puppet_data = np.concatenate(all_puppet_data, axis=0)  # 合并所有从控端数据
    
    if verbose:
        print(f"  📊 数据统计:")
        print(f"    master: {all_master_data.shape}")
        print(f"    puppet: {all_puppet_data.shape}")
    
    # 计算分位数
    quantiles = {
        "master": {
            "q01": [
                np.percentile(all_master_data[:, 6], 1),   # 第7维的1%分位数
                np.percentile(all_master_data[:, 13], 1)   # 第14维的1%分位数
            ],
            "q99": [
                np.percentile(all_master_data[:, 6], 99),  # 第7维的99%分位数
                np.percentile(all_master_data[:, 13], 99)  # 第14维的99%分位数
            ]
        },
        "puppet": {
            "q01": [
                np.percentile(all_puppet_data[:, 6], 1),   # 第7维的1%分位数
                np.percentile(all_puppet_data[:, 13], 1)   # 第14维的1%分位数
            ],
            "q99": [
                np.percentile(all_puppet_data[:, 6], 99),  # 第7维的99%分位数
                np.percentile(all_puppet_data[:, 13], 99)  # 第14维的99%分位数
            ]
        }
    }
    
    if verbose:
        print(f"  📈 计算完成的分位数:")
        print(f"    master q01: {quantiles['master']['q01']}")
        print(f"    master q99: {quantiles['master']['q99']}")
        print(f"    puppet q01: {quantiles['puppet']['q01']}")
        print(f"    puppet q99: {quantiles['puppet']['q99']}")
    
    return quantiles  # 返回计算的分位数


def load_raw_episode_data(  # 定义加载原始剧集数据的函数
    ep_path: Path,  # 剧集文件路径
    quantiles: dict | None = None,  # 分位数字典，用于数据归一化
    verbose: bool = False,  # 是否打印详细信息
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:  # 返回图像字典、状态、动作、速度、力矩的元组
    with h5py.File(ep_path, "r") as ep:  # 打开HDF5文件进行读取
        # 检查是否为新格式
        is_new_format = "/arm/jointStatePosition" in ep  # 检查是否存在新的关节位置结构
        
        if is_new_format:  # 如果是新格式
            # 新格式：从 /arm/jointStatePosition/ 读取关节位置数据
            puppet_left = ep["/arm/jointStatePosition/puppetLeft"][:]  # 读取左臂关节位置数据
            puppet_right = ep["/arm/jointStatePosition/puppetRight"][:]  # 读取右臂关节位置数据
            master_left = ep["/arm/jointStatePosition/masterLeft"][:]  # 读取左臂主控关节位置数据
            master_right = ep["/arm/jointStatePosition/masterRight"][:]  # 读取右臂主控关节位置数据
            
            # 合并左右臂数据 [right_arm, left_arm]
            state = np.concatenate([puppet_right, puppet_left], axis=1)  # 合并从控端数据
            action = np.concatenate([master_right, master_left], axis=1)  # 合并主控端数据
            
            if verbose:
                print(f"新格式 - 状态数据形状: {state.shape}")  # 打印状态数据形状
                print(f"新格式 - 动作数据形状: {action.shape}")  # 打印动作数据形状
        else:  # 兼容旧格式
            # 旧格式：从 /puppet/arm_joint_position 和 /master/arm_joint_position 读取数据
            state = ep["/puppet/arm_joint_position"][:]  # 读取从控端关节位置数据
            action = ep["/master/arm_joint_position"][:]  # 读取主控端关节位置数据
            if verbose:
                print(f"旧格式 - 状态数据形状: {state.shape}")  # 打印状态数据形状
                print(f"旧格式 - 动作数据形状: {action.shape}")  # 打印动作数据形状

        # 归一化第7和第14维并裁剪到[0,1]
        if quantiles is not None:  # 如果分位数数据不为空
            if verbose:
                print(f"  🔢 使用计算的分位数进行归一化:")
                print(f"    puppet q01: {quantiles['puppet']['q01']}, q99: {quantiles['puppet']['q99']}")
                print(f"    master q01: {quantiles['master']['q01']}, q99: {quantiles['master']['q99']}")

            # puppet（state）第7维和第14维
            denom_puppet_7 = (quantiles["puppet"]["q99"][0] - quantiles["puppet"]["q01"][0]) or 1e-9
            denom_puppet_14 = (quantiles["puppet"]["q99"][1] - quantiles["puppet"]["q01"][1]) or 1e-9
            state[:, 6] = np.clip(
                (state[:, 6] - quantiles["puppet"]["q01"][0]) / denom_puppet_7,
                0.0,
                1.0,
            )
            state[:, 13] = np.clip(
                (state[:, 13] - quantiles["puppet"]["q01"][1]) / denom_puppet_14,
                0.0,
                1.0,
            )

            # master（action）第7维和第14维
            denom_master_7 = (quantiles["master"]["q99"][0] - quantiles["master"]["q01"][0]) or 1e-9
            denom_master_14 = (quantiles["master"]["q99"][1] - quantiles["master"]["q01"][1]) or 1e-9
            action[:, 6] = np.clip(
                (action[:, 6] - quantiles["master"]["q01"][0]) / denom_master_7,
                0.0,
                1.0,
            )
            action[:, 13] = np.clip(
                (action[:, 13] - quantiles["master"]["q01"][1]) / denom_master_14,
                0.0,
                1.0,
            )
        else:
            if verbose:
                print("  ⚠️ 警告: 没有提供分位数数据，跳过归一化")

        state = torch.from_numpy(state).to(torch.float32)  # 将状态数据转换为PyTorch张量
        action = torch.from_numpy(action).to(torch.float32)  # 将动作数据转换为PyTorch张量

        # 处理速度数据
        velocity = None  # 初始化速度数据为None
        if is_new_format:  # 如果是新格式
            if "/arm/jointStateVelocity" in ep:  # 如果存在新的速度结构
                vel_left = ep["/arm/jointStateVelocity/puppetLeft"][:]  # 读取左臂速度数据
                vel_right = ep["/arm/jointStateVelocity/puppetRight"][:]  # 读取右臂速度数据
                velocity = np.concatenate([vel_right, vel_left], axis=1)  # 合并速度数据
                velocity = torch.from_numpy(velocity).to(torch.float32)  # 转换为张量
        else:  # 兼容旧格式
            if "/observations/qvel" in ep:  # 如果存在速度观察数据
                velocity = torch.from_numpy(ep["/observations/qvel"][:])  # 读取速度数据并转换为张量

        # 处理力矩数据
        effort = None  # 初始化力矩数据为None
        if is_new_format:  # 如果是新格式
            if "/arm/jointStateEffort" in ep:  # 如果存在新的力矩结构
                effort_left = ep["/arm/jointStateEffort/puppetLeft"][:]  # 读取左臂力矩数据
                effort_right = ep["/arm/jointStateEffort/puppetRight"][:]  # 读取右臂力矩数据
                effort = np.concatenate([effort_right, effort_left], axis=1)  # 合并力矩数据
                effort = torch.from_numpy(effort).to(torch.float32)  # 转换为张量
        else:  # 兼容旧格式
            if "/observations/effort" in ep:  # 如果存在力矩观察数据
                effort = torch.from_numpy(ep["/observations/effort"][:])  # 读取力矩数据并转换为张量

        imgs_per_cam = load_raw_images_per_camera(  # 加载每个相机的原始图像
            ep,  # HDF5文件对象
            [  # 相机列表
                "cam_high",  # 高位相机
                # "cam_low",  # 低位相机（已注释）
                "cam_left_wrist",  # 左腕相机
                "cam_right_wrist",  # 右腕相机
            ],
            verbose=verbose,  # 传递verbose参数
        )

        # 处理位置数据（localization/pose）- 可选功能
        if is_new_format and "/localization/pose" in ep:  # 如果存在位置数据
            pose_left = ep["/localization/pose/puppetLeft"][:]  # 读取左臂位置数据
            pose_right = ep["/localization/pose/puppetRight"][:]  # 读取右臂位置数据
            if verbose:
                print(f"位置数据 - 左臂形状: {pose_left.shape}, 右臂形状: {pose_right.shape}")  # 打印位置数据形状
            # 注意：位置数据目前只是打印，如果需要可以添加到返回的元组中

    return imgs_per_cam, state, action, velocity, effort  # 返回图像字典、状态、动作、速度、力矩


def populate_dataset(  # 定义填充数据集的函数
    dataset: LeRobotDataset,  # LeRobot数据集对象
    hdf5_files: list[Path],  # HDF5文件路径列表
    task: str,  # 任务名称
    episodes: list[int] | None = None,  # 要处理的剧集索引列表，可选
    quantiles: dict | None = None,  # 分位数字典，用于数据归一化
    verbose: bool = False,  # 是否打印详细信息
) -> LeRobotDataset:  # 返回填充后的数据集
    if episodes is None:  # 如果没有指定剧集
        episodes = range(len(hdf5_files))  # 处理所有剧集

    if verbose:
        print(f"🎬 开始处理 {len(episodes)} 个剧集")
    
    for ep_idx in tqdm.tqdm(episodes):  # 遍历所有剧集，显示进度条
        ep_path = hdf5_files[ep_idx]  # 获取当前剧集文件路径
        if verbose:
            print(f"\n📁 处理剧集 {ep_idx}: {ep_path.name}")

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path, quantiles=quantiles, verbose=verbose)  # 加载剧集数据
        num_frames = state.shape[0]  # 获取帧数
        
        if verbose:
            print(f"📊 剧集数据统计:")
            print(f"  🖼️ 图像数据: {len(imgs_per_cam)} 个相机")
            for cam, imgs in imgs_per_cam.items():
                print(f"    {cam}: {imgs.shape} ({imgs.size * imgs.dtype.itemsize / 1024 / 1024:.2f} MB)")
            print(f"  🤖 状态数据: {state.shape}")
            print(f"  🎯 动作数据: {action.shape}")
            if velocity is not None:
                print(f"  ⚡ 速度数据: {velocity.shape}")
            if effort is not None:
                print(f"  💪 力矩数据: {effort.shape}")

        frames_with_images = 0  # 统计有图像的帧数
        frames_without_images = 0  # 统计没有图像的帧数
        
        for i in range(num_frames):  # 遍历所有帧
            frame = {  # 创建帧数据字典
                "observation.state": state[i],  # 添加状态观察
                "action": action[i],  # 添加动作
            }

            has_images = False  # 标记是否有图像数据
            for camera, img_array in imgs_per_cam.items():  # 遍历所有相机
                if i < len(img_array):  # 确保索引不超出范围
                    frame[f"observation.images.{camera}"] = img_array[i]  # 添加相机图像观察
                    has_images = True  # 标记有图像
                else:
                    if verbose:
                        print(f"  ⚠️ 警告: 帧 {i} 超出 {camera} 图像范围 ({len(img_array)} 张图像)")
            
            if has_images:
                frames_with_images += 1
            else:
                frames_without_images += 1
                if verbose:
                    print(f"  ⚠️ 警告: 帧 {i} 没有图像数据")

            if velocity is not None:  # 如果存在速度数据
                frame["observation.velocity"] = velocity[i]  # 添加速度观察
            if effort is not None:  # 如果存在力矩数据
                frame["observation.effort"] = effort[i]  # 添加力矩观察

            frame["task"] = task  # 添加任务标签

            dataset.add_frame(frame)  # 将帧添加到数据集

        if verbose:
            print(f"📈 剧集 {ep_idx} 完成:")
            print(f"  ✅ 有图像的帧: {frames_with_images}")
            print(f"  ❌ 无图像的帧: {frames_without_images}")
            print(f"  📊 总帧数: {num_frames}")
        
        dataset.save_episode()  # 保存剧集
        if verbose:
            print(f"  💾 剧集已保存")

    if verbose:
        print(f"\n🎉 所有剧集处理完成!")
    return dataset  # 返回填充后的数据集


def port_aloha(  # 定义转换Aloha数据的主函数
    raw_dir: Path,  # 原始数据目录路径
    repo_id: str,  # 仓库ID
    raw_repo_id: str | None = None,  # 原始仓库ID，可选
    task: str = "DEBUG",  # 任务名称，默认为"DEBUG"
    *,  # 强制关键字参数分隔符
    episodes: list[int] | None = None,  # 要处理的剧集列表，可选
    push_to_hub: bool = False,  # 是否推送到Hub，默认为False
    is_mobile: bool = False,  # 是否为移动机器人，默认为False
    mode: Literal["video", "image"] = "image",  # 数据模式，默认为图像
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,  # 数据集配置
):
    if (HF_LEROBOT_HOME / repo_id).exists():  # 如果数据集目录已存在
        print("数据集目录已存在",HF_LEROBOT_HOME / repo_id)
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)  # 删除现有目录

    # if not raw_dir.exists():  # 注释掉的原始目录检查
    #     if raw_repo_id is None:  # 注释掉的原始仓库ID检查
    #         raise ValueError("raw_repo_id must be provided if raw_dir does not exist")  # 注释掉的错误抛出
    #     download_raw(raw_dir, repo_id=raw_repo_id)  # 注释掉的原始数据下载

    hdf5_files = [  # 查找所有HDF5文件
        Path(dirpath) / filename  # 构建文件路径
        for dirpath, _, filenames in os.walk(raw_dir, followlinks=True)  # 遍历目录树
        for filename in filenames  # 遍历文件名
        if filename.endswith(".hdf5")  # 筛选HDF5文件
    ]
    
    if dataset_config.verbose:
        print(f"🔍 找到 {len(hdf5_files)} 个HDF5文件:")
        for i, hdf5_file in enumerate(hdf5_files):
            file_size = hdf5_file.stat().st_size / 1024 / 1024  # 文件大小（MB）
            print(f"  {i}: {hdf5_file.name} ({file_size:.2f} MB)")

        # 检查第一个文件的结构
        if hdf5_files:
            print(f"\n🔬 分析第一个文件: {hdf5_files[0].name}")
            try:
                with h5py.File(hdf5_files[0], "r") as f:
                    print("文件结构:")
                    def print_structure(name, obj, level=0):
                        indent = "  " * level
                        if isinstance(obj, h5py.Dataset):
                            size_mb = obj.size * obj.dtype.itemsize / 1024 / 1024
                            print(f"{indent}📄 {name} ({obj.dtype}, {obj.shape}, {size_mb:.2f} MB)")
                        else:
                            print(f"{indent}📁 {name}/")
                            if level < 2:  # 只显示前两层
                                for key in obj.keys():
                                    print_structure(key, obj[key], level + 1)
                    
                    f.visititems(print_structure)
            except Exception as e:
                print(f"❌ 无法读取文件: {e}")

    dataset = create_empty_dataset(  # 创建空数据集
        repo_id,  # 仓库ID
        robot_type="mobile_aloha" if is_mobile else "aloha",  # 根据是否为移动机器人选择机器人类型
        mode=mode,  # 数据模式
        has_effort=has_effort(hdf5_files),  # 检查是否包含力矩数据
        has_velocity=has_velocity(hdf5_files),  # 检查是否包含速度数据
        dataset_config=dataset_config,  # 数据集配置
    )
    
    if dataset_config.verbose:
        print(f"\n🏗️ 创建数据集: {repo_id}")
        print(f"  🤖 机器人类型: {'mobile_aloha' if is_mobile else 'aloha'}")
        print(f"  📊 数据模式: {mode}")
        print(f"  ⚡ 包含速度: {has_velocity(hdf5_files)}")
        print(f"  💪 包含力矩: {has_effort(hdf5_files)}")
    
    # 计算分位数
    quantiles = calculate_quantiles(hdf5_files, verbose=dataset_config.verbose)
    
    task='Put the fork in the box.'  # 设置任务名称为"Clean the table."
    dataset = populate_dataset(  # 填充数据集
        dataset,  # 数据集对象
        hdf5_files,  # HDF5文件列表
        task=task,  # 任务名称
        episodes=episodes,  # 剧集列表
        quantiles=quantiles,  # 传递计算的分位数
        verbose=dataset_config.verbose,  # 传递verbose参数
    )

    if push_to_hub:  # 如果需要推送到Hub
        dataset.push_to_hub()  # 推送数据集到Hub


if __name__ == "__main__":  # 如果作为主程序运行
    tyro.cli(port_aloha)  # 使用tyro命令行接口调用port_aloha函数
