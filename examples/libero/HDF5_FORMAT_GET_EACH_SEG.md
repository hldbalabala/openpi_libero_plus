# HDF5 文件格式说明 - get_each_seg 模式

本文档描述在 `get_each_seg=True` 模式下保存的 HDF5 文件格式。

## 文件结构

HDF5 文件采用层次结构，每个成功的 episode 保存为一个顶级组（group）。

```
replay_success.hdf5
├── task{task_id}_episode{episode_idx}/
│   ├── (attributes)
│   ├── images
│   ├── wrist_images
│   ├── img_segs/          # Group containing object-specific segmentations
│   │   ├── robot
│   │   ├── cabinet
│   │   ├── bowl
│   │   └── ...
│   ├── states
│   └── actions
├── task{task_id}_episode{episode_idx}_dup1/
│   └── ...
└── ...
```

## Group 结构

每个 episode group 的命名格式：`task{task_id}_episode{episode_idx}`

如果组名冲突，会自动添加 `_dup{suffix}` 后缀（如 `task0_episode0_dup1`）。

### Group Attributes（组属性）

每个 episode group 包含以下属性：

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `task_id` | int | 任务 ID（从 0 开始） |
| `task_description` | str | 任务描述文本 |
| `episode_idx` | int | Episode 索引（从 0 开始） |

### Datasets（数据集）

#### 1. `images` - Agent View 图像

- **类型**: Dataset (numpy array)
- **形状**: `(num_frames, height, width, channels)`
- **数据类型**: `uint8`
- **压缩**: gzip
- **说明**: Agent view 的原始 RGB 图像序列
- **预处理**: 
  - 图像翻转 180 度（`[::-1, ::-1]`）
  - Resize 并 padding 到 `resize_size x resize_size`（默认 224x224）
  - 转换为 uint8

#### 2. `wrist_images` - Wrist View 图像

- **类型**: Dataset (numpy array)
- **形状**: `(num_frames, height, width, channels)`
- **数据类型**: `uint8`
- **压缩**: gzip
- **说明**: Wrist camera 的原始 RGB 图像序列
- **预处理**: 同 `images`

#### 3. `img_segs` - 分割图像组

- **类型**: Group（在 `get_each_seg` 模式下）
- **说明**: 包含每个对象的分割图像数据集

##### `img_segs/robot` - Robot 分割

- **类型**: Dataset (numpy array)
- **形状**: `(num_frames, height, width, channels)`
- **数据类型**: `uint8`
- **压缩**: gzip
- **说明**: Robot 的分割图像序列
- **特点**: **总是存在**（每个 episode 都会保存 robot 的分割）

##### `img_segs/{object_name}` - 对象分割

- **类型**: Dataset (numpy array)
- **形状**: `(num_frames, height, width, channels)`
- **数据类型**: `uint8`
- **压缩**: gzip
- **说明**: 特定对象的分割图像序列
- **对象名**: 可能包括以下对象（取决于任务描述）：
  - `cabinet`
  - `bowl`
  - `plate`
  - `stove`
  - `cream cheese`
  - `rack`
  - `wine bottle`
- **保存规则**: 
  - 只有当对象名出现在 `task_description_new` 中时才会保存
  - 如果对象在某一帧不存在，该帧用全零图像填充（与第一帧的形状相同）

#### 4. `states` - 机器人状态

- **类型**: Dataset (numpy array)
- **形状**: `(num_frames, state_dim)`
- **数据类型**: `float32`
- **压缩**: gzip
- **说明**: 机器人状态序列
- **状态维度**: 7（3 位置 + 3 旋转轴角 + 1 夹爪）
- **组成**:
  - `robot0_eef_pos` (3D): 末端执行器位置
  - `quat2axisangle(robot0_eef_quat)` (3D): 末端执行器旋转（轴角表示）
  - `robot0_gripper_qpos` (1D): 夹爪开合状态

#### 5. `actions` - 动作序列

- **类型**: Dataset (numpy array)
- **形状**: `(num_frames, action_dim)`
- **数据类型**: `float32`
- **压缩**: gzip
- **说明**: 执行的动作序列
- **动作维度**: 7（与状态维度相同）
- **格式**: `[x, y, z, rx, ry, rz, gripper]`

## 数据预处理流程

### 图像预处理

1. **原始图像获取**: 从环境观察中获取 `agentview_image` 和 `robot0_eye_in_hand_image`
2. **翻转**: 图像翻转 180 度（`[::-1, ::-1]`）以匹配训练时的预处理
3. **分割提取**: 
   - Robot: 总是提取
   - 其他对象: 仅当对象名在任务描述中时提取
4. **分割应用**: 将分割掩码应用到原始图像上
5. **Resize 和 Padding**: 使用 `image_tools.resize_with_pad()` 调整到目标尺寸
6. **类型转换**: 转换为 `uint8`

### 分割图像生成

对于每个对象的分割图像：

```python
# 1. 获取对象的分割掩码
obj_mask = env.get_segmentation_input_obj(segmentation_instance, obj_name)

# 2. 翻转掩码
obj_mask = obj_mask[::-1, ::-1]

# 3. 应用到原始图像
obj_seg = (original_image * obj_mask).astype(original_image.dtype)

# 4. Resize 和转换
obj_seg_processed = convert_to_uint8(resize_with_pad(obj_seg, resize_size, resize_size))
```

## 示例：读取 HDF5 文件

### Python 示例

```python
import h5py
import numpy as np

# 打开 HDF5 文件
with h5py.File("replay_success.hdf5", "r") as f:
    # 列出所有 episode groups
    episode_names = list(f.keys())
    print(f"Total episodes: {len(episode_names)}")
    
    # 读取第一个 episode
    episode_name = episode_names[0]
    episode = f[episode_name]
    
    # 读取属性
    task_id = episode.attrs["task_id"]
    task_description = episode.attrs["task_description"]
    episode_idx = episode.attrs["episode_idx"]
    
    print(f"Task ID: {task_id}")
    print(f"Task Description: {task_description}")
    print(f"Episode Index: {episode_idx}")
    
    # 读取图像数据
    images = episode["images"][:]  # Shape: (num_frames, H, W, 3)
    wrist_images = episode["wrist_images"][:]  # Shape: (num_frames, H, W, 3)
    
    # 读取分割图像（get_each_seg 模式）
    img_segs_group = episode["img_segs"]
    
    # Robot 分割（总是存在）
    robot_segs = img_segs_group["robot"][:]  # Shape: (num_frames, H, W, 3)
    
    # 其他对象分割（如果存在）
    available_objects = list(img_segs_group.keys())
    print(f"Available objects: {available_objects}")
    
    for obj_name in available_objects:
        if obj_name != "robot":
            obj_segs = img_segs_group[obj_name][:]  # Shape: (num_frames, H, W, 3)
            print(f"{obj_name} segmentation shape: {obj_segs.shape}")
    
    # 读取状态和动作
    states = episode["states"][:]  # Shape: (num_frames, 7)
    actions = episode["actions"][:]  # Shape: (num_frames, 7)
    
    print(f"Number of frames: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"State shape: {states[0].shape}")
    print(f"Action shape: {actions[0].shape}")
```

### 检查对象是否存在

```python
def has_object_segmentation(h5_file, episode_name, obj_name):
    """检查指定 episode 是否包含某个对象的分割"""
    episode = h5_file[episode_name]
    if "img_segs" in episode:
        img_segs_group = episode["img_segs"]
        return obj_name in img_segs_group
    return False

# 使用示例
with h5py.File("replay_success.hdf5", "r") as f:
    episode_name = list(f.keys())[0]
    has_cabinet = has_object_segmentation(f, episode_name, "cabinet")
    print(f"Has cabinet segmentation: {has_cabinet}")
```

## 注意事项

1. **帧数一致性**: 所有数据集（`images`, `wrist_images`, `img_segs/*`, `states`, `actions`）的第一维（帧数）必须相同

2. **对象存在性**: 
   - `robot` 分割总是存在
   - 其他对象的分割仅当对象名出现在任务描述中时才存在
   - 如果对象在某一帧不存在，该帧的分割图像为全零

3. **图像尺寸**: 所有图像（包括分割图像）都经过 resize 和 padding，尺寸统一为 `resize_size x resize_size`（默认 224x224）

4. **数据类型**: 
   - 图像数据: `uint8` (0-255)
   - 状态和动作: `float32`

5. **压缩**: 所有数据集都使用 gzip 压缩以节省存储空间

## 与普通模式的对比

| 特性 | 普通模式 (`get_each_seg=False`) | get_each_seg 模式 (`get_each_seg=True`) |
|------|--------------------------------|------------------------------------------|
| `img_segs` 类型 | Dataset (单个数组) | Group (包含多个数据集) |
| 分割图像数量 | 1 个（所有对象合并） | 多个（每个对象单独） |
| Robot 分割 | 包含在合并分割中 | 单独数据集 `img_segs/robot` |
| 对象分割 | 不单独保存 | 每个对象单独数据集 |
| 文件大小 | 较小 | 较大（多个分割图像） |

## 文件命名

- 单个 worker: `replay_success.hdf5`
- 多进程模式: `replay_success_worker{worker_id}.hdf5`（每个 worker 一个文件）
- 合并后: `replay_success.hdf5`（所有 worker 文件合并后的最终文件）

