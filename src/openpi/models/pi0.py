import logging  # 日志库

import einops  # 张量重排/复制工具
import flax.nnx as nnx  # Flax NNX 模块
import flax.nnx.bridge as nnx_bridge  # 旧模块到 NNX 的桥接
import jax  # JAX 主库
import jax.numpy as jnp  # JAX 的 numpy 接口
from typing_extensions import override  # 方法重写标注

from openpi.models import model as _model  # 模型基类与类型
from openpi.models import pi0_config  # Pi0 配置
import openpi.models.gemma as _gemma  # 语言模型（Gemma）
import openpi.models.siglip as _siglip  # 视觉编码器（SigLIP）
from openpi.shared import array_typing as at  # 形状/类型注释

logger = logging.getLogger("openpi")  # 获取 openpi 日志记录器


def make_attn_mask(input_mask, mask_ar):  # 构造注意力掩码（因果/分段）
    """改编自 big_vision。

    Token 只能注意到其累计 mask_ar 值小于或等于自身的有效输入 token。
    因此 `mask_ar`（bool[?B, N]）可用于设置多种注意力形态，例如：

      [[1 1 1 1 1 1]]：纯因果注意力。

      [[0 0 0 1 1 1]]：前缀语言模型注意力（prefix-lm）。前 3 个 token 可互相注意，
          后 3 个 token 采用因果注意力。第一个条目即使为 1 也不改变行为。

      [[1 0 1 0 1 0 0 1 0 0]]：分为 4 个块的因果注意力。每个块中的 token 可注意所有
          先前块以及同一块中的所有 token。

    参数:
      input_mask: bool[B, N]，为 True 表示真实输入，False 表示填充。
      mask_ar: bool[?B, N]，为 True 表示之前的 token 不能依赖当前 token；
        为 False 表示与前一个 token 共享同一注意力掩码。
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)  # 广播到[B, N]
    cumsum = jnp.cumsum(mask_ar, axis=1)  # 累积表示段落/时间顺序
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # 查询只能看不晚于自身的键
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  # 仅允许有效 token 互相注意
    return jnp.logical_and(attn_mask, valid_mask)  # 合并两类约束


@at.typecheck  # 运行时形状/类型检查
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """为标量位置计算正弦-余弦位置嵌入向量。"""
    if embedding_dim % 2 != 0:  # 维度必须为偶数
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)  # [0,1] 等间隔
    period = min_period * (max_period / min_period) ** fraction  # 对数均匀周期
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,  # 位置标量（如时间）
        1.0 / period * 2 * jnp.pi,  # 频率
        precision=jax.lax.Precision.HIGHEST,  # 最高数值精度
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)  # 拼接 sin/cos


class Pi0(_model.BaseModel):  # Pi0 策略模型
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):  # 初始化结构
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)  # 基类初始化
        self.pi05 = config.pi05  # 是否使用 pi0.5（adaRMS）
        self.action_loss_weight = config.action_loss_weight
        self.image_denoise_loss_weight = config.image_denoise_loss_weight
        paligemma_config = _gemma.get_config(config.paligemma_variant)  # 获取 PaliGemma 配置
        action_expert_config = _gemma.get_config(config.action_expert_variant)  # 获取动作专家配置
        # TODO：将 gemma 重写为 NNX 版本。目前先使用桥接实现。
        llm = nnx_bridge.ToNNX(  # 将 Gemma 模块桥接至 NNX
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],  # 两路配置
                embed_dtype=config.dtype,  # 嵌入精度
                adarms=config.pi05,  # 是否启用 adaRMS 条件
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])  # 延迟初始化
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 与 LLM 宽度对齐
                variant="So400m/14",  # SigLIP 变体
                pool_type="none",  # 不做池化
                scan=True,  # 扫描（并行）
                dtype_mm=config.dtype,  # 精度设置
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)  # 用假样本初始化形状
        self.PaliGemma = nnx.Dict(llm=llm, img=img)  # 聚合视觉与语言模块
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)  # 动作输入投影
        if config.pi05:  # pi0.5：时间嵌入经 MLP 提供给 adaRMS
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:  # 纯 pi0：显式状态 token + 动作时间拼接 MLP
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)  # 输出动作残差
        # 图像 decoder：将 token 映射回图像 patch
        # SigLIP patch_size=14，每个 patch 是 14x14x3 = 588 像素
        patch_size = 14
        patch_pixels = patch_size * patch_size * 3
        self.image_out_proj = nnx.Linear(paligemma_config.width, patch_pixels, rngs=rngs)  # 图像输出投影

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True  # 控制确定性行为（默认 True）

    @at.typecheck  # 前缀（图像+文本）嵌入
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
    ]:
        input_mask = []  # 有效 token 掩码列表
        ar_mask = []  # 自回归/分段掩码（False 允许互相注意）
        tokens = []  # 收集所有前缀 token
        image_token_mask = []  # 仅图像 token 的位置
        # embed images
        for name in obs.images:  # 遍历各路图像输入
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)  # 视觉编码输出序列

            tokens.append(image_tokens)  # 收集图像 token
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],  # 每张图像对应的有效掩码
                    "b -> b s",
                    s=image_tokens.shape[1],  # 扩展到图像 token 维度
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]  # 图像 token 之间全连接注意力
            image_token_mask.append(jnp.ones((image_tokens.shape[1],), dtype=jnp.bool_))

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:  # 若存在文本输入
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")  # LLM 嵌入
            tokens.append(tokenized_inputs)  # 加入文本 token
            input_mask.append(obs.tokenized_prompt_mask)  # 文本有效掩码
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]  # 图文之间允许全注意
            image_token_mask.append(jnp.zeros((tokenized_inputs.shape[1],), dtype=jnp.bool_))
        tokens = jnp.concatenate(tokens, axis=1)  # 拼接前缀 token
        input_mask = jnp.concatenate(input_mask, axis=1)  # 拼接掩码
        ar_mask = jnp.array(ar_mask)  # 列表转数组
        image_token_mask = jnp.concatenate(image_token_mask, axis=0)
        return tokens, input_mask, ar_mask, image_token_mask  # 返回前缀序列与掩码

    @at.typecheck  # 后缀（状态/动作/时间）嵌入
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []  # 有效 token 掩码
        ar_mask = []  # 自回归掩码
        tokens = []  # 收集后缀 token
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]  # 将状态映射为单个 token
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))  # 状态 token 有效
            # image/language inputs do not attend to state or actions
            ar_mask += [True]  # 阻断与此前缀混合的回看关系

        action_tokens = self.action_in_proj(noisy_actions)  # 动作序列线性投影
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)  # 时间嵌入
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)  # 时间条件 MLP（供 adaRMS）
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens  # pi0.5 使用纯动作 token
            adarms_cond = time_emb  # 作为 adaRMS 条件
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)  # 扩展到时间维
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)  # 拼接动作和时间
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens  # 融合后的 token
            adarms_cond = None  # 无 adaRMS 条件
        tokens.append(action_expert_tokens)  # 追加动作相关 token
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))  # 后缀 token 全有效
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))  # 仅首个位置阻断
        tokens = jnp.concatenate(tokens, axis=1)  # 拼接后缀序列
        input_mask = jnp.concatenate(input_mask, axis=1)  # 拼接掩码
        ar_mask = jnp.array(ar_mask)  # 列表转数组
        return tokens, input_mask, ar_mask, adarms_cond  # 返回序列与掩码和 adaRMS 条件

    @override  # 训练时的损失计算
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, img_seg: at.Array | None = None, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, image_noise_rng = jax.random.split(rng, 4)  # 随机数拆分
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)  # 观测预处理

        batch_shape = actions.shape[:-2]  # 批形状
        noise = jax.random.normal(noise_rng, actions.shape)  # 高斯噪声
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # t∈(0,1]，避免端点
        time_expanded = time[..., None, None]  # 扩展到 (b, ah, 1)
        x_t = time_expanded * noise + (1 - time_expanded) * actions  # 线性插值的带噪动作
        u_t = noise - actions  # 目标速度/残差

        # 一次性完成前缀与后缀的联合前向传播
        prefix_tokens, prefix_mask, prefix_ar_mask, image_token_mask = self.embed_prefix(observation)  # 前缀序列
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)  # 后缀序列
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)  # 合并掩码
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)  # 合并自回归掩码
        attn_mask = make_attn_mask(input_mask, ar_mask)  # 构造注意力掩码
        positions = jnp.cumsum(input_mask, axis=1) - 1  # 绝对位置
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # 仅用最后 ah 个位置

        # 计算 action loss，按维度归一化（对 action_dim 维度求平均，然后除以 action_horizon）
        action_sq_error = jnp.square(v_t - u_t)  # (b, ah, ad)
        # 对 action_dim 维度求平均，得到每个时间步的 loss，形状保持为 (b, ah)
        action_loss = jnp.mean(action_sq_error, axis=-1)  # (b, ah)
        
        # 计算图像 loss，完全仿照 action_loss 的方式
        # 初始化 image_loss 为 (b, ah) 形状，与 action_loss 保持一致
        image_loss = jnp.zeros(action_loss.shape, dtype=action_loss.dtype)  # (b, ah)
        
        if self.image_denoise_loss_weight > 0.0 and img_seg is not None:
            # 获取 groundtruth 图像
            target_image = img_seg  # (b, h, w, c)
            
            # 确保 target_image 在 [-1, 1] 范围内（与 action 类似）
            target_max = jnp.max(target_image)
            target_min = jnp.min(target_image)
            needs_normalization = (target_max > 1.0) | (target_min < -1.0)
            is_uint8_range = (target_max > 1.0) & (target_min >= 0.0)
            is_float01_range = (target_max <= 1.0) & (target_min >= 0.0) & (target_max > 0.0)
            
            target_image = jnp.where(
                needs_normalization,
                jnp.where(
                    is_uint8_range,
                    target_image.astype(jnp.float32) / 255.0 * 2.0 - 1.0,  # [0, 255] -> [-1, 1]
                    jnp.where(
                        is_float01_range,
                        target_image * 2.0 - 1.0,  # [0, 1] -> [-1, 1]
                        (target_image - target_min) / jnp.maximum(target_max - target_min, 1e-8) * 2.0 - 1.0
                    )
                ),
                target_image
            )
            
            # 完全仿照 action_loss 的方式计算图像 loss
            # 1. 生成图像噪声（与 action 的 noise 类似）
            image_noise = jax.random.normal(image_noise_rng, target_image.shape, dtype=target_image.dtype)  # (b, h, w, c)
            
            # 2. 使用相同的时间（与 action 的 time 相同）
            # time_expanded 是 (b, 1, 1)，需要扩展到 (b, ah, 1, 1, 1) 以匹配图像维度
            time_expanded_image = einops.repeat(
                time_expanded, "b 1 1 -> b ah 1 1 1", ah=self.action_horizon
            )  # (b, ah, 1, 1, 1)
            
            # 3. 计算带噪图像（与 action 的 x_t 类似）
            # 将 target_image 扩展到 (b, ah, h, w, c) 以匹配 action_horizon
            target_image_expanded = einops.repeat(
                target_image, "b h w c -> b ah h w c", ah=self.action_horizon
            )  # (b, ah, h, w, c)
            image_noise_expanded = einops.repeat(
                image_noise, "b h w c -> b ah h w c", ah=self.action_horizon
            )  # (b, ah, h, w, c)
            image_x_t = time_expanded_image * image_noise_expanded + (1.0 - time_expanded_image) * target_image_expanded  # (b, ah, h, w, c)
            
            # 4. 计算目标速度（与 action 的 u_t 类似）
            image_u_t = image_noise_expanded - target_image_expanded  # (b, ah, h, w, c)
            
            # 5. 从模型输出解码图像并计算预测速度（与 action 的 v_t 类似）
            # 提取图像 token
            batched_image_mask = jnp.broadcast_to(image_token_mask[None, :], prefix_mask.shape)
            valid_image_mask = jnp.logical_and(batched_image_mask, prefix_mask)
            
            patch_size = 14
            image_h, image_w = 224, 224
            num_patches_h = image_h // patch_size  # 16
            num_patches_w = image_w // patch_size  # 16
            tokens_per_image = num_patches_h * num_patches_w  # 256
            
            batch_size = prefix_out.shape[0]
            first_image_token_outputs = prefix_out[:, :tokens_per_image]  # (b, 256, emb_dim)
            
            # 使用 decoder 将 token 映射到 patch 像素值
            patch_pixels = self.image_out_proj(first_image_token_outputs)  # (b, 256, patch_size*patch_size*3)
            
            # Reshape 到图像格式
            patch_pixels = patch_pixels.reshape(
                batch_size, num_patches_h, num_patches_w, patch_size, patch_size, 3
            )
            decoded_image = einops.rearrange(
                patch_pixels,
                "b h_patches w_patches h_patch w_patch c -> b (h_patches h_patch) (w_patches w_patch) c"
            )  # (b, 224, 224, 3)
            
            # 确保尺寸匹配
            if decoded_image.shape[1:3] != target_image.shape[1:3]:
                from openpi.shared import image_tools
                decoded_image = image_tools.resize_with_pad(
                    decoded_image, target_image.shape[1], target_image.shape[2]
                )
            
            # 将 decoded_image 扩展到 (b, ah, h, w, c) 以匹配 action_horizon
            decoded_image_expanded = einops.repeat(
                decoded_image, "b h w c -> b ah h w c", ah=self.action_horizon
            )  # (b, ah, h, w, c)
            
            # 计算预测速度（与 action 的 v_t 类似）
            image_v_t = decoded_image_expanded - target_image_expanded  # (b, ah, h, w, c)
            
            # 6. 计算平方误差（与 action_sq_error 类似）
            image_sq_error = jnp.square(image_v_t - image_u_t)  # (b, ah, h, w, c)
            
            # 7. 对空间和通道维度（h, w, c）求平均，得到每个时间步的 loss，形状保持为 (b, ah)
            image_loss = jnp.mean(image_sq_error, axis=(-3, -2, -1))  # (b, ah)

        return self.action_loss_weight * action_loss + self.image_denoise_loss_weight * image_loss

    @override  # 推理：从噪声逐步去噪得到动作
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)  # 预处理
        # 注意：这里采用扩散文献中更常见的约定，t=1 表示纯噪声，t=0 表示目标分布。
        # 是的，这与 Pi0 论文的表述相反，抱歉带来困惑。
        dt = -1.0 / num_steps  # 负时间步长（从 t=1 走到 0）
        batch_size = observation.state.shape[0]  # 批大小
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))  # 初始噪声

        # 首先通过一次前缀前向传播填充 KV 缓存
        prefix_tokens, prefix_mask, prefix_ar_mask, _ = self.embed_prefix(observation)  # 计算前缀
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)  # 前缀自注意掩码
        positions = jnp.cumsum(prefix_mask, axis=1) - 1  # 前缀位置
        (prefix_out_large, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)  # 缓存 KV

        def step(carry):  # 单步积分
            x_t, time = carry  # 当前动作与时间标量
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` 形状为 (b, suffix_len, suffix_len)，表示后缀 token 之间的相互注意关系
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)  # 后缀内部注意
            # `prefix_attn_mask` 形状为 (b, suffix_len, prefix_len)，表示后缀 token 如何注意到前缀 token
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])  # 后缀看前缀
            # `combined_mask` 形状为 (b, suffix_len, prefix_len + suffix_len)，表示后缀 token（产生 queries）
            # 如何注意到完整的前缀+后缀序列（产生 keys 与 values）
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)  # 合并成全掩码
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` 形状为 (b, suffix_len)，表示每个后缀 token 的绝对位置
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1  # 后缀绝对位置

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(  # 仅解码后缀，复用 KV
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None  # 不应返回前缀输出
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # 动作速度预测

            return x_t + dt * v_t, time + dt  # 欧拉步进更新

        def cond(carry):  # while 循环条件
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))  # 从噪声积分到 t≈0
        return x_0  # 返回预测动作

