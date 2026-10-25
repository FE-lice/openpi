import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


# ---------------------------------------------------------------------------
# 工厂函数：从训练好的 checkpoint 创建一个可直接推理的 Policy
#
# 它把「训练配置 + 权重目录 + 数据变换 + 归一化统计」这些零散信息，
# 组装成一个对外统一的策略对象（见 policy.Policy）
# ---------------------------------------------------------------------------
def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.
    （从训练好的 checkpoint 创建一个策略对象）

    Args:
        train_config: 用于创建模型的训练配置。
        checkpoint_dir: 加载模型的权重目录（本地路径或可下载路径）
        repack_transforms: 可选的、最先执行的一层数据重排变换
        sample_kwargs: 传给 `sample_actions` 的参数；不提供则使用模型默认值
        default_prompt: 默认任务提示词；若输入里没有 prompt 字段会自动注入
        norm_stats: 归一化统计量；不提供时从 checkpoint 目录中加载
        pytorch_device: PyTorch 模型使用的设备（如 "cpu"、"cuda"、"cuda:0"）
            若为 None 且是 PyTorch 模型，则自动选择 cuda（可用时）或 cpu

    Note:
        该函数通过检查 checkpoint 目录是否存在 model.safetensors
        来自动判断模型是 PyTorch 还是 JAX 实现
    """
    # 若未提供重排变换，则用空 Group 占位（保证后面的 * 解包不会报错）
    repack_transforms = repack_transforms or transforms.Group()
    # 支持远程路径：若是 URL/仓库路径则先下载到本地缓存
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    # 通过是否存在 model.safetensors 来判断模型类型
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        # ---------- PyTorch 分支 ----------
        # 从 safetensors 加载权重
        model = train_config.model.load_pytorch(train_config, weight_path)
        # 只把部分参数转成 bfloat16，兼顾显存占用与数值精度
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # ---------- JAX 分支 ----------
        # 从 Orbax checkpoint 还原参数树，并指定精度为 bfloat16
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    # 根据训练配置创建数据配置（决定输入/输出的变换管线）
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        # 从 checkpoint 而不是配置的 assets 目录加载归一化统计量，
        # 以保证推理时用的归一化参数与训练时完全一致。
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    # 为 PyTorch 模型选择运行设备：有 GPU 用 cuda，否则退回 cpu
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        # ---------- 推理前的输入处理链 ----------
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        # ---------- 推理后的输出处理链（与输入基本对称、顺序相反） ----------
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        # 策略元数据（如机器人类型、action 维度等）随策略一起保存
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        # 只有 PyTorch 模型才需要设备参数，JAX 传 None
        pytorch_device=pytorch_device if is_pytorch else None,
    )
