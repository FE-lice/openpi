from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

# 类型别名：策略基类来自 openpi_client，外部环境只依赖这个抽象接口
BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    """策略封装层：把「模型推理」包装成统一的策略接口。

    对外只暴露 infer(obs) -> dict，内部负责：
    1. 输入数据变换（transforms）；
    2. JAX / PyTorch 两种后端的数据格式与设备适配；
    3. 调用 model.sample_actions 采样动作；
    4. 输出数据变换（output_transforms）与耗时统计。
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.
        （初始化策略封装层）

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.

    infer(obs)
   │
   ├─ 深拷贝 obs :dict   防止transform内置函数修改数据来源的内容，只允许修改拷贝后的数据x
   ├─ 输入变换（transforms）
   ├─ 推理时可能没有batch维但为了适应算法加 batch 维度 + 根据后端转成 jax.Array 或 torch.Tensor并给sample_rng_or_pytorch_device
   ├─ 组装 sample_kwargs（拼接 noise），里面包含action方法需要的参数如总时间步长，温度等
   ├─ 从x 得到 Observation对象
   ├─ 调用 sample_actions 
   ├─ 去 batch + 转 NumPy ，pytorch还需detach + 搬回 CPU
   ├─ 输出变换（output_transforms）  反归一成关节之类的具体物理量
   ├─ 计算耗时
   └─ 返回 {actions, state, policy_timing}
        """
        self._model = model
        # 把多个输入/输出变换函数按顺序组合成一条流水线
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        # 采样时额外传给 model.sample_actions 的参数（如 num_steps、noise 等）
        self._sample_kwargs = sample_kwargs or {}
        # 附加元数据，随策略一起保存/读取
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            # ---------- PyTorch 后端 ----------
            # 把模型搬到目标设备并切到推理模式（关闭 dropout / BN 更新等）
            self._model = self._model.to(pytorch_device)
            self._model.eval()        # 自动设置self.deterministic参数
            # 复用模型的 sample_actions 方法
            self._sample_actions = model.sample_actions
        else:
            # ---------- JAX 后端 ----------
            # 用 module_jit 包装采样函数，把 NNX 模块状态和参数一起编译进 jit
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            # JAX 采样需要随机数 key；未提供时使用固定的默认 key(0)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """执行一次策略推理。

        Args:
            obs: 原始观测字典（来自真实/仿真环境的输出）。
            noise: 可选的初始噪声。传入固定噪声可让扩散/流匹配采样过程可复现。

        Returns:
            包含 "actions"（动作）、"state"（状态）和 "policy_timing"（耗时）的字典。
        """
        # 先深拷贝一份（用 tree.map 逐叶子复制），因为变换函数可能会原地修改输入
        inputs = jax.tree.map(lambda x: x, obs)
        # 1) 输入变换：把环境原始观测整理成模型需要的字段与格式
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # 2) JAX 分支：给每个叶子增加 batch 维度并转成 jax.Array
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            # 拆分随机数 key：一个留给下次调用，一个用于本次采样
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # 2) PyTorch 分支：转成张量、搬到目标设备并增加 batch 维度
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            # PyTorch 不用随机 key，这里复用它传递设备名，保持下面调用代码统一
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        # 3) 组装采样参数（复制一份，避免污染 self._sample_kwargs）
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            # 根据后端把噪声转成对应张量类型
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # 4) 把输入字典转成模型内部的 Observation 数据结构
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            # 5) 调用采样函数：JAX 传 rng key，PyTorch 传设备字符串
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        # 6) 去掉 batch 维度并转成 NumPy（PyTorch 还需 detach + 搬回 CPU）
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # 7) 输出变换：把模型输出转回环境期望的动作格式
        outputs = self._output_transform(outputs)
        # 8) 附加本次推理耗时（毫秒）
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        # 返回初始化时传入的元数据（只读属性）
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """把策略的每次行为（输入 + 输出）记录到磁盘，便于调试与数据采集。"""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy
        logging.info(f"Dumping policy records to: {record_dir}")
        # 记录目录：不存在则递归创建
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        # 记录步数计数器，用于生成 step_xxx 文件名
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # 先正常执行一次推理
        results = self._policy.infer(obs)

        # 把输入输出打包成嵌套字典并拍平成 "a/b/c" 形式的扁平键
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        # 每次调用写入一个新的 step_xxx.npy 文件
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
