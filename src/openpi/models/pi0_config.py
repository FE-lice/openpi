import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    # 模型运行时使用的dtype
    dtype: str = "bfloat16"
    # 视觉语言模型使用的 Gemma 变体。
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # 动作专家使用的 Gemma 变体。
    action_expert_variant: _gemma.Variant = "gemma_300m"


    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore

    # Pi05 与 Pi0 的两个关键差异：
    # 1. 状态输入被当作离散语言 token 一部分，而不是连续 suffix 输入；
    # 2. 动作专家使用 adaRMSNorm 注入 flow matching timestep。
    pi05: bool = False
    # 这个配置项本身不直接参与模型计算，但会被 ModelTransformFactory 读取，
    # 用来决定是否将状态当作离散输入进行处理。
    discrete_state_input: bool = None  # type: ignore

    # PyTorch 编译模式；如果不为 None，则必须是允许的编译模式之一。
    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        # 如果没有显式设置 max_token_len，则根据版本选择默认值。
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)

        # 如果 discrete_state_input 未指定，则默认和 pi05 保持一致。
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

        # 检查 PyTorch 编译模式是否合法，避免传入不支持的字符串。
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # 根据是否启用 pi05，返回对应的模型类型。
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        # 按当前配置构造 Pi0 模型，并为其初始化随机数生成器。
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # 每个相机的输入张量形状： [batch, H, W, 3]，类型是 float32。
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        # 缺失相机的图像做mask，形状： [batch]，类型是 bool。
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        # 关闭类型检查以便构造 Observation 的 spec，避免过于严格的静态类型约束，
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                # 状态输入 shape: [batch, action_dim]
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                # tokenized_prompt shape: [batch, max_token_len]
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )

        # 动作输出的 spec: [batch, action_horizon, action_dim]
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        # 该方法用于设计“冻结哪些参数”的过滤器，通常用于微调/LoRA 场景
        filters = []
        has_lora = False

        # NNX 模型被 nnx.split 后，参数是一个嵌套的PyTree（字典）
        # 匹配 Gemma 主干部分参数，路径中有llm的，两大块模型都包含了
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        # 匹配 action expert 相关参数，通常对应 llm且含_1 这一层结构
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")

        if "lora" in self.paligemma_variant:   # paligemma_variant字符串上面定义，含lora则使用lora
            # 如果 paligemma 使用 LoRA，则添加 Gemma 主干参数。
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # 若 action expert 没有使用 LoRA，则要额外排除 action expert 参数
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            # 如果 action expert 使用 LoRA，则冻结该部分参数。
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # 如果存在 LoRA，则排除所有 LoRA 参数本身，避免它们也被冻结。
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        # 如果没有任何冻结过滤条件，则不冻结任何参数。
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)   # 所有filters取交集，
