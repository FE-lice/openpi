import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """LoRA 低秩适配的配置参数。"""

    # LoRA rank.
    rank: int
    # LoRA 更新量的缩放系数 alpha。
    alpha: float = 1.0
    # LoRA 低秩参数的初始化方法    参数接收一个「初始化函数」，默认用均值 0、标准差 0.01 的高斯正态分布来初始化权重
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # 是否启用 rank-stabilized LoRA；启用时缩放分母使用 sqrt(rank)。
    rslora: bool = False
    # 张量形状: (A, B, C, D)  最后两个轴应用低秩分解   (B, H, m, n)  →  (B, H, m, r) @ (B, H, r, n)
    axes: tuple[int, int] = (-2, -1)
    # einsum 方法中代表 LoRA 低秩中间维度的标签，？
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        # 标准 LoRA 使用 alpha/r；RS-LoRA 使用 alpha/sqrt(r) 调整更新幅度
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank

# torch.einsum 是用一套统一的索引符号来表达更一般的张量收缩（如不在后两维度的矩阵乘法，外积....），这里扩展了支持lora的einsum方法
class Einsum(nn.Module):  
    """核心思路是：原始权重 w 不动，额外挂两个低秩矩阵 w_a、w_b训练时只更新它们"""

    # 主权重张量的形状。
    shape: tuple[int, ...]
    # 主权重张量的初始化方法
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # -
    lora_config: LoRAConfig | None = None

    # 初始化方法:setup()函数会在模型实例化后被调用，用于定义模型的参数和子模块，将它们注册到Flax的参数树中，以便在训练和推理过程中使用。
    def setup(self):
        # 创建主权重原始值
        self.w = self.param("w", self.init_fn, self.shape)

        # :=海象运算符，执行true判断，并将表达式的结果赋值给变量,这里是对于lora分支的低秩矩阵
        if config := self.lora_config:
            shape_a, shape_b = list(self.shape), list(self.shape)
            # 在指定权重轴插入 rank 维度，A[...,ij]->[...,i,r]    B[...,jk]->[...,r,k] 
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)   # 显式加上lora命名用于filter
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision

        # 先根据原始 einsum 方程计算主权重的输出，...ij,...jk->...ik
        # 回顾下原始方法：字母顺序代表维度，前后相同字母会进行对应相乘，若字母在输出中不出现相应地再会被求和消除，逗号分隔不同的输入张量，可以理解为一个循环加求和！！
        # 例如:torch.einsum('bij,bjk->jk', A, B)  等价于C[j, k] = Σ_b Σ_i  A[b, i, j] * B[b, j, k]
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        # lora分支
        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            # 相当于x与AB的连乘运算得到lora部分输出
            # ...ij,...jL->...iL  
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            # ...iL,...Lk->...ik  
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            # 主权重输出是冻结的lora部分是不断更新的
            result = result + lora * config.scaling_value

        return result

    # eqn为主权重W和输入X 的einsum计算方法，这里将它f分别分解为 A,B 和输入Xf对应的einsum计算方法
    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        # L 保留给低秩中间维度使用，因此原方程不能已经包含该标签
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        # 要求符合标准的“输入,权重->输出” einsum 方程，同时赋值了m
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        # 使用grouos方法提取对应字符，  lhs对应 “...ij”  rhs对应 “...jk”  out对应 “...ik”
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        # a_label :"j", b_label  : "k"
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label   #  "L"

        # ...ij,...jL->...iL  wA对应运算部分
        a_rhs = rhs.replace(b_label, label)   # 把“k”换成“L”
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"


        # ...iL,...Lk->...ik  wB对应运算部分
        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """门控mlp:GeGLU,在输入端会多带一个线性层输出门控值,这种模式hiddendim更小
       带可选 LoRA 更新的前馈网络模块"""

    features: int    # x的最后一位特征维
    hidden_dim: int  # mlp隐藏维
    lora_config: LoRAConfig | None = None

    def setup(self):
        # 两组里层输入对应权重：一组产生 GeGLU 门控值，另一组产生待门控特征   (2,D,hiddendim)
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        # 外层权重
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # 分别为门控投影和最后的线性投影创建低秩 A/B 参数  （2,D,hiddendim）-> (2,D,r) @ (2,r,hiddendim)
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  
        # 第一组投影经 GELU 激活后s输出门控值，与第二组投影逐元素相乘。
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        # 核心公式
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        assert outputs.dtype == dtype
        return outputs
    
    # 修改成带lora分支的前向矩阵乘法，先计算基础线性层；若提供 LoRA 权重，则再计算 x @ A @ B 并叠加
    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
