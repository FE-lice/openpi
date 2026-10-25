import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:
    Token可以关注那些累积mask_ar小于等于自己的有效输入token实现多种不同的注意力掩码方式

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.   是图像的真实输入则为true
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on   注意力的实现方式
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)    #  将mask_ar广播到与input_mask相同的形状
    cumsum = jnp.cumsum(mask_ar, axis=1)                     #  cumsum计算mask_ar沿着N维度的前缀累积和
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]     
    # 首先多加入一个none维度，分别是[B,1,N]和[B,N,1]，做运算时会同时拉齐到[B,N,N]，但左边沿第二维拉齐右边沿第三维，最后比大小得到第i个q是否可以看到第j个k
    """左边（每行相同）：
[[1, 1, 2, 2],
 [1, 1, 2, 2],
 [1, 1, 2, 2],
 [1, 1, 2, 2]]
右边（每列相同）：
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [2, 2, 2, 2],
 [2, 2, 2, 2]]
 """
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  #同上，做逐元素乘法得到有效的注意力掩码
    return jnp.logical_and(attn_mask, valid_mask)     # 取交集
ll

@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction   #  与原论文10000方法略不同
    sinusoid_input = jnp.einsum(         #  两个一维序列所有元素乘起来得到[N,D]，N是pos的长度，D是embedding_dim/2
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    # Pi0Config加载配置（模型定义，输入形状检查，冻结器）
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        # 是否使用 pi05 版本
        self.pi05 = config.pi05

        # 读取语言模型和动作专家的配置，创建两个 Gemma 相关模块
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # TODO: rewrite gemma in NNX. For now, use bridge.
        # 暂时使用桥接方式把原来的Liner逻辑下的 Gemma 模型包装成 NNX 可管理模块
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],     # Module 的内部设计接收多个配置，在同一个模块树里创建多个子结构的模型路径
                embed_dtype=config.dtype,
                adarms=config.pi05,                    # Pi0.5 变体，启用 AdaRMS归一化 ，时间步t首先通过一个正弦位置编码，然后送入小的 MLP 网络，生成每层所需的调制参数
            )
        )
        # 对大语言模型初始化参数；若启用 pi05，则使用特殊 adarms 配置。
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        # 视觉编码器，用于从图像中提取特征。
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,   # 设置图像特征维度与语言模型宽度相匹配
                variant="So400m/14",                # 400M
                pool_type="none",                   # 不使用池化层，保留所有patch
                scan=True,                          # 启用扫描：不会for循环展开成所有的层结构，会在每个时间步重复使用同一层结构，然后传入参数，实现参数去堆叠而形状仅多一个层维度，节省内存
                dtype_mm=config.dtype,
            )
        )
        # 让图像编码器根据一个示例输入初始化参数
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # 将 llm 和 img 组合到一个字典里，方便后面统一访问。
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # 动作层
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)    # torch框架下会开始就self init 初始化，这里要显式调用rngs初始化

        # 根据 PI0 / PI05 版本选择不同的时间嵌入与状态处理路径
        # 时间步 t → 正弦编码 → time_mlp_in → 激活 → time_mlp_out → c  c 就是 AdaRMSNorm 的条件信号，状态和动作噪声走正常输入路径，和时间注入解耦
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # state层 ，另一个mlp
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            # h_action + h_time位置编码 → 维度上拼接在一起 → action_time_mlp_in → 激活 → action_time_mlp_out → h_cond
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # 输出投影层：将动作专家输出映射回 action_dim。
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # 这个属性会由 model.train() / model.eval() 自动设置，
        # 用于控制模型的确定性行为（如 dropout 关闭/开启）
        self.deterministic = True

    # 图像和文本的前缀embed，返回 token 序列、输入掩码和自回归掩码
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False) # PaliGemma.img 是SigLIP模型，返回[b, N, emb]

            tokens.append(image_tokens)     # 堆叠操作
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],  #  [batch,]，类型是 bool,对应相机是否有效
                    "b -> b s",             #  einops.repeat做维度扩展，batch不变，s即每个图像token的数量
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other  这是用于控制图像token间的注意力信息流动，因为图像token之间可以互相注意所以全部设为false不开启新块
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")  # obs的tokenized_prompt已经 token化并填充过了
            tokens.append(tokenized_inputs)                   # tokenized_inputs: [batch, max_token_len, emb]
            input_mask.append(obs.tokenized_prompt_mask)      # tokenized_prompt_mask: [batch, max_token_len]  填充位置要false
            # full attention between image and language inputs  同上
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)   # 序列维拼接
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    # 后缀处理：obs(一般包含图像和机器人状态)、noisy_actions、timestep
    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]    # 投影状态并添加序列维度
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))   # 添加状态掩码（全为1）
            # image/language inputs do not attend to state or actions
            ar_mask += [True]           # 第二块

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb            #  time_emb 作为 AdaRMSNorm 的条件信号
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)     # 重复时间嵌入以匹配动作序列长度 [b s emb]
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)          # emb 维度拼接
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)                                   # Swith激活？
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))             # 添加掩码（全为1）[B,ah]
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))                # 第一个为1
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)             # list.append 只是把元素加到列表末尾，不会自动拼接成数组
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond
    
    #loss 定义
    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)    # 分割随机数生成器为三部分，用于不同的随机操作，但基于同一个父key以复现
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)   # 返回处理过的 observation对象

        batch_shape = actions.shape[:-2]    # 去掉最后两维,形状为(B,)，赋值后成为1维列表
        noise = jax.random.normal(noise_rng, actions.shape)        # !!生成与动作形状相同的高斯噪声，每个样本ah的ad都不同 [b, ah, ad]
        # # 使用的是Beta分布采样时间不是均匀，（1.5, 1）对应f(x)=1.5*x^0.5(1−x)^0，偏向于1，pi0中t=0:数据，t=1:噪声，这样更多的训练样本会落在纯噪声附近
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001   # [B个time值]  [0.333,0.444....]
        time_expanded = time[..., None, None]    #  [B,1,1]，用于广播到动作维度   [[[0.333]]  [[0.444]] ......]
        x_t = time_expanded * noise + (1 - time_expanded) * actions   # 0->1，动作逐渐被噪声替代
        u_t = noise - actions    

        # 总结：批次不同，父key不同;样本轨迹不同，time不同;horizen动作序列不同，噪声不同但时间步相同         

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)        #  实现q  k之间的注意力流动关系
        positions = jnp.cumsum(input_mask, axis=1) - 1         #  ！！为每个有效 token 计算一个连续的位置索引

        # ！！！输入是两个序列的列表：[prefix_tokens, suffix_tokens]；输出也是两个序列的元组：(prefix_out, suffix_out)；adarms_cond 也是两个元素的列表：[None, adarms_cond]，模型内部不是将token简单拼接，而是共享注意力再过自己的FFN层
        # configs=[paligemma_config, action_expert_config]  llm定义时就使用了两个配置
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])    #  只取动作维度

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)   # 返回预测噪声和真实噪声之间的均方误差

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,       # 扩散过程的步数，默认为10步             
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # 默认从1到0是去噪，note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps      
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix  先单独运行一次前缀部分生成kv缓存
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry   # # carry数组包含当前状态和时间
            # 将时间广播到批次维度，并嵌入后缀（状态和动作）
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other  
            # 创建后缀内部的注意力掩码，形状为(批次, 后缀长度, 后缀长度
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            # 创建后缀对前缀的注意力掩码，形状为(批次, 后缀长度, 前缀长度)
            # 将 prefix_mask（图像文本是否有效）沿维度s复制到匹配suffix_tokens数，表示：第 b 个样本里，第 i 个 suffix token 能不能 attend 到第 j 个 prefix token
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            # 拼接掩码，形状为(批次, 后缀长度, 前缀长度+后缀长度)，控制后缀token（生成查询）如何关注完整序列（生成键和值）
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            # 计算后缀token的位置编码
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # ！！！！使用KV缓存进行高效的前向传递
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
             # 且确保前缀输出为None（因为使用了KV缓存）
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt   #  赋值到carry中

        def cond(carry):       # 迭代终止器，当time小于-dt/2=-0.05时返回False终止，保证最后一步能到0
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
