import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)  类型检查器，强调“必须是这个上界联合类型的子类型
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):      #  模型种类赋值
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images  三相机
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.  标准分辨率
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):  # 存储和管理模型的输入数据
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images图像数据类型注解, 不符静态检查器会提示  in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.默认赋值None
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod    #  从Pytree字典数据创建结构化的 `Observation` 对象以在内部运算
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":   # cls 类方法而不是实例方法，Pytree[ArrayT]类型注解，表示数据是一个符合ArrayT的树状结构
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together. fast方法两个同时存在
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8 [0,255], convert them to [-1, 1] float32.  key是’image‘下的子键
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:    # 加了个数据有dtype属性的判断，防止报错
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(         # 从字典数据创建结构化的 `Observation` 对象 
            images=data["image"],  
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:   #  从内部对象创建Pytree字典数据，方便保存和传输到外部
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)         #   重新转为字典结构
        result["image"] = result.pop("images")    #   重命名
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.   ah` 表示动作时间步长，`ad` 表示每个动作的维度
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,   
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    # 先校验所有必须的相机图像是否都在 observation.images 中；  # set是必须的图像键集合，issubset 检查前者是否是后者的子集
    if not set(image_keys).issubset(observation.images):    
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    #.shape[:-1] [*b,s] ->  一个新元组[*b,]
    batch_shape = observation.state.shape[:-1]

    # out_images 用于收集处理后的所有图像，并保留原始图像 key -> 处理后图像的映射。
    out_images = {}
    for key in image_keys:
        # 取出当前相机对应的图片张量。
        image = observation.images[key]

        # 统一到标准大小224  # [1:3]是索引1,2
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)     # 采用等比例缩放后再零填充

        # 训练模式下，做数据增强
        if train:
            # augmax 需要图像在 [0, 1] 区间；原始数据通常是 [-1, 1]，所以先做转换
            image = image / 2.0 + 0.5

            # 构造增强链：范围形式
            # - 非 wrist 相机: 位置随机裁剪 + 重缩放 + 随机角度旋转
            # - 所有图像: 颜色抖动（亮度/对比度/饱和度）
            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),   
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]

            # rng是父key,假设（2,）:[0 0],由它决定jax.random.split生成结果
            # 但split方法会依靠父keyz在batch个子key上生成新的随机数序列, 如[0,3242,0.3556] [0.3278, 0.5678] ....
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)     
            # vmap是并行批处理, 传入的sub_rngs和image按batch维度拆分, 每个子key对应一张图像, 依次调用augmax.Chain(*transforms)做数据增强

            # 最后转换回原始的 [-1,1] 范围，保证后续模型输入格式一致
            image = image * 2.0 - 1.0

        out_images[key] = image

    # 计算图像 mask：
    # - 如果某个图像没有对应 mask，则默认该图像有效（全为 True）
    # - 如果有 mask，则直接保留原始 mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    # 返回一个新的 Observation 对象
    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    初始化，加载，输入规格基础定义
    """

    # 动作空间维度
    action_dim: int
    # 动作块序列长度
    action_horizon: int
    # 语言 prompt 的最大 token 长度。
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""
        # 这个属性要求每个具体模型都声明自己属于哪一种模型类型 PI0 / PI0_FAST / PI05 

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""
        # 这个抽象方法要求所有子类实现“如何创建并初始化一个模型实例”，通常会在训练开始时被调用，以生成一个<新模型对象>

    #  从纯参数树恢复模型,用来恢复 checkpoint
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        # 传入create方法，eval_shape 创建模型骨架，但无数据
        model = nnx.eval_shape(self.create, jax.random.key(0))
        # 将骨架分为  静态结构（层、连接关系）  与  动态参数  两部分
        graphdef, state = nnx.split(model)
        # 过滤掉与当前模型结构不匹配的键名，避免导致加载失败
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        # 检查两棵参数树结构是否一致；check_shapes=True形状必须匹配，之后传入
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    # PyTorch模型参数加载方法？？
    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        # 加载 PyTorch 版本模型，通常用于从 safetensors 权重恢复模型
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""
        # 这个方法返回模型输入输出的“形状-类型规格”，
        # 例如 observation 的每个字段 shape 和 dtype，以及 action 的 shape
        # 这样可以在训练前做静态检查，不需要先构造真实样本

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        # 生成一个虚拟观察值，用于快速测试模型的输入结构是否合理
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        # 生成一个虚拟动作输出，用于快速测试模型输出形状是否匹配
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    # 模型的动作维度
    action_dim: int
    # 模型预测的动作序列长度
    action_horizon: int
    # 语言 token 的最大长度
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...
        # 这个抽象方法定义：模型如何根据 observation 和 action 计算 loss。
        # 训练阶段通常会调用它来更新参数；train 参数表示当前是否启用训练增强/训练模式。

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...
        # 这个抽象方法定义：给定当前观测，模型如何采样下一步动作
        # 例如在推理或 rollout 时调用，用于生成 robot 需要执行的动作序列


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.
    拿到参数树,与上面的load方法配合使用
    参数：
    params_path:本地检查点目录的路径
    restore_type:参数恢复的类型。可设置为`np.ndarray`以将参数加载为numpy数组
    dtype:用于恢复所有参数的数据类型。若未指定，将使用检查点中的原始数据类型
    sharding:用于参数的分片方式。若未指定，则参数将在所有设备上复制

    Returns:
        The restored params.
    """
    # 如果传入的是本地路径，则转成绝对路径；如果是 gs:// 路径，则保持原样。
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    # 默认情况下，如果恢复成 JAX Array 且没有指定 sharding，
    # 就把模型参数复制到整个设备 mesh 上，避免参数只落在单机单卡上。
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 使用 Orbax 的 PyTreeCheckpointer 来读取 checkpoint，
    # 这里的 metadata["params"] 表示参数树所在的位置。
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        # 执行恢复：指定 item 和每个数组的 restore 参数。
        # 这里将 params 整体恢复为一个 PyTree，其中每个张量都设置了 sharding / dtype / restore_type。
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # 如果参数是通过 nnx.State.save_state 保存出来的，
    # 那么每个 key 路径末尾通常会带一个 "value".
    # 例如 {'layer': {'kernel': {'value': ...}}} 这种结构。
    # 这不是纯粹的参数字典，所以这里删除尾部 "value"，统一返回 NNX 习惯的 pure dict 结构。
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
