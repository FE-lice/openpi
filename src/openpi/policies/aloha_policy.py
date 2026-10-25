import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy.
    （构造一个随机的 Aloha 示例输入，用于测试/调试数据管线）
    """
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.
    （Aloha 策略的输入变换：把真实 Aloha 环境的原始观测，转换成模型需要的格式）

    期望的输入：
    - images: 字典 [相机名, 图像]，图像形状为 [通道, 高, 宽]，相机名必须在 EXPECTED_CAMERAS 中
    - state: [14] 维状态向量
    - actions: [action_horizon, 14] 动作序列（只在训练时提供）
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    # 若为 True，会把关节角与夹爪值从「标准 Aloha 空间」转换到「pi 内部运行时空间」（即预训练基础模型使用的坐标空间）。
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    # 期望的相机名称集合：所有输入相机都必须在此集合内； 缺失的相机会用全黑图像填充，并把对应的 image_mask 置为 False。
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        """执行变换：原始 Aloha 观测 -> 模型输入。"""
        # 1) 解码：翻转关节符号、把夹爪从线性空间转回角度空间、图像转成 HWC 布局
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        # 校验输入相机名是否合法，出现未知相机直接报错
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        # 假定基准相机 cam_high 一定存在，用它作为 base_0_rgb
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        # 添加两个腕部相机（左/右），并记录它们是否存在
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                # 存在则直接使用，并标记 mask 为 True
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                # 不存在则用黑图占位，并把 mask 置为 False（模型会忽略该视角）
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # 组装模型输入：图像 + 图像掩码 + 状态
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        # 动作只在训练数据里存在，推理时不会走到这里
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # 把动作从 pi 空间反变换回标准 Aloha 空间（用于训练目标对齐）
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        # 文本提示词直接透传
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy.
    （Aloha 策略的输出变换：把模型输出的动作转回真实 Aloha 环境可执行的格式）
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    # 若为 True，会在输出时把关节角与夹爪值从 pi 空间转回标准 Aloha 空间。
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        # 只取前 14 维（模型输出可能更宽，这里只保留 Aloha 需要的 7+7 维）
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles.
    （用于在 Aloha 与 pi 两套关节角约定之间做符号转换：
     元素为 -1 的关节需要取反，其余保持原值）
    """
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    # 线性归一化：把 [min_val, max_val] 映射到 [0, 1]
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    # 反归一化：把 [0, 1] 还原到 [min_val, max_val]
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    #
    # （把 Aloha 的线性夹爪位置反变换回角度空间:pi0 是在角度空间预训练的，所以要与其保持一致）
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    # （这是 Interbotix 代码中「角度 -> 线性」变换的逆运算）
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    # （以下常数取自 Interbotix 官方代码）
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    #
    # （pi0 的夹爪数据在编码器计数 (2405, 3110) 之间被归一化到 (0, 1)；
    #   编码器总计数为 4096，Aloha 的零点为 2048；换算成弧度后范围约为 (0.5476, 1.6296)）
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.
    #
    # （把 pi0 使用的夹爪位置转换回 Aloha 使用的夹爪位置；
    #   注意单位仍是角度，只是取值范围不同）

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    #
    # （不额外做缩放，因为 trossen 模型的预测本身就是弧度制；
    #   常数推导见 _gripper_to_angular）
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    # （这两个边界值来自 Aloha 代码中的 PUPPET_GRIPPER_JOINT_OPEN / CLOSE）
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    # （_gripper_from_angular 的逆函数，用于训练时把动作反变换回去）
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    #
    # （state 的组成：[左臂 6 个关节角, 左夹爪, 右臂 6 个关节角, 右夹爪]，
    #   各段维度为 [6, 1, 6, 1]，合计 14 维）
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        # 若图像是浮点型（通常已归一化到 [0,1]），乘 255 转成 uint8
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        # 把通道优先布局 [C, H, W] 转成 [H, W, C]
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    # 把 Aloha 原生状态转换到 pi 空间
    if adapt_to_pi:
        # Flip the joints.
        # 按符号掩码翻转需要取反的关节
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        # 撤销 Aloha 运行时对夹爪（第 6、13 维）做的线性化变换，转回角度空间
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    # 输出时：把 pi 空间的动作转换回标准 Aloha 空间，供真实机器人执行
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    # 训练时：把标准 Aloha 动作转换到 pi 空间，作为模型的学习目标
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
