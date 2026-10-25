"""
负责模型状态的保存和恢复，比如管理训练状态的序列化，包括：
    模型参数
    优化器状态
    EMA 参数（如果使用）

模型初始化流程	训练步骤流程	与 models 模块的交互	检查点管理流程
   init_train_state(config, rng, mesh)
   ├── 创建模型：model = config.model.create(rng)
   ├── 加载权重：partial_params = config.weight_loader.load(params)
   ├── 设置冻结参数：params = state_map(params, config.freeze_filter, ...)
   ├── 创建优化器：tx = create_optimizer(config.optimizer, config.lr_schedule)
   └── 返回 TrainState	   
   
   train_step(config, rng, state, batch)
   ├── 计算梯度：loss, grads = value_and_grad(model.compute_loss)()
   ├── 更新参数：updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
   ├── 应用更新：new_params = optax.apply_updates(params, updates)
   ├── 更新 EMA 参数（如果配置）
   └── 返回 new_state, info	   - 训练系统加载模型定义 (`BaseModel`)
   - 处理模型参数的保存和加载
   - 调用模型的 `compute_loss` 方法计算损失——详见上文的「1.2.4.3 损失函数 `compute_loss`」	  
    
    save_state(checkpoint_manager, state, data_loader, step)
   ├── _split_params(state) → 分离训练状态和推理参数
   ├── 保存归一化统计数据到 assets 目录
   └── checkpoint_manager.save() → 保存检查点
   
   restore_state(checkpoint_manager, state, data_loader)
   ├── checkpoint_manager.restore() → 恢复检查点
   └── _merge_params() → 合并恢复的参数
"""

from __future__ import annotations

import asyncio  
import concurrent.futures as futures  
import dataclasses 
import logging 
from typing import Protocol  

from etils import epath  
import jax  
import orbax.checkpoint as ocp  
import orbax.checkpoint.future as future 

from openpi.shared import array_typing as at  
import openpi.shared.normalize as _normalize  
import openpi.training.data_loader as _data_loader 
import openpi.training.utils as training_utils 


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    """初始化检查点目录，并返回检查点管理器！！与是否处于“恢复训练”状态！！

    参数:
        checkpoint_dir: 检查点目录路径。
        keep_period: 每隔多少步保留一个检查点（None 表示不周期保留）。
        overwrite: 若目录已存在，是否清空并覆盖。
        resume: 若目录已存在，是否从已有检查点继续训练。

    返回:
        (检查点管理器, resuming) 二元组，resuming 表示是否真的需要恢复检查点。
    """
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            # 覆盖模式：删除整个目录后重新创建
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            # 恢复模式：标记为需要恢复训练
            resuming = True
        else:
            # 目录已存在且未指定 overwrite / resume，抛出错误提示用户
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    # 确保目录存在（存在的目录不会报错）
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 创建Orbax 提供的检查点管理器  item_handlers：定义每个要保存的条目，以及用什么 handler 来保存/恢复  options：管理器的行为配置。
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            # assets：通过下面自定义保存归一化统计数据,配置信息等非Pytree资产
            "assets": CallbackHandler(),
            # train_state：JAX PyTree,优化器状态,当前 step,随机数生成器状态,其他训练相关变量
            "train_state": ocp.PyTreeCheckpointHandler(),
            # params：模型参数，可单独加载用于推理，与训练解耦
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,  # 最多保留 1 个最近的检查点
            keep_period=keep_period,  # 周期性保留间隔
            create=False,  # 目录已在上方手动创建，这里不再自动创建
            async_options=ocp.AsyncOptions(timeout_secs=7200),  # 异步保存超时（秒）
        ),
    )

    # 特殊情况：检查点目录已存在且用户请求恢复训练，但训练尚未保存出第一个检查点。
    # 此时不希望训练脚本尝试恢复检查点（因为会失败），故直接取消恢复。
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    """保存当前训练状态。

    会：1) 保存归一化统计数据到 assets；2) 分离可推理参数并分别保存 train_state 与 params。
    """
    def save_assets(directory: epath.Path):
        # 保存归一化统计数据（norm stats）到 directory / data_config.asset_id, norm_stats
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats) # 把计算好的 norm_stats序列化为一个标准的 JSON 文件，并保存到指定目录asset_id能区别数据集

    with at.disable_typechecking():
        train_state, params = _split_params(state)
    # manager 内各项对应的内容
    items = {
        "assets": save_assets,  # 资产由函数在保存时生成
        "train_state": train_state,  # 训练状态
        "params": {"params": params},  # 推理参数
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    """从检查点恢复训练状态，并合并回可推理参数。"""
    del data_loader  # 恢复时不需要数据加载器

    with at.disable_typechecking():
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    # 将恢复的 train_state 与 params 合并为完整的 TrainState
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    """从 assets 目录加载指定资产 ID 的归一化统计数据。"""
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    """回调协议：接收一个目录路径，执行自定义保存逻辑。"""

    def __call__(self, directory: epath.Path) -> None: ...

"""
配置阶段：
  CheckpointManager(item_handlers={"assets": CallbackHandler(), ...})
      ↓
  告诉 Orbax：以后 items 里 key 为 "assets" 的东西，交给 CallbackHandler 处理

保存阶段：
  save_state(...)
      ↓
  items = {"assets": save_assets, "train_state": ..., "params": ...}
      ↓
  checkpoint_manager.save(step, items)
      ↓
  Orbax 路由：
    "assets" → CallbackHandler → 异步线程 → 主进程执行 save_assets(directory)
                                                  ↓
                                          _normalize.save(...)
                                                  ↓
                                          写 norm_stats.json
"""
class CallbackHandler(ocp.AsyncCheckpointHandler):
    """以异步方式调用任意函数的 CheckpointHandler。仅用于保存，不支持恢复。"""

    def save(self, directory: epath.Path, args: CallbackSave):
        # 仅由主进程执行保存，避免多进程重复写入
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        # 将同步 save 放到独立线程执行，并以 Future 形式返回
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        # 该处理器不支持恢复
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    """保存回调参数：包含待执行的 callback 函数。"""
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    """分离训练状态与可推理参数。

    若存在 EMA 参数，则把 EMA 参数作为推理参数，并从 train_state 中清空 ema_params；
    否则把普通 params 作为推理参数，并从 train_state 中清空 params。
    """
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    """合并回 _split_params 拆分出的参数。

    依据 _split_params 的逻辑：若 train_state.params 非空，说明拆分时使用的是 EMA 参数，
    因此合并时应恢复为 ema_params；否则恢复为普通 params。
    """
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
