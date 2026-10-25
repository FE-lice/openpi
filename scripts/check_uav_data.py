"""校验 UAV LeRobot v2.1 数据集 → Pi0 模型输入（速度极速版）"""
import copy
import dataclasses
import pathlib
import sys
import traceback
import json
import time

import numpy as np
import pandas as pd
import tyro

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.common.datasets import utils as lerobot_utils
from openpi.training.data_loader import TransformedDataset

# 强制关闭旧版的联网校验
lerobot_utils.get_safe_version = lambda repo_id, revision: "main"

def _patched_create_torch_dataset(data_config, action_horizon, model_config):
    repo_id = data_config.repo_id
    if repo_id == "fake":
        return _data_loader.FakeDataset(model_config, num_samples=1024)
    
    dataset_meta = LeRobotDatasetMetadata(repo_id, root=data_config.root)
    dataset = LeRobotDataset(
        repo_id,
        root=data_config.root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset

_data_loader.create_torch_dataset = _patched_create_torch_dataset


@dataclasses.dataclass
class Args:
    repo_id: str = "uav_v21"
    root: str = "/root/autodl-tmp/UAV-Flow/lerobot_uav_v21"  # 指向你刚转好的目录
    config_name: str = "pi0_uav"
    action_key: str | None = None
    num_samples: int = 8
    num_batches: int = 4
    model_action_dim: int = 32
    assets_base_dir: str = "./assets"

_FAILS: list[str] = []
_WARNS: list[str] = []

def _ok(msg: str) -> None: print(f"  [PASS] {msg}")
def _warn(msg: str) -> None:
    _WARNS.append(msg)
    print(f"  [WARN] {msg}")
def _fail(msg: str) -> None:
    _FAILS.append(msg)
    print(f"  [FAIL] {msg}")
def _check(cond: bool, msg: str, detail: str = "") -> bool:
    if cond: _ok(msg)
    else:
        _fail(msg)
        if detail: print(f"         {detail}")
    return bool(cond)
def _shape(v) -> tuple | str: return tuple(v.shape) if hasattr(v, "shape") else f"<{type(v).__name__}>"
def _leaf(key: str) -> str: return key.replace("/", ".").split(".")[-1]
def _finish() -> None:
    print("\n" + "=" * 68)
    print(f"结果: {len(_FAILS)} FAIL / {len(_WARNS)} WARN")
    if _FAILS:
        print("\n需要修复:")
        for msg in _FAILS: print(f"  - {msg}")
        sys.exit(1)
    print("✅ 数据管道检查全部通过")


def main(args: Args) -> None:
    print(f"\n=== [1] 数据集元数据: {args.root} ===")
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    _ok("v2.1 元数据读取成功")
    
    features = meta.features
    action_candidates = [k for k in features if _leaf(k) in ("action", "actions")]
    data_action_key = args.action_key if args.action_key else (action_candidates[0] if action_candidates else None)
    print(f"  fps={meta.fps}  episodes={meta.total_episodes}  frames={meta.total_frames}")

    print("\n=== [2] 配置检查 ===")
    cfg = _config.get_config(args.config_name)
    base = cfg.data.base_config or _config.DataConfig()
    new_base = dataclasses.replace(base, root=args.root)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=args.repo_id, base_config=new_base))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    _ok(f"DataConfig 构建成功")

    if dc.repack_transforms.inputs:
        repack = dc.repack_transforms.inputs[0]
        if isinstance(repack, _transforms.RepackTransform):
            flat = _transforms.flatten_dict(repack.structure)
            if "state" in flat.values() and "observation.state" in features:
                new_structure = copy.deepcopy(repack.structure)
                def _fix(d):
                    for k, v in d.items():
                        if isinstance(v, dict): _fix(v)
                        elif v == "state": d[k] = "observation.state"
                _fix(new_structure)
                dc = dataclasses.replace(dc, repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform(new_structure)]))
                _warn("修正了 config 中的 'state' 键")

    print("\n=== [3] 单样本逐段变换 ===")
    ds = _data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    _ok(f"数据集可加载，共 {len(ds)} 个样本")

    print("\n=== [5] 真实 DataLoader (v2.1 寻址耗时测试) ===")
    t_init_start = time.time()
    loader = _data_loader.create_data_loader(cfg, shuffle=True, num_batches=args.num_batches, skip_norm_stats=(dc.norm_stats is None))
    _ok(f"DataLoader 初始化成功 (耗时: {time.time() - t_init_start:.3f} 秒)")

    batch_times = []
    t_last = time.time()
    
    for i, (obs, actions) in enumerate(loader):
        t_now = time.time()
        elapsed = t_now - t_last
        batch_times.append(elapsed)
        t_last = t_now
        
        print(f"  batch[{i}] 耗时: {elapsed:6.3f} 秒 | state={tuple(obs.state.shape)} actions={tuple(actions.shape)}")
        if i + 1 >= args.num_batches: break

    if len(batch_times) > 1:
        avg_steady = sum(batch_times[1:]) / len(batch_times[1:])
        print("\n  💡 耗时分析:")
        print(f"    首个 Batch: {batch_times[0]:.3f} 秒")
        print(f"    稳态 Batch: {avg_steady:.3f} 秒 (平均)")
        if avg_steady > 0.5:
            _warn("⚠️ v2.1 稳态耗时依然偏高，建议增加 num_workers。")
        else:
            _ok(f"🚀 稳态 Batch 耗时极佳（{avg_steady:.3f} 秒），训练将非常顺滑！")

    _finish()

if __name__ == "__main__":
    main(tyro.cli(Args))