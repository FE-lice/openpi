import json
import os
import gc
import tqdm
import numpy as np
import pandas as pd  
from datasets import load_dataset
from ledataset.datasets.lerobot_dataset import LeRobotDataset

def transform_to_local_frame(current_pose, next_pose):
    """
    核心动作生成逻辑：将全局坐标系下的位移，转换到无人机当前机头朝向的局部坐标系。
    输入位姿格式: [x, y, z, yaw(弧度)]
    """
    current_pos = current_pose[:3]
    current_yaw = current_pose[3]
    next_pos = next_pose[:3]
    next_yaw = next_pose[3]
    
    # 构建绕 Z 轴的 2D 旋转矩阵
    cos_yaw = np.cos(current_yaw)
    sin_yaw = np.sin(current_yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    
    # 计算全局相对位移，然后旋转到局部坐标系
    relative_pos = next_pos - current_pos
    local_pos = np.linalg.inv(R) @ relative_pos
    
    # 计算相对偏航角，并强制归一化到 [-pi, pi] 区间（处理 360 度跨越）
    relative_yaw = next_yaw - current_yaw
    relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi
    
    return np.array([local_pos[0], local_pos[1], local_pos[2], relative_yaw], dtype=np.float32)


def process_to_lerobot(parquet_path, output_dir):
    # 1. 配置 LeRobot 特征字典（⚠️ 核心修改：全部改为 4 维）
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "z", "yaw"]
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["dx_local", "dy_local", "dz_local", "dyaw_local"]
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"]
        }
    }

    # 2. 初始化 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id="uav_flow_local",
        root=output_dir,
        fps=5, 
        features=features,
        robot_type="uav",
        use_videos=True,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 3. 流式读取与轨迹拼接
    ds = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
    trajectory_dict = dict()

    for row in tqdm.tqdm(ds, desc=f"Processing {os.path.basename(parquet_path)}"):
        trajectory_id = row['id']
        frame_idx = row['frame_idx']
        
        if trajectory_id not in trajectory_dict:
            log = json.loads(row['log'])
            trajectory_dict[trajectory_id] = {
                'instruction': log.get('instruction_unified', log.get('instruction', '')),
                'states': log['preprocessed_logs'], # 原始 6 维
                'length': len(log['preprocessed_logs']),
                'images': {}
            }
        
        trajectory_dict[trajectory_id]['images'][frame_idx] = row['image']

        # 4. 凑齐一条轨迹后，开始处理
        if len(trajectory_dict[trajectory_id]['images']) == trajectory_dict[trajectory_id]['length']:
            imgs = [trajectory_dict[trajectory_id]['images'][k] for k in sorted(trajectory_dict[trajectory_id]['images'].keys())]
            raw_states = trajectory_dict[trajectory_id]['states']
            task = trajectory_dict[trajectory_id]['instruction']
            
            # --- 核心：提取 4维状态，并计算 4维局部动作 ---
            states_4d = []
            for s in raw_states:
                # 提取 [x, y, z, yaw]。根据源码，原 6 维中索引 4 是 yaw
                x, y, z, yaw = s[0], s[1], s[2], s[4]
                # 角度转弧度 (官方源码: trajectory_raw[:, 3] = np.deg2rad(...))
                yaw_rad = np.deg2rad(yaw)
                states_4d.append([x, y, z, yaw_rad])
            
            actions = []
            for i in range(len(states_4d)):
                if i < len(states_4d) - 1:
                    # 调用局部坐标系转换函数
                    action = transform_to_local_frame(states_4d[i], states_4d[i+1])
                else:
                    # 最后一帧没有下一帧，动作补 0
                    action = np.zeros(4, dtype=np.float32)
                actions.append(action)
            
            # 将轨迹逐帧喂入 LeRobot
            valid_len = min(len(imgs), len(states_4d), len(actions))
            for i in range(valid_len):
                frame_data = {
                    "observation.state": np.array(states_4d[i], dtype=np.float32),
                    "action": actions[i],
                    "observation.images.front": imgs[i],
                    "task": task
                }
                dataset.add_frame(frame_data)
            
            # 保存当前轨迹
            dataset.save_episode()
            
            # 清理内存，防止无卡模式 OOM
            trajectory_dict.pop(trajectory_id, None)
            del imgs, raw_states, states_4d, actions
            gc.collect()

    # ================= ⚠️ 关键补丁：生成兼容旧版 openpi 的 tasks.jsonl =================
    tasks_parquet_path = os.path.join(output_dir, "meta", "tasks.parquet")
    tasks_jsonl_path = os.path.join(output_dir, "meta", "tasks.jsonl")
    if os.path.exists(tasks_parquet_path):
        df = pd.read_parquet(tasks_parquet_path)
        with open(tasks_jsonl_path, 'w') as f:
            for idx, row in df.iterrows():
                task_str = row['task'] if 'task' in df.columns else row[df.columns[0]]
                task_idx = row['task_index'] if 'task_index' in df.columns else idx
                f.write(json.dumps({"task_index": int(task_idx), "task": str(task_str)}) + "\n")
        print("✅ tasks.jsonl 生成完毕！")
    # ==============================================================================

if __name__ == "__main__":
    parquet_file = '/root/autodl-tmp/UAV-Flow/train-00000-of-00054.parquet'
    output_dir = '/root/autodl-tmp/UAV-Flow/lerobot_uav_data_v3'
    
    process_to_lerobot(parquet_file, output_dir)