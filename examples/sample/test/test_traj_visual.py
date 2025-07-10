import numpy as np
import datasets
from imitation.data import huggingface_utils, rollout
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time
dataset_path = os.path.join("/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/")
dataset_0 = datasets.load_from_disk(dataset_path+"traj_1")
# dataset_1 = datasets.load_from_disk(dataset_path+"traj_2")
dataset = datasets.concatenate_datasets([dataset_0])
transitions = huggingface_utils.TrajectoryDatasetSequence(dataset)
time_1 = time.time()
obs = dataset_0[0]["obs"]
print("time:", time.time()-time_1)

transitions = rollout.flatten_trajectories(list(transitions))
transitions[0]
for i in range(len(transitions)):
    images = transitions[i]["obs"]
    if np.isnan(images).any():
        print("图像数据中存在 NaN！")
        nan_indices = np.where(np.isnan(images))
        if len(nan_indices[0]) > 0:
            print(
                f"NaN 出现在以下位置: 样本={nan_indices[0]}, 帧={nan_indices[1]},"
                f" 通道={nan_indices[2]}, 行={nan_indices[3]}, 列={nan_indices[4]}")
    if not np.isfinite(images).all():
        print("图像数据中存在 Inf 或 -Inf！")
    min_val = np.min(images)
    max_val = np.max(images)
    # 如果超出合理范围
    if min_val < -3 or max_val > 3:
        print("警告：像素值范围异常，可能未正确标准化！")

    actions = transitions[i]["acts"]
    if np.isnan(actions).any():
        print("动作数据中存在 NaN！")
        # 定位具体出问题的样本
        nan_actions = np.any(np.isnan(actions), axis=1)
        if np.sum(nan_actions) > 0:
            print(f"NaN 动作出现在样本: {np.where(nan_actions)[0]}")
    if not np.isfinite(actions).all():
        print("动作数据中存在 Inf 或 -Inf！")
    if actions.min() < -1 or actions.max() > 1:
        print(f"动作值超出 [-1,1]，实际范围: [{actions.min()}, {actions.max()}]")
    else:
        print(f"动作合理: [{actions.min()}, {actions.max()}]")
print("all right")

import numpy as np
import os
import matplotlib.pyplot as plt

# 假设 images 是形状为 (1200,5,3,224,224) 的 numpy 数组
num_groups = 20
save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

# 随机抽取组索引
group_indices = np.random.choice(len(transitions), size=num_groups, replace=False)
group_indices = np.arange(1, 200)

def denormalize_image(image):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    denorm_image = image * std + mean
    denorm_image = np.clip(denorm_image, 0, 1)
    denorm_image = (denorm_image * 255).astype(np.uint8)
    denorm_image = denorm_image.transpose(1, 2, 0)
    return denorm_image


for i, group_idx in enumerate(group_indices):
    group = transitions[int(group_idx)]["obs"]

    for t in range(5):
        frame = group[t]
        frame_rgb = denormalize_image(frame)
        save_path = os.path.join(save_dir, f"group_{i}_frame_{t}.png")
        plt.imsave(save_path, frame_rgb)

