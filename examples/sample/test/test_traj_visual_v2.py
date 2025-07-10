import numpy as np
import datasets
from imitation.data import huggingface_utils, rollout
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time
import cv2

# dataset_path = os.path.join("/data/log/sample/trajs_dam_v2/traj_2")
dataset_path = os.path.join("/data/log/sample/trajs_openwater/")
dataset_0 = datasets.load_from_disk(dataset_path+"traj_0")
# dataset_1 = datasets.load_from_disk(dataset_path+"traj_3")
dataset = datasets.concatenate_datasets([dataset_0])
transitions = huggingface_utils.TrajectoryDatasetSequence(dataset)

time_1 = time.time()
obs = dataset_0[0]["obs"]
print("time:", time.time()-time_1)

test_path = os.path.join(dataset_path, "test")
test_path_left = os.path.join(test_path, "left")
test_path_right = os.path.join(test_path, "right")
test_path_sonar = os.path.join(test_path, "sonar")
os.makedirs(test_path, exist_ok=True)
os.makedirs(test_path_left, exist_ok=True)
os.makedirs(test_path_right, exist_ok=True)
os.makedirs(test_path_sonar, exist_ok=True)

transitions = rollout.flatten_trajectories(list(transitions))
print(len(transitions))
for i in range(len(transitions)):
    flat_images = transitions[i]["obs"]
    # 计算每个图像的元素数量
    left_size = 3 * 224 * 224
    right_size = 3 * 224 * 224
    sonar_size = 1 * 128 * 128
    state_size = 1
    # 检查输入数据是否符合预期大小
    if len(flat_images) != left_size + right_size + sonar_size + state_size:
        raise ValueError(f"输入数据大小不匹配: 预期 {left_size + right_size + sonar_size}, 实际 {len(flat_images)}")
    
    # 切分并重塑数据
    left_camera = flat_images[0:left_size].reshape(3, 224, 224)
    right_camera = flat_images[left_size:left_size+right_size].reshape(3, 224, 224)
    sonar = flat_images[left_size+right_size:left_size+right_size+sonar_size].reshape(1, 128, 128)

    left_camera_cv = left_camera.transpose(1, 2, 0).astype(np.uint8)
    left_camera_cv_bgr = cv2.cvtColor(left_camera_cv, cv2.COLOR_RGB2BGR)
    right_camera_cv = right_camera.transpose(1, 2, 0).astype(np.uint8)
    right_camera_cv_bgr = cv2.cvtColor(right_camera_cv, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(test_path_left, f"sample_{i}.jpg"), left_camera_cv_bgr)
    cv2.imwrite(os.path.join(test_path_right, f"sample_{i}.jpg"), right_camera_cv_bgr)

    print("sonar max:", sonar.max())
    sonar_data = sonar[0].astype(np.uint8)
    cv2.imwrite(os.path.join(test_path_sonar, f"sample_{i}.jpg"), sonar_data)



import matplotlib.pyplot as plt
import numpy as np
import os
import time

def visualize_sonar_polar(sonar_data, save_path, frame_index=0):
    """
    将声呐数据可视化为极坐标图并保存
    
    参数:
        sonar_data: 形状为(1,128,128)的声呐数据
        save_path: 保存图像的路径
        frame_index: 帧索引（用于文件名）
    """
    # 确保数据形状正确
    if sonar_data.shape != (1, 128, 128):
        raise ValueError(f"声呐数据形状应为(1,128,128)，实际为{sonar_data.shape}")
    
    # 提取单通道数据
    sonar = sonar_data[0]
    
    # 初始化极坐标参数（基于HoloOcean示例）
    azi = 120  # 方位角范围（度）
    minR = 1  # 最小距离
    maxR = 50  # 最大距离
    binsR = 128  # 距离采样点数
    binsA = 128  # 方位角采样点数
    
    # 创建图形
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 6))
    ax.set_theta_zero_location("N")
    ax.set_thetamin(-azi/2)
    ax.set_thetamax(azi/2)
    
    # 创建网格
    theta = np.linspace(-azi/2, azi/2, binsA) * np.pi/180
    r = np.linspace(minR, maxR, binsR)
    T, R = np.meshgrid(theta, r)
    
    # 标准化声呐数据到[0,1]范围
    normalized_data = sonar.astype(np.float32)/225
    
    # 绘制热图
    plot = ax.pcolormesh(T, R, normalized_data, cmap='gray', shading='auto', vmin=0, vmax=1)
    plt.grid(False)
    plt.colorbar(plot, ax=ax, label="Intensity")
    plt.title("Sonar Visualization (Polar Coordinates)")
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_path, f"sonar_polar_{frame_index}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return os.path.join(save_path, f"sonar_polar_{frame_index}.png")

# 使用示例
def save_sonar_visualizations(transitions, sonar_path):
    """保存所有声呐数据的可视化"""
    os.makedirs(sonar_path, exist_ok=True)
    
    for i in range(len(transitions)):
        start_time = time.time()
        
        # 获取展平数据并重塑
        flat_images = transitions[i]["obs"]
        left_size = 3 * 224 * 224
        right_size = 3 * 224 * 224
        sonar_size = 1 * 128 * 128
        
        # 提取并重塑声呐数据
        sonar = flat_images[left_size+right_size:].reshape(1, 128, 128)
        
        # 生成两种可视化: 极坐标和常规灰度图
        # 1. 极坐标可视化
        polar_path = visualize_sonar_polar(sonar, sonar_path, i)
        
        # 2. 常规灰度图可视化 (简单方法)
        sonar_normalized = (sonar[0] - sonar[0].min()) / (sonar[0].max() - sonar[0].min() + 1e-8)
        sonar_uint8 = (sonar_normalized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(sonar_path, f"sample_{i}.jpg"), sonar_uint8)
        
        print(f"处理帧 {i}, 耗时: {time.time() - start_time:.4f}秒")
    
    print(f"所有声呐图像保存到: {sonar_path}")



    







