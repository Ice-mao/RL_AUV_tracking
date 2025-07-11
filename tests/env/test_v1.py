import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import argparse
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import auv_env
from config_loader import load_config

def test_gym_compliance():
    """
    测试 'v1-state-norender' 环境 (使用 WorldAuvV1) 是否符合 Gymnasium API 标准。
    """
    print("\n" + "="*40)
    print("  执行 Gymnasium 环境合规性测试  ")
    print("="*40 + "\n")
    print("正在创建环境 'v1-state-norender'...")
    # 'v1-state-norender' 在 auv_env/__init__.py 中注册，
    # 它内部使用 WorldAuvV1 并且禁用了渲染，适合自动化测试。
    env = gym.make("v1-state-norender")
    print("环境创建成功。")

    print("\n正在使用 check_env 工具检查环境...")
    try:
        # check_env 会对环境的 API、数据类型、空间等进行一系列检查
        check_env(env.unwrapped)
        print("\n✅ 成功: 环境 'v1-state-norender' 符合 Gymnasium API 标准！")
    except Exception as e:
        print(f"\n❌ 失败: 环境检查失败：{e}")
        print("环境不符合 Gymnasium API 标准。请根据错误信息进行修复。")
    finally:
        # 关闭环境以释放资源
        env.close()
        print("\n环境已关闭。")
        print("\n" + "="*40)
        print("         测试结束         ")
        print("="*40 + "\n")

def main():
    # 加载配置文件
    config_path = os.path.join('configs', 'v1_config.yml')
    config = load_config(config_path)
    
    # 使用加载的配置创建环境
    # 其他参数如 num_targets, eval 等现在从 config 文件中读取
    env = auv_env.make("AUVTracking_v1", 
                       config=config,
                       eval=True, t_steps=200,
                       show_viewport=True,
                       )

    obs, info = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminate, truncate, info = env.step(action)
        if terminate or truncate:
            obs, info = env.reset()
    env.close()

if __name__ == '__main__':
    main()