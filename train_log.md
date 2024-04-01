# record of train

## 3.20训练
state_space:目标状态估计+agent状态估计+最近障碍物信息
action_space:生成局部路径点的均值和标准差
reward_space:协方差+碰撞惩罚

操作：把相机和渲染关闭，并使用cpu进行训练

### 训练过程问题
- reward中要增加对转动角度的限制因子
