import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')  # 或其他风格
import os
def smooth(csv_path,weight=0.85):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['None','Step','Value'])
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    return save
    # save.to_csv('smooth_'+csv_path)

# 读取第一个CSV文件
csv_file_path1 = '../../data_record/sac.csv'
df1 = smooth(csv_file_path1, weight=0.6)
x_data1 = df1.iloc[1:, 0].astype(float)
y_data1 = df1.iloc[1:, 1].astype(float)

# 设置绘图风格，使用科学论文常见的线条样式和颜色
print(plt.style.available)
plt.style.use('seaborn-v0_8-bright')

# 设置字体和字号
font = {'family': 'sans-serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        }
plt.rc('font', **font)

# 绘制第一幅图像
fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True)
axs[0].plot(x_data1, y_data1, color='darkcyan', linewidth=2)
axs[0].set_ylabel('平均奖励')

x = np.array([0, 120000, 240000, 360000, 480000])
y1 = np.array([0, 0.250123825,1.035009114,0.782395382,0.817854787])
std1 = np.array([0, 0.340349493,0.3244951,0.409584298,0.346551393])
axs[1].plot(x, y1, linestyle='-', color='blue', label='Mean Line')
axs[1].fill_between(x, y1-std1, y1 + std1, color='blue', alpha=0.2, label='Standard Deviation')
axs[1].set_ylabel('J')

y2 = np.array([0, 0.4132,0.92,0.8104,0.8517])
std2 = np.array([0, 0.311300112,0.240008333,0.309163128,0.261649785])
axs[2].plot(x, y2, linestyle='-', color='gray', label='Mean Line')
axs[2].fill_between(x, y2-std2, y2 + std2, color='gray', alpha=0.2, label='Standard Deviation')
axs[2].set_ylabel('视野内时间')

y3 = np.array([0, 0.0409,0.0656,0.0657,0.0304])
std3 = np.array([0, 0.066020376,0.0851985657,0.051350581,0.025451939])
axs[3].plot(x, y3, linestyle='-', color='goldenrod', label='Mean Line')
axs[3].fill_between(x, y3-std3, y3 + std3, color='goldenrod', alpha=0.2, label='Standard Deviation')
axs[3].set_ylabel('碰撞率')

axs[-1].set_xlabel('时间步')

# plt.legend(["Reward", "PPO", "RPPO"])
plt.tight_layout()
plt.grid()
# 调整布局使得图像不溢出
plt.savefig('reward.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
