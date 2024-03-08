import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
csv_file_path1 = 'data/DQN_1.csv'  # 替换为第一个CSV文件的路径
df1 = smooth(csv_file_path1, weight=0.6)
# df1 = pd.read_csv(csv_file_path1, header=None, skiprows=1)

# 提取第一个CSV文件的x和y轴数据
x_data1 = df1.iloc[1:, 0].astype(float)
y_data1 = df1.iloc[1:, 1].astype(float)

# # 读取第二个CSV文件
# csv_file_path2 = 'train_loss.csv'  # 替换为第二个CSV文件的路径
# df2 = pd.read_csv(csv_file_path2, header=None, skiprows=1)
#
# # 提取第二个CSV文件的x和y轴数据
# x_data2 = df2.iloc[1:, 1].astype(float)
# y_data2 = df2.iloc[1:, 2].astype(float)

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
plt.figure(1)
plt.plot(x_data1, y_data1, color='#424e66', linewidth=2)
plt.xlabel('timestep')
plt.ylabel('ep_rew_mean')
plt.title('reward_mean')

plt.tight_layout()
plt.grid()
# 调整布局使得图像不溢出
plt.savefig('reward.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
