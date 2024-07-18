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
csv_file_path1 = '../../data_record/2d_dqn.csv'
df1 = smooth(csv_file_path1, weight=0.6)
x_data1 = df1.iloc[1:, 0].astype(float)
y_data1 = df1.iloc[1:, 1].astype(float)
# 读取第二个CSV文件
csv_file_path2 = '../../data_record/2d_ppo.csv'
df2 = smooth(csv_file_path2, weight=0.6)
x_data2 = df2.iloc[1:, 0].astype(float)
y_data2 = df2.iloc[1:, 1].astype(float)
# 读取第一个CSV文件
csv_file_path3 = '../../data_record/2d_rppo.csv'
df3 = smooth(csv_file_path3, weight=0.6)
x_data3 = df3.iloc[1:, 0].astype(float)
y_data3 = df3.iloc[1:, 1].astype(float)
# 读取第一个CSV文件
csv_file_path4 = '../../data_record/2d_sac.csv'
df4 = smooth(csv_file_path4, weight=0.6)
x_data4 = df4.iloc[1:, 0].astype(float)
y_data4 = df4.iloc[1:, 1].astype(float)

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
plt.figure()
plt.plot(x_data1, y_data1, color='darkcyan', linewidth=2)
plt.plot(x_data2, y_data2, color='brown', linewidth=2)
plt.plot(x_data3, y_data3, color='royalblue', linewidth=2)
plt.plot(x_data4, y_data4, color='#424e66', linewidth=2)
plt.xlabel('时间步')
plt.ylabel('阶段平均奖励')
plt.title('奖励结果')

plt.legend(["DQN", "PPO", "RPPO", "SAC"])
plt.tight_layout()
plt.grid()
# 调整布局使得图像不溢出
plt.savefig('reward.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
