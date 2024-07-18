import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = [20,50,100,150,200]
y_1 = [1.122243674,0.985509598,0.773531469,0.638185617,0.690832139]
error_1 = [0.10528848,0.298536017,0.350118492,0.435033094,0.431219645]  # 误差

# 绘制数据图
ax1.errorbar(x, y_1, yerr=error_1, fmt='o', capsize=5, color='royalblue', label='RL_AUV_tracking')
ax1.plot(x, y_1, linestyle='--', color='royalblue', label='RL_AUV_tracking')

y_2 = [0.854879514,0.586442136,0.629314479,0.471551671,0.747944887]
error_2 = [0.514110513,0.428479877,0.407682844,0.352469024,0.413395247] # 误差
# 绘制数据图
ax1.errorbar(x, y_2, yerr=error_2, fmt='o', capsize=5, color='goldenrod', label='Greedy_50')
ax1.plot(x, y_2, linestyle='--', color='goldenrod', label='Greedy_50')

y_3 = [1.016417901,0.973250249,0.628638047,0.611861194,0.762327101]
error_3 = [0.420374443,0.46115359,0.490057424,0.528638271,0.424663826]  # 误差
# 绘制数据图
ax1.errorbar(x, y_3, yerr=error_3, fmt='o', capsize=5, color='gray', label='Greedy_500')
ax1.plot(x, y_3, linestyle='--', color='gray', label='Greedy_500')
ax1.set_ylabel('J')


t_1 = [0.976,0.937,0.816,0.6585,0.73]
t_2 = [0.766,0.6545,0.7275,0.6185,0.7765]
t_3 = [0.874,0.868,0.679,0.6395,0.796]
ax2.plot(x, t_1, linestyle='-', color='royalblue', label='RL_AUV_tracking', marker='o')
ax2.plot(x, t_2, linestyle='-', color='goldenrod', label='Greedy_50', marker='o')
ax2.plot(x, t_3, linestyle='-', color='gray', label='Greedy_500', marker='o')
# 设置第二个 y 轴的刻度标签和范围
ax2.set_ylabel('insight time')
ax2.set_ylim(0, 1)


plt.xscale('log')
plt.xticks(x, [str(val) for val in x])
ax2.legend(['RL_AUV_tracking', 'Greedy_50', 'Greedy_500'], loc='lower left')

# 设置图形标题和轴标签
plt.title('')

# 显示图形
ax1.grid(True)
plt.show()
