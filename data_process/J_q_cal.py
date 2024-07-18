import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = [0.02, 0.1, 0.5, 1.0, 2.0, 5.0]
y_1 = [0.167224183, 0.395629721,1.061060811,1.702206201,1.700513958,1.249746734]
error_1 = [0.1, 0.2, 0.1, 0.3, 0.15, 0.1]  # 误差

# 绘制数据图
ax1.errorbar(x, y_1, yerr=error_1, fmt='o', capsize=5, color='royalblue', label='RL_AUV_tracking')
ax1.plot(x, y_1, linestyle='--', color='royalblue', label='RL_AUV_tracking')

y_2 = [0.241126819,0.424053521,0.70662138,0.876630481,1.596788096,0.677]
error_2 = [0.081553275,0.189102219,0.510244017,0.712386897,0.903725476,1.419955454] # 误差
# 绘制数据图
ax1.errorbar(x, y_2, yerr=error_2, fmt='o', capsize=5, color='goldenrod', label='Greedy_50')
ax1.plot(x, y_2, linestyle='--', color='goldenrod', label='Greedy_50')

y_3 = [0.247992264,0.537910746,0.910986942,1.262397513,1.422526791,2.369115856]
error_3 = [0.112482451,0.221193705,0.533381092,0.672546518,0.925056143,1.21210147]  # 误差
# 绘制数据图
ax1.errorbar(x, y_3, yerr=error_3, fmt='o', capsize=5, color='gray', label='Greedy_500')
ax1.plot(x, y_3, linestyle='--', color='gray', label='Greedy_500')
ax1.set_ylabel('J')


t_1 = [0.802, 0.937, 0.981, 0.9955, 0.8635, 0.677]
t_2 = [0.847, 0.8075, 0.669, 0.659, 0.8095, 0.6785]
t_3 = [0.803, 0.8395, 0.771, 0.8135, 0.768, 0.8445]
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
