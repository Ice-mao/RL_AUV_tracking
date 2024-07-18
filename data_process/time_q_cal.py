import matplotlib.pyplot as plt

x = [0.02, 0.1, 0.5, 1.0, 2.0, 5.0]
y_1 = [0.802,0.937,0.981,0.9955,0.8635,0.677]

# 绘制数据图
plt.plot(x, y_1, linestyle='-', color='royalblue', label='RL_AUV_tracking')

y_2 = [0.847,0.8075,0.669,0.659,0.8095,0.6785]
# 绘制数据图
plt.plot(x, y_2, linestyle='-', color='goldenrod', label='Greedy_50')

y_3 = [0.803,0.8395,0.771,0.8135,0.768,0.8445]
# 绘制数据图
plt.plot(x, y_3, linestyle='-', color='gray', label='Greedy_500')


plt.xscale('log')
plt.xticks(x, [str(val) for val in x])
plt.ylim(0, 1)
plt.legend(['RL_AUV_tracking', 'Greedy_50', 'Greedy_500'])

# 设置图形标题和轴标签
plt.title('')
plt.xlabel('q')
plt.ylabel('J')

# 显示图形
plt.grid(True)
plt.show()
