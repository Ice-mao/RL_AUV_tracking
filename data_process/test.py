import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
std_dev = np.array([0.1, 0.2, 0.1, 0.3, 0.15])  # 标准差

# 绘制数据点和均值线
plt.plot(x, y, linestyle='-', color='blue', label='Mean Line')

# 绘制标准差范围的背景
plt.fill_between(x, y - std_dev, y + std_dev, color='gray', alpha=0.5, label='Standard Deviation')

# 设置图例
plt.legend()

# 设置图形标题和轴标签
plt.title('Mean and Standard Deviation Plot')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.grid(True)
plt.show()
