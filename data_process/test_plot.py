import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np

# 创建一个旋转的 shapely Polygon 对象
rectangle = ShapelyPolygon([(1, 1), (3, 1), (2, 3)])

# 提取旋转矩形的顶点坐标
x, y = rectangle.exterior.xy

# 创建 matplotlib 中的旋转矩形对象
polygon_patch = Polygon(np.column_stack((x, y)), closed=True, edgecolor='black', facecolor='blue')

# 创建图形并添加旋转矩形
fig, ax = plt.subplots()
ax.add_patch(polygon_patch)

# 设置坐标轴范围
ax.set_xlim(min(x)-0.1, max(x)+0.1)
ax.set_ylim(min(y)-0.1, max(y)+0.1)

# 显示图形
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rotated Rectangle from Shapely Polygon')
plt.grid(True)
plt.axis('equal')
plt.show()
