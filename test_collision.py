from shapely.geometry import Point, Polygon
from auv_env.obstacle import polygon_2_points
#
#
# # 定义点
# point1 = Point(0, 0)
# point2 = Point(0, 10)
# point3 = Point(10, 10)
# point4 = Point(10, 0)
# # 定义多边形
# # polygon = Polygon([point1, point2, point3, point4])
# point = Point(5, 2)
#
# points = polygon_2_points([0, 0], [8, 8, 1], 1, [0, 0], 45)
# print(points)
# polygon = Polygon(points)
# print(polygon.boundary)
#
# # 检测点是否在多边形内
# if polygon.contains(point):
#     print("Point is inside the polygon!")
# else:
#     print("Point is outside the polygon.")
from shapely.geometry import Point, Polygon
import numpy as np

# 定义圆心坐标和半径
circle_center = Point(0, 0)
circle_radius = 1.0

# 创建一个正多边形来近似表示圆
num_vertices = 30  # 假设圆被近似为一个有30条边的正多边形
vertices = [
    (circle_center.x + circle_radius * np.cos(2 * np.pi * i / num_vertices),
     circle_center.y + circle_radius * np.sin(2 * np.pi * i / num_vertices))
    for i in range(num_vertices)
]
circle_polygon = Polygon(vertices)

# 定义多边形的顶点坐标
polygon_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
polygon = Polygon(polygon_points)

# 检测圆和多边形是否相交
if circle_polygon.intersects(polygon) or circle_polygon.within(polygon):
    print("Circle intersects with or is within the polygon.")
else:
    print("Circle does not intersect with or is not within the polygon.")
