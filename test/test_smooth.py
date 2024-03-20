import numpy as np
import matplotlib.pyplot as plt
import bezier

path = np.load('path.npy')
num_point = len(path)
path = path.T
count = 0
curves = []
if num_point < 6:
    curves.append(bezier.Curve(path, degree=num_point-1))
else:
    # get the first curve
    count = 6
    nodes = path[:, :count]
    curves.append(bezier.Curve(nodes, degree=5))
    while count < num_point:
        # pick the last node of the last curve
        Q0 = path[:, count - 1]
        # create the new helper node
        Q1 = Q0 + (Q0 - path[:, count - 2]) / 2
        tmp = np.stack((Q0, Q1), axis=-1)
        if num_point-count >= 4:
            count += 4
            # pick the new node
            nodes = path[:, count-4:count]
            nodes = np.hstack((tmp, nodes))
            curves.append(bezier.Curve(nodes, degree=5))
        else:
            tmp_num = num_point-count
            nodes = path[:, count:num_point]
            nodes = np.hstack((tmp, nodes))
            count = num_point
            curves.append(bezier.Curve(nodes, degree=tmp_num+1))


# 定义控制点

# 在曲线上采样点
num_sample_points = 3 * num_point
num_curve = len(curves)
pick_point = int(num_sample_points / num_curve)+1
point_on_curves = None
for curve in curves:
    s_vals = np.linspace(0, 1, pick_point)
    point_on_curve = curve.evaluate_multi(s_vals[:-1])
    if point_on_curves is None:
        point_on_curves = point_on_curve
    else:
        point_on_curves = np.hstack((point_on_curves, point_on_curve))
point_on_curves = np.hstack((point_on_curves, path[:, -1].reshape(-1, 1)))


# 和直接全局优化的曲线对比
global_curve = bezier.Curve(path, degree=num_point-1)
# 绘制贝塞尔曲线
plt.plot(point_on_curves[0], point_on_curves[1], label='Bezier Curve')
plt.plot(global_curve.evaluate_multi(np.linspace(0, 1, num_sample_points))[0], global_curve.evaluate_multi(np.linspace(0, 1, num_sample_points))[1], label='Global Curve')

plt.scatter(path[0], path[1], color='red', label='Control Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Bezier Curve')
plt.legend()
plt.grid(True)
plt.show()
