import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from auto_epsilon import auto_epsilon

def set_grid(q, P, epsilon):
    cell_size = epsilon / np.sqrt(2)

    def get_cell_idx(point):
        return tuple(map(int, np.floor(point / cell_size)))

    cells = {}
    for pt in P:
        cell_idx = get_cell_idx(pt)
        if cell_idx not in cells:
            cells[cell_idx] = []
        cells[cell_idx].append(pt)

    return cells

path = '/Users/linustse/Desktop/data/RN_50K_50P_1S.csv'
P = np.array(pd.read_csv(path))

q = np.array([0.19, 0.92])
epsilon = auto_epsilon(P)
print(epsilon)

cells = set_grid(q, P, epsilon)
# 画网格
x_min, y_min, x_max, y_max = 0, 0, 1, 1
cell_size = epsilon / np.sqrt(2)

def visualize_nearby_grids(q, cells, epsilon):
    plt.figure(figsize=(8, 8))

    cell_size = epsilon / np.sqrt(2)

    def get_cell_idx(point):
        return tuple(map(int, np.floor(np.array(point) / cell_size)))

    q_idx = get_cell_idx(q)

    # 绘制 q 点所在的和附近的网格中的点
    for i in range(-10, 11):  # 调整范围以包括更多或更少的网格
        for j in range(-10, 11):
            nearby_idx = (q_idx[0] + i, q_idx[1] + j)
            if nearby_idx in cells:
                for pt in cells[nearby_idx]:
                    plt.scatter(pt[0], pt[1], c='blue', s=1)

    # 绘制 q 点
    plt.scatter(q[0], q[1], c='red', s=10)

    # 绘制网格线
    x_lines = np.arange(0, 1, cell_size)
    y_lines = np.arange(0, 1, cell_size)
    for x in x_lines:
        plt.axvline(x=x, color='grey', linestyle='--', linewidth=0.2)
    for y in y_lines:
        plt.axhline(y=y, color='grey', linestyle='--', linewidth=0.2)

    plt.xlim(0.1, 0.3)
    plt.ylim(0.8, 1)
    '''plt.xlim(0, 1)
    plt.ylim(0, 1)'''
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# 调用函数
visualize_nearby_grids(q, cells, epsilon)
