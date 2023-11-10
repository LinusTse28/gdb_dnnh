from sklearn.cluster import DBSCAN
from math import sqrt
from auto_epsilon import auto_epsilon
from utils import *
import matplotlib.pyplot as plt
import numpy as np

def load_data_from_csv(file_path):
    return pd.read_csv(file_path).values

def set_grid(data, cell_size):
    # 确定网格大小
    grid_shape = (int(np.ceil(1 / cell_size)), int(np.ceil(1 / cell_size)))
    # 初始化网格
    grid = [[[] for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]
    # 分配数据点到网格
    for point in data:
        x_idx, y_idx = int(point[0] / cell_size), int(point[1] / cell_size)
        grid[y_idx][x_idx].append(point)
    return grid, grid_shape

def spiral_order(q, grid_shape, cell_size):
    # 螺旋顺序索引的初始化实现
    spiral_indices = []
    x, y = int(q[0] / cell_size), int(q[1] / cell_size)
    dx, dy = 0, -1
    for i in range(max(grid_shape)**2):
        if (0 <= x < grid_shape[1]) and (0 <= y < grid_shape[0]):
            spiral_indices.append((y, x))
        if x == y or (x < y and x + y == grid_shape[0] - 1) or (x > y and x + y == grid_shape[1]):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
    return spiral_indices

def calculate_layer(cell_idx, q_idx):
    # 计算网格层级
    return max(abs(cell_idx[0] - q_idx[0]), abs(cell_idx[1] - q_idx[1]))

def get_points_in_spiral_cells(grid, spiral_indices, layer):
    # 获取层级内的所有点
    pts_in_cells = []
    for idx in spiral_indices:
        if calculate_layer(idx, spiral_indices[0]) <= layer:
            pts_in_cells.extend(grid[idx[0]][idx[1]])
    return np.array(pts_in_cells)

def perform_clustering(data, q, minPts):
    eps = auto_epsilon(data)
    cell_size = eps / sqrt(2)
    grid, grid_shape = set_grid(data, cell_size)
    q_idx = (int(q[0] / cell_size), int(q[1] / cell_size))
    spiral_indices = spiral_order(q, grid_shape, cell_size)
    cluster_dict = {}
    clusterID = 0

    for layer in range(1, max(grid_shape)):
        pts_in_cells = get_points_in_spiral_cells(grid, spiral_indices, layer)
        if len(pts_in_cells) >= minPts:
            db = DBSCAN(eps=eps, min_samples=minPts).fit(pts_in_cells)
            unique_labels = set(db.labels_)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            if len(unique_labels) > 0:
                for k in unique_labels:
                    class_member_mask = (db.labels_ == k)
                    cluster_dict[clusterID] = pts_in_cells[class_member_mask]
                    clusterID += 1
            else:
                break

    return cluster_dict


def visualize_clusters(data, clusters):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制原始数据点
    ax.scatter(data[:, 0], data[:, 1], c='grey', alpha=0.5, label='Data Points', s=1)

    # 绘制每个聚类的数据点
    for cluster_id, points in clusters.items():
        ax.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster_id}', s=1)

    # 设置图例
    ax.legend()

    # 设置坐标轴标签
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # 设置坐标轴范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 展示图形
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# 用于测试的数据和参数
data = load_data_from_csv('/Users/linustse/Desktop/data/RN_50K_50P_1S.csv')
eps = auto_epsilon(data)
minPts = 5
q = np.array([0.19, 0.92])

cell_size = eps / sqrt(2)
grid_shape = (round(1 / cell_size), round(1 / cell_size))
layers = int(round(1 / cell_size) / 2) + 1
# 执行聚类搜索
clusters = perform_clustering(data, q, minPts)

visualize_clusters(data, clusters)

