from sklearn.cluster import DBSCAN
from math import sqrt
from auto_epsilon import auto_epsilon
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import os
import time
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

def get_grid_index(x, cell_size):
    return int(x // cell_size)

def spiral_cells(qx, qy, layers, grid_shape, cell_size):
    cells = []
    x, y = get_grid_index(qx, cell_size), get_grid_index(qy, cell_size)
    #y = grid_shape[0] - 1 - y
    x_offset = qx % cell_size - 0.5 * cell_size
    y_offset = qy % cell_size - 0.5 * cell_size

    for layer in range(layers):
        # Right
        for i in range(2 * layer + 1):
            if 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]:
                cells.append((y, x))
            x += 1
            if layer == 0 and i == 0:
                x += get_grid_index(x_offset, cell_size)

        # Up
        for i in range(2 * layer + 1):
            if 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]:
                cells.append((y, x))
            y -= 1
            if layer == 0 and i == 0:
                y -= get_grid_index(y_offset, cell_size)

        # Left
        for i in range(2 * layer + 2):
            if 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]:
                cells.append((y, x))
            x -= 1

        # Down
        for i in range(2 * layer + 2):
            if 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]:
                cells.append((y, x))
            y += 1

    return cells

def calculate_layer(idx):
    layer = 0
    while True:
        if idx < (2*layer+1)**2:
            break
        layer += 1
    return layer

def get_pts_in_layer(grid, spiral_idxs, layer):
    # 获取层级内的所有点
    pts_in_layer = []
    for idx in spiral_idxs:
        if calculate_layer(idx) <= layer:
            pts_in_layer.extend(grid[idx[0]][idx[1]])
    return np.array(pts_in_layer)

def get_pts_in_cells(grid, cells):
    pts_in_cells = []
    for cell in cells:  # Ensure 'cell' is a tuple of (y, x)
        pts_in_cells.extend(grid[cell[0]][cell[1]])  # Correct indexing
    return np.array(pts_in_cells)

def get_inner_spiral_cell(idx, spiral_idxs):
    layer = calculate_layer(idx)
    if layer == 0:
        return None
    inner_layer_start = (2*(layer-1)-1)**2
    inner_layer_end = (2*layer-1)**2 - 1
    # find nearest cell from current cell
    inner_cells = range(inner_layer_start, inner_layer_end+1)
    inner_idx = min(inner_cells, key=lambda x: abs(x-idx))
    return spiral_idxs.index(inner_idx) if inner_idx in spiral_idxs else None

def is_corner_cell(idx):
    layer = calculate_layer(idx)
    if layer == 0:
        return False  # 中心点不是边角单元格

    # 计算当前层的起始和结束索引
    layer_start = (2*(layer-1)+1)**2
    layer_end = (2*layer+1)**2 - 1

    # 计算当前层四个角落的索引
    top_right = layer_start
    top_left = top_right + 2*layer - 1
    bottom_left = top_left + 2*layer
    bottom_right = layer_end

    # 判断idx是否为四个角落的索引之一
    return idx == top_right or idx == top_left or idx == bottom_left

def is_layer_end_cell(idx):
    layer = calculate_layer(idx)
    return idx == (2*layer+1)**2 - 1

def perform_clustering(data, q, minPts, cell_size, eps, spiral_idx, grid, grid_shape):
    cluster_dict = {}
    clusterID = 0

    idx = 0
    idxs = len(spiral_idxs)
    while idx < idxs:
        layer = calculate_layer(idx)  # 用于确定当前索引所在的层级
        pts_in_cells = []

        if idx == 0:
            cells = [spiral_idxs[0]]

        elif layer == 2:
            cells = [spiral_idxs[idx], spiral_idxs[idx-1], spiral_idxs[0]]

        elif layer >= 3:
            inner_idx = get_inner_spiral_cell(idx, spiral_idxs)
            layer_start_idx = (2 * (layer - 1) + 1) ** 2
            if is_layer_end_cell(idx):
                cells = [spiral_idxs[idx],
                         spiral_idxs[idx-1],
                         spiral_idxs[inner_idx],
                         spiral_idxs[layer_start_idx]]

            elif is_corner_cell(idx):
                cells = [spiral_idxs[idx],
                         spiral_idxs[idx-1],
                         spiral_idxs[inner_idx]]

            else:
                cells = [spiral_idxs[idx],
                         spiral_idxs[idx-1],
                         spiral_idxs[inner_idx],
                         spiral_idxs[inner_idx+1],
                         spiral_idxs[inner_idx-1]]

        pts_in_cells = get_pts_in_cells(grid, cells)
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

        if any(len(cluster) > minPts * 5 for cluster in cluster_dict.values()):
            idxs = (layer + 1) ** 2

        idx += 1

    delta_values = {cluster_id: delta_(q, cluster) for cluster_id, cluster in cluster_dict.items()}

    # 找到delta值最小的聚类ID
    min_delta_cluster_id = min(delta_values, key=delta_values.get)

    # 返回delta值最小的聚类
    return cluster_dict[min_delta_cluster_id], min_delta_cluster_id


def visualization(data, q, cluster, cluster_id, cell_size, spiral_idxs, grid, grid_shape):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制查询点
    ax.scatter(q[0], q[1], c='red', label='Query Point', s=50, zorder=5)
    # 绘制数据点
    ax.scatter(data[:, 0], data[:, 1], c='grey', alpha=0.5, label='Data Points', s=10, zorder=3)
    # 绘制delta最小的聚类
    ax.scatter(cluster[:, 0], cluster[:, 1], c='blue', label=f'Cluster {cluster_id}', s=10, zorder=4)

    # 绘制网格线
    ax.set_xticks(np.arange(0, 1, cell_size))
    ax.set_yticks(np.arange(0, 1, cell_size))
    ax.grid(which='both', color='black', linestyle='-', linewidth=0.5, alpha=0.7)

    '''# 标记格子的顺序
    for i, (y, x) in enumerate(spiral_idxs):
        # 计算格子中心点坐标
        center_x, center_y = (x * cell_size + cell_size / 2), (y * cell_size + cell_size / 2)
        # 如果该格子在目标聚类中，添加注释
        if any(np.all(min_delta_cluster == point, axis=1) for point in grid[y][x]):
            ax.text(center_x, center_y, str(i), color="black", ha="center", va="center", fontsize=8)'''

    # 设置图例和坐标轴标签
    ax.legend()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')



    ax.set_xlim(0.1, 0.3)
    ax.set_ylim(0.8, 1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# 用于测试的数据和参数
#data = load_data_from_csv('/Users/linustse/Desktop/data/RN_50K_50P_1S.csv')

# 执行聚类搜索
base_path = '/Users/linustse/Desktop/data/'

variants = [
    '0.0S', '0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '1.0S', '1.5S', '2.0S', '2.5S', '3.0S'
]
file_names = [f"RN_{variant}_100K_50P.csv" for variant in variants]
variants = [
    '1', '5', '10', '15', '20'
]
variant = ['20']
file_names = [f"RN_{variant}0K_50P_1S.csv" for variant in variants]
data_paths = [os.path.join(base_path, file_name) for file_name in file_names]

for data_path in data_paths:
    data = load_data_from_csv(data_path)
    eps = auto_epsilon(data)
    minPts = 5
    q = np.array([0.19, 0.92])

    cell_size = eps / sqrt(2)
    #grid_shape = (round(1 / cell_size), round(1 / cell_size))
    layers = int(round(1 / cell_size) / 2) + 1
    grid, grid_shape = set_grid(data, cell_size)
    qx, qy = q[0], q[1]
    q_idx = (int(q[0] / cell_size), int(q[1] / cell_size))
    spiral_idxs = spiral_cells(qx, qy, layers, grid_shape, cell_size)
    start_time = time.time()
    min_delta_cluster, min_delta_cluster_id = perform_clustering(data, q, minPts, cell_size, eps, spiral_idxs, grid, grid_shape)

    print(data_path, ': ', time.time()-start_time, 's')
    #min_delta_cluster, min_delta_cluster_id = perform_clustering(data, q, minPts, cell_size, eps, )

    visualization(data, q, min_delta_cluster, min_delta_cluster_id, cell_size, spiral_idxs, grid, grid_shape)
