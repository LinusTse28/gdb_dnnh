from math import sqrt
from auto_epsilon import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Rectangle
import os
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def load_data_from_csv(file_path):
    return np.array(pd.read_csv(file_path))

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
    cells.append((y, x))

    directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    steps = 1

    for layer in range(1, layers):
        for dx, dy in directions:
            for _ in range(steps):
                x += dx
                y += dy
                if 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]:
                    cells.append((y, x))
            if dx == 1 or dx == -1:
                steps += 1

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

def get_inner_spiral_cell(idx):
    layer = calculate_layer(idx)
    if layer == 0:
        return None
    inner_layer_start = (2*(layer-1)-1)**2
    inner_layer_end = (2*layer-1)**2 - 1
    # find nearest cell from current cell
    inner_cells = range(inner_layer_start, inner_layer_end+1)
    inner_idx = min(inner_cells, key=lambda x: abs(x-idx))
    #print('inner_idx: ', inner_idx)
    return inner_idx

def get_cells(idx):
    cells = []
    layer = calculate_layer(idx)
    if idx == 0:
        cells = [spiral_idxs[0]]

    elif 1 <= idx <= 8:
        cells = [spiral_idxs[idx], spiral_idxs[idx - 1], spiral_idxs[0]]

    else:
        inner_idx = get_inner_spiral_cell(idx)

        layer_start_idx = (2 * (layer - 1) + 1) ** 2
        if is_layer_end_cell(idx):
            cells = [spiral_idxs[idx],
                     spiral_idxs[idx - 1],
                     spiral_idxs[inner_idx],
                     spiral_idxs[layer_start_idx]]

        elif is_corner_cell(idx):
            cells = [spiral_idxs[idx],
                     spiral_idxs[idx - 1],
                     spiral_idxs[inner_idx]]

        else:
            cells = [spiral_idxs[idx],
                     spiral_idxs[idx - 1],
                     spiral_idxs[inner_idx],
                     spiral_idxs[inner_idx + 1],
                     spiral_idxs[inner_idx - 1]]
    return cells

def is_corner_cell(idx):
    layer = calculate_layer(idx)
    if layer == 0:
        return False

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

def is_mergeable(new_points, existing_cluster):
    # 对于 new_points 中的每个点
    for point in new_points:
        # 检查这个点是否在 existing_cluster 中
        if np.any(np.all(existing_cluster == point, axis=1)):
            return True
    return False

def perform_clustering(data, q, minPts, eps, cell_size):

    layers = int(round(1 / cell_size) / 2) + 1
    grid, grid_shape = set_grid(data, cell_size)

    spiral_idxs = spiral_cells(qx, qy, layers, grid_shape, cell_size)

    cells_to_eps = [spiral_idxs[i] for i in range(10)]
    pts_to_eps = get_pts_in_cells(grid, cells_to_eps)
    print(pts_to_eps)
    grid_eps = auto_epsilon(pts_to_eps, minPts=minPts)
    print(grid_eps)
    cluster_dict = {}
    clusterID = 0

    idx = 0
    idxs = len(spiral_idxs)
    #print(idx)
    while idx < idxs:
        #print('idx: ', idx)
        layer = calculate_layer(idx)  # 用于确定当前索引所在的层级
        cells = get_cells(idx)
        pts_in_cells = get_pts_in_cells(grid, cells)
        #print('idx: ', idx, 'cells: ', cells, 'len(pts): ', len(pts_in_cells))
        if len(pts_in_cells) >= minPts:
            pts_scale = StandardScaler().fit_transform(pts_in_cells)
            db = DBSCAN(eps=grid_eps, min_samples=minPts).fit(pts_scale)
            for label in set(db.labels_):
                if label == -1:
                    continue  # 忽略噪声点
                class_member_mask = (db.labels_ == label)
                new_points = pts_in_cells[class_member_mask]

                merged = False
                for cluster_id, existing_cluster in cluster_dict.items():
                    if is_mergeable(new_points, existing_cluster):
                        cluster_dict[cluster_id] = np.vstack((existing_cluster, new_points))
                        merged = True
                        break
                if not merged:
                    cluster_dict[clusterID] = new_points
                    clusterID += 1
        #print(cluster_dict)
        if any(len(cluster) > minPts * 2 for cluster in cluster_dict.values()):
            idxs = (2*layer + 1) ** 2
            idx += 1
            continue

        idx += 1
    print(' final idx: ', idx)
    delta_values = {cluster_id: delta(q, cluster) for cluster_id, cluster in cluster_dict.items()}
    min_delta_cluster_id = min(delta_values, key=delta_values.get)

    return cluster_dict[min_delta_cluster_id], min_delta_cluster_id

def visualization(data, q, cluster, cluster_id, cell_size, spiral_idxs, x0_range, x1_range):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制查询点
    ax.scatter(q[0], q[1], c='red', label='Query Point', s=50, zorder=10)
    # 绘制数据点
    ax.scatter(data[:, 0], data[:, 1], c='grey', alpha=0.5, label='Data Points', s=5, zorder=3)
    # 绘制delta最小的聚类
    ax.scatter(cluster[:, 0], cluster[:, 1], c='blue', label=f'Cluster {cluster_id}', s=5, zorder=4)

    # 绘制网格线
    ax.set_xticks(np.arange(0, 1, cell_size))
    ax.set_yticks(np.arange(0, 1, cell_size))
    ax.grid(which='both', color='black', linestyle='-', linewidth=0.5, alpha=0.7)


    cell_idx = spiral_idxs[2]
    cell_row, cell_col = cell_idx
    cell_left = cell_col * cell_size
    cell_right = (cell_col + 1) * cell_size
    cell_bottom = cell_row * cell_size
    cell_top = (cell_row + 1) * cell_size
    rect = Rectangle((cell_left, cell_bottom), cell_size, cell_size,
                     linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    ax.legend()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    ax.set_xlim(x0_range)
    ax.set_ylim(x1_range)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

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
file_names = [f"RN_{variant}0K_50P_1S.csv" for variant in variants]
variants = [
        '1', '5', '10', '15', '20'
    ]
file_names = [f"UN_{variant}0K.csv" for variant in variants]


variants = [
        '0.0S', '0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '1.0S', '1.5S', '2.0S', '2.5S', '3.0S'
    ]

file_names = [f"RN_{variant}_100K_50P.csv" for variant in variants]
data_paths = [os.path.join(base_path, file_name) for file_name in file_names]
#data_paths = ['/Users/linustse/Desktop/data/RN_200K_50P_1S.csv']
data_paths = ['/Users/linustse/Desktop/data/labeled/rn/RN_100K_50P_0.1S.csv']
for data_path in data_paths:
    #data = load_data_from_csv(data_path)
    data, labels = load_data_from_csv_labeled(data_path)
    #print('data\n', data)
    eps = auto_epsilon(data)
    print(eps)
    minPts = 5
    #q = np.array([0.19, 0.92])
    q = np.array([0.21808777, 0.72194386])
    x0_range = (q[0] - 0.1, q[0] + 0.1)  # 假设范围为点的横坐标减小0.1到加大0.1
    x1_range = (q[1] - 0.1, q[1] + 0.1)
    print(q)
    cell_size = eps / sqrt(2)
    #print(cell_size)
    #grid_shape = (round(1 / cell_size), round(1 / cell_size))
    layers = int(round(1 / cell_size) / 2) + 1
    grid, grid_shape = set_grid(data, cell_size)
    print('len(grid): ', len(grid))
    qx, qy = q[0], q[1]
    q_idx = (int(q[0] / cell_size), int(q[1] / cell_size))
    #print(qx//cell_size, qy//cell_size)
    spiral_idxs = spiral_cells(qx, qy, layers, grid_shape, cell_size)
    #print(spiral_idxs[:10])
    start_time = time.time()
    cluster, cluster_id = perform_clustering(data, q, minPts, eps, cell_size)
   # print(cluster)
    print(time.time()-start_time)

    visualization(data, q, cluster, cluster_id, cell_size, spiral_idxs, x0_range, x1_range)

