from math import sqrt
from auto_epsilon import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os
import time
from sklearn.preprocessing import StandardScaler

def load_data_from_csv(file_path):
    return np.array(pd.read_csv(file_path))

def set_grid(data, cell_size):
    grid_shape = (int(np.ceil(1 / cell_size)), int(np.ceil(1 / cell_size)))
    grid = [[[] for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]
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

def get_cells_in_layers(spiral_idxs, layer):
    if layer == 0:
        cells = [spiral_idxs[0]]
    else:
        #start = (2 * (layer - 1) - 1) ** 2
        end = (2 * layer + 1) ** 2 - 1
        cells = [spiral_idxs[i] for i in range(0, end + 1)]
    return cells

def get_pts_in_layers(grid, spiral_idxs, layer):
    pts_in_cells = []
    cells = get_cells_in_layers(spiral_idxs, layer)

    for cell in cells:  # Ensure 'cell' is a tuple of (y, x)
        pts_in_cells.extend(grid[cell[0]][cell[1]])  # Correct indexing
    return np.array(pts_in_cells)


def perform_clustering(data, q, minPts, eps, cell_size):

    layers = int(round(1 / cell_size) / 2) + 1
    grid, grid_shape = set_grid(data, cell_size)
    qx, qy = q
    spiral_idxs = spiral_cells(qx, qy, layers, grid_shape, cell_size)

    cells_to_eps = [spiral_idxs[i] for i in range(10)]
    pts_to_eps = get_pts_in_layers(grid, spiral_idxs, layer=3)
    #print(pts_to_eps)
    grid_eps = auto_epsilon(pts_to_eps, minPts=minPts)
    #print(grid_eps)
    cluster_dict = {}

    layer = 0
    while layer < layers:
        clusterID = 0
        pts_in_cells = get_pts_in_layers(grid, spiral_idxs, layer)
        if len(pts_in_cells) >= minPts:
            pts_scale = StandardScaler().fit_transform(pts_in_cells)
            db = DBSCAN(eps=grid_eps, min_samples=minPts).fit(pts_scale)
            non_noise_labels = set(label for label in set(db.labels_) if label != -1)

            for label in non_noise_labels:
                class_member_mask = (db.labels_ == label)
                new_points = pts_in_cells[class_member_mask]
                cluster_dict[clusterID] = new_points
                clusterID += 1
        #print(cluster_dict)
        if any(len(cluster) > minPts * 2 for cluster in cluster_dict.values()):
            break
        #print(layer)
        layer += 1
    #print(' final layer: ', layer)
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

def run(data_path):
    q = np.random.rand(2)
    # print(q)
    x0_range = (q[0] - 0.1, q[0] + 0.1)
    x1_range = (q[1] - 0.1, q[1] + 0.1)
    data, labels = load_data_from_csv_labeled(data_path)
    minPts = 4
    eps = auto_epsilon(data, minPts)
    #print(eps)
    cell_size = eps / sqrt(2)
    layers = int(round(1 / cell_size) / 2) + 1
    grid, grid_shape = set_grid(data, cell_size)
    start_time = time.time()
    cluster, cluster_id = perform_clustering(data, q, minPts, eps, cell_size)
    execution_time = time.time() - start_time
    nearest_idx = find_nearest_points_kd(data, cluster)
    predict_labels = [labels[idx] for idx in nearest_idx]
    counts = np.bincount(predict_labels)
    label = np.argmax(counts)
    if len(cluster) < 50:
        extended_array = predict_labels + [0] * (50 - len(predict_labels))
        predict_labels = np.array(extended_array)

    TP = np.count_nonzero(predict_labels == label)
    FP = len(cluster) - counts[-1]

    TN = 0
    FN = (lambda x: 50 if x >= 50 else 50 - x)(counts[-1])

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = TP / (TP + 1 / 2 * (FP + FN))
    qx, qy = q
    spiral_idxs = spiral_cells(qx, qy, layers, grid_shape, cell_size)
    x0_range = (q[0] - 0.1, q[0] + 0.1)
    x1_range = (q[1] - 0.1, q[1] + 0.1)
    #visualization(data, q, cluster, cluster_id, cell_size, spiral_idxs, x0_range, x1_range)
    return {
        "time": execution_time,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

base_path = '/Users/linustse/Desktop/data/'

variants = [
    '1', '5', '10', '15', '20'
]
file_names = [f"RN_{variant}0K_50P_1S.csv" for variant in variants]
variants = [
    '1', '5', '10', '15', '20'
]
file_names = [f"UN_{variant}0K.csv" for variant in variants]
variants = [
    '0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '1.0S', '1.5S', '2.0S', '2.5S', '3.0S'
]

file_names = [f"RN_{variant}_100K_50P.csv" for variant in variants]
base_path = '/Users/linustse/Desktop/data/labeled/rn/'
variants = [
      '0.0S', '0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '1.0S', '1.5S', '2.0S'
]

file_names = [f"RN_100K_50P_{variant}.csv" for variant in variants]
data_paths = [os.path.join(base_path, file_name) for file_name in file_names]
#data_paths = ['/Users/linustse/Desktop/data/RN_200K_50P_1S.csv']
#data_paths = ['/Users/linustse/Desktop/data/labeled/rn/RN_100K_50P_0.1S.csv']
all_results = []
for data_path in data_paths:
    experiment_results = []
    for i in range(30):
        result = run(data_path)
        result['data_path'] = data_path
        experiment_results.append(result)

    all_results.extend(experiment_results)

df_results = pd.DataFrame(all_results)
df_results.to_excel('/Users/linustse/Desktop/expand_grid_RN_distance.xlsx', index=False)
