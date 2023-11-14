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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from tabulate import tabulate

def load_data_from_csv(file_path):
    return np.array(pd.read_csv(file_path))


def load_data_from_csv_labeled(file_path):
    # 加载数据，假设CSV文件的前两列是x和y坐标，第三列是标签
    df = pd.read_csv(file_path)

    # 分离特征和标签
    X = df.iloc[:, :2].values  # 前两列是特征
    y = df.iloc[:, 2].values  # 第三列是标签

    return X, y
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

'''def spiral_cells(qx, qy, layers, grid_shape, cell_size):
    cells = []
    x, y = get_grid_index(qx, cell_size), get_grid_index(qy, cell_size)
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
    return cells'''


def spiral_cells(qx, qy, layers, grid_shape, cell_size):
    cells = []
    # 将查询点转换为网格索引
    x, y = get_grid_index(qx, cell_size), get_grid_index(qy, cell_size)

    # 从查询点所在单元格开始
    cells.append((y, x))

    # 定义移动方向 (右, 上, 左, 下)
    directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]

    # 定义初始步数
    steps = 1

    # 开始生成螺旋路径
    for layer in range(1, layers):
        # 遍历每个方向
        for dx, dy in directions:
            # 对于每个方向，根据当前层进行适当次数的移动
            for _ in range(steps):
                x += dx
                y += dy
                # 检查坐标是否在网格内
                if 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]:
                    cells.append((y, x))
            # 每完成一个方向，增加步数
            if dx == 1 or dx == -1:  # 在水平移动之后增加步数
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


'''def is_mergeable(new_points, existing_cluster, eps, minPts):
    nn = NearestNeighbors(radius=eps)

    # 将新点和现有聚类中的点结合起来
    all_points = np.vstack((new_points, existing_cluster))

    # 训练最近邻模型
    nn.fit(all_points)

    # 对每个新点进行半径查询
    for point in new_points:
        # 如果新点在eps半径内有足够多的邻居，则认为是可合并的
        if len(nn.radius_neighbors([point], return_distance=False)[0]) >= minPts:
            return True

    return False'''
def is_mergeable(new_points, existing_cluster):
    # 对于 new_points 中的每个点
    for point in new_points:
        # 检查这个点是否在 existing_cluster 中
        if np.any(np.all(existing_cluster == point, axis=1)):
            return True
    return False

def perform_clustering(data, q, minPts, cell_size, eps, spiral_idxs, grid, grid_shape):
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
            db = DBSCAN(eps=eps, min_samples=minPts).fit(pts_in_cells)
            for label in set(db.labels_):
                if label == -1:
                    continue  # 忽略噪声点
                class_member_mask = (db.labels_ == label)
                new_points = pts_in_cells[class_member_mask]

                # 尝试将新点合并到现有的聚类中，或者创建一个新聚类
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

    delta_values = {cluster_id: delta(q, cluster) for cluster_id, cluster in cluster_dict.items()}
    min_delta_cluster_id = min(delta_values, key=delta_values.get)

    return cluster_dict[min_delta_cluster_id], min_delta_cluster_id

def evaluate_clustering(labels_true, labels_pred):

    precision, recall, f1, _ = precision_recall_fscore_support(labels_true, labels_pred, average='weighted')
    return precision, recall, f1

def visualization(data, q, cluster, cluster_id, cell_size, spiral_idxs, grid, grid_shape):
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

    ax.set_xlim(0.15, 0.25)
    ax.set_ylim(0.85, 0.95)

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

#data_paths = ['/Users/linustse/Desktop/data/RN_200K_50P_1S.csv']
base_path = '/Users/linustse/Desktop/data/labeled/rn/'
variants = [
    '0.0S', '0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '1.0S', '1.5S', '2.0S'
]

file_names = [f"RN_100K_50P_{variant}.csv" for variant in variants]

data_paths = [os.path.join(base_path, file_name) for file_name in file_names]

#data_paths = ['/Users/linustse/Desktop/data/labeled/rn/RN_100K_50P_1.0S.csv']
for data_path in data_paths:
    #data = load_data_from_csv(data_path)
    data, labels = load_data_from_csv_labeled(data_path)
    #print('data\n', data)
    eps = auto_epsilon(data)
    minPts = 5
    q = np.array([0.19, 0.92])

    cell_size = eps / sqrt(2)
    #print(cell_size)
    #grid_shape = (round(1 / cell_size), round(1 / cell_size))
    layers = int(round(1 / cell_size) / 2) + 1
    grid, grid_shape = set_grid(data, cell_size)
    qx, qy = q[0], q[1]
    q_idx = (int(q[0] / cell_size), int(q[1] / cell_size))
    #print(qx//cell_size, qy//cell_size)
    spiral_idxs = spiral_cells(qx, qy, layers, grid_shape, cell_size)
    #print(spiral_idxs[:10])
    start_time = time.time()
    cluster, cluster_id = perform_clustering(data, q, minPts, cell_size, eps, spiral_idxs, grid, grid_shape)
    print(time.time() - start_time)
    cluster = np.unique(cluster, axis=0)
    nearest_idx = find_nearest_points_kd(data, cluster)
    predict_labels = [labels[idx] for idx in nearest_idx]

    counts = np.bincount(predict_labels)
    label = np.argmax(counts)
    if len(cluster) < 50:
        extended_array = predict_labels + [0] * (50 - len(predict_labels))
        predict_labels = np.array(extended_array)

    TP = np.count_nonzero(predict_labels == label)
    FP = len(cluster)-counts[-1]
    #print(predict_labels)
    #print(counts[-1])

    TN = 0
    FN = 50 - counts[-1]

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = TP / (TP + 1 / 2 * (FP + FN))
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    #print('nearest_points_indices:', predict_labels)
    '''real_labels = np.array([label] * 50)
    #print(real_labels)
    accuracy = accuracy_score(real_labels, predict_labels)

    confusion = confusion_matrix(real_labels, predict_labels)

    precision = precision_score(real_labels, predict_labels, average='weighted')

    recall = recall_score(real_labels, predict_labels, average='weighted')

    f1 = f1_score(real_labels, predict_labels, average='weighted')

    report = classification_report(real_labels, predict_labels)
    print("分类报告:")
    print(report)'''


    visualization(data, q, cluster, cluster_id, cell_size, spiral_idxs, grid, grid_shape)

