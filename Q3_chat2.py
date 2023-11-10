from utils import *
import numpy as np

def find_neighbors_in_grid(p, P, epsilon, grid, grid_position, visited):
    neighbors = []
    cells_to_search = get_adjacent_cells(grid, grid_position)

    for cell in cells_to_search:
        for pt, idx in P:
            if np.linalg.norm(p - pt) <= epsilon and not visited[idx]:
                neighbors.append((pt, idx))

    return neighbors


def expand_cluster(grid, labels, idx_p, neighbors, clusterId, epsilon, minPts, visited):
    labels[idx_p] = clusterId
    i = 0

    while i < len(neighbors):
        neighbor, neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1 and not visited[neighbor_idx]:
            labels[neighbor_idx] = clusterId
            visited[neighbor_idx] = True
            new_neighbors = find_neighbors_in_grid(neighbor, grid[neighbor_idx], epsilon, grid,
                                                   get_grid_position(neighbor), visited)
            if len(new_neighbors) >= minPts:
                neighbors.extend(new_neighbors)

        i += 1


def get_adjacent_cells(grid, position):
    x, y = position
    adjacent_cells = []

    for i in range(max(0, x - 1), min(len(grid), x + 2)):
        for j in range(max(0, y - 1), min(len(grid[0]), y + 2)):
            adjacent_cells.extend(grid[i][j])

    return adjacent_cells


# 此函数根据点的坐标返回其在网格中的位置
def get_grid_position(point, cell_size):
    x, y = point
    grid_x = x // cell_size
    grid_y = y // cell_size
    return int(grid_x), int(grid_y)


def grid_expand(query_point, points, cell_size, epsilon, minPts):
    # 初始化 grid
    max_x, max_y = np.max(points, axis=0)
    min_x, min_y = np.min(points, axis=0)
    grid_shape = (int((max_x - min_x) // cell_size) + 1, int((max_y - min_y) // cell_size) + 1)
    grid = [[[] for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]

    # 将点放入对应的格子中
    for idx, point in enumerate(points):
        x, y = ((point - [min_x, min_y]) // cell_size).astype(int)
        grid[x][y].append((point, idx))

    # 初始化访问和标签列表
    visited = [False] * len(points)
    labels = [-1] * len(points)  # cluster 的标签

    # 找到 query_point 所在的 cell
    qx, qy = ((query_point - [min_x, min_y]) // cell_size).astype(int)

    clusterId = 0
    # 根据您之前的描述，这里可能是螺旋搜索的逻辑
    # 您可能需要根据自己的需要进一步实现和调整这部分代码
    for layer in range(max(grid_shape)):
        cells_to_search = spiral_cells(qx, qy, layer, grid_shape)  # 这个函数需要您实现
        for cell_position in cells_to_search:
            for p, idx_p in grid[cell_position[0]][cell_position[1]]:
                if not visited[idx_p]:
                    visited[idx_p] = True
                    neighbors = find_neighbors_in_grid(p, points, epsilon, grid, cell_position, visited)
                    if len(neighbors) >= minPts:
                        expand_cluster(grid, labels, idx_p, neighbors, clusterId, epsilon, minPts, visited)
                        clusterId += 1

    # 您可以在此处添加更多的代码来找到具有最小 delta 的 cluster
    # ...
    min_delta = float('inf')
    best_cluster = None

    unique_labels = set(labels)
    for cluster_id in unique_labels:
        if cluster_id == -1:  # 噪声点
            continue
        cluster_points = [points[idx] for idx, label in enumerate(labels) if label == cluster_id]
        if len(cluster_points) > 1:  # 需要至少两个点来计算密度和 delta
            cluster_delta = delta(query_point, np.array(cluster_points))
            if cluster_delta < min_delta:
                min_delta = cluster_delta
                best_cluster = (cluster_id, cluster_points)

    return best_cluster


    return labels  # 或者返回您想要的其他内容，比如具有最小 delta 的 cluster


