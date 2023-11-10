from utils import *
from itertools import product
import matplotlib.pyplot as plt


def setGrid(q, P, cellSize, gridSize):
    """根据给定的q，设置网格并返回每个网格包含的点
    
    Parameters
    ----------
        q : array-like, shape (n_features,)
            查询点
        P : array-like, shape (n_samples, n_features)
            数据集
        cellSize : int
            有多少个单元格，我这里认为是用来划分一个 cellSize * cellSize 的网格
        gridSize : float
            每个网格（正方形）的大小，

    Returns
    -------
        cells : dict
            每个网格包含的点
    """
    q = np.array(q)
    min_val = q - (gridSize * cellSize / 2.0)
    max_val = q + (gridSize * cellSize / 2.0)

    # 划分网格.
    grid_indices = product(np.arange(int(min_val[0]), int(max_val[0]), gridSize),
                           np.arange(int(min_val[1]), int(max_val[1]), gridSize))
    gird_index = product(
        range(0, cellSize),
        range(0, cellSize)
    )
    
    # 创建一个空的字典来存储每个单元格中的点.
    cells = {}
    for idx in gird_index:
        cells[idx] = []
        
    # 分配每个点到相应的单元格.
    for point, idx in P:
        cell_idx = (
            (point[0] - min_val[0]) // gridSize,
            (point[1] - min_val[1]) // gridSize
        )
        # 需要把它转换成整数.
        cell_idx = tuple(map(int, cell_idx))

        # 如果点在当前的范围内，就把它加到相应的cell.
        if cell_idx in cells:
            cells[cell_idx].append([point, idx])
        else:
            # 否则，我们 skip 掉这个点.
            pass
    
    return cells


def find_neighbors(p, P, epsilon):
    """查找给定点 p 在 P 中的邻居"""
    neighbors = []
    for pt, idx in P:
        if np.linalg.norm(p - pt) <= epsilon:
            neighbors.append(pt)
    return neighbors

def expand_cluster(P, labels, idx_p, p, neighbors, clusterId, epsilon, minPts):
    labels[idx_p] = clusterId
    i = 0

    while i < len(neighbors):
        neighbor, neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = clusterId
            new_neighbors = find_neighbors(neighbor, P, epsilon)
            if len(new_neighbors) >= minPts:
                neighbors.extend(new_neighbors)
        i += 1

def spiral_order(q, cells, cellSize):
    """生成以 q 为中心的螺旋顺序的元素"""

    assert cellSize % 2 == 1
    # 这里我们假设 len(cells) % 2 == 1
    # 因为这样可以保证 q 位于最中心。
    # 1 2 3
    # 4 5 6
    # 7 8 9
    # 例如上面，我们希望 q 位于 5 中。
    cx, cy = cellSize // 2, cellSize // 2
    result = [cells[(cx, cy)]]

    dirs = [
        [0, 1],  # 右
        [1, 0],  # 下
        [0, -1], # 左
        [-1, 0]  # 上
    ]
    curDir = 0 # 初始为向右

    center = cellSize // 2 # 中心
    limit = 1 # 当前螺旋的边界 [center - limit, center + limit]
    checkX = 1 # 当前是检查 X 的边界嘛
    cnt = 0 # 用来判断是否需要增加 limit

    for _ in range(cellSize**2):
        dx, dy = dirs[curDir]
        cx += dx
        cy += dy
        # 如果该 cells 中有点，就加入到结果中
        if (cx, cy) in cells:
            result.append(cells[(cx, cy)])
        else:
            # 否则，我们加入一个空的列表
            result.append([])
        

        if checkX:
            # 如果需要检查 X 边界
            if cx == center + limit or \
                cx == center - limit:
                checkX ^= 1 # 切换到检查 Y
                curDir = (curDir + 1) % 4 # 切换方向
                cnt += 1 # 增加计数
        else:
            # 同理，检查 Y 边界
            if cy == center + limit or \
                cy == center - limit:
                checkX ^= 1
                curDir = (curDir + 1) % 4
                cnt += 1
        # 如果计数到 4，就增加 limit，也就是增加螺旋的边界
        # a b c d e f
        # g h i j k l
        # m n o p q r
        # s t u v w x
        # y z 1 2 3 4
        # 如上，当我们 [o, p, v, u, t, n, h, i, j] 访问玩了
        # 下一步我们是访问 k，所以要扩展边界。
        if cnt == 4:
            limit += 1
            cnt = 0

    return result


def grid_expand_algorithm(P, q, cellSize, gridSize, epsilon, minPts):
    """Grid expand 算法实现"""
    # Add index for P.
    P = np.array([[p, idx] for idx, p in enumerate(P)],
                 dtype=object)
    # 按距离进行一次排序。
    P = np.array(sorted(P, key=lambda p: distance.euclidean(p[0], q)))
    bound = float('inf')
    P_cur = []

    # 初始化标签
    labels = np.full(P.shape[0], -1, dtype=int)

    density_threshold = 2

    clusterId = 0
    curCell = 0 # 当前的单元格
    cells = setGrid(q, P, cellSize, gridSize)
    # 获取访问的路径
    Pt = spiral_order(q, cells, cellSize)

    while curCell < len(Pt):
        # 按照螺旋顺序遍历每个单元格
        P_cur = Pt[curCell]

        # 当前 cell 中所有的点都得在 bound 中才能继续运行
        if np.any([np.linalg.norm(p - q) > bound for p, _ in P_cur]):
            break

        # 如果是当前 cell 空的，就跳过
        if len(P_cur) == 0:
            curCell += 1
            continue

        # 寻找离q最近的点
        p, idx = min(P_cur, key=
            lambda x: distance.euclidean(x[0], q))

        #! 这里不确定是 P_cur 还是 Pt 或者 P？不过你可以看着改
        neighbors = find_neighbors(p, P_cur, epsilon)

        # 如果邻居数小于 minPts，就标记为噪声
        if len(neighbors) < minPts:
            labels[p] = -1
        else:
            expand_cluster(
                P,
                labels,
                idx,
                p,
                neighbors,
                clusterId,
                epsilon,
                minPts
            )
            clusterId += 1

        for cur_clusterId in range(clusterId):
            # 选择当前的 cluster_id.
            Ci = P[(labels == cur_clusterId)]

            #! density_threshold 可以看着修改
            if len(Ci) >= minPts * density_threshold:
                bound = np.sqrt(2) * cellSize * (curCell + 1)

    # Remove index for P.
    P = rmv_idx(P)
    clusters = [[i, P[labels == i]] for i in range(clusterId)]
    clusters = sorted(clusters, key=lambda c: delta(q, c[1]))

    # 其中 best_cluster 是 clusters[0]，
    # 其中第一个代表 clusterId, 第二个代表 cluster 本身。
    return clusters, labels

if __name__ == "__main__":
    P = create_fake_data()
    q = np.array([0, 0])

    clusters, labels = grid_expand_algorithm(
        P,
        q,
        cellSize=5,
        gridSize=2,
        epsilon=0.5,
        minPts=3
    )
    for i, cluster_data in clusters:
        if i == clusters[0][0]:
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color='yellow', s=1)
            # print('plot best cluster is ', clusters[0][1], len(clusters[0][1]))
    plt.scatter(q[0], q[1], color='red', marker='+', s=100)
