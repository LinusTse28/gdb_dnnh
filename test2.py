import matplotlib.pyplot as plt
from auto_epsilon import *
from math import *


def spiral_cells(qx, qy, layers, grid_shape, cell_size):
    cells = []
    x, y = get_grid_index(qx, cell_size), get_grid_index(qy, cell_size)
    y = grid_shape[0] - 1 - y
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


def get_spiral_order_indices(qx, qy, layers, grid_shape, cell_size):
    # 使用您提供的函数生成螺旋索引
    spiral_indices = spiral_cells(qx, qy, layers, grid_shape, cell_size)

    # 根据索引返回顺序
    return spiral_indices


# 此外，确保 get_grid_index 函数返回的坐标是正确的，例如：
def get_grid_index(x, cell_size):
    return int(x // cell_size)


def assign_points_to_cells(data, qx, qy, layers, grid_shape, cell_size):
    spiral_order_indices = spiral_cells(qx, qy, layers, grid_shape, cell_size)

    cells_dict = {idx: [] for idx in spiral_order_indices}

    for point in data:
        x_idx, y_idx = get_grid_index(point[0], cell_size), get_grid_index(point[1], cell_size)
        y_idx = grid_shape[0] - 1 - y_idx  # 如果需要倒置y轴索引

        if (y_idx, x_idx) in cells_dict:
            cells_dict[(y_idx, x_idx)].append(point.tolist())

    return cells_dict


def visualize_spiral(grid_shape, qx, qy, layers, cell_size, data):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim([0, grid_shape[1] * cell_size])
    ax.set_ylim([0, grid_shape[0] * cell_size])

    # 绘制网格线
    for x in range(grid_shape[1]):
        ax.axvline(x * cell_size, color='grey', linewidth=0.8)
    for y in range(grid_shape[0]):
        ax.axhline(y * cell_size, color='grey', linewidth=0.8)

    # 标记螺旋顺序的格子
    spiral_order_indices = get_spiral_order_indices(qx, qy, layers, grid_shape, cell_size)
    print(spiral_order_indices)
    for idx, (row, col) in enumerate(spiral_order_indices):
        rect = plt.Rectangle((col * cell_size, (grid_shape[0] - row - 1) * cell_size), cell_size, cell_size,
                             fill=None, edgecolor='blue', linewidth=1)
        ax.add_patch(rect)
        ax.text(col * cell_size + cell_size / 2, (grid_shape[0] - row - 1) * cell_size + cell_size / 2,
                str(idx + 1), color="blue", ha="center", va="center", fontsize=9)

    # 绘制数据点
    cells_dict = assign_points_to_cells(data, qx, qy, layers, grid_shape, cell_size)
    for (row, col), points in cells_dict.items():
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        ax.scatter(x_values, y_values, s=10, c='red')  # 红色点表示数据点

    # 标记查询点 q
    ax.scatter(qx, qy, c='green', s=50, marker='x')  # 绿色x表示查询点

    # 关闭matplotlib的默认网格显示，并手动设置坐标轴标签
    plt.grid(False)
    plt.xticks(np.arange(0, grid_shape[1] * cell_size, cell_size))
    plt.yticks(np.arange(0, grid_shape[0] * cell_size, cell_size))
    ax.set_xticklabels(np.around(np.arange(0, grid_shape[1] * cell_size, cell_size), 2))
    ax.set_yticklabels(np.around(np.arange(0, grid_shape[0] * cell_size, cell_size), 2))

    # 翻转y轴以匹配常用的坐标系
    plt.gca().invert_yaxis()

    # 显示图形
    plt.xlim(0.1, 0.3)
    plt.ylim(0.8, 1)
    plt.show()


# Example usage
data = np.array(pd.read_csv('/Users/linustse/Desktop/data/RN_50K_50P_1S.csv'))
eps = auto_epsilon(data)

cell_size = eps / sqrt(2)
grid_shape = (round(1 / cell_size), round(1 / cell_size))
layers = int(round(1 / cell_size) / 2) + 1

print('eps: ', eps, '\ncellsize :', cell_size, '\nlayers: ', layers)
'''cell_size = 0.1  # Cell size of 0.01x0.01 in the data range 0-1
grid_shape = (10, 10)  # 100x100 grids

layers = 10  # Number of layers to spiral out'''
qx, qy = 0.19, 0.92  # Starting point in the data range
visualize_spiral(grid_shape, qx, qy, layers, cell_size, data)

'''cells = spiral_cells(qx, qy, layers, grid_shape, cell_size)
print(cells)
'''
'''assigned_cells = assign_points_to_cells(data, qx, qy, layers, grid_shape, cell_size)

# 打印前9个网格内的数据点
for idx, (cell_idx, points) in enumerate(assigned_cells.items()):
    if idx < 9:  # 我们只想打印前9个网格
        print(f"Cell {cell_idx} has {len(points)} points")
        for point in points:
            print(point)
    else:
        break'''
