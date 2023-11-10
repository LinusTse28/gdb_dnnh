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
    grid = np.zeros(grid_shape)

    cells_dict = assign_points_to_cells(data, qx, qy, layers, grid_shape, cell_size)

    # 获取螺旋形的单元格
    spiral_indices = spiral_cells(qx, qy, layers, grid_shape, cell_size)

    # 为每个单元格填充颜色
    for count, (row, col) in enumerate(spiral_indices, 1):
        ax.add_patch(plt.Rectangle((col * cell_size, row * cell_size), cell_size, cell_size, fill=None, alpha=1,
                                   edgecolor='blue'))

        # 显示前9个网格的数据点数量
        if count <= 9:  # 仅检查前9个网格
            points_in_cell = cells_dict.get((row, col), [])
            ax.text(col * cell_size + cell_size / 2, row * cell_size + cell_size / 2, f'{len(points_in_cell)}',
                    va='center', ha='center', size=8)

        # 画出数据点
        for cell, points in cells_dict.items():
            y, x = cell
            x_values = [p[0] for p in points]
            y_values = [p[1] for p in points]
            ax.scatter(x_values, y_values, s=0.5, c='black')

    # Create a normalized grid for displaying gray-scale
    normalized_grid = grid / len(cells)

    plt.imshow(normalized_grid, cmap='gray_r', extent=(0, grid_shape[1] * cell_size, grid_shape[0] * cell_size, 0))
    for (y, x), grid_value in np.ndenumerate(grid):
        if grid_value:
            plt.text(x * cell_size + cell_size / 2, y * cell_size + cell_size / 2,
                     f"{int(grid_value)}",
                     ha='center', va='center', color='black', fontsize=6)
    plt.grid(which='both', linestyle='-', linewidth=0.5, color='gray')
    # plt.xticks(np.arange(0, grid_shape[1] * cell_size, cell_size))
    # plt.yticks(np.arange(0, grid_shape[0] * cell_size, cell_size))
    x_values = data[:, 0]  # 假设第一列是x值
    y_values = data[:, 1]  # 假设第二列是y值
    plt.scatter(x_values, y_values, s=0.5)
    plt.title("Spiral Order")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(qx, qy, c='red', s=2)
    plt.show()


# Example usage
data = np.array(pd.read_csv('/Users/linus/Desktop/data/RN_50K_50P_1S.csv'))
eps = auto_epsilon(data)

cell_size = eps / sqrt(2)
grid_shape = (round(1 / cell_size), round(1 / cell_size))
layers = int(round(1 / cell_size) / 2) + 1

print('eps: ', eps, '\ncellsize :', cell_size, '\nlayers: ', layers)
'''cell_size = 0.1  # Cell size of 0.01x0.01 in the data range 0-1
grid_shape = (10, 10)  # 100x100 grids

layers = 10  # Number of layers to spiral out'''
qx, qy = 0.19, 0.92  # Starting point in the data range
# visualize_spiral(grid_shape, qx, qy, layers, cell_size, data)

'''cells = spiral_cells(qx, qy, layers, grid_shape, cell_size)
print(cells)
'''
assigned_cells = assign_points_to_cells(data, qx, qy, layers, grid_shape, cell_size)

# 打印前9个网格内的数据点
for idx, (cell_idx, points) in enumerate(assigned_cells.items()):
    if idx < 9:  # 我们只想打印前9个网格
        print(f"Cell {cell_idx} has {len(points)} points")
        for point in points:
            print(point)
    else:
        break

visualize_spiral(grid_shape, qx, qy, layers, cell_size, data)