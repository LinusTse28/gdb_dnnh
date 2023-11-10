import numpy as np

def create_grid(P, eps):
    """
    创建网格并将数据点分配到网格单元中
    :param P: 数据点集合，每行一个点 [x, y]
    :param eps: 网格单元大小
    :return: 网格字典，字典的键是网格单元的坐标 (x, y)，值是该网格单元中的点集合
    """
    grid = {}
    for point in P:
        x, y = point
        grid_x, grid_y = int(x // eps), int(y // eps)
        grid_cell = (grid_x, grid_y)

        if grid_cell in grid:
            grid[grid_cell].append(point)
        else:
            grid[grid_cell] = [point]

    return grid

def spiral_search(grid, q):
    """
    螺旋方向搜索网格单元中的数据点
    :param grid: 网格字典
    :param q: 查询点 [x, y]
    :return: 从查询点开始螺旋方向搜索的数据点
    """
    # 计算查询点所在的网格单元
    x, y = q
    grid_x, grid_y = int(x // eps), int(y // eps)
    grid_cell = (grid_x, grid_y)

    # 定义螺旋方向移动的偏移量
    dx, dy = 1, 0
    turns = 0  # 螺旋转数

    # 从查询点所在的网格单元开始螺旋搜索
    result = []
    while True:
        if grid_cell in grid:
            result.extend(grid[grid_cell])

        # 检查是否达到查询点附近的网格单元
        if abs(grid_cell[0] - grid_x) <= turns // 2 and abs(grid_cell[1] - grid_y) <= turns // 2:
            break

        # 移动到下一个网格单元
        grid_cell = (grid_cell[0] + dx, grid_cell[1] + dy)

        # 在螺旋转角时改变移动方向
        if grid_cell[0] == grid_x + turns // 2 and grid_cell[1] == grid_y - turns // 2:
            dx, dy = 0, 1
        elif grid_cell[0] == grid_x - turns // 2 and grid_cell[1] == grid_y - turns // 2:
            dx, dy = -1, 0
        elif grid_cell[0] == grid_x - turns // 2 and grid_cell[1] == grid_y + turns // 2:
            dx, dy = 0, -1
        elif grid_cell[0] == grid_x + turns // 2 and grid_cell[1] == grid_y + turns // 2:
            dx, dy = 1, 0
            turns += 1

    return result

# 设置网格单元大小
eps = 0.1

# 生成示例数据集
P = np.random.rand(100, 2)

# 查询点
q = [0.5, 0.5]

# 创建网格
grid = create_grid(P, eps)

# 从查询点开始螺旋方向搜索数据点
result = spiral_search(grid, q)

# 打印结果
print("查询点附近的数据点：", result)
