import os
import matplotlib.pyplot as plt
from auto_epsilon import *

# 找到最近的集群
def find_closest_cluster(data, query_point, epsilon):
    data_scale = StandardScaler().fit_transform(data)
    # 使用 DBSCAN 对数据进行聚类
    db = DBSCAN(eps=epsilon, min_samples=5).fit(data_scale)
    # 得到聚类标签
    labels = db.labels_
    # 找到最小的 Δ 对应的密集组
    clusters = [data[labels == i] for i in range(max(labels) + 1) if np.sum(labels == i) > 1]
    if len(clusters) > 0:
        delta_values = [delta_(query_point, cluster) for cluster in clusters]
        # 找到 Δ 值最小的集群
        closest_cluster = clusters[np.argmin(delta_values)]
    return closest_cluster, labels

# 可视化结果
def visualize_results(data, query_point, labels, closest_cluster, data_path):
    # print(labels)
    # labels[i] = -1 means that the point i is an outlier
    plt.scatter(data[:, 0], data[:, 1], c=labels, label="other", s=1)

    plt.scatter(closest_cluster[:, 0], closest_cluster[:, 1], c='red', label="closest cluster", s=1)
    plt.scatter(query_point[0], query_point[1], c='red', marker='X', label="query point")
    #plt.legend(loc='upper right')
    plt.title(f"Clustering Result of {data_path}")
    plt.xlim(0, 0.5)
    plt.ylim(0.5, 1)
    plt.show()

def run(path, q):
    P = np.array(pd.read_csv(path))
    eps = auto_epsilon(P)
    #eps =0.01093
    #print(eps)

    startTime = time.time()
    closest_cluster, labels = find_closest_cluster(P, q, epsilon=eps)
    print(time.time() - startTime)
    visualize_results(P, q, labels, closest_cluster, path)
    #print(len(closest_cluster))
# 测试代码
if __name__ == "__main__":
    '''data = create_fake_data()

    query_point = np.array([0.0, 0.0]) # 查询点'''
    q = np.array([0.19, 0.92])
    #q = np.random.rand(2)
    #print(q)
    base_path = '/Users/linus/Desktop/data/'

    '''variants = [
        '0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '1.0S', '1.5S', '2.0S', '2.5S', '3.0S'
    ]

    file_names = [f"RN_{variant}_100K_50P.csv" for variant in variants]'''
    variants = [
        '1', '5', '10', '15', '20'
    ]
    file_names = [f"RN_{variant}0K_50P_1S.csv" for variant in variants]
    data_paths = [os.path.join(base_path, file_name) for file_name in file_names]
    for data_path in data_paths:
        run(data_path, q)

    '''closest_cluster, labels = find_closest_cluster(data, q)
    visualize_results(data, q, labels, closest_cluster)'''

