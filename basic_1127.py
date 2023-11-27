import matplotlib.pyplot as plt
from auto_epsilon import *

# 找到最近的集群
def find_closest_cluster(data, query_point, epsilon, minPts):
    data_scale = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=epsilon, min_samples=minPts).fit(data_scale)
    labels = db.labels_
    clusters = [data[labels == i] for i in range(max(labels) + 1) if np.sum(labels == i) >= (3 * minPts)]
    closest_cluster = None
    print('nums of clusters: ', len(clusters))
    if len(clusters) > 0:
        #print('len(clusters): ', len(clusters))
        delta_values = [delta_(query_point, cluster) for cluster in clusters]
        closest_cluster = clusters[np.argmin(delta_values)]
        #print(closest_cluster)
    return closest_cluster, labels

# 可视化结果
def visualize_results(data, query_point, labels, closest_cluster, data_path, x0_range, x1_range):
    # print(labels)
    # labels[i] = -1 means that the point i is an outlier
    plt.scatter(data[:, 0], data[:, 1], c=labels, label="other", s=1)

    plt.scatter(closest_cluster[:, 0], closest_cluster[:, 1], c='red', label="closest cluster", s=1)
    plt.scatter(query_point[0], query_point[1], c='red', marker='X', label="query point")
    #plt.legend(loc='upper right')
    plt.title(f"Clustering Result of {data_path}")
    plt.xlim(x0_range)
    plt.ylim(x1_range)
    '''plt.xlim(0, 1)
    plt.ylim(0, 1)'''

    plt.show()

def run(path):
    q = np.random.rand(2)
    # print(q)
    x0_range = (q[0] - 0.1, q[0] + 0.1)
    x1_range = (q[1] - 0.1, q[1] + 0.1)
    P, labels = load_data_from_csv_labeled(data_path)
    minPts = 4
    eps = auto_epsilon(P, minPts)
    print(eps)
    startTime = time.time()
    closest_cluster, labels = find_closest_cluster(P, q, epsilon=eps, minPts=minPts)
    execution_time = time.time() - startTime
    visualize_results(P, q, labels, closest_cluster, path, x0_range, x1_range)
    cluster = closest_cluster
    nearest_idx = find_nearest_points_kd(P, cluster)
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

    return {
        "time": execution_time,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# 测试代码
if __name__ == "__main__":

    #q = np.array([0.19, 0.92])

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
    all_results = []
    data_paths = ['/Users/linustse/Desktop/data/labeled/rn/RN_100K_50P_0.1S.csv']

    for data_path in data_paths:
        experiment_results = []
        for i in range(1):
            result = run(data_path)
            '''result['data_path'] = data_path
            experiment_results.append(result)

        all_results.extend(experiment_results)

    df_results = pd.DataFrame(all_results)
    df_results.to_excel('/Users/linustse/Desktop/experiment_results.xlsx', index=False)
'''


