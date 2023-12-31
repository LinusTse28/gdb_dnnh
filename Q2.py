import os
from auto_epsilon import auto_epsilon
from utils import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


def find_neighbors(p, P, epsilon, visited):
    neighbors = []
    for pt, idx in P:
        if np.linalg.norm(p - pt) <= epsilon and visited[idx]==False:
            neighbors.append([pt, idx])

    return neighbors

def expand_cluster(P, labels, idx_p, neighbors, clusterId, epsilon, minPts, visited):
    labels[idx_p] = clusterId
    # assigned_indices = []
    i = 0

    while i < len(neighbors):
        #print(f"Starting expand_cluster loop with i={i}...")
        neighbor, neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1 and visited[neighbor_idx] == False:
            labels[neighbor_idx] = clusterId
            visited[neighbor_idx] = True
            new_neighbors = find_neighbors(neighbor, P, epsilon, visited)
            if len(new_neighbors) >= minPts and visited[neighbor_idx] == False:
                neighbors.extend(new_neighbors)

        i += 1
        # print(f"Ending expand_cluster loop with i={i}...")


def expand_algorithm(P, q, epsilon, minPts=5):
    # Add index for P
    P = np.array([[p, idx] for idx, p in enumerate(P)], dtype=object)
    P_save = P.copy()

    # Use mask to denote whether the point is available
    mask = np.full(P.shape[0], True, dtype=bool)
    # Use labels to denote the cluster id
    labels = np.full(P.shape[0], -1, dtype=int)

    clusterId = -1
    bound = 10000
    density_threshold = 1
    visited = np.full(P.shape[0], False, dtype=bool)
    while True:
        # print("Starting main while loop...\n")
        update = False
        Delta_old = 10000
        idx = 0

        while idx < len(P[mask]):
            if visited[idx] == True:
                idx = idx + 1
                continue
            #print('starting processing', idx, ' th data point')
            visited[idx] = True
            mask = mask & np.array([np.linalg.norm(pt - q) <= bound for pt, idx in P_save])
            # mask = mask_save & (~visited)
            if idx >= len(P[mask]):
                update = False
                break

            p, idx = P[mask][idx]

            # print(idx, bound, len(P[mask]))
            neighbors = find_neighbors(p, P[mask], epsilon, visited)

            #print(len(neighbors))
            if len(neighbors) >= minPts:  # change the condition here
                if labels[idx] == -1:  # only expand the cluster if the point is currently noise
                    clusterId += 1
                    #print(clusterId)
                    expand_cluster(P[mask & ~visited], labels, idx, neighbors, clusterId, epsilon, minPts, visited)
                    visited[labels != -1] = True
                    # plot_current_cluster(P_save, labels, clusterId, q, bound)

                    '''for _, neighbor_idx in neighbors:
                        visited[neighbor_idx] = True'''
            else:
                idx = idx + 1
                continue

            if clusterId >= 0:
                Ci = P[(labels == clusterId) & mask]
                #print('len(Ci): ',len(Ci))
            else:
                idx = idx + 1
                continue

            if len(Ci) > 1:
                # Calculate new_mask
                new_bound = max([distance.euclidean(q, pt) for pt, idx in Ci])
                #print('curId is ', clusterId, 'new bound is ', new_bound, ' and old bound is ', bound, 'len(Ci): ',len(Ci))

                new_mask = mask & np.array(
                    [np.linalg.norm(pt - q) <= new_bound for pt, idx in P_save])

                if new_bound < bound:
                    # Calculate Delta_old and Delta_new
                    #Delta_new = delta(q, rmv_idx(P[(labels == clusterId)]))
                    Delta_new = delta(q, rmv_idx(P[(labels == clusterId)]))
                    # print("clusterId:", clusterId, ',', "len(Ci):", len(Ci), ',', "Delta_new:", Delta_new, ',', "Delta_old:", Delta_old)

                    if len(Ci) >= minPts * density_threshold and Delta_new < Delta_old or clusterId == 0:
                        update = True
                        bound = new_bound
                        mask = new_mask
                        Delta_old = Delta_new  # Update Delta_old
                # print('cluster ID: ', clusterId, ', len(Ci): ', len(Ci))
            else:
                labels[~mask] = -1
                idx = idx + 1
                continue

            labels[~mask] = -1
            idx = idx + 1

        # ("Ending main while loop...")

        # End condition
        if not update or np.all(labels != -1) or density_threshold * minPts > len(P[mask]):
            break

    # Remove index for P
    P = rmv_idx(P)
    # print(clusterId)
    clusters = [[i, P[(labels == i) & mask]] for i in range(clusterId + 1) if len(P[(labels == i) & mask]) > 0]
    # clusters = [[i, P[(labels == i) & mask]] for i in range(clusterId) if len(P[(labels == i) & mask]) > 0]
    clusters = sorted(clusters, key=lambda c: delta(q, c[1]))
   # clusters = sorted(clusters, key=lambda c: delta(q, c[1]))

    # Return clusters, labels, and bound
    return clusters, labels, bound


def plot_clusters(data, clusters, labels, q, bound, data_path):
    plt.figure()
    # Create colormap
    cmap = cm.get_cmap('rainbow', len(clusters))

    # Create color list
    colors = ['grey' if label == -1 else cmap(label) for label in labels]

    # Plot data pts
    plt.scatter(data[:, 0], data[:, 1], c='gray', s=1)

    # Plot clusters
    for i, cluster_data in clusters:
        color = cmap(i)
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color, s=1)
        if i == clusters[0][0]:
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color='yellow', s=1)
            # print('plot best cluster is ', clusters[0][1], len(clusters[0][1]))
    plt.scatter(q[0], q[1], color='red', marker='+', s=100)

    # Plot boundary circle
    circle = plt.Circle((q[0], q[1]), bound, color='blue', fill=False)

    plt.gca().add_patch(circle)
    plt.title(f"Clustering Result of {data_path}")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.xlim(0.1, 0.3)
    plt.ylim(0.8, 1)

    plt.show()


from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_clusters(clusters, labeled_data):
    # 从标记数据中获取真实标签
    true_labels = labeled_data[:, 2]

    # 获取从您的 clusters 结果中提取的预测标签
    predicted_labels = -np.ones(len(labeled_data), dtype=int)  # 初始化为-1（未分类标签）
    for cluster_id, cluster_points in clusters:
        mask = np.isin(labeled_data[:, :2], cluster_points).all(axis=1)
        predicted_labels[mask] = cluster_id

    # 仅评估已分类的数据点
    mask = predicted_labels != -1
    true_labels = true_labels[mask]
    predicted_labels = predicted_labels[mask]

    # 计算性能指标
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')

    return precision, recall, f1


def run(path, q):

    P = np.array(pd.read_csv(path))
    P = np.array(sorted(P, key=lambda p: np.linalg.norm(p - q)))
    #P = np.array([[p, idx] for idx, p in enumerate(P_sorted)], dtype=object)

    eps = auto_epsilon(P)

    #print(path, 'epsilon time is ', time.time() - startTime, '\nepsilon is ', eps)
    # q = np.array([0.5, 0.5])
    # Sort by distances between q and P

    startTime = time.time()
    clusters, labels, bound = expand_algorithm(P, q, epsilon=eps)
    #print('processing time is ', time.time() - startTime)
    print(time.time() - startTime)
    plot_clusters(P, clusters, labels, q, bound, path)
    # print(clusters)
    '''labeled_data = df
    precision, recall, f1 = evaluate_clusters(clusters, labeled_data)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")'''

if __name__ == "__main__":
    q = np.array([0.19, 0.92])
    #q = np.random.rand(2)
    #print(q)
    base_path = '/Users/linus/Desktop/data/'

    variants = [
        '0.0S', '0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '1.0S', '1.5S', '2.0S', '2.5S', '3.0S'
    ]
    file_names = [f"RN_{variant}_100K_50P.csv" for variant in variants]
    variants = [
        '1', '5', '10', '15', '20'
    ]
    file_names = [f"RN_{variant}0K_50P_1S.csv" for variant in variants]
    data_paths = [os.path.join(base_path, file_name) for file_name in file_names]

    for data_path in data_paths:
        run(data_path, q)
