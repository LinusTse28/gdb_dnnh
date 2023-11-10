import pandas as pd

from utils import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def find_neighbors(p, P, epsilon):
    neighbors = []
    for pt, idx in P:
        if np.linalg.norm(p - pt) <= epsilon:
            neighbors.append([pt, idx])
    return neighbors


def expand_cluster(P, labels, idx_p, neighbors, clusterId, epsilon, minPts):
    labels[idx_p] = clusterId
    #assigned_indices = []
    i = 0

    while i < len(neighbors):
        neighbor, neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = clusterId
            #assigned_indices.append(neighbor_idx)
            new_neighbors = find_neighbors(neighbor, P, epsilon)
            if len(new_neighbors) >= minPts:
                neighbors.extend(new_neighbors)
                #new_neighbors_indices = [idx for _, idx in new_neighbors]
                #assigned_indices.extend(new_neighbors_indices)
        i += 1

    #return assigned_indices

def expand_algorithm(P, q, epsilon=0.009, minPts=5):
    # Add index for P
    P = np.array([[p, idx] for idx, p in enumerate(P)], dtype=object)
    P_save = P.copy()

    # Use mask to denote whether the point is available
    mask = np.full(P.shape[0], True, dtype=bool)
    # Use labels to denote the cluster id
    labels = np.full(P.shape[0], -1, dtype=int)

    clusterId = -1
    bound = 10000
    density_threshold = 2
    visited = np.full(P.shape[0], False, dtype=bool)
    while True:
        update = False
        Delta_old = 10000
        idx = 0

        while idx < len(P[mask]):
            visited[idx] = True
            mask = mask & np.array([np.linalg.norm(pt - q) <= bound for pt, idx in P_save]) & ~visited
            p, idx = P[mask][idx]
            if idx > len(P[mask]):
                break
            #print(idx, bound, len(P[mask]))
            neighbors = find_neighbors(p, P[mask], epsilon)
            if len(neighbors) >= minPts:  # change the condition here
                if labels[idx] == -1:  # only expand the cluster if the point is currently noise
                    clusterId += 1
                    expand_cluster(P[mask], labels, idx, neighbors, clusterId, epsilon, minPts)

            if clusterId >= 0:
                Ci = P[(labels == clusterId) & mask]
            else:
                continue

            if len(Ci) > 0:
                # Calculate new_mask
                new_bound = max([distance.euclidean(q, pt) for pt, idx in Ci])
                print('curId is ', clusterId, 'new bound is ', new_bound, ' and old bound is ', bound)
                new_mask = mask & np.array(
                    [np.linalg.norm(pt - q) <= new_bound for pt, idx in P_save]) & ~visited

                if new_bound < bound:
                    # Calculate Delta_old and Delta_new
                    Delta_new = delta(q, rmv_idx(P[(labels == clusterId) & new_mask]))
                    print("clusterId:", clusterId, ',', "len(Ci):", len(Ci),',',"Delta_new:", Delta_new,',',"Delta_old:", Delta_old)
                    if clusterId == 0 and len(Ci) >= minPts * density_threshold:
                        update = True
                        bound = new_bound
                        mask = new_mask
                        Delta_old = Delta_new  # Update Delta_old
                    elif len(Ci)>= minPts * density_threshold and Delta_new < Delta_old:

                        update = True
                        bound = new_bound
                        mask = new_mask
                        Delta_old = Delta_new  # Update Delta_old
            else:
                idx = idx + 1
                mask = mask & ~visited
                continue

            '''for cur_clusterId in range(clusterId):
                # Select the current cluster_id
                Ci = P[(labels == cur_clusterId) & mask]

                if len(Ci) > 0:
                    # Calculate new_mask
                    new_bound = max([distance.euclidean(q, pt) for pt, idx in Ci])
                    print('curId is ', cur_clusterId, 'new bound is ',new_bound, ' and old bound is ', bound)
                    new_mask = mask & np.array([np.linalg.norm(pt - q) <= new_bound for pt, idx in P_save]) & ~visited

                    if new_bound < bound:
                    # Calculate Delta_old and Delta_new
                        Delta_new = delta(q, rmv_idx(P[(labels == cur_clusterId) & new_mask]))
                        if cur_clusterId == 0 and len(Ci) >= minPts * density_threshold:
                            update = True
                            bound = new_bound
                            mask = new_mask
                            Delta_old = Delta_new  # Update Delta_old
                        elif lenC(Ci) >= minPts * density_threshold and Delta_new < Delta_old:

                            update = True
                            bound = new_bound
                            mask = new_mask
                            Delta_old = Delta_new  # Update Delta_old

                    else:
                        break'''

            mask = mask & ~visited

            outside_mask = np.array([np.linalg.norm(pt - q) > bound for pt, idx in P_save])
            labels[outside_mask] = -1
            idx = idx + 1
            if idx >= len(P[mask]):
                break

        outside_mask = np.array([np.linalg.norm(pt - q) > bound for pt, idx in P_save])
        labels[outside_mask] = -1
        # End condition
        if not update or np.all(labels != -1) or not np.any(np.array(density_threshold) * minPts <= len(P[mask])):
            break

        #print(update)


    # Remove index for P
    P = rmv_idx(P)
    clusters = [[i, P[(labels == i) & mask]] for i in range(clusterId) if len(P[(labels == i) & mask]) > 0]
    clusters = sorted(clusters, key=lambda c: delta(q, c[1]))

    # Return clusters, labels, and bound
    return clusters, labels, bound


def plot_clusters(data, clusters, labels, q, bound):
    plt.figure()
    # Create colormap
    cmap = cm.get_cmap('rainbow', len(clusters))

    # Create color list
    colors = ['grey' if label == -1 else cmap(label) for label in labels]

    # Plot data pts
    plt.scatter(data[:, 0], data[:, 1], c='gray')

    # Plot clusters
    for i, cluster_data in clusters:
        '''cluster_data = np.array(cluster_data)  # ensure cluster_data is a 2D data
        centroid = np.mean(cluster_data, axis=0)
        color = cmap(i)
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color)
        plt.scatter(centroid[0], centroid[1], marker='x', color='black', s=100)
        plt.text(centroid[0], centroid[1], str(i), color='black', fontsize=12)'''
        if i == clusters[0][0]:
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color='yellow', edgecolor='black', linewidth=1)
            #print('plot best cluster is ', clusters[0][1], len(clusters[0][1]))
    plt.scatter(q[0], q[1], color='red', marker='+', s=100)

    # Plot boundary circle
    circle = plt.Circle((q[0], q[1]), bound, color='blue', fill=False)

    plt.gca().add_patch(circle)
    plt.title("Clustering Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.show()


if __name__ == "__main__":
    P = pd.read_csv('/Users/shirley/Desktop/data/RN_30P_100K_1S.csv')
    q = np.array([0.5, 0.5])

    # Sort by distances between q and P
    P = np.array(sorted(P, key=lambda p: distance.euclidean(p[0], q)))

    clusters, labels, bound = expand_algorithm(P, q)

    plot_clusters(P, clusters, labels, q, bound)
    #print(clusters, len(clusters[0][1]))
    print("Best Cluster: ",clusters[0][0], ',',  bound)
