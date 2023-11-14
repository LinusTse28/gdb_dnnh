from auto_epsilon import *
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def load_data_from_csv(file_path):
    return np.array(pd.read_csv(file_path))

data_paths = ['/Users/linustse/Desktop/data/RN_50K_50P_1S.csv']

for data_path in data_paths:
    X = load_data_from_csv(data_path)
    eps = auto_epsilon(X)
    print(eps)

    minPts = 5
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=5)

    # 执行DBSCAN聚类算法
    y_db = db.fit_predict(X)

    n_clusters_ = len(set(y_db)) - (1 if -1 in y_db else 0)
    n_noise_ = list(y_db).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # 可视化聚类结果
    # 给不同的簇分配不同的颜色
    unique_labels = set(y_db)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (y_db == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=1)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
