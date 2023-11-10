import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from scipy.spatial import distance

def create_fake_data(n_samples=300,
                     n_features=2, 
                     centers=10,
                     cluster_std=1.0,
                     random_state=17):

    data, _ = make_blobs(n_samples=n_samples,
                         n_features=n_features,
                         centers=centers,
                         cluster_std=cluster_std,
                         random_state=random_state)
    return data


def density(cluster):
    dist_sum = sum([distance.euclidean(i, j) for i in cluster for j in cluster])
    return dist_sum / len(cluster)

def delta(q, cluster):
    return distance.euclidean(q, np.mean(cluster, axis=0)) ** 2 / density(cluster)

def approximate_density(cluster):

    x_values = [p[0] for p in cluster]
    y_values = [p[1] for p in cluster]
    
    x_median = np.median(x_values)
    y_median = np.median(y_values)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    approximate_area = min(abs(x_min - x_median), abs(x_max - x_median)) * \
                       min(abs(y_min - y_median), abs(y_max - y_median))
    
    return len(cluster) / approximate_area

def delta_(q, cluster):
    return distance.euclidean(q, np.mean(cluster, axis=0)) ** 2 / approximate_density(cluster)

def rmv_idx(cluster):
    return np.array([pt for pt, idx in cluster])

