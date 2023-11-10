import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
import pandas as pd
from utils import *


def auto_epsilon(P, minPts=5):
    X_scaled = StandardScaler().fit_transform(P)
    k = minPts

    neigh = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nbrs = neigh.fit(X_scaled)
    # distances, meadia method, we obtain the average of the matrix of distances, we leave out the first neighbor since it is the same point.
    distances1, indices = nbrs.kneighbors(X_scaled)
    distances1 = np.sort(distances1, axis=0)
    distances1 = distances1[:, 1:].mean(1)

    # método del último vecino
    distances_last, indices_last = nbrs.kneighbors(X_scaled)
    distances_last = np.sort(distances_last, axis=0)[:, k - 1]

    kn = KneeLocator(range(1, len(distances1) + 1), distances1, curve='convex', direction='increasing',
                     S=1, )  # interp_method='polynomial'
    #print('The value for the optimal epsilon is in: ', kn.knee)


    """ Last kneighbors distances"""
    kn2 = KneeLocator(range(1, len(distances_last) + 1), distances_last, curve='convex', direction='increasing', S=1,
                      interp_method='polynomial')  # interp_method='polynomial'
    ''''/Users/shirley/Desktop/data/RN_1.0S_100K_50P.csv'''

    distancia = round(distances_last[kn.knee], 5)
    epsilon = round(distances_last[kn2.knee], 5)
    #print('eps1: ', distancia, ', eps2: ', epsilon)

    return distancia

def eps_time(path):

    P = np.array(pd.read_csv(path))
    startTime = time.time()
    eps = auto_epsilon(P)
    print(eps)
    print('time of ', path, 'is: ', time.time()-startTime)

'''if __name__ == '__main__':
    paths = ['/Users/shirley/Desktop/data/UN_10K.csv',
             '/Users/shirley/Desktop/data/UN_50K.csv',
             '/Users/shirley/Desktop/data/UN_100K.csv',
             '/Users/shirley/Desktop/data/UN_150K.csv',
             '/Users/shirley/Desktop/data/UN_200K.csv']
    for path in paths:
        eps_time(path)
base_path = '/Users/linus/Desktop/data/'
variants = [
        '0.0S', '0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '1.0S', '1.5S', '2.0S', '2.5S', '3.0S'
    ]

file_names = [f"RN_{variant}_100K_50P.csv" for variant in variants]
data_paths = [os.path.join(base_path, file_name) for file_name in file_names]
for path in data_paths:
    eps_time(path)'''
