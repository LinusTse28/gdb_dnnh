from auto_epsilon import auto_epsilon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data_path = '/Users/linus/Desktop/data/labeled/rn/RN_100K_50P_0.0S.csv'

'''X = np.array(pd.read_csv(data_path))
X_scale = StandardScaler().fit_transform(X)
# 使用DBSCAN进行聚类
epsilon =auto_epsilon(X_scale)
print(epsilon)
db = DBSCAN(eps=0.04, min_samples=5)
labels = db.fit_predict(X_scale)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=1)
plt.title('DBSCAN clustering')
plt.colorbar()
plt.show()
'''
data = pd.read_csv(data_path)
X = data.iloc[:, :2].values
y = data.iloc[:, 2].values

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=1)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.title('Visualization of Labeled Data')
plt.show()