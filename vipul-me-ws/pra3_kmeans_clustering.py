import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('cancer_data.csv')
X = dataset.iloc[:, [1, 30]].values

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.xlabel('No of clusters (k)')
plt.xlabel('WCSS - Within-Cluster Sum of Square')
plt.title('The elbow method')
plt.show()


