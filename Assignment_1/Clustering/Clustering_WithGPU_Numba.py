import pandas as pd
import numpy as np
import time
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, adjusted_rand_score
import numba
from numba import jit

# Fetch dataset from UCI repo
dataset = fetch_ucirepo(id=891)
df = dataset.data.features  # Feature dataframe
y = dataset.data.targets.values.ravel()  # Convert targets to numpy array

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Store true labels for evaluation
true_labels = y

@jit(nopython=True)
def euclidean_distance(X, centroids):
    n_samples, n_features = X.shape
    n_clusters = centroids.shape[0]
    distances = np.empty((n_samples, n_clusters))

    for i in range(n_samples):
        for j in range(n_clusters):
            distances[i, j] = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))
    
    return distances

@jit(nopython=True)
def kmeans_numba(X, n_clusters, max_iter=300):
    np.random.seed(42)
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]  # Initialize centroids
    
    for _ in range(max_iter):
        distances = euclidean_distance(X, centroids)
        labels = np.argmin(distances, axis=1)
        
        # Ensure centroids have the same shape throughout
        new_centroids = np.zeros((n_clusters, n_features))  
        
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if cluster_points.shape[0] > 0:  # Avoid empty cluster issue
                new_centroids[i] = cluster_points.sum(axis=0) / cluster_points.shape[0]
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

# Run KMeans with Numba (Without Parallelization)
start_time = time.time()
cluster_labels_numba, centroids_numba = kmeans_numba(scaled_features, 2)  # k=3
numba_time = time.time() - start_time

# Compute RMSE & ARI for Numba
rmse_numba = np.sqrt(mean_squared_error(scaled_features, centroids_numba[cluster_labels_numba]))
ari_numba = adjusted_rand_score(true_labels, cluster_labels_numba)

print(f"Numba KMeans -> RMSE: {rmse_numba:.4f}, ARI: {ari_numba:.4f}, Time: {numba_time:.4f} seconds")

# Output: k=3
# Numba KMeans -> RMSE: 0.9157, ARI: 0.0835, Time: 6.8645 seconds

# Output: k=2 better ARI
# Numba KMeans -> RMSE: 0.9416, ARI: 0.1691, Time: 14.2344 seconds



