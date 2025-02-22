import pandas as pd
import numpy as np
import time
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, adjusted_rand_score
import numba
from numba import cuda
import math

# Fetch dataset from UCI repo (CDC Diabetes Health)
dataset = fetch_ucirepo(id=891)
df = dataset.data.features
y = dataset.data.targets.values.ravel()

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
true_labels = y

@cuda.jit
def compute_distances_kernel(X, centroids, distances):
    # Get 2D thread indices
    row, col = cuda.grid(2)
    
    # Check array boundaries
    if row < X.shape[0] and col < centroids.shape[0]:
        # Compute distance from point row to centroid col
        dist = 0.0
        for k in range(X.shape[1]):
            diff = X[row, k] - centroids[col, k]
            dist += diff * diff
        distances[row, col] = math.sqrt(dist)

def kmeans_gpu(X, n_clusters, max_iter=300):
    np.random.seed(42)
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly from the dataset
    idx = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[idx].copy()
    
    # Allocate memory for distances matrix
    distances = np.zeros((n_samples, n_clusters), dtype=np.float32)
    
    # Copy data to GPU
    d_X = cuda.to_device(X)
    d_distances = cuda.to_device(distances)
    
    # Configure CUDA kernel launch parameters (similar to regression code)
    threadsperblock = (16, 16)
    blockspergrid_x = (n_samples + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n_clusters + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Track labels from previous iteration for convergence
    prev_labels = np.zeros(n_samples, dtype=np.int32)
    
    for iteration in range(max_iter):
        # Copy current centroids to device memory
        d_centroids = cuda.to_device(centroids)
        
        # Launch the CUDA kernel with 2D grid configuration
        compute_distances_kernel[blockspergrid, threadsperblock](d_X, d_centroids, d_distances)
        
        # Synchronize to ensure computation is complete
        cuda.synchronize()
        
        # Copy distances back to host
        distances = d_distances.copy_to_host()
        
        # Assign points to nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.all(labels == prev_labels):
            break
        prev_labels = labels.copy()
        
        # Update centroids on CPU
        new_centroids = np.zeros((n_clusters, n_features), dtype=np.float32)
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]
        centroids = new_centroids
    
    # Clean up GPU memory
    d_X.copy_to_host()
    d_distances.copy_to_host()
    
    return labels, centroids

# Run the GPU-accelerated K-means and measure performance
start_time = time.time()
cluster_labels_gpu, centroids_gpu = kmeans_gpu(scaled_features, 2)
gpu_time = time.time() - start_time

# Compute validation metrics
rmse_gpu = np.sqrt(mean_squared_error(scaled_features, centroids_gpu[cluster_labels_gpu]))
ari_gpu = adjusted_rand_score(true_labels, cluster_labels_gpu)

print(f"Numba GPU KMeans -> RMSE: {rmse_gpu:.4f}, ARI: {ari_gpu:.4f}, Time: {gpu_time:.4f} seconds")

# For comparison: CPU implementation
# from sklearn.cluster import KMeans
# start_time = time.time()
# kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaled_features)
# sklearn_time = time.time() - start_time

# rmse_sklearn = np.sqrt(mean_squared_error(scaled_features, kmeans.cluster_centers_[kmeans.labels_]))
# ari_sklearn = adjusted_rand_score(true_labels, kmeans.labels_)

# print(f"sklearn KMeans (CPU) -> RMSE: {rmse_sklearn:.4f}, ARI: {ari_sklearn:.4f}, Time: {sklearn_time:.4f} seconds")