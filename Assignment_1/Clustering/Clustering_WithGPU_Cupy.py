import cupy as cp
import numpy as np
import time
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, adjusted_rand_score

# Fetch dataset from UCI repo
dataset = fetch_ucirepo(id=891)
df = dataset.data.features  # Feature dataframe
y = dataset.data.targets.values.ravel()  # Convert targets to numpy array

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Store true labels for evaluation
true_labels = y

def kmeans_cupy(X, n_clusters, max_iter=300):
    """Optimized K-Means clustering using CuPy for GPU acceleration"""
    X_gpu = cp.array(X)  # Move data to GPU
    n_samples, n_features = X_gpu.shape

    # Initialize centroids randomly
    random_indices = cp.random.choice(n_samples, n_clusters, replace=False)
    centroids = X_gpu[random_indices]

    for _ in range(max_iter):
        # Compute distances using CuPy broadcasting (faster than loops)
        distances = cp.sqrt(cp.sum((X_gpu[:, cp.newaxis, :] - centroids) ** 2, axis=2))
        labels = cp.argmin(distances, axis=1)  # Assign clusters

        # Compute new centroids efficiently using CuPy
        new_centroids = cp.zeros((n_clusters, n_features))
        counts = cp.bincount(labels, minlength=n_clusters).astype(cp.float32).reshape(-1, 1)

        for i in range(n_clusters):
            new_centroids[i] = cp.sum(X_gpu[labels == i], axis=0) / (counts[i] + 1e-10)  # Avoid division by zero

        # Check convergence
        if cp.allclose(centroids, new_centroids, atol=1e-4):  # More stable convergence check
            break
        centroids = new_centroids

    return cp.asnumpy(labels), cp.asnumpy(centroids)  # Move results back to CPU

# Run KMeans with CuPy (GPU) for k=3
start_time = time.time()
cluster_labels_cupy, centroids_cupy = kmeans_cupy(scaled_features, 2)  # k=2
cupy_time = time.time() - start_time

# Compute RMSE & ARI for CuPy
rmse_cupy = np.sqrt(mean_squared_error(scaled_features, centroids_cupy[cluster_labels_cupy]))
ari_cupy = adjusted_rand_score(true_labels, cluster_labels_cupy)

print(f"CuPy (GPU) KMeans -> RMSE: {rmse_cupy:.4f}, ARI: {ari_cupy:.4f}, Time: {cupy_time:.4f} seconds")

# Output:
# CuPy (GPU) KMeans -> RMSE: 0.9416, ARI: 0.1690, Time: 0.9852 seconds