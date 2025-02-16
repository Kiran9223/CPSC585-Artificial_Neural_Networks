import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, adjusted_rand_score
from ucimlrepo import fetch_ucirepo

# Load dataset (Update with correct file path)
dataset = fetch_ucirepo(id=891)
df = dataset.data.features  # DataFrame of features
y = dataset.data.targets

true_labels = y['Diabetes_binary'].values

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Start CPU Timer
start_time = time.time()

# Run KMeans on CPU (Using all cores)
kmeans_cpu = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels_cpu = kmeans_cpu.fit_predict(scaled_features)

# Compute execution time
cpu_time = time.time() - start_time

# Compute RMSE (Internal Index)
rmse_cpu = np.sqrt(mean_squared_error(scaled_features, kmeans_cpu.cluster_centers_[cluster_labels_cpu]))

# Compute ARI (External Index)
ari_cpu = adjusted_rand_score(true_labels, cluster_labels_cpu)

# Print results
print("\n===== K-Means Clustering (CPU) Results =====")
print(f"RMSE: {rmse_cpu:.4f}")
print(f"ARI: {ari_cpu:.4f}")
print(f"Execution Time: {cpu_time:.4f} seconds")


# Output: k=2 better alignment because of better ARI
# ===== K-Means Clustering (CPU) Results =====
# RMSE: 0.9416
# ARI: 0.1697
# Execution Time: 0.9405 seconds

# Output: k=3
# ===== K-Means Clustering (CPU) Results =====
# RMSE: 0.9155
# ARI: 0.1438
# Execution Time: 1.0388 seconds