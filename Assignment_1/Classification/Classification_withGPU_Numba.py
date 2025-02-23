import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from numba import cuda
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import time
import platform
import psutil 


print(f"OS: {platform.system()} {platform.release()}")
print(f"CPU: {platform.processor()}")
print(f"Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
print(f"RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
# Fetch the dataset
# Load dataset
dataset = fetch_ucirepo(id=891)
X = dataset.data.features.to_numpy()
y = dataset.data.targets.to_numpy().flatten()  # Convert to 1D array

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept term (bias) by adding a column of ones
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to NumPy arrays for GPU processing
X_train_gpu = cuda.to_device(X_train)
y_train_gpu = cuda.to_device(y_train)

# -----------------------------
# Define CUDA-Accelerated Function to Compute Normal Equation for Logistic Regression
# -----------------------------
@cuda.jit
def compute_xtx_xty(X, y, XtX, Xty):
    row, col = cuda.grid(2)
    
    if row < XtX.shape[0] and col < XtX.shape[1]:
        # Compute X^T * X
        XtX[row, col] = 0.0
        for k in range(X.shape[0]):
            XtX[row, col] += X[k, row] * X[k, col]
    
    if row < Xty.shape[0] and col == 0:
        # Compute X^T * y
        Xty[row] = 0.0
        for k in range(X.shape[0]):
            Xty[row] += X[k, row] * y[k]

# -----------------------------
# Compute Weights Using Normal Equation
# -----------------------------
beta_gpu = cuda.device_array(X_train.shape[1])
XtX_gpu = cuda.device_array((X_train.shape[1], X_train.shape[1]))
Xty_gpu = cuda.device_array(X_train.shape[1])

threadsperblock = (16, 16)
blockspergrid_x = (X_train.shape[1] + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (X_train.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

start_time = time.time()
compute_xtx_xty[blockspergrid, threadsperblock](X_train_gpu, y_train_gpu, XtX_gpu, Xty_gpu)
cuda.synchronize()

# Solve for beta using NumPy after moving back to CPU
XtX = XtX_gpu.copy_to_host()
Xty = Xty_gpu.copy_to_host()
beta = np.linalg.solve(XtX, Xty)
numba_time = time.time() - start_time

# -----------------------------
# Predictions on Testing Set
# -----------------------------
y_test_pred = 1 / (1 + np.exp(-np.dot(X_test, beta)))  # Sigmoid function
y_test_pred_bin = (y_test_pred > 0.5).astype(float)  # Convert to binary labels

# -----------------------------
# Calculate Performance Metrics
# -----------------------------
conf_matrix = confusion_matrix(y_test, y_test_pred_bin)
accuracy = accuracy_score(y_test, y_test_pred_bin)
precision = precision_score(y_test, y_test_pred_bin)

# -----------------------------
# Results
# -----------------------------
print("Coefficient Vector (first 10 shown):", beta[:10])
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Computation Time: {:.2f} seconds".format(numba_time))

# Get GPU device information
gpu = cuda.get_current_device()
print(f"GPU Name: {gpu.name}")

