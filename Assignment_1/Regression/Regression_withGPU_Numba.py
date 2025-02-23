import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import platform
import psutil
from ucimlrepo import fetch_ucirepo
from numba import cuda

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------

# System Specifications
print(f"OS: {platform.system()} {platform.release()}")
print(f"CPU: {platform.processor()}")
print(f"Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
print(f"RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")

# Fetch the dataset from UCI Repository
individual_household_electric_power_consumption = fetch_ucirepo(id=235)

# Load the dataset
X = individual_household_electric_power_consumption.data.features
features = ['Global_reactive_power', 'Voltage', 'Global_intensity']
target = 'Global_active_power'

# Convert columns to numeric (replace non-numeric entries with NaN)
cols_to_convert = features + [target]
for col in cols_to_convert:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Drop rows with missing values
X_clean = X.dropna(subset=cols_to_convert)

# Extract features and target, converting them to float
X = X_clean[features].astype(np.float16)
y = X_clean[target].astype(np.float16)

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features to improve numerical stability
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Generate polynomial features (degree 3)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Convert targets to NumPy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Move data to GPU
X_train_gpu = cuda.to_device(X_train_poly)
y_train_gpu = cuda.to_device(y_train)

# -----------------------------
# Define CUDA-Accelerated Function for Linear Regression
# -----------------------------
@cuda.jit
def compute_beta_gpu(X, y, beta, XtX, Xty):
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
# Compute Regression Coefficients Using CUDA
# -----------------------------
beta_gpu = cuda.device_array(X_train_poly.shape[1])
XtX_gpu = cuda.device_array((X_train_poly.shape[1], X_train_poly.shape[1]))
Xty_gpu = cuda.device_array(X_train_poly.shape[1])

threadsperblock = (16, 16)
blockspergrid_x = (X_train_poly.shape[1] + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (X_train_poly.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

start_time = time.time()
compute_beta_gpu[blockspergrid, threadsperblock](X_train_gpu, y_train_gpu, beta_gpu, XtX_gpu, Xty_gpu)
cuda.synchronize()

# Solve for beta using NumPy after moving back to CPU
XtX = XtX_gpu.copy_to_host()
Xty = Xty_gpu.copy_to_host()
beta = np.linalg.solve(XtX, Xty)
numba_time = time.time() - start_time

# -----------------------------
# Predictions on Training and Testing Sets
# -----------------------------
y_train_pred = np.dot(X_train_poly, beta)
y_test_pred = np.dot(X_test_poly, beta)

# -----------------------------
# Calculate Performance Metrics
# -----------------------------
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# -----------------------------
# Results
# -----------------------------
print("Coefficient Vector (first 10 shown):", beta[:10])
print("Training RMSE:", train_rmse)
print("Training R^2:", train_r2)
print("Testing RMSE:", test_rmse)
print("Testing R^2:", test_r2)
print("Computation Time: {:.2f} seconds".format(numba_time))
