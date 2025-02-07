import pandas as pd
import numpy as np
import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
from ucimlrepo import fetch_ucirepo

# Fetch the dataset from UCI Repository
individual_household_electric_power_consumption = fetch_ucirepo(id=235)

#Loading the dataset
X = individual_household_electric_power_consumption.data.features
y = individual_household_electric_power_consumption.data.targets  #returns null

# Cleaning the dataset
features = ['Global_reactive_power', 'Voltage', 'Global_intensity']
target = 'Global_active_power'

cols_to_convert = features + [target]
for col in cols_to_convert:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X_clean = X.dropna(subset=cols_to_convert)

X = X_clean[features].astype(float)
y = X_clean[target].astype(float)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying polynomial regression with degree 3
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

y_train = y_train.to_numpy()
X_train_gpu = cp.asarray(X_train_poly)
y_train_gpu = cp.asarray(y_train)
X_test_gpu = cp.asarray(X_test_poly)

# Solve Linear Regression via Normal Equation on GPU
start_time = time.time()
# Compute (X^T X)
XtX = cp.dot(X_train_gpu.T, X_train_gpu)
# Compute inverse of (X^T X)
XtX_inv = cp.linalg.inv(XtX)
# Compute (X^T y)
Xty = cp.dot(X_train_gpu.T, y_train_gpu)
# Regression coefficients: beta = (X^T X)^{-1} (X^T y)
beta = cp.dot(XtX_inv, Xty)
cupy_time = time.time() - start_time

# Compute predictions on training and testing sets on GPU
y_train_pred_gpu = cp.dot(X_train_gpu, beta)
y_test_pred_gpu = cp.dot(cp.asarray(X_test_poly), beta)

# Convert predictions back to CPU and calculate metrics
y_train_pred = cp.asnumpy(y_train_pred_gpu)
y_test_pred = cp.asnumpy(y_test_pred_gpu)
y_train_cpu = y_train
y_test_cpu = y_test.values

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_cpu, y_train_pred))
train_r2 = r2_score(y_train_cpu, y_train_pred)
test_rmse = np.sqrt(mean_squared_error(y_test_cpu, y_test_pred))
test_r2 = r2_score(y_test_cpu, y_test_pred)

print("Coefficient Vector (first 10 shown):", cp.asnumpy(beta)[:10])
print("Training RMSE:", train_rmse)
print("Training R^2:", train_r2)
print("Testing RMSE:", test_rmse)
print("Testing R^2:", test_r2)
print("Computation Time: {:.2f} seconds".format(cupy_time))


"""
Coefficient Vector (first 10 shown): [ 1.10914257e+00 -1.82740718e-02  1.45644238e-02  1.08444781e+00
 -7.68885769e-03  9.76884441e-04  1.30442423e-02  3.17867085e-06
  1.31582852e-02 -1.01784094e-02]
Training RMSE: 0.03815021642565045
Training R^2: 0.9986962545071121
Testing RMSE: 0.03781098890162714
Testing R^2: 0.9987279868563419
Computation Time: 1.91 seconds

"""