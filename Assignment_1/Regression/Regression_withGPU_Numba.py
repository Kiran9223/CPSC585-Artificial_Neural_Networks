import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
from ucimlrepo import fetch_ucirepo
from numba import jit

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
# Fetch the dataset from UCI Repository
individual_household_electric_power_consumption = fetch_ucirepo(id=235)

# Load the dataset
X = individual_household_electric_power_consumption.data.features
# The targets field is null so we extract the target from the features
# Define the columns we want to use
features = ['Global_reactive_power', 'Voltage', 'Global_intensity']
target = 'Global_active_power'

# Convert columns to numeric (replace non-numeric entries with NaN)
cols_to_convert = features + [target]
for col in cols_to_convert:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Drop rows with missing values
X_clean = X.dropna(subset=cols_to_convert)

# Extract features and target, converting them to float
X = X_clean[features].astype(float)
y = X_clean[target].astype(float)

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

# -----------------------------
# Define Numba-Accelerated Function for Linear Regression
# -----------------------------
@jit(nopython=True)
def compute_beta(X, y):
    # Compute X^T * X
    XtX = np.dot(X.T, X)
    # Compute X^T * y
    Xty = np.dot(X.T, y)
    # Solve (X^T X) * beta = (X^T y) for beta
    beta = np.linalg.solve(XtX, Xty)
    return beta

# -----------------------------
# Compute Regression Coefficients Using Numba
# -----------------------------
start_time = time.time()
beta = compute_beta(X_train_poly, y_train)
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

"""
Coefficient Vector (first 10 shown): [ 1.10914257e+00 -1.82740718e-02  1.45644238e-02  1.08444781e+00
 -7.68885769e-03  9.76884441e-04  1.30442423e-02  3.17867099e-06
  1.31582852e-02 -1.01784094e-02]
Training RMSE: 0.038150216425650446
Training R^2: 0.9986962545071121
Testing RMSE: 0.03781098890162548
Testing R^2: 0.998727986856342
Computation Time: 8.27 seconds
"""
