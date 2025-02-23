import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time
import platform
import psutil
from ucimlrepo import fetch_ucirepo

# System Specifications
print(f"OS: {platform.system()} {platform.release()}")
print(f"CPU: {platform.processor()}")
print(f"Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
print(f"RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")

# Fetch the dataset
dataset = fetch_ucirepo(id=235)

# Load features
X = dataset.data.features

# Extract target manually
target = 'Global_active_power'
features = ['Global_reactive_power', 'Voltage', 'Global_intensity']

# Convert columns to numeric and handle missing values
cols_to_convert = features + [target]
for col in cols_to_convert:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X_clean = X.dropna(subset=cols_to_convert)

X = X_clean[features].astype(np.float16)
y = X_clean[target].astype(np.float16)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Polynomial Regression
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train the model and measure time
start_time = time.time()
model = LinearRegression()  
model.fit(X_train_poly, y_train)
cpu_time = time.time() - start_time

# Predictions
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Compute metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("Coefficient Vector:", model.coef_)
print("Training RMSE:", train_rmse)
print("Training R^2:", train_r2)
print("Testing RMSE:", test_rmse)
print("Testing R^2:", test_r2)
print("Computation Time: {:.2f} seconds".format(cpu_time))
