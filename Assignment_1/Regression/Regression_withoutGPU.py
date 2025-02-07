import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
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

# Training the model and computing the CPU time
model = LinearRegression()
start_time = time.time()
model.fit(X_train_poly, y_train)
cpu_time = time.time() - start_time

# Evaluating the model
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Calculating the RMSE and R^2 scores for both training and testing sets
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


"""
Coefficient Vector: [ 0.00000000e+00 -1.82740718e-02  1.45644238e-02  1.08444781e+00
 -7.68885769e-03  9.76884441e-04  1.30442423e-02  3.17867084e-06
  1.31582852e-02 -1.01784094e-02  2.57671991e-04  9.76227291e-06
  1.36060029e-03  2.66713069e-04 -3.33764988e-04 -2.50430898e-03
  4.08670282e-05 -1.95398677e-04  1.11936907e-05  1.36815508e-03]
Training RMSE: 0.038150216425650446
Training R^2: 0.9986962545071121
Testing RMSE: 0.03781098890162699
Testing R^2: 0.9987279868563419
Computation Time: 1.44 seconds

"""