import pandas as pd
import numpy as np
import torch
from torch import nn, optim
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

# Defining the device to use (GPU if available) and converting tensors to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train_tensor = torch.tensor(X_train_poly, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_poly, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

# Defining the model, loss function, and optimizer
input_dim = X_train_poly.shape[1]
model = nn.Linear(input_dim, 1)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the model
num_epochs = 1000
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

training_time = time.time() - start_time

# Evaluating the model on the test set
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

# Converting the predictions back to numpy arrays
train_preds_np = train_preds.cpu().numpy()
test_preds_np = test_preds.cpu().numpy()
y_train_np = y_train_tensor.cpu().numpy()
y_test_np = y_test_tensor.cpu().numpy()

# Calculating the evaluation metrics
train_rmse = np.sqrt(mean_squared_error(y_train_np, train_preds_np))
train_r2 = r2_score(y_train_np, train_preds_np)
test_rmse = np.sqrt(mean_squared_error(y_test_np, test_preds_np))
test_r2 = r2_score(y_test_np, test_preds_np)

coef_vector = model.weight.cpu().detach().numpy().flatten()
intercept = model.bias.cpu().detach().numpy()


print("Coefficient Vector:", coef_vector)
print("Intercept:", intercept)
print("Training RMSE:", train_rmse)
print("Training R^2:", train_r2)
print("Testing RMSE:", test_rmse)
print("Testing R^2:", test_r2)
print("Training Time: {:.2f} seconds".format(training_time))

"""
Coefficient Vector: [ 6.4146119e-01 -1.8298244e-02  1.4632550e-02  1.0843211e+00
 -7.6395827e-03  9.3480153e-04  1.2879177e-02  3.1506224e-05
  1.3379974e-02 -9.8162713e-03  2.5569097e-04 -8.6955570e-06
  1.3403122e-03  2.7792080e-04 -2.6589434e-04 -2.4383487e-03
  4.3624110e-05 -1.9571137e-04 -6.3329964e-05  1.2902220e-03]
Intercept: [0.4674425]
Training RMSE: 0.038150926003644246
Training R^2: 0.9986962080001831
Testing RMSE: 0.03781137474115747
Testing R^2: 0.9987279772758484
Training Time: 2.06 seconds

"""