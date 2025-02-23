import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import time
import platform
import psutil 

# System Info
print(f"OS: {platform.system()} {platform.release()}")
print(f"CPU: {platform.processor()}")
print(f"Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
print(f"RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")

# Load dataset
dataset = fetch_ucirepo(id=891)
X = dataset.data.features.to_numpy(dtype='float64')  # Convert to float64
y = dataset.data.targets.to_numpy().flatten()  # Convert to 1D array

# Convert to binary classification
y = (y > 0).astype(int)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add Bias Term (Intercept)
X = cp.c_[cp.ones((X.shape[0], 1)), X]  # Adding a column of ones for bias term

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Move data to GPU
X_train_gpu = cp.asarray(X_train)
y_train_gpu = cp.asarray(y_train)
X_test_gpu = cp.asarray(X_test)
y_test_gpu = cp.asarray(y_test)

# Sigmoid function (GPU)
def sigmoid(z):
    return 1 / (1 + cp.exp(-z))

# Optimized Logistic Regression using CuPy
def logistic_regression_gpu(X, y):
    start_time = time.time()

    # Compute weights using Pseudo-inverse (Faster than lstsq)
    weights = cp.dot(cp.linalg.pinv(X), y)  # (X^T * X)^(-1) * X^T * y

    computation_time = time.time() - start_time
    return weights, computation_time

# Train the model on GPU
weights_gpu, cupy_time = logistic_regression_gpu(X_train_gpu, y_train_gpu)

# Predict function using Normal Equation weights
def predict_gpu(X, weights):
    linear_model = cp.dot(X, weights)  # No bias term needed
    predictions = sigmoid(linear_model)
    return (predictions > 0.5).astype(int)

# Predict on GPU
y_pred_gpu = predict_gpu(X_test_gpu, weights_gpu)

# Move results back to CPU
y_pred = cp.asnumpy(y_pred_gpu)

# Print metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("Precision:", precision_score(y_test, y_pred))
print("Computation Time: {:.4f} seconds".format(cupy_time))
