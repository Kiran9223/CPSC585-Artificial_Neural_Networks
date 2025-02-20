import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from numba import cuda, float32
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

# Fetch the dataset
dataset = fetch_ucirepo(id=891)  # CDC Diabetes Health Indicators dataset

# Check if data is loaded properly
print(f"Dataset: {dataset}")
print(f"Data: {dataset.data}")

# If the dataset is fetched properly, proceed with the preprocessing
data = dataset.data

if data is not None:
    # Preprocessing
    data = data.dropna()  # Dropping rows with missing values
    X = data.drop(columns=["diabetes"])
    y = data["diabetes"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert y_train and y_test to numpy arrays for compatibility
    y_train = y_train.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    # Define the logistic regression kernel for GPU using Numba
    @cuda.jit
    def logistic_regression_gpu(X, y, w, learning_rate=0.01, epochs=1000):
        i = cuda.grid(1)
        if i < X.shape[0]:
            # Compute the logistic regression update for each sample
            linear_output = 0.0
            for j in range(X.shape[1]):
                linear_output += X[i, j] * w[j]
            prediction = 1 / (1 + np.exp(-linear_output))  # Sigmoid function
            error = y[i] - prediction

            for j in range(X.shape[1]):
                # Update weights
                w[j] += learning_rate * error * X[i, j]

    # Initialize weights and move data to GPU
    X_train_gpu = cuda.to_device(X_train.astype(np.float32))
    y_train_gpu = cuda.to_device(y_train)
    w_gpu = cuda.device_array(X_train.shape[1], dtype=np.float32)

    # Launch kernel with one block and multiple threads
    logistic_regression_gpu[256, 256](X_train_gpu, y_train_gpu, w_gpu)

    # After training, you can use the weights for predictions and evaluation
    # Moving weights back to the CPU
    w_cpu = w_gpu.copy_to_host()

    # Make predictions on test set
    X_test_gpu = cuda.to_device(X_test.astype(np.float32))
    y_pred_gpu = np.dot(X_test_gpu, w_cpu)  # Linear predictions
    y_pred_gpu = 1 / (1 + np.exp(-y_pred_gpu))  # Sigmoid function

    # Convert to binary prediction
    y_pred_bin = (y_pred_gpu > 0.5).astype(np.float32)

    # Evaluate the model's performance
    conf_matrix_gpu = confusion_matrix(y_test, y_pred_bin)
    accuracy_gpu = accuracy_score(y_test, y_pred_bin)
    precision_gpu = precision_score(y_test, y_pred_bin)

    print("Confusion Matrix (GPU):", conf_matrix_gpu)
    print("Accuracy (GPU):", accuracy_gpu)
    print("Precision (GPU):", precision_gpu)

else:
    print("Failed to load dataset.")
