import time
import numpy as np
import platform
import psutil
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml

# Function to get machine specifications
def get_system_specs():
    print("\n=== Machine Specifications ===")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    
    # CPU details
    cpu_freq = psutil.cpu_freq()
    print(f"CPU: {psutil.cpu_count(logical=False)} cores, {psutil.cpu_count(logical=True)} threads")
    print(f"CPU Max Frequency: {cpu_freq.max:.2f} MHz")
    
    # RAM details
    ram = psutil.virtual_memory()
    print(f"Total RAM: {ram.total / (1024 ** 3):.2f} GB")
    
    # GPU details (if available)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Cores: {torch.cuda.device_count()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("GPU: Not available")
    print("==============================\n")

# Display machine specifications
get_system_specs()

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int64)

# Split into training and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Standardize data
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Scikit-learn MLP Model
mlp_sklearn = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', alpha=0.0001,
                             max_iter=50, random_state=42, early_stopping=True)

# Train model
start_time = time.time()
mlp_sklearn.fit(X_train_scaled, y_train)
sklearn_time = time.time() - start_time

# Predictions
y_pred_sklearn = mlp_sklearn.predict(X_test_scaled)
sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
sklearn_precision = precision_score(y_test, y_pred_sklearn, average='macro')

# Print results
print(f"Scikit-learn MLP - Accuracy: {sklearn_accuracy:.4f}, Precision: {sklearn_precision:.4f}, Training Time: {sklearn_time:.2f} sec")

#Scikit-learn MLP - Accuracy: 0.9779, Precision: 0.9777, Training Time: 20.87 sec
