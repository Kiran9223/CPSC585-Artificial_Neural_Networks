import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from ucimlrepo import fetch_ucirepo
import time

# Load dataset
dataset = fetch_ucirepo(id=891)
X = dataset.data.features.to_numpy()
y = dataset.data.targets.to_numpy().flatten()  # Convert to 1D array

# Convert to binary classification: 0 = No Diabetes, 1 = Diabetes
y = (y > 0).astype(int)

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

# Logistic Regression using Normal Equation (GPU with CuPy)
def logistic_regression_normal_eq_gpu(X, y):
    start_time = time.time()

    # Compute weights using Normal Equation
    X_T_X = cp.dot(X.T, X)  # X^T * X
    X_T_y = cp.dot(X.T, y)  # X^T * y

    # Compute inverse and multiply (X^T X)^(-1) * X^T * y
    weights = cp.dot(cp.linalg.inv(X_T_X), X_T_y)

    computation_time = time.time() - start_time

    return weights, computation_time

# Train the model on GPU
weights_gpu, cupy_time = logistic_regression_normal_eq_gpu(X_train_gpu, y_train_gpu)

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
print("Computation Time: {:.2f} seconds".format(cupy_time))

