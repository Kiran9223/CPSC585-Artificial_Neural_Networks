import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define MLP model
mlp_sklearn = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Two hidden layers with 128 and 64 neurons
    activation='relu',            # ReLU activation function
    solver='adam',                # Adam optimizer
    alpha=0.0001,                 # L2 regularization term
    batch_size=128,               # Mini-batch size
    learning_rate_init=0.001,     # Initial learning rate
    max_iter=20,                  # Number of epochs
    early_stopping=True,          # Early stopping
    random_state=42
)

# Train the model and measure training time
start_time = time.time()
mlp_sklearn.fit(X_train, y_train)
training_time = time.time() - start_time

# Evaluate the model
y_pred = mlp_sklearn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')

print(f"Scikit-learn MLP Training Time: {training_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")

# Scikit-learn MLP Training Time: 41.14 seconds
# Accuracy: 0.9751
# Precision: 0.9752