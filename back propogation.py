import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

def train(X, Y, learning_rate, num_iterations):
    # Initialize random weights and biases
    np.random.seed(0)
    weights = 2 * np.random.random((2, 1)) - 1
    biases = np.zeros((1, 1))

    for i in range(num_iterations):
        # Forward propagation
        layer_output = sigmoid(np.dot(X, weights) + biases)

        # Calculate the error
        error = Y - layer_output

        # Backpropagation
        delta = error * sigmoid_derivative(layer_output)
        weights += learning_rate * np.dot(X.T, delta)
        biases += learning_rate * np.sum(delta)

    return weights, biases

# Example training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0, 1, 1, 0]]).T

# Train the neural network
weights, biases = train(X, Y, learning_rate=0.1, num_iterations=10000)

# Make predictions
layer_output = sigmoid(np.dot(X, weights) + biases)
predictions = np.round(layer_output)

print("Predictions:", predictions)
