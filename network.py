import numpy as np

# XOR Input and Output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

# Weights and biases initialization
np.random.seed(1)

w1 = np.random.randn(2, 2)  
b1 = np.zeros((1, 2))       # Biases for hidden layer

w2 = np.random.randn(2, 1)  
b2 = np.zeros((1, 1))       # Biases for output layer

# Training loop
for epoch in range(10000):  
    # Forward pass
    a1 = sigmoid(np.dot(X, w1) + b1)
    output = sigmoid(np.dot(a1, w2) + b2)

    # Calculate error
    error = Y - output
    d_output = error * derivative(output)  
    d_hidden = d_output.dot(w2.T) * derivative(a1)  

    # Backpropagation (update weights and biases)
    w2 += a1.T.dot(d_output) * 0.1
    b2 += np.sum(d_output, axis=0, keepdims=True) * 0.1
    w1 += X.T.dot(d_hidden) * 0.1
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * 0.1

    if epoch % 1000 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# Final predictions
print("\nPredictions:")
print(output.round(3))