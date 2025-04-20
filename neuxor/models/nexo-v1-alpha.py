import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

np.random.seed(1)

w1 = np.random.rand(2, 4)
b1 = np.zeros((1, 4))

w2 = np.random.rand(4, 1)
b2 = np.zeros((4, 1))

for epoch in range(90000):  
    a1 = sigmoid(np.dot(X, w1) + b1)
    output = sigmoid(np.dot(a1, w2) + b2)
    error = Y - output
    d_output = error * derivative(output)
    d_hidden = d_output.dot(w2.T) * derivative(a1)
    w2 += a1.T.dot(d_output) * 0.1
    b2 += np.sum(d_output, axis=0, keepdims=True) * 0.1
    w1 += X.T.dot(d_hidden) * 0.1
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * 0.1
    if epoch % 1000 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

print("\nFinal Predictions:")
print(output.round(3))