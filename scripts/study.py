import numpy as np

# Set the learning rate
learning_rate = 0.1

# Initialize the weights and biases for the first linear layer
w1 = np.array([[2.0], [0.0]])
b1 = np.zeros((1, 1))

# Initialize the weights and biases for the second linear layer
w2 = np.array([[0.0], [0.5]])
b2 = np.zeros((1, 1))

# Define the input vector
x = np.array([[1], [2]])

# Compute the output of the first linear layer
y1 = np.dot(w1, x) + b1

# Compute the output of the second linear layer
y2 = np.dot(w2, y1) + b2

# Set the target output
t = np.array([[4], [1.41421356]])

# Compute the MSE loss
loss = np.mean((y2 - t)**2) / 2

# Backpropagate the loss through the second linear layer
grad_y2 = y2 - t
grad_w2 = np.outer(grad_y2, y1)
grad_b2 = grad_y2
grad_y1 = np.dot(w2.T, grad_y2)

# Backpropagate the loss through the first linear layer
grad_w1 = np.outer(grad_y1, x)
grad_b1 = grad_y1

# Update the weights and biases using gradient descent
w1 -= learning_rate * grad_w1
b1 -= learning_rate * grad_b1
w2 -= learning_rate * grad_w2
b2 -= learning_rate * grad_b2