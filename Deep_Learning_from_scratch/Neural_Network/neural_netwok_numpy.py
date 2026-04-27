# Build a 2 layer Neural network for regression problem use sigmoid actiation function and MSE loss function

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, lr=0.1):
        self.lr = lr
        
        # W1: (hidden_size, input_size)
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size, 1))
        
        # W2: (1, hidden_size)
        self.W2 = np.random.randn(1, hidden_size)
        self.b2 = np.zeros((1, 1))

    # Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Derivative of sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        x = x.reshape(-1, 1)
        
        # z1 = W1x + b1
        self.z1 = np.dot(self.W1, x) + self.b1
        
        # a1 = sigmoid(z1)
        self.a1 = self.sigmoid(self.z1)
        
        # z2 = W2a1 + b2
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        
        # output
        self.y_hat = self.z2
        
        return self.y_hat

    def backward(self, x, y):
        x = x.reshape(-1, 1)
        y = np.array([[y]])

        # Loss derivative: dL/dy_hat = (y_hat - y)
        dL_dyhat = self.y_hat - y
        
        # dL/dW2 = dL/dy_hat * a1^T
        dL_dW2 = np.dot(dL_dyhat, self.a1.T)
        
        # dL/db2 = dL/dy_hat
        dL_db2 = dL_dyhat
        
        # dL/da1 = W2^T * dL/dy_hat
        dL_da1 = np.dot(self.W2.T, dL_dyhat)
        
        # da1/dz1 = sigmoid'(z1)
        da1_dz1 = self.sigmoid_derivative(self.a1)
        
        # dL/dz1 = dL/da1 * da1/dz1
        dL_dz1 = dL_da1 * da1_dz1
        
        # dL/dW1 = dL/dz1 * x^T
        dL_dW1 = np.dot(dL_dz1, x.T)
        
        # dL/db1 = dL/dz1
        dL_db1 = dL_dz1
        
        # Gradient Descent updates
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            for i in range(len(X)):
                self.forward(X[i])
                self.backward(X[i], y[i])

    def predict(self, X):
        preds = []
        for x in X:
            y_hat = self.forward(x)
            preds.append(y_hat[0][0])
        return np.array(preds)

#Testing
# Create dummy dataset (y = x1 + x2)
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

y = np.array([3, 5, 7, 9])

# Initialize network
nn = NeuralNetwork(input_size=2, hidden_size=4, lr=0.01)

# Train
nn.train(X, y, epochs=2000)

# Predict
preds = nn.predict(X)

print("Predictions:", preds)
print("Actual:", y)