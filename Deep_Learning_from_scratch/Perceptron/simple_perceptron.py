import numpy as np

class Perceptron:
  def __init__(self, lr = 0.1 , epoch = 100):
    self.lr = lr
    self.epoch = epoch

  def fit(self, x, y):
    # get the samles and the faetures
    n_samples , n_features = x.shape
    
    # Intialize the weights and bias as 0
    self.w = np.zeros(n_features)
    self.b = 0

    # Convert the labels is -1 and 1 for helping in a weight and bias update
    y_ = []
    for i in y:
      y_.append(1 if i > 0 else -1)


    for _ in range(self.epoch):
      for i in range(n_samples):
        x_i = x[i]
        y_i = y_[i]

        # first update the weights and bias with a random valuses (0) by dot product with the input samples
        # This basically create a line in a 2d data , plane in 3d data and hyper plane 3d + data
        linear_output = (x_i @ self.w) + self.b

        # apply a step function or you say activation function
        y_pred = np.sign(linear_output)

        if y_pred != y_i:
          self.w += self.lr * y_i * x_i
          self.b += self.lr * y_i

  def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output) 

# Testing
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=42)

model = Perceptron()
model.fit(X, y)

predictions = model.predict(X)

accuracy = np.mean(predictions == np.where(y <= 0, -1, 1))
print("Accuracy:", accuracy)