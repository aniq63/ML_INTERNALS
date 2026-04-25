import numpy as np

class Perceptron:
  def __init__(self, lr = 0.1, epoch = 100):
    self.lr = lr
    self.epoch = epoch
  
  def fit(self, x, y):
    # Featch the no of samples (rows) and no of feaures from input data on which our model basically train
    n_samples , n_features = x.shape

    # We intialize the randome weights and biases
    # In this case we intialize it with zeros
    self.w = np.zeros(n_features)
    self.b = 0

    # we conver the labels in the -1 and 1 
    y_ = np.where(y <= 0, -1, 1)


    for _ in range (self.epoch):
      for i in range(n_samples):
        x_i = x[i]
        y_i = y_[i]

        # Now first we compute with the random weights
        margin = y_i * (x_i @ self.w) + self.b

        if margin < 1:
          self.w -= self.lr * (-y_i * x_i)
          self.b -= self.lr * (-y_i)

  def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, -1)

# Testing
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=42)

model = Perceptron(epoch=1000)
model.fit(X, y)

predictions = model.predict(X)

accuracy = np.mean(predictions == np.where(y <= 0, -1, 1))
print("Accuracy:", accuracy)