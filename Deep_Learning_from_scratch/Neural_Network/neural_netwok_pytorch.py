# 2-layer Neural Network (manual) for binary classification

import torch

class pytorchNN:
  def __init__(self, X_train, y_train, hidden_size=8):
    self.X = torch.tensor(X_train, dtype=torch.float32)
    self.y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    input_size = self.X.shape[1]

    # layer 1
    self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
    self.b1 = torch.zeros(hidden_size, requires_grad=True)

    # layer 2
    self.W2 = torch.randn(hidden_size, 1, requires_grad=True)
    self.b2 = torch.zeros(1, requires_grad=True)

  def forward_propgation(self, X):
    z1 = X @ self.W1 + self.b1
    a1 = torch.relu(z1)

    z2 = a1 @ self.W2 + self.b2
    y_pred = torch.sigmoid(z2)

    return y_pred

  def loss_function(self, y_pred, y):
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    loss = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()
    return loss

  def train(self, epochs=100, lr=0.01):
    for epoch in range(epochs):

      # forward
      y_pred = self.forward_propgation(self.X)

      # loss
      loss = self.loss_function(y_pred, self.y)

      # backward
      loss.backward()

      # update weights
      with torch.no_grad():
        self.W1 -= lr * self.W1.grad
        self.b1 -= lr * self.b1.grad
        self.W2 -= lr * self.W2.grad
        self.b2 -= lr * self.b2.grad

        # reset gradients
        self.W1.grad.zero_()
        self.b1.grad.zero_()
        self.W2.grad.zero_()
        self.b2.grad.zero_()

      if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

  def predict(self, X_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_pred = self.forward_propgation(X_test)
    return (y_pred > 0.5).float()




# Testing on titanic dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# basic preprocessing
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# handle missing
df['Age'].fillna(df['Age'].mean(), inplace=True)

# encode
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# split
X = df.drop('Survived', axis=1).values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scale (important)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# initialize model
model = pytorchNN(X_train, y_train)

# train
model.train(epochs=100, lr=0.01)

# predict
preds = model.predict(X_test)

# accuracy
y_test_tensor = torch.tensor(y_test).view(-1, 1)
accuracy = (preds == y_test_tensor).float().mean()

print("Accuracy:", accuracy.item())