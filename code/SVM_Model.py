"""
A custom stochastic gradient descent model for support vector machines.

Author: Anthony Andriano
"""

import numpy as np

class SVM_Model:
  """
  A custom implementation of a Stochastic Gradient Descent
  model for support vector analysis.
  """

  def __init__(self, learning_rate = 0.001, lambda_param = 0.01, iterations = 1000):
    """
    Initially configure the model.
    """
    self.lr = learning_rate
    self.lambda_param = lambda_param
    self.n_iters = iterations
    self.w = None
    self.b = None

  def fit(self, X, y):
    """
    Fit the model to the training data.
    """
    n_samples, n_features = X.shape
    y_ = np.where(y <= 0, -1, 1)

    self.w = np.zeros(n_features)
    self.b = 0

    for _ in range(self.n_iters):
      for idx, x_i in enumerate(X):
        condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
        if condition:
          self.w -= self.lr * (2 * self.lambda_param * self.w)
        else:
          self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
          self.b -= self.lr * y_[idx]

  def predict(self, X):
    """
    Make predictions using the learned parameters and bias.
    """
    approx = np.dot(X, self.w) - self.b
    return np.sign(approx)
