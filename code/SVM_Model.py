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
    self.learning_rate = learning_rate
    self.lambda_parameter = lambda_param
    self.num_iterations = iterations
    self.parameters = None
    self.offset = None

  def fit(self, X, y):
    """
    Fit the model to the training data.
    """
    num_sambles, n_features = X.shape
    self.parameters = np.zeros(n_features)
    self.offset = 0

    y_derived = np.where(y <= 0, -1, 1)

    for _ in range(self.n_iters):
      for count, x_i in enumerate(X):
        if y_derived[count] * (np.dot(x_i, self.parameters) - self.offset) >= 1:
          self.parameters -= self.lr * (2 * self.lambda_parameter * self.parameters)
        else:
          self.offset -= self.lr * y_derived[count]
          self.parameters -= self.lr * (2 * self.lambda_parameter * self.parameters - np.dot(x_i, y_derived[count]))

  def predict(self, X):
    """
    Make predictions using the learned parameters and bias.
    """
    approx = np.dot(X, self.parameters) - self.offset
    return np.sign(approx)
