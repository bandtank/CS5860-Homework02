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
    _, n_features = X.shape
    self.parameters = np.zeros(n_features)
    self.offset = 0

    for _ in range(self.num_iterations):
      for count, x_i in enumerate(X):
        if y[count] * (np.dot(x_i, self.parameters) - self.offset) >= 1:
          self.parameters -= self.learning_rate * (2 * self.lambda_parameter * self.parameters)
        else:
          self.parameters -= self.learning_rate * (2 * self.lambda_parameter * self.parameters - np.dot(x_i, y[count]))
          self.offset -= self.learning_rate * y[count]

  def predict(self, X):
    """
    Make predictions using the learned parameters and bias.
    """
    approx = np.dot(X, self.parameters) - self.offset
    return np.sign(approx)

if __name__ == "__main__":
  from sklearn import datasets
  from sklearn.model_selection import train_test_split

  X, y = datasets.make_blobs(
      n_samples=500, n_features=2, centers=2, cluster_std=1.05, random_state=1
  )

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)

  clf = SVM_Model(lambda_param=0.0001, learning_rate=0.01, iterations=1000)
  clf.fit(X_train, y_train)
  predictions = clf.predict(X_test)

  def accuracy(y_true, y_pred):
      accuracy = np.sum(y_true==y_pred) / len(y_true)
      return accuracy

  print("SVM Accuracy: ", accuracy(y_test, predictions))