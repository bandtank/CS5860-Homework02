"""
Compare different SVM models using multiple datasets. The models are
implemented without using any libraries. The datasets are loaded from
CSV files and split into training and testing sets. The models are
trained and evaluated using the mean squared error, mean absolute error,
and R^2 score. The results are displayed in a table and a scatter plot.

Author: Anthony Andriano
"""
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from SVM_Model import SVM_Model
from data import Data

class SVM:
  """
  Implement a Stochastic Gradient Descent model for support vector machine analysis.
  """

  def __init__(self, args):
    """
    Load the dataset based on the user's choice.
    """

    # Get the data based on the user's input
    X, y = Data(args.dataset).get_data()

    # Split the data into training and testing sets
    X_train, X_test, self.y_train, self.y_test = train_test_split(
      X,
      y,
      test_size = args.test_ratio,
      random_state = args.random_seed
    )

    # Normalize feature values
    scaler = StandardScaler()
    self.X_train = scaler.fit_transform(X_train)
    self.X_test = scaler.transform(X_test)

    #self.model= SVM_Model(
    #  learning_rate = args.learning_rate,
    #  lambda_param = args.lambda_param,
    #  iterations = args.iterations,
    #)

    #data = load_iris()
    #self.X = data.data[:, :2]
    #y = data.target
    #self.y = np.where(y == 0, -1, 1)

  def run(self):
    """
    Train and evaluate the Stochastic Gradient Descent algorithm on the dataset.
    """

    return

  def plot_decision_boundary(self):
    plt.scatter(self.X[:, 0], self.X[:, 1], c = self.y, cmap = 'bwr')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = self.model.predict(xy).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha = 0.3, cmap = 'bwr')
    plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Stochastic Gradient Descent for Support Vector Machines")
  parser.add_argument('--dataset', required = True, choices = ['relax', 'skin'])
  parser.add_argument("--test_ratio", type = float, default = 0.2)
  parser.add_argument("--random_seed", type = int, default = 42)

  parser.add_argument("--lambda_param", type = float, default = 0.01)
  parser.add_argument("--iterations", type = int, default = 1000)
  parser.add_argument("--learning_rate", type = float, default = 0.001)

  args = parser.parse_args()

  svm = SVM(args)
  svm.run()
