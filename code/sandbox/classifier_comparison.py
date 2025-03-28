"""
Compare machine learning classifiers using 2-feature datasets.
Visualize the datasets and decision boundaries.

Based on https://scikit-learn.org/stable/datasets/sample_generators.html

Author: Anthony Andriano

Example Invocations:
 python compare.py
 python compare.py --samples 100
 python compare.py --samples 50 --random_state 27
"""

import argparse
import random

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sklearn.datasets as ds
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class Compare:
  """
  Compare various classifiers and datasets.
  """

  def __init__(self, args):
    samples = args.samples or 100
    self.random_state = args.random_state or random.randint(10,100)
    self.figure = plt.figure(figsize=(27, 9))
    self.subplot_cnt = 1

    self.classifiers = [
      {"name": "Nearest Neighbors", "model": KNeighborsClassifier(3)},
      {"name": "Linear SVM", "model": SVC(kernel="linear", C=0.025, random_state=42)},
      {"name": "RBF SVM", "model": SVC(gamma=2, C=1, random_state=42)},
      {"name": "Gaussian Process", "model": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)},
      {"name": "Decision Tree", "model": DecisionTreeClassifier(max_depth=5, random_state=42)},
      {"name": "Random Forest", "model": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)},
      {"name": "Neural Net", "model": MLPClassifier(alpha=1, max_iter=1000, random_state=42)},
      {"name": "AdaBoost", "model": AdaBoostClassifier(random_state=42)},
      {"name": "Naive Bayes", "model": GaussianNB()},
      {"name": "QDA", "model": QuadraticDiscriminantAnalysis()},
    ]

    self.datasets = [
        ds.make_moons(n_samples=samples, noise=0.3, random_state=self.random_state * 2),
        ds.make_circles(n_samples=samples, noise=0.2, factor=0.5, random_state=self.random_state // 2),
        ds.make_blobs(n_samples=samples, centers=3, cluster_std=0.60, random_state = self.random_state + 5),
        ds.make_gaussian_quantiles(n_samples=samples, n_features=2, n_classes=3, random_state = self.random_state * 3),
        ds.make_classification(n_samples=samples, n_features=2, n_redundant=0, n_informative=2, random_state = self.random_state * 2 - 3, n_clusters_per_class=1),
    ]

  def run(self):
    for count, ds in enumerate(self.datasets):
      self.process_dataset(count, ds)

    plt.tight_layout()
    plt.show()

  def process_dataset(self, count, ds):
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = self.random_state)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Plot the dataset
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(self.datasets), len(self.classifiers) + 1, self.subplot_cnt)
    if count == 0:
        ax.set_title("Input data")

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")

    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    self.subplot_cnt += 1

    for classifier in self.classifiers:
        ax = plt.subplot(len(self.datasets), len(self.classifiers) + 1, self.subplot_cnt)

        clf = make_pipeline(StandardScaler(), classifier["model"])
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")

        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if count == 0:
            ax.set_title(classifier["name"])
        ax.text(x_max - 0.3, y_min + 0.3, ("%.3f" % score).lstrip("0"), size=15, horizontalalignment="right")

        self.subplot_cnt += 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Compare classifiers using 2-feature datasets")

  parser.add_argument("--random_state", type = int, default = None)
  parser.add_argument("--samples", type = int, default = None)

  args = parser.parse_args()

  c = Compare(args)
  c.run()
