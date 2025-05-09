"""
Compare different regression models using multiple datasets. The
regression models are implemented using libraries. The datasets are
loaded from CSV files and split into training and testing sets. The
feature values are normalized using the StandardScaler class. The
regression models are trained and evaluated using the mean squared
error, mean absolute error, and R^2 score. The results are displayed
in a table and a scatter plot.

Author: Anthony Andriano

Example Invocations:
 python compare.py --all --dataset relax
 python compare.py --all --dataset relax --verbosity 2
 python compare.py --regressors --dataset skin
 python compare.py --regressors --dataset skin --random_state 28
 python compare.py --classifiers --dataset skin
 python compare.py --classifiers --dataset skin --test_ratio 0.3
"""

import argparse
import tabulate
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import sklearn.discriminant_analysis as discriminant_analysis
import sklearn.gaussian_process as gaussian_process
import sklearn.tree as tree
import sklearn.naive_bayes as naive_bayes
import sklearn.neural_network as neural_network
import xgboost
import lightgbm as lgb
import catboost

from SVM_Model import SVM_Model
from data import Data

class Compare:
  """
  Compare learning techniques for data analysis.
  """

  def __init__(self, args):
    """
    Initialize the class and load the data.
    """

    ### Start initialization
    time_start = time.time()

    self.args = args

    self.results = {
      "regressors": [],
      "classifiers": [],
    }

    ### Get the data based on the user's input
    time_previous = time_start

    X, y = Data(args.dataset).get_data()

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to load data")

    ### Split the data into training and testing sets
    time_previous = time.time()

    X_train, X_test, self.y_train, self.y_test = train_test_split(
      X,
      y,
      test_size = args.test_ratio,
      shuffle = True,
      random_state = args.random_state
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to split data")

    ### Normalize feature values
    time_previous = time.time()

    scaler = StandardScaler()
    self.X_train = scaler.fit_transform(X_train)
    self.X_test = scaler.transform(X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to normalize data")

    ### Finalize
    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Total time to initialize")
      print()

  def run(self):
    """
    Train and evaluate various models.
    """

    if self.args.classifiers or self.args.all:
      self.run_classifiers()

    if self.args.regressors or self.args.all:
      self.run_regressors()

  def run_classifiers(self):
    """
    Train and evaluate various classification models.
    """

    print("Running Classifiers...")

    for C in [0.0001, 5.0]:
      for learning_rate in [0.00001, 0.1]:
        self.CustomSVM(
          learning_rate = learning_rate,
          C = C,
          iterations = 100,
        )

    # K Nearest Neighbors
    for weight in ["uniform"]:
      for n_neighbors in [3, 7]:
        for algorithm in ["ball_tree"]:
          for leaf_size in [10, 30]:
            for p in [1, 2]:
              self.KNearestClassifier(
                n_neighbors = n_neighbors,
                weights = weight,
                leaf_size = leaf_size,
                algorithm = algorithm,
                p = p
              )

    # SVC
    for C in [0.1, 1.0]:
      for tol in [0.0001]:
        for kernel in ["linear"]:
          for gamma in [0.0001, 0.01]:
            self.SVC(
              C = C,
              tol = tol,
              kernel = kernel,
              gamma = gamma,
              random_state = self.args.random_state,
            )

    # LinearSVC
    for loss in ["squared_hinge", "hinge"]:
      for C in [0.01, 1.0]:
        for tol in [0.0001, 0.01]:
          for max_iter in [1000, 10000]:
            self.LinearSVC(
              loss = loss,
              C = C,
              tol = tol,
              max_iter = max_iter,
              random_state = self.args.random_state,
            )

    # SGD Classifier
    for loss in ["hinge", "squared_hinge"]:
      for alpha in [0.0001, 0.01]:
        for tol in [0.0001, 0.01]:
          self.SGDClassifier(
            loss = loss,
            alpha = alpha,
            tol = tol,
            random_state = self.args.random_state,
          )

    # Random Forests
    for n_estimators in [10, 200]:
      for max_depth in [2, 4]:
        for bootstrap in [True, False]:
          self.RandomForests(
            n_estimators = n_estimators,
            max_depth = max_depth,
            bootstrap = bootstrap,
            random_state = self.args.random_state,
          )

    # AdaBoost
    for n_estimators in [50, 300]:
      for learning_rate in [0.1, 0.3]:
        self.AdaBoost(
          n_estimators = n_estimators,
          learning_rate = learning_rate,
          random_state = self.args.random_state,
        )

    # XGBoost
    for max_depth in [3, 5]:
      for learning_rate in [0.01, 0.3]:
        for n_estimators in [100, 300]:
          for gamma in [0, 2]:
            for tree_method in ["hist"]:
              self.XGBClassifier(
                max_depth = max_depth,
                learning_rate = learning_rate,
                n_estimators = n_estimators,
                gamma = gamma,
                tree_method = tree_method,
                random_state = self.args.random_state,
              )

    # Gaussian Process Classifier
    self.GaussianProcessClassifier(random_state = self.args.random_state)

    # Quadratic Discriminant Analysis
    for reg_param in [0.1, 0.2]:
      for tol in [0.0001, 0.01]:
        self.QuadraticDiscriminantAnalysis(
          reg_param = reg_param,
          tol = tol,
        )

    # Gaussian Naive Bayes
    for var_smoothing in [1e-9, 1e-7]:
      self.GaussianNB(
        var_smoothing = var_smoothing,
      )

    # Decision Tree Classifier
    for criterion in ["gini", "entropy"]:
      for max_depth in [2, 4]:
        self.DecisionTreeClassifier(
          criterion = criterion,
          max_depth = max_depth,
          random_state = self.args.random_state,
        )

    # LightGBM
    for boosting_type in ["gbdt", "dart"]:
      for num_leaves in [5, 50]:
        for max_depth in [5]:
          for learning_rate in [0.01, 0.3]:
            for n_estimators in [200]:
              for objective in ["binary"]:
                self.LightGBMClassifier(
                  boosting_type = boosting_type,
                  num_leaves = num_leaves,
                  max_depth = max_depth,
                  learning_rate = learning_rate,
                  n_estimators = n_estimators,
                  objective = objective,
                  random_state = self.args.random_state,
                )

    # CatBoost
    for iterations in [200]:
      for depth in [5]:
        for learning_rate in [0.01, 0.3]:
          for l2_leaf_reg in [1, 3]:
            self.CatBoost(
              iterations = iterations,
              depth = depth,
              learning_rate = learning_rate,
              l2_leaf_reg = l2_leaf_reg,
              random_state = self.args.random_state,
            )

    # Neural Network Classifier
    for activation in ["relu"]:
      for solver in ["sgd"]:
        for alpha in [0.0001, 0.01]:
          for learning_rate in ["constant"]:
            for learning_rate_init in [0.001, 0.01]:
              for max_iter in [500]:
                self.MLPClassifier(
                  activation = activation,
                  solver = solver,
                  alpha = alpha,
                  learning_rate = learning_rate,
                  learning_rate_init = learning_rate_init,
                  max_iter = max_iter,
                  random_state = self.args.random_state,
                )

    self.print_classifier_results()

  def print_classifier_results(self):
    data = sorted(self.results["classifiers"], key = lambda d: float(d['Accuracy (%)']), reverse = True)

    print()
    print(tabulate.tabulate(
      data,
      headers = "keys",
      colalign = ["left", "center", "center"],
      floatfmt = "0.3f",
    ))
    print()

  def run_regressors(self):
    """
    Train and evaluate various regression models.
    """

    print("Running Regressors...")
    for weight in ["uniform", "distance"]:
      for n_neighbors in [3, 5, 7]:
        for leaf_size in [10, 20, 30]:
          for p in [1, 2]:
            self.KNeighborsRegressor(
              n_neighbors = n_neighbors,
              weights = weight,
              leaf_size = leaf_size,
              p = p
            )
    for max_depth in [2, 3, 4]:
      for learning_rate in [0.1, 0.2, 0.3]:
        for n_estimators in [100, 200, 300]:
          for gamma in [0, 1, 2]:
            self.XGBRegressor(
              max_depth = max_depth,
              learning_rate = learning_rate,
              n_estimators = n_estimators,
              gamma = gamma,
              random_state = self.args.random_state,
            )
    for C in [0.1, 0.5, 1.0]:
      for tol in [0.0001, 0.001, 0.01]:
        for max_iter in [100, 200, 300]:
          for solver in ['lbfgs', 'liblinear', 'sag', 'saga', 'newton-cg']:
            self.LogisticRegression(
              C = C,
              tol = tol,
              max_iter = max_iter,
              solver = solver,
              random_state = self.args.random_state,
            )

    data = sorted(self.results["regressors"], key = lambda d: float(d['MSE']), reverse = False)

    print()
    print(tabulate.tabulate(
      data,
      headers = "keys",
      #tablefmt = "pretty",
      colalign = ["left", "center", "center", "center", "center", "center"],
      floatfmt = "0.3f",
    ))

  def NearestNeighbors(self, n_neighbors = 5, algorithm = "ball_tree", leaf_size = 30):
    """
    Nearest Neighbors is an unsupervised technique that finds the
    k-nearest neighbors of a target data point.

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors

    Default Parameters:
      NearestNeighbors(
        *,
        n_neighbors = 5,
        radius = 1.0,
        algorithm = 'auto',
        leaf_size = 30,
        metric = 'minkowski',
        p = 2,
        metric_params = None,
        n_jobs = None
      )
    """

    ### Initialize
    name = "Nearest Neighbors"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = neighbors.NearestNeighbors(
      n_neighbors = n_neighbors,
      algorithm = algorithm,
      leaf_size = leaf_size,
      n_jobs = -1, # Use all CPU cores
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time.time()

    machine.fit(self.X_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Query the model
    time_previous = time.time()

    distances, indices = machine.kneighbors(self.X_test[:10])

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to query model")

    ### Evaluate the model
    time_previous = time.time()

    brute_force_distances = np.argsort(
      metrics.pairwise_distances(self.X_train, self.X_test[:1], metric='euclidean'),
      axis = 0
    )[:5].flatten()
    correct_matches = np.intersect1d(indices, brute_force_distances).size

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to brute force")

    ### Finalize
    if args.verbosity > 0:
      print()

    # Print the indices of nearest neighbors and their distances
    if self.args.verbosity > 1:
      for i, (d, idx) in enumerate(zip(distances, indices)):
          print(f"Test sample {i}: {self.X_test[i]}")
          print(f"  Nearest neighbors (indices): {idx}")
          print(f"  Distances: {d}")
          print()

    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": correct_matches / 5 * 100,
      "Time (s)": time.time() - time_start,
      "n_neighbors": 5,
      "algorithm": "ball_tree",
      "leaf_size": 30,
    })

  def KNearestClassifier(self, n_neighbors = 5, weights = "uniform", algorithm = "ball_tree", leaf_size = 30, p = 2):
    """
    KNN Classifier is a supervised technique that attempts to predict
    the class of a target data point by computing the local probability.

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

    Default Parameters:
      KNeighborsClassifier(
        n_neighbors = 5,
        *,
        weights = 'uniform',
        algorithm = 'auto',
        leaf_size = 30,
        p = 2,
        metric = 'minkowski',
        metric_params = None,
        n_jobs = None
      )
    """

    ### Initialize
    name = "K Nearest Neighbors"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = neighbors.KNeighborsClassifier(
      n_neighbors = n_neighbors,
      weights = weights,
      algorithm = algorithm,
      leaf_size = leaf_size,
      p = p,
      n_jobs = 8, # Use all CPU cores
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time.time()

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"n_neighbors={n_neighbors}, weights={weights}, algorithm={algorithm}, leaf_size={leaf_size}, p={p}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def KNeighborsRegressor(self, n_neighbors = 5, weights = "uniform", leaf_size = 30, p = 2):
    """
    KNN Regressor is a supervised technique that attempts to predict
    the value of a target data point by computing the local average.

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor

    Default Parameters:
      KNeighborsRegressor(
        n_neighbors = 5,
        *,
        weights = 'uniform',
        algorithm = 'auto',
        leaf_size = 30,
        p = 2,
        metric = 'minkowski',
        metric_params = None,
        n_jobs = None
      )
    """

    ### Initialize
    name = "K Nearest Neighbors"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = neighbors.KNeighborsRegressor(
      n_neighbors = n_neighbors,
      weights = weights,
      leaf_size = leaf_size,
      p = p,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time.time()

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"n_neighbors={n_neighbors}, weights={weights}, leaf_size={leaf_size}, p={p}"
    self.results["regressors"].append({
      "Method": f"{name} ({weights})",
      "MSE": metrics.mean_squared_error(self.y_test, y_pred),
      "RMSE": metrics.root_mean_squared_error(self.y_test, y_pred),
      "MAE": metrics.mean_absolute_error(self.y_test, y_pred),
      "R^2": metrics.r2_score(self.y_test, y_pred),
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def CustomSVM(self, learning_rate = 0.001, C = 1.0, iterations = 1000):
    """
    A custom SVM classifier from SVM_Model.py.
    """

    ### Initialize
    name = "Custom SVM"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = SVM_Model(
      learning_rate = learning_rate,
      lambda_param = C,
      iterations = iterations,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"C={C}, rate={learning_rate}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def SVC(self, C = 1.0, kernel = "linear", tol = 0.001, gamma = "scale", random_state = 42):
    """
    Support Vector Classifier is a supervised technique that attempts
    to find the hyperplane that best separates the classes.

    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

    Default Parameters:
      SVC(
        *,
        C = 1.0,
        kernel = 'rbf',
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        shrinking = True,
        probability = False,
        tol = 0.001,
        cache_size = 200,
        class_weight = None,
        verbose = False,
        max_iter = -1,
        decision_function_shape = 'ovr',
        break_ties = False,
        random_state = None
      )
    """

    ### Initialize
    name = "SVC"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = svm.SVC(
      C = C,
      kernel = kernel,
      tol = tol,
      gamma = gamma,
      random_state = random_state,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"C={C}, kernel={kernel}, tol={tol}, gamma={gamma}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def LinearSVC(self, loss = 'hinge', C = 1.0, tol = 0.0001, max_iter = 1000, random_state = 42):
    """
    Linear Support Vector Classifier is a supervised technique that
    attempts to find the hyperplane that best separates the classes.

    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    Default Parameters:
      LinearSVC(
        *,
        penalty = 'l2',
        loss = 'squared_hinge',
        dual = True,
        tol = 1e-4,
        C = 1.0,
        multi_class = 'ovr',
        fit_intercept = True,
        intercept_scaling = 1,
        class_weight = None,
        verbose = 0,
        random_state = None,
        max_iter = 1000
      )
    """

    ### Initialize
    name = "Linear SVC"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = svm.LinearSVC(
      loss = loss,
      C = C,
      tol = tol,
      max_iter = max_iter,
      random_state = random_state,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Print the results
    if args.verbosity > 0:
      print()

    configuration = f"loss={loss}, C={C}, tol={tol}, max_iter={max_iter}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def SGDClassifier(self, loss = "hinge", alpha = 0.0001, tol = 0.001, random_state = 42):
    """
    Stochastic Gradient Descent Classifier is a supervised technique
    that attempts to find the hyperplane that best separates the classes.

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier

    Default Parameters:
      SGDClassifier(
        *,
        loss = 'hinge',
        penalty = 'l2',
        alpha = 0.0001,
        l1_ratio = 0.15,
        fit_intercept = True,
        max_iter = 1000,
        tol = 0.001,
        shuffle = True,
        verbose = 0,
        epsilon = 0.1,
        n_jobs = None,
        random_state = None,
        learning_rate = 'optimal',
        eta0 = 0.0,
        power_t = 0.5,
        early_stopping = False,
        validation_fraction = 0.1,
        n_iter_no_change = 5,
        class_weight = None,
        warm_start = False,
        average = False
      )

    Why use SGDClassifier instead of SVC/Standard SVM?

    Feature            SVC (Standard SVM)                SGDClassifier (SVM with SGD)
    ---------          -------------------               --------------------------
    Dataset Size       Works well for small datasets     Scales well to large datasets
    Training Speed     Slower for large data             Faster, trains with mini-batches
    Memory Usage       Uses full dataset (high memory)   Uses one sample at a time (low memory)
    Online Learning    No (batch training)               Yes (can update model with new data)
    """

    ### Initialize
    name = "SGD"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = linear_model.SGDClassifier(
      loss = loss,
      alpha = alpha,
      tol = tol,
      random_state = random_state,
      n_jobs = -1,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"loss={loss}, alpha={alpha}, tol={tol}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def RandomForests(self, n_estimators = 100, max_depth = 2, bootstrap = True, random_state = 0):
    """
    Random Forests is an ensemble technique that uses multiple decision
    trees to predict the target values.

    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Default Parameters:
      RandomForestClassifier(
        n_estimators = 100,
        *,
        criterion = 'gini',
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        min_weight_fraction_leaf = 0.0,
        max_features = 'auto',
        max_leaf_nodes = None,
        min_impurity_decrease = 0.0,
        min_impurity_split = None,
        bootstrap = True,
        oob_score = False,
        n_jobs = None,
        random_state = None,
        verbose = 0,
        warm_start = False,
        class_weight = None,
        ccp_alpha = 0.0,
        max_samples = None
      )
    """

    ### Initialize
    name = "Random Forests"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = ensemble.RandomForestClassifier(
      n_estimators = n_estimators,
      max_depth = max_depth,
      bootstrap = bootstrap,
      random_state = random_state,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"n_estimators={n_estimators}, max_depth={max_depth}, bootstrap={bootstrap}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def AdaBoost(self, n_estimators = 50, learning_rate = 1.0, random_state = 42):
    """
    AdaBoost is an ensemble technique that uses multiple weak learners
    to predict the target values.

    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

    Default Parameters:
      AdaBoostClassifier(
        base_estimator = None,
        n_estimators = 50,
        *,
        learning_rate = 1.0,
        algorithm = 'SAMME.R',
        random_state = None
      )
    """

    ### Initialize
    name = "AdaBoost"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = ensemble.AdaBoostClassifier(
      n_estimators = n_estimators,
      learning_rate = learning_rate,
      random_state = random_state,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"n_estimators={n_estimators}, learning_rate={learning_rate}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def LogisticRegression(self, C = 1.0, tol = 0.0001, max_iter = 100, solver = 'lbfgs', random_state = 42):
    """
    Logistic Regression is a supervised technique that attempts to
    predict the class of a target data point by computing the local
    probability.

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    Default Parameters:
      LogisticRegression(
        *,
        penalty = 'l2',
        dual = False,
        tol = 0.0001,
        C = 1.0,
        fit_intercept = True,
        intercept_scaling = 1,
        class_weight = None,
        random_state = None,
        solver = 'lbfgs',
        max_iter = 100,
        multi_class = 'auto',
        verbose = 0,
        warm_start = False,
        n_jobs = None,
        l1_ratio = None
      )
    """
    ### Initialize
    name = "Logistic"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = linear_model.LogisticRegression(
      max_iter = 1000,
      solver = 'lbfgs'
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"C={C}, tol={tol}, max_iter={max_iter}, solver={solver}"
    self.results["regressors"].append({
      "Method": name,
      "MSE": metrics.mean_squared_error(self.y_test, y_pred),
      "RMSE": metrics.root_mean_squared_error(self.y_test, y_pred),
      "MAE": metrics.mean_absolute_error(self.y_test, y_pred),
      "R^2": metrics.r2_score(self.y_test, y_pred),
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def XGBClassifier(self, max_depth = 3, learning_rate = 0.1, n_estimators = 100, gamma = 0, random_state = 42, tree_method = "hist"):
    """
    XGBClassifier is an ensemble technique that uses multiple decision
    trees to classify the target values.

    https://xgboost.readthedocs.io/en/latest/python/python_api.html

    Default Parameters:
      XGBClassifier(
        max_depth = 3,
        learning_rate = 0.1,
        n_estimators = 100,
        verbosity = 1,
        objective = 'binary:logistic',
        booster = 'gbtree',
        tree_method = 'auto',
        n_jobs = 1,
        gpu_id = -1,
        gamma = 0,
        min_child_weight = 1,
        max_delta_step = 0,
        subsample = 1,
        colsample_bytree = 1,
        colsample_bylevel = 1,
        colsample_bynode = 1,
        reg_alpha = 0,
        reg_lambda = 1,
        scale_pos_weight = 1,
        base_score = 0.5,
        random_state = 0,
        missing = None,
        num_parallel_tree = 1,
        monotone_constraints = None,
        interaction_constraints = None,
        importance_type = 'gain',
        validate_parameters = True
    """

    ### Initialize
    name = "XGBoost"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = xgboost.XGBClassifier(
      tree_method = tree_method,
      max_depth = max_depth,
      learning_rate = learning_rate,
      n_estimators = n_estimators,
      gamma = gamma,
      random_state = random_state,
      early_stopping_rounds = 20,
      verbosity = 0,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Encode the target values
    time_previous = time.time()

    le = LabelEncoder()
    y_train = le.fit_transform(self.y_train)
    y_test = le.fit_transform(self.y_test)

    eval_set = [(self.X_train, y_train), (self.X_test, y_test)]

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to encode values")

    ### Fit to training data
    time_previous = time_start

    #machine.fit(self.X_train, self.y_train, verbose = False)
    machine.fit(self.X_train, y_train, eval_set = eval_set, verbose = False)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}, gamma={gamma}, tree_method={tree_method}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred, average='micro') * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred, average='micro') * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred, average='micro') * 100,
      "Time (s)" : time.time() - time_start,
      "Configuration": configuration,
    })

  def XGBRegressor(self, max_depth = 3, learning_rate = 0.1, n_estimators = 100, gamma = 0, random_state = 42):
    """
    XGBRegressor is an ensemble technique that uses multiple decision
    trees to predict the target values.

    https://xgboost.readthedocs.io/en/latest/python/python_api.html

    Default Parameters:
      XGBRegressor(
        max_depth = 3,
        learning_rate = 0.1,
        n_estimators = 100,
        verbosity = 1,
        objective = 'reg:squarederror',
        booster = 'gbtree',
        tree_method = 'auto',
        n_jobs = 1,
        gamma = 0,
        min_child_weight = 1,
        max_delta_step = 0,
        subsample = 1,
        colsample_bytree = 1,
        colsample_bylevel = 1,
        colsample_bynode = 1,
        reg_alpha = 0,
        reg_lambda = 1,
        scale_pos_weight = 1,
        base_score = 0.5,
        random_state = 0,
        missing = None,
        num_parallel_tree = 1,
        monotone_constraints = None,
        interaction_constraints = None,
        importance_type = 'gain',
        gpu_id = -1,
        validate_parameters = True
      )
    """

    ### Initialize
    name = "XGBoost"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = xgboost.XGBRegressor(
      max_depth = max_depth,
      learning_rate = learning_rate,
      n_estimators = n_estimators,
      gamma = gamma,
      random_state = random_state,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}, gamma={gamma}"
    self.results["regressors"].append({
      "Method": name,
      "MSE": metrics.mean_squared_error(self.y_test, y_pred),
      "RMSE": metrics.root_mean_squared_error(self.y_test, y_pred),
      "MAE": metrics.mean_absolute_error(self.y_test, y_pred),
      "R^2": metrics.r2_score(self.y_test, y_pred),
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def QuadraticDiscriminantAnalysis(self, reg_param = 0.0, tol = 0.0001):
    """
    Quadratic Discriminant Analysis is a supervised technique that
    attempts to find the hyperplane that best separates the classes.

    https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html

    Default Parameters:
      QuadraticDiscriminantAnalysis(
        priors = None,
        reg_param = 0.0,
        store_covariance = False,
        tol = 0.0001
      )
    """
    ### Initialize
    name = "Quadratic Discriminant Analysis"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = discriminant_analysis.QuadraticDiscriminantAnalysis(
      reg_param = reg_param,
      tol = tol,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"reg_param={reg_param}, tol={tol}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def GaussianProcessClassifier(self, random_state = 42):
    """
    Gaussian Process Classifier is a supervised technique that
    attempts to find the hyperplane that best separates the classes.

    https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html

    Default Parameters:
      GaussianProcessClassifier(
        kernel = None,
        optimizer = 'fmin_l_bfgs_b',
        n_restarts_optimizer = 0,
        alpha = 1e-10,
        copy_X_train = True,
        random_state = None,
        multi_class = 'one_vs_rest',
        warm_start = False,
        max_iter_predict = 100,
        n_jobs = None
      )
    """

    ### Initialize
    name = "Gaussian Process Classifier"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = gaussian_process.GaussianProcessClassifier(
      kernel = 1.0 * gaussian_process.kernels.RBF(1.0),
      random_state = random_state,
      copy_X_train = False
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = ""
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def GaussianNB(self, var_smoothing = 1e-9):
    """
    Gaussian Naive Bayes is a supervised technique that attempts to
    find the hyperplane that best separates the classes.

    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

    Default Parameters:
      GaussianNB(
        priors = None,
        var_smoothing = 1e-9
      )
    """

    ### Initialize
    name = "Gaussian Naive Bayes"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = naive_bayes.GaussianNB(
      var_smoothing = var_smoothing,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"var_smoothing={var_smoothing}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def DecisionTreeClassifier(self, criterion = "gini", max_depth = None, random_state = 42):
    """
    Decision Tree Classifier is a supervised technique that attempts
    to find the hyperplane that best separates the classes.

    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

    Default Parameters:
      DecisionTreeClassifier(
        *,
        criterion = 'gini',
        splitter = 'best',
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        min_weight_fraction_leaf = 0.0,
        max_features = None,
        random_state = None,
        max_leaf_nodes = None,
        min_impurity_decrease = 0.0,
        class_weight = None,
        ccp_alpha = 0.0
      )
    """

    ### Initialize
    name = "Decision Tree Classifier"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = tree.DecisionTreeClassifier(
      criterion = criterion,
      max_depth = max_depth,
      random_state = random_state,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"criterion={criterion}, max_depth={max_depth}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def LightGBMClassifier(self, boosting_type = "gbdt", num_leaves = 31, max_depth = -1, learning_rate = 0.1, n_estimators = 100, objective = None, random_state = None):
    """
    LightGBM is a gradient boosting framework that uses tree-based learning
    algorithms. It is designed for distributed and efficient training.

    https://lightgbm.readthedocs.io/en/latest/
    https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

    Default Parameters:
      LightGBM(
        boosting_type = 'gbdt',
        num_leaves = 31,
        max_depth = -1,
        learning_rate = 0.1,
        n_estimators = 100,
        objective = None,
        class_weight = None,
        random_state = None,
        n_jobs = -1,
        silent = True
      )
    """

    ### Initialize
    name = "LightGBM"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = lgb.LGBMClassifier(
      boosting_type = boosting_type,
      num_leaves = num_leaves,
      max_depth = max_depth,
      learning_rate = learning_rate,
      n_estimators = n_estimators,
      objective = objective,
      random_state = random_state,
      force_col_wise = True,
      verbosity = -1,
      n_jobs = 8,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"boosting_type={boosting_type}, num_leaves={num_leaves}, max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}, objective={objective}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def CatBoost(self, iterations = 1000, learning_rate = 0.1, depth = 6, l2_leaf_reg = 3, random_state = 42):
    """
    Catboost is a gradient boosting framework that uses tree-based learning
    algorithms. It is designed for distributed and efficient training.

    https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier

    Default Parameters:
      CatBoostClassifier(
        iterations = 1000,
        learning_rate = 0.1,
        depth = 6,
        l2_leaf_reg = 3,
        random_seed = None,
        loss_function = 'Logloss',
        eval_metric = None,
        verbose = True
      )
    """

    ### Initialize
    name = "CatBoost"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = catboost.CatBoostClassifier(
      iterations = iterations,
      learning_rate = learning_rate,
      depth = depth,
      l2_leaf_reg = l2_leaf_reg,
      random_seed = random_state,
      verbose = False,
      allow_writing_files = False,
      thread_count = 8,
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"iterations={iterations}, learning_rate={learning_rate}, depth={depth}, l2_leaf_reg={l2_leaf_reg}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

  def MLPClassifier(self, activation = 'relu', solver = 'adam', alpha = 0.0001, learning_rate = 'constant', learning_rate_init = 0.001, max_iter = 200, random_state = 42):
    """
    Multi-layer Perceptron Classifier is a supervised technique that
    attempts to find the hyperplane that best separates the classes.

    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

    Default Parameters:
      MLPClassifier(
        hidden_layer_sizes = (100,),
        *,
        activation = 'relu',
        solver = 'adam',
        alpha = 0.0001,
        batch_size = 'auto',
        learning_rate = 'constant',
        learning_rate_init = 0.001,
        power_t = 0.5,
        max_iter = 200,
        shuffle = True,
        random_state = None,
        tol = 0.0001,
        verbose = False,
        warm_start = False,
        momentum = 0.9,
        nesterovs_momentum = True,
        early_stopping = False,
        validation_fraction = 0.1,
        beta_1 = 0.9,
        beta_2 = 0.999,
        epsilon = 1e-08,
        n_iter_no_change = 10
      )
    """

    ### Initialize
    name = "MLP Classifier"
    if self.args.verbosity > 0:
      print(name)

    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = neural_network.MLPClassifier(
      activation=activation,
      solver=solver,
      alpha=alpha,
      learning_rate=learning_rate,
      learning_rate_init=learning_rate_init,
      max_iter=max_iter,
      random_state=random_state
    )

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to create model")

    ### Fit to training data
    time_previous = time_start

    machine.fit(self.X_train, self.y_train)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to train model")

    ### Make predictions
    time_previous = time.time()

    y_pred = machine.predict(self.X_test)

    if self.args.verbosity > 0:
      print(f"  {time.time() - time_previous:.4f}  Time to generate predictions")

    ### Finalize
    if args.verbosity > 0:
      print()

    configuration = f"activation={activation}, solver={solver}, alpha={alpha}, learning_rate={learning_rate}, learning_rate_init={learning_rate_init}, max_iter={max_iter}"
    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": metrics.accuracy_score(self.y_test, y_pred) * 100,
      "F1": metrics.f1_score(self.y_test, y_pred) * 100,
      "Precision": metrics.precision_score(self.y_test, y_pred) * 100,
      "Recall": metrics.recall_score(self.y_test, y_pred) * 100,
      "Time (s)": time.time() - time_start,
      "Configuration": configuration,
    })

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Compare learning methods for homework 2")

  parser.add_argument('--dataset', required = True, choices = ['relax', 'skin'])
  parser.add_argument("--test_ratio", type = float, default = 0.2)
  parser.add_argument("--random_state", type = int, default = 42)
  parser.add_argument("--verbosity", type = int, default = 0)
  parser.add_argument("--regressors", action = "store_true", help = "Run regression models")
  parser.add_argument("--classifiers", action = "store_true", help = "Run classification models")
  parser.add_argument("--all", action = "store_true", help = "Run all models")

  args = parser.parse_args()

  c = Compare(args)
  c.run()
