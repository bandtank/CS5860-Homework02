"""
Compare different regression models using multiple datasets. The
regression models are implemented using libraries. The datasets are
loaded from CSV files and split into training and testing sets. The
feature values are normalized using the StandardScaler class. The
regression models are trained and evaluated using the mean squared
error, mean absolute error, and R^2 score. The results are displayed
in a table and a scatter plot.

Author: Anthony Andriano
"""

import argparse
import tabulate
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
import xgboost

from data import Data

class Compare:
  """
  Compare learning techniques for data analysis.
  """

  def __init__(self, args):
    """
    Initialise the class and load the data.
    """

    ### Start initialization
    print("Initializing...")
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
      random_state = args.random_seed
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
    print(f"  {time.time() - time_previous:.4f}  Total time to initialize")
    print()

  def run(self):
    """
    Train and evaluate various models.
    """

    if self.args.classifiers or self.args.all:
      print("#######################")
      print("# Running Classifiers #")
      print("#######################")
      self.NearestNeighbors()
      self.KNearestClassifier()
      self.SVC()
      self.LinearSVC()
      self.SGDClassifier()
      self.RandomForests()
      self.AdaBoost()
      self.XGBClassifier()

      data = sorted(self.results["classifiers"], key = lambda d: float(d['Accuracy (%)']), reverse = True)

      print(tabulate.tabulate(
        data,
        headers = "keys",
        tablefmt = "pretty",
        colalign=["left", "center", "center"],
      ))
      print()

    if self.args.regressors or self.args.all:
      print("######################")
      print("# Running Regressors #")
      print("######################")
      self.KNeighborsRegressor()
      self.XGBRegressor()

      print(tabulate.tabulate(
        self.results["regressors"],
        headers = "keys",
        tablefmt = "pretty",
        colalign=["left", "center", "center", "center", "center", "center"],
      ))

  def NearestNeighbors(self):
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
    print(name)
    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = neighbors.NearestNeighbors(
      n_neighbors = 5,
      algorithm = 'ball_tree',   # for efficiency
      n_jobs = -1                # Use all CPU cores
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
      "Accuracy (%)": f"{correct_matches / 5 * 100:.2f}",
      "Time (s)": f"{time.time() - time_start:.4f}",
    })

  def KNearestClassifier(self):
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
    print(name)
    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = neighbors.KNeighborsClassifier(
      n_neighbors = 5,
      algorithm = 'ball_tree',  # for efficiency
      n_jobs = -1               # Use all CPU cores
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

    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": f"{metrics.accuracy_score(self.y_test, y_pred) * 100:.2f}",
      "Time (s)": f"{time.time() - time_start:.4f}",
    })

  def KNeighborsRegressor(self):
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
    n_neighbors = 5

    for i, weights in enumerate(["uniform", "distance"]):
        print(f"{name} (Weights = {weights})")
        time_start = time.time()

        ### Create the model
        time_previous = time_start

        machine = neighbors.KNeighborsRegressor(
          n_neighbors,
          weights = weights
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

        self.results["regressors"].append({
          "Method": f"{name} ({weights})",
          "MSE": f"{metrics.mean_squared_error(self.y_test, y_pred):.2f}",
          "RMSE": f"{metrics.root_mean_squared_error(self.y_test, y_pred):.2f}",
          "MAE": f"{metrics.mean_absolute_error(self.y_test, y_pred):.2f}",
          "R^2": f"{metrics.r2_score(self.y_test, y_pred):.2f}",
          "Time (s)": f"{time.time() - time_start:.4f}",
        })

  def SVC(self):
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
    print(name)
    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = svm.SVC(
      kernel = 'linear'
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

    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": f"{metrics.accuracy_score(self.y_test, y_pred) * 100:.2f}",
      "Time (s)": f"{time.time() - time_start:.4f}",
    })

  def LinearSVC(self):
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
    print(name)
    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = svm.LinearSVC(
      loss = 'hinge',
      max_iter = 100000
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

    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": f"{metrics.accuracy_score(self.y_test, y_pred) * 100:.2f}",
      "Time (s)": f"{time.time() - time_start:.4f}",
    })

  def SGDClassifier(self):
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
    print(name)
    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = linear_model.SGDClassifier()

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

    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": f"{metrics.accuracy_score(self.y_test, y_pred) * 100:.2f}",
      "Time (s)": f"{time.time() - time_start:.4f}",
    })

  def RandomForests(self):
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
    print(name)
    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = ensemble.RandomForestClassifier(
      n_estimators = 100,
      max_depth = 2,
      random_state = 0
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

    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": f"{metrics.accuracy_score(self.y_test, y_pred) * 100:.2f}",
      "Time (s)": f"{time.time() - time_start:.4f}",
    })

  def AdaBoost(self):
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
    print(name)
    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = ensemble.AdaBoostClassifier(
      n_estimators = 100
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

    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": f"{metrics.accuracy_score(self.y_test, y_pred) * 100:.2f}",
      "Time (s)": f"{time.time() - time_start:.4f}",
    })

  def XGBClassifier(self):
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
    print(name)
    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = xgboost.XGBClassifier(
      tree_method = "hist",
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

    self.results["classifiers"].append({
      "Method": name,
      "Accuracy (%)": f"{metrics.accuracy_score(y_test, y_pred) * 100:.2f}",
      "Time (s)" : f"{time.time() - time_start:.4f}",
    })

  def XGBRegressor(self):
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
    print(name)
    time_start = time.time()

    ### Create the model
    time_previous = time_start

    machine = xgboost.XGBRegressor(
      n_estimators = 100,
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

    self.results["regressors"].append({
      "Method": name,
      "MSE": f"{metrics.mean_squared_error(self.y_test, y_pred):.2f}",
      "RMSE": f"{metrics.root_mean_squared_error(self.y_test, y_pred):.2f}",
      "MAE": f"{metrics.mean_absolute_error(self.y_test, y_pred):.2f}",
      "R^2": f"{metrics.r2_score(self.y_test, y_pred):.2f}",
      "Time (s)": f"{time.time() - time_start:.4f}",
    })

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Compare learning methods for homework 2")

  parser.add_argument('--dataset', required = True, choices = ['relax', 'skin'])
  parser.add_argument("--test_ratio", type = float, default = 0.2)
  parser.add_argument("--random_seed", type = int, default = 42)
  parser.add_argument("--verbosity", type = int, default = 0)
  parser.add_argument("--regressors", action = "store_true", help = "Run regression models")
  parser.add_argument("--classifiers", action = "store_true", help = "Run classification models")
  parser.add_argument("--all", action = "store_true", help = "Run all models")

  args = parser.parse_args()

  c = Compare(args)
  c.run()
