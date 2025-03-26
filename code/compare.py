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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost

import matplotlib.pyplot as plt

from data import Data

class Compare:
  """
  Compare learning techniques for data analysis.
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

    self.results = []

  def run(self):
    """
    Train and evaluate various models.
    """

    # Nearest Neighbors is an unsupervised technique that finds the
    # k-nearest neighbors of a target data point.
    self.NearestNeighbors()

    # KNN Classifier is a supervised technique that attempts to predict
    # the class of a target data point by computing the local probability.
    self.KNearestClassifier()

    # KNN Regressor is a supervised technique that attempts to predict
    # the value of a target data point by computing the local average.
    self.KNeighborsRegressor()

    # Support Vector Classifier is a supervised technique that attempts
    # to find the hyperplane that best separates the classes.
    self.SVC()

    # Linear Support Vector Classifier is a supervised technique that
    # attempts to find the hyperplane that best separates the classes.
    self.LinearSVC()

    # Stochastic Gradient Descent Classifier is a supervised technique
    # that attempts to find the hyperplane that best separates the classes.
    self.SGDClassifier()

    # Random Forests is an ensemble technique that uses multiple decision
    # trees to predict the target values.
    self.RandomForests()

    # AdaBoost is an ensemble technique that uses multiple weak learners
    # to predict the target values.
    self.AdaBoost()

    # XGBClassifier is an ensemble technique that uses multiple decision trees
    # to classify the target values.
    self.XGBClassifier()

    # XGBRegressor is an ensemble technique that uses multiple decision trees
    # to predict the target values.
    self.XGBRegressor()

  def NearestNeighbors(self):
    """
    Use the Nearest Neighbors model to find the k-nearest neighbors.

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

    machine = neighbors.NearestNeighbors(
      n_neighbors = 5,
      algorithm = 'ball_tree',   # for efficiency
      n_jobs = -1                # Use all CPU cores
    )

    # Fit to training data
    machine.fit(self.X_train)

    # Find the 5 nearest neighbors for the first 10 test samples
    distances, indices = machine.kneighbors(self.X_test[:10])

    # Print the indices of nearest neighbors and their distances
    #for i, (d, idx) in enumerate(zip(distances, indices)):
    #    print(f"Test sample {i}:")
    #    print(f"  Nearest neighbors (indices): {idx}")
    #    print(f"  Distances: {d}\n")

  def KNearestClassifier(self):
    """
    Use the k-Nearest Neighbors classifier to predict the target values.

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
    machine = neighbors.KNeighborsClassifier(
      n_neighbors = 5,
      algorithm = 'ball_tree',  # for efficiency
      n_jobs = -1               # Use all CPU cores
    )

    # Train the model
    machine.fit(self.X_train, self.y_train)

    # Predict the target values
    y_pred = machine.predict(self.X_test)

    # Evaluate accuracy
    self.results.append([
      "KNeighborsClassifier",
      accuracy_score(self.y_test, y_pred),
    ])

  def KNeighborsRegressor(self):
    """
    Use the k-Nearest Neighbors regression model to predict the target values.

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
    n_neighbors = 5

    for i, weights in enumerate(["uniform", "distance"]):
        machine = neighbors.KNeighborsRegressor(
          n_neighbors,
          weights = weights
        )

        # Train the model
        machine.fit(self.X_train, self.y_train)

        # Predict the target values
        y_pred = machine.predict(self.X_test)

        # Evaluate accuracy
        self.results.append([
          f"KNeighborsRegressor - {weights}",
          mean_squared_error(self.y_test, y_pred),
        ])

        #plt.subplot(2, 1, i + 1)
        #plt.scatter(self.X_train, self.y_train, color="darkorange", label="data")
        #plt.plot(self.X_test, y_pred, color="navy", label="prediction")
        #plt.axis("tight")
        #plt.legend()
        #plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

    #plt.tight_layout()
    #plt.show()

  def SVC(self):
    """
    Use the Support Vector Classifier to predict the target values.

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

    machine = svm.SVC(
      kernel = 'linear'
    )

    # Train the model
    machine.fit(self.X_train, self.y_train)

    # Predict the target values
    y_pred = machine.predict(self.X_test)

    # Evaluate accuracy
    self.results.append([
      "SVC",
      accuracy_score(self.y_test, y_pred),
    ])

  def LinearSVC(self):
    """
    Use the Linear Support Vector Classifier to predict the target values.

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

    machine = svm.LinearSVC(
      loss = 'hinge',
      max_iter = 100000
    )

    # Train the model
    machine.fit(self.X_train, self.y_train)

    # Predict the target values
    y_pred = machine.predict(self.X_test)

    # Evaluate accuracy
    self.results.append([
      "LinearSVC",
      accuracy_score(self.y_test, y_pred),
    ])

  def SGDClassifier(self):
    """
    Use the Stochastic Gradient Descent classifier to predict the target values.

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

    machine = linear_model.SGDClassifier()

    # Train the model
    machine.fit(self.X_train, self.y_train)

    # Predict the target values
    y_pred = machine.predict(self.X_test)

    # Evaluate accuracy
    self.results.append([
      "SGDClassifier",
      accuracy_score(self.y_test, y_pred),
    ])

  def RandomForests(self):
    """
    Use the Random Forests model to predict the target values.

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

    machine = ensemble.RandomForestClassifier(
      n_estimators = 100,
      max_depth = 2,
      random_state = 0
    )

    # Train the model
    machine.fit(self.X_train, self.y_train)

    # Predict the target values
    y_pred = machine.predict(self.X_test)

    # Evaluate accuracy
    self.results.append([
      "RandomForestClassifier",
      accuracy_score(self.y_test, y_pred),
    ])

  def AdaBoost(self):
    """
    Use the AdaBoost model to predict the target values.

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

    machine = ensemble.AdaBoostClassifier(
      n_estimators = 100
    )

    # Train the model
    machine.fit(self.X_train, self.y_train)

    # Predict the target values
    y_pred = machine.predict(self.X_test)

    # Evaluate accuracy
    self.results.append([
      "AdaBoostClassifier",
      accuracy_score(self.y_test, y_pred),
    ])

  def XGBClassifier(self):
    """
    Use the XGBClassifier to classify the target values.

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

    machine = xgboost.XGBClassifier(
      tree_method = "hist",
      early_stopping_rounds = 20,
      verbosity = 0,
    )

    le = LabelEncoder()
    y_train = le.fit_transform(self.y_train)
    y_test = le.fit_transform(self.y_test)

    eval_set = [(self.X_train, y_train), (self.X_test, y_test)]

    # Train the model
    machine.fit(self.X_train, y_train, eval_set = eval_set, verbose = False)

    # Predict the target values
    y_pred = machine.predict(self.X_test)

    # Evaluate accuracy
    self.results.append([
      "XGBoostClassifier",
      accuracy_score(self.y_test, y_pred),
    ])

    #results = machine.evals_result()

    #plt.figure(figsize=(10,7))
    #plt.plot(results["validation_0"]["logloss"], label="Training loss")
    #plt.plot(results["validation_1"]["logloss"], label="Validation loss")
    #plt.axvline(machine.best_iteration, color="gray", label="Optimal tree number")
    #plt.xlabel("Number of trees")
    #plt.ylabel("Loss")
    #plt.legend()
    #plt.show()

  def XGBRegressor(self):
    """
    Use the XGBRegressor to predict the target values.

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

    machine = xgboost.XGBRegressor(
      n_estimators = 100,
    )

    # Train the model
    machine.fit(self.X_train, self.y_train)

    # Predict the target values
    y_pred = machine.predict(self.X_test)

    # Evaluate accuracy
    self.results.append([
      "XGBoostRegressor",
      mean_squared_error(self.y_test, y_pred),
    ])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Compare learning methods for homework 2")
  parser.add_argument('--dataset', required = True, choices = ['relax', 'skin'])
  parser.add_argument("--test_ratio", type = float, default = 0.2)
  parser.add_argument("--random_seed", type = int, default = 42)
  args = parser.parse_args()

  c = Compare(args)
  c.run()
