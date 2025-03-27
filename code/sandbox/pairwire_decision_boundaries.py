import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from itertools import combinations

# Generate synthetic data with 4 features
X, y = make_classification(n_samples=100, n_features=4, n_informative=4, n_redundant=0, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create pairwise plots of decision boundaries
feature_combinations = combinations(range(X.shape[1]), 2)
for i, (feature1, feature2) in enumerate(feature_combinations):
    plt.figure(figsize=(8, 6))

    # Create a meshgrid of points
    x_min, x_max = X[:, feature1].min() - 1, X[:, feature1].max() + 1
    y_min, y_max = X[:, feature2].min() - 1, X[:, feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Predict the class for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())]).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

    # Plot the data points
    plt.scatter(X[:, feature1], X[:, feature2], c=y, cmap=plt.cm.RdBu, edgecolors='k')

    plt.xlabel(f"Feature {feature1}")
    plt.ylabel(f"Feature {feature2}")
    plt.title(f"Decision Boundary for Features {feature1} and {feature2}")
    plt.show()