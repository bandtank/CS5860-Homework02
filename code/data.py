import argparse
import sys

import pandas as pd

class Data:
  """
  Fetch the data for Homework 2
  """

  def __init__(self, dataset):
    """
    Load the dataset based on the user's choice.
    """

    if dataset == 'skin':
      self.X, self.y = self.skin()

    elif dataset == 'relax':
      self.X, self.y = self.relax()

    else:
      print("Invalid dataset choice.")
      sys.exit()

  def get_data(self):
    return self.X, self.y

  def skin(self):
    """
    Load the Skin Segmentation dataset.
    """

    # Load the data
    path = "datasets/skin.txt"
    df = pd.read_csv(path, sep = '\t', encoding='utf-8').values

    # Split the data into features and labels
    X = df[:,:-1].astype(int)
    y = df[:,-1].astype(int)

    return X, y

  def relax(self):
    """
    Load the Planning Relax dataset.
    """

    # Load the data
    path = "datasets/relax.txt"
    df = pd.read_csv(path, sep = '\t', encoding='utf-8').values

    # Split the data into features and labels
    X = df[:,:-1].astype(float)
    y = df[:,-1].astype(int)

    return X, y

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Get the data for homework 2")
  parser.add_argument('--dataset', choices = ['skin', 'relax'])
  args = parser.parse_args()

  data = Data(args.dataset)
  X, y = data.get_data()
  print(f"X Shape: {X.shape}")
  print(f"Y Shape: {y.shape}")