from ucimlrepo import fetch_ucirepo

# fetch dataset
skin_segmentation = fetch_ucirepo(id=229)

# data (as pandas dataframes)
X = skin_segmentation.data.features
y = skin_segmentation.data.targets

# metadata
print(skin_segmentation.metadata)

# variable information
print(skin_segmentation.variables)