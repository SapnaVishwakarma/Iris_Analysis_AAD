from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Display the first five rows
print("First five rows of the dataset:")
print(iris_df.head())

# Display the dataset's shape
print("\nDataset shape:")
print(iris_df.shape)

# Summary statistics
print("\nSummary statistics:")
print(iris_df.describe())