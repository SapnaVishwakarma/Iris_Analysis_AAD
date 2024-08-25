from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()

# Create DataFrame for features
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create Series for the target variable
iris_target = pd.Series(iris.target, name='species')

# Combine features and target into a single DataFrame
iris_df['species'] = iris_target

# Define features (X) and target (y)
X = iris_df.drop('species', axis=1)  # Features
y = iris_df['species']  # Target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the number of samples in each set
print(f"Number of samples in training set: {X_train.shape[0]}")
print(f"Number of samples in testing set: {X_test.shape[0]}")