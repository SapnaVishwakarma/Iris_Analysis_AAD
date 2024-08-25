# Required libraries import karein
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dataset create karein (YearsExperience aur Salary ke saath)
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 65000, 70000, 85000, 90000, 105000, 115000, 125000]
}

df = pd.DataFrame(data)

# Features aur target variable define karein
X = df[['YearsExperience']]  # Features (YearsExperience)
y = df['Salary']  # Target variable (Salary)

# Data ko training aur testing sets me split karein (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model create karein
model = LinearRegression()

# Training data par model fit karein
model.fit(X_train, y_train)

# Testing set par prediction karein
y_pred = model.predict(X_test)

# Model ke performance ko Mean Squared Error (MSE) se evaluate karein
mse = mean_squared_error(y_test, y_pred)

# Results print karein
print("Predicted Salaries:", y_pred)
print("Mean Squared Error (MSE):", mse)
