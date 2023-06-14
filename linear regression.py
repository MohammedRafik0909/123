import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample input features and target variable
X = np.array([[1], [2], [3], [4], [5]])  # Input features
y = np.array([2, 4, 6, 8, 10])  # Target variable

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
X_test = np.array([[6], [7], [8]])  # New input features
y_pred = model.predict(X_test)

# Plot the data points and the regression line
plt.scatter(X, y, color='b', label='Actual data')
plt.plot(X_test, y_pred, color='r', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
