import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Sample input features and target variable
X = np.array([[1], [2], [3], [4], [5]])  # Input features
y = np.array([2, 4, 6, 8, 10])  # Target variable

# Linear Regression
# Create a linear regression model
linear_model = LinearRegression()

# Fit the model to the data
linear_model.fit(X, y)

# Make predictions
X_test = np.array([[0], [6]])  # New input features
y_linear_pred = linear_model.predict(X_test)

# Polynomial Regression
# Transform input features into polynomial features
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Create a polynomial regression model
poly_model = LinearRegression()

# Fit the model to the transformed data
poly_model.fit(X_poly, y)

# Transform test features into polynomial features
X_test_poly = poly_features.transform(X_test)
y_poly_pred = poly_model.predict(X_test_poly)

# Plot the actual data points and the regression lines
plt.scatter(X, y, color='b', label='Actual data')

plt.plot(X_test, y_linear_pred, color='r', label='Linear regression')
plt.plot(X_test, y_poly_pred, color='g', label='Polynomial regression')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear and Polynomial Regression')
plt.legend()
plt.show()
