import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample input features and target variable
X = np.array([[1], [2], [3], [4], [5], [6]])  # Input features
y = np.array([0, 0, 0, 1, 1, 1])  # Target variable

# Create a logistic regression model
model = LogisticRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
X_test = np.array([[0], [7]])  # New input features
y_pred = model.predict(X_test)
print(y_pred)
# Plot the data points and the decision boundary
plt.scatter(X, y, color='b', label='Actual data')
plt.plot(X_test, model.predict_proba(X_test)[:, 1], color='r', label='Decision boundary')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Logistic Regression')
plt.legend()
plt.show()
