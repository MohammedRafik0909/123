import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

x = ([[1],[2],[3],[4],[5]])
y=([0,0,0,1,1])

model = LogisticRegression()
model.fit(x,y)
X_new = ([[0],[7]])
y_pred = model.predict(X_new)

print(y_pred)

plt.scatter(x,y,color='b',label='actual data')
plt.plot(X_new,model.predict_proba(X_new)[:,1], color='r',label='regression line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("LINEAR REGRESSION")
plt.legend()
plt.show()
