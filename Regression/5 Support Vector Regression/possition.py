# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print('Level')
print(x)
print('Salaries')
print(y)

y = y.reshape(len(y), 1)
print(y)

# feature scaling
sc_x = StandardScaler()
xt = sc_x.fit_transform(x)

sc_y = StandardScaler()
yt = sc_y.fit_transform(y)

print('transformed X')
print(xt)
print('transformed y')
print(yt)

regressor = SVR(kernel='rbf')
regressor.fit(xt, yt)

x1 = [[6.5]]
x1t = sc_x.transform(x1)
y1t = regressor.predict(x1t)
y1 = sc_y.inverse_transform(y1t)
print('predicted salary for ' + str(x1))
print(y1)

# Visualising the Training set results
plt.scatter(x, y, color='red')

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid)), 1)
x_grid_t = sc_x.transform(x_grid)
y_pred = regressor.predict(x_grid_t)
y_pred_inv = sc_y.inverse_transform(y_pred)
plt.plot(x_grid, y_pred_inv, color='blue')
plt.title('Salary vs Position (Support Vector Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
