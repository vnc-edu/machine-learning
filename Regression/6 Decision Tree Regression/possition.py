# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print('Level')
print(x)
print('Salaries')
print(y)


regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

x1 = [[6.5]]
y1 = regressor.predict(x1)
print('predicted salary for ' + str(x1))
print(y1)

# Visualising the Training set results
plt.scatter(x, y, color='red')

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid)), 1)
y_pred = regressor.predict(x_grid)
plt.plot(x_grid, y_pred, color='blue')
plt.title('Salary vs Position (Decision Tree Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
