# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print('Level')
print(x)
print('Salaries')
print(y)

# Training the Linear Regression model on the whole dataset
lin_reg_1 = LinearRegression()
lin_reg_1.fit(x, y)

# Training the Polynomial Regression model on the whole dataset
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Getting the final linear regression equation with the values of the coefficients
print(lin_reg_1.coef_)
print(lin_reg_1.intercept_)



# Predicting the Test set results
y_pred = lin_reg_1.predict(x)

# Visualising the Training set results
plt.scatter(x, y, color='red')
plt.plot(x, y_pred, color='blue')
plt.title('Salary vs Position (linear)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Predicting the Test set results
y_poly_pred = lin_reg_2.predict(x_poly)
# Visualising the Training set results
plt.scatter(x, y, color='red')
plt.plot(x, y_poly_pred, color='blue')
plt.title('Salary vs Position (Polynomial)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# predict
x1 = [[6.5]]
y1 = lin_reg_1.predict(x1)
print('As per linear regression salary for ' + str(x1) + ' is :' + str(y1))

x_poly2 = poly_reg.fit_transform(x1)
y_poly_pred2 = lin_reg_2.predict(x_poly2)
print('As per polynomial regression salary for ' + str(x1) + ' is :' + str(y_poly_pred2))