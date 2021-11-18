# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print('Years of experience')
print(x)
print('Salaries')
print(y)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('x train')
print(x_train)
print('x test')
print(x_test)
print('y train')
print(y_train)
print('y test')
print(y_train)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
# Salary=9312.57512673×YearsExperience+26780.09915062818

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train, y_train, color='green')
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training-Green, Test-Red)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
