import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import libraries
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

# Encoding categorical data
# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('x train')
print(x_train)
print('y train')
print(y_train)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Getting the final multiple regression equation with the values of the coefficients
print('co efficents')
print(regressor.coef_)
print('constant')
print(regressor.intercept_)
# Salary=9312.57512673×YearsExperience+26780.09915062818

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Visualising the Training set results
np.set_printoptions(precision=2)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), axis=1))


# Making a single prediction (for example the profit of a startup with R&D Spend = 160000,
# Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))