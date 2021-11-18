# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Simple Linear Regression model on the Training set
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = linear_regressor.predict(x_test)

lr = r2_score(y_test, y_pred)
print('R2 for linear regression')
print(lr)

# Training the Polynomial Regression model on the whole dataset
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y_train)

x_test_poly  = poly_reg.transform(x_test)
y_test_pred_poly = lin_reg_2.predict(x_test_poly)
pr = r2_score(y_test, y_test_pred_poly)
print('R2 for polynomial regression')
print(pr)


# support vector
y1 = y.reshape(len(y), 1)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y1, test_size=0.2, random_state=0)

sc_x = StandardScaler()
xt = sc_x.fit_transform(x_train1)
sc_y = StandardScaler()
yt = sc_y.fit_transform(y_train1)
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(xt, yt)


x1t = sc_x.transform(x_test1)
y1t = svr_regressor.predict(x1t)
y_pred_svr = sc_y.inverse_transform(y1t)
svr_r2 = r2_score(y_test1, y_pred_svr)
print('R2 for support vector')
print(svr_r2)


#decision tree
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train, y_train)


y_pred_dcn = regressor.predict(x_test)
dcn_r2 = r2_score(y_test, y_pred_dcn)
print('R2 for decision tree')
print(dcn_r2)

# random forest
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x_train, y_train)


y_pred_rndf = regressor.predict(x_test)
rndf_r2 = r2_score(y_test, y_pred_rndf)
print('R2 for random forest')
print(rndf_r2)
