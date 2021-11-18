import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataframe = pd.read_csv('Social_Network_Ads.csv')
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Random forest classifier model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting a new result
x1 = [[30, 87000]]
x1_test = sc.transform(x1)
y1_test = classifier.predict(x1_test)
print(y1_test)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print('Prediction on test set')
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print('Accuracy')
ac = accuracy_score(y_test, y_pred)
print(ac)
