import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataframe = pd.read_csv('cancer_data.csv')
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the decision tree model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting a new result

# Predicting the Test set results
y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print('test set')
print(y_test)
print('Prediction on test set')
print(y_pred)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('Confusion Matrix')
print(cm)
print('Accuracy')
print(ac)
print(report)
