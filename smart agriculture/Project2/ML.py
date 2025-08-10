import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

#importing csv file
h = pd.read_csv("data.csv")

#Null Values
h.isnull()

#Drop
X=h.drop("label",axis=1)
y=h["label"]

#Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Prediction
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)
tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)

#Cross_val_score
cross_val_score(tree_clf, X_train, y_train, cv=3, scoring="accuracy")

#Accuracy
accuracy = accuracy_score(y_test, y_pred)

#Feature importances
feature_importances = tree_clf.feature_importances_

#Modeling
tree_clf = DecisionTreeClassifier(random_state=42)

#fitting
tree_clf.fit(X_train, y_train)

#prediction
y_pred = tree_clf.predict(X_test)

#Accuracy
accuracy = accuracy_score(y_test, y_pred)


#Report
print(f"Accuracy: {accuracy * 100:.2f}%")

#Amount values
N = float(input("Enter the Nitrogen value: "))
P = float(input("Enter the Phosphorus value: "))
K = float(input("Enter the Potassium value: "))
temperature = float(input("Enter the temperature value: "))
humidity = float(input("Enter the humidity value: "))
ph = float(input("Enter the pH value: "))
rainfall = float(input("Enter the rainfall value: "))
user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                          columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
predicted_crop_label = tree_clf.predict(user_input)
print("Predicted Crop Label:", predicted_crop_label[0])

#Prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Fitting
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)
tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)

#Accuracy
print(f"\nAccuracy = {tree_clf.score(X_test, y_test)*100}%")
