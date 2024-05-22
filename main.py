import pandas as pd
import numpy as np
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import tkinter as tk
from tkinter import messagebox

myDF = pd.read_csv("diabetes.csv")

print(myDF.info())

#Data has no null entries

#Split data into training and testing sets

#Here we have X = features

X = myDF.drop('Outcome', axis=1)

#Y is equal to only the outcome column
y = myDF['Outcome']

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Next, I scale the data using standardization (subtracting mean and dividng stdev from every feature)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Next step is performing Machine Learning Classification (whether a patient has diabetes or not)
# Implementing Logistic Regression, because it is suitable for binary classification

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#Now, I use the test data on the model and pick up some statistics such as Accuracy, Precision, F1, and Recall scores.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')

# Function to predict diabetes
def predict_diabetes(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]


# Example usage
input_data = [6, 148, 70, 20, 85, 32.1, 0.6, 25]  # Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI,
                                                  # Diabetes Pedigree Function, Age
prediction = predict_diabetes(input_data)
print("Prediction:", "Diabetic" if prediction == 1 else "Not Diabetic")
