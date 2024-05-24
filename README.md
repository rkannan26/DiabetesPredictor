# Diabetes Prediction Model

## Overview
This project implements a machine learning model to predict the likelihood of diabetes in patients based on various health metrics. The model is built using logistic regression and Data Analysis is performed before and after model creation.

## Features

- **Pregnancies**: Number of pregnancies the patient has had.
- **Glucose**: The patient's Glucose Level.
- **BloodPressure**: The patient's Blood Pressure
- **SkinThickness**: Skin Thickness (mm).
- **Insulin**: The patient's Insulin Level.
- **BMI**: Patient's Body Mass Index (weight in kg/(height in m)^2).
- **Diabetes Pedigree Function**: Diabetes Pedigree Function (a function that represents how likely the patient is to have diabetes based on family history /ancestry).
- **Age**: Age of the patient.

## Files

- **diabetes.csv**: Dataset containing the input features and target variable.
- **main.py**: Contains model training, evaluation, and prediction code.
- **model.pkl**: Saved logistic regression model using joblib.
- **scaler.pkl**: Saved StandardScaler used to preprocess the data.
- **styles.css**: CSS file with application design implementation.
- **index.html**: Application's first page HTML implementation which is a form to collect User Input.
- **result.html**: Application's second page HTML implementation that reveals Diabetes Prediction.
- **analysis.py**: Preliminary analysis on the dataset, involving violin, box, hist, and strip plots.



