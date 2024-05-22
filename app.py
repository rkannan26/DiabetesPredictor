from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
def predict_diabetes(input_data):
    """
    Function to predict diabetes using the trained model.

    Parameters:
    input_data (list or array): A list or array of input features.

    Returns:
    int: The predicted class (0 for non-diabetic, 1 for diabetic).
    """
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form['pregnancies']),
                  float(request.form['glucose']),
                  float(request.form['bloodpressure']),
                  float(request.form['skinthickness']),
                  float(request.form['insulin']),
                  float(request.form['bmi']),
                  float(request.form['dpf']),
                  float(request.form['age'])]
    prediction = predict_diabetes(input_data)
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)