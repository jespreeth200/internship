from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('medical_insurance_model.pkl', 'rb'))

# Define home page
@app.route('/')
def home():
    return render_template('index.html')

# Define predict page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the customer attributes input from the user
    age = int(request.form['age'])
    diabetes = int(request.form['diabetes'])
    blood_pressure = int(request.form['blood_pressure'])
    transplants = int(request.form['transplants'])
    chronic_diseases = int(request.form['chronic_diseases'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    allergies = int(request.form['allergies'])
    cancer_history = int(request.form['cancer_history'])
    major_surgeries = int(request.form['major_surgeries'])

    # Create a numpy array for the input data
    input_data = np.array([[age, diabetes, blood_pressure, transplants, chronic_diseases, height, weight, allergies, cancer_history, major_surgeries]])

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)

    # Round off the predicted value to 2 decimal places
    prediction = round(prediction[0], 2)

    # Render the predicted value on a new page
    return render_template('index.html', prediction_text='Predicted Health Insurance Premium: $ {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
app.run()
