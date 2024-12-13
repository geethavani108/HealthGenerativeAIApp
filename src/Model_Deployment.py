from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    
    # Prepare the input data for prediction
    input_data = np.array([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,MaxHR,ST_Slope]])
    input_data = scaler.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data).round()[0]
    
    # Map the prediction to the output
    result = 'Heart Disease' if prediction == 1 else 'No Heart Disease'
    
    return jsonify({
        'prediction': result
    })


#Function to provide evidence-based recommendations
def provide_recommendations(patient_data):
    prediction = model.predict(patient_data).round()[0][0]
    if prediction == 1:
        return "Recommend further testing for heart disease."
    else:
        return "No immediate concern for heart disease."

if __name__ == '__main__':
    app.run(debug=True)
