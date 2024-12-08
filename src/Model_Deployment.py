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
    age = data['age']
    blood_pressure = data['blood_pressure']
    cholesterol = data['cholesterol']
    
    # Prepare the input data for prediction
    input_data = np.array([[age, blood_pressure, cholesterol]])
    input_data = scaler.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data).round()[0]
    
    # Map the prediction to the output
    result = 'Heart Disease' if prediction == 1 else 'No Heart Disease'
    
    return jsonify({
        'prediction': result
    })

if __name__ == '__main__':
    app.run(debug=True)
