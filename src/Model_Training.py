import pickle
import  joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

def save_model(model, pickle_path, joblib_path):
    with open(pickle_path, 'wb') as file:
        pickle.dump(model, file)
        joblib.dump(model, joblib_path)
    print(f"Model saved as {pickle_path} and {joblib_path}")
    

# Load the dataset
df = pd.read_csv("../tmp/Heart_predict_clean.csv")
# Features and target
# Define features (X) and target variable (y) 
X = df.drop(columns=["HeartDisease"]) 
X.to_csv("../tmp/Heart_predict_Xtest.csv")
y = df["HeartDisease"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
pickle_file_path = 'hs_model.pkl'
save_model(model, 'hs_model.pkl', 'hs_model.joblib')
# Evaluate the model
y_pred = model.predict(X_test).round()
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
# Function to provide evidence-based recommendations
def provide_recommendations(patient_data):
    prediction = model.predict(patient_data).round()[0][0]
    if prediction == 1:
        return "Recommend further testing for heart disease."
    else:
        return "No immediate concern for heart disease."

# Example usage
#new_patient = [-1.3344393286432252,1,2,-0.0995213315645555,-0.5022087520249551,-0.4372657700276822,1,-0.1009634347862968,0,-0.8374576075963182,2]
#recommendation = provide_recommendations(new_patient)
#print(recommendation)

