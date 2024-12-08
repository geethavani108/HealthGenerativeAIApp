import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Features and target
X = data[['Age', 'Blood_Pressure', 'Cholesterol']]
y = data['Heart_Disease']

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

# Evaluate the model
y_pred = model.predict(X_test).round()
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
# Function to provide evidence-based recommendations
def provide_recommendations(patient_data):
    patient_data = scaler.transform([patient_data])
    prediction = model.predict(patient_data).round()[0][0]
    if prediction == 1:
        return "Recommend further testing for heart disease."
    else:
        return "No immediate concern for heart disease."

# Example usage
new_patient = [30, 130, 215]
recommendation = provide_recommendations(new_patient)
print(recommendation)

