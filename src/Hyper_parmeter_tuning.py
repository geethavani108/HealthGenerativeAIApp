/* Hyperparameter tuning is the process of optimizing the hyperparameters of a machine learning model to improve its performance. 
A common approach to hyperparameter tuning is using techniques such as Grid Search or Random Search with cross-validation.
pip install scikit-learn */


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("heart_failure_prediction_synthetic.csv")

# Encode categorical variables
categorical_columns = df.select_dtypes(include=["object"]).columns
for column in categorical_columns:
    df[column] = pd.Categorical(df[column]).codes

# Define features and target variable
X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the hyperparameters grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the best model with the best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

/*Explanation
--------------
Load the Dataset: Load the dataset and encode categorical variables.
Split the Data: Split the data into training and testing sets.
Define the Model: Define the RandomForestClassifier model.
Define the Hyperparameters Grid: Specify the hyperparameters and their values to search.
Initialize GridSearchCV: Use GridSearchCV to perform an exhaustive search over the specified hyperparameters.
Fit GridSearchCV: Fit the GridSearchCV on the training data to find the best hyperparameters.
Best Hyperparameters: Retrieve and print the best hyperparameters found by the grid search.
Train the Best Model: Train the model with the best hyperparameters on the training data.
Evaluate the Model: Make predictions on the test data and evaluate the model’s performance.
Hyperparameter tuning helps in finding the optimal set of hyperparameters that improve the model’s performance */
