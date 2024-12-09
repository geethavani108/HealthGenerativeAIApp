import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

# Define models to validate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC()
}

# Perform K-Fold Cross-Validation
k = 5  # Number of folds
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=k)
    print(f"{name}: Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Train and evaluate the best model on the test set
best_model = RandomForestClassifier(random_state=42)  # Assuming Random Forest performed best
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model (Random Forest): Test Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


/* Explanation:
Cross-Validation: cross_val_score is used to perform K-Fold Cross-Validation and obtain the average accuracy and standard deviation.

Best Model Selection: After cross-validation, the best-performing model (Random Forest in this example) is selected and trained on the entire training set.

Final Evaluation: The best model is then evaluated on the test set to get an unbiased estimate of its performance.

Model validation ensures that the selected model generalizes well to new data and helps in choosing the best model for your problem.
                                                                                                                  */
