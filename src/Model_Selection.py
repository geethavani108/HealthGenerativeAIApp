#Choose a machine learning model that is appropriate for your problem.
#Common models for binary classification problems like heart disease prediction include:

# Load the data
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("../tmp/heart_failure_prediction.csv")

# Encode categorical variables
categorical_columns = df.select_dtypes(include=["object"]).columns
for column in categorical_columns:
    df[column] = pd.Categorical(df[column]).codes

#X=df['Age', 'ChestPainType', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'Age_Cholesterol_Ratio']
# Define features and target variable
X = df.drop(columns=['RestingECG', 'FastingBS','Sex','ExerciseAngina','HeartDisease'])
y = df["HeartDisease"]
df.to_csv("../tmp/Heart_predict_clean.csv")
#Scale numerical features to a similar range:
from sklearn.preprocessing import StandardScaler

# Identify numerical features
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
numerical_columns = numerical_columns.drop('HeartDisease')
print(numerical_columns)
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 2: Cross-Validation Accuracy
#Perform cross-validation to estimate the performance of different models
#Logistic Regression
#Random Forest
#Gradient Boosting
#Support Vector Machines (SVM)
#K-Nearest Neighbors (KNN)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Evaluate models using cross-validation
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: Cross-Validation Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")


#Step 3: Train and Evaluate Each Model
#--------------------------------------
#Train each model on the training data and evaluate its performance on the test data.

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}:")
    print(f"  Test Accuracy: {accuracy:.2f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"  Classification Report:\n{classification_report(y_test, y_pred)}")
    print()
"""
Load the Data: The dataset is loaded, and categorical variables are encoded. The data is then split into training and testing sets.

Cross-Validation Accuracy:

Purpose: To estimate the model's performance on unseen data and compare different models.

How It Works: The cross_val_score function performs cross-validation by splitting the training data into several folds, training the model on some folds while validating it on the remaining fold. This process is repeated multiple times (5 times in this example), and the results are averaged to obtain the cross-validation accuracy.

Train and Evaluate Each Model:

Purpose: To train each model on the entire training dataset and evaluate its performance on the test dataset.

How It Works: Each model is trained using the fit method, predictions are made using the predict method, and the model's performance is evaluated using metrics like accuracy, confusion matrix, and classification report.
    Loading the Dataset: The dataset is loaded from a CSV file.

Encoding Categorical Variables: Categorical variables are encoded into numerical values.

Splitting the Data: The data is split into training and testing sets.

Initializing Models: Various classification models are initialized, including Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine, and K-Nearest Neighbors.

Cross-Validation: Each model is evaluated using cross-validation to compare their performance.

Training and Evaluation: Each model is trained on the training data and evaluated on the test data. The accuracy, confusion matrix, and classification report are printed for each model.*/
"""
