import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("../tmp/heart_failure_prediction.csv")
# Scale numerical features

#Scale numerical features to a similar range:
from sklearn.preprocessing import StandardScaler

# Identify numerical features
numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns
numerical_columns = numerical_columns.drop('HeartDisease')
print(numerical_columns)
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Prepare features (X) and target (y)
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']
#  Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate the correlation matrix
corr_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#Recursive Feature Elimination (RFE)
#--------------------------------------
#Use RFE with a model to recursively remove the least important features.

# Identify highly correlated features with the target variable
correlation_threshold = 0.1  # You can adjust this threshold
relevant_features = corr_matrix.index[abs(corr_matrix["HeartDisease"]) > correlation_threshold].tolist()
print("Method :1 Relevant features based on correlation with HeartDisease:", relevant_features)

#2. Recursive Feature Elimination (RFE)
#----------------------------------
#Use RFE with a model to recursively remove the least important features.
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# Define the model
model = RandomForestClassifier(random_state=42)

# Define RFE( Recursive Feature Elimination)
rfe = RFE(estimator=model, n_features_to_select=7)  
# Adjust the number of features to select
# Fit RFE
rfe.fit(X_train, y_train)

# Get selected features
selected_features = X_train.columns[rfe.support_].tolist()
# Get the ranking of features 
ranking = rfe.ranking_
#  # Create a DataFrame to display features and their rankings
feature_importance = pd.DataFrame({ 'Feature': X_train.columns, 'Ranking': ranking })
#  # Sort features by their importance 
sorted_features = feature_importance.sort_values(by='Ranking') 
#print(sorted_features)
print("Method:2 Selected features:", selected_features)
print("Method:2 Sorted features:", sorted_features)

#3. Feature Importance with RandomForest
#---------------------------------
#Use a RandomForest classifier to compute feature importance scores.
from sklearn.ensemble import RandomForestClassifier

# Define the model
model = RandomForestClassifier(random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("Method :3 Feature importances:")
print(feature_importances)



#Summary of Selected Features
#Use any of these methods to identify and select the most relevant features for your model. Here’s a summary of the process:
#Correlation Matrix: Identify features correlated with the target variable.
#RFE: Use Recursive Feature Elimination with a model to select features.
#Feature Importance: Use feature importance scores from a RandomForest model.
