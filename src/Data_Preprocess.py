import pandas as pd

# Load the dataset
df = pd.read_csv("../tmp/heart_without_outliers.csv")

# Display the first few rows of the dataset
print(df.head())
# Check for missing values
print(df.isnull().sum())
"""
# Fill missing numerical values with the mean
#df.fillna(df.mean(), inplace=True)

# Fill missing categorical values with the mode
for column in df.select_dtypes(include=["object"]).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Verify that there are no missing values left
print(df.isnull().sum())
"""
#encding categorical columns
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
categorical_columns = df.select_dtypes(include=["object"]).columns
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Display the first few rows of the encoded dataset
print(df.head())


# Display the first few rows of the scaled dataset
print(df.head())

#Feature EngineeringCreate new features or modify existing ones if needed:


# Display the first few rows of the dataset with the new feature
print(df.head())
df.to_csv("../tmp/heart_failure_prediction.csv", index=False)
