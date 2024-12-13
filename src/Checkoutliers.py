import pandas as pd
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

import pandas as pd

# Load the dataset
df = pd.read_csv("heart.csv")

# Display the first few rows of the dataset
print(df.head())
# Check for missing values
print(df.isnull().sum())
# Fill missing numerical values with the mean
#df.fillna(df.mean(), inplace=True)


# Verify that there are no missing values left
print(df.isnull().sum())

# Load the dataset
#df = pd.read_csv("heart_failure_prediction.csv")
print("Descriptive Summary statistics")
print(df.describe())

# 
# Check numerical columns using box plot for finding outliers and correct them 
# Building a box-plot to look at emissions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.boxplot(x=df['Age'], ax=axes[0, 0])
axes[0, 0].set_title('Age')

sns.boxplot(x=df['RestingBP'], ax=axes[0, 1])
axes[0, 1].set_title('RestingBP')

sns.boxplot(x=df['Cholesterol'], ax=axes[1, 0])
axes[1, 0].set_title('Cholesterol')

sns.boxplot(x=df['MaxHR'], ax=axes[1, 1])
axes[1, 1].set_title('MaxHR')

plt.tight_layout()
plt.show()

# Remove rows where RestingBP or Cholesterol are 0
df = df[(df['RestingBP'] != 0) & (df['Cholesterol'] != 0)]

# there are many outliers in RestingBP and Cholestrol
# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from RestingBP and Cholesterol
df = remove_outliers_iqr(df, 'RestingBP')
df = remove_outliers_iqr(df, 'Cholesterol')

print(df)

#Distribution Analysis
# Histograms for numerical features
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()





fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.boxplot(x=df['Age'], ax=axes[0, 0])
axes[0, 0].set_title('Age')

sns.boxplot(x=df['RestingBP'], ax=axes[0, 1])
axes[0, 1].set_title('RestingBP')

sns.boxplot(x=df['Cholesterol'], ax=axes[1, 0])
axes[1, 0].set_title('Cholesterol')

sns.boxplot(x=df['MaxHR'], ax=axes[1, 1])
axes[1, 1].set_title('MaxHR')

plt.tight_layout()
plt.show()
df.to_csv("../:tmp/heart_without_outliers.csv", index=False)




