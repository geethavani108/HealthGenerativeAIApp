# Summary statistics
print(df.describe())

# Count of unique values in categorical columns
print(df['Sex'].value_counts())
print(df['ChestPainType'].value_counts())
print(df['RestingECG'].value_counts())
print(df['ExerciseAngina'].value_counts())
print(df['ST_Slope'].value_counts())

#Distribution Analysis
# Histograms for numerical features
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

#Correlation Analysis
# Boxplots for numerical features
for column in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='HeartDisease', y=column, data=df)
    plt.title(f'{column} vs HeartDisease')
    plt.show()
# Correlation matrix
corr_matrix = df.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#Categorical Analysis
# Count plots for categorical features
sns.countplot(x='Sex', hue='HeartDisease', data=df)
plt.title('Sex vs HeartDisease')
plt.show()

sns.countplot(x='ChestPainType', hue='HeartDisease', data=df)
plt.title('ChestPainType vs HeartDisease')
plt.show()

sns.countplot(x='RestingECG', hue='HeartDisease', data=df)
plt.title('RestingECG vs HeartDisease')
plt.show()

sns.countplot(x='ExerciseAngina', hue='HeartDisease', data=df)
plt.title('ExerciseAngina vs HeartDisease')
plt.show()

sns.countplot(x='ST_Slope', hue='HeartDisease', data=df)
plt.title('ST_Slope vs HeartDisease')
plt.show()



#Histograms
# Histograms for numerical features
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Boxplots for numerical features vs target variable
for column in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='HeartDisease', y=column, data=df)
    plt.title(f'{column} vs HeartDisease')
    plt.show()

#Heat maps
# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Ma# Count plots for categorical features vs target variable
sns.countplot(x='Sex', hue='HeartDisease', data=df)
plt.title('Sex vs HeartDisease')
plt.show()

#count plots are useful for visualizing the frequency distribution of categorical features.
sns.countplot(x='ChestPainType', hue='HeartDisease', data=df)
plt.title('ChestPainType vs HeartDisease')
plt.show()

sns.countplot(x='RestingECG', hue='HeartDisease', data=df)
plt.title('RestingECG vs HeartDisease')
plt.show()

sns.countplot(x='ExerciseAngina', hue='HeartDisease', data=df)
plt.title('ExerciseAngina vs HeartDisease')
plt.show()

sns.countplot(x='ST_Slope', hue='HeartDisease', data=df)
plt.title('ST_Slope vs HeartDisease')
plt.show()
trix')
plt.show()

#Summary of Findings
--------------------
#Descriptive Statistics: Provides insights into the central tendency, dispersion, and shape of the dataset.
#Distribution Analysis: Helps identify the spread and potential outliers in numerical features.
#Correlation Analysis: Reveals the relationships between different features.
#Categorical Analysis: Examines the relationship between categorical features and the target variable.
