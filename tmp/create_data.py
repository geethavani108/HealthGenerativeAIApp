import pandas as pd
import numpy as np

# Set the number of rows for the dataset
num_rows = 500

# Generate random data for each column
data = {
    "Age": np.random.randint(18, 90, size=num_rows),
    "Sex": np.random.choice(["M", "F"], size=num_rows),
    "ChestPainType": np.random.choice(["ATA", "NAP", "ASY", "TA"], size=num_rows),
    "RestingBP": np.random.randint(90, 200, size=num_rows),
    "Cholesterol": np.random.randint(100, 400, size=num_rows),
    "FastingBS": np.random.choice([0, 1], size=num_rows),
    "RestingECG": np.random.choice(["Normal", "ST", "LVH"], size=num_rows),
    "MaxHR": np.random.randint(60, 202, size=num_rows),
    "ExerciseAngina": np.random.choice(["Y", "N"], size=num_rows),
    "Oldpeak": np.random.uniform(0.0, 6.2, size=num_rows).round(1),
    "ST_Slope": np.random.choice(["Up", "Flat", "Down"], size=num_rows),
    "HeartDisease": np.random.choice([0, 1], size=num_rows)
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv("heart_failure_prediction_synthetic.csv", index=False)
