import numpy as np
import pandas as pd

# Generate synthetic data
np.random.seed(42)
num_samples = 100

# Create features
ages = np.random.randint(20, 80, num_samples)
blood_pressures = np.random.randint(110, 180, num_samples)
cholesterols = np.random.randint(150, 300, num_samples)
heart_disease = np.random.randint(0, 2, num_samples)  # 0 = No, 1 = Yes

# Combine into a DataFrame
data = pd.DataFrame({
    'Age': ages,
    'Blood_Pressure': blood_pressures,
    'Cholesterol': cholesterols,
    'Heart_Disease': heart_disease
})

# Display the first few rows
print(data.head())
