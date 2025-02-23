import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic transit data
data_size = 5000
data = pd.DataFrame({
    'Time': pd.date_range(start='2024-01-01', periods=data_size, freq='H'),
    'Traffic_Level': np.random.randint(1, 10, data_size),
    'Weather_Condition': np.random.choice(['Clear', 'Rainy', 'Snowy'], data_size),
    'Passenger_Count': np.random.randint(20, 300, data_size)  # Simulated demand
})

# Convert categorical data to numerical format
data = pd.get_dummies(data, columns=['Weather_Condition'])

# Save dataset to CSV file
data.to_csv("synthetic_transportation_data.csv", index=False)

print("🚀 Synthetic dataset generated successfully and saved as 'synthetic_transportation_data.csv'")
