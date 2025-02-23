import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("synthetic_transportation_data.csv")

# Extract hour from timestamp
data['Hour'] = pd.to_datetime(data['Time']).dt.hour
data.drop(columns=['Time'], inplace=True)

# Define features and target
X = data.drop(columns=['Passenger_Count'])
y = data['Passenger_Count']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Machine Learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"🚀 Model Performance:\nMean Absolute Error: {mae}\nR² Score: {r2}")

# Save the trained model & scaler
import pickle
pickle.dump(model, open("demand_prediction_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Visualize Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", color='blue')
plt.plot(y_pred, label="Predicted", color='red', linestyle='dashed')
plt.xlabel("Sample Index")
plt.ylabel("Passenger Demand")
plt.title("Actual vs Predicted Transit Demand")
plt.legend()
plt.show()
