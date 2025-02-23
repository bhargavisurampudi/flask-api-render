from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open("demand_prediction_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Smart Transit API is Running! Use /predict to make requests."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        hour = int(data['hour'])
        traffic_level = int(data['traffic_level'])
        weather = data['weather']

        # Convert weather to numerical format
        weather_clear = 1 if weather == "Clear" else 0
        weather_rainy = 1 if weather == "Rainy" else 0
        weather_snowy = 1 if weather == "Snowy" else 0

        # Prepare input features
        features = np.array([[hour, traffic_level, weather_clear, weather_rainy, weather_snowy]])
        features_scaled = scaler.transform(features)

        # Predict demand
        prediction = model.predict(features_scaled)[0]

        return jsonify({"Predicted Passenger Demand": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
