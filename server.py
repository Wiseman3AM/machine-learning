from flask import Flask, request, jsonify 
import numpy as np
import pandas as pd
import joblib
from flask_cors import CORS  # Import CORS




app = Flask(__name__)
CORS(app)

# File paths
model_filepath = 'best_ml.pkl'
scaler_filepath = 'scaler.pkl'


try:
    saved_model = joblib.load(model_filepath)
    scaler = joblib.load(scaler_filepath)
  

except Exception as e:
    print("Error loading model or scaler:", e)
    saved_model = None
    scaler = None


@app.route("/api/ml/predict", methods=['POST'])
def predict():
    if saved_model is None or scaler is None:
        return jsonify({"error": "Model or scaler is not loaded correctly"}), 500

    # Get the data sent from the front end (as JSON)
    data = request.get_json()

     
     # Prepare features as a DataFrame
    features = pd.DataFrame([{
        'HUMIDITY': data.get('HUMIDITY', 0),
        'RAINFALL': data.get('RAINFALL', 0),
        'TEMPERATURE': data.get('TEMPERATURE', 0),
        'TREND': data.get('TREND', 0),
        'REGION_Angola': data.get('REGION_Angola', 0),
        'REGION_Botswana': data.get('REGION_Botswana', 0),
        'REGION_Eswatini': data.get('REGION_Eswatini', 0),
        'REGION_Lesotho': data.get('REGION_Lesotho', 0),
        'REGION_Malawi': data.get('REGION_Malawi', 0),
        'REGION_Mozambique': data.get('REGION_Mozambique', 0),
        'REGION_South Africa': data.get('REGION_South Africa', 0),
        'REGION_United Republic of Tanzania': data.get('REGION_United Republic of Tanzania', 0),
        'REGION_Zambia': data.get('REGION_Zambia', 0),
        'REGION_Zimbabwe': data.get('REGION_Zimbabwe', 0),
        'CROP_Maize (corn)': data.get('CROP_Maize (corn)', 0),
        'CROP_Rice': data.get('CROP_Rice', 0),
        'CROP_Wheat': data.get('CROP_Wheat', 0),
        'HUMIDITY_TEMPERATURE': data.get('HUMIDITY_TEMPERATURE', 0)
    }])

    try:
        # Scale the features
        features_scaled = scaler.transform(features)

        # Make the prediction
        y_prediction = saved_model.predict(features_scaled)

        # Inverse the log1p transformation to get the original yield value
        original_yield = np.expm1(y_prediction)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Return the original yield as a JSON response
    return jsonify({'predicted_yield': float(original_yield[0])})




@app.route("/api/ml/test", methods=['GET'])
def test():

    # saved_model.predict()
    return {
        "status": 'running'
    }

if __name__ == "__main__":
    app.run(debug=True)
