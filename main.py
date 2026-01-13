from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from model import LogisticRegressionScratch

app = Flask(__name__)

# Load Model & Scaler
try:
    with open("cardio_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler_min, scaler_max = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler_min, scaler_max = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        features = [
            float(data['age']),
            int(data['gender']),
            float(data['height']),
            float(data['weight']),
            float(data['ap_hi']),
            float(data['ap_lo']),
            int(data['cholesterol']),
            int(data['gluc']),
            int(data['smoke']),
            int(data['alco']),
            int(data['active'])
        ]
        
        # Normalize features
        features_arr = np.array(features).reshape(1, -1)
        if scaler_min is not None and scaler_max is not None:
             features_arr = (features_arr - scaler_min) / (scaler_max - scaler_min)

        # Predict
        prob = model.predict_proba(features_arr)[0]
        prediction = 1 if prob >= 0.5 else 0
        
        return jsonify({
            'probability': float(prob),
            'prediction': prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
