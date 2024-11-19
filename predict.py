import joblib
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify

from utils import prepare_features

# Load the saved pipeline and XGBoost model
pipeline = joblib.load('xgb_pipeline.pkl')

app = Flask('House Price Prediction')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the target value for the input data sent via a POST request.

    Expects:
    - JSON payload with feature values.

    Returns:
    - JSON response containing the predicted house price.
    """
    # Parse input data from the request
    input_data = request.get_json()

    if not input_data:
        return jsonify({'error': 'Invalid input, no data provided'}), 400

    try:
        # Preprocess input data
        input_data = prepare_features(input_data)
        
        # Perform prediction
        y_pred = pipeline.predict([input_data])
        price = np.expm1(float(y_pred[0]))

        # Prepare response
        result = {
            'predicted_value': np.round(price, 3)  # Extract the single predicted value
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=9696)
