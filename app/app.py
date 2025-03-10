from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from flask_cors import CORS
import datetime

app = Flask(__name__)

CORS(app)

# Directory For Saving Trained Models
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check to verify service is running"""
    return jsonify({"status": "healthy", 'service': 'cashly-ai-service'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)