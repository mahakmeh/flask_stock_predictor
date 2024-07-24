import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from azure.storage.blob import BlobServiceClient
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Azure Blob Storage connection
connect_str = os.getenv('AZURE_CONNECTION_STRING')
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = os.getenv('AZURE_CONTAINER_NAME')

# List of stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL']

# Function to download a blob from Azure Blob Storage
def download_blob(blob_service_client, container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    download_stream = blob_client.download_blob()
    return io.BytesIO(download_stream.readall())

# Load models and scalers from Azure Blob Storage
models = {}
scalers = {}

for ticker in tickers:
    model_data = download_blob(blob_service_client, container_name, f'{ticker}_lstm_model.h5')
    scaler_data = download_blob(blob_service_client, container_name, f'{ticker}_scaler.pkl')
    models[ticker] = tf.keras.models.load_model(model_data)
    scalers[ticker] = joblib.load(scaler_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data['ticker']
    features = np.array(data['features']).reshape(1, -1, 1)
    prediction = models[ticker].predict(features)
    return jsonify({'prediction': prediction[0][0]})

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
