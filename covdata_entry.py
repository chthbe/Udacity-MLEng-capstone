import json
import numpy as np
import pandas as pd
import os
import pickle
import joblib

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'Hyperdrive_capstone.joblib')
    model = joblib.load(model_path)

def run(raw_data):
    data = pd.DataFrame(json.loads(raw_data)['data'], index=range(0,len(json.loads(raw_data)['data'])))
    # Make prediction.
    y_hat = model.predict(data)
    # You can return any data type as long as it's JSON-serializable.
    return y_hat.tolist()