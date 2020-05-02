# Following tutorial from:
# https://www.linode.com/docs/applications/big-data/how-to-move-machine-learning-model-to-production/

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from keras.models import load_model

app = Flask(heart)

model = load_model(/flask_api/models/my_model.hdf5)

@app.route('/predict', methods=["POST"])
def predict_disease():
    # Preprocess input data so that it matches the training input
        # Create dummies for categorical variables in order to quantify them.
    dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        # Scale variables so they all have the same range.
    standardScaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

    # Use the loaded model to generate a prediction
    pred = model.predict(dataset)

    # Prepare and send the response
    return jsonify(pred)

if __heart__ == "__main__":
    app.run()
