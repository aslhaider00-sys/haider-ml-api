from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load your model
model = load_model("Model/keras_model.h5")
labels = [line.strip() for line in open("Model/labels.txt", "r").readlines()]

@app.route('/predict/', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        index = np.argmax(prediction)
        gesture = labels[index]

        return jsonify({"gesture": gesture})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()