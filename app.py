import os
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Force TensorFlow to use CPU only (Disables GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Load the trained model (without compiling to avoid warnings)
model = load_model('accident_detection_model.h5', compile=False)

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    if file and allowed_file(file.filename):
        try:
            # Read the image from memory
            img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values

            # Make prediction
            prediction = model.predict(img_array)[0][0]
            probability = round((1 - prediction) * 100, 2) if prediction < 0.5 else round(prediction * 100, 2)

            # If confidence is below 80%, classify as "Not Accident"
            class_label = "Accident" if probability >= 80 else "Not Accident"

            return jsonify({
                "result": class_label,
                "confidence": probability
            }), 200

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type."}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Render sets PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=False)
