import tensorflow as tf
import os
import warnings
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Limit TensorFlow memory usage
physical_devices = tf.config.list_physical_devices('CPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow logs
warnings.filterwarnings("ignore", category=UserWarning)  # Hide model warnings

app = Flask(__name__)

# Load the model without recompiling it
model = load_model('accident_detection_model.h5', compile=False)  # Avoids warning

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
            img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction = model.predict(img_array)[0][0]
            class_label = "Accident" if prediction < 0.99 else "Not Accident"
            probability = round((1 - prediction) * 100, 2) if prediction < 0.5 else round(prediction * 100, 2)

            return jsonify({"result": class_label, "confidence": probability}), 200

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type."}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Change port if needed
    app.run(host="0.0.0.0", port=port, debug=False)
