# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# model = load_model('accident_detection_model.h5')
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded."}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "Empty filename."}), 400
#     if file and allowed_file(file.filename):
#         filename = file.filename
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         try:
#             img = image.load_img(filepath, target_size=(224, 224))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array /= 255.0
#             prediction = model.predict(img_array)[0][0]
#             class_label = "Accident" if prediction < 0.99 else "Not Accident"
#             probability = round((1 - prediction) * 100, 2) if prediction < 0.5 else round(prediction * 100, 2)
#             return jsonify({
#                 "result": class_label
#             }), 200
#         except Exception as e:
#             return jsonify({"error": f"Error processing image: {str(e)}"}), 500
#     return jsonify({"error": "Invalid file type."}), 400
# if __name__ == '__main__':
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     app.run(debug=True)
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)

# Load the trained model
model = load_model('accident_detection_model.h5')

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
            # Read the image without saving
            img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values

            # Make prediction
            prediction = model.predict(img_array)[0][0]
            class_label = "Accident" if prediction < 0.99 else "Not Accident"
            probability = round((1 - prediction) * 100, 2) if prediction < 0.5 else round(prediction * 100, 2)

            return jsonify({
                "result": class_label,
                "confidence": probability
            }), 200

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type."}), 400

if __name__ == '__main__':
    app.run(debug=True)
