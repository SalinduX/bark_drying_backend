from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
import time
import tensorflow as tf

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow requests from React Native mobile app

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'model/model.tflite')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
PORT = int(os.getenv('PORT', 5000))

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input size (check Edge Impulse impulse settings)
INPUT_SIZE = (96, 96)  # Change to (128, 128) if your model uses 128x128

# Classes (your project labels)
CLASSES = ['dry', 'mid_level_dry', 'wet']

print("TensorFlow Lite model loaded successfully!")
print("Input size:", INPUT_SIZE)
print("Classes:", CLASSES)

@app.route('/')
def home():
    return jsonify({"message": "Bark Drying Backend - Ready!"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files['image']
    filename = f"bark_{int(time.time())}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize(INPUT_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_idx = np.argmax(output)
        predicted_class = CLASSES[predicted_idx]
        confidence = float(output[predicted_idx])
        
        is_optimal = "yes" if predicted_class.lower() == "dry" else "no"
        
        response = {
            "condition": predicted_class,
            "confidence": round(confidence, 4),
            "is_optimal": is_optimal,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"Prediction: {predicted_class} ({confidence:.2%})")
        return jsonify(response)
        
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)