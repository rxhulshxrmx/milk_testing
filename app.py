# Import necessary libraries
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import datetime
from flask import Flask, request, render_template, send_file
import io
import base64
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import json
import requests

# Config parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2
MODEL_PATH = 'milk_quality_model.keras'

# Download model if not present
def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Model not found. Downloading from Google Drive...")
        file_id = "1n9MjH-Opw6F1HKMPAeEiO5tClBWV_Kjg"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("[INFO] Model download complete ✅")
        else:
            print("[ERROR] Failed to download model. Status code:", response.status_code)

download_model_if_missing()

class MilkQualityTester:
    def __init__(self):
        self.model = self._build_model()
        self.threshold = 0.5

    def _build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_image(self, image_path):
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def extract_roi(self, image):
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size_x = int(w * 0.6)
        roi_size_y = int(h * 0.6)
        start_x = max(0, center_x - roi_size_x // 2)
        start_y = max(0, center_y - roi_size_y // 2)
        end_x = min(w, center_x + roi_size_x // 2)
        end_y = min(h, center_y + roi_size_y // 2)
        roi = image[start_y:end_y, start_x:end_x]
        return roi

    def color_analysis(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mean_h = np.mean(hsv[:,:,0])
        mean_s = np.mean(hsv[:,:,1])
        mean_v = np.mean(hsv[:,:,2])
        std_h = np.std(hsv[:,:,0])
        std_s = np.std(hsv[:,:,1])
        std_v = np.std(hsv[:,:,2])
        blue_ratio = np.mean(image[:,:,2]) / (np.mean(image[:,:,0]) + np.mean(image[:,:,1]) + 1e-10)
        pink_ratio = np.mean(image[:,:,0]) / (np.mean(image[:,:,1]) + np.mean(image[:,:,2]) + 1e-10)
        return {
            'mean_h': mean_h,
            'mean_s': mean_s,
            'mean_v': mean_v,
            'std_h': std_h,
            'std_s': std_s,
            'std_v': std_v,
            'blue_ratio': blue_ratio,
            'pink_ratio': pink_ratio
        }

    def predict(self, image_path):
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path
        roi = self.extract_roi(img)
        color_features = self.color_analysis(roi)
        processed_img = self.preprocess_image(roi)
        prediction = self.model.predict(processed_img)[0][0]
        quality_class = "Acceptable" if prediction < self.threshold else "Unacceptable"
        return {
            'prediction': float(prediction),
            'quality_class': quality_class,
            'color_features': color_features,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def load_model(self, model_path=MODEL_PATH):
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"No model found at {model_path}")

app = Flask(__name__)
milk_tester = MilkQualityTester()

try:
    milk_tester.load_model(MODEL_PATH)
except:
    print("No pre-trained model found.")

@app.route('/')
def home():
    return """
    <html>
    <head><title>Milk Quality Testing</title></head>
    <body style='font-family: Arial; padding: 40px;'>
        <h2>Upload a Milk Sample Image</h2>
        <form method="POST" action="/analyze" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br><br>
            <button type="submit">Analyze</button>
        </form>
    </body>
    </html>
    """

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    img_stream = file.read()
    nparr = np.frombuffer(img_stream, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = milk_tester.predict(img)
    roi = milk_tester.extract_roi(img)
    roi_pil = Image.fromarray(roi)
    buf = io.BytesIO()
    roi_pil.save(buf, format='JPEG')
    roi_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return f"""
    <html>
    <head>
        <title>Milk Quality Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f6f8; color: #333; padding: 30px; }}
            .container {{ max-width: 700px; margin: auto; background: white; padding: 25px 40px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; }}
            img {{ max-width: 300px; border-radius: 12px; }}
            .result {{ margin-top: 20px; padding: 20px; border-radius: 12px; background-color: {'#d4edda' if result['quality_class'] == 'Acceptable' else '#f8d7da'}; color: {'#155724' if result['quality_class'] == 'Acceptable' else '#721c24'}; }}
            .btn {{ margin-top: 20px; padding: 10px 20px; background-color: #007BFF; color: white; border: none; border-radius: 8px; text-decoration: none; font-size: 16px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Milk Quality Analysis Results</h1>
            <img src='data:image/jpeg;base64,{roi_b64}' alt='Analyzed Sample'><br>

            <div class="result">
                <h2>{result['quality_class']}</h2>
                <p><strong>Confidence:</strong> {round((result['prediction'] if result['quality_class']=='Unacceptable' else 1 - result['prediction']) * 100, 2)}%</p>
                <p><strong>Analysis Time:</strong> {result['timestamp']}</p>
            </div>

            <h3>Color Features</h3>
            <ul style="list-style-type:none; padding: 0;">
                <li><strong>Blue Ratio:</strong> {round(result['color_features']['blue_ratio'], 3)}</li>
                <li><strong>Pink Ratio:</strong> {round(result['color_features']['pink_ratio'], 3)}</li>
                <li><strong>Saturation:</strong> {round(result['color_features']['mean_s'], 1)}</li>
            </ul>

            <a href="/" class="btn">Test Another Sample</a>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)