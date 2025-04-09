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

# Config parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2  # Acceptable vs Unacceptable
MODEL_PATH = 'milk_quality_model.keras'

class MilkQualityTester:
    def __init__(self):
        self.model = self._build_model()
        self.threshold = 0.5  # Default threshold for binary classification

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
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        return model

    def train(self, train_data_dir):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation'
        )

        checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[checkpoint, early_stop]
        )
        return history

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

# Flask App
app = Flask(__name__)
milk_tester = MilkQualityTester()

try:
    milk_tester.load_model('milk_quality_model.keras')
except:
    print("No pre-trained model found.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
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
    return render_template('result.html', result=result, roi_image=roi_b64)

@app.route('/download_report/<timestamp>')
def download_report(timestamp):
    report_str = f"Milk Quality Analysis Report\nTimestamp: {timestamp}\n"
    buffer = io.BytesIO()
    buffer.write(report_str.encode())
    buffer.seek(0)
    return send_file(buffer, download_name='milk_quality_report.txt', as_attachment=True)

@app.route('/templates/index.html')
def get_index_template():
    return """<html>... (same content as before) ...</html>"""

@app.route('/templates/result.html')
def get_result_template():
    return """<html>... (same content as before) ...</html>"""

if __name__ == '__main__':
    milk_tester.train('training_data')
    app.run(debug=True)