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
MODEL_PATH = 'milk_quality_model.h5'

class MilkQualityTester:
    def __init__(self):
        self.model = self._build_model()
        self.threshold = 0.5  # Default threshold for binary classification
        
    def _build_model(self):
        """Build and compile the transfer learning model using ResNet50"""
        # Load pre-trained model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    
    def train(self, train_data_dir):
        """Train the model using data from the specified directory"""
        # Data augmentation for training
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
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training'
        )
        
        # Load validation data
        validation_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation'
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', 
                                    save_best_only=True, mode='max')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, 
                                  restore_best_weights=True)
        
        # Train model
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
        """Preprocess image for prediction"""
        if isinstance(image_path, str):
            # Load image from file
            img = cv2.imread(image_path)
        else:
            # If image is already loaded (numpy array)
            img = image_path
            
        # Convert BGR to RGB if necessary
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, IMG_SIZE)
        
        # Normalize
        img = img / 255.0
        
        # Expand dimensions for model
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def extract_roi(self, image):
        """Extract region of interest (well area) from the image"""
        # For MVP, we'll use a simple approach - assume the well is centered
        # In a real implementation, this would detect the actual well position
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Extract a region around the center (60% of image size)
        roi_size_x = int(w * 0.6)
        roi_size_y = int(h * 0.6)
        
        start_x = max(0, center_x - roi_size_x // 2)
        start_y = max(0, center_y - roi_size_y // 2)
        end_x = min(w, center_x + roi_size_x // 2)
        end_y = min(h, center_y + roi_size_y // 2)
        
        roi = image[start_y:end_y, start_x:end_x]
        return roi
    
    def color_analysis(self, image):
        """Extract color features from the image"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate mean values for each channel
        mean_h = np.mean(hsv[:,:,0])
        mean_s = np.mean(hsv[:,:,1])
        mean_v = np.mean(hsv[:,:,2])
        
        # Calculate standard deviation for each channel
        std_h = np.std(hsv[:,:,0])
        std_s = np.std(hsv[:,:,1])
        std_v = np.std(hsv[:,:,2])
        
        # For resazurin, blue → pink → colorless
        # Calculate blue and pink ratios in RGB
        blue_ratio = np.mean(image[:,:,2]) / (np.mean(image[:,:,0]) + np.mean(image[:,:,1]) + 1e-10)
        pink_ratio = np.mean(image[:,:,0]) / (np.mean(image[:,:,1]) + np.mean(image[:,:,2]) + 1e-10)
        
        features = {
            'mean_h': mean_h,
            'mean_s': mean_s,
            'mean_v': mean_v,
            'std_h': std_h,
            'std_s': std_s,
            'std_v': std_v,
            'blue_ratio': blue_ratio,
            'pink_ratio': pink_ratio
        }
        
        return features
    
    def predict(self, image_path):
        """Predict milk quality from image"""
        if isinstance(image_path, str):
            # Load image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path
            
        # Extract ROI
        roi = self.extract_roi(img)
        
        # Extract color features (for reporting)
        color_features = self.color_analysis(roi)
        
        # Preprocess for model
        processed_img = self.preprocess_image(roi)
        
        # Make prediction
        prediction = self.model.predict(processed_img)[0][0]
        
        # Determine class
        quality_class = "Acceptable" if prediction < self.threshold else "Unacceptable"
        
        result = {
            'prediction': float(prediction),
            'quality_class': quality_class,
            'color_features': color_features,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    
    def load_model(self, model_path=MODEL_PATH):
        """Load a saved model"""
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"No model found at {model_path}")

# Flask Web Application
app = Flask(__name__)
milk_tester = MilkQualityTester()

# Try to load the model if it exists
try:
    milk_tester.load_model()
except:
    print("No pre-trained model found. You'll need to train before prediction.")

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
    
    # Read image
    img_stream = file.read()
    nparr = np.frombuffer(img_stream, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get prediction
    result = milk_tester.predict(img)
    
    # Convert ROI image to base64 for display
    roi = milk_tester.extract_roi(img)
    roi_pil = Image.fromarray(roi)
    buf = io.BytesIO()
    roi_pil.save(buf, format='JPEG')
    roi_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return render_template('result.html', 
                          result=result,
                          roi_image=roi_b64)

@app.route('/download_report/<timestamp>')
def download_report(timestamp):
    # Generate report as a CSV
    # In a real application, this would create a detailed PDF report
    report_str = f"Milk Quality Analysis Report\nTimestamp: {timestamp}\n"
    # Add more details here
    
    # Create a downloadable file
    buffer = io.BytesIO()
    buffer.write(report_str.encode())
    buffer.seek(0)
    
    return send_file(buffer, 
                    attachment_filename='milk_quality_report.txt',
                    as_attachment=True)

# HTML templates (simplified for MVP)
@app.route('/templates/index.html')
def get_index_template():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Milk Quality Testing</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; }
            .btn { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Rapid Milk Quality Testing</h1>
                <p>Upload an image of your milk sample with resazurin indicator</p>
            </div>
            
            <div class="upload-form">
                <form action="/analyze" method="POST" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*" required>
                    <br><br>
                    <button type="submit" class="btn">Analyze Sample</button>
                </form>
            </div>
            
            {% if error %}
            <div class="error">
                <p>{{ error }}</p>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """

@app.route('/templates/result.html')
def get_result_template():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .result { padding: 20px; border: 1px solid #ccc; margin-bottom: 20px; }
            .acceptable { background-color: #DFF2BF; }
            .unacceptable { background-color: #FFBABA; }
            .img-container { text-align: center; margin: 20px 0; }
            .btn { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Milk Quality Analysis Results</h1>
            </div>
            
            <div class="img-container">
                <h3>Analyzed Sample</h3>
                <img src="data:image/jpeg;base64,{{ roi_image }}" alt="Sample Image" style="max-width: 300px;">
            </div>
            
            <div class="result {{ 'acceptable' if result.quality_class == 'Acceptable' else 'unacceptable' }}">
                <h2>Quality Classification: {{ result.quality_class }}</h2>
                <p>Confidence: {{ (result.prediction if result.quality_class == 'Unacceptable' else 1 - result.prediction) * 100 | round(2) }}%</p>
                <p>Analysis Time: {{ result.timestamp }}</p>
                
                <h3>Color Analysis:</h3>
                <ul>
                    <li>Blue-to-other ratio: {{ result.color_features.blue_ratio | round(3) }}</li>
                    <li>Pink-to-other ratio: {{ result.color_features.pink_ratio | round(3) }}</li>
                    <li>Color saturation: {{ result.color_features.mean_s | round(1) }}</li>
                </ul>
                
                <a href="/download_report/{{ result.timestamp }}" class="btn">Download Full Report</a>
            </div>
            
            <div>
                <a href="/" style="text-decoration: none;">
                    <button class="btn">Test Another Sample</button>
                </a>
            </div>
        </div>
    </body>
    </html>
    """

# Data collection helper functions
def collect_training_data(save_dir='training_data'):
    """
    Function to help collect and organize training data
    This would be used during the development phase
    """
    os.makedirs(os.path.join(save_dir, 'acceptable'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'unacceptable'), exist_ok=True)
    
    print(f"Created directories for training data at {save_dir}")
    print("Place your images in the following directories:")
    print(f" - {os.path.join(save_dir, 'acceptable')} for good quality milk samples")
    print(f" - {os.path.join(save_dir, 'unacceptable')} for contaminated milk samples")

# Run the application
if __name__ == '__main__':
    # Create data collection directories
    collect_training_data()
    
    # Run Flask app
    app.run(debug=True)
