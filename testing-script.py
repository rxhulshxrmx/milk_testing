# Import necessary libraries
import requests
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import json
import sys
import time

# Import our milk quality tester module
from milk_quality_testing_mvp import MilkQualityTester

def simulate_tests():
    """Simulate milk testing with synthetic data for demo purposes"""
    print("==== Rapid Milk Quality Testing Demo ====")
    print("Initializing tester...")
    
    # Initialize our tester
    tester = MilkQualityTester()
    
    # Generate synthetic test images - in real MVP these would come from camera
    # Blue (good quality)
    blue_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    blue_img[:, :, 0] = 50  # Low red
    blue_img[:, :, 1] = 50  # Low green
    # Add some noise for realism
    blue_img = blue_img + np.random.normal(0, 10, blue_img.shape).astype(np.uint8)
    blue_img = np.clip(blue_img, 0, 255)
    
    # Pink-ish (borderline quality)
    pink_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    pink_img[:, :, 1] = 150  # Medium green
    pink_img[:, :, 2] = 150  # Medium blue
    pink_img = pink_img + np.random.normal(0, 10, pink_img.shape).astype(np.uint8)
    pink_img = np.clip(pink_img, 0, 255)
    
    # Colorless/yellowish (bad quality)
    yellow_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    yellow_img[:, :, 2] = 100  # Low blue
    yellow_img = yellow_img + np.random.normal(0, 10, yellow_img.shape).astype(np.uint8)
    yellow_img = np.clip(yellow_img, 0, 255)
    
    # Save images for demo
    if not os.path.exists('demo_samples'):
        os.makedirs('demo_samples')
    
    cv2.imwrite('demo_samples/good_sample.jpg', cv2.cvtColor(blue_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('demo_samples/borderline_sample.jpg', cv2.cvtColor(pink_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('demo_samples/bad_sample.jpg', cv2.cvtColor(yellow_img, cv2.COLOR_RGB2BGR))
    
    print("Generated test samples in demo_samples directory")
    
    # Normally we'd load a trained model here, but for demo, we'll use feature extraction
    # and simple rule-based classification instead of ML prediction
    
    def analyze_sample(img, sample_name):
        """Demo analysis of a sample"""
        print(f"\n=== Analyzing {sample_name} ===")
        
        # Extract ROI
        roi = tester.extract_roi(img)
        
        # Get color features
        features = tester.color_analysis(roi)
        
        # For demo, use simple rule-based classification (in real MVP we'd use the trained model)
        blue_to_pink = features['blue_ratio'] / (features['pink_ratio'] + 0.001)
        
        if blue_to_pink > 1.5:
            quality = "Acceptable"
            confidence = min(blue_to_pink / 3, 0.99)
        elif blue_to_pink > 0.8:
            quality = "Borderline"
            confidence = 0.5 + (blue_to_pink - 0.8) / 1.4
        else:
            quality = "Unacceptable"
            confidence = max(1 - blue_to_pink, 0.7)
        
        # Display results
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(roi)
        plt.title(f"Sample: {sample_name}")
        
        plt.subplot(1, 2, 2)
        labels = list(features.keys())
        values = list(features.values())
        y_pos = np.arange(len(labels))
        
        plt.barh(y_pos, values)
        plt.yticks(y_pos, labels)
        plt.title("Color Features")
        plt.tight_layout()
        
        print(f"Quality Classification: {quality}")
        print(f"Confidence: {confidence*100:.1f}%")
        print("Color Features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")
        
        # In a real app, we'd save this data and generate reports
        
        return quality, confidence, features
    
    # Demo all samples
    analyze_sample(blue_img, "Good Quality Sample")
    analyze_sample(pink_img, "Borderline Sample")
    analyze_sample(yellow_img, "Poor Quality Sample")
    
    print("\n=== Demo Complete ===")
    print("In a real deployment:")
    print("1. The web app would be running for users to upload images")
    print("2. The ML model would be trained on real milk samples")
    print("3. Results would be saved in a database and downloadable as reports")

# Run the demo
if __name__ == "__main__":
    simulate_tests()
    
    print("\nTo start the web application, run:")
    print("python milk_quality_testing_mvp.py")
