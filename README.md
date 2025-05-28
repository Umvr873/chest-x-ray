# Chest X-ray Classification System

![Demo Screenshot](demo_screenshot.png) <!-- Add a screenshot later -->

A deep learning-powered web application for classifying chest X-ray images into four categories: COVID-19, Lung Opacity, Normal, and Viral Pneumonia. The system provides Grad-CAM heatmaps to visualize model decisions and clinical recommendations for each diagnosis.

## Features

- 🏥 Medical Image Classification: Identifies 4 pulmonary conditions
- 🔍 Explainable AI: Visual model explanations using Grad-CAM
- 🚨 Clinical Guidance: Professional medical recommendations
- 📊 Probability Scores: Shows confidence levels for each diagnosis
- 🖥️ Web Interface: User-friendly Streamlit application

## Technical Specifications

- Model Architecture: ResNet50 (custom final layer)
- Input: 224×224 RGB chest X-ray images
- Output: Classification probabilities with explanations
- Framework: PyTorch + TorchVision
- Interface: Streamlit web app

# Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Umvr873/chest-xray-classifier.git
   cd chest-xray-classifier

##Install Independencies:
pip install -r requirements.txt

##Or manually install:
pip install streamlit torch torchvision pillow numpy matplotlib







