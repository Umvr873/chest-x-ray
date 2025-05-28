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

##Download the pre-trained model weights:
resnet50_covid_classifier.pth

##NOTE:
Make sure all files are in the same folder
app.py
requirements.txt
packages.txt
resnet50_covid_classifier.pth

chest-xray-classifier/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── resnet50_covid_classifier.pth  # Model weights
└── README.md             # Documentation

##USAGE

##Run the application:
streamlit run app.py

Upload a chest X-ray image (JPG/PNG) through the web interface
View results including:
Classification probabilities
Model explanation
Clinical recommendations
Grad-CAM heatmap visualization

##Dataset Information
Model was trained on COVID-19_Radiography_Dataset(4 directories, 5 files)

##Medical Disclaimer
⚠️Important: This tool is for research/educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.







