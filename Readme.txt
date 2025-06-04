CropAI: Crop Recommendation & Plant Disease Classification App
==============================================================

Overview
--------
CropAI is an integrated machine learning-powered Streamlit application that helps farmers and agriculture enthusiasts by:

1. Recommending the most suitable crop to cultivate based on soil nutrients and environmental conditions.
    # Crop Recommendation System Using Machine Learning
    This is a machine learning-based recommendation system designed to suggest the most suitable crop to cultivate based on environmental and soil conditions. This project aims to improve agricultural productivity by leveraging data-driven insights from soil nutrients, temperature, humidity, pH levels, and rainfall.

2. Classifying plant leaf images into disease categories: Healthy, Powdery mildew, and Rust.
    # Plant Disease Classification Using CNN
    This project implements a Convolutional Neural Network (CNN) to classify plant leaf images into three categories: **Healthy**, **Powdery mildew**, and **Rust**. The dataset is organized into training, validation, and testing folders and stored on Google Drive. The model is trained using TensorFlow and Keras on Google Colab.

The app combines classical machine learning (Random Forest) for crop prediction with deep learning (CNN) for disease classification in a single user-friendly interface.

---

Features
--------
- Crop recommendation based on inputs:
  • Nitrogen (N), Phosphorus (P), Potassium (K)
  • Temperature (°C), Humidity (%), Soil pH, Rainfall (mm)
- Plant disease detection from leaf images using CNN.
- Clean UI with input validation and instant prediction.
- Uses pre-trained models for fast and accurate results.

---

Dataset Structure
-----------------
Dataset can be downloaded from the following links:
- Plant Disease Classification Using CNN (https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset/)
The datasets for crop recommendation and plant disease classification are organized as follows:

plant_dataset/
├── Train/
│   └── Train/
│       ├── Healthy/
│       ├── Powdery/
│       └── Rust/
├── Validation/
│   └── Validation/
│       ├── Healthy/
│       ├── Powdery/
│       └── Rust/
└── Test/
    └── Test/
        ├── Healthy/
        ├── Powdery/
        └── Rust/

Crop Recommendation Dataset
---------------------------
The crop recommendation model uses a dataset with features such as:
crop.csv
Crop_recommendation.csv
to generate combined_crop_data.csv

Tech Stack
----------
- Language: Python 3.11
- Framework: Streamlit
- Libraries:
  • pandas, numpy, scikit-learn (for crop recommendation)
  • tensorflow, keras (for CNN plant disease model)
  • pillow (PIL) for image processing
  • pickle for loading saved ML models

---

Setup Instructions
------------------
1. Clone or download the project files to your local machine.
2. Install dependencies:
   pip install -r requirements.txt
   Ensure you have Python 3.11 installed.
3. Place the following model files in the specified paths or update paths in the code:
   - best_rf_model.pkl (Random Forest crop model)
   - scaler.pkl (feature scaler)
   - label_encoder.pkl (label encoder for crops)
   - plant_disease_model.keras (CNN for plant disease classification)

4. Run the Streamlit app:
   streamlit run app.py

---

Usage
-----
Crop Recommendation Tab
- Enter values for soil nutrients and environmental parameters.
- Click “Predict Crop” to get the recommended crop based on your inputs.

Plant Disease Classifier Tab
- Upload a clear image of a plant leaf (jpg/png/jpeg).
- Click “Predict Disease” to classify if the leaf is Healthy, Powdery mildew, or Rust.
- The app displays the predicted class and confidence percentage.

---

Code Structure Highlights
-------------------------
- Models are loaded with caching to speed up app responsiveness.
- Crop data inputs are scaled before prediction.
- CNN model resizes and normalizes images for accurate predictions.
- Two tabs separate crop recommendation and plant disease detection.

---

Future Improvements
-------------------
- Add real-time weather API integration to enhance crop recommendations.
- Build a mobile app version for field use.
- Use more advanced deep learning architectures for disease classification.
- Extend crop dataset and disease categories for broader coverage.

---

Contact
-------
Author: Samyak Deshar
Email: sdeshar9803@gmail.com
