# 🌾 CropAI: Smart Crop Recommendation & Plant Disease Classification App

**Empowering Farmers with AI-Driven Agricultural Insights**  
A Streamlit-based application that combines **machine learning** and **deep learning** to provide actionable recommendations for crop cultivation and plant disease detection.

CropAI is an integrated machine learning-powered Streamlit application that helps farmers and agriculture enthusiasts by:

1. Recommending the most suitable crop to cultivate based on soil nutrients and environmental conditions.
    # Crop Recommendation System Using Machine Learning
    This is a machine learning-based recommendation system designed to suggest the most suitable crop to cultivate based on environmental and soil conditions. This project aims to improve agricultural productivity by leveraging data-driven insights from soil nutrients, temperature, humidity, pH levels, and rainfall.

2. Classifying plant leaf images into disease categories: Healthy, Powdery mildew, and Rust.
    # Plant Disease Classification Using CNN
    This project implements a Convolutional Neural Network (CNN) to classify plant leaf images into three categories: **Healthy**, **Powdery mildew**, and **Rust**. The dataset is organized into training, validation, and testing folders and stored on Google Drive. The model is trained using TensorFlow and Keras on Google Colab.

The app combines classical machine learning (Random Forest) for crop prediction with deep learning (CNN) for disease classification in a single user-friendly interface.

---

## 🚀 Overview

**CropAI** is an integrated, user-friendly web app designed to assist farmers, agriculturalists, and researchers by:

### 🌱 Crop Recommendation
> *Predict the most suitable crop based on environmental and soil parameters using machine learning.*

- Utilizes Random Forest for accurate crop predictions.
- Inputs: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature (°C), Humidity (%), Soil pH, and Rainfall (mm).

### 🧪 Plant Disease Classification
> *Identify plant leaf diseases with a Convolutional Neural Network (CNN).*

- Classifies leaf images into: **Healthy**, **Powdery Mildew**, or **Rust**.
- Built with TensorFlow & Keras, trained on a publicly available plant disease dataset.

---

## ✨ Features

- ✅ **Crop Recommendation Engine**
- 🌿 **Plant Disease Detection from Leaf Images**
- 🧠 **Pre-trained ML & DL Models** for fast and reliable results
- 🎯 **Clean, Interactive UI** with instant feedback
- ⚡ **Streamlit Interface** — accessible via web

---

## 🗂️ Dataset Information

### 📁 Plant Disease Dataset Structure  
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset/)

```
plant_dataset/
├── Train/
│   ├── Healthy/
│   ├── Powdery/
│   └── Rust/
├── Validation/
│   ├── Healthy/
│   ├── Powdery/
│   └── Rust/
└── Test/
    ├── Healthy/
    ├── Powdery/
    └── Rust/
```

### 🌾 Crop Recommendation Dataset  
Files:
- `crop.csv`
- `Crop_recommendation.csv`
- Merged into `combined_crop_data.csv`

---

## 🧰 Tech Stack

- **Language:** Python 3.11  
- **Framework:** Streamlit  
- **Libraries:**
  - `pandas`, `numpy`, `scikit-learn` — crop model
  - `tensorflow`, `keras` — CNN for plant classification
  - `PIL` (`pillow`) — image processing
  - `pickle` — loading ML models

---

## ⚙️ Setup Instructions

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/CropAI.git
   cd CropAI
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Models**  
   Place these files in the root or correct paths (or modify paths in code):
   - `best_rf_model.pkl` – Random Forest model for crops  
   - `scaler.pkl` – Scaler for input features  
   - `label_encoder.pkl` – Label encoder for crop names  
   - `plant_disease_model.keras` – CNN model for disease classification

4. **Run the App**  
   ```bash
   streamlit run app.py
   ```

---

## 🧪 How to Use

### 🌱 Crop Recommendation
- Input values for **soil nutrients** and **climatic conditions**.
- Click **“Predict Crop”** to receive an optimal crop suggestion.

### 🌿 Plant Disease Classification
- Upload a **leaf image** (JPG/PNG).
- Click **“Predict Disease”** to classify it as:
  - **Healthy**
  - **Powdery Mildew**
  - **Rust**

---

## 🧩 Code Highlights

- Models are **cached** to optimize performance.
- Crop inputs are **scaled** before prediction.
- Leaf images are **resized and normalized** before classification.
- Clean **tab-based UI** separates features for intuitive use.

---

For Better Understanding go thorough the README_CropAI.md and Plant_Disease_Classification_README.md

---


## 📬 Contact

**Author:** Samyak Deshar  
📧 [sdeshar9803@gmail.com]
