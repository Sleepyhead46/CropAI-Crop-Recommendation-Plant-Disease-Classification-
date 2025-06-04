
# 🌾 CropAI: Crop Recommendation System Using Machine Learning

---

## 📌 Overview

**CropAI** is a machine learning-based recommendation system designed to suggest the most suitable crop to cultivate based on environmental and soil conditions. This project aims to improve agricultural productivity by leveraging data-driven insights from soil nutrients, temperature, humidity, pH levels, and rainfall.

Using a range of powerful classification algorithms, the system predicts which crop is most appropriate for given conditions, helping farmers make informed decisions.

---

## 🧩 Key Features

- Loads and preprocesses agricultural datasets
- Handles missing values and outlier detection
- Performs exploratory data analysis (EDA) to identify trends and anomalies
- Applies feature scaling to standardize input variables
- Trains multiple machine learning models, including:
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Gradient Boosting Classifier
  - Bagging Classifier
  - Extra Trees Classifier
- Evaluates models using metrics like accuracy, precision, recall, F1-score, and confusion matrix
- Visualizes feature importance for explainability
- Saves the best-performing model for deployment

---

## 🧪 Tech Stack

- **Language:** Python 3.x
- **Libraries:**
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`
  - Utilities: `warnings`, `os`, `joblib`

---

## 📂 Dataset Requirements

Ensure your dataset is in CSV format and located in the project directory. It must contain the following columns:

| Feature       | Description                          |
|---------------|--------------------------------------|
| N             | Nitrogen content in soil             |
| P             | Phosphorus content in soil           |
| K             | Potassium content in soil            |
| temperature   | Temperature in degrees Celsius       |
| humidity      | Relative humidity in %               |
| ph            | Soil pH value                        |
| rainfall      | Rainfall in mm                       |
| label         | Target variable (crop name)          |

File name should be: `Crop_recommendation.csv`

---

## ✅ Usage Instructions

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### 2. Run Jupyter Notebook

If using a notebook:

```bash
jupyter notebook crop_recommendation.ipynb
```

Or if using a Python script:

```bash
python crop_recommendation.py
```

### 3. Launch the Web App (Optional)

If you’ve built a Streamlit-based web app:

```bash
streamlit run app.py
```

---

## 🔍 Workflow Explanation

### 1. Import Required Libraries
Import essential libraries for data processing, visualization, and machine learning.

### 2. Load and Understand Data
Load the dataset and examine its structure, including class distribution and feature statistics.

### 3. Exploratory Data Analysis (EDA)
Visualize feature distributions and relationships between features and the target crop.

### 4. Address Class Imbalance
Apply under-sampling if needed to ensure balanced representation of crop types.

### 5. Feature Scaling
Standardize the features to avoid bias due to feature magnitude differences.

### 6. Data Splitting
Split the dataset into training and test sets for fair model evaluation.

### 7. Train Multiple Models
Train various classifiers including ensemble, boosting, and linear models.

### 8. Evaluate Models
Evaluate models using classification metrics and visualize confusion matrices.

### 9. Select the Best Model
Choose the most accurate and generalizable model for deployment.

### 10. Save the Model
Serialize the selected model using `joblib` for future predictions.

---

## 📊 Example Output

- Accuracy scores of all trained models
- Confusion matrix heatmaps
- Feature importance chart from Random Forest or XGBoost
- Saved model file: `best_model.pkl`

---

## 📬 Author

- **Name:** Samyak Deshar  
- **Project:** CropAI  
- **Contact:** [youremail@example.com] (replace with your real contact if you wish)

---

## 📜 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🚀 Future Improvements

- Integrate real-time weather API for dynamic predictions
- Add mobile app interface
- Use deep learning models for improved accuracy
- Incorporate geolocation-based soil datasets

---



# Plant Disease Classification Using CNN

This project implements a Convolutional Neural Network (CNN) to classify plant leaf images into three categories: **Healthy**, **Powdery mildew**, and **Rust**. The dataset is organized into training, validation, and testing folders and stored on Google Drive. The model is trained using TensorFlow and Keras on Google Colab.

---

## Project Overview

This project trains a CNN model to detect common plant leaf diseases from images. It uses image augmentation for training and evaluates performance on a separate validation and test set. The model is saved and can be loaded for prediction on new images.

---

## Dataset Structure

The dataset is stored on Google Drive with the following directory structure:

```
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
```

Each folder contains images of leaves belonging to the respective class.

---

## Environment Setup

- Python 3.11
- TensorFlow 
- Keras
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab (recommended for easy GPU usage)

---

## How to Run

1. **Mount Google Drive in Colab**

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Set paths** to your dataset folders on Google Drive.

3. **Run the notebook/code** to:
   - Count images per class
   - Display sample images
   - Prepare image data generators
   - Build and train the CNN model
   - Evaluate on validation and test sets
   - Plot training history and confusion matrix
   - Save the trained model
   - Predict on single images

---

## Model Architecture

- Input: 225x225 RGB images
- Conv2D (32 filters, 3x3) + ReLU + MaxPooling
- Conv2D (64 filters, 3x3) + ReLU + MaxPooling
- Flatten
- Dense (64 units, ReLU)
- Output Dense (3 units, Softmax)

---

## Training Details

- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 5 (can be increased for better performance)
- Batch size: 32
- Data augmentation applied to training images (rescale, shear, zoom, horizontal flip)

---

## Results and Evaluation

- The model is evaluated on a separate test set.
- Confusion matrix and accuracy/loss plots are generated.
- Sample predictions on test images are printed.
- Final test accuracy is displayed and plotted.

---
