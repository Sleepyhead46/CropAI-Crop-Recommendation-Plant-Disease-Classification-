
# Plant Disease Classification Using CNN

This project implements a Convolutional Neural Network (CNN) to classify plant leaf images into three categories: **Healthy**, **Powdery mildew**, and **Rust**. The dataset is organized into training, validation, and testing folders and stored on Google Drive. The model is trained using TensorFlow and Keras on Google Colab.

---

## Project Overview

This project trains a CNN model to detect common plant leaf diseases from images. It uses image augmentation for training and evaluates performance on a separate validation and test set. The model is saved and can be loaded for prediction on new images.

---

##  Key Features

- Loads training, validation, and test datasets from Google Drive
- Visualizes sample leaf images
- Applies real-time image augmentation using `ImageDataGenerator`
- Builds a custom CNN architecture
- Trains the model and monitors performance using accuracy/loss plots
- Evaluates performance with:
  - Confusion Matrix
  - Accuracy metrics
- Predicts disease class from single images
- Saves trained model in `.keras` format for reuse

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


- **Language:** Python 3.11
- **Libraries:**
  - Deep Learning: `tensorflow`, `keras`
  - Visualization: `matplotlib`, `seaborn`
  - Data Handling: `numpy`, `os`
  - Evaluation: `sklearn.metrics`
- **Platform:** Google Colab (preferred)

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
