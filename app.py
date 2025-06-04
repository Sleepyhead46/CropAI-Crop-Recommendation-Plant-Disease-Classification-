import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

import streamlit as st
import pandas as pd
import pickle

# Load the saved model (pickle)
with open('D:/projects/crop/p/best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
# Load scaler and label encoder with pickle
with open('D:/projects/crop/p/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('D:/projects/crop/p/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Crop Recommendation Prediction

st.title("Crop Recommendation Prediction")

N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=500.0, value=20.0, step=0.1)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=500.0, value=30.0, step=0.1)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=500.0, value=40.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=10.0, step=0.1)
ph = st.number_input("Soil pH", min_value=3.5, max_value=10.0, value=5.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=3000.0, value=100.0, step=0.1)

input_data = pd.DataFrame([{
    'N': N,
    'P': P,
    'K': K,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall
}])

input_scaled = scaler.transform(input_data)

if st.button("Predict Crop"):
    prediction_encoded = model.predict(input_scaled)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)
    st.success(f"Recommended Crop: {prediction_label[0]}")





# Load Model 
model = load_model('D:/projects/crop/plant_disease_model.keras')
class_names = ['Healthy', 'Powdery', 'Rust']

# Prediction Function 
def predict(img):
    img = img.resize((225, 225))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Streamlit UI 
st.title("Plant Disease Classifier")
st.write("Upload a leaf image to detect whether it's Healthy, Powdery, or Rust affected.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        label, confidence = predict(image_data)
        st.success(f"Prediction: **{label}** with **{confidence*100:.2f}%** confidence.")
