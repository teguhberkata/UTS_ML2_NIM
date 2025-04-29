import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="crop_recommendation.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Crop Recommendation System")
st.write("Masukkan informasi lingkungan untuk mendapatkan rekomendasi tanaman yang cocok.")

# Form input pengguna
N = st.number_input("Kandungan Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
P = st.number_input("Kandungan Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
K = st.number_input("Kandungan Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
temperature = st.number_input("Suhu (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("pH Tanah", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=300.0, value=100.0)

if st.button("Rekomendasikan Tanaman"):
    # Preprocessing input
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    crop_name = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Rekomendasi tanaman: **{crop_name.upper()}**")
