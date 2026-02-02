# ==========================================
# STREAMLIT APP - MRI BINARY CLASSIFICATION
# Using MobileNetV2 Model
# ==========================================

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import os

# ----------------------------
# CONSTANTS
# ----------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "MobileNet_Binary_MRI_best.keras"

st.set_page_config(
    page_title="MRI Binary Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 MRI Binary Classification")
st.write("Upload an image to classify it using our MobileNetV2 model.")
st.write(f"**Model:** MobileNetV2 | **TensorFlow:** {tf.__version__}")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    """Load the MobileNetV2 binary classification model"""
    try:
        # Check if file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            st.stop()
        
        # Get file size
        file_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
        
        # Load model with safe_mode=False for compatibility
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            compile=False,
            safe_mode=False
        )
        
        st.sidebar.success(f"✅ Model loaded successfully!")
        st.sidebar.write(f"📂 File size: {file_size:.2f} MB")
        st.sidebar.write(f"🏗️ Input shape: {model.input_shape}")
        st.sidebar.write(f"📤 Output shape: {model.output_shape
