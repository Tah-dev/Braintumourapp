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
MODEL_PATH = "MobileNet_Binary_MRI.keras"

st.set_page_config(
    page_title="MRI Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 MRI Image Classification")
st.write("Upload an image to check if it is an MRI scan.")
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
            compile=False
        )
        
        st.sidebar.success(f"✅ Model loaded successfully!")
        st.sidebar.write(f"📂 File size: {file_size:.2f} MB")
        st.sidebar.write(f"🏗️ Input shape: {model.input_shape}")
        st.sidebar.write(f"📤 Output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load the model
model = load_model()

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# SIDEBAR INFO
# ----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses a **MobileNetV2** deep learning model 
    to classify images as MRI scans or non-MRI images.
    """)
    
    st.divider()
    
    st.header("📊 Model Details")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Task:** Binary Classification")
    st.write("**Classes:** MRI vs Non-MRI")
    st.write("**Image Size:** 224x224")
    
    st.divider()
    
    st.header("⚠️ Disclaimer")
    st.error("""
    **Educational purposes only.**
    
    This tool is NOT for medical diagnosis.
    Always consult healthcare professionals.
    """)

# ----------------------------
# MAIN APP - IMAGE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "📁 Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG, or PNG image"
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"📐 **Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"🎨 **Mode:** {image.mode}")
    
    # Preprocess image
    img_input = preprocess_image(image)
    
    # Make prediction
    with st.spinner("🔍 Analyzing image..."):
        time.sleep(0.5)
        
        try:
            # Get prediction
            prediction = model.predict(img_input, verbose=0)
            
            # Extract probabilities
            non_mri_prob = float(prediction[0][0])
            mri_prob = 1.0 - non_mri_prob
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.stop()
    
    # Display results
    with col2:
        st.subheader("Classification Result")
        
        # Show prediction
        if mri_prob >= 0.5:
            st.success("🧲 **MRI IMAGE DETECTED**")
            st.metric("Confidence", f"{mri_prob:.2%}", delta=None)
        else:
            st.error("🚫 **NOT AN MRI IMAGE**")
            st.metric("Confidence", f"{non_mri_prob:.2%}", delta=None)
        
        # Confidence level
        max_prob = max(mri_prob, non_mri_prob)
        
        if max_prob >= 0.95:
            st.info("🎯 **Very High Confidence**")
        elif max_prob >= 0.85:
            st.info("✅ **High Confidence**")
        elif max_prob >= 0.70:
            st.info("⚠️ **Moderate Confidence**")
        else:
            st.warning("⚠️ **Low Confidence** - Result uncertain")
    
    # Detailed probabilities
    st.markdown("---")
    st.subheader("📊 Probability Breakdown")
    
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.metric(
            label="🧲 MRI Probability",
            value=f"{mri_prob:.4f}",
            delta=f"{mri_prob:.2%}"
        )
        st.progress(mri_prob)
    
    with prob_col2:
        st.metric(
            label="🚫 Non-MRI Probability",
            value=f"{non_mri_prob:.4f}",
            delta=f"{non_mri_prob:.2%}"
        )
        st.progress(non_mri_prob)
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    if mri_prob >= 0.5:
        st.write(f"""
        The model predicts this is **an MRI image** with {mri_prob:.2%} confidence.
        
        This means the model is {mri_prob:.1%} certain that the uploaded image 
        is a medical MRI scan based on its learned features.
        """)
    else:
        st.write(f"""
        The model predicts this is **NOT an MRI image** with {non_mri_prob:.2%} confidence.
        
        This means the model is {non_mri_prob:.1%} certain that the uploaded image 
        is not a medical MRI scan.
        """)
    
    # Technical details (expandable)
    with st.expander("🔧 Technical Details"):
        st.write("**Raw Model Output:**")
        st.code(f"Prediction: {prediction[0][0]:.6f}")
        st.write("**Interpretation:**")
        st.write(f"- Values close to 0 → MRI")
        st.write(f"- Values close to 1 → Non-MRI")
        st.write(f"- Threshold: 0.5")
        
        st.write("\n**Preprocessing:**")
        st.write(f"- Resized to: {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
        st.write(f"- Normalized: [0, 1] range")
        st.write(f"- Color mode: RGB")

# ----------------------------
# INSTRUCTIONS (when no image uploaded)
# ----------------------------
else:
    st.info("👆 Upload an image using the file uploader above to get started.")
    
    st.markdown("---")
    st.subheader("📖 How to Use")
    
    st.write("""
    1. **Click** on "Browse files" above
    2. **Select** an image from your computer (JPG, JPEG, or PNG)
    3. **Wait** for the model to analyze the image
    4. **View** the classification result and confidence score
    
    **Tips for Best Results:**
    - Use clear, high-quality images
    - Ensure the image is properly oriented
    - Medical MRI scans work best
    """)
    
    st.markdown("---")
    st.subheader("🧪 Example Use Cases")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        st.write("**✅ Will Detect as MRI:**")
        st.write("- Brain MRI scans")
        st.write("- Spine MRI images")
        st.write("- Any medical MRI scan")
    
    with example_col2:
        st.write("**❌ Will Detect as Non-MRI:**")
        st.write("- Regular photos")
        st.write("- X-ray images")
        st.write("- CT scans")
        st.write("- Ultrasound images")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <p>🧠 MRI Classification System | Powered by MobileNetV2 & TensorFlow</p>
        <p><small>For Educational and Research Purposes Only</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
