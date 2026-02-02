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
    """
    Preprocess image for model prediction using MobileNetV2 preprocessing
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to model input size
    image = image.resize(IMG_SIZE)
    
    # Convert to array
    img_array = np.array(image).astype(np.float32)
    
    # MobileNetV2 preprocessing: scale to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ----------------------------
# SIDEBAR INFO
# ----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses a **MobileNetV2** deep learning model 
    for binary image classification.
    """)
    
    st.divider()
    
    st.header("📊 Model Details")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Task:** Binary Classification")
    st.write("**Image Size:** 224x224")
    st.write("**Preprocessing:** MobileNetV2 standard")
    
    st.divider()
    
    st.header("🔧 Environment")
    st.write(f"**TensorFlow:** {tf.__version__}")
    st.write(f"**NumPy:** {np.__version__}")
    
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
        st.image(image, caption="Uploaded Image", use_container_width=True)
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
            # Assuming output is sigmoid: close to 0 = class 0, close to 1 = class 1
            class_1_prob = float(prediction[0][0])
            class_0_prob = 1.0 - class_1_prob
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.stop()
    
    # Display results
    with col2:
        st.subheader("Classification Result")
        
        # Determine predicted class
        if class_1_prob >= 0.5:
            st.success("✅ **CLASS 1 DETECTED**")
            st.metric("Confidence", f"{class_1_prob:.2%}", delta=None)
        else:
            st.info("ℹ️ **CLASS 0 DETECTED**")
            st.metric("Confidence", f"{class_0_prob:.2%}", delta=None)
        
        # Confidence level
        max_prob = max(class_0_prob, class_1_prob)
        
        if max_prob >= 0.95:
            st.success("🎯 **Very High Confidence**")
        elif max_prob >= 0.85:
            st.info("✅ **High Confidence**")
        elif max_prob >= 0.70:
            st.warning("⚠️ **Moderate Confidence**")
        else:
            st.error("❗ **Low Confidence** - Result uncertain")
    
    # Detailed probabilities
    st.markdown("---")
    st.subheader("📊 Probability Breakdown")
    
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.metric(
            label="Class 0 Probability",
            value=f"{class_0_prob:.4f}",
            delta=f"{class_0_prob:.2%}"
        )
        st.progress(class_0_prob)
    
    with prob_col2:
        st.metric(
            label="Class 1 Probability",
            value=f"{class_1_prob:.4f}",
            delta=f"{class_1_prob:.2%}"
        )
        st.progress(class_1_prob)
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    if class_1_prob >= 0.5:
        st.write(f"""
        The model predicts this is **Class 1** with {class_1_prob:.2%} confidence.
        
        This means the model is {class_1_prob:.1%} certain that the uploaded image 
        belongs to Class 1 based on its learned features.
        """)
    else:
        st.write(f"""
        The model predicts this is **Class 0** with {class_0_prob:.2%} confidence.
        
        This means the model is {class_0_prob:.1%} certain that the uploaded image 
        belongs to Class 0 based on its learned features.
        """)
    
    # Technical details (expandable)
    with st.expander("🔧 Technical Details"):
        st.write("**Raw Model Output:**")
        st.code(f"Prediction: {prediction[0][0]:.6f}")
        
        st.write("\n**Interpretation:**")
        st.write(f"- Values close to 0 → Class 0")
        st.write(f"- Values close to 1 → Class 1")
        st.write(f"- Threshold: 0.5")
        
        st.write("\n**Preprocessing:**")
        st.write(f"- Resized to: {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
        st.write(f"- Normalized to: [-1, 1] range (MobileNetV2)")
        st.write(f"- Color mode: RGB")
        
        st.write("\n**Input Array Shape:**")
        st.code(f"{img_input.shape}")

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
    - Images will be automatically resized to 224x224
    """)
    
    st.markdown("---")
    st.subheader("🔍 What This Model Does")
    
    st.write("""
    This deep learning model performs **binary classification** on uploaded images.
    It uses a **MobileNetV2** architecture trained to distinguish between two classes.
    
    The model:
    - Processes images of any size (resized to 224×224)
    - Outputs probability scores for each class
    - Provides confidence levels for predictions
    - Uses transfer learning for improved accuracy
    """)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <p>🧠 Binary Classification System | Powered by MobileNetV2 & TensorFlow</p>
        <p><small>For Educational and Research Purposes Only</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
