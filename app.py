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

# Page configuration
st.set_page_config(
    page_title="MRI Binary Classification",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown('<p class="main-header">🧠 MRI Binary Classification System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by MobileNetV2 Deep Learning Model</p>', unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the MobileNetV2 binary classification model"""
    try:
        # Check if file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            st.info("📋 Please ensure 'MobileNet_Binary_MRI_best.keras' is in the app directory.")
            st.stop()
        
        # Get file size
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Convert to MB
        
        # Load model with safe_mode=False for compatibility
        with st.spinner("🔄 Loading model... Please wait..."):
            model = tf.keras.models.load_model(
                MODEL_PATH, 
                compile=False,
                safe_mode=False
            )
        
        return model, file_size
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 This might be due to TensorFlow version mismatch. Please check the deployment logs.")
        st.stop()

# Load the model
model, file_size = load_model()

# ----------------------------
# SIDEBAR - MODEL INFO
# ----------------------------
with st.sidebar:
    st.header("ℹ️ Model Information")
    
    st.markdown("**Architecture:**")
    st.info("MobileNetV2")
    
    st.markdown("**Task:**")
    st.info("Binary Classification")
    
    st.markdown("**Model File:**")
    st.code(MODEL_PATH, language=None)
    
    st.markdown("**File Size:**")
    st.success(f"{file_size:.2f} MB")
    
    st.markdown("**Input Shape:**")
    st.code(f"{model.input_shape}", language=None)
    
    st.markdown("**Output Shape:**")
    st.code(f"{model.output_shape}", language=None)
    
    st.markdown("**Total Parameters:**")
    st.metric("Parameters", f"{model.count_params():,}")
    
    st.divider()
    
    # Environment Info
    st.header("🔧 Environment")
    st.markdown(f"""
    **Training Environment:**
    - Python: 3.10.19
    - TensorFlow: 2.10.1
    - NumPy: 1.24.3
    
    **Current Environment:**
    - TensorFlow: {tf.__version__}
    - NumPy: {np.__version__}
    """)
    
    st.divider()
    
    # Instructions
    st.header("📖 How to Use")
    st.markdown("""
    1. **Upload** an image (JPG, JPEG, PNG)
    2. **Wait** for the model to analyze
    3. **View** classification results
    4. **Check** confidence scores
    """)
    
    st.divider()
    
    # Disclaimer
    st.header("⚠️ Disclaimer")
    st.error("""
    **For Educational Use Only**
    
    This tool is NOT intended for medical diagnosis. 
    Always consult qualified healthcare professionals 
    for medical advice.
    """)

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
# MAIN APP - FILE UPLOAD
# ----------------------------
st.markdown("---")

uploaded_file = st.file_uploader(
    "📁 Upload an Image for Classification",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Load image
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        st.stop()
    
    # Display image and info
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_container_width=True, caption="Original Image")
        
        # Image info
        st.markdown("**Image Details:**")
        st.write(f"- **Size:** {image.size[0]} × {image.size[1]} pixels")
        st.write(f"- **Mode:** {image.mode}")
        st.write(f"- **Format:** {image.format if image.format else 'Unknown'}")
        
    # Preprocess and predict
    with col2:
        st.subheader("🔍 Analysis Results")
        
        # Preprocess image
        with st.spinner("🔄 Preprocessing image..."):
            img_input = preprocess_image(image)
            time.sleep(0.3)  # Small delay for UX
        
        # Make prediction
        with st.spinner("🧠 Running model inference..."):
            try:
                prediction = model.predict(img_input, verbose=0)
                pred_value = float(prediction[0][0])
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.stop()
        
        # Calculate probabilities
        # Output is sigmoid: close to 0 = class 0, close to 1 = class 1
        class_1_prob = pred_value
        class_0_prob = 1.0 - pred_value
        
        # Determine predicted class (threshold = 0.5)
        predicted_class = 1 if pred_value >= 0.5 else 0
        confidence = max(class_0_prob, class_1_prob)
        
        # Display prediction
        st.markdown("### 🎯 Prediction")
        
        if predicted_class == 1:
            st.success("✅ **CLASS 1 DETECTED**")
            st.metric("Confidence", f"{class_1_prob:.2%}", delta=None)
        else:
            st.info("ℹ️ **CLASS 0 DETECTED**")
            st.metric("Confidence", f"{class_0_prob:.2%}", delta=None)
        
        # Confidence level indicator
        if confidence >= 0.95:
            st.success("🎯 **Very High Confidence**")
        elif confidence >= 0.85:
            st.info("✅ **High Confidence**")
        elif confidence >= 0.70:
            st.warning("⚠️ **Moderate Confidence**")
        else:
            st.error("❗ **Low Confidence** - Result uncertain")
    
    # ----------------------------
    # DETAILED PROBABILITIES
    # ----------------------------
    st.markdown("---")
    st.subheader("📊 Probability Breakdown")
    
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.markdown("**Class 0 Probability**")
        st.progress(class_0_prob)
        st.metric(
            label="Class 0",
            value=f"{class_0_prob:.4f}",
            delta=f"{class_0_prob:.2%}"
        )
    
    with prob_col2:
        st.markdown("**Class 1 Probability**")
        st.progress(class_1_prob)
        st.metric(
            label="Class 1",
            value=f"{class_1_prob:.4f}",
            delta=f"{class_1_prob:.2%}"
        )
    
    # ----------------------------
    # INTERPRETATION
    # ----------------------------
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    with st.expander("📖 Understanding the Results", expanded=True):
        st.write(f"""
        **Predicted Class:** {'Class 1' if predicted_class == 1 else 'Class 0'}
        
        **Confidence Level:** {confidence:.2%}
        
        **What does this mean?**
        - The model outputs a probability between 0 and 1
        - Values **close to 0** → Class 0
        - Values **close to 1** → Class 1
        - Threshold for classification: **0.5**
        
        **Raw Output:** {pred_value:.6f}
        
        **Model Decision:**
        {f"Since {pred_value:.4f} ≥ 0.5, the model predicts **Class 1**" if predicted_class == 1 
          else f"Since {pred_value:.4f} < 0.5, the model predicts **Class 0**"}
        """)
    
    # ----------------------------
    # TECHNICAL DETAILS
    # ----------------------------
    with st.expander("🔧 Technical Details"):
        st.write(f"""
        **Preprocessing Pipeline:**
        1. Convert to RGB color mode
        2. Resize to {IMG_SIZE[0]}×{IMG_SIZE[1]} pixels
        3. Normalize to [-1, 1] range (MobileNetV2 preprocessing)
        4. Add batch dimension
        
        **Model Architecture:**
        - Base: MobileNetV2 (ImageNet pretrained)
        - Custom classification head
        - Output: Single sigmoid neuron
        
        **Classification:**
        - Binary classification task
        - Sigmoid activation function
        - Decision threshold: 0.5
        
        **Raw Model Output:**
```
        {prediction[0][0]:.8f}
```
        
        **Input Shape:** {img_input.shape}
        """)

# ----------------------------
# INSTRUCTIONS (No image uploaded)
# ----------------------------
else:
    st.info("👆 **Upload an image using the file uploader above to begin classification**")
    
    st.markdown("---")
    
    # Example use cases
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("✅ Supported Images")
        st.markdown("""
        - JPG/JPEG format
        - PNG format
        - RGB or grayscale images
        - Any resolution (will be resized)
        """)
    
    with col_b:
        st.subheader("📝 Best Practices")
        st.markdown("""
        - Use clear, well-lit images
        - Ensure proper image orientation
        - Avoid heavily compressed images
        - Check image quality before upload
        """)
    
    st.markdown("---")
    
    # Model capabilities
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
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p style='font-size: 1.1rem; font-weight: bold;'>🧠 MRI Binary Classification System</p>
        <p>Powered by <strong>MobileNetV2</strong> & <strong>TensorFlow</strong></p>
        <p style='font-size: 0.9rem;'><em>For Educational and Research Purposes Only</em></p>
        <p style='font-size: 0.8rem; margin-top: 10px;'>
            Model trained with: Python 3.10.19 | TensorFlow 2.10.1 | NumPy 1.24.3
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
