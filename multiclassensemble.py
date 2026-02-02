# ==========================================
# STREAMLIT APP - ENSEMBLE MULTICLASS CLASSIFICATION
# Using MobileNetV2, DenseNet121, and ResNet50
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

# Model paths
MODEL_PATHS = {
    "MobileNetV2": "MobileNetV2_Multiclass_best.keras",
    "DenseNet121": "DenseNet121_Multiclass_best.keras",
    "ResNet50": "ResNet50_Multiclass_best.keras"
}

# Class names - Brain Tumor Types
CLASS_NAMES = ['Glioma', 'Meningioma', 'Normal', 'Pituitary']
CLASS_INDICES = {'glioma': 0, 'meningioma': 1, 'normal': 2, 'pituitary': 3}
NUM_CLASSES = len(CLASS_NAMES)

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Ensemble Brain Tumor Classification")
st.write("Upload an MRI image to classify brain tumors using an ensemble of three deep learning models.")
st.write(f"**Models:** MobileNetV2 + DenseNet121 + ResNet50 | **TensorFlow:** {tf.__version__}")

# ----------------------------
# PREPROCESSING FUNCTIONS
# ----------------------------

def preprocess_mobilenet(image: Image.Image) -> np.ndarray:
    """MobileNetV2 preprocessing: scale to [-1, 1]"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_resized = image.resize(IMG_SIZE)
    img_array = np.array(image_resized).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_densenet(image: Image.Image) -> np.ndarray:
    """DenseNet121 preprocessing: subtract mean RGB values"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_resized = image.resize(IMG_SIZE)
    img_array = np.array(image_resized).astype(np.float32)
    # DenseNet preprocessing
    img_array[..., 0] -= 103.939
    img_array[..., 1] -= 116.779
    img_array[..., 2] -= 123.68
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_resnet(image: Image.Image) -> np.ndarray:
    """ResNet50 preprocessing: subtract mean RGB values"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_resized = image.resize(IMG_SIZE)
    img_array = np.array(image_resized).astype(np.float32)
    # ResNet preprocessing
    img_array[..., 0] -= 103.939
    img_array[..., 1] -= 116.779
    img_array[..., 2] -= 123.68
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():
    """Load all three models for ensemble prediction"""
    models = {}
    model_info = {}
    
    for model_name, model_path in MODEL_PATHS.items():
        try:
            if not os.path.exists(model_path):
                st.warning(f"⚠️ {model_name} model not found: {model_path}")
                continue
            
            # Load model
            model = tf.keras.models.load_model(model_path, compile=False)
            models[model_name] = model
            
            # Get model info
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            
            model_info[model_name] = {
                'file_size': file_size,
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            }
            
        except Exception as e:
            st.error(f"❌ Error loading {model_name}: {str(e)}")
            continue
    
    if len(models) == 0:
        st.error("❌ No models could be loaded! Please check model files.")
        st.stop()
    
    return models, model_info

# Load all models
models, model_info = load_models()

# ----------------------------
# SIDEBAR - MODEL INFO
# ----------------------------
with st.sidebar:
    st.header("🤖 Ensemble Models")
    
    if len(models) == 3:
        st.success(f"✅ All 3 models loaded successfully!")
    else:
        st.warning(f"⚠️ {len(models)}/3 models loaded")
    
    st.write(f"**Number of Classes:** {NUM_CLASSES}")
    st.write(f"**Classes:** {', '.join(CLASS_NAMES)}")
    
    st.divider()
    
    # Display info for each loaded model
    for model_name in models.keys():
        with st.expander(f"📊 {model_name}"):
            info = model_info[model_name]
            st.write(f"**File Size:** {info['file_size']:.2f} MB")
            st.write(f"**Input Shape:** {info['input_shape']}")
            st.write(f"**Output Shape:** {info['output_shape']}")
    
    st.divider()
    
    st.header("ℹ️ About Ensemble")
    st.write("""
    This app uses an **ensemble** of three state-of-the-art 
    deep learning models to classify brain MRI images:
    
    - **MobileNetV2**: Lightweight and efficient
    - **DenseNet121**: Dense connections for feature reuse
    - **ResNet50**: Deep residual learning
    
    The final prediction is based on **majority voting** 
    and **average probability** from all models.
    """)
    
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
    "📁 Choose a brain MRI image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG, or PNG image"
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    
    # Create layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, caption="Original MRI Image", use_column_width=True)
        st.write(f"📐 **Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"🎨 **Mode:** {image.mode}")
    
    # Make predictions with all models
    with st.spinner("🔍 Running ensemble prediction..."):
        time.sleep(0.3)
        
        predictions = {}
        preprocessing_functions = {
            "MobileNetV2": preprocess_mobilenet,
            "DenseNet121": preprocess_densenet,
            "ResNet50": preprocess_resnet
        }
        
        try:
            for model_name, model in models.items():
                # Preprocess image for this specific model
                preprocess_func = preprocessing_functions[model_name]
                img_input = preprocess_func(image)
                
                # Get prediction
                pred_probs = model.predict(img_input, verbose=0)[0]
                pred_class = np.argmax(pred_probs)
