# ==========================================
# STREAMLIT APP - ENSEMBLE MRI CLASSIFICATION
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
    "MobileNetV2": "MobileNet_Binary_MRI_best.keras",
    "DenseNet121": "DenseNet121_Binary_MRI_best.keras",
    "ResNet50": "ResNet50_Binary_MRI_best.keras"
}

# Class names
CLASS_NAMES = {0: "MRI", 1: "Non-MRI"}

st.set_page_config(
    page_title="Ensemble MRI Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Ensemble MRI Binary Classification")
st.write("Upload an image to classify it using an ensemble of three deep learning models.")
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
    """DenseNet121 preprocessing: scale to [0, 1] and normalize"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_resized = image.resize(IMG_SIZE)
    img_array = np.array(image_resized).astype(np.float32)
    # DenseNet preprocessing: per-channel mean subtraction
    # Mean values for ImageNet: [103.939, 116.779, 123.68]
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
    # ResNet preprocessing: per-channel mean subtraction
    # Mean values for ImageNet: [103.939, 116.779, 123.68]
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
    deep learning models to classify images:
    
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
    "📁 Choose an image...",
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
        st.image(image, caption="Original Image", use_column_width=True)
        st.write(f"📐 **Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"🎨 **Mode:** {image.mode}")
    
    # Make predictions with all models
    with st.spinner("🔍 Running ensemble prediction..."):
        time.sleep(0.5)
        
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
                pred = model.predict(img_input, verbose=0)
                non_mri_prob = float(pred[0][0])
                mri_prob = 1.0 - non_mri_prob
                
                predictions[model_name] = {
                    'mri_prob': mri_prob,
                    'non_mri_prob': non_mri_prob,
                    'predicted_class': 0 if mri_prob >= 0.5 else 1
                }
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.stop()
    
    # Calculate ensemble results
    avg_mri_prob = np.mean([p['mri_prob'] for p in predictions.values()])
    avg_non_mri_prob = np.mean([p['non_mri_prob'] for p in predictions.values()])
    
    # Majority voting
    votes = [p['predicted_class'] for p in predictions.values()]
    ensemble_class = 0 if votes.count(0) > votes.count(1) else 1
    ensemble_confidence = max(avg_mri_prob, avg_non_mri_prob)
    
    # Display ensemble results
    with col2:
        st.subheader("🎯 Ensemble Result")
        
        if ensemble_class == 0:
            st.success("🧲 **MRI IMAGE DETECTED**")
            st.metric("Ensemble Confidence", f"{avg_mri_prob:.2%}")
        else:
            st.error("🚫 **NON-MRI IMAGE DETECTED**")
            st.metric("Ensemble Confidence", f"{avg_non_mri_prob:.2%}")
        
        # Confidence level
        if ensemble_confidence >= 0.95:
            st.success("🎯 **Very High Confidence**")
        elif ensemble_confidence >= 0.85:
            st.info("✅ **High Confidence**")
        elif ensemble_confidence >= 0.70:
            st.warning("⚠️ **Moderate Confidence**")
        else:
            st.error("❗ **Low Confidence** - Result uncertain")
        
        # Voting breakdown
        st.write(f"**Voting:** {votes.count(0)} MRI, {votes.count(1)} Non-MRI")
    
    # Individual model predictions
    st.markdown("---")
    st.subheader("🤖 Individual Model Predictions")
    
    model_cols = st.columns(len(models))
    
    for idx, (model_name, pred) in enumerate(predictions.items()):
        with model_cols[idx]:
            st.write(f"**{model_name}**")
            
            if pred['predicted_class'] == 0:
                st.success("MRI")
                st.progress(pred['mri_prob'])
                st.write(f"{pred['mri_prob']:.2%}")
            else:
                st.error("Non-MRI")
                st.progress(pred['non_mri_prob'])
                st.write(f"{pred['non_mri_prob']:.2%}")
    
    # Detailed probabilities
    st.markdown("---")
    st.subheader("📊 Ensemble Probability Breakdown")
    
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.metric(
            label="🧲 MRI Probability (Average)",
            value=f"{avg_mri_prob:.4f}",
            delta=f"{avg_mri_prob:.2%}"
        )
        st.progress(avg_mri_prob)
    
    with prob_col2:
        st.metric(
            label="🚫 Non-MRI Probability (Average)",
            value=f"{avg_non_mri_prob:.4f}",
            delta=f"{avg_non_mri_prob:.2%}"
        )
        st.progress(avg_non_mri_prob)
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    if ensemble_class == 0:
        st.write(f"""
        The ensemble predicts this is **an MRI image** with {avg_mri_prob:.2%} average confidence.
        
        **Voting Results:**
        - {votes.count(0)} out of {len(votes)} models predicted MRI
        - {votes.count(1)} out of {len(votes)} models predicted Non-MRI
        
        The ensemble combines the strengths of three different architectures to provide 
        a more robust and reliable prediction.
        """)
    else:
        st.write(f"""
        The ensemble predicts this is **NOT an MRI image** with {avg_non_mri_prob:.2%} average confidence.
        
        **Voting Results:**
        - {votes.count(0)} out of {len(votes)} models predicted MRI
        - {votes.count(1)} out of {len(votes)} models predicted Non-MRI
        
        The ensemble combines the strengths of three different architectures to provide 
        a more robust and reliable prediction.
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.write("**Ensemble Method:**")
        st.write("- Voting: Majority vote from all models")
        st.write("- Probability: Average of all model probabilities")
        st.write("- Decision Threshold: 0.5")
        
        st.write("\n**Individual Model Predictions:**")
        for model_name, pred in predictions.items():
            st.write(f"- {model_name}: {pred['non_mri_prob']:.6f} (Raw output)")
        
        st.write("\n**Preprocessing Methods:**")
        st.write("- MobileNetV2: Scale to [-1, 1]")
        st.write("- DenseNet121: Subtract mean RGB [103.939, 116.779, 123.68]")
        st.write("- ResNet50: Subtract mean RGB [103.939, 116.779, 123.68]")
        
        st.write("\n**Image Processing:**")
        st.write(f"- Original size: {image.size[0]}x{image.size[1]}")
        st.write(f"- Resized to: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
        st.write(f"- Color mode: RGB")

# ----------------------------
# INSTRUCTIONS (when no image uploaded)
# ----------------------------
else:
    st.info("👆 Upload an image using the file uploader above to get started.")
    
    st.markdown("---")
    
    # Two columns for information
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.subheader("📖 How to Use")
        st.write("""
        1. **Click** on "Browse files" above
        2. **Select** an image from your computer
        3. **Wait** for all three models to analyze
        4. **View** the ensemble prediction results
        
        **Tips for Best Results:**
        - Use clear, high-quality images
        - Ensure proper orientation
        - Medical MRI scans work best
        """)
        
        st.subheader("🎯 Ensemble Advantages")
        st.write("""
        **Why use an ensemble?**
        - **Higher Accuracy**: Combines multiple models
        - **More Reliable**: Reduces individual model bias
        - **Robust**: Better handles edge cases
        - **Confidence**: Voting provides consensus
        """)
    
    with info_col2:
        st.subheader("🧪 Example Use Cases")
        
        st.write("**✅ Will Detect as MRI:**")
        st.write("- Brain MRI scans")
        st.write("- Spine MRI images")
        st.write("- Knee MRI scans")
        st.write("- Any medical MRI scan")
        
        st.write("\n**❌ Will Detect as Non-MRI:**")
        st.write("- Regular photos")
        st.write("- X-ray images")
        st.write("- CT scans")
        st.write("- Ultrasound images")
        st.write("- Everyday objects")
        
        st.subheader("🤖 The Three Models")
        st.write("""
        - **MobileNetV2**: Fast, efficient, mobile-optimized
        - **DenseNet121**: Dense connections, feature reuse
        - **ResNet50**: Deep architecture, residual learning
        """)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <p>🧠 Ensemble MRI Classification System</p>
        <p><strong>MobileNetV2 + DenseNet121 + ResNet50</strong></p>
        <p><small>For Educational and Research Purposes Only</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
