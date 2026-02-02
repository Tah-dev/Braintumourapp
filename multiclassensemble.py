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
                
                predictions[model_name] = {
                    'probabilities': pred_probs,
                    'predicted_class': pred_class,
                    'confidence': pred_probs[pred_class]
                }
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.stop()
    
    # Calculate ensemble results
    avg_probs = np.mean([p['probabilities'] for p in predictions.values()], axis=0)
    ensemble_class = np.argmax(avg_probs)
    ensemble_confidence = avg_probs[ensemble_class]
    
    # Majority voting
    votes = [p['predicted_class'] for p in predictions.values()]
    vote_counts = np.bincount(votes, minlength=NUM_CLASSES)
    majority_class = np.argmax(vote_counts)
    
    # Display ensemble results
    with col2:
        st.subheader("🎯 Ensemble Result")
        
        # Color code based on tumor type
        predicted_class_name = CLASS_NAMES[ensemble_class]
        
        if predicted_class_name == "Normal":
            st.success(f"✅ **{predicted_class_name}**")
        else:
            st.warning(f"⚠️ **{predicted_class_name} Tumor Detected**")
        
        st.metric("Ensemble Confidence", f"{ensemble_confidence:.2%}")
        
        # Confidence level
        if ensemble_confidence >= 0.90:
            st.success("🎯 **Very High Confidence**")
        elif ensemble_confidence >= 0.75:
            st.info("✅ **High Confidence**")
        elif ensemble_confidence >= 0.60:
            st.warning("⚠️ **Moderate Confidence**")
        else:
            st.error("❗ **Low Confidence** - Result uncertain")
        
        # Voting information
        st.write(f"**Majority Vote:** {CLASS_NAMES[majority_class]} ({vote_counts[majority_class]}/{len(models)} votes)")
    
    # Individual model predictions
    st.markdown("---")
    st.subheader("🤖 Individual Model Predictions")
    
    model_cols = st.columns(len(models))
    
    for idx, (model_name, pred) in enumerate(predictions.items()):
        with model_cols[idx]:
            pred_class = pred['predicted_class']
            confidence = pred['confidence']
            pred_class_name = CLASS_NAMES[pred_class]
            
            st.write(f"**{model_name}**")
            
            if pred_class_name == "Normal":
                st.success(f"**{pred_class_name}**")
            else:
                st.warning(f"**{pred_class_name}**")
            
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(float(confidence))
    
    # Ensemble probability breakdown
    st.markdown("---")
    st.subheader("📊 Ensemble Probability Breakdown")
    
    # Create columns for each class
    prob_cols = st.columns(NUM_CLASSES)
    
    for idx, class_name in enumerate(CLASS_NAMES):
        with prob_cols[idx]:
            st.metric(
                label=f"🧠 {class_name}",
                value=f"{avg_probs[idx]:.2%}"
            )
            st.progress(float(avg_probs[idx]))
    
    # Detailed breakdown table
    st.markdown("---")
    st.subheader("📋 Detailed Model Breakdown")
    
    import pandas as pd
    
    breakdown_data = []
    for model_name, pred in predictions.items():
        pred_class = pred['predicted_class']
        breakdown_data.append({
            'Model': model_name,
            'Prediction': CLASS_NAMES[pred_class],
            'Confidence': f"{pred['confidence']:.2%}",
            'Top 3 Predictions': ', '.join([
                f"{CLASS_NAMES[i]} ({pred['probabilities'][i]:.1%})" 
                for i in np.argsort(pred['probabilities'])[::-1][:3]
            ])
        })
    
    df = pd.DataFrame(breakdown_data)
    st.dataframe(df, use_container_width=True)
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    st.write(f"""
    The ensemble predicts this MRI scan as **{CLASS_NAMES[ensemble_class]}** with {ensemble_confidence:.2%} average confidence.
    
    **Voting Results:**
    """)
    
    for i, class_name in enumerate(CLASS_NAMES):
        if vote_counts[i] > 0:
            st.write(f"- **{class_name}**: {vote_counts[i]} out of {len(models)} models")
    
    if CLASS_NAMES[ensemble_class] == "Normal":
        st.info("""
        ℹ️ The models indicate **no tumor detected** in this MRI scan. 
        This means the brain tissue appears normal according to the trained models.
        """)
    else:
        st.warning(f"""
        ⚠️ The models detected a **{CLASS_NAMES[ensemble_class]} tumor**. 
        
        **About {CLASS_NAMES[ensemble_class]}:**
        """)
        
        if CLASS_NAMES[ensemble_class] == "Glioma":
            st.write("- Most common type of brain tumor originating from glial cells")
            st.write("- Can be benign or malignant")
        elif CLASS_NAMES[ensemble_class] == "Meningioma":
            st.write("- Tumor that forms on membranes covering the brain and spinal cord")
            st.write("- Usually benign and slow-growing")
        elif CLASS_NAMES[ensemble_class] == "Pituitary":
            st.write("- Tumor that forms in the pituitary gland")
            st.write("- Usually benign but can affect hormone production")
    
    st.write("""
    
    **Remember:** This ensemble combines three different architectures to provide 
    a more robust prediction, but it should NOT replace professional medical diagnosis.
    """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.write("**Ensemble Method:**")
        st.write("- Final Prediction: Based on average probabilities")
        st.write("- Majority Vote: Most common prediction across models")
        st.write("- Individual predictions are weighted equally")
        
        st.write("\n**Individual Model Predictions:**")
        for model_name, pred in predictions.items():
            st.write(f"- {model_name}: {CLASS_NAMES[pred['predicted_class']]} ({pred['confidence']:.4f})")
        
        st.write("\n**Average Probabilities (Ensemble):**")
        for idx, class_name in enumerate(CLASS_NAMES):
            st.write(f"- {class_name}: {avg_probs[idx]:.4f}")
        
        st.write("\n**Class Mapping:**")
        for class_name, idx in CLASS_INDICES.items():
            st.write(f"- {class_name.capitalize()}: Class {idx}")
        
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
    st.info("👆 Upload a brain MRI image using the file uploader above to get started.")
    
    st.markdown("---")
    
    # Two columns for information
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.subheader("📖 How to Use")
        st.write("""
        1. **Click** on "Browse files" above
        2. **Select** a brain MRI image from your computer
        3. **Wait** for all three models to analyze
        4. **View** the ensemble prediction results
        
        **Tips for Best Results:**
        - Use clear, high-quality MRI images
        - Ensure proper orientation
        - Brain MRI scans work best
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
        st.subheader("🧠 Tumor Types")
        
        st.write("**Glioma**")
        st.write("- Most common brain tumor type")
        st.write("- Originates from glial cells")
        
        st.write("\n**Meningioma**")
        st.write("- Forms on brain/spinal cord membranes")
        st.write("- Usually benign and slow-growing")
        
        st.write("\n**Pituitary**")
        st.write("- Forms in the pituitary gland")
        st.write("- Can affect hormone production")
        
        st.write("\n**Normal**")
        st.write("- No tumor detected")
        st.write("- Brain tissue appears healthy")
        
        st.subheader("🤖 The Three Models")
        st.write("""
        - **MobileNetV2**: Fast, efficient, mobile-optimized
        - **DenseNet121**: Dense connections, feature reuse
        - **ResNet50**: Deep architecture, residual learning
        
        Each model has been trained on brain MRI images
        and optimized for tumor classification.
        """)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <p>🧠 Ensemble Brain Tumor Classification System</p>
        <p><strong>MobileNetV2 + DenseNet121 + ResNet50</strong></p>
        <p><small>For Educational and Research Purposes Only • Not for Medical Diagnosis</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
