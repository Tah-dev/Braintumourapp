# ==========================================
# STREAMLIT APP - TWO-STAGE ENSEMBLE CLASSIFICATION
# Stage 1: Binary Classification (MRI vs Non-MRI)
# Stage 2: Multiclass Classification (Tumor Type)
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

# Stage 1: Binary Classification Models (MRI vs Non-MRI)
BINARY_MODEL_PATHS = {
    "MobileNetV2": "MobileNet_Binary_MRI_best.keras",
    "DenseNet121": "DenseNet121_Binary_MRI_best.keras",
    "ResNet50": "ResNet50_Binary_MRI_best.keras"
}

# Stage 2: Multiclass Classification Models (Tumor Types)
MULTICLASS_MODEL_PATHS = {
    "MobileNetV2": "MobileNetV2_Multiclass_best.keras",
    "DenseNet121": "DenseNet121_Multiclass_best.keras",
    "ResNet50": "ResNet50_Multiclass_best.keras"
}

# Binary class names
BINARY_CLASS_NAMES = {0: "MRI", 1: "Non-MRI"}

# Multiclass class names
MULTICLASS_CLASS_NAMES = ['Glioma', 'Meningioma', 'Normal', 'Pituitary']
MULTICLASS_CLASS_INDICES = {'glioma': 0, 'meningioma': 1, 'normal': 2, 'pituitary': 3}
NUM_CLASSES = len(MULTICLASS_CLASS_NAMES)

st.set_page_config(
    page_title="Two-Stage MRI Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Two-Stage Ensemble MRI Classification")
st.write("**Stage 1:** Verify if image is an MRI scan | **Stage 2:** Classify tumor type")
st.write(f"**6 Models Total** | **TensorFlow:** {tf.__version__}")

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
    img_array[..., 0] -= 103.939
    img_array[..., 1] -= 116.779
    img_array[..., 2] -= 123.68
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_all_models():
    """Load all 6 models (3 binary + 3 multiclass)"""
    binary_models = {}
    multiclass_models = {}
    binary_info = {}
    multiclass_info = {}
    
    # Load Binary Models
    st.sidebar.write("Loading Stage 1 models...")
    for model_name, model_path in BINARY_MODEL_PATHS.items():
        try:
            if not os.path.exists(model_path):
                st.warning(f"⚠️ Binary {model_name} not found: {model_path}")
                continue
            
            model = tf.keras.models.load_model(model_path, compile=False)
            binary_models[model_name] = model
            
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            binary_info[model_name] = {
                'file_size': file_size,
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            }
            
        except Exception as e:
            st.error(f"❌ Error loading Binary {model_name}: {str(e)}")
            continue
    
    # Load Multiclass Models
    st.sidebar.write("Loading Stage 2 models...")
    for model_name, model_path in MULTICLASS_MODEL_PATHS.items():
        try:
            if not os.path.exists(model_path):
                st.warning(f"⚠️ Multiclass {model_name} not found: {model_path}")
                continue
            
            model = tf.keras.models.load_model(model_path, compile=False)
            multiclass_models[model_name] = model
            
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            multiclass_info[model_name] = {
                'file_size': file_size,
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            }
            
        except Exception as e:
            st.error(f"❌ Error loading Multiclass {model_name}: {str(e)}")
            continue
    
    if len(binary_models) == 0:
        st.error("❌ No binary models loaded! Cannot proceed.")
        st.stop()
    
    return binary_models, multiclass_models, binary_info, multiclass_info

# Load all models
binary_models, multiclass_models, binary_info, multiclass_info = load_all_models()

# ----------------------------
# SIDEBAR - MODEL INFO
# ----------------------------
with st.sidebar:
    st.header("🤖 Two-Stage Pipeline")
    
    st.subheader("Stage 1: Binary Classification")
    if len(binary_models) == 3:
        st.success(f"✅ All 3 binary models loaded")
    else:
        st.warning(f"⚠️ {len(binary_models)}/3 binary models loaded")
    
    st.subheader("Stage 2: Multiclass Classification")
    if len(multiclass_models) == 3:
        st.success(f"✅ All 3 multiclass models loaded")
    else:
        st.warning(f"⚠️ {len(multiclass_models)}/3 multiclass models loaded")
    
    st.divider()
    
    with st.expander("📊 Binary Models (Stage 1)"):
        for model_name in binary_models.keys():
            info = binary_info[model_name]
            st.write(f"**{model_name}**")
            st.write(f"Size: {info['file_size']:.2f} MB")
    
    with st.expander("📊 Multiclass Models (Stage 2)"):
        for model_name in multiclass_models.keys():
            info = multiclass_info[model_name]
            st.write(f"**{model_name}**")
            st.write(f"Size: {info['file_size']:.2f} MB")
    
    st.divider()
    
    st.header("ℹ️ How It Works")
    st.write("""
    **Stage 1: MRI Detection**
    - Checks if the image is an MRI scan
    - Uses 3 binary classification models
    - If NOT MRI → Stop processing
    
    **Stage 2: Tumor Classification**
    - Only runs if Stage 1 detects MRI
    - Classifies tumor type
    - Uses 3 multiclass models
    """)
    
    st.divider()
    
    st.header("🧠 Tumor Types")
    st.write("- **Glioma**: Most common brain tumor")
    st.write("- **Meningioma**: Benign, slow-growing")
    st.write("- **Pituitary**: Affects hormone production")
    st.write("- **Normal**: No tumor detected")
    
    st.divider()
    
    st.header("⚠️ Disclaimer")
    st.error("""
    **Educational purposes only.**
    
    NOT for medical diagnosis.
    Consult healthcare professionals.
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
    
    # ==========================================
    # STAGE 1: BINARY CLASSIFICATION (MRI vs Non-MRI)
    # ==========================================
    
    st.markdown("---")
    st.header("🔍 Stage 1: MRI Detection")
    
    with st.spinner("Running binary classification..."):
        time.sleep(0.3)
        
        binary_predictions = {}
        preprocessing_functions = {
            "MobileNetV2": preprocess_mobilenet,
            "DenseNet121": preprocess_densenet,
            "ResNet50": preprocess_resnet
        }
        
        try:
            for model_name, model in binary_models.items():
                preprocess_func = preprocessing_functions[model_name]
                img_input = preprocess_func(image)
                
                pred = model.predict(img_input, verbose=0)
                non_mri_prob = float(pred[0][0])
                mri_prob = 1.0 - non_mri_prob
                
                binary_predictions[model_name] = {
                    'mri_prob': mri_prob,
                    'non_mri_prob': non_mri_prob,
                    'predicted_class': 0 if mri_prob >= 0.5 else 1
                }
        
        except Exception as e:
            st.error(f"❌ Stage 1 Error: {str(e)}")
            st.stop()
    
    # Calculate Stage 1 ensemble results
    avg_mri_prob = np.mean([p['mri_prob'] for p in binary_predictions.values()])
    avg_non_mri_prob = np.mean([p['non_mri_prob'] for p in binary_predictions.values()])
    
    binary_votes = [p['predicted_class'] for p in binary_predictions.values()]
    binary_ensemble_class = 0 if binary_votes.count(0) > binary_votes.count(1) else 1
    binary_confidence = max(avg_mri_prob, avg_non_mri_prob)
    
    # Display Stage 1 Results
    with col2:
        st.subheader("Stage 1 Result")
        
        if binary_ensemble_class == 0:
            st.success("✅ **MRI IMAGE DETECTED**")
            st.metric("Confidence", f"{avg_mri_prob:.2%}")
            st.info("✓ Proceeding to Stage 2: Tumor Classification")
        else:
            st.error("🚫 **NON-MRI IMAGE DETECTED**")
            st.metric("Confidence", f"{avg_non_mri_prob:.2%}")
            st.warning("⚠️ Stage 2 will NOT run - Not an MRI scan")
        
        st.write(f"**Voting:** {binary_votes.count(0)} MRI, {binary_votes.count(1)} Non-MRI")
    
    # Stage 1 Individual Predictions
    st.subheader("Stage 1: Individual Model Predictions")
    
    binary_cols = st.columns(len(binary_models))
    
    for idx, (model_name, pred) in enumerate(binary_predictions.items()):
        with binary_cols[idx]:
            st.write(f"**{model_name}**")
            if pred['predicted_class'] == 0:
                st.success("MRI")
                st.progress(pred['mri_prob'])
                st.write(f"{pred['mri_prob']:.2%}")
            else:
                st.error("Non-MRI")
                st.progress(pred['non_mri_prob'])
                st.write(f"{pred['non_mri_prob']:.2%}")
    
    # ==========================================
    # STAGE 2: MULTICLASS CLASSIFICATION (Tumor Type)
    # Only runs if Stage 1 detected MRI
    # ==========================================
    
    if binary_ensemble_class == 0:  # MRI detected
        st.markdown("---")
        st.header("🧠 Stage 2: Tumor Classification")
        
        with st.spinner("Running tumor classification..."):
            time.sleep(0.3)
            
            multiclass_predictions = {}
            
            try:
                for model_name, model in multiclass_models.items():
                    preprocess_func = preprocessing_functions[model_name]
                    img_input = preprocess_func(image)
                    
                    pred_probs = model.predict(img_input, verbose=0)[0]
                    pred_class = np.argmax(pred_probs)
                    
                    multiclass_predictions[model_name] = {
                        'probabilities': pred_probs,
                        'predicted_class': pred_class,
                        'confidence': pred_probs[pred_class]
                    }
            
            except Exception as e:
                st.error(f"❌ Stage 2 Error: {str(e)}")
                st.stop()
        
        # Calculate Stage 2 ensemble results
        avg_probs = np.mean([p['probabilities'] for p in multiclass_predictions.values()], axis=0)
        multiclass_ensemble_class = np.argmax(avg_probs)
        multiclass_confidence = avg_probs[multiclass_ensemble_class]
        
        multiclass_votes = [p['predicted_class'] for p in multiclass_predictions.values()]
        vote_counts = np.bincount(multiclass_votes, minlength=NUM_CLASSES)
        majority_class = np.argmax(vote_counts)
        
        # Display Stage 2 Results
        st.subheader("🎯 Stage 2 Result")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            predicted_tumor = MULTICLASS_CLASS_NAMES[multiclass_ensemble_class]
            
            if predicted_tumor == "Normal":
                st.success(f"✅ **{predicted_tumor}**")
                st.write("No tumor detected in the MRI scan")
            else:
                st.warning(f"⚠️ **{predicted_tumor} Tumor Detected**")
            
            st.metric("Confidence", f"{multiclass_confidence:.2%}")
            
            if multiclass_confidence >= 0.90:
                st.success("🎯 Very High Confidence")
            elif multiclass_confidence >= 0.75:
                st.info("✅ High Confidence")
            elif multiclass_confidence >= 0.60:
                st.warning("⚠️ Moderate Confidence")
            else:
                st.error("❗ Low Confidence")
        
        with result_col2:
            st.write("**Voting Results:**")
            for i, class_name in enumerate(MULTICLASS_CLASS_NAMES):
                if vote_counts[i] > 0:
                    st.write(f"- {class_name}: {vote_counts[i]}/{len(multiclass_models)} votes")
        
        # Stage 2 Individual Predictions
        st.subheader("Stage 2: Individual Model Predictions")
        
        multi_cols = st.columns(len(multiclass_models))
        
        for idx, (model_name, pred) in enumerate(multiclass_predictions.items()):
            with multi_cols[idx]:
                pred_class = pred['predicted_class']
                confidence = pred['confidence']
                pred_class_name = MULTICLASS_CLASS_NAMES[pred_class]
                
                st.write(f"**{model_name}**")
                
                if pred_class_name == "Normal":
                    st.success(f"**{pred_class_name}**")
                else:
                    st.warning(f"**{pred_class_name}**")
                
                st.write(f"{confidence:.2%}")
                st.progress(float(confidence))
        
        # Stage 2 Probability Breakdown
        st.subheader("Stage 2: Probability Breakdown")
        
        prob_cols = st.columns(NUM_CLASSES)
        
        for idx, class_name in enumerate(MULTICLASS_CLASS_NAMES):
            with prob_cols[idx]:
                st.metric(
                    label=f"{class_name}",
                    value=f"{avg_probs[idx]:.2%}"
                )
                st.progress(float(avg_probs[idx]))
        
        # Tumor Information
        if predicted_tumor != "Normal":
            st.markdown("---")
            st.subheader(f"ℹ️ About {predicted_tumor}")
            
            if predicted_tumor == "Glioma":
                st.write("- Most common type of brain tumor originating from glial cells")
                st.write("- Can be benign or malignant")
                st.write("- Requires medical evaluation and treatment planning")
            elif predicted_tumor == "Meningioma":
                st.write("- Tumor forming on membranes covering brain and spinal cord")
                st.write("- Usually benign and slow-growing")
                st.write("- May require monitoring or surgical intervention")
            elif predicted_tumor == "Pituitary":
                st.write("- Tumor forming in the pituitary gland")
                st.write("- Usually benign but can affect hormone production")
                st.write("- May cause vision problems or hormonal imbalances")
    
    else:  # Non-MRI detected in Stage 1
        st.markdown("---")
        st.info("""
        ℹ️ **Stage 2 Skipped**
        
        The image was classified as **Non-MRI** in Stage 1, so tumor classification 
        was not performed. Please upload an MRI scan to use the full two-stage pipeline.
        """)
    
    # ==========================================
    # PIPELINE SUMMARY
    # ==========================================
    
    st.markdown("---")
    st.header("📊 Pipeline Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.subheader("Stage 1: MRI Detection")
        st.write(f"**Result:** {BINARY_CLASS_NAMES[binary_ensemble_class]}")
        st.write(f"**Confidence:** {binary_confidence:.2%}")
        st.write(f"**Models Used:** {len(binary_models)}")
    
    with summary_col2:
        st.subheader("Stage 2: Tumor Classification")
        if binary_ensemble_class == 0:
            st.write(f"**Result:** {MULTICLASS_CLASS_NAMES[multiclass_ensemble_class]}")
            st.write(f"**Confidence:** {multiclass_confidence:.2%}")
            st.write(f"**Models Used:** {len(multiclass_models)}")
        else:
            st.write("**Status:** Skipped (Not an MRI)")
            st.write("**Reason:** Stage 1 detected Non-MRI")
    
    # Technical Details
    with st.expander("🔧 Technical Details"):
        st.write("**Two-Stage Pipeline:**")
        st.write("1. Stage 1 (Binary): Validates image is MRI scan")
        st.write("2. Stage 2 (Multiclass): Only runs if Stage 1 detects MRI")
        st.write("3. Each stage uses 3 models (MobileNetV2, DenseNet121, ResNet50)")
        
        st.write("\n**Stage 1 - Binary Classification:**")
        st.write(f"- MRI Probability: {avg_mri_prob:.4f}")
        st.write(f"- Non-MRI Probability: {avg_non_mri_prob:.4f}")
        st.write(f"- Final Decision: {BINARY_CLASS_NAMES[binary_ensemble_class]}")
        
        if binary_ensemble_class == 0:
            st.write("\n**Stage 2 - Multiclass Classification:**")
            for idx, class_name in enumerate(MULTICLASS_CLASS_NAMES):
                st.write(f"- {class_name}: {avg_probs[idx]:.4f}")
        
        st.write("\n**Preprocessing:**")
        st.write("- MobileNetV2: Scale to [-1, 1]")
        st.write("- DenseNet121: Subtract mean RGB")
        st.write("- ResNet50: Subtract mean RGB")
        
        st.write("\n**Total Models:** 6 (3 binary + 3 multiclass)")

# ----------------------------
# INSTRUCTIONS (when no image uploaded)
# ----------------------------
else:
    st.info("👆 Upload an image to begin the two-stage classification pipeline.")
    
    st.markdown("---")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.subheader("📖 How to Use")
        st.write("""
        1. **Upload** an image (MRI or regular photo)
        2. **Stage 1** verifies if it's an MRI scan
        3. **Stage 2** runs ONLY if MRI is detected
        4. **View** comprehensive results from 6 models
        """)
        
        st.subheader("🔄 Two-Stage Pipeline")
        st.write("""
        **Stage 1: MRI Detection**
        - 3 Binary classification models
        - Separates MRI from Non-MRI images
        - Acts as a gatekeeper
        
        **Stage 2: Tumor Classification**
        - 3 Multiclass models
        - Only runs if Stage 1 detects MRI
        - Classifies tumor type (or Normal)
        """)
    
    with info_col2:
        st.subheader("🎯 Why Two Stages?")
        st.write("""
        **Efficiency:**
        - Saves computation on non-MRI images
        - Stage 2 only processes relevant images
        
        **Accuracy:**
        - Specialized models for each task
        - Better performance than single model
        
        **Reliability:**
        - 6 models provide robust predictions
        - Ensemble voting reduces errors
        """)
        
        st.subheader("✅ Will Detect as MRI")
        st.write("- Brain MRI scans")
        st.write("- Spine MRI images")
        st.write("- Any medical MRI scan")
        
        st.subheader("❌ Will Detect as Non-MRI")
        st.write("- Regular photos")
        st.write("- X-rays, CT scans")
        st.write("- Everyday objects")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <p>🧠 Two-Stage Ensemble MRI Classification System</p>
        <p><strong>Stage 1: Binary (3 Models) → Stage 2: Multiclass (3 Models)</strong></p>
        <p><small>For Educational and Research Purposes Only • Not for Medical Diagnosis</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
