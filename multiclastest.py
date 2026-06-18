"""
Streamlit App — Brain Tumour Multiclass Classifier
============================================================
EfficientNetB0, 4-class: glioma / meningioma / notumor / pituitary.

Model hosted on Hugging Face (NdahTah/BrainTumor, public repo),
downloaded at runtime via huggingface_hub — same pattern used for
the binary tumour and dementia apps.

Class mapping (confirmed at training time):
    glioma      -> 0
    meningioma  -> 1
    notumor     -> 2
    pituitary   -> 3
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download

# ── Config ──────────────────────────────────────────────────────────────

HF_REPO_ID = "NdahTah/BrainTumor"
MODEL_FILENAME = "efficientnetb0_braintumor_multiclass_best.h5"

INPUT_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

st.set_page_config(page_title="Brain Tumour Type Classifier", page_icon="🧠", layout="centered")

# ── Model loading (cached so download/load happens once per session) ───

@st.cache_resource
def load_tumor_multiclass_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        cache_dir="./model_cache",
    )
    return load_model(model_path, compile=False)


try:
    model = load_tumor_multiclass_model()
    model_load_error = None
except Exception as e:
    model = None
    model_load_error = str(e)

# ── UI ──────────────────────────────────────────────────────────────────

st.title("🧠 Brain Tumour Type Classifier")
st.write(
    "Upload a brain MRI image to classify the tumour type: "
    "**Glioma**, **Meningioma**, **No Tumor**, or **Pituitary**."
)

if model is None:
    st.error(f"Could not load model from Hugging Face.\n\nDetails: {model_load_error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_rgb = image.convert("RGB")

    st.image(image_rgb, caption="Uploaded image", use_container_width=True)

    resized = image_rgb.resize(INPUT_SIZE)
    img_array = np.array(resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    with st.spinner("Classifying..."):
        prediction = model.predict(img_array, verbose=0)[0]  # shape: (4,)

    predicted_idx = int(np.argmax(prediction))
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = float(prediction[predicted_idx])

    st.subheader("Result")

    if predicted_idx == 2:   # No Tumor
        st.success(f"**{predicted_label}**")
    else:
        st.error(f"**{predicted_label}**")

    st.write(f"Confidence: {confidence:.2%}")

    with st.expander("Full probability breakdown"):
        for class_name, prob in zip(CLASS_NAMES, prediction):
            st.write(f"{class_name}: {prob:.2%}")
            st.progress(float(prob))

    with st.expander("How to read this"):
        st.write(
            "The model outputs a probability for each of the four "
            "categories, and the highest-probability class is shown "
            "as the result."
        )

st.markdown("---")
st.caption("Built with Streamlit • Model hosted on Hugging Face")
