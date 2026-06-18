"""
Streamlit App — Brain Tumour Binary Classifier
============================================================
EfficientNetB0 binary classification: tumour present (yes) vs
not present (no). Retrained model, saved directly as .h5.

Model hosted on Hugging Face (NdahTah/BrainTumor, public repo),
downloaded at runtime via huggingface_hub — same pattern used
for the dementia diagnosis apps.

Class mapping (confirmed at training time):
  no   -> 0   (no tumour)
  yes  -> 1   (tumour present)
  sigmoid output = P(yes / tumour present)
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download

# ── Config ──────────────────────────────────────────────────────────────

HF_REPO_ID = "NdahTah/BrainTumor"
MODEL_FILENAME = "efficientnetb0_braintumor_binary_best.h5"

INPUT_SIZE = (224, 224)
THRESHOLD = 0.5

st.set_page_config(page_title="Brain Tumour Classifier", page_icon="🧠", layout="centered")

# ── Model loading (cached so download/load happens once per session) ───

@st.cache_resource
def load_tumor_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        cache_dir="./model_cache",
    )
    return load_model(model_path, compile=False)


try:
    model = load_tumor_model()
    model_load_error = None
except Exception as e:
    model = None
    model_load_error = str(e)

# ── UI ──────────────────────────────────────────────────────────────────

st.title("🧠 Brain Tumour Classifier")
st.write("Upload a brain MRI image to check for the presence of a tumour.")

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

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array, verbose=0)
        probability_yes = float(prediction[0][0])   # P(tumour present)
        probability_no = 1 - probability_yes

    predicted_class = "Tumour Present" if probability_yes > THRESHOLD else "No Tumour"
    confidence = probability_yes if probability_yes > THRESHOLD else probability_no

    st.subheader("Result")

    if probability_yes > THRESHOLD:
        st.error(f"**{predicted_class}**")
    else:
        st.success(f"**{predicted_class}**")

    st.write(f"Confidence: {confidence:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tumour probability", f"{probability_yes:.2%}")
    with col2:
        st.metric("No tumour probability", f"{probability_no:.2%}")

    with st.expander("How to read this"):
        st.write(
            "The model outputs a single probability representing the "
            "likelihood a tumour is present. Values above 50% are "
            "classified as tumour present; values below are classified "
            "as no tumour."
        )

st.markdown("---")
st.caption("Built with Streamlit • Model hosted on Hugging Face")
