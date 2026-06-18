"""
Streamlit App — Sequential Brain Tumour Diagnosis Pipeline
============================================================
Stage 1: Tumour presence gate (EfficientNetB0, binary: yes/no)
Stage 2: Tumour type classification (EfficientNetB0, 4-class)
          — only runs if Stage 1 detects a tumour (yes)

Both models hosted on Hugging Face (NdahTah/BrainTumor, public
repo) and downloaded at runtime via huggingface_hub.

Class mappings (confirmed at training time):
  Stage 1 — Tumour gate:
    no   -> 0   (no tumour)
    yes  -> 1   (tumour present)
    sigmoid output = P(yes / tumour present)

  Stage 2 — Tumour type:
    glioma      -> 0
    meningioma  -> 1
    notumor     -> 2   (not reachable in practice — Stage 1 already
                         filters these out, kept for completeness)
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
BINARY_FILENAME = "efficientnetb0_braintumor_binary_best.h5"
MULTICLASS_FILENAME = "efficientnetb0_braintumor_multiclass_best.h5"

INPUT_SIZE = (224, 224)
TUMOR_GATE_THRESHOLD = 0.5   # P(yes/tumour present) above this = passes gate

TUMOR_TYPE_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

st.set_page_config(page_title="Brain Tumour Diagnosis Pipeline", page_icon="🧠", layout="centered")

# ── Model loading (cached so downloads/loads happen once per session) ───

@st.cache_resource
def load_binary_gate_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=BINARY_FILENAME,
        cache_dir="./model_cache",
    )
    return load_model(model_path, compile=False)


@st.cache_resource
def load_multiclass_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MULTICLASS_FILENAME,
        cache_dir="./model_cache",
    )
    return load_model(model_path, compile=False)


def preprocess_image(image: Image.Image) -> tuple:
    """Shared preprocessing for both models — same training-time pipeline."""
    image_rgb = image.convert("RGB")
    resized = image_rgb.resize(INPUT_SIZE)
    img_array = np.array(resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, image_rgb


# ── UI ──────────────────────────────────────────────────────────────────

st.title("🧠 Brain Tumour Diagnosis Pipeline")
st.write(
    "Upload a brain MRI image. The image is first checked for tumour "
    "presence — if a tumour is detected, its type is then classified as "
    "**Glioma**, **Meningioma**, or **Pituitary**."
)

try:
    with st.spinner("Loading models from Hugging Face (first run only)..."):
        binary_gate_model = load_binary_gate_model()
        multiclass_model = load_multiclass_model()
except Exception as e:
    st.error(f"Could not load models from Hugging Face.\n\nDetails: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array, image_rgb = preprocess_image(image)

    st.image(image_rgb, caption="Uploaded image", use_container_width=True)

    # ── Stage 1: Tumour Presence Gate ────────────────────────────────

    st.subheader("Stage 1 — Tumour Detection")

    with st.spinner("Checking for tumour presence..."):
        gate_prediction = binary_gate_model.predict(img_array, verbose=0)
        probability_yes = float(gate_prediction[0][0])   # P(tumour present)
        probability_no = 1 - probability_yes

    tumor_detected = probability_yes > TUMOR_GATE_THRESHOLD

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tumour probability", f"{probability_yes:.2%}")
    with col2:
        st.metric("No tumour probability", f"{probability_no:.2%}")

    if tumor_detected:
        st.error("⚠ Tumour detected — proceeding to type classification.")
    else:
        st.success("✓ No tumour detected.")
        st.stop()

    # ── Stage 2: Tumour Type Classification (only if tumour detected) ──

    st.subheader("Stage 2 — Tumour Type Classification")

    with st.spinner("Classifying tumour type..."):
        type_prediction = multiclass_model.predict(img_array, verbose=0)[0]  # shape: (4,)

    predicted_idx = int(np.argmax(type_prediction))
    predicted_label = TUMOR_TYPE_NAMES[predicted_idx]
    confidence = float(type_prediction[predicted_idx])

    st.error(f"**{predicted_label}**")
    st.write(f"Confidence: {confidence:.2%}")

    with st.expander("Full probability breakdown"):
        for class_name, prob in zip(TUMOR_TYPE_NAMES, type_prediction):
            st.write(f"{class_name}: {prob:.2%}")
            st.progress(float(prob))

with st.expander("ℹ️ How this pipeline works"):
    st.markdown(
        """
        **Stage 1 — Tumour Detection:** confirms whether a tumour is
        present in the uploaded MRI before attempting to classify its
        type.

        **Stage 2 — Tumour Type Classification:** only runs if Stage 1
        detects a tumour. Classifies it as Glioma, Meningioma, or
        Pituitary.

        Both models are EfficientNetB0, hosted on Hugging Face and
        downloaded automatically on first use.
        """
    )

st.markdown("---")
st.caption("Built with Streamlit • Models hosted on Hugging Face")
