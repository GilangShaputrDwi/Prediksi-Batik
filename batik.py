# batik.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# --- Load model .sav ---
@st.cache_resource
def load_cnn_model():
    with open('cnn_batik_model.sav', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_cnn_model()

# Kelas sesuai urutan training
CLASS_NAMES = ['batik-bali', 'batik-megamendung', 'batik-parang', 'batik-sekarjagad']  # ganti sesuai dataset kamu

# --- Tampilan Streamlit ---
st.set_page_config(page_title="Prediksi Batik", page_icon="🧵", layout="centered")

st.title("🧵 Prediksi Motif Batik")
st.markdown("""
Upload gambar batik untuk memprediksi jenis motifnya menggunakan model CNN.
""")

# --- Upload gambar ---
uploaded_file = st.file_uploader("Pilih gambar batik", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    # Preprocessing
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.subheader(f"Prediksi: **{CLASS_NAMES[predicted_class]}**")
    st.write(f"Tingkat keyakinan: **{confidence*100:.2f}%**")

    # Tampilkan semua probabilitas
    st.markdown("### Probabilitas masing-masing kelas:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")
