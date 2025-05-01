import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Muat Model CNN yang sudah tersimpan
model = tf.keras.models.load_model("model.h5")

# Daftar Nama Breed Kucing
cat_breed_names = [
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "British_Shorthair", "Egyptian_Mau", "Maine_Coon",
    "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx"
]

IMG_SIZE = 160

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan batch dimension
    return image

# UI Streamlit
st.title("Klasifikasi Breed Kucing")
st.write("Unggah gambar kucing untuk mendapatkan prediksi breed-nya.")

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Cek format file berdasarkan ekstensi
    file_extension = uploaded_file.name.split(".")[-1].lower()
    allowed_extensions = ["jpg", "png", "jpeg"]

    if file_extension not in allowed_extensions:
        st.error("❌ Format file tidak didukung! Harap unggah file dengan format JPG, PNG, atau JPEG.")
    else:
        # Jika format benar, lanjutkan ke proses prediksi
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        max_prob = np.max(prediction)  # Probabilitas tertinggi dari prediksi

        if max_prob < 0.7:  # Threshold 70% untuk validasi
            st.warning("❌ Model tidak yakin ini adalah kucing! Coba unggah gambar lain.")
        else:
            predicted_label = np.argmax(prediction)
            predicted_breed = cat_breed_names[predicted_label]

            st.write(f"Prediksi: **{predicted_breed}**")