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

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    predicted_breed = cat_breed_names[predicted_label]

    st.write(f"Prediksi: **{predicted_breed}**")