import os
import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Klasifikasi Breed Kucing",
    page_icon="🐱",
    layout="centered",
    menu_items=None,
)

st.markdown(
    """<style>
            .stMarkdown h3,h4{text-align: center;},
            [data-testid="stFullScreenFrame"]{padding:0rem; margin:0rem;}
            </style>""",
    unsafe_allow_html=True,
)

tf_enable_onednn_opts = os.getenv("TF_ENABLE_ONEDNN_OPTS", "0")


# Daftar Nama Breed Kucing
cat_breed_names = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
]

IMG_SIZE = 160


def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan batch dimension
    return image


# UI Streamlit

col1, col2 = st.columns([1, 3])
with col1:
    st.image(
        "./assets/images.png",
        use_container_width=True,
    )
with col2:
    st.title("Klasifikasi Breed Kucing")

    st.markdown(
        "> Aplikasi ini menggunakan model **CNN** untuk mengklasifikasikan **breed kucing** berdasarkan gambar kucing yang diunggah."
    )


uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png", "jpeg"])
st.markdown(
    """
    ---
    """
)

if uploaded_file is not None:
    # Cek format file berdasarkan ekstensi
    file_extension = uploaded_file.name.split(".")[-1].lower()
    allowed_extensions = ["jpg", "png", "jpeg"]

    if file_extension not in allowed_extensions:
        st.error(
            "❌ Format file tidak didukung! Harap unggah file dengan format JPG, PNG, atau JPEG."
        )
    else:
        # load model hanya saat akan digunakan dan masukkan ke session state sehingga tidak perlu memuat ulang setiap kali dipanggil
        if "model" not in st.session_state:
            st.session_state["model"] = tf.keras.models.load_model("model.h5")
        model = st.session_state["model"]
        # Jika format benar, lanjutkan ke proses prediksi
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("memprediksi gambar :cat2: :black_cat: ..."):
            time.sleep(2)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            max_prob = np.max(prediction)  # Probabilitas tertinggi dari prediksi

        if max_prob < 0.7:  # Threshold 70% untuk validasi
            st.warning(
                "❌ Model tidak yakin ini adalah kucing! Coba unggah gambar lain."
            )
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        else:
            predicted_label = np.argmax(prediction)
            predicted_breed = cat_breed_names[predicted_label]
            st.success("✅ Kucing terdeteksi!")
            st.markdown(
                "### **Breed yang terdeteksi:**",
                unsafe_allow_html=True,
            )
            st.markdown(f"### **{predicted_breed}**")
            st.markdown(f"#### Probabilitas: {max_prob*100:.2f}%")

            st.image(image, caption="Gambar yang diunggah", use_container_width=True)


st.markdown(
    """
    ---
    """
)
st.header("Kelompok 4 ")
st.code(
    """Nama Anggota: \n1.Alfi Khairani (2802562075) \n2.Arief Rahmat Nugraha (2802537735) \n3.Muhammad Ihsan Firzatullah Simbolon (2802518572) \n4.Muhammad Lutfi (2802535521) \n5.Sugeng Wahyudi (2802550541)
    """
)
