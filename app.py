# Library heart disease prediction
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import base64
from io import BytesIO
from PIL import Image


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — biar tampilan lebih rapi & menarik
# =========================================================
st.markdown("""
<style>
    /* Header utama */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff8a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 16px;
        color: #b0b0b0;
        margin-top: -10px;
    }
    .author {
        font-size: 14px;
        color: #888;
    }

    /* Card untuk gambar */
    .img-card {
        background-color: #1e1e26;
        border-radius: 16px;
        padding: 14px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }
    .img-card:hover {
        transform: translateY(-4px);
    }
    .img-card img {
        border-radius: 12px;
        width: 100%;
        height: auto;
        object-fit: cover;
    }
    .img-caption {
        margin-top: 8px;
        font-size: 14px;
        font-weight: 600;
        color: #e0e0e0;
    }

    /* Section header */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 6px;
        border-left: 5px solid #ff4b4b;
        padding-left: 10px;
    }

    /* Hasil prediksi */
    .result-box-positive {
        background-color: rgba(255, 75, 75, 0.12);
        border: 1px solid #ff4b4b;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        color: #ff6b6b;
    }
    .result-box-negative {
        background-color: rgba(75, 200, 120, 0.12);
        border: 1px solid #4bc878;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        color: #4bc878;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNGSI BANTUAN
# =========================================================
def image_to_base64(img: Image.Image, size=None) -> str:
    """Konversi gambar PIL ke base64 agar bisa dibungkus HTML/CSS custom."""
    if size:
        img = img.resize(size)
    buffered = BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode()


def show_image_card(image_path, caption, size=(280, 200)):
    """Tampilkan gambar dalam card rounded yang lebih menarik."""
    img = Image.open(image_path)
    img_b64 = image_to_base64(img, size)
    st.markdown(f"""
        <div class="img-card">
            <img src="data:image/jpeg;base64,{img_b64}" />
            <div class="img-caption">{caption}</div>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">❤️ Dashboard Prediksi Penyakit Jantung</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="author">Dibuat oleh: '
    '<a href="https://www.linkedin.com/in/apriyanto19/" target="_blank">Apriyanto</a></div>',
    unsafe_allow_html=True
)
st.write("")

st.markdown("""
<div class="sub-title">
Aplikasi ini memprediksi <b>potensi penyakit jantung</b> berdasarkan data klinis pasien
menggunakan model Machine Learning. Dataset training berasal dari
<a href="https://archive.ics.uci.edu/dataset/45/heart+disease" target="_blank">Heart Disease Dataset (UCI ML)</a>.
</div>
""", unsafe_allow_html=True)

st.write("---")


# =========================================================
# GAMBAR ILUSTRASI (man / woman heart attack)
# =========================================================
st.markdown('<div class="section-header">⚠️ Kenali Gejala Sakit Jantung</div>', unsafe_allow_html=True)
st.write("Nyeri dada yang menjalar dan tekanan di area dada bisa menjadi tanda peringatan penyakit jantung, baik pada pria maupun wanita.")

col1, col2 = st.columns(2)
with col1:
    show_image_card("public/man-heart-attack.jpg", "Gejala pada Pria")
with col2:
    show_image_card("public/woman-heart-attack.jpg", "Gejala pada Wanita")

st.write("---")


# =========================================================
# SIDEBAR — INPUT DATA
# =========================================================
st.sidebar.markdown("## 🩺 Menu Aplikasi")
add_selectitem = st.sidebar.selectbox("Pilih Tools:", ("Prediksi Penyakit Jantung",))

st.sidebar.markdown("---")
st.sidebar.header("📥 INPUT DATA")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        st.sidebar.markdown("### ✍️ Input Manual")

        sex = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
        sex = 0 if sex == "Perempuan" else 1

        age = st.sidebar.slider("Usia", 20, 100, 50)

        cp = st.sidebar.slider('Jenis Nyeri Dada (cp)', 1, 4, 1)
        if cp == 1:
            wcp = "Nyeri dada tipe angina"
        elif cp == 2:
            wcp = "Nyeri dada tipe nyeri tidak stabil"
        elif cp == 3:
            wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
        else:
            wcp = "Nyeri dada tidak terkait dengan masalah jantung"
        st.sidebar.caption(f"ℹ️ {wcp}")

        thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", 71, 202, 80)
        slope = st.sidebar.slider("Kemiringan segmen ST (slope)", 0, 2, 1)
        oldpeak = st.sidebar.slider("Depresi segmen ST (oldpeak)", 0.0, 6.2, 1.0)
        
        exang = st.sidebar.slider("Exercise Induced Angina (exang)", 0, 1, 1)
        if exang == 0:
            wexang = "Tidak"
        else:
            wexang = "Ya"
        st.sidebar.caption(f"ℹ️ {wexang}")
        
        ca = st.sidebar.slider("Jumlah Pembuluh Darah Utama (ca)", 0, 3, 1)
        
        thal = st.sidebar.slider("Hasil Tes Thalium (thal)", 1, 3, 1)
        if thal == 1:
            wthal = "Normal"
        elif thal == 2
            wthal = "Ada defek tetap"
        else:
            wthal = "Ada defek dapat dipulihkan"
        st.sidebar.caption(f"ℹ️ {wthal}")
    

        data = {
            'cp': cp,
            'thalach': thalach,
            'slope': slope,
            'oldpeak': oldpeak,
            'exang': exang,
            'ca': ca,
            'thal': thal,
            'sex': sex,
            'age': age
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Lakukan Prediksi", use_container_width=True)


# =========================================================
# DATA INPUT PREVIEW
# =========================================================
loaded_model = None

if predict_btn:
    df = input_df
    st.markdown('<div class="section-header">📋 Data Input</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    with open("model/best_model.pkl", 'rb') as file:
        loaded_model = pickle.load(file)


# =========================================================
# HASIL PREDIKSI
# =========================================================
if loaded_model is not None:
    prediction = loaded_model.predict(df)

    st.markdown('<div class="section-header">🎯 Hasil Prediksi</div>', unsafe_allow_html=True)

    with st.spinner('Sedang memproses prediksi...'):
        time.sleep(2)

    if (prediction == 0).any():
        st.markdown('<div class="result-box-negative">✅ TIDAK TERINDIKASI PENYAKIT JANTUNG</div>', unsafe_allow_html=True)
        result_img = "public/strong-heart.jpg"
    else:
        st.markdown('<div class="result-box-positive">⚠️ ADA POTENSI PENYAKIT JANTUNG</div>', unsafe_allow_html=True)
        result_img = "public/heart-disease.jpg"

    st.write("")
    img_col, _ = st.columns([1, 2])
    with img_col:
        show_image_card(result_img, "")
