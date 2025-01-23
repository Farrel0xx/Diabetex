import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Model dan Dataset
diabetes_model = joblib.load("model/diabetes_model.sav")
df = pd.read_csv('diabetes_dataset_clean.csv')

X = df.drop(columns='diabetes', axis=1)
scaler = StandardScaler()

# Konfigurasi Tampilan
st.set_page_config(
    page_title="APD: Aplikasi Prediksi Diabetes",
    layout="wide",
    page_icon="💉"
)

# Judul Halaman
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>APD 💉<br>Aplikasi Prediksi Diabetes</h1>",
    unsafe_allow_html=True
)

st.markdown("<p style='text-align: center; color: #555;'>Masukkan data di bawah ini untuk mengetahui prediksi kondisi diabetes Anda!</p>", unsafe_allow_html=True)

# Input Form
col1, col2 = st.columns(2)

with col1:
    Gender = st.radio('💁‍♀️ Jenis Kelamin:', ['Perempuan', 'Laki-laki'])
    Age = st.number_input("📅 Umur Anda:", min_value=1, max_value=120, step=1)
    Hipertension = st.radio('🩺 Hipertensi:', ['Tidak', 'Ya'])
    Heart_disease = st.radio('❤️ Penyakit Jantung:', ['Tidak', 'Ya'])

with col2:
    Smoking_history = st.selectbox(
        "🚬 Riwayat Merokok:", 
        ['Tidak Ada Info', 'Tidak Pernah', 'Mantan Perokok', 'Sedang Merokok', 'Tidak Saat Ini', 'Pernah Merokok']
    )
    bmi = st.number_input('📊 Indeks Massa Tubuh (BMI):', min_value=10.0, max_value=100.0, format="%.1f")
    HbA1c_level = st.number_input("🩸 Level Hemoglobin A1c:", min_value=4.0, max_value=20.0, format="%.1f")
    Blood_glucose = st.number_input('🩸 Level Glukosa Darah:', min_value=50.0, max_value=500.0, format="%.1f")

# Mapping nilai input
gender_map = {'Perempuan': 0, 'Laki-laki': 1}
binary_map = {'Tidak': 0, 'Ya': 1}
smoking_map = {
    'Tidak Ada Info': -1, 'Tidak Pernah': 0, 'Mantan Perokok': 1,
    'Sedang Merokok': 2, 'Tidak Saat Ini': 3, 'Pernah Merokok': 4
}

input_data = np.array([[
    gender_map[Gender],
    Age,
    binary_map[Hipertension],
    binary_map[Heart_disease],
    smoking_map[Smoking_history],
    bmi,
    HbA1c_level,
    Blood_glucose
]])

# Normalisasi Data
scaler.fit(X)
std_data = scaler.transform(input_data)

# Prediksi
diabetes_diagnosis = "Silakan lengkapi data terlebih dahulu."

if st.button('Prediksi Sekarang 🔍'):
    diabetes_prediction = diabetes_model.predict(std_data)
    if diabetes_prediction[0] == 0:
        diabetes_diagnosis = "✨ Anda TIDAK terkena diabetes! Tetap jaga pola hidup sehat ya! 💪"
    else:
        diabetes_diagnosis = "⚠️ Anda TERKENA diabetes. Segera konsultasikan ke dokter untuk penanganan lebih lanjut. 🙏"

# Hasil Prediksi
st.markdown("<h3 style='text-align: center;'>Hasil Prediksi:</h3>", unsafe_allow_html=True)
st.success(diabetes_diagnosis)

# Footer
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>Dibuat dengan ❤️ oleh Farrel0xx Contact=farrelprasraya@gmail.com</p>",
    unsafe_allow_html=True
)
