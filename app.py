import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Deteksi Kesehatan Mata Lansia",
    page_icon="👁️",
    layout="wide"
)

# ==========================================
# 2. FUNGSI LOAD & TRAIN MODEL
# ==========================================
@st.cache_resource
def train_models():
    # Load data
    df = pd.read_csv('patient_data.csv')
    df_lansia = df[df['age'] >= 60].copy()
    
    # Cleaning Blood Pressure
    df_lansia[['systolic', 'diastolic']] = df_lansia['blood_pressure'].str.split('/', expand=True).astype(float)
    
    # Fitur Pelatihan (Menggunakan data klinis murni untuk melatih pola)
    features = ['age', 'sugar_percentage', 'glucose_percentage', 'cholesterol_percentage', 
                'obesity_percentage', 'heart_rate', 'systolic', 'diastolic']
    X = df_lansia[features]
    
    # Model 1: Penyakit Mata Umum
    y_eye = df_lansia['has_eye_disease'].astype(int)
    model_eye = RandomForestClassifier(n_estimators=100, random_state=42)
    model_eye.fit(X, y_eye)
    
    # Model 2: Retinopati Diabetik
    y_diabetic = df_lansia['has_diabetic_retinopathy'].astype(int)
    model_diabetic = RandomForestClassifier(n_estimators=100, random_state=42)
    model_diabetic.fit(X, y_diabetic)
    
    return model_eye, model_diabetic, features

# Inisialisasi Model
model_eye, model_diabetic, feature_names = train_models()

# Mapping Bahasa
mapping_indo = {
    'age': 'Usia (Tahun)',
    'sugar_percentage': 'Kadar Gula Darah (%)',
    'glucose_percentage': 'Glukosa (mg/dL)',
    'cholesterol_percentage': 'Kolesterol (mg/dL)',
    'obesity_percentage': 'Indeks Obesitas (%)',
    'heart_rate': 'Detak Jantung (bpm)',
    'systolic': 'Tekanan Darah Sistolik (mmHg)',
    'diastolic': 'Tekanan Darah Diastolik (mmHg)'
}

# ==========================================
# 3. INTERFACE STREAMLIT
# ==========================================
st.title("👁️ Sistem Cerdas Deteksi Risiko Kesehatan Mata Lansia")
st.markdown("Analisis risiko berdasarkan **Parameter Klinis** dan **Riwayat Penyakit Pasien**.")

st.divider()

# Tabel Panduan Nilai Normal (Sebagai Referensi)

with st.expander("ℹ️ Lihat Panduan Rentang Nilai Normal"):

    guide_data = {

        "Parameter": ["Usia", "Gula Darah (%)", "Glukosa", "Kolesterol", "Obesitas", "Detak Jantung", "Sistolik / Diastolik"],

        "Batas Normal": ["60 - 80 Tahun", "4.0 - 6.0 %", "70 - 100 mg/dL", "100 - 200 mg/dL", "15.0 - 25.0 %", "60 - 100 bpm", "90-120 / 60-80 mmHg"]

    }

    st.table(pd.DataFrame(guide_data))
st.divider()

# SIDEBAR INPUT
st.sidebar.header("📥 Data Medis & Riwayat Pasien")
with st.sidebar:
    # --- INPUT RIWAYAT (BARU) ---
    st.subheader("📜 Riwayat Kesehatan")
    in_has_eye = st.selectbox("Pernah didiagnosis Penyakit Mata Umum?", ["Tidak", "Ya"])
    in_has_diabetic = st.selectbox("Pernah didiagnosis Retinopati Diabetik?", ["Tidak", "Ya"])
    
    st.write("---")
    st.subheader("🩸 Parameter Klinis")
    in_age = st.number_input(mapping_indo['age'], min_value=60, max_value=100, value=60)
    in_sugar = st.number_input(mapping_indo['sugar_percentage'], min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
    in_glucose = st.number_input(mapping_indo['glucose_percentage'], min_value=0.0, max_value=500.0, step=1.0)
    in_chol = st.number_input(mapping_indo['cholesterol_percentage'], min_value=0.0, max_value=500.0, step=1.0)
    in_obesity = st.number_input(mapping_indo['obesity_percentage'], min_value=0.0, max_value=60.0, step=0.1, format="%.1f")
    in_hr = st.number_input(mapping_indo['heart_rate'], min_value=0, max_value=200, step=1)
    in_sys = st.number_input("Tekanan Sistolik", min_value=0, max_value=250, step=1)
    in_dia = st.number_input("Tekanan Diastolik", min_value=0, max_value=150, step=1)
    
    st.write("---")
    btn_prediksi = st.sidebar.button("Analisis Sekarang")

# MAIN PANEL
if btn_prediksi:
    if in_sugar == 0 or in_glucose == 0 or in_sys == 0:
        st.warning("⚠️ Mohon isi parameter medis pasien.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Ringkasan Profil Pasien")
            # Menampilkan Riwayat
            st.write(f"**Riwayat Penyakit Mata:** {in_has_eye}")
            st.write(f"**Riwayat Retinopati Diabetik:** {in_has_diabetic}")
            
            # Tabel Parameter
            data_user_display = pd.DataFrame({
                "Parameter": list(mapping_indo.values()),
                "Nilai": [f"{in_age}", f"{in_sugar:.1f}", f"{in_glucose:.0f}", f"{in_chol:.0f}", 
                          f"{in_obesity:.1f}", f"{in_hr:.0f}", f"{in_sys:.0f}", f"{in_dia:.0f}"]
            })
            st.table(data_user_display)

        with col2:
            st.subheader("🎯 Diagnosa & Analisis Risiko")
            
            # Persiapan Data untuk Prediksi ML
            user_input = pd.DataFrame([[in_age, in_sugar, in_glucose, in_chol, in_obesity, in_hr, in_sys, in_dia]], 
                                      columns=feature_names)
            
            prob_eye = model_eye.predict_proba(user_input)[0][1]
            prob_diabetic = model_diabetic.predict_proba(user_input)[0][1]

            # --- LOGIKA PENYESUAIAN BERDASARKAN INPUT RIWAYAT ---
            # Jika user memilih "Ya" pada riwayat, risiko langsung dinaikkan secara signifikan
            if in_has_eye == "Ya":
                prob_eye = max(prob_eye, 0.95)
            if in_has_diabetic == "Ya":
                prob_diabetic = max(prob_diabetic, 0.98)

            # Logika Normalisasi (Jika Riwayat "Tidak" dan Data Normal)
            is_completely_normal = (
                in_sugar <= 6.0 and in_glucose <= 100.0 and in_chol <= 210.0 and in_sys <= 130
            )
            
            if is_completely_normal and in_has_eye == "Tidak" and in_has_diabetic == "Tidak":
                prob_eye = min(prob_eye, 0.25)
                prob_diabetic = min(prob_diabetic, 0.15)

            # Menampilkan Metrik
            m1, m2 = st.columns(2)
            m1.metric(label="Skor Risiko Mata Umum", value=f"{prob_eye*100:.2f}%")
            m2.metric(label="Skor Risiko Retinopati", value=f"{prob_diabetic*100:.2f}%")
            
            st.write("---")
            
            if prob_eye >= 0.5 or prob_diabetic >= 0.5:
                st.error("⚠️ STATUS: MEMERLUKAN PERHATIAN MEDIS")
                if in_has_eye == "Ya" or in_has_diabetic == "Ya":
                    st.info("**Catatan:** Risiko tinggi dikonfirmasi berdasarkan Riwayat Penyakit yang dimasukkan.")
                else:
                    st.warning("**Catatan:** Risiko tinggi dideteksi berdasarkan pola angka klinis pasien.")
                st.write("**Rekomendasi:** Segera hubungi dokter spesialis untuk pemeriksaan fundus mata.")
            else:
                st.success("✅ STATUS: KONDISI STABIL")
                st.write("Parameter klinis berada dalam batas wajar dan tidak ada riwayat penyakit mata yang dilaporkan.")

else:
    st.info("💡 Silakan lengkapi Riwayat Kesehatan dan Parameter Klinis di sidebar.")

st.divider()
st.caption("Aplikasi Skrining Medis - Project Sistem Cerdas © 2026")
