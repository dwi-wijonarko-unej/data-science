import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# CONFIGURATION & PAGE SETTING
# ==========================================
st.set_page_config(
    page_title="Email Domain Eligibility Classifier", page_icon="📧", layout="centered"
)


# Load Model dan Scaler yang sudah disimpan dari Google Colab
@st.cache_resource
def load_machine_learning_artifacts():
    try:
        # Menggunakan Logistic Regression sebagai model utama agar aman dari overfitting ekstrim
        model = joblib.load("model_email_fraud.pkl")
        scaler = joblib.load("scaler_email.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error(
            "⚠️ File 'model_email_fraud.pkl' atau 'scaler_email.pkl' tidak ditemukan. Pastikan Anda sudah mengunduhnya dari Google Colab."
        )
        return None, None


model, scaler = load_machine_learning_artifacts()

# ==========================================
# HEADER INTERFACES
# ==========================================
st.title("📧 Sistem Klasifikasi Kelayakan Domain Pengirim Email")
st.write("""
Aplikasi berbasis *Supervised Learning* untuk memprediksi tingkat risiko sebuah domain pengirim email berdasarkan performa volume dan keberhasilan pengiriman (*Delivery Rate*).
""")
st.divider()

# Jika berkas pkl berhasil dimuat, tampilkan form input
if model is not None and scaler is not None:
    st.subheader("📊 Masukkan Parameter Profiling User")

    # Form Input untuk Prediksi
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            total_emails = st.number_input(
                "Total Emails Sent",
                min_value=1,
                value=1000,
                step=1,
                help="Jumlah total email yang dikirim oleh domain tersebut.",
            )

            email_sender_count = st.number_input(
                "Email Sender Count",
                min_value=1,
                value=1,
                step=1,
                help="Jumlah entitas/akun pengirim unik di dalam domain tersebut.",
            )

        with col2:
            delivery_rate = st.slider(
                "Delivery Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=0.1,
                help="Persentase email yang berhasil terkirim ke kotak masuk/spam tanpa bounce.",
            )

        submit_button = st.form_submit_button(label="🔮 Klasifikasikan Kelayakan")

    # ==========================================
    # PROCESS PREDICTION
    # ==========================================
    if submit_button:
        # 1. Menampung data input menjadi DataFrame sesuai dengan urutan fitur saat training
        input_data = pd.DataFrame(
            [
                {
                    "Total_Emails": total_emails,
                    "Delivery_Rate": delivery_rate,
                    "Email_Sender": email_sender_count,
                }
            ]
        )

        # 2. Mengkonstruksi Data menggunakan Scaler (Feature Scaling)
        input_scaled = scaler.transform(input_data)

        # 3. Prediksi menggunakan Model
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================
        st.subheader("🎯 Hasil Analisis Model:")

        if prediction[0] == 1:
            st.success("🟢 STATUS: AMAN (SAFE DOMAIN)")
            st.write(
                f"Model meyakini sebesar **{prediction_proba[0][1] * 100:.2f}%** bahwa domain ini memenuhi standar kualitas pengiriman."
            )
        else:
            st.error("🔴 STATUS: BERISIKO (HIGH RISK / SPAMMER)")
            st.write(
                f"Model mendeteksi indikasi fraud/spam sebesar **{prediction_proba[0][0] * 100:.2f}%**. Direkomendasikan untuk pembatasan (suspension)."
            )

        # Menampilkan Nilai Fitur yang diinput dalam bentuk tabel kecil
        st.json(
            {
                "Total_Emails_Input": total_emails,
                "Delivery_Rate_Input": f"{delivery_rate}%",
                "Email_Sender_Input": email_sender_count,
            }
        )

else:
    st.info(
        "ℹ️ Petunjuk Lulus Uji Kompetensi: Jalankan baris kode pengeksportan `.pkl` di Google Colab Anda, unduh filenya, lalu letakkan di folder yang sama dengan file `app.py` ini."
    )
