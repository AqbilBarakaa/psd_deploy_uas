import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go # Library baru untuk grafik interaktif

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Strawberry Quality Check",
    page_icon="🍓",
    layout="centered"
)

# CSS CUSTOMIZATION
st.markdown("""
<style>
    /* Mengatur Font Global */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Header Utama */
    .main-header {
        background: linear-gradient(90deg, #C70039 0%, #900C3F 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 14px;
        margin-top: 5px;
        opacity: 0.9;
    }

    /* Styling Tombol */
    .stButton>button {
        width: 100%;
        background-color: #C70039;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        height: 50px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #900C3F;
        box-shadow: 0 5px 15px rgba(199, 0, 57, 0.4);
        transform: translateY(-2px);
    }

    /* Result Card Styling */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        border-top: 5px solid #ccc;
        color: #333;
    }
    
    /* Status Colors */
    .status-authentic {
        border-top-color: #28a745 !important;
    }
    .status-adulterated {
        border-top-color: #dc3545 !important;
    }
    
    .result-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-desc {
        font-size: 16px;
        color: #666;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL & SCALER
@st.cache_resource
def load_resources():
    try:
        # Load Model dan Scaler
        model = joblib.load('model_strawberry.pkl')
        scaler = joblib.load('scaler_strawberry.pkl')
        return model, scaler
    except Exception as e:
        return None, None

# HEADER INTERFACE
st.markdown("""
<div class='main-header'>
    <h1>SISTEM DETEKSI KEASLIAN STROBERI</h1>
    <p>Analisis Spektroskopi Berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Load resources
model, scaler = load_resources()

if model is None or scaler is None:
    st.error("SYSTEM ERROR: File model atau scaler tidak ditemukan. Harap unggah file .pkl.")
    st.stop()

# SIDEBAR 
with st.sidebar:
    st.markdown("### INFORMASI")
    st.info("""
    - **Metode:** Random Forest Classifier
    - *Input:** 235 Titik Gelombang Spektrum
    - **Output:** Binary Classification
    """)
    
    st.markdown("---")
    st.markdown("**Panduan:**")
    st.text("1. Siapkan file .txt/.csv hasil Scan Spektroskopi.")
    st.text("2. Pastikan format data numerik.")
    st.text("3. Klik tombol analisis.")

# UPLOAD DATA
st.write("### 1. Upload Data Sampel")
uploaded_file = st.file_uploader("", type=['txt', 'csv'], help="Unggah file spektroskopi mentah di sini")

input_data = None 

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        
        # Deteksi pemisah
        if "," in content.split('\n')[0]:
            sep = ","
        else:
            sep = r"\s+"
            
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=None, sep=sep)
        data_values = df.values
        
        # Handling Kolom
        if data_values.shape[1] > 235:
            st.toast(f"Info: Memotong kolom label. Menggunakan 235 fitur terakhir.", icon=None)
            input_data = data_values[:, 1:] 
        elif data_values.shape[1] == 235:
            input_data = data_values
        else:
            st.error(f"FORMAT ERROR: Data memiliki {data_values.shape[1]} kolom. Dibutuhkan 235 kolom.")
            st.stop()

        # Ambil sampel pertama
        input_data = input_data.astype(float)
        input_data = input_data[0].reshape(1, -1)
        
        # Tampilkan preview data mini dalam expander
        with st.expander("Lihat Data Mentah"):
            st.dataframe(pd.DataFrame(input_data), hide_index=True)

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

# VISUALISASI & PREDIKSI
if input_data is not None:
    
    st.write("### 2. Analisis Spektrum")
    
    # CHART
    x_axis = np.arange(235)
    y_axis = input_data[0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, 
        y=y_axis, 
        mode='lines', 
        name='Sampel',
        line=dict(color='#C70039', width=3),
        fill='tozeroy', 
        fillcolor='rgba(199, 0, 57, 0.1)'
    ))
    
    fig.update_layout(
        title="Visualisasi Gelombang Spektroskopi",
        xaxis_title="Titik Panjang Gelombang",
        yaxis_title="Intensitas Absorbansi",
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x"
    )
    st.plotly_chart(fig, use_container_width=True)

    # TOMBOL PREDIKSI
    st.write("### 3. Hasil Diagnosa")
    
    col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])
    with col_btn_2:
        predict_btn = st.button("JALANKAN ANALISIS AI")
    
    if predict_btn:
        with st.spinner("Sedang memproses algoritma..."):
            import time
            time.sleep(0.8)
            
            # PREDIKSI
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            # Hitung Probabilitas
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_scaled)[0]
                confidence = np.max(probs) * 100
            else:
                confidence = 100.0
            
            # LOGIKA HASIL
            if prediction == 1:
                label_text = "AUTHENTIC (ASLI)"
                desc_text = "Sampel terverifikasi memiliki profil kimia stroberi murni."
                status_class = "status-authentic"
                text_color = "#28a745"
                gauge_color = "#28a745"
            else:
                label_text = "ADULTERATED (OPLOSAN)"
                desc_text = "Terdeteksi anomali komposisi kimia pada sampel."
                status_class = "status-adulterated"
                text_color = "#dc3545"
                gauge_color = "#dc3545"

            # TAMPILAN HASIL DASHBOARD
            col_res1, col_res2 = st.columns([1.5, 1])
            
            # Kolom Kiri: Teks Penjelasan
            with col_res1:
                st.markdown(f"""
                <div class='result-card {status_class}'>
                    <div style='color: #888; font-size: 12px; margin-bottom: 5px;'>HASIL PREDIKSI</div>
                    <div class='result-title' style='color: {text_color};'>{label_text}</div>
                    <div class='result-desc'>{desc_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Tampilkan detail probabilitas jika ada
                if hasattr(model, "predict_proba"):
                    st.caption("Distribusi Probabilitas Model:")
                    prob_df = pd.DataFrame(probs.reshape(1, -1), columns=model.classes_)
                    st.bar_chart(prob_df.T, color=text_color, height=150)

            # Kolom Kanan: Gauge Chart (Spidometer)
            with col_res2:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence,
                    title = {'text': "Tingkat Keyakinan AI", 'font': {'size': 14, 'color': '#555'}},
                    number = {'suffix': "%", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#eee",
                        'steps': [
                            {'range': [0, 50], 'color': '#f9f9f9'},
                            {'range': [50, 100], 'color': '#f0f0f0'}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)