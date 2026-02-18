import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="House Price Prediction - Sleman",
    page_icon="🏠",
    layout="wide"
)

# =========================
# HELPERS
# =========================
def format_rupiah_adaptive(x):
    """Format harga jadi Rp xxx Juta / Miliar sesuai nominal"""
    try:
        x = float(x)
        if x >= 1_000_000_000:
            return f"Rp {x/1_000_000_000:.2f} Miliar"
        elif x >= 1_000_000:
            return f"Rp {x/1_000_000:.2f} Juta"
        else:
            return f"Rp {x:,.0f}"
    except:
        return str(x)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# =========================
# LOAD ASSETS
# =========================
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("preprocessor.pkl")

    best_model_type = None
    model = None

    if os.path.exists("best_model_tf.h5"):
        model = keras.models.load_model("best_model_tf.h5")
        best_model_type = "DNN"

    elif os.path.exists("best_model.pkl"):
        model = joblib.load("best_model.pkl")
        # cek json untuk lihat model terbaik
        if os.path.exists("model_results.json"):
            with open("model_results.json", "r") as f:
                results = json.load(f)
            best_model_name = max(results.items(), key=lambda x: x[1]["test_r2"])[0]
            best_model_type = best_model_name
        else:
            best_model_type = "Linear Regression"  # default fallback
        model = joblib.load("best_model.pkl")

    else:
        raise FileNotFoundError("best_model_tf.h5 / best_model.pkl tidak ditemukan")

    results = {}
    if os.path.exists("model_results.json"):
        with open("model_results.json", "r") as f:
            results = json.load(f)

    return preprocessor, model, best_model_type, results

try:
    preprocessor, best_model, best_model_type, results_dict = load_assets()
except Exception as e:
    st.error(f"❌ Error loading assets: {e}")
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🏠 House Prediction App")
menu = st.sidebar.radio("Menu", ["Prediksi Harga", "Evaluasi Model"])

st.sidebar.markdown("---")
st.sidebar.write("**Model Terbaik:**", f"**{best_model_type}**")

# =========================
# PREDICTION PAGE
# =========================
if menu == "Prediksi Harga":
    st.title("🏠 Prediksi Harga Rumah Sleman")
    st.write("Isi data rumah di bawah, lalu klik tombol **Predict**.")

    col1, col2 = st.columns(2)

    with col1:
        bed = st.number_input("Bedrooms (bed)", min_value=0, max_value=20, value=2, step=1)
        bath = st.number_input("Bathrooms (bath)", min_value=0, max_value=20, value=1, step=1)
        carport = st.number_input("Carport", min_value=0, max_value=20, value=1, step=1)
        surface_area = st.number_input(
            "Surface Area / Luas Tanah (m²)",
            min_value=1.0, max_value=10000.0,
            value=70.0, step=1.0
        )
        building_area = st.number_input(
            "Building Area / Luas Bangunan (m²)",
            min_value=1.0, max_value=10000.0,
            value=60.0, step=1.0
        )

    with col2:
        try:
            location_list = (
                preprocessor.named_transformers_["cat"]
                .categories_[0]
                .tolist()
            )
        except:
            location_list = []

        if len(location_list) == 0:
            listing_location = st.text_input("Listing Location", value="Sleman")
        else:
            listing_location = st.selectbox("Listing Location", location_list)

    if st.button("Predict", use_container_width=True):
        try:
            input_df = pd.DataFrame([{
                "listing-location": listing_location,
                "bed": bed,
                "bath": bath,
                "carport": carport,
                "surface_area": surface_area,
                "building_area": building_area
            }])

            # Model DNN
            if isinstance(best_model, tf.keras.Model):
                X_processed = preprocessor.transform(input_df).toarray()
                pred = best_model.predict(X_processed, verbose=0).flatten()[0]
            else:
                pred = best_model.predict(input_df)[0]

            pred = max(pred, 0)

            st.markdown(f"""
            <div style='padding:15px; background:linear-gradient(135deg, #667eea, #764ba2); 
                        border-radius:12px; color:white; text-align:center; font-size:1.8rem; font-weight:bold;'>
                Predicted Price:<br>{format_rupiah_adaptive(pred)}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# =========================
# EVALUATION PAGE
# =========================
elif menu == "Evaluasi Model":
    st.title("📊 Evaluasi Model")
    st.write("Berikut hasil evaluasi model dari file **model_results.json**.")

    if not results_dict:
        st.warning("⚠️ model_results.json tidak ditemukan atau kosong.")
        st.stop()

    metrics_df = pd.DataFrame(results_dict).T.reset_index()
    metrics_df.rename(columns={"index": "Model"}, inplace=True)

    for col in ["test_r2", "test_mae", "test_rmse"]:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    metrics_df.rename(columns={
        "test_r2": "R²",
        "test_mae": "MAE",
        "test_rmse": "RMSE"
    }, inplace=True)

    metrics_df = metrics_df[["Model", "R²", "MAE", "RMSE"]]
    metrics_show = metrics_df.copy()
    metrics_show["R²"] = metrics_show["R²"].round(4)
    metrics_show["MAE"] = metrics_show["MAE"].apply(format_rupiah_adaptive)
    metrics_show["RMSE"] = metrics_show["RMSE"].apply(format_rupiah_adaptive)

    st.subheader("📌 Tabel Metrics (R² / MAE / RMSE)")
    st.dataframe(metrics_show, use_container_width=True)

    st.subheader("📌 Visualisasi Performa Model")
    fig = px.bar(metrics_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"]),
                 x="Model", y="value", color="variable",
                 barmode="group", text_auto=True,
                 labels={"value": "Rp", "variable": "Metric"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Model Aktif di Aplikasi")
    st.info(f"Model aktif sekarang: **{best_model_type}**")
