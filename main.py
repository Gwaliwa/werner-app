# main.py

import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------- Page Setup --------------------
st.set_page_config(page_title="Werner Deconvolution App", layout="wide")
st.title("🏠 Werner Deconvolution - Home")

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("Navigation")
st.sidebar.page_link("main.py", label="🏠 Home")
st.sidebar.page_link("pages/Dashboard.py", label="📊 Dashboard")
st.sidebar.page_link("pages/analysis.py", label="🧪 Analysis")

# -------------------- Universal File Reader --------------------
def read_uploaded_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
        if df.shape[1] < 2:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep="\t", header=None)
        if df.shape[1] < 2:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",", header=None)
        if df.shape[1] < 2:
            raise ValueError("Unable to detect valid delimiter or not enough columns.")

        df = df.iloc[:, :2]
        df.columns = ["x", "tmag"]
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        return df
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        return None

# -------------------- Derivative Computation --------------------
def compute_horizontal_derivative(x, tmag):
    if len(tmag) < 7:
        return None, None
    x_der, h_der = [], []
    for i in range(3, len(tmag) - 3):
        dx = (0.01667 * (tmag[i + 3] - tmag[i - 3])
              - 0.15 * (tmag[i + 2] - tmag[i - 2])
              + 0.75 * (tmag[i + 1] - tmag[i - 1]))
        x_der.append(x[i])
        h_der.append(dx)
    return x_der, h_der

# -------------------- File Upload --------------------
uploaded = st.file_uploader("📁 Upload a magnetic data file (.dat, .txt, .csv)", type=["dat", "txt", "csv"])
if uploaded:
    st.session_state.uploaded_file = uploaded

# -------------------- File Handling --------------------
uploaded_file = st.session_state.get("uploaded_file", None)
if uploaded_file:
    df = read_uploaded_file(uploaded_file)
    if df is not None:
        st.session_state.df = df

        st.success(f"✅ Loaded {len(df)} data points.")
        st.subheader("📄 Raw Uploaded Data")
        st.dataframe(df)

        # Magnetic anomaly plot
        fig = px.line(df, x="x", y="tmag", title="📉 Distance vs Magnetic Anomalies",
                      labels={"x": "Distance", "tmag": "Tmag"})
        st.plotly_chart(fig, use_container_width=True)

        # Derivative plot
        x = df["x"].values
        tmag = df["tmag"].values
        x_der, h_der = compute_horizontal_derivative(x, tmag)

        if x_der:
            st.subheader("📈 Horizontal Derivative (dT/dx)")
            fig2 = px.scatter(x=x_der, y=h_der, labels={"x": "Distance", "y": "dT/dx"},
                              title="Horizontal Derivative")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("⚠️ Not enough data points to compute derivative (minimum 7).")
else:
    st.info("Please upload a `.dat`, `.txt`, or `.csv` file to begin.")
