import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans

# -------------------- Page Setup --------------------
st.set_page_config(page_title="Werner Deconvolution Dashboard", layout="wide")
st.title("📊 Werner Deconvolution Dashboard")

# -------------------- Check for Uploaded Data --------------------
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No uploaded data found. Please upload a file on the Home page.")
    st.stop()

df = st.session_state.df.copy()

# -------------------- Werner Deconvolution Logic --------------------
def werner_deconvolution(x, tmag):
    if len(x) < 7 or len(tmag) < 7:
        return pd.DataFrame(columns=["Distance", "Dip", "Susceptibility"])

    results = []
    for i in range(3, len(x) - 3):
        try:
            dx = (0.01667*(tmag[i+3] - tmag[i-3])
                  - 0.15*(tmag[i+2] - tmag[i-2])
                  + 0.75*(tmag[i+1] - tmag[i-1]))
            dip = np.arctan(dx) * 180 / np.pi  # Convert to degrees
            sus = abs(dx)  # Dummy value; ensure non-negative for plotting
            results.append([x[i], dip, sus])
        except Exception:
            continue

    return pd.DataFrame(results, columns=["Distance", "Dip", "Susceptibility"])

# -------------------- Run Deconvolution --------------------
x = df["x"].astype(float).values
tmag = df["tmag"].astype(float).values
df_results = werner_deconvolution(x, tmag)

if df_results.empty:
    st.warning("⚠️ Not enough data to compute Werner Deconvolution.")
    st.stop()

st.subheader("📋 Deconvolution Results")
st.dataframe(df_results)

# -------------------- K-Means Clustering --------------------
st.subheader("🔍 Cluster Analysis")

num_clusters = st.slider("Select number of clusters", 2, 6, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df_results["Cluster"] = kmeans.fit_predict(df_results[["Distance", "Dip"]])

# -------------------- Cluster Map --------------------
st.subheader("🧩 Cluster Map")
fig = px.scatter(
    df_results,
    x="Distance",
    y="Dip",
    color=df_results["Cluster"].astype(str),
    size=df_results["Susceptibility"],
    hover_data=["Distance", "Dip", "Susceptibility"],
    title="K-Means Clustered Map (Dip vs Distance)",
)
st.plotly_chart(fig, use_container_width=True)

# -------------------- 3D Deconvolution Visualization --------------------
st.subheader("🌐 3D Magnetic Deconvolution")
fig3d = px.scatter_3d(
    df_results,
    x="Distance",
    y="Dip",
    z="Susceptibility",
    color=df_results["Cluster"].astype(str),
    size="Susceptibility",
    labels={"Distance": "Distance", "Dip": "Dip", "Susceptibility": "Susceptibility"},
    title="3D Visualization of Magnetic Anomalies"
)
fig3d.update_traces(marker=dict(size=4))
st.plotly_chart(fig3d, use_container_width=True)

# -------------------- Download Button --------------------
st.subheader("⬇️ Download Results")
csv = df_results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="werner_deconvolution_results.csv",
    mime="text/csv"
)
