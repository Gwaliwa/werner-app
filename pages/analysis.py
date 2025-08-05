import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import streamlit as st
from io import BytesIO

# -------------------- Page Config --------------------
st.set_page_config(page_title="Magnetic Anomaly Analysis", layout="wide")
st.sidebar.title("Navigation")
st.sidebar.page_link("main.py", label="🏠 Home")
st.sidebar.page_link("pages/Dashboard.py", label="📊 Dashboard")
st.sidebar.page_link("pages/analysis.py", label="🧪 Analysis")

st.title("🧪 Magnetic Anomaly Cluster Analysis")

# -------------------- Helper Functions --------------------
def read_uploaded_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        lines = content.strip().split('\n')
        data = [list(map(float, line.strip().split())) for line in lines if len(line.strip().split()) >= 2]
        if not data:
            raise ValueError("No valid data found in file.")
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None

def preprocess_data(df):
    if df.shape[1] == 3:
        df.columns = ["Distance", "Dip", "Susceptibility"]
    elif df.shape[1] == 2:
        df.columns = ["Distance", "Susceptibility"]
        df["Dip"] = 0
    else:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]}")
    return df.dropna()

def plot_data(df):
    sns.pairplot(df, diag_kind='kde')
    plt.suptitle("Pairplot of Magnetic Features", y=1.02)
    st.pyplot(plt.gcf())
    plt.clf()

def fit_kmeans(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df[["Dip", "Susceptibility"]])
    return df, kmeans

def plot_clusters(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Dip", y="Susceptibility", hue="Cluster", palette="viridis")
    plt.title("Magnetic Anomalies Clustering")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_clusters_by_distance(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Distance", y="Susceptibility", hue="Cluster", palette="tab10")
    plt.title("Clusters by Distance vs Susceptibility")
    st.pyplot(plt.gcf())
    plt.clf()

def detect_anomalies(df):
    df["Z_Score"] = (df["Susceptibility"] - df["Susceptibility"].mean()) / df["Susceptibility"].std()
    return df[np.abs(df["Z_Score"]) > 2.5]

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Clustered Data", index=False)
    return output.getvalue()

def summarize_findings(df, anomalies):
    summary = f"📊 The dataset contains {len(df)} observations. "
    summary += f"KMeans clustering identified {df['Cluster'].nunique()} clusters. "
    summary += f"{len(anomalies)} anomalies were detected based on Z-score thresholds. "
    summary += "These indicate regions with unusual magnetic susceptibility possibly linked to geological structures."
    return summary

# -------------------- File Handling --------------------
uploaded_file = st.session_state.get("uploaded_file")

if not uploaded_file:
    try:
        with open("Igunda1.DAT", "rb") as f:
            uploaded_file = BytesIO(f.read())
            st.info("🗂 Using default file: Igunda1.DAT")
    except FileNotFoundError:
        st.warning("⚠️ Please upload a magnetic data file from the Home page.")
        st.stop()

df = read_uploaded_file(uploaded_file)
if df is None:
    st.stop()

# -------------------- Main Analysis --------------------
try:
    df = preprocess_data(df)
    if df.empty:
        st.warning("⚠️ Data file contains no usable records.")
        st.stop()

    st.subheader("📄 Raw Data")
    st.write(df.head())

    st.subheader("📊 Pairwise Data Distribution")
    plot_data(df)

    st.subheader("🔍 Cluster Analysis")
    cluster_count = st.slider("Select number of clusters", 2, 10, 3)
    df, model = fit_kmeans(df, n_clusters=cluster_count)
    st.write(df.head())

    plot_clusters(df)
    plot_clusters_by_distance(df)

    st.subheader("🚨 Anomaly Detection")
    anomalies = detect_anomalies(df)
    st.write(anomalies)

    st.subheader("⬇️ Download Clustered Data")
    st.download_button(
        label="Download as Excel",
        data=convert_df_to_excel(df),
        file_name="clustered_magnetic_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.subheader("🧠 Summary of Findings")
    st.write(summarize_findings(df, anomalies))

except Exception as e:
    st.error(f"An error occurred during analysis: {e}")
