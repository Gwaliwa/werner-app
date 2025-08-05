# analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="Magnetic Anomaly Analysis", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.page_link("main.py", label="🏠 Home")
st.sidebar.page_link("pages/Dashboard.py", label="📊 Dashboard")
st.sidebar.page_link("pages/analysis.py", label="🧪 Analysis")


def read_uploaded_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        lines = content.strip().split('\n')
        data = []
        for line in lines:
            try:
                values = list(map(float, line.strip().split()))
                if len(values) >= 2:
                    data.append(values)
            except ValueError:
                continue
        if not data:
            raise ValueError("No valid data found in file.")
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def preprocess_data(df):
    col_count = df.shape[1]
    if col_count == 3:
        df.columns = ["Distance", "Dip", "Susceptibility"]
    elif col_count == 2:
        df.columns = ["Distance", "Susceptibility"]
        df["Dip"] = 0
    else:
        raise ValueError(f"Unexpected number of columns: {col_count}")
    df.dropna(inplace=True)
    return df


def plot_data(df):
    sns.pairplot(df, diag_kind='kde')
    plt.suptitle("Pairplot of Magnetic Features", y=1.02)
    st.pyplot(plt)
    plt.clf()


def fit_kmeans(df, n_clusters=3):
    X = df[["Dip", "Susceptibility"]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    return df, kmeans


def plot_clusters(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Dip", y="Susceptibility", hue="Cluster", palette="viridis")
    plt.title("Magnetic Anomalies Clustering")
    plt.xlabel("Dip")
    plt.ylabel("Susceptibility")
    plt.legend(title="Cluster")
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()


def plot_clusters_by_distance(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Distance", y="Susceptibility", hue="Cluster", palette="tab10")
    plt.title("Clusters by Distance vs Susceptibility")
    plt.xlabel("Distance")
    plt.ylabel("Susceptibility")
    plt.legend(title="Cluster")
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()


def detect_anomalies(df):
    df['Z_Score'] = (df['Susceptibility'] - df['Susceptibility'].mean()) / df['Susceptibility'].std()
    anomalies = df[np.abs(df['Z_Score']) > 2.5]
    return anomalies


def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Clustered Data', index=False)
    processed_data = output.getvalue()
    return processed_data


def summarize_findings(df, anomalies):
    if df.empty:
        return "No data available to summarize."
    summary = f"The dataset contains {len(df)} observations. "
    summary += f"KMeans clustering identified {df['Cluster'].nunique()} unique clusters. "
    summary += f"A total of {len(anomalies)} anomalies were detected based on Z-score thresholds, "
    summary += "indicating regions with unusually high or low susceptibility. "
    summary += "These findings suggest the presence of localized magnetic anomalies that could reflect geological structures or variations."
    return summary


# Streamlit page logic
st.title("🧪 Magnetic Anomaly Cluster Analysis")

if "uploaded_file" in st.session_state:
    uploaded_file = st.session_state.uploaded_file
    df = read_uploaded_file(uploaded_file)

    if df is not None:
        try:
            df = preprocess_data(df)

            if df.empty:
                st.warning("The uploaded file contains no usable data.")
            else:
                st.subheader("Raw Data")
                st.write(df.head())

                st.subheader("Pairwise Data Distribution")
                plot_data(df)

                st.subheader("Cluster Analysis")
                cluster_count = st.slider("Select number of clusters", 2, 10, 3)
                df, model = fit_kmeans(df, n_clusters=cluster_count)
                st.write(df.head())

                plot_clusters(df)
                plot_clusters_by_distance(df)

                st.subheader("Anomaly Detection")
                anomalies = detect_anomalies(df)
                st.write(anomalies)

                st.subheader("📥 Download Clustered Data")
                st.download_button(
                    label="Download as Excel",
                    data=convert_df_to_excel(df),
                    file_name='clustered_magnetic_data.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

                st.subheader("🧠 Summary of Findings")
                summary = summarize_findings(df, anomalies)
                st.write(summary)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
st.warning("Please upload a magnetic data file from the Home page.")
