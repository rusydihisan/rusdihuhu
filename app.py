import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Konfigurasi halaman
st.set_page_config(page_title="Mall Customer Analysis", layout="wide")

st.title("📊 Analisis Data Mall Customers")
st.write("Aplikasi Streamlit untuk analisis pelanggan mall.")

# Upload file
uploaded_file = st.file_uploader("Upload Dataset Mall_Customers.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Menampilkan data
    st.subheader("Dataset")
    st.dataframe(df)

    # Informasi dataset
    st.subheader("Informasi Dataset")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Jumlah Data:", df.shape[0])
        st.write("Jumlah Kolom:", df.shape[1])

    with col2:
        st.write("Kolom Dataset:")
        st.write(df.columns.tolist())

    # Statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

    # Visualisasi umur
    st.subheader("Distribusi Umur Pelanggan")
    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=10)
    ax.set_xlabel("Umur")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

    # Scatter Plot
    st.subheader("Income vs Spending Score")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["Annual_Income_(k$)"], df["Spending_Score"])
    ax2.set_xlabel("Annual Income (k$)")
    ax2.set_ylabel("Spending Score")
    st.pyplot(fig2)

    # Clustering K-Means
    st.subheader("Clustering Pelanggan (K-Means)")

    k = st.slider("Jumlah Cluster", 2, 10, 5)

    X = df[["Annual_Income_(k$)", "Spending_Score"]]

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    fig3, ax3 = plt.subplots()
    scatter = ax3.scatter(
        df["Annual_Income_(k$)"],
        df["Spending_Score"],
        c=df["Cluster"]
    )

    ax3.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=200
    )

    ax3.set_xlabel("Annual Income (k$)")
    ax3.set_ylabel("Spending Score")
    ax3.set_title("Hasil Clustering Pelanggan")

    st.pyplot(fig3)

    st.subheader("Data Hasil Cluster")
    st.dataframe(df)

else:
    st.info("Silakan upload file Mall_Customers.csv terlebih dahulu.")
