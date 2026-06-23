import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Mall Customer Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# Rename kolom agar lebih rapi
df.columns = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income",
    "Spending Score"
]

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("🎛 Dashboard Filter")

gender = st.sidebar.multiselect(
    "Pilih Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

age_range = st.sidebar.slider(
    "Rentang Umur",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (
        int(df["Age"].min()),
        int(df["Age"].max())
    )
)

filtered_df = df[
    (df["Gender"].isin(gender))
    & (df["Age"] >= age_range[0])
    & (df["Age"] <= age_range[1])
]

# ==========================
# HEADER
# ==========================
st.title("📊 Mall Customer Dashboard")
st.markdown("Dashboard Analisis Pelanggan Mall")

# ==========================
# KPI
# ==========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customer",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Rata-rata Umur",
        round(filtered_df["Age"].mean(),1)
    )

with col3:
    st.metric(
        "Rata-rata Income",
        round(filtered_df["Annual Income"].mean(),1)
    )

with col4:
    st.metric(
        "Rata-rata Spending",
        round(filtered_df["Spending Score"].mean(),1)
    )

st.divider()

# ==========================
# CHARTS
# ==========================
col1, col2 = st.columns(2)

with col1:
    fig_gender = px.pie(
        filtered_df,
        names="Gender",
        title="Distribusi Gender"
    )
    st.plotly_chart(fig_gender, use_container_width=True)

with col2:
    fig_age = px.histogram(
        filtered_df,
        x="Age",
        nbins=15,
        title="Distribusi Umur"
    )
    st.plotly_chart(fig_age, use_container_width=True)

# ==========================
# INCOME VS SPENDING
# ==========================
fig_scatter = px.scatter(
    filtered_df,
    x="Annual Income",
    y="Spending Score",
    color="Gender",
    size="Age",
    title="Income vs Spending Score"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)

# ==========================
# K-MEANS CLUSTERING
# ==========================
st.subheader("🤖 Customer Segmentation")

k = st.slider(
    "Jumlah Cluster",
    2,
    10,
    5
)

X = filtered_df[
    ["Annual Income",
     "Spending Score"]
]

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

filtered_df["Cluster"] = kmeans.fit_predict(X)

fig_cluster = px.scatter(
    filtered_df,
    x="Annual Income",
    y="Spending Score",
    color=filtered_df["Cluster"].astype(str),
    title="K-Means Clustering"
)

st.plotly_chart(
    fig_cluster,
    use_container_width=True
)

# ==========================
# DATA TABLE
# ==========================
st.subheader("📄 Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)
