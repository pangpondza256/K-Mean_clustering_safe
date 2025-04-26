import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io

# Page title
st.title("🔍 K-Means Clustering App with Iris Dataset by Thanakorn Risub")

# Load dataset from pkl
st.sidebar.header("Dataset Information")
try:
    with open('kmeans_model.pkl', 'rb') as file:
        X = pickle.load(file)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
    st.sidebar.success("✅ Loaded 'kmeans_model.pkl' successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load 'kmeans_model.pkl': {e}")
    st.stop()

# Sidebar - Number of clusters
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 1, 10, 3)

# Run K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
reduced_df = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
reduced_df["Cluster"] = labels

# Get cluster centers (projected into PCA space)
centers = pca.transform(kmeans.cluster_centers_)

# Plot clusters
fig, ax = plt.subplots(figsize=(8, 6))
for cluster in range(k):
    cluster_data = reduced_df[reduced_df["Cluster"] == cluster]
    ax.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}", s=40)

# Plot cluster centers
ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centers')

ax.set_title("Clusters (2D PCA Projection)", fontsize=16)
ax.set_xlabel("PCA1", fontsize=14)
ax.set_ylabel("PCA2", fontsize=14)
ax.legend()
ax.grid(True)

# Show plot
st.pyplot(fig)

# Download plot button
buf = io.BytesIO()
fig.savefig(buf, format="png")
st.download_button(
    label="📥 Download Plot as PNG",
    data=buf.getvalue(),
    file_name="cluster_plot.png",
    mime="image/png"
)

# Show sample of clustered data
st.subheader("Sample of clustered data:")
st.dataframe(reduced_df.head(10))