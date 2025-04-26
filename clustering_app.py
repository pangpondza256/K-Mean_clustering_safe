import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

# Page title
st.title("🔍 K-Means Clustering App with Iris Dataset by Thanakorn Risub")

# Load model from uploaded file
with open("kmeans_model.pkl", "rb") as file:
    kmeans = joblib.load(file)

# Load dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Sidebar for number of clusters (force to match loaded model)
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, kmeans.n_clusters)

# Re-cluster if user changes k (optional fallback)
if k != kmeans.n_clusters:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

# Predict clusters
labels = kmeans.predict(X)

# PCA for 2D visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
reduced_df = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
reduced_df["Cluster"] = labels

# Plot clusters
fig, ax = plt.subplots()
for cluster in range(k):
    cluster_data = reduced_df[reduced_df["Cluster"] == cluster]
    ax.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}")
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend()

# Show plot and data
st.pyplot(fig)
st.dataframe(reduced_df.head(10))
