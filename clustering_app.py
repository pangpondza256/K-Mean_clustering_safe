import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

# Page title
st.title("🔍 K-Means Clustering App (Iris Dataset)")

# Load the KMeans model
with open("kmeans_model.pkl", "rb") as file:
    kmeans = joblib.load(file)

# Load Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X_full = pd.DataFrame(iris.data, columns=iris.feature_names)

# Match features used during training
feature_names_model = getattr(kmeans, "feature_names_in_", None)

if feature_names_model is not None:
    # Select only the features the model was trained on
    X = X_full[feature_names_model]
else:
    st.error("❌ Error: The model does not contain feature names. Please retrain using a DataFrame.")
    st.stop()

# Sidebar configuration
st.sidebar.header("Clustering Settings")
k = st.sidebar.slider("Number of clusters (k)", 2, 10, kmeans.n_clusters)

# Refit if user selects different k
if k != kmeans.n_clusters:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

# Predict cluster labels
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

# Display plot and preview of data
st.pyplot(fig)
st.dataframe(reduced_df.head(10))
