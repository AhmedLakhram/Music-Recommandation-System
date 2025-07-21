# 📦 Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

# ---------------------------- #
# 📁 Load Data
# ---------------------------- #
base_path = r"C:\Users\DELL\OneDrive\Bureau\New folder\Recommandation_System"
genre_data = pd.read_csv(base_path + r"\data_by_genres.csv")
song_data = pd.read_csv(base_path + r"\data.csv")

# ---------------------------- #
# ✅ Genre Clustering
# ---------------------------- #
print("\n--- Genre Clustering ---")

# Select numeric features
genre_features = genre_data.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
genre_scaled = scaler.fit_transform(genre_features)

# Apply PCA
pca_genre = PCA(n_components=0.95, random_state=42)
genre_pca = pca_genre.fit_transform(genre_scaled)
print(f"📉 PCA reduced genre dimensions to {genre_pca.shape[1]} components.")

# Silhouette analysis for best K
print("\nFinding best number of clusters for genres...")
k_range = range(2, 21)
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(genre_pca)
    sil = silhouette_score(genre_pca, labels)
    sil_scores.append(sil)

best_k_genres = k_range[np.argmax(sil_scores)]
print(f"✅ Best K for genres: {best_k_genres} (Silhouette = {max(sil_scores):.3f})")

# Plot silhouette score
plt.figure(figsize=(8, 4))
plt.plot(k_range, sil_scores, marker='o')
plt.title("Silhouette Score vs K (Genres)")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Apply KMeans with best K
kmeans_genre = KMeans(n_clusters=best_k_genres, random_state=42, n_init='auto')
genre_clusters = kmeans_genre.fit_predict(genre_pca)
genre_data['cluster'] = genre_clusters
genre_sil = silhouette_score(genre_pca, genre_clusters)
print(f"🎯 Final Silhouette Score (Genres): {genre_sil:.3f}")

# t-SNE visualization
tsne = TSNE(n_components=2, perplexity=15, random_state=42)
genre_tsne = tsne.fit_transform(genre_pca)
genre_data['tsne-1'] = genre_tsne[:, 0]
genre_data['tsne-2'] = genre_tsne[:, 1]

fig1 = px.scatter(
    genre_data, x='tsne-1', y='tsne-2',
    color=genre_data['cluster'].astype(str),
    hover_data=['genres'],
    title=f'🎧 t-SNE Genre Clusters (Silhouette: {genre_sil:.2f})',
    width=800, height=600
)
fig1.show()

# ---------------------------- #
# ✅ Song Clustering
# ---------------------------- #
print("\n--- Song Clustering ---")

# Select numeric features
song_features = song_data.select_dtypes(include=['float64', 'int64'])
song_scaled = scaler.fit_transform(song_features)

# Apply PCA
pca_song = PCA(n_components=0.95, random_state=42)
song_pca = pca_song.fit_transform(song_scaled)
print(f"📉 PCA reduced song dimensions to {song_pca.shape[1]} components.")

# Use fixed 25 clusters as per instructions
kmeans_song = KMeans(n_clusters=25, random_state=42, n_init='auto')
song_clusters = kmeans_song.fit_predict(song_pca)
song_data['cluster'] = song_clusters

# Evaluate clustering
song_sil = silhouette_score(song_pca, song_clusters)
print(f"🎯 Silhouette Score (Songs): {song_sil:.3f}")

# PCA visualization
pca_2d = PCA(n_components=2)
song_2d = pca_2d.fit_transform(song_scaled)
explained_var = pca_2d.explained_variance_ratio_.sum()
song_data['pca-1'] = song_2d[:, 0]
song_data['pca-2'] = song_2d[:, 1]

hover_cols = [col for col in ['name', 'artists'] if col in song_data.columns]

fig2 = px.scatter(
    song_data, x='pca-1', y='pca-2',
    color=song_data['cluster'].astype(str),
    hover_data=hover_cols,
    title=f'🎵 PCA Song Clusters (Silhouette: {song_sil:.2f}, Explained Variance: {explained_var:.1%})',
    width=800, height=600
)
fig2.show()
