# 📦 Import required libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px

# 📁 Load your datasets
base_path = r"C:\Users\DELL\OneDrive\Bureau\New folder\Recommandation_System"
genre_data = pd.read_csv(base_path + r"\data_by_genres.csv")
song_data = pd.read_csv(base_path + r"\data.csv")

# ---------------------------- #
# ✅ 1. KMeans Clustering on Genres (12 clusters)
# ---------------------------- #

# Select only numeric features for clustering
genre_features = genre_data.select_dtypes(include=['float64', 'int64'])

# Scale the features
scaler = StandardScaler()
genre_scaled = scaler.fit_transform(genre_features)

# Apply KMeans
kmeans_genre = KMeans(n_clusters=12, random_state=42)
genre_clusters = kmeans_genre.fit_predict(genre_scaled)

# Add the cluster labels to the genre data
genre_data['cluster'] = genre_clusters

# ---------------------------- #
# ✅ 2. t-SNE Visualization for Genres
# ---------------------------- #

tsne = TSNE(n_components=2, perplexity=15, random_state=42)
tsne_genre = tsne.fit_transform(genre_scaled)

genre_data['tsne-1'] = tsne_genre[:, 0]
genre_data['tsne-2'] = tsne_genre[:, 1]

fig = px.scatter(
    genre_data, x='tsne-1', y='tsne-2',
    color=genre_data['cluster'].astype(str),
    hover_data=['genres'],
    title='🎧 t-SNE Clustering of Music Genres'
)
fig.show()

# ---------------------------- #
# ✅ 3. KMeans Clustering on Songs (25 clusters)
# ---------------------------- #

# Drop non-numeric columns (like song name, artists if present)
song_features = song_data.select_dtypes(include=['float64', 'int64'])

# Scale song data
song_scaled = scaler.fit_transform(song_features)

# Apply KMeans
kmeans_song = KMeans(n_clusters=25, random_state=42)
song_clusters = kmeans_song.fit_predict(song_scaled)

# Add cluster labels
song_data['cluster'] = song_clusters

# ---------------------------- #
# ✅ 4. PCA Visualization for Songs
# ---------------------------- #

pca = PCA(n_components=2, random_state=42)
pca_song = pca.fit_transform(song_scaled)

song_data['pca-1'] = pca_song[:, 0]
song_data['pca-2'] = pca_song[:, 1]

# Add hover text if you have 'name' or 'artists' column
hover_cols = [col for col in ['name', 'artists'] if col in song_data.columns]

fig = px.scatter(
    song_data, x='pca-1', y='pca-2',
    color=song_data['cluster'].astype(str),
    hover_data=hover_cols,
    title='🎵 PCA Clustering of Songs'
)
fig.show()
