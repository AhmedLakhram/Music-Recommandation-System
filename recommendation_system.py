# recommendation_system.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

# Load data
base_path = r"C:\Users\DELL\OneDrive\Bureau\New folder\Recommandation_System"
dataset = pd.read_csv(base_path + r"\data.csv")

# ✅ Fix artist formatting (e.g., ['Artist1', 'Artist2'] → Artist1, Artist2)
dataset['artists'] = dataset['artists'].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else x)

# ✅ Drop unnecessary columns
columns_to_use = ['name', 'artists', 'year', 'acousticness', 'danceability', 'energy',
                  'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo', 'speechiness']
dataset = dataset[columns_to_use].dropna()

# ✅ Scale numerical features
features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'valence', 'loudness', 'tempo', 'speechiness']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset[features])

# ✅ Recommendation function
def recommend_songs(song_name, top_n=5):
    matches = dataset[dataset['name'].str.lower() == song_name.lower()]
    if matches.empty:
        return f"❌ Song '{song_name}' not found."

    song_index = matches.index[0]
    song_vector = scaled_data[song_index].reshape(1, -1)

    similarities = cosine_similarity(song_vector, scaled_data)[0]
    indices = similarities.argsort()[::-1][1:top_n+1]

    # ✅ Clean output
    recommended = dataset.iloc[indices][['name', 'artists', 'year']].copy()
    recommended.columns = ['Song Name', 'Artist(s)', 'Year']
    return recommended
