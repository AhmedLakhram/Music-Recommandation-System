import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from dotenv import load_dotenv
import os

# Load credentials
load_dotenv()
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

# Load dataset
base_path = r"C:\Users\DELL\OneDrive\Bureau\New folder\Recommandation_System"
dataset = pd.read_csv(base_path + r"\data.csv")
dataset['artists'] = dataset['artists'].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else x)

# Select features
features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'valence', 'loudness', 'tempo', 'speechiness']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset[features])


# 🔹 Get song features from Spotify API
def get_song_features_from_spotify(song_name):
    try:
        results = sp.search(q=song_name, limit=1, type='track')
        if not results['tracks']['items']:
            return None, None

        track = results['tracks']['items'][0]
        track_id = track['id']
        artist = track['artists'][0]['name']
        audio_features = sp.audio_features(track_id)[0]

        if not audio_features:
            return None, None

        # Extract features
        song_vector = np.array([audio_features[feature] for feature in features]).reshape(1, -1)
        return song_vector, artist

    except Exception as e:
        print(f"API error: {e}")
        return None, None


# 🔹 Main recommendation function
def recommend_songs(song_name, top_n=5):
    # Try API first
    song_vector, artist = get_song_features_from_spotify(song_name)

    if song_vector is not None:
        scaled_vector = scaler.transform(song_vector)
        similarities = cosine_similarity(scaled_vector, scaled_data)[0]
        indices = similarities.argsort()[::-1][:top_n]
        recommendations = dataset.iloc[indices][['name', 'artists', 'year']]
        recommendations.columns = ['Song Name', 'Artist(s)', 'Year']
        print(f"\n🎧 Found using Spotify API: '{song_name}' by {artist}")
        return recommendations

    # Fallback to dataset
    matches = dataset[dataset['name'].str.lower() == song_name.lower()]
    if matches.empty:
        return f"❌ Song '{song_name}' not found in Spotify or local dataset."

    song_index = matches.index[0]
    song_vector = scaled_data[song_index].reshape(1, -1)

    similarities = cosine_similarity(song_vector, scaled_data)[0]
    indices = similarities.argsort()[::-1][1:top_n+1]

    recommendations = dataset.iloc[indices][['name', 'artists', 'year']]
    recommendations.columns = ['Song Name', 'Artist(s)', 'Year']
    print(f"\n📁 Found in local dataset: '{song_name}'")
    return recommendations
