# recommendation_system.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load environment variables from .env file
load_dotenv()

# Load dataset
data = pd.read_csv("data.csv")  # Use relative path if deploying or sharing

# Spotify credentials from .env
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))


# STEP 3: Song lookup and data functions
def find_song(name, year):
    try:
        results = sp.search(q=f'track:{name} year:{year}', limit=1)
        items = results['tracks']['items']
        return items[0] if items else None
    except:
        return None

def get_song_data(song, spotify_data):
    try:
        result = spotify_data[
            (spotify_data['name'].str.lower() == song['name'].lower()) &
            (spotify_data['year'] == song['year'])
        ].iloc[0]
        return result
    except:
        s = find_song(song['name'], song['year'])
        if s:
            features = sp.audio_features(s['id'])[0]
            return pd.Series({
                'name': song['name'],
                'year': song['year'],
                'explicit': int(s['explicit']),
                'popularity': s['popularity'],
                **features
            })
        else:
            return None

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(f"⚠️  Song not found: {song['name']} ({song['year']})")
            continue
        numeric = song_data[song_data.map(np.isreal)]
        song_vectors.append(numeric.values)

    if not song_vectors:
        raise ValueError("No valid songs found. Cannot compute recommendations.")
    
    return np.mean(song_vectors, axis=0)


def recommend_songs(song_list, spotify_data, n_recommendations=10):
    metadata_cols = ['name', 'year']
    song_center = get_mean_vector(song_list, spotify_data)

    song_features = spotify_data.select_dtypes(include=np.number)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(song_features)

    similarity = cosine_similarity([song_center], scaled_data)
    index = similarity[0].argsort()[::-1]

    recommendations = []
    for idx in index:
        name = spotify_data.iloc[idx]['name']
        year = spotify_data.iloc[idx]['year']
        if not any((s['name'] == name and s['year'] == year) for s in song_list):
            recommendations.append(spotify_data.iloc[idx][metadata_cols])
        if len(recommendations) >= n_recommendations:
            break

    return pd.DataFrame(recommendations)

# STEP 4: Test the system
if __name__ == "__main__":
    input_songs = [
        {"name": "Blinding Lights", "year": 2020},
        {"name": "Shape of You", "year": 2017}
    ]
    recommended = recommend_songs(input_songs, data)
    print("\n🎧 Recommended Songs:")
    print(recommended)
