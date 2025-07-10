import streamlit as st
import pandas as pd
from recommendation_system import recommend_songs
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load dataset
base_path = r"C:\Users\DELL\OneDrive\Bureau\New folder\Recommandation_System"
data = pd.read_csv("data.csv")
st.set_page_config(page_title="🎵 Spotify Recommender", layout="centered")
st.title("🎧 Music Recommendation System")

song_name = st.text_input("Enter a song name", value="Shape of You")
song_year = st.number_input("Enter release year", min_value=1900, max_value=2025, value=2017)

if st.button("Recommend"):
    with st.spinner("Finding similar songs..."):
        input_song = [{"name": song_name, "year": int(song_year)}]
        recommendations = recommend_songs(input_song, data)

        if recommendations.empty:
            st.warning("No recommendations found. Try a different song.")
        else:
            st.success("Here are your top 10 recommendations:")
            st.dataframe(recommendations)
