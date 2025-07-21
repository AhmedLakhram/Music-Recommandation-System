import streamlit as st
from recommendation_system import recommend_songs

st.set_page_config(page_title="🎧 Music Recommendation System", layout="centered")

st.title("🎵 Song Recommendation App")
st.markdown("Find similar songs based on your favorite track.")

# Input fields
song = st.text_input("Enter a song name")
result = recommend_songs(song)


# Recommend button
if st.button("🔍 Recommend Songs"):
    with st.spinner("Finding recommendations..."):
        recommendations = recommend_songs(song_name, year)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.success("Here are your recommended songs:")
            st.dataframe(recommendations)
