import streamlit as st
from recommendation_system import recommend_songs

st.set_page_config(page_title="🎵 Song Recommendation App", layout="centered")

st.title("🎵 Song Recommendation App")
st.markdown("Find similar songs based on your favorite track using Spotify API or the local dataset.")

song_name = st.text_input("🎧 Enter a song name")

if song_name:
    with st.spinner("🔎 Searching for recommendations..."):
        result = recommend_songs(song_name)

    if isinstance(result, str):
        st.error(result)
    else:
        st.success(f"✅ Recommendations for '{song_name}':")
        st.dataframe(result)
