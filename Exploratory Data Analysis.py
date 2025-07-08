# Import libraries 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS


# Load the data from my local folder loading all csv files into memory
base_path = r"C:\Users\DELL\OneDrive\Bureau\New folder\Recommandation_System"
data = pd.read_csv(base_path + r"\data.csv")
genre_data = pd.read_csv(base_path + r"\data_by_genres.csv")
year_data = pd.read_csv(base_path + r"\data_by_year.csv")
artist_data = pd.read_csv(base_path + r"\data_by_artist.csv")
data['decade'] = data['year'].apply(lambda x: str(x)[:3] + '0s')

# ✅ 1. Visualize the distribution of tracks across different decades
plt.figure(figsize=(10, 5))
sns.countplot(data=data, x='decade', order=sorted(data['decade'].unique()))
plt.title('Distribution of Tracks by Decade')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ✅ 2. Plot sound features trend over years
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=sound_features, title='Trends of Sound Features Over Time')
fig.show()

# ✅ 3. Plot loudness over decades
fig = px.line(year_data, x='year', y='loudness', title='Trend of Loudness Over Decades')
fig.show()

# ✅ 4. Top 10 genres by popularity and sound features
top10_genres = genre_data.sort_values('popularity', ascending=False).head(10)
fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'],
             barmode='group', title='Top 10 Genres - Sound Features')
fig.show()

# ✅ 5–6. Word cloud of genres
comment_words = ' '.join(genre_data['genres'].dropna().astype(str))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800, background_color='white',
                      stopwords=stopwords, max_words=40, min_font_size=10).generate(comment_words)
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Genres Word Cloud")
plt.show()

# ✅ 7–8. Word cloud of artists
comment_words = ' '.join(artist_data['artists'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=800, background_color='white',
                      stopwords=stopwords, max_words=40, min_word_length=3, min_font_size=10).generate(comment_words)
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Artists Word Cloud")
plt.show()

# ✅ 9. Top 10 artists with most songs
top10_most_song_produced_artists = artist_data['artists'].value_counts().reset_index()
top10_most_song_produced_artists.columns = ['artists', 'count']
print("\n🎤 Top 10 Artists with Most Songs:")
print(top10_most_song_produced_artists.head(10))

# ✅ 10. Top 10 artists with highest popularity
top10_popular_artists = artist_data[['artists', 'popularity']].drop_duplicates()
top10_popular_artists = top10_popular_artists.sort_values('popularity', ascending=False)
print("\n🔥 Top 10 Popular Artists:")
print(top10_popular_artists.head(10))

# ✅ 11. Conclusion
print("\n📌 Conclusion:")
print("1. Most tracks are from recent decades (2000s–2020s).")
print("2. Danceability and energy have increased over time.")
print("3. Popular genres tend to be more energetic and upbeat.")
print("4. Some artists dominate both in quantity and popularity.")
