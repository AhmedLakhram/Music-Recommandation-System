import pandas as pd 

data=pd.read_csv("data.csv")

genre_data=pd.read_csv("data_by_genres.csv")

year_data=pd.read_csv("data_by_year.csv")

artist_data=pd.read_csv("data_by_artist.csv")



print("🎧Main Data:")
print(data.head(2), "\n")

print("🎼 Genre Data:")
print(genre_data.head(2), "\n")

print("📆 Year Data:")
print(year_data.head(2), "\n")

print("🎤 Artist Data:")
print(artist_data.head(2), "\n")

#  Basic Info of Main and Genre Data ===
print("ℹ️ Main Data Info:")
print(data.info(), "\n")

print("ℹ️ Genre Data Info:")
print(genre_data.info(), "\n")

#  Add 'Decade' Column to Main Data ===

if 'year' in data.columns:
    data['decade'] = data['year'].apply(lambda x: str(int(x))[:3] + '0s')
    print("🕒 Data with Decade Column:")
    print(data[['year', 'decade']].head())
else:
    print("⚠️ 'year' column not found in main data. Please check the dataset.")