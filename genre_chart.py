import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("movies_cleaned.csv")

genres = df["Genre"].str.split(", ").explode()

top_genres = genres.value_counts().head(10)

top_genres.plot(kind="bar")

plt.title("Top 10 Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Count")

plt.show()