import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("movies_cleaned.csv")

top_languages = df["Original_Language"].value_counts().head(10)

top_languages.plot(kind="bar")

plt.title("Top 10 Movie Languages")
plt.xlabel("Language")
plt.ylabel("Number of Movies")

plt.show()