import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("movies_cleaned.csv")

plt.hist(df["Vote_Average"], bins=10)

plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Movies")

plt.show()