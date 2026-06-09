import pandas as pd

df = pd.read_csv("movies_cleaned.csv")

Q1 = df["Popularity"].quantile(0.25)
Q3 = df["Popularity"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Popularity"] < lower) |
    (df["Popularity"] > upper)
]

print("Number of Outliers:", len(outliers))