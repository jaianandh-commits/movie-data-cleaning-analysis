import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("📦 Loading dataset...")
df = pd.read_csv("movies.csv", engine="python", on_bad_lines="skip")
print(f"✅ Loaded {df.shape[0]} movies with {df.shape[1]} columns\n")

# ── 2. DATA CLEANING ──────────────────────────────────────────────────────────
print("🧹 Cleaning data...")
df.dropna(subset=["Title", "Vote_Average", "Popularity", "Genre"], inplace=True)
df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df["Year"] = df["Release_Date"].dt.year
df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")
df.dropna(subset=["Vote_Average", "Popularity", "Year"], inplace=True)
df = df[df["Vote_Average"] > 0]
print(f"✅ Clean dataset: {df.shape[0]} movies\n")

# ── 3. STATISTICAL SUMMARY ────────────────────────────────────────────────────
print("=" * 55)
print("        📊 STATISTICAL SUMMARY")
print("=" * 55)
print(f"  Total Movies       : {len(df)}")
print(f"  Average Rating     : {df['Vote_Average'].mean():.2f} / 10")
print(f"  Highest Rating     : {df['Vote_Average'].max():.1f}")
print(f"  Lowest Rating      : {df['Vote_Average'].min():.1f}")
print(f"  Average Popularity : {df['Popularity'].mean():.2f}")
print(f"  Most Popular Movie : {df.loc[df['Popularity'].idxmax(), 'Title']}")
print(f"  Highest Rated Movie: {df.loc[df['Vote_Average'].idxmax(), 'Title']}")
print(f"  Languages          : {df['Original_Language'].nunique()}")
print(f"  Year Range         : {int(df['Year'].min())} - {int(df['Year'].max())}")
print("=" * 55)
print()

# ── 4. GENRE PROCESSING ───────────────────────────────────────────────────────
all_genres = []
for g in df["Genre"].dropna():
    for genre in str(g).split(","):
        all_genres.append(genre.strip())

genre_series = pd.Series(all_genres)
top_genres = genre_series.value_counts().head(10)

# ── 5. VISUALIZATIONS ─────────────────────────────────────────────────────────
os.makedirs("eda_charts", exist_ok=True)
sns.set_theme(style="darkgrid")
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
          "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#7B2D8B"]

# Chart 1: Top 10 Genres
print("📊 Generating Chart 1: Top 10 Genres...")
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_genres.index[::-1], top_genres.values[::-1], color=COLORS)
ax.set_title("Top 10 Movie Genres", fontsize=16, fontweight="bold")
ax.set_xlabel("Number of Movies")
for bar, val in zip(bars, top_genres.values[::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10)
plt.tight_layout()
plt.savefig("eda_charts/1_top_genres.png", dpi=150)
plt.close()

# Chart 2: Vote Average Distribution
print("📊 Generating Chart 2: Rating Distribution...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["Vote_Average"], bins=30, color="#2E86AB", edgecolor="white", alpha=0.85)
ax.axvline(df["Vote_Average"].mean(), color="#C73E1D", linestyle="--",
           linewidth=2, label=f"Mean: {df['Vote_Average'].mean():.2f}")
ax.set_title("Distribution of Movie Ratings", fontsize=16, fontweight="bold")
ax.set_xlabel("Vote Average")
ax.set_ylabel("Number of Movies")
ax.legend()
plt.tight_layout()
plt.savefig("eda_charts/2_rating_distribution.png", dpi=150)
plt.close()

# Chart 3: Popularity vs Rating scatter
print("📊 Generating Chart 3: Popularity vs Rating...")
fig, ax = plt.subplots(figsize=(10, 6))
df_sample = df[df["Popularity"] < 3000]
scatter = ax.scatter(df_sample["Vote_Average"], df_sample["Popularity"],
                     alpha=0.4, color="#A23B72", s=15)
ax.set_title("Popularity vs Vote Average", fontsize=16, fontweight="bold")
ax.set_xlabel("Vote Average")
ax.set_ylabel("Popularity Score")
plt.tight_layout()
plt.savefig("eda_charts/3_popularity_vs_rating.png", dpi=150)
plt.close()

# Chart 4: Top 10 Languages
print("📊 Generating Chart 4: Top Languages...")
top_lang = df["Original_Language"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(top_lang.index, top_lang.values, color=COLORS)
ax.set_title("Top 10 Movie Languages", fontsize=16, fontweight="bold")
ax.set_xlabel("Language")
ax.set_ylabel("Number of Movies")
for i, (lang, val) in enumerate(zip(top_lang.index, top_lang.values)):
    ax.text(i, val + 10, str(val), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("eda_charts/4_top_languages.png", dpi=150)
plt.close()

# Chart 5: Movies released per year
print("📊 Generating Chart 5: Movies Per Year...")
movies_per_year = df.groupby("Year").size()
movies_per_year = movies_per_year[movies_per_year.index >= 2000]
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(movies_per_year.index, movies_per_year.values,
        color="#F18F01", linewidth=2.5, marker="o", markersize=4)
ax.fill_between(movies_per_year.index, movies_per_year.values,
                alpha=0.2, color="#F18F01")
ax.set_title("Number of Movies Released Per Year (2000+)", fontsize=16, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Movies")
plt.tight_layout()
plt.savefig("eda_charts/5_movies_per_year.png", dpi=150)
plt.close()

# Chart 6: Average rating per genre
print("📊 Generating Chart 6: Average Rating by Genre...")
genre_rows = []
for _, row in df.iterrows():
    for genre in str(row["Genre"]).split(","):
        genre_rows.append({"Genre": genre.strip(), "Vote_Average": row["Vote_Average"]})

genre_df = pd.DataFrame(genre_rows)
genre_avg = genre_df.groupby("Genre")["Vote_Average"].mean().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(genre_avg.index[::-1], genre_avg.values[::-1], color=COLORS)
ax.set_title("Average Rating by Genre (Top 10)", fontsize=16, fontweight="bold")
ax.set_xlabel("Average Vote Score")
for bar, val in zip(bars, genre_avg.values[::-1]):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=10)
ax.set_xlim(0, 10)
plt.tight_layout()
plt.savefig("eda_charts/6_avg_rating_by_genre.png", dpi=150)
plt.close()

# Chart 7: Correlation heatmap
print("📊 Generating Chart 7: Correlation Heatmap...")
corr = df[["Popularity", "Vote_Count", "Vote_Average"]].corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, square=True)
ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_charts/7_correlation_heatmap.png", dpi=150)
plt.close()

# Chart 8: Top 10 most popular movies
print("📊 Generating Chart 8: Top 10 Most Popular Movies...")
top_movies = df.nlargest(10, "Popularity")[["Title", "Popularity"]]
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_movies["Title"][::-1], top_movies["Popularity"][::-1], color=COLORS)
ax.set_title("Top 10 Most Popular Movies", fontsize=16, fontweight="bold")
ax.set_xlabel("Popularity Score")
for bar, val in zip(bars, top_movies["Popularity"][::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f"{val:.0f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("eda_charts/8_top_popular_movies.png", dpi=150)
plt.close()

print("\n✅ All 8 charts saved in eda_charts/ folder!")
print("\n🎉 EDA Project Complete!")
print("=" * 55)
print("  Charts saved in: eda_charts/")
print("  1. Top 10 Genres")
print("  2. Rating Distribution")
print("  3. Popularity vs Rating")
print("  4. Top 10 Languages")
print("  5. Movies Per Year")
print("  6. Average Rating by Genre")
print("  7. Correlation Heatmap")
print("  8. Top 10 Most Popular Movies")
print("=" * 55)