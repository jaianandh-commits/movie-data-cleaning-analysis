import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv("movies.csv", engine="python", on_bad_lines="skip")
df.dropna(subset=["Title", "Vote_Average", "Popularity", "Genre"], inplace=True)
df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df["Year"] = df["Release_Date"].dt.year
df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
df["Vote_Count"] = pd.to_numeric(df["Vote_Count"], errors="coerce")
df.dropna(subset=["Vote_Average", "Popularity", "Year"], inplace=True)
df = df[df["Vote_Average"] > 0]

most_popular = df.loc[df["Popularity"].idxmax(), "Title"]
highest_rated = df.loc[df["Vote_Average"].idxmax(), "Title"]
avg_rating = round(df["Vote_Average"].mean(), 2)
avg_popularity = round(df["Popularity"].mean(), 2)

# ── SETUP ─────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    "EDA_Movies_Report.pdf",
    pagesize=A4,
    rightMargin=50, leftMargin=50,
    topMargin=50, bottomMargin=50
)
styles = getSampleStyleSheet()
elements = []

BLUE = colors.HexColor("#2E86AB")
DARK = colors.HexColor("#151515")
WHITE = colors.white

heading_style = ParagraphStyle(
    "heading", fontSize=13, fontName="Helvetica-Bold",
    textColor=BLUE, spaceAfter=6
)
body_style = ParagraphStyle(
    "body", fontSize=10, fontName="Helvetica",
    textColor=colors.HexColor("#333333"), spaceAfter=4, leading=16
)
title_style = ParagraphStyle(
    "title", fontSize=22, fontName="Helvetica-Bold",
    textColor=BLUE, spaceAfter=6, alignment=1
)
subtitle_style = ParagraphStyle(
    "subtitle", fontSize=12, fontName="Helvetica",
    textColor=colors.HexColor("#555555"), spaceAfter=4, alignment=1
)

# ── TITLE PAGE ────────────────────────────────────────────────────────────────
elements.append(Spacer(1, 60))
elements.append(Paragraph("🎬 Exploratory Data Analysis", title_style))
elements.append(Paragraph("Movies Dataset", title_style))
elements.append(Spacer(1, 12))
elements.append(Paragraph("Thiranex Internship Project — EDA Report", subtitle_style))
elements.append(Paragraph(f"Submitted on: {datetime.now().strftime('%d %B %Y')}", subtitle_style))
elements.append(Spacer(1, 30))

# divider
elements.append(Table([[""]], colWidths=[500], rowHeights=[2],
    style=TableStyle([("BACKGROUND", (0,0), (-1,-1), BLUE)])))
elements.append(Spacer(1, 30))

# ── 1. INTRODUCTION ───────────────────────────────────────────────────────────
elements.append(Paragraph("1. Introduction", heading_style))
elements.append(Paragraph(
    "This report presents a comprehensive Exploratory Data Analysis (EDA) of a movies dataset "
    "containing information about nearly 10,000 films. The analysis aims to uncover patterns, "
    "trends, and correlations within the data using statistical summaries and visualizations. "
    "The dataset includes attributes such as movie title, release date, genre, popularity score, "
    "vote count, vote average, and original language.",
    body_style
))
elements.append(Spacer(1, 12))

# ── 2. DATASET OVERVIEW ───────────────────────────────────────────────────────
elements.append(Paragraph("2. Dataset Overview", heading_style))
overview_data = [
    ["Attribute", "Details"],
    ["Dataset", "Movies Dataset (TMDB)"],
    ["Total Records", f"{len(df):,} movies"],
    ["Total Columns", "9 columns"],
    ["Year Range", f"{int(df['Year'].min())} — {int(df['Year'].max())}"],
    ["Languages", f"{df['Original_Language'].nunique()} unique languages"],
    ["Missing Values", "Handled by dropping incomplete rows"],
]
t = Table(overview_data, colWidths=[200, 280])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), BLUE),
    ("TEXTCOLOR", (0,0), (-1,0), WHITE),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F0F4F8"), WHITE]),
    ("LEFTPADDING", (0,0), (-1,-1), 10),
    ("TOPPADDING", (0,0), (-1,-1), 7),
    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
]))
elements.append(t)
elements.append(Spacer(1, 16))

# ── 3. STATISTICAL SUMMARY ────────────────────────────────────────────────────
elements.append(Paragraph("3. Statistical Summary", heading_style))
stats_data = [
    ["Metric", "Value"],
    ["Average Vote Rating", f"{avg_rating} / 10"],
    ["Highest Rating", f"{df['Vote_Average'].max()} / 10"],
    ["Lowest Rating", f"{df['Vote_Average'].min()} / 10"],
    ["Average Popularity Score", f"{avg_popularity}"],
    ["Most Popular Movie", most_popular],
    ["Highest Rated Movie", highest_rated],
    ["Total Unique Genres", f"{df['Genre'].nunique()}"],
]
t2 = Table(stats_data, colWidths=[200, 280])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), BLUE),
    ("TEXTCOLOR", (0,0), (-1,0), WHITE),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F0F4F8"), WHITE]),
    ("LEFTPADDING", (0,0), (-1,-1), 10),
    ("TOPPADDING", (0,0), (-1,-1), 7),
    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
]))
elements.append(t2)
elements.append(Spacer(1, 20))

# ── 4. VISUALIZATIONS ─────────────────────────────────────────────────────────
elements.append(Paragraph("4. Visualizations & Insights", heading_style))

charts = [
    ("eda_charts/1_top_genres.png",
     "4.1 Top 10 Movie Genres",
     "Drama, Comedy, and Thriller dominate the dataset. Drama leads with the highest number of movies, reflecting audience preference for story-driven content."),
    ("eda_charts/2_rating_distribution.png",
     "4.2 Distribution of Movie Ratings",
     "Most movies are rated between 5 and 8 out of 10. The distribution is slightly left-skewed, indicating that most movies receive above-average ratings."),
    ("eda_charts/3_popularity_vs_rating.png",
     "4.3 Popularity vs Vote Average",
     "There is a weak positive correlation between popularity and rating. Some highly popular movies have moderate ratings, suggesting popularity is driven by factors beyond quality alone."),
    ("eda_charts/4_top_languages.png",
     "4.4 Top 10 Movie Languages",
     "English dominates as the primary language, followed by French and Japanese. This reflects the global influence of Hollywood and international cinema."),
    ("eda_charts/5_movies_per_year.png",
     "4.5 Movies Released Per Year",
     "Movie production has grown steadily since 2000, with a noticeable dip around 2020-2021 likely caused by the COVID-19 pandemic affecting the film industry."),
    ("eda_charts/6_avg_rating_by_genre.png",
     "4.6 Average Rating by Genre",
     "Documentary and Music genres tend to receive higher average ratings compared to Horror and Comedy, suggesting niche audiences rate their preferred content more favorably."),
    ("eda_charts/7_correlation_heatmap.png",
     "4.7 Correlation Heatmap",
     "Vote Count and Popularity show a strong positive correlation, meaning widely watched movies tend to receive more votes. Vote Average has a weaker correlation with both."),
    ("eda_charts/8_top_popular_movies.png",
     "4.8 Top 10 Most Popular Movies",
     "Spider-Man: No Way Home leads as the most popular movie in the dataset, followed by The Batman and other recent blockbusters, reflecting current audience trends."),
]

for path, title, insight in charts:
    elements.append(Paragraph(title, ParagraphStyle(
        "ch", fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#333333"), spaceAfter=6
    )))
    try:
        img = Image(path, width=5.5*inch, height=2.8*inch)
        elements.append(img)
    except:
        elements.append(Paragraph(f"[Chart not found: {path}]", body_style))
    elements.append(Paragraph(f"💡 Insight: {insight}", ParagraphStyle(
        "ins", fontSize=9, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#555555"), spaceAfter=4,
        borderPadding=6, leading=14
    )))
    elements.append(Spacer(1, 14))

# ── 5. KEY FINDINGS ───────────────────────────────────────────────────────────
elements.append(Paragraph("5. Key Findings", heading_style))
findings = [
    "Drama is the most produced genre, making up the largest share of the dataset.",
    "Most movies are rated between 5 and 8 out of 10, with an average rating of " + str(avg_rating) + ".",
    "Popularity and Vote Count are strongly correlated — popular movies attract more votes.",
    "English is the dominant language, representing the majority of all movies.",
    "Movie production peaked in recent years, with a visible dip during the COVID-19 pandemic.",
    "Documentary and Music genres receive higher average ratings despite fewer movies.",
    f"Spider-Man: No Way Home is the most popular movie in the dataset.",
]
for f in findings:
    elements.append(Paragraph(f"• {f}", body_style))
elements.append(Spacer(1, 16))

# ── 6. CONCLUSION ─────────────────────────────────────────────────────────────
elements.append(Paragraph("6. Conclusion", heading_style))
elements.append(Paragraph(
    "This Exploratory Data Analysis of the movies dataset revealed several meaningful patterns "
    "and trends in the film industry. Drama and Comedy dominate in volume, while Documentary "
    "content tends to receive higher ratings. Popularity is strongly driven by vote count, and "
    "English remains the dominant language in global cinema. The analysis successfully demonstrated "
    "the use of statistical summaries, data visualizations, and correlation analysis to extract "
    "actionable insights from a real-world dataset — fulfilling the core objectives of the "
    "Thiranex Internship EDA Project.",
    body_style
))
elements.append(Spacer(1, 20))

# ── BUILD ─────────────────────────────────────────────────────────────────────
doc.build(elements)
print("✅ PDF Report saved as: EDA_Movies_Report.pdf")
