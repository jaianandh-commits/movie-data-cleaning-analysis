import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("📦 Loading dataset...")
df = pd.read_csv("train.csv")
print(f"✅ Loaded {df.shape[0]} records with {df.shape[1]} columns\n")

# ── 2. DATA CLEANING ──────────────────────────────────────────────────────────
print("🧹 Cleaning data...")
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Month_Name"] = df["Order Date"].dt.strftime("%b")
df["Days_to_Ship"] = (df["Ship Date"] - df["Order Date"]).dt.days
df.dropna(inplace=True)
print(f"✅ Clean dataset: {df.shape[0]} records\n")

# ── 3. STATISTICAL SUMMARY ────────────────────────────────────────────────────
print("=" * 55)
print("        📊 STATISTICAL SUMMARY")
print("=" * 55)
print(f"  Total Orders       : {len(df):,}")
print(f"  Total Revenue      : ${df['Sales'].sum():,.2f}")
print(f"  Average Sale       : ${df['Sales'].mean():,.2f}")
print(f"  Highest Sale       : ${df['Sales'].max():,.2f}")
print(f"  Lowest Sale        : ${df['Sales'].min():,.2f}")
print(f"  Total Customers    : {df['Customer ID'].nunique():,}")
print(f"  Total Products     : {df['Product ID'].nunique():,}")
print(f"  Categories         : {df['Category'].nunique()}")
print(f"  Regions            : {df['Region'].nunique()}")
print(f"  Avg Days to Ship   : {df['Days_to_Ship'].mean():.1f} days")
print("=" * 55)
print()

# ── 4. VISUALIZATIONS ─────────────────────────────────────────────────────────
os.makedirs("retail_charts", exist_ok=True)
sns.set_theme(style="darkgrid")
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
          "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#7B2D8B"]

# Chart 1: Sales by Category
print("📊 Generating Chart 1: Sales by Category...")
cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(cat_sales.index, cat_sales.values, color=COLORS[:3])
ax.set_title("Total Sales by Category", fontsize=16, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Total Sales ($)")
for bar, val in zip(bars, cat_sales.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("retail_charts/1_sales_by_category.png", dpi=150)
plt.close()

# Chart 2: Sales by Region
print("📊 Generating Chart 2: Sales by Region...")
reg_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
wedges, texts, autotexts = ax.pie(
    reg_sales.values, labels=reg_sales.index,
    autopct="%1.1f%%", colors=COLORS[:4], startangle=140
)
ax.set_title("Sales Distribution by Region", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("retail_charts/2_sales_by_region.png", dpi=150)
plt.close()

# Chart 3: Monthly Sales Trend
print("📊 Generating Chart 3: Monthly Sales Trend...")
monthly = df.groupby(["Year", "Month"])["Sales"].sum().reset_index()
monthly["Period"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)
monthly = monthly.sort_values("Period")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly["Period"], monthly["Sales"], color="#2E86AB",
        linewidth=2, marker="o", markersize=4)
ax.fill_between(range(len(monthly)), monthly["Sales"], alpha=0.15, color="#2E86AB")
ax.set_xticks(range(0, len(monthly), 3))
ax.set_xticklabels(monthly["Period"].iloc[::3], rotation=45, ha="right")
ax.set_title("Monthly Sales Trend", fontsize=16, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Total Sales ($)")
plt.tight_layout()
plt.savefig("retail_charts/3_monthly_sales_trend.png", dpi=150)
plt.close()

# Chart 4: Top 10 Sub-Categories
print("📊 Generating Chart 4: Top Sub-Categories...")
sub_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(sub_sales.index[::-1], sub_sales.values[::-1], color=COLORS)
ax.set_title("Top 10 Sub-Categories by Sales", fontsize=16, fontweight="bold")
ax.set_xlabel("Total Sales ($)")
for bar, val in zip(bars, sub_sales.values[::-1]):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
            f"${val:,.0f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("retail_charts/4_top_subcategories.png", dpi=150)
plt.close()

# Chart 5: Sales by Segment
print("📊 Generating Chart 5: Sales by Segment...")
seg_sales = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(seg_sales.index, seg_sales.values, color=COLORS[:3])
ax.set_title("Total Sales by Customer Segment", fontsize=16, fontweight="bold")
ax.set_xlabel("Segment")
ax.set_ylabel("Total Sales ($)")
for bar, val in zip(bars, seg_sales.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("retail_charts/5_sales_by_segment.png", dpi=150)
plt.close()

# Chart 6: Top 10 States by Sales
print("📊 Generating Chart 6: Top 10 States...")
state_sales = df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(state_sales.index[::-1], state_sales.values[::-1], color=COLORS)
ax.set_title("Top 10 States by Sales", fontsize=16, fontweight="bold")
ax.set_xlabel("Total Sales ($)")
for bar, val in zip(bars, state_sales.values[::-1]):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
            f"${val:,.0f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("retail_charts/6_top_states.png", dpi=150)
plt.close()

# Chart 7: Ship Mode Distribution
print("📊 Generating Chart 7: Ship Mode...")
ship_counts = df["Ship Mode"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(ship_counts.index, ship_counts.values, color=COLORS[:4])
ax.set_title("Orders by Ship Mode", fontsize=16, fontweight="bold")
ax.set_xlabel("Ship Mode")
ax.set_ylabel("Number of Orders")
for bar, val in zip(bars, ship_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(val), ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("retail_charts/7_ship_mode.png", dpi=150)
plt.close()

# Chart 8: Sales Distribution (histogram)
print("📊 Generating Chart 8: Sales Distribution...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df[df["Sales"] < 1000]["Sales"], bins=40,
        color="#2E86AB", edgecolor="white", alpha=0.85)
ax.axvline(df["Sales"].mean(), color="#C73E1D", linestyle="--",
           linewidth=2, label=f"Mean: ${df['Sales'].mean():.2f}")
ax.set_title("Sales Amount Distribution (Under $1000)", fontsize=16, fontweight="bold")
ax.set_xlabel("Sales Amount ($)")
ax.set_ylabel("Number of Orders")
ax.legend()
plt.tight_layout()
plt.savefig("retail_charts/8_sales_distribution.png", dpi=150)
plt.close()

print("\n✅ All 8 charts saved in retail_charts/ folder!")

# ── 5. PREDICTION MODEL ───────────────────────────────────────────────────────
print("\n🤖 Training Sales Prediction Model...")

le_cat = LabelEncoder()
le_seg = LabelEncoder()
le_reg = LabelEncoder()
le_ship = LabelEncoder()

df["Category_enc"] = le_cat.fit_transform(df["Category"])
df["Segment_enc"] = le_seg.fit_transform(df["Segment"])
df["Region_enc"] = le_reg.fit_transform(df["Region"])
df["ShipMode_enc"] = le_ship.fit_transform(df["Ship Mode"])

features = ["Category_enc", "Segment_enc", "Region_enc",
            "ShipMode_enc", "Month", "Year", "Days_to_Ship"]
X = df[features]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Trained!")
print(f"   RMSE     : ${rmse:.2f}")
print(f"   R² Score : {r2:.4f}")

# Chart 9: Predicted vs Actual
print("📊 Generating Chart 9: Predicted vs Actual...")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.3, color="#2E86AB", s=15)
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()], "r--", linewidth=2)
ax.set_title("Predicted vs Actual Sales", fontsize=16, fontweight="bold")
ax.set_xlabel("Actual Sales ($)")
ax.set_ylabel("Predicted Sales ($)")
plt.tight_layout()
plt.savefig("retail_charts/9_predicted_vs_actual.png", dpi=150)
plt.close()

print("\n🎉 Retail Project Complete!")
print("=" * 55)
print(f"  Total Revenue    : ${df['Sales'].sum():,.2f}")
print(f"  Model RMSE       : ${rmse:.2f}")
print(f"  Model R² Score   : {r2:.4f}")
print("=" * 55)