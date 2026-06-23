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
df = pd.read_csv("train.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Days_to_Ship"] = (df["Ship Date"] - df["Order Date"]).dt.days
df.dropna(inplace=True)

# ── STYLES ────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    "Retail_Sales_Report.pdf", pagesize=A4,
    rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50
)
styles = getSampleStyleSheet()
BLUE = colors.HexColor("#2E86AB")
WHITE = colors.white

title_style = ParagraphStyle("title", fontSize=22, fontName="Helvetica-Bold",
    textColor=BLUE, spaceAfter=6, alignment=1)
subtitle_style = ParagraphStyle("subtitle", fontSize=12, fontName="Helvetica",
    textColor=colors.HexColor("#555555"), spaceAfter=4, alignment=1)
heading_style = ParagraphStyle("heading", fontSize=13, fontName="Helvetica-Bold",
    textColor=BLUE, spaceAfter=6)
body_style = ParagraphStyle("body", fontSize=10, fontName="Helvetica",
    textColor=colors.HexColor("#333333"), spaceAfter=4, leading=16)
insight_style = ParagraphStyle("insight", fontSize=9, fontName="Helvetica-Oblique",
    textColor=colors.HexColor("#555555"), spaceAfter=4, leading=14)
chart_title_style = ParagraphStyle("ch", fontSize=11, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#333333"), spaceAfter=6)

def divider():
    return Table([[""]], colWidths=[500], rowHeights=[2],
        style=TableStyle([("BACKGROUND", (0,0), (-1,-1), BLUE)]))

def make_table(data, col_widths):
    t = Table(data, colWidths=col_widths)
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
    return t

elements = []

# ── TITLE PAGE ────────────────────────────────────────────────────────────────
elements.append(Spacer(1, 60))
elements.append(Paragraph("🛒 Real-World Data Project", title_style))
elements.append(Paragraph("Retail Sales Analysis & Prediction", title_style))
elements.append(Spacer(1, 12))
elements.append(Paragraph("Thiranex Internship Project — Finance, Health or Retail", subtitle_style))
elements.append(Paragraph(f"Submitted on: {datetime.now().strftime('%d %B %Y')}", subtitle_style))
elements.append(Spacer(1, 30))
elements.append(divider())
elements.append(Spacer(1, 30))

# ── 1. INTRODUCTION ───────────────────────────────────────────────────────────
elements.append(Paragraph("1. Introduction", heading_style))
elements.append(Paragraph(
    "This report presents an end-to-end data analysis and sales prediction project "
    "using a real-world Retail Sales dataset from a global superstore. The project "
    "applies data science techniques including data cleaning, exploratory analysis, "
    "visualization, and machine learning to uncover business insights and predict "
    "future sales. The dataset contains 9,800 orders across multiple categories, "
    "regions, and customer segments.", body_style))
elements.append(Spacer(1, 12))

# ── 2. DATASET OVERVIEW ───────────────────────────────────────────────────────
elements.append(Paragraph("2. Dataset Overview", heading_style))
overview_data = [
    ["Attribute", "Details"],
    ["Dataset", "Global Superstore Retail Sales"],
    ["Total Records", f"{len(df):,} orders"],
    ["Total Columns", "18 columns"],
    ["Date Range", f"{df['Year'].min()} — {df['Year'].max()}"],
    ["Categories", "Furniture, Office Supplies, Technology"],
    ["Regions", "East, West, Central, South"],
    ["Customer Segments", "Consumer, Corporate, Home Office"],
]
elements.append(make_table(overview_data, [200, 280]))
elements.append(Spacer(1, 16))

# ── 3. STATISTICAL SUMMARY ────────────────────────────────────────────────────
elements.append(Paragraph("3. Statistical Summary", heading_style))
stats_data = [
    ["Metric", "Value"],
    ["Total Revenue", f"${df['Sales'].sum():,.2f}"],
    ["Average Sale Amount", f"${df['Sales'].mean():,.2f}"],
    ["Highest Single Sale", f"${df['Sales'].max():,.2f}"],
    ["Lowest Single Sale", f"${df['Sales'].min():,.2f}"],
    ["Total Unique Customers", f"{df['Customer ID'].nunique():,}"],
    ["Total Unique Products", f"{df['Product ID'].nunique():,}"],
    ["Average Days to Ship", f"{df['Days_to_Ship'].mean():.1f} days"],
    ["Most Popular Category", df.groupby('Category')['Sales'].sum().idxmax()],
    ["Top Region by Sales", df.groupby('Region')['Sales'].sum().idxmax()],
]
elements.append(make_table(stats_data, [200, 280]))
elements.append(Spacer(1, 20))

# ── 4. VISUALIZATIONS ─────────────────────────────────────────────────────────
elements.append(Paragraph("4. Visualizations & Insights", heading_style))

charts = [
    ("retail_charts/1_sales_by_category.png",
     "4.1 Sales by Category",
     "Technology leads in total revenue, followed by Furniture and Office Supplies. Technology products have higher unit prices which drives the total sales value."),
    ("retail_charts/2_sales_by_region.png",
     "4.2 Sales by Region",
     "The West region contributes the highest share of total sales, followed by the East. The Central region has the lowest contribution, indicating a potential growth opportunity."),
    ("retail_charts/3_monthly_sales_trend.png",
     "4.3 Monthly Sales Trend",
     "Sales show a clear upward trend over the years with seasonal peaks towards the end of each year (Q4), likely driven by holiday shopping and year-end business purchases."),
    ("retail_charts/4_top_subcategories.png",
     "4.4 Top Sub-Categories by Sales",
     "Phones and Chairs are the top-selling sub-categories, reflecting strong demand for technology and office furniture in the corporate segment."),
    ("retail_charts/5_sales_by_segment.png",
     "4.5 Sales by Customer Segment",
     "The Consumer segment generates the most revenue, followed by Corporate and Home Office. Targeted marketing for Corporate clients could significantly boost revenue."),
    ("retail_charts/6_top_states.png",
     "4.6 Top 10 States by Sales",
     "California leads all states by a wide margin, followed by New York and Texas. These three states alone account for a significant portion of total national revenue."),
    ("retail_charts/7_ship_mode.png",
     "4.7 Orders by Ship Mode",
     "Standard Class is the most preferred shipping mode, used for majority of orders. Same Day delivery is least used, suggesting customers prioritize cost over speed."),
    ("retail_charts/8_sales_distribution.png",
     "4.8 Sales Amount Distribution",
     "Most orders fall below $500, with the distribution heavily right-skewed. A small number of high-value orders significantly impact the total revenue figures."),
    ("retail_charts/9_predicted_vs_actual.png",
     "4.9 Predicted vs Actual Sales",
     "The Linear Regression model captures the general trend of sales. The spread around the diagonal line indicates room for improvement with more advanced models like Random Forest."),
]

for path, title, insight in charts:
    elements.append(Paragraph(title, chart_title_style))
    try:
        img = Image(path, width=5.5*inch, height=2.8*inch)
        elements.append(img)
    except:
        elements.append(Paragraph(f"[Chart not found: {path}]", body_style))
    elements.append(Paragraph(f"💡 Insight: {insight}", insight_style))
    elements.append(Spacer(1, 14))

# ── 5. PREDICTION MODEL ───────────────────────────────────────────────────────
elements.append(Paragraph("5. Sales Prediction Model", heading_style))
elements.append(Paragraph(
    "A Linear Regression model was trained to predict sales amounts based on features "
    "including product category, customer segment, region, shipping mode, month, year, "
    "and days to ship. The dataset was split 80/20 for training and testing.",
    body_style))
elements.append(Spacer(1, 8))
model_data = [
    ["Metric", "Value"],
    ["Algorithm", "Linear Regression"],
    ["Training Split", "80% Train / 20% Test"],
    ["RMSE", "$699.62"],
    ["R² Score", "-0.0007"],
    ["Features Used", "Category, Segment, Region, Ship Mode, Month, Year, Days to Ship"],
]
elements.append(make_table(model_data, [200, 280]))
elements.append(Spacer(1, 8))
elements.append(Paragraph(
    "The low R² score indicates that sales are influenced by many other factors beyond "
    "the available features, such as promotions, discounts, and market trends. "
    "Future improvements could include using Random Forest or XGBoost models with "
    "additional features like discount rates and profit margins.",
    body_style))
elements.append(Spacer(1, 16))

# ── 6. KEY FINDINGS ───────────────────────────────────────────────────────────
elements.append(Paragraph("6. Key Findings", heading_style))
findings = [
    "Technology is the highest revenue-generating category with strong unit prices.",
    "The West region leads all regions in total sales contribution.",
    "Sales follow a seasonal pattern with peaks in Q4 each year.",
    "Phones and Chairs are the top-selling sub-categories by revenue.",
    "California, New York, and Texas are the top 3 states by sales.",
    "Most orders are below $500, but high-value orders drive total revenue.",
    "Standard Class shipping is preferred by the majority of customers.",
    "Consumer segment generates the most revenue among all customer segments.",
]
for f in findings:
    elements.append(Paragraph(f"• {f}", body_style))
elements.append(Spacer(1, 16))

# ── 7. CONCLUSION ─────────────────────────────────────────────────────────────
elements.append(Paragraph("7. Conclusion", heading_style))
elements.append(Paragraph(
    "This project successfully demonstrated an end-to-end real-world data analysis "
    "on a retail sales dataset. Through data cleaning, statistical analysis, and "
    "9 detailed visualizations, meaningful business insights were uncovered about "
    "customer behavior, regional performance, and product trends. A machine learning "
    "model was also built to predict sales values. This project fulfills all the "
    "requirements of the Thiranex Internship Real-World Data Project — applying "
    "data science skills in a real retail business context.",
    body_style))

# ── BUILD PDF ─────────────────────────────────────────────────────────────────
doc.build(elements)
print("✅ PDF Report saved as: Retail_Sales_Report.pdf")
