import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("movies_cleaned.csv")
print("Shape:", df.shape)
print(df.columns.tolist())

# -------------------------------
# 2. Select Features & Target
# -------------------------------
target = "Vote_Average"
features = ["Popularity", "Vote_Count"]

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Train Models
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    print(f"\n{name}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R2:   {r2:.3f}")

# -------------------------------
# 5. Compare Models (Bar Chart)
# -------------------------------
results_df = pd.DataFrame(results).T
print("\nModel Comparison:\n", results_df)

results_df["R2"].plot(kind="bar", title="Model Comparison - R2 Score", color="skyblue")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# -------------------------------
# 6. Best Model - Predicted vs Actual
# -------------------------------
best_model_name = results_df["R2"].idxmax()
best_model = models[best_model_name]
preds = best_model.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Vote Average")
plt.ylabel("Predicted Vote Average")
plt.title(f"Predicted vs Actual ({best_model_name})")
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.show()