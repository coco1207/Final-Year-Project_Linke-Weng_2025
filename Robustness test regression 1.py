import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import random

# Load data
file1_path = "/Users/cocoweng/Desktop/database/Step 3 Robustness Checks/Regression 1/Regression1Data.xlsx"
df = pd.read_excel(file1_path)
df["Province"] = df["Province"].astype(str)
df["Year"] = df["Year"].astype(int)
df = df.set_index(["Province", "Year"])

# Model setup (Baseline)
y = df["QoL"]
X = df[["Nhotel", "Nattract", "Nagent", "trrate", "pdrate"]]
X = sm.add_constant(X)
base_model = PanelOLS(y, X, time_effects=True).fit()
print("=== Baseline Fixed Effects Model Results ===")
print(base_model.summary)

# Step 1: Extract unique provinces
provinces = df.index.get_level_values("Province").unique()
print(f"Total provinces: {len(provinces)}")

# Step 2: Calculate mean Nhotel per province to identify medium-scale provinces
province_means = df.groupby("Province")["Nhotel"].mean()
# Sort provinces by Nhotel and select those in the middle 50% (avoid extremes)
quantiles = province_means.quantile([0.25, 0.75])
medium_provinces = province_means[(province_means >= quantiles[0.25]) & (province_means <= quantiles[0.75])].index.tolist()

# Step 3: Randomly select one province to drop from medium-scale provinces
if len(medium_provinces) == 0:
    medium_provinces = provinces.tolist()  # Fallback to all provinces if none in middle range
dropped_province = random.choice(medium_provinces)
print(f"Randomly dropped province: {dropped_province}")

# Step 4: Create new dataset excluding the dropped province
df_reduced = df[df.index.get_level_values("Province") != dropped_province]

# Step 5: Rerun the model with reduced dataset
y_reduced = df_reduced["QoL"]
X_reduced = df_reduced[["Nhotel", "Nattract", "Nagent", "trrate", "pdrate"]]
X_reduced = sm.add_constant(X_reduced)
reduced_model = PanelOLS(y_reduced, X_reduced, time_effects=True).fit()
print(f"=== Fixed Effects Model Results (Excluding {dropped_province}) ===")
print(reduced_model.summary)

# Step 6: Compare coefficients
base_coeffs = base_model.params
reduced_coeffs = reduced_model.params
coeff_comparison = pd.DataFrame({
    "Base": base_coeffs,
    "Reduced": reduced_coeffs,
    "Change (%)": ((reduced_coeffs - base_coeffs) / base_coeffs * 100).abs()
})
print("=== Coefficient Comparison ===")
print(coeff_comparison)

# Step 7: Check stability (e.g., changes < 10%)
stable = (coeff_comparison["Change (%)"] < 10).all()
print(f"\nResults stable (all coefficient changes < 10%)? {stable}")