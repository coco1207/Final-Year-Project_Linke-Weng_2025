import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, compare

# Load data
file_path = "/Users/cocoweng/Desktop/database/Step 2 Regression/Regression 1/Regression1Data.xlsx"
df = pd.read_excel(file_path)
df["Province"] = df["Province"].astype(str)
df["Year"] = df["Year"].astype(int)
df = df.set_index(["Province", "Year"])

# Model setup
y = df["QoL"]
X = df[["Nhotel", "Nattract", "Nagent", "trrate", "pdrate"]]
X = sm.add_constant(X)
base_model = PanelOLS(y, X, time_effects=True).fit()
print("=== Baseline Fixed Effects Model Results ===")
print(base_model.summary)

# Robustness Check: Hausman Test (FE vs RE)
re_model = RandomEffects(y, X).fit()
hausman_result = compare({"Fixed Effects": base_model, "Random Effects": re_model})
print("\n=== Hausman Test Results ===")
print(hausman_result)
