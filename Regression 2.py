import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, compare
from linearmodels.iv import IV2SLS

# Load data
file_path = "/Users/cocoweng/Desktop/database/Step 2 Regression/Regression 2/Regression2Data.xlsx"
df = pd.read_excel(file_path)

# Calculate interaction terms
df["interaction1"] = df["Nhotel"] * df["STIndex"]
df["interaction2"] = df["Nattract"] * df["STIndex"]
df["interaction3"] = df["Nagent"] * df["STIndex"]

# Set panel structure
df["Province"] = df["Province"].astype(str)
df["Year"] = df["Year"].astype(int)
df = df.set_index(["Province", "Year"])

# Model setup
y = df["QoL"]
X = df[["interaction1", "interaction2", "interaction3", "trrate", "pdrate"]]
X = sm.add_constant(X)

# Baseline Fixed Effects Model
base_model = PanelOLS(y, X, time_effects=True, drop_absorbed=False).fit()
print("=== Baseline Fixed Effects Model Results ===")
print(base_model.summary)

# Robustness Check: Hausman Test (FE vs RE)
re_model = RandomEffects(y, X).fit()
hausman_result = compare({"Fixed Effects": base_model, "Random Effects": re_model})
print("\n=== Hausman Test Results ===")
print(hausman_result)
