# %% [markdown]
# # BullRun -- Risk Model (Notebook 04)
# ### Trailing Volatility and Maximum Drawdown metrics
# ---

# %% [markdown]
# ## Step 0: Setup & Imports

# %%
import pandas as pd
import numpy as np
import os, warnings
from datetime import datetime

warnings.filterwarnings('ignore')

PROJECT_DIR = r"e:/Bull_Run"
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports", "model_reports")

os.makedirs(REPORT_DIR, exist_ok=True)

TECH_FEATURES_PATH = os.path.join(PROCESSED_DIR, "technical_features.csv")
RISK_OUT_PATH = os.path.join(PROCESSED_DIR, "risk_features.csv")

print("[OK] Step 0 -- Paths configured")

# %% [markdown]
# ## Step 1: Data Input

# %%
print("=" * 60)
print("STEP 1 -- DATA LOAD")
print("=" * 60)

df = pd.read_csv(TECH_FEATURES_PATH, parse_dates=['Date'])
df.sort_values(['Stock', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# We only need High, Low, Close for risk metrics
risk_df = df[['Date', 'Stock', 'High', 'Low', 'Close']].copy()

print(f"  [OK] Data loaded. Shape: {risk_df.shape}")

# %% [markdown]
# ## Step 2: Risk Feature Engineering

# %%
print("=" * 60)
print("STEP 2 -- VOLATILITY AND DRAWDOWN")
print("=" * 60)

# Calculate Daily Returns
risk_df['Return'] = risk_df.groupby('Stock')['Close'].pct_change()

# Metric 1: 14-Day Trailing Volatility (Annualized)
risk_df['Volatility_14d'] = risk_df.groupby('Stock')['Return'].rolling(window=14).std().reset_index(0, drop=True) * np.sqrt(252) * 100

# Metric 2: 30-Day Maximum Drawdown
# Drawdown = (Price / Highest_Price_in_Window) - 1
def calculate_max_drawdown(series):
    rolling_max = series.rolling(window=30, min_periods=1).max()
    drawdown = (series / rolling_max) - 1.0
    return drawdown * 100 # percentage

risk_df['Max_Drawdown_30d'] = risk_df.groupby('Stock')['Close'].transform(calculate_max_drawdown)

# Fill NaNs from rolling windows (forward fill, then 0 for initial rows)
risk_df.fillna(0, inplace=True)

print(f"  [OK] Risk features calculated successfully.")

# %% [markdown]
# ## Step 3: Save Processed Data

# %%
print("=" * 60)
print("STEP 3 -- OUTPUT GENERATION")
print("=" * 60)

out_df = risk_df[['Date', 'Stock', 'Volatility_14d', 'Max_Drawdown_30d']]
out_df.to_csv(RISK_OUT_PATH, index=False)

print(f"  [v] Final Risk Features saved to: {RISK_OUT_PATH}")
print(f"  Output shape: {out_df.shape}")

# %% [markdown]
# ## Step 4: Report Generation

# %%
print("=" * 60)
print("STEP 4 -- REPORT GENERATION")
print("=" * 60)

avg_vol = out_df['Volatility_14d'].mean()
avg_dd = out_df['Max_Drawdown_30d'].mean()

report = f"""# BullRun -- Risk Model Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Context
The Risk Model generates two critical continuous features that capture local market conditions for each stock. This acts as a dampening shield for the Meta Model.

## 2. Risk Metrics
- **Volatility_14d:** Annualized trailing 14-day volatility (%). If volatility is exploding, the Meta Model can either scale down position size or veto trades to avoid chop.
  - *Average Volatility in dataset:* {avg_vol:.2f}%
- **Max_Drawdown_30d:** Measures the largest peak-to-trough drop over the last 30 trading days. Helps the system identify if it's trading into a falling knife or a gentle retracement.
  - *Average Drawdown in dataset:* {avg_dd:.2f}%

## 3. Pipeline Output
- Final dataset serialized to: `{RISK_OUT_PATH}`
- Shape: {out_df.shape}

**Status:** Ready for Meta layer integration.
"""

report_path = os.path.join(REPORT_DIR, "risk_model_report.md")
with open(report_path, "w", encoding="utf-8") as rf:
    rf.write(report)
print(f"  [OK] Report saved: {report_path}")
