# %% [markdown]
# # BullRun -- Price Prediction Model (Notebook 03)
# ### LightGBM Regressor for 5-Day Forward Return
# ---

# %% [markdown]
# ## Step 0: Setup & Imports

# %%
import pandas as pd
import numpy as np
import os, warnings, joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

warnings.filterwarnings('ignore')

PROJECT_DIR = r"e:/Bull_Run"
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "price_model")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports", "model_reports")

for d in [MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

TECH_FEATURES_PATH = os.path.join(PROCESSED_DIR, "technical_features.csv")
PRICE_OUT_PATH = os.path.join(PROCESSED_DIR, "price_prediction_features.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "lgb_price_model.pkl")

print("[OK] Step 0 -- Paths configured")

# %% [markdown]
# ## Step 1: Data Input & Target Creation

# %%
print("=" * 60)
print("STEP 1 -- DATA LOAD & TARGET CREATION")
print("=" * 60)

df = pd.read_csv(TECH_FEATURES_PATH, parse_dates=['Date'])
df.sort_values(['Stock', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Generate 5-day forward continuous return target
df['Target_5d_Return'] = df.groupby('Stock')['Close'].transform(lambda x: (x.shift(-5) - x) / x * 100)

# Drop NaNs at the end caused by shift
df = df.dropna(subset=['Target_5d_Return'])
df.reset_index(drop=True, inplace=True)

print(f"  [OK] Data loaded and 5-day forward return target created.")
print(f"  Shape: {df.shape}")

# %% [markdown]
# ## Step 2: Feature Engineering & Train/Test Split

# %%
print("=" * 60)
print("STEP 2 -- TRAIN/TEST SPLIT (CHRONOLOGICAL)")
print("=" * 60)

# Features: Exclude non-predictive/target columns
exclude_cols = ['Date', 'Stock', 'Target', 'Target_5d_Return']
features = [c for c in df.columns if c not in exclude_cols]

# Chronological Split (Train < 2024, Test >= 2024 to match V3 test period)
train_df = df[df['Date'] < '2024-01-01'].copy()
test_df = df[df['Date'] >= '2024-01-01'].copy()

X_train, y_train = train_df[features], train_df['Target_5d_Return']
X_test, y_test = test_df[features], test_df['Target_5d_Return']

print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"  Number of features: {len(features)}")

# %% [markdown]
# ## Step 3: Model Training (LightGBM)

# %%
print("=" * 60)
print("STEP 3 -- LIGHTGBM MODEL TRAINING")
print("=" * 60)

# Fast regressor suitable for continuous financial data
model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

print(f"  [OK] Model trained. Best iteration: {model.best_iteration_}")

# %% [markdown]
# ## Step 4: Model Evaluation

# %%
print("=" * 60)
print("STEP 4 -- EVALUATION")
print("=" * 60)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"  Test RMSE: {rmse:.4f}%")
print(f"  Test MAE:  {mae:.4f}%")
print(f"  Test R2:   {r2:.4f}")

# Overall Directional Accuracy (is the sign right?)
direction_acc = np.mean(np.sign(preds) == np.sign(y_test))
print(f"  Directional Accuracy: {direction_acc*100:.2f}%")

# Generate predictions for ENTIRE dataset so Meta Model has history
df['Pred_Return_5d'] = model.predict(df[features])

# Save output
signal_df = df[['Date', 'Stock', 'Pred_Return_5d']]
signal_df.to_csv(PRICE_OUT_PATH, index=False)
print(f"  [OK] Saved price predictions to: {PRICE_OUT_PATH}")

# Save Model
joblib.dump(model, MODEL_SAVE_PATH)
print(f"  [OK] Model saved to: {MODEL_SAVE_PATH}")

# %% [markdown]
# ## Step 5: Report Generation

# %%
print("=" * 60)
print("STEP 5 -- REPORT GENERATION")
print("=" * 60)

imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
imp = imp.sort_values(by='Importance', ascending=False).head(5)

report = f"""# BullRun -- Price Prediction Model Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Context
The Price Prediction Model uses a `LightGBM Regressor` to forecast the continuous 5-day forward return percentage. This provides the Meta Model with a quantitative target expectation rather than just a categorical Buy/Sell assumption.

## 2. Evaluation Metrics (Test Set: 2024+)
| Metric | Value | Interpretation |
|---|---|---|
| **RMSE** | {rmse:.4f}% | Standard deviation of prediction errors. |
| **MAE** | {mae:.4f}% | Average absolute error in percentage points. |
| **R2 Score** | {r2:.4f} | Explained variance (typically very low in finance). |
| **Directional Accuracy** | {direction_acc*100:.2f}% | How often the model correctly predicts the sign (+/-). |

## 3. Top 5 Drivers (Feature Importance)
"""
for idx, row in imp.iterrows():
    report += f"- **{row['Feature']}**: {row['Importance']}\n"

report += f"""
## 4. Pipeline Outputs
- Predictions saved to: `{PRICE_OUT_PATH}`
- Model serialized to: `{MODEL_SAVE_PATH}`

**Status:** Ready for Meta layer integration.
"""

report_path = os.path.join(REPORT_DIR, "price_model_report.md")
with open(report_path, "w", encoding="utf-8") as rf:
    rf.write(report)
print(f"  [OK] Report saved: {report_path}")
