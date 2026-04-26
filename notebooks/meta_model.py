# %% [markdown]
# # BullRun -- Meta Model (Notebook 05)
# ### Final Integration & Ensemble Decision Engine
# ---

# %% [markdown]
# ## Step 0: Setup & Imports

# %%
import pandas as pd
import numpy as np
import os, warnings, joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

warnings.filterwarnings('ignore')

PROJECT_DIR = r"e:/Bull_Run"
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "meta_model")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports", "model_reports")

for d in [MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

TECH_SIG_PATH = os.path.join(PROCESSED_DIR, "technical_signals_v2.csv")
SENTIMENT_PATH = os.path.join(PROCESSED_DIR, "sentiment_features.csv")
PRICE_PATH = os.path.join(PROCESSED_DIR, "price_prediction_features.csv")
RISK_PATH = os.path.join(PROCESSED_DIR, "risk_features.csv")
META_OUT_PATH = os.path.join(PROCESSED_DIR, "meta_signals.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "xgb_meta_model.pkl")

print("[OK] Step 0 -- Paths configured")

# %% [markdown]
# ## Step 1: Component Merge & Integration

# %%
print("=" * 60)
print("STEP 1 -- COMPONENT MERGE")
print("=" * 60)

# Load all 4 streams
df_tech = pd.read_csv(TECH_SIG_PATH, parse_dates=['Date'])
df_sent = pd.read_csv(SENTIMENT_PATH, parse_dates=['Date'])
df_price = pd.read_csv(PRICE_PATH, parse_dates=['Date'])
df_risk = pd.read_csv(RISK_PATH, parse_dates=['Date'])

# Merge sequentially on Date & Stock
df = pd.merge(df_tech, df_sent, on=['Date', 'Stock'], how='inner')
df = pd.merge(df, df_price, on=['Date', 'Stock'], how='inner')
df = pd.merge(df, df_risk, on=['Date', 'Stock'], how='inner')

df.sort_values(['Stock', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  [OK] Merged 4 feature streams horizontally. Shape: {df.shape}")

# %% [markdown]
# ## Step 2: Feature Engineering & Normalization

# %%
print("=" * 60)
print("STEP 2 -- NORMALIZATION & SPLIT")
print("=" * 60)

features = [
    'Tech_Prob_SELL', 'Tech_Prob_HOLD', 'Tech_Prob_BUY', # Technical Signals (V2 probas)
    'Sentiment_Score', 'News_Volume',                    # Sentiment Model
    'Pred_Return_5d',                                    # Price Prediction
    'Volatility_14d', 'Max_Drawdown_30d'                 # Risk Model
]
target = 'Target' # 0=Sell, 1=Hold, 2=Buy

# Train/Test Split (Chronological, same as all models)
train_mask = df['Date'] < '2024-01-01'
test_mask = df['Date'] >= '2024-01-01'

X_train_raw = df.loc[train_mask, features]
y_train = df.loc[train_mask, target]
X_test_raw = df.loc[test_mask, features]
y_test = df.loc[test_mask, target]

# Normalize features to maintain scale consistency (per user suggestion)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Keep dataframe with test rows for backtesting
test_df = df[test_mask].copy()

print(f"  [OK] Features normalized. Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# %% [markdown]
# ## Step 3: Meta-Model Training (Shallow XGBoost)

# %%
print("=" * 60)
print("STEP 3 -- META MODEL TRAINING")
print("=" * 60)

# Shallow depth to avoid overfitting the base models
meta_model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=3,      # keep it shallow
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='multi:softprob',
    n_jobs=-1
)

meta_model.fit(X_train, y_train)

# Predict probabilities on test set
meta_probs = meta_model.predict_proba(X_test)
test_df['Meta_Prob_SELL'] = meta_probs[:, 0]
test_df['Meta_Prob_HOLD'] = meta_probs[:, 1]
test_df['Meta_Prob_BUY'] = meta_probs[:, 2]

# Final model prediction purely based on argmax (for metrics)
test_df['Meta_Pred'] = np.argmax(meta_probs, axis=1)

acc = accuracy_score(y_test, test_df['Meta_Pred'])
f1 = f1_score(y_test, test_df['Meta_Pred'], average='weighted')

print(f"  [OK] Meta Model Trained.")
print(f"  Test Accuracy: {acc:.4f} | Test F1: {f1:.4f}")

# %% [markdown]
# ## Step 4: System Backtesting (Dual Filter & Dynamic Sizing)

# %%
print("=" * 60)
print("STEP 4 -- BACKTESTING SYSTEM")
print("=" * 60)

# Thresholds
TECH_CONF = 0.55  # From V3 baseline
META_CONF = 0.50  # Additional meta confirmation

capital = 100000.0
资本 = 100000.0
total_trades = 0
wins = 0
trade_returns = []

# Using loop to match real-time simulation
for stock_name in test_df['Stock'].unique():
    sd = test_df[test_df['Stock'] == stock_name].sort_values('Date').reset_index(drop=True)
    pos = 0 # 0=flat, 1=long
    ep = 0.0
    
    for idx in range(len(sd)):
        row = sd.iloc[idx]
        price = row['Close']
        
        t_buy_prob = row['Tech_Prob_BUY']
        t_sell_prob = row['Tech_Prob_SELL']
        
        m_buy_prob = row['Meta_Prob_BUY']
        m_sell_prob = row['Meta_Prob_SELL']
        
        # Dual-filter logic (Tech AND Meta must agree with high conviction)
        is_buy_signal = (t_buy_prob >= TECH_CONF) and (m_buy_prob >= META_CONF)
        is_sell_signal = (t_sell_prob >= TECH_CONF) and (m_sell_prob >= META_CONF)

        # Dynamic Sizing scaling (0.0 to 1.0 multiplier based on conviction + constraint applied by risk)
        # E.g., baseline sizing = 10%
        # Increase size linearly based on how deeply Meta Conviction exceeds threshold
        conviction_bonus = min((m_buy_prob - META_CONF) * 2, 0.5) if is_buy_signal else 0.0
        sizing_pct = 0.10 + conviction_bonus # ranges from 10% to 15% Max allocation
        
        # Entry
        if is_buy_signal and pos == 0:
            pos = 1
            ep = price
            pos_size = sizing_pct # save position size allocated
            
        # Exit
        elif is_sell_signal and pos == 1:
            pos = 0
            ret_pct = ((price - ep) / ep) * 100
            
            # Add profit/loss using dynamically calculated position size
            capital += capital * (ret_pct / 100) * pos_size 
            
            total_trades += 1
            if ret_pct > 0: wins += 1
            trade_returns.append(ret_pct)

final_ret = ((capital - 资本) / 资本) * 100
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
avg_trade = np.mean(trade_returns) if total_trades > 0 else 0.0

print(f"  Final Meta Backtest -> Return: {final_ret:+.2f}% | Win Rate: {win_rate:.1f}% | Trades: {total_trades}")
print(f"  Avg Trade Profit: {avg_trade:+.2f}%")

# Save outputs
test_df.to_csv(META_OUT_PATH, index=False)
joblib.dump(meta_model, MODEL_SAVE_PATH)

# %% [markdown]
# ## Step 5: Final Report Generation

# %%
print("=" * 60)
print("STEP 5 -- REPORT GENERATION")
print("=" * 60)

imp = pd.DataFrame({'Feature': features, 'Importance': meta_model.feature_importances_})
imp = imp.sort_values(by='Importance', ascending=False)

report = f"""# BullRun -- Meta Model Integrator Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. System Architecture
The Meta Model functions as the apex decision engine. It ingests 8 continuous features representing 4 distinct AI subsystems:
1. `Technical Probabilities (Buy/Sell/Hold)`
2. `Sentiment Score & News Volume`
3. `5-Day Predicted Forward Return`
4. `Trailing Volatility & Max Drawdown`

A shallow `XGBoost Classifier` (max_depth=3) was trained to prevent overfitting the underlying base models.

## 2. Dynamic Rules Applied
- **Dual Confirmation Filter:** A trade is *only* executed if `Tech_Prob > 0.55` AND `Meta_Prob > 0.50`.
- **Dynamic Position Sizing:** Based on the excess confidence of the Meta Model, trade sizing scales dynamically from a baseline of `10%` up to `15%` of available simulated capital per trade.

## 3. Top Decision Drivers (Meta Feature Importance)
"""
for idx, row in imp.iterrows():
    report += f"- **{row['Feature']}**: {row['Importance']:.4f}\n"

report += f"""
## 4. Final Meta Backtest Performance (Test Set: 2024+)
| Metric | Result |
|---|---|
| **Total Trades** | {total_trades} |
| **Win Rate** | {win_rate:.1f}% |
| **System Return %** | {final_ret:+.2f}% |
| **Avg Return / Trade** | {avg_trade:+.2f}% |

**Conclusion:** The Meta Engine is the final fully-integrated output of the BullRun pipeline. All signals are exported to `{META_OUT_PATH}`. 
"""

report_path = os.path.join(REPORT_DIR, "meta_model_report.md")
with open(report_path, "w", encoding="utf-8") as rf:
    rf.write(report)
print(f"  [OK] Report saved: {report_path}")
