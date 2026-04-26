# %% [markdown]
# # 🐂 BullRun -- Technical Model Pipeline
# ### Notebook 01: XGBoost-based Buy/Sell/Hold Classifier
# ---

# %% [markdown]
# ## Step 0 -- Setup & Imports

# %%
import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# --- Paths ---
PROJECT_DIR = r"e:/Bull_Run"
DATA_PATH = os.path.join(PROJECT_DIR, "dataset", "combined_data.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "technical_model")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports", "model_reports")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiments")

for d in [MODEL_DIR, REPORT_DIR, PROCESSED_DIR, EXPERIMENT_DIR]:
    os.makedirs(d, exist_ok=True)

print("[OK] Step 0 Complete -- Libraries loaded, paths configured.")

# %% [markdown]
# ## Step 1 -- Load Dataset

# %%
print("=" * 60)
print("STEP 1 -- LOADING DATASET")
print("=" * 60)

df_raw = pd.read_csv(DATA_PATH)
print(f"  Rows    : {df_raw.shape[0]:,}")
print(f"  Columns : {df_raw.shape[1]}")
print(f"  Stocks  : {df_raw['Stock'].nunique()}")
print(f"  Columns : {list(df_raw.columns)}")

# %% [markdown]
# ## Step 2 -- Data Understanding

# %%
print("=" * 60)
print("STEP 2 -- DATA UNDERSTANDING")
print("=" * 60)

print("\n--- Data Types ---")
print(df_raw.dtypes)

print("\n--- Null Counts ---")
print(df_raw.isnull().sum())

print("\n--- Date Range ---")
print(f"  Min : {df_raw['Date'].min()}")
print(f"  Max : {df_raw['Date'].max()}")

print("\n--- Rows Per Stock (sample) ---")
stock_counts = df_raw['Stock'].value_counts()
print(stock_counts.head(10))

print(f"\n--- Stocks with < 500 rows (will be excluded) ---")
small_stocks = stock_counts[stock_counts < 500].index.tolist()
print(f"  {small_stocks if small_stocks else 'None'}")

# %% [markdown]
# ## Step 3 -- Data Cleaning

# %%
print("=" * 60)
print("STEP 3 -- DATA CLEANING")
print("=" * 60)

df = df_raw.copy()

# 3.1 Drop Adj Close (all NaN) and Source (not needed)
df.drop(columns=['Adj Close', 'Source'], inplace=True, errors='ignore')
print("  [v] Dropped 'Adj Close' (all NaN) and 'Source'")

# 3.2 Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
print("  [v] Converted Date to datetime")

# 3.3 Sort by Stock + Date
df.sort_values(['Stock', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)
print("  [v] Sorted by Stock + Date")

# 3.4 Remove stocks with too little data
min_rows = 500
stock_counts = df['Stock'].value_counts()
valid_stocks = stock_counts[stock_counts >= min_rows].index.tolist()
removed = [s for s in df['Stock'].unique() if s not in valid_stocks]
df = df[df['Stock'].isin(valid_stocks)].copy()
print(f"  [v] Kept {len(valid_stocks)} stocks, removed {len(removed)}: {removed}")

# 3.5 Remove duplicate date rows per stock
before = len(df)
df.drop_duplicates(subset=['Stock', 'Date'], keep='first', inplace=True)
print(f"  [v] Removed {before - len(df)} duplicate rows")

# 3.6 Remove rows with zero/negative prices
mask = (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)
bad_rows = (~mask).sum()
df = df[mask].copy()
print(f"  [v] Removed {bad_rows} rows with invalid prices")

# 3.7 Remove rows with zero volume
zero_vol = (df['Volume'] == 0).sum()
df = df[df['Volume'] > 0].copy()
print(f"  [v] Removed {zero_vol} rows with zero volume")

print(f"\n  Final shape: {df.shape}")
print("[OK] Step 3 Complete -- Data cleaned.")

# %% [markdown]
# ## Step 4 -- Feature Engineering (Technical Indicators)

# %%
print("=" * 60)
print("STEP 4 -- FEATURE ENGINEERING")
print("=" * 60)

def compute_features(group):
    """Compute technical indicators for a single stock group."""
    g = group.copy()
    close = g['Close']
    high = g['High']
    low = g['Low']
    volume = g['Volume']

    # --- Trend Indicators ---
    g['SMA_10'] = close.rolling(10).mean()
    g['SMA_20'] = close.rolling(20).mean()
    g['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    g['EMA_26'] = close.ewm(span=26, adjust=False).mean()

    # --- MACD ---
    g['MACD'] = g['EMA_12'] - g['EMA_26']
    g['MACD_Signal'] = g['MACD'].ewm(span=9, adjust=False).mean()
    g['MACD_Hist'] = g['MACD'] - g['MACD_Signal']

    # --- RSI (14-day) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    g['RSI_14'] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    g['BB_Upper'] = sma20 + 2 * std20
    g['BB_Lower'] = sma20 - 2 * std20
    g['BB_Width'] = (g['BB_Upper'] - g['BB_Lower']) / (sma20 + 1e-10)

    # --- ATR (14-day) ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    g['ATR_14'] = tr.rolling(14).mean()

    # --- Volume Features ---
    g['Volume_SMA_10'] = volume.rolling(10).mean()
    g['Volume_Change_Pct'] = volume.pct_change() * 100

    # --- Price-Derived ---
    g['Daily_Return_Pct'] = close.pct_change() * 100
    g['HL_Range_Pct'] = ((high - low) / (close + 1e-10)) * 100
    g['CO_Change_Pct'] = ((close - g['Open']) / (g['Open'] + 1e-10)) * 100

    # --- Price Position ---
    g['Price_vs_SMA20'] = (close - sma20) / (sma20 + 1e-10)

    return g

# Apply per stock (loop for reliability across pandas versions)
result_parts = []
for stock_name, stock_group in df.groupby('Stock'):
    featured = compute_features(stock_group)
    result_parts.append(featured)
df = pd.concat(result_parts, ignore_index=True)
df.sort_values(['Stock', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  [v] Applied to {len(result_parts)} stocks")

feature_cols = [
    'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI_14',
    'BB_Upper', 'BB_Lower', 'BB_Width',
    'ATR_14',
    'Volume_SMA_10', 'Volume_Change_Pct',
    'Daily_Return_Pct', 'HL_Range_Pct', 'CO_Change_Pct',
    'Price_vs_SMA20'
]

print(f"  [v] Created {len(feature_cols)} features:")
for i, f in enumerate(feature_cols, 1):
    print(f"    {i:2d}. {f}")
print("[OK] Step 4 Complete -- Features engineered.")

# %% [markdown]
# ## Step 5 -- Target Creation (Buy / Sell / Hold)

# %%
print("=" * 60)
print("STEP 5 -- TARGET CREATION")
print("=" * 60)

FORWARD_DAYS = 5
THRESHOLD_PCT = 2.0

def create_target(group):
    g = group.copy()
    future_close = g['Close'].shift(-FORWARD_DAYS)
    pct_change = ((future_close - g['Close']) / g['Close']) * 100

    g['Target'] = 1  # HOLD by default
    g.loc[pct_change >= THRESHOLD_PCT, 'Target'] = 2   # BUY
    g.loc[pct_change <= -THRESHOLD_PCT, 'Target'] = 0   # SELL
    return g

result_parts2 = []
for stock_name, stock_group in df.groupby('Stock'):
    labeled = create_target(stock_group)
    result_parts2.append(labeled)
df = pd.concat(result_parts2, ignore_index=True)
df.sort_values(['Stock', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  Forward window : {FORWARD_DAYS} days")
print(f"  Threshold      : +/-{THRESHOLD_PCT}%")
print(f"\n  Target Distribution:")
target_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
for val, label in target_map.items():
    count = (df['Target'] == val).sum()
    pct = count / len(df) * 100
    print(f"    {label} ({val}): {count:,} rows ({pct:.1f}%)")

print("[OK] Step 5 Complete -- Targets created.")

# %% [markdown]
# ## Step 6 -- Drop NaN Rows

# %%
print("=" * 60)
print("STEP 6 -- DROP NaN ROWS")
print("=" * 60)

before = len(df)
cols_to_check = feature_cols + ['Target']
df.dropna(subset=cols_to_check, inplace=True)
df.reset_index(drop=True, inplace=True)
after = len(df)

print(f"  Rows before : {before:,}")
print(f"  Rows after  : {after:,}")
print(f"  Dropped     : {before - after:,} ({(before - after)/before*100:.1f}%)")
print("[OK] Step 6 Complete.")

# %% [markdown]
# ## Step 7 -- Save Processed Data

# %%
processed_path = os.path.join(PROCESSED_DIR, "technical_features.csv")
df.to_csv(processed_path, index=False)
print(f"  [v] Saved processed data to: {processed_path}")
print(f"    Shape: {df.shape}")

# %% [markdown]
# ## Step 8 -- Time-Based Train / Test Split

# %%
print("=" * 60)
print("STEP 8 -- TRAIN / TEST SPLIT (TIME-BASED)")
print("=" * 60)

TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'

train_df = df[df['Date'] <= TRAIN_END].copy()
test_df = df[df['Date'] >= TEST_START].copy()

X_train = train_df[feature_cols].values
y_train = train_df['Target'].astype(int).values
X_test = test_df[feature_cols].values
y_test = test_df['Target'].astype(int).values

print(f"  Train : {X_train.shape[0]:,} rows  |  {train_df['Date'].min().date()} -> {train_df['Date'].max().date()}")
print(f"  Test  : {X_test.shape[0]:,} rows  |  {test_df['Date'].min().date()} -> {test_df['Date'].max().date()}")
print(f"  Train target dist: SELL={sum(y_train==0)}, HOLD={sum(y_train==1)}, BUY={sum(y_train==2)}")
print(f"  Test  target dist: SELL={sum(y_test==0)}, HOLD={sum(y_test==1)}, BUY={sum(y_test==2)}")
print("[OK] Step 8 Complete -- Chronological split done. No data leakage.")

# %% [markdown]
# ## Step 9 -- Feature Scaling

# %%
print("=" * 60)
print("STEP 9 -- FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # fit on TRAIN only

print(f"  [v] StandardScaler fitted on training data only")
print(f"  [v] Applied to both train and test")

# Save scaler
scaler_path = os.path.join(MODEL_DIR, "scaler_v1.pkl")
joblib.dump(scaler, scaler_path)
print(f"  [v] Scaler saved: {scaler_path}")
print("[OK] Step 9 Complete.")

# %% [markdown]
# ## Step 10 -- Batch Training (Walk-Forward)

# %%
print("=" * 60)
print("STEP 10 -- BATCH TRAINING (WALK-FORWARD)")
print("=" * 60)

# Define walk-forward windows
batch_windows = [
    ('2015-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2015-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2015-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2015-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
]

batch_results = []
for i, (tr_start, tr_end, te_start, te_end) in enumerate(batch_windows, 1):
    tr_mask = (df['Date'] >= tr_start) & (df['Date'] <= tr_end)
    te_mask = (df['Date'] >= te_start) & (df['Date'] <= te_end)

    X_tr = df.loc[tr_mask, feature_cols].values
    y_tr = df.loc[tr_mask, 'Target'].astype(int).values
    X_te = df.loc[te_mask, feature_cols].values
    y_te = df.loc[te_mask, 'Target'].astype(int).values

    if len(X_te) == 0:
        continue

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    model_b = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, verbosity=0
    )
    model_b.fit(X_tr_s, y_tr)
    y_pred_b = model_b.predict(X_te_s)

    acc = accuracy_score(y_te, y_pred_b)
    f1 = f1_score(y_te, y_pred_b, average='weighted')

    batch_results.append({
        'batch': i, 'train': f'{tr_start}->{tr_end}',
        'test': f'{te_start}->{te_end}',
        'train_size': len(X_tr), 'test_size': len(X_te),
        'accuracy': round(acc, 4), 'f1_weighted': round(f1, 4)
    })
    print(f"  Batch {i}: Train {tr_start}->{tr_end} | Test {te_start}->{te_end}")
    print(f"           Acc={acc:.4f} | F1={f1:.4f} | Train={len(X_tr):,} | Test={len(X_te):,}")

print("\n[OK] Step 10 Complete -- Walk-forward batch training done.")

# %% [markdown]
# ## Step 11 -- Final Model Training

# %%
print("=" * 60)
print("STEP 11 -- FINAL MODEL TRAINING (XGBoost)")
print("=" * 60)

model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, verbosity=0
)

model.fit(X_train_scaled, y_train)
print(f"  [v] XGBoost trained on {X_train_scaled.shape[0]:,} samples, {X_train_scaled.shape[1]} features")

# Feature importance
importances = model.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
print("\n  Top 10 Feature Importances:")
for name, imp in feat_imp[:10]:
    bar = '#' * int(imp * 100)
    print(f"    {name:25s} : {imp:.4f} {bar}")

print("[OK] Step 11 Complete.")

# %% [markdown]
# ## Step 12 -- Model Evaluation

# %%
print("=" * 60)
print("STEP 12 -- MODEL EVALUATION")
print("=" * 60)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_prec = precision_score(y_test, y_test_pred, average='weighted')
test_rec = recall_score(y_test, y_test_pred, average='weighted')

print(f"\n  {'Metric':<20} {'Train':>10} {'Test':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*10}")
print(f"  {'Accuracy':<20} {train_acc:>10.4f} {test_acc:>10.4f}")
print(f"  {'F1 (weighted)':<20} {'':>10} {test_f1:>10.4f}")
print(f"  {'Precision (wt)':<20} {'':>10} {test_prec:>10.4f}")
print(f"  {'Recall (wt)':<20} {'':>10} {test_rec:>10.4f}")

print(f"\n--- Classification Report (Test Set) ---\n")
print(classification_report(y_test, y_test_pred, target_names=['SELL', 'HOLD', 'BUY']))

print(f"--- Confusion Matrix (Test Set) ---")
cm = confusion_matrix(y_test, y_test_pred)
print(f"             Predicted")
print(f"             SELL  HOLD   BUY")
for i, row_label in enumerate(['SELL ', 'HOLD ', 'BUY  ']):
    print(f"  Actual {row_label} {cm[i]}")

# %% [markdown]
# ## Step 13 -- Overfitting Check

# %%
print("=" * 60)
print("STEP 13 -- OVERFITTING CHECK")
print("=" * 60)

gap = train_acc - test_acc
print(f"  Train Accuracy : {train_acc:.4f}")
print(f"  Test  Accuracy : {test_acc:.4f}")
print(f"  Gap            : {gap:.4f}")

if gap > 0.10:
    print("  [!]  WARNING: Potential overfitting detected (gap > 10%)")
    print("  -> Consider: reduce max_depth, increase regularization, fewer estimators")
elif gap > 0.05:
    print("  [!] MODERATE: Slight overfitting (gap 5-10%). Monitor closely.")
else:
    print("  [OK] HEALTHY: No significant overfitting detected.")

# %% [markdown]
# ## Step 14 -- Backtesting

# %%
print("=" * 60)
print("STEP 14 -- BACKTESTING (Simulated Trading)")
print("=" * 60)

test_bt = test_df.copy()
test_bt = test_bt.dropna(subset=feature_cols + ['Target'])
test_bt = test_bt.reset_index(drop=True)

X_bt = scaler.transform(test_bt[feature_cols].values)
test_bt['Predicted'] = model.predict(X_bt)

initial_capital = 100000.0
capital = initial_capital
position = 0  # 0 = no position, 1 = holding
entry_price = 0.0
total_trades = 0
wins = 0
losses = 0
trade_log = []

# Run simulation per stock independently
for stock_name in test_bt['Stock'].unique():
    stock_data = test_bt[test_bt['Stock'] == stock_name].sort_values('Date').reset_index(drop=True)
    pos = 0
    ep = 0.0

    for idx in range(len(stock_data)):
        row = stock_data.iloc[idx]
        pred = int(row['Predicted'])
        price = row['Close']

        if pred == 2 and pos == 0:  # BUY signal, not holding
            pos = 1
            ep = price

        elif pred == 0 and pos == 1:  # SELL signal, holding
            pos = 0
            ret_pct = ((price - ep) / ep) * 100
            capital += capital * (ret_pct / 100) * 0.1  # 10% allocation per trade
            total_trades += 1
            if ret_pct > 0:
                wins += 1
            else:
                losses += 1
            trade_log.append({
                'stock': stock_name, 'entry': round(ep, 2),
                'exit': round(price, 2), 'return_pct': round(ret_pct, 2)
            })

final_return = ((capital - initial_capital) / initial_capital) * 100
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

print(f"  Initial Capital : ₹{initial_capital:,.0f}")
print(f"  Final Capital   : ₹{capital:,.0f}")
print(f"  Total Return    : {final_return:+.2f}%")
print(f"  Total Trades    : {total_trades}")
print(f"  Wins            : {wins}")
print(f"  Losses          : {losses}")
print(f"  Win Rate        : {win_rate:.1f}%")

if trade_log:
    returns = [t['return_pct'] for t in trade_log]
    print(f"  Avg Trade Return: {np.mean(returns):.2f}%")
    print(f"  Best Trade      : {max(returns):+.2f}%")
    print(f"  Worst Trade     : {min(returns):+.2f}%")

print("[OK] Step 14 Complete -- Backtesting done.")

# %% [markdown]
# ## Step 15 -- Save Model

# %%
print("=" * 60)
print("STEP 15 -- SAVE MODEL")
print("=" * 60)

model_path = os.path.join(MODEL_DIR, "technical_model_v1.pkl")
joblib.dump(model, model_path)
print(f"  [v] Model saved : {model_path}")

best_path = os.path.join(MODEL_DIR, "technical_model_best.pkl")
joblib.dump(model, best_path)
print(f"  [v] Best model  : {best_path}")

# Save feature list
feat_path = os.path.join(MODEL_DIR, "feature_list.json")
with open(feat_path, 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f"  [v] Feature list: {feat_path}")

print("[OK] Step 15 Complete -- Model saved.")

# %% [markdown]
# ## Step 16 -- Report Generation

# %%
print("=" * 60)
print("STEP 16 -- REPORT GENERATION")
print("=" * 60)

report = f"""# BullRun -- Technical Model Report (v1)
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Data Summary
| Item | Value |
|---|---|
| Total Rows (raw) | {df_raw.shape[0]:,} |
| Total Rows (cleaned) | {df.shape[0]:,} |
| Stocks | {df['Stock'].nunique()} |
| Date Range | {df['Date'].min().date()} to {df['Date'].max().date()} |
| Columns Dropped | Adj Close (all NaN), Source |

## 2. Features Used ({len(feature_cols)} total)
| # | Feature |
|---|---|
"""
for i, f_name in enumerate(feature_cols, 1):
    report += f"| {i} | {f_name} |\n"

report += f"""
## 3. Target Configuration
| Parameter | Value |
|---|---|
| Forward Window | {FORWARD_DAYS} days |
| Threshold | +/-{THRESHOLD_PCT}% |
| BUY count | {sum(y_train==2) + sum(y_test==2):,} |
| HOLD count | {sum(y_train==1) + sum(y_test==1):,} |
| SELL count | {sum(y_train==0) + sum(y_test==0):,} |

## 4. Model Configuration
| Parameter | Value |
|---|---|
| Algorithm | XGBoost (XGBClassifier) |
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| Train Period | up to {TRAIN_END} |
| Test Period | {TEST_START} onwards |

## 5. Performance
| Metric | Train | Test |
|---|---|---|
| Accuracy | {train_acc:.4f} | {test_acc:.4f} |
| F1 (weighted) | -- | {test_f1:.4f} |
| Precision (wt) | -- | {test_prec:.4f} |
| Recall (wt) | -- | {test_rec:.4f} |
| Overfit Gap | {gap:.4f} | -- |

## 6. Batch Training Results
| Batch | Train Window | Test Window | Accuracy | F1 |
|---|---|---|---|---|
"""
for b in batch_results:
    report += f"| {b['batch']} | {b['train']} | {b['test']} | {b['accuracy']} | {b['f1_weighted']} |\n"

report += f"""
## 7. Backtesting Results
| Metric | Value |
|---|---|
| Initial Capital | Rs.{initial_capital:,.0f} |
| Final Capital | Rs.{capital:,.0f} |
| Total Return | {final_return:+.2f}% |
| Total Trades | {total_trades} |
| Win Rate | {win_rate:.1f}% |

## 8. Top Features
| Feature | Importance |
|---|---|
"""
for name, imp in feat_imp[:10]:
    report += f"| {name} | {imp:.4f} |\n"

report += f"""
## 9. Insights
- Model uses {len(feature_cols)} technical indicators as input features
- Walk-forward batch training validates consistency across multiple time periods
- Chronological train/test split ensures no data leakage
- Backtest simulates realistic trading with 10% capital allocation per trade

## 10. Recommended Next Steps
- Tune hyperparameters with Optuna
- Add class weighting to handle HOLD imbalance
- Integrate sentiment signals from Notebook 02
- Consider ensemble with LSTM for sequence memory
"""

report_path = os.path.join(REPORT_DIR, "technical_model_report_v1.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  [v] Report saved: {report_path}")

# Save experiment log
exp_log = {
    'run_id': 'run_001_technical_xgb',
    'timestamp': datetime.now().isoformat(),
    'model': 'XGBoost',
    'version': 'v1',
    'features': len(feature_cols),
    'train_rows': int(X_train.shape[0]),
    'test_rows': int(X_test.shape[0]),
    'train_accuracy': round(train_acc, 4),
    'test_accuracy': round(test_acc, 4),
    'test_f1': round(test_f1, 4),
    'backtest_return': round(final_return, 2),
    'backtest_win_rate': round(win_rate, 1),
    'total_trades': total_trades,
}
exp_path = os.path.join(EXPERIMENT_DIR, "runs", "run_001_technical_xgb.json")
with open(exp_path, 'w') as f:
    json.dump(exp_log, f, indent=2)
print(f"  [v] Experiment log: {exp_path}")

# Save technical signals for Meta Model
signal_cols = ['Date', 'Stock', 'Close', 'Target', 'Predicted'] if 'Predicted' in test_bt.columns else ['Date', 'Stock', 'Close', 'Target']
# Generate predictions on full dataset for downstream use
X_all_scaled = scaler.transform(df[feature_cols].values)
df['Tech_Pred'] = model.predict(X_all_scaled)
proba = model.predict_proba(X_all_scaled)
df['Tech_Prob_SELL'] = proba[:, 0]
df['Tech_Prob_HOLD'] = proba[:, 1]
df['Tech_Prob_BUY'] = proba[:, 2]

signal_path = os.path.join(PROCESSED_DIR, "technical_signals.csv")
df[['Date', 'Stock', 'Close', 'Target', 'Tech_Pred',
    'Tech_Prob_SELL', 'Tech_Prob_HOLD', 'Tech_Prob_BUY']].to_csv(signal_path, index=False)
print(f"  [v] Signals saved: {signal_path}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE -- Technical Model v1 is ready!")
print("=" * 60)
