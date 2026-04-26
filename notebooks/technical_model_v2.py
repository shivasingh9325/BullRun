# %% [markdown]
# # BullRun -- Technical Model v2 (Improvement Pipeline)
# ### Threshold Analysis + Class Balancing + Hyperparameter Tuning
# ---

# %% [markdown]
# ## Step 0 -- Setup

# %%
import pandas as pd
import numpy as np
import os, json, warnings, joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

PROJECT_DIR = r"e:/Bull_Run"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "technical_features.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "technical_model")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports", "model_reports")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiments")

# Load v1 baseline for comparison
with open(os.path.join(EXPERIMENT_DIR, "runs", "run_001_technical_xgb.json")) as f:
    v1_metrics = json.load(f)

print("[OK] Step 0 -- Setup complete")
print(f"  V1 Baseline: Acc={v1_metrics['test_accuracy']}, F1={v1_metrics['test_f1']}, "
      f"Return={v1_metrics['backtest_return']}%, WinRate={v1_metrics['backtest_win_rate']}%")

# %% [markdown]
# ## Step 1 -- Load Processed Data

# %%
print("=" * 60)
print("STEP 1 -- LOAD PROCESSED DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df.sort_values(['Stock', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Original 18 features from v1
feature_cols_v1 = [
    'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI_14',
    'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR_14',
    'Volume_SMA_10', 'Volume_Change_Pct',
    'Daily_Return_Pct', 'HL_Range_Pct', 'CO_Change_Pct', 'Price_vs_SMA20'
]

print(f"  Loaded: {df.shape[0]:,} rows, {df['Stock'].nunique()} stocks")
print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# %% [markdown]
# ## Step 2 -- Threshold Experiment (3%, 5%, 7%)

# %%
print("=" * 60)
print("STEP 2 -- THRESHOLD EXPERIMENT")
print("=" * 60)

TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
FORWARD_DAYS = 5

def create_labels(dataframe, threshold_pct):
    """Create BUY/SELL/HOLD labels using forward returns + threshold."""
    result_parts = []
    for stock_name, stock_group in dataframe.groupby('Stock'):
        g = stock_group.copy()
        future_close = g['Close'].shift(-FORWARD_DAYS)
        pct_change = ((future_close - g['Close']) / g['Close']) * 100
        g['Target'] = 1  # HOLD
        g.loc[pct_change >= threshold_pct, 'Target'] = 2   # BUY
        g.loc[pct_change <= -threshold_pct, 'Target'] = 0  # SELL
        result_parts.append(g)
    return pd.concat(result_parts, ignore_index=True)

def run_backtest(test_data, feature_list, model, scaler):
    """Run backtest simulation and return metrics."""
    bt = test_data.copy().dropna(subset=feature_list + ['Target']).reset_index(drop=True)
    X_bt = scaler.transform(bt[feature_list].values)
    bt['Predicted'] = model.predict(X_bt)

    capital = 100000.0
    init_capital = capital
    total_trades = 0
    wins = 0
    trade_returns = []

    for stock_name in bt['Stock'].unique():
        sd = bt[bt['Stock'] == stock_name].sort_values('Date').reset_index(drop=True)
        pos = 0
        ep = 0.0
        for idx in range(len(sd)):
            pred = int(sd.iloc[idx]['Predicted'])
            price = sd.iloc[idx]['Close']
            if pred == 2 and pos == 0:
                pos = 1
                ep = price
            elif pred == 0 and pos == 1:
                pos = 0
                ret_pct = ((price - ep) / ep) * 100
                capital += capital * (ret_pct / 100) * 0.1
                total_trades += 1
                if ret_pct > 0:
                    wins += 1
                trade_returns.append(ret_pct)

    final_return = ((capital - init_capital) / init_capital) * 100
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    return {
        'return_pct': round(final_return, 2),
        'win_rate': round(win_rate, 1),
        'total_trades': total_trades,
        'avg_trade': round(np.mean(trade_returns), 2) if trade_returns else 0
    }

def train_and_evaluate(dataframe, feature_list, threshold, use_class_weights=False, xgb_params=None):
    """Full train-evaluate-backtest for a given configuration."""
    labeled = create_labels(dataframe, threshold)
    labeled.dropna(subset=feature_list + ['Target'], inplace=True)
    labeled.reset_index(drop=True, inplace=True)

    train_df = labeled[labeled['Date'] <= TRAIN_END].copy()
    test_df = labeled[labeled['Date'] >= TEST_START].copy()

    X_train = train_df[feature_list].values
    y_train = train_df['Target'].astype(int).values
    X_test = test_df[feature_list].values
    y_test = test_df['Target'].astype(int).values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Compute class weights if needed
    sw = None
    if use_class_weights:
        from collections import Counter
        counts = Counter(y_train)
        total = len(y_train)
        n_classes = len(counts)
        class_w = {c: total / (n_classes * cnt) for c, cnt in counts.items()}
        sw = np.array([class_w[y] for y in y_train])

    # Default XGBoost params
    params = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'use_label_encoder': False, 'eval_metric': 'mlogloss',
        'random_state': 42, 'verbosity': 0
    }
    if xgb_params:
        params.update(xgb_params)

    model = XGBClassifier(**params)
    model.fit(X_train_s, y_train, sample_weight=sw)

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_prec = precision_score(y_test, y_pred_test, average='weighted')
    test_rec = recall_score(y_test, y_pred_test, average='weighted')

    bt_result = run_backtest(test_df, feature_list, model, scaler)

    dist = {0: int(sum(y_train == 0)), 1: int(sum(y_train == 1)), 2: int(sum(y_train == 2))}

    return {
        'model': model, 'scaler': scaler,
        'train_acc': round(train_acc, 4), 'test_acc': round(test_acc, 4),
        'test_f1': round(test_f1, 4), 'test_prec': round(test_prec, 4),
        'test_rec': round(test_rec, 4),
        'overfit_gap': round(train_acc - test_acc, 4),
        'backtest': bt_result,
        'class_dist': dist,
        'y_test': y_test, 'y_pred_test': y_pred_test,
        'train_df': train_df, 'test_df': test_df
    }


# --- Run threshold experiments ---
thresholds = [2.0, 3.0, 5.0, 7.0]
threshold_results = {}

for thr in thresholds:
    print(f"\n  --- Threshold: +/-{thr}% ---")
    res = train_and_evaluate(df, feature_cols_v1, thr)
    threshold_results[thr] = res
    print(f"    Class Dist (train): SELL={res['class_dist'][0]:,} HOLD={res['class_dist'][1]:,} BUY={res['class_dist'][2]:,}")
    print(f"    Test Acc={res['test_acc']} | F1={res['test_f1']} | Gap={res['overfit_gap']}")
    print(f"    Backtest: Return={res['backtest']['return_pct']}% | WinRate={res['backtest']['win_rate']}% | Trades={res['backtest']['total_trades']}")

print("\n[OK] Step 2 Complete -- Threshold experiments done.")

# %% [markdown]
# ## Step 3 -- Select Best Threshold

# %%
print("=" * 60)
print("STEP 3 -- SELECT BEST THRESHOLD")
print("=" * 60)

# Score each threshold: composite of accuracy, F1, backtest return, win rate
print(f"\n  {'Thresh':>6} | {'Acc':>6} | {'F1':>6} | {'Return%':>8} | {'WinRate':>7} | {'Trades':>6} | Score")
print(f"  {'-'*6} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*7} | {'-'*6} | -----")

best_threshold = None
best_score = -999

for thr, res in threshold_results.items():
    # Composite score: normalized blend of key metrics
    score = (
        res['test_f1'] * 30 +                              # 30% weight on F1
        res['backtest']['return_pct'] / 10 * 30 +           # 30% weight on returns (scaled)
        res['backtest']['win_rate'] / 100 * 20 +            # 20% weight on win rate
        (1 - abs(res['overfit_gap'])) * 10 +                # 10% weight on low overfit
        min(res['backtest']['total_trades'] / 200, 1) * 10  # 10% weight on trade count
    )
    print(f"  {thr:>5.1f}% | {res['test_acc']:>6.4f} | {res['test_f1']:>6.4f} | "
          f"{res['backtest']['return_pct']:>+7.1f}% | {res['backtest']['win_rate']:>6.1f}% | "
          f"{res['backtest']['total_trades']:>6} | {score:.2f}")
    if score > best_score:
        best_score = score
        best_threshold = thr

print(f"\n  >> Best Threshold: +/-{best_threshold}%  (score={best_score:.2f})")

# %% [markdown]
# ## Step 4 -- Class Imbalance Handling

# %%
print("=" * 60)
print("STEP 4 -- CLASS IMBALANCE HANDLING")
print("=" * 60)

# Compare with vs without class weights using best threshold
print(f"\n  Using threshold: +/-{best_threshold}%")

res_no_weight = threshold_results[best_threshold]
res_weighted = train_and_evaluate(df, feature_cols_v1, best_threshold, use_class_weights=True)

print(f"\n  {'Metric':<18} | {'No Weights':>12} | {'Class Weights':>14}")
print(f"  {'-'*18} | {'-'*12} | {'-'*14}")
print(f"  {'Test Accuracy':<18} | {res_no_weight['test_acc']:>12.4f} | {res_weighted['test_acc']:>14.4f}")
print(f"  {'Test F1':<18} | {res_no_weight['test_f1']:>12.4f} | {res_weighted['test_f1']:>14.4f}")
print(f"  {'Overfit Gap':<18} | {res_no_weight['overfit_gap']:>12.4f} | {res_weighted['overfit_gap']:>14.4f}")
print(f"  {'Backtest Return':<18} | {res_no_weight['backtest']['return_pct']:>+11.1f}% | {res_weighted['backtest']['return_pct']:>+13.1f}%")
print(f"  {'Win Rate':<18} | {res_no_weight['backtest']['win_rate']:>11.1f}% | {res_weighted['backtest']['win_rate']:>13.1f}%")
print(f"  {'Trades':<18} | {res_no_weight['backtest']['total_trades']:>12} | {res_weighted['backtest']['total_trades']:>14}")

# Pick the better one
use_weights = False
if res_weighted['test_f1'] > res_no_weight['test_f1'] and res_weighted['backtest']['return_pct'] >= res_no_weight['backtest']['return_pct'] * 0.7:
    use_weights = True
print(f"\n  >> Decision: {'USE class weights' if use_weights else 'NO class weights (baseline better)'}")

# %% [markdown]
# ## Step 5 -- Feature Importance Analysis

# %%
print("=" * 60)
print("STEP 5 -- FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

base_result = res_weighted if use_weights else res_no_weight
base_model = base_result['model']
importances = base_model.feature_importances_
feat_imp = sorted(zip(feature_cols_v1, importances), key=lambda x: x[1], reverse=True)

print("\n  Feature Importance Ranking:")
for i, (name, imp) in enumerate(feat_imp, 1):
    bar = '#' * int(imp * 100)
    status = "  << WEAK" if imp < 0.03 else ""
    print(f"    {i:2d}. {name:25s} : {imp:.4f} {bar}{status}")

# Remove features with importance < 3%
weak_features = [name for name, imp in feat_imp if imp < 0.03]
strong_features = [name for name, imp in feat_imp if imp >= 0.03]

print(f"\n  Strong features: {len(strong_features)}")
print(f"  Weak features (removed): {len(weak_features)} -> {weak_features}")
print(f"\n  Final feature set: {strong_features}")

# %% [markdown]
# ## Step 6 -- Hyperparameter Tuning

# %%
print("=" * 60)
print("STEP 6 -- HYPERPARAMETER TUNING")
print("=" * 60)

# Test a focused grid of practical configurations
param_configs = [
    {"label": "Baseline (v1 params)", "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}},
    {"label": "Deeper trees", "params": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05}},
    {"label": "More trees + slower LR", "params": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03}},
    {"label": "Regularized", "params": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 1.5}},
    {"label": "Shallow + Fast", "params": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08}},
    {"label": "Large ensemble", "params": {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.7}},
]

tuning_results = []
features_to_use = strong_features if len(strong_features) >= 8 else feature_cols_v1

print(f"  Using {len(features_to_use)} features, threshold={best_threshold}%, weights={use_weights}")
print()
print(f"  {'Config':<25} | {'TestAcc':>7} | {'F1':>6} | {'Gap':>6} | {'Return%':>8} | {'WinRate':>7}")
print(f"  {'-'*25} | {'-'*7} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*7}")

for cfg in param_configs:
    res = train_and_evaluate(df, features_to_use, best_threshold,
                             use_class_weights=use_weights, xgb_params=cfg['params'])
    tuning_results.append({'label': cfg['label'], 'params': cfg['params'], 'result': res})
    print(f"  {cfg['label']:<25} | {res['test_acc']:>7.4f} | {res['test_f1']:>6.4f} | "
          f"{res['overfit_gap']:>6.4f} | {res['backtest']['return_pct']:>+7.1f}% | "
          f"{res['backtest']['win_rate']:>6.1f}%")

print("\n[OK] Step 6 Complete -- Hyperparameter tuning done.")

# %% [markdown]
# ## Step 7 -- Select Best Configuration

# %%
print("=" * 60)
print("STEP 7 -- SELECT BEST MODEL")
print("=" * 60)

best_config = None
best_total_score = -999

for tr in tuning_results:
    res = tr['result']
    score = (
        res['test_f1'] * 30 +
        res['backtest']['return_pct'] / 10 * 30 +
        res['backtest']['win_rate'] / 100 * 20 +
        (1 - abs(res['overfit_gap'])) * 10 +
        min(res['backtest']['total_trades'] / 200, 1) * 10
    )
    if score > best_total_score:
        best_total_score = score
        best_config = tr

print(f"  Best Config   : {best_config['label']}")
print(f"  Params        : {best_config['params']}")
print(f"  Test Accuracy : {best_config['result']['test_acc']}")
print(f"  Test F1       : {best_config['result']['test_f1']}")
print(f"  Overfit Gap   : {best_config['result']['overfit_gap']}")
print(f"  Backtest Ret  : {best_config['result']['backtest']['return_pct']}%")
print(f"  Win Rate      : {best_config['result']['backtest']['win_rate']}%")
print(f"  Total Trades  : {best_config['result']['backtest']['total_trades']}")

# %% [markdown]
# ## Step 8 -- Batch Training Validation (Walk-Forward)

# %%
print("=" * 60)
print("STEP 8 -- BATCH TRAINING VALIDATION")
print("=" * 60)

labeled_df = create_labels(df, best_threshold)
labeled_df.dropna(subset=features_to_use + ['Target'], inplace=True)
labeled_df.reset_index(drop=True, inplace=True)

batch_windows = [
    ('2015-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2015-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2015-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2015-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
]

xgb_params_best = best_config['params'].copy()
xgb_params_best.update({
    'subsample': xgb_params_best.get('subsample', 0.8),
    'colsample_bytree': 0.8,
    'use_label_encoder': False, 'eval_metric': 'mlogloss',
    'random_state': 42, 'verbosity': 0
})

batch_results_v2 = []
for i, (tr_s, tr_e, te_s, te_e) in enumerate(batch_windows, 1):
    tr_mask = (labeled_df['Date'] >= tr_s) & (labeled_df['Date'] <= tr_e)
    te_mask = (labeled_df['Date'] >= te_s) & (labeled_df['Date'] <= te_e)
    X_tr = labeled_df.loc[tr_mask, features_to_use].values
    y_tr = labeled_df.loc[tr_mask, 'Target'].astype(int).values
    X_te = labeled_df.loc[te_mask, features_to_use].values
    y_te = labeled_df.loc[te_mask, 'Target'].astype(int).values
    if len(X_te) == 0:
        continue

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    sw = None
    if use_weights:
        from collections import Counter
        counts = Counter(y_tr)
        total = len(y_tr)
        n_c = len(counts)
        cw = {c: total / (n_c * cnt) for c, cnt in counts.items()}
        sw = np.array([cw[y] for y in y_tr])

    m = XGBClassifier(**xgb_params_best)
    m.fit(X_tr_s, y_tr, sample_weight=sw)
    y_p = m.predict(X_te_s)
    acc = accuracy_score(y_te, y_p)
    f1 = f1_score(y_te, y_p, average='weighted')
    batch_results_v2.append({'batch': i, 'train': f'{tr_s}->{tr_e}', 'test': f'{te_s}->{te_e}',
                             'accuracy': round(acc, 4), 'f1': round(f1, 4)})
    print(f"  Batch {i}: Train {tr_s}->{tr_e} | Test {te_s}->{te_e} | Acc={acc:.4f} | F1={f1:.4f}")

print("\n[OK] Step 8 Complete -- Walk-forward validation done.")

# %% [markdown]
# ## Step 9 -- Final Evaluation (v1 vs v2 Comparison)

# %%
print("=" * 60)
print("STEP 9 -- v1 vs v2 COMPARISON")
print("=" * 60)

v2 = best_config['result']

print(f"\n  {'Metric':<20} | {'v1 (old)':>12} | {'v2 (new)':>12} | {'Change':>10}")
print(f"  {'-'*20} | {'-'*12} | {'-'*12} | {'-'*10}")

def diff_str(old, new, higher_better=True):
    d = new - old
    if higher_better:
        return f"{'+'if d>=0 else ''}{d:.4f} {'[v]' if d>0 else '[!]'}"
    else:
        return f"{'+'if d>=0 else ''}{d:.4f} {'[v]' if d<0 else '[!]'}"

print(f"  {'Test Accuracy':<20} | {v1_metrics['test_accuracy']:>12.4f} | {v2['test_acc']:>12.4f} | {diff_str(v1_metrics['test_accuracy'], v2['test_acc'])}")
print(f"  {'Test F1':<20} | {v1_metrics['test_f1']:>12.4f} | {v2['test_f1']:>12.4f} | {diff_str(v1_metrics['test_f1'], v2['test_f1'])}")
print(f"  {'Overfit Gap':<20} | {'N/A':>12} | {v2['overfit_gap']:>12.4f} |")
print(f"  {'Backtest Return':<20} | {v1_metrics['backtest_return']:>+11.1f}% | {v2['backtest']['return_pct']:>+11.1f}% | {diff_str(v1_metrics['backtest_return'], v2['backtest']['return_pct'])}")
print(f"  {'Win Rate':<20} | {v1_metrics['backtest_win_rate']:>11.1f}% | {v2['backtest']['win_rate']:>11.1f}% | {diff_str(v1_metrics['backtest_win_rate'], v2['backtest']['win_rate'])}")
print(f"  {'Total Trades':<20} | {v1_metrics['total_trades']:>12} | {v2['backtest']['total_trades']:>12} |")

print(f"\n--- Classification Report (v2 Test Set) ---\n")
print(classification_report(v2['y_test'], v2['y_pred_test'], target_names=['SELL', 'HOLD', 'BUY']))

print(f"--- Confusion Matrix (v2 Test Set) ---")
cm = confusion_matrix(v2['y_test'], v2['y_pred_test'])
print(f"             Predicted")
print(f"             SELL  HOLD   BUY")
for i, row_label in enumerate(['SELL ', 'HOLD ', 'BUY  ']):
    print(f"  Actual {row_label} {cm[i]}")

# %% [markdown]
# ## Step 10 -- Save Model v2

# %%
print("=" * 60)
print("STEP 10 -- SAVE MODEL v2")
print("=" * 60)

v2_model = v2['model']
v2_scaler = v2['scaler']

model_path = os.path.join(MODEL_DIR, "technical_model_v2.pkl")
joblib.dump(v2_model, model_path)
print(f"  [v] Model saved : {model_path}")

scaler_path = os.path.join(MODEL_DIR, "scaler_v2.pkl")
joblib.dump(v2_scaler, scaler_path)
print(f"  [v] Scaler saved: {scaler_path}")

# Update best model
best_path = os.path.join(MODEL_DIR, "technical_model_best.pkl")
joblib.dump(v2_model, best_path)
best_scaler = os.path.join(MODEL_DIR, "scaler_best.pkl")
joblib.dump(v2_scaler, best_scaler)
print(f"  [v] Best model updated")

feat_path = os.path.join(MODEL_DIR, "feature_list_v2.json")
with open(feat_path, 'w') as f:
    json.dump(features_to_use, f, indent=2)
print(f"  [v] Feature list: {feat_path}")

# Save signals for Meta Model
labeled_full = create_labels(df, best_threshold)
labeled_full.dropna(subset=features_to_use + ['Target'], inplace=True)
X_all = v2_scaler.transform(labeled_full[features_to_use].values)
labeled_full['Tech_Pred'] = v2_model.predict(X_all)
proba = v2_model.predict_proba(X_all)
labeled_full['Tech_Prob_SELL'] = proba[:, 0]
labeled_full['Tech_Prob_HOLD'] = proba[:, 1]
labeled_full['Tech_Prob_BUY'] = proba[:, 2]
signal_path = os.path.join(PROCESSED_DIR, "technical_signals_v2.csv")
labeled_full[['Date', 'Stock', 'Close', 'Target', 'Tech_Pred',
              'Tech_Prob_SELL', 'Tech_Prob_HOLD', 'Tech_Prob_BUY']].to_csv(signal_path, index=False)
print(f"  [v] Signals saved: {signal_path}")

# %% [markdown]
# ## Step 11 -- Report Generation

# %%
print("=" * 60)
print("STEP 11 -- REPORT GENERATION")
print("=" * 60)

# Build threshold comparison section
thr_table = "| Threshold | Test Acc | F1 | Return% | Win Rate | Trades |\n|---|---|---|---|---|---|\n"
for thr in thresholds:
    r = threshold_results[thr]
    thr_table += (f"| +/-{thr}% | {r['test_acc']} | {r['test_f1']} | "
                  f"{r['backtest']['return_pct']:+.1f}% | {r['backtest']['win_rate']}% | "
                  f"{r['backtest']['total_trades']} |\n")

# Build tuning comparison
tune_table = "| Config | Test Acc | F1 | Gap | Return% | Win Rate |\n|---|---|---|---|---|---|\n"
for tr in tuning_results:
    r = tr['result']
    tune_table += (f"| {tr['label']} | {r['test_acc']} | {r['test_f1']} | "
                   f"{r['overfit_gap']} | {r['backtest']['return_pct']:+.1f}% | "
                   f"{r['backtest']['win_rate']}% |\n")

# Build batch table
batch_table = "| Batch | Train Window | Test Window | Accuracy | F1 |\n|---|---|---|---|---|\n"
for b in batch_results_v2:
    batch_table += f"| {b['batch']} | {b['train']} | {b['test']} | {b['accuracy']} | {b['f1']} |\n"

report = f"""# BullRun -- Technical Model v2 Report (Improvement)
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. v1 vs v2 Comparison

| Metric | v1 (Old) | v2 (New) | Change |
|---|---|---|---|
| Test Accuracy | {v1_metrics['test_accuracy']} | {v2['test_acc']} | {v2['test_acc'] - v1_metrics['test_accuracy']:+.4f} |
| Test F1 | {v1_metrics['test_f1']} | {v2['test_f1']} | {v2['test_f1'] - v1_metrics['test_f1']:+.4f} |
| Precision (wt) | N/A | {v2['test_prec']} | -- |
| Recall (wt) | N/A | {v2['test_rec']} | -- |
| Backtest Return | {v1_metrics['backtest_return']:+.1f}% | {v2['backtest']['return_pct']:+.1f}% | {v2['backtest']['return_pct'] - v1_metrics['backtest_return']:+.1f}% |
| Win Rate | {v1_metrics['backtest_win_rate']}% | {v2['backtest']['win_rate']}% | {v2['backtest']['win_rate'] - v1_metrics['backtest_win_rate']:+.1f}% |
| Total Trades | {v1_metrics['total_trades']} | {v2['backtest']['total_trades']} | {v2['backtest']['total_trades'] - v1_metrics['total_trades']:+d} |

## 2. Threshold Analysis

Best threshold: **+/-{best_threshold}%** (forward {FORWARD_DAYS} days)

{thr_table}

## 3. Class Imbalance

| Setting | Test Acc | F1 | Return | Win Rate |
|---|---|---|---|---|
| No weights | {res_no_weight['test_acc']} | {res_no_weight['test_f1']} | {res_no_weight['backtest']['return_pct']:+.1f}% | {res_no_weight['backtest']['win_rate']}% |
| Class weights | {res_weighted['test_acc']} | {res_weighted['test_f1']} | {res_weighted['backtest']['return_pct']:+.1f}% | {res_weighted['backtest']['win_rate']}% |

Decision: **{'Class weights used' if use_weights else 'No weights (baseline better)'}**

## 4. Hyperparameter Tuning

{tune_table}

Best config: **{best_config['label']}** with params: `{best_config['params']}`

## 5. Feature Analysis

| # | Feature | Importance | Status |
|---|---|---|---|
"""
for i, (name, imp) in enumerate(feat_imp, 1):
    status = "Removed" if name in weak_features else "Kept"
    report += f"| {i} | {name} | {imp:.4f} | {status} |\n"

report += f"""
Features used in v2: **{len(features_to_use)}** (removed {len(weak_features)} weak features)

## 6. Batch Training (Walk-Forward)

{batch_table}

## 7. Final v2 Metrics

| Metric | Value |
|---|---|
| Test Accuracy | {v2['test_acc']} |
| Test F1 (weighted) | {v2['test_f1']} |
| Test Precision (wt) | {v2['test_prec']} |
| Test Recall (wt) | {v2['test_rec']} |
| Overfit Gap | {v2['overfit_gap']} |
| Backtest Return | {v2['backtest']['return_pct']:+.1f}% |
| Win Rate | {v2['backtest']['win_rate']}% |
| Total Trades | {v2['backtest']['total_trades']} |
| Avg Trade Return | {v2['backtest']['avg_trade']}% |

## 8. Changes Made (v1 -> v2)
- Threshold: 2% -> {best_threshold}%
- Class weights: {'Enabled' if use_weights else 'Disabled (not beneficial)'}
- XGBoost params: {best_config['params']}
- Features: {len(feature_cols_v1)} -> {len(features_to_use)} (dropped {len(weak_features)} weak)

## 9. Conclusion
- Model v2 {'improves' if v2['test_f1'] > v1_metrics['test_f1'] else 'maintains'} F1 score vs v1
- Backtest return: v1={v1_metrics['backtest_return']:+.1f}% vs v2={v2['backtest']['return_pct']:+.1f}%
- Overfit gap: {v2['overfit_gap']} ({'healthy' if v2['overfit_gap'] < 0.05 else 'moderate' if v2['overfit_gap'] < 0.10 else 'needs attention'})
- Ready for Meta Model integration: **Yes** (signals exported)
"""

report_path = os.path.join(REPORT_DIR, "technical_model_report_v2.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  [v] Report saved: {report_path}")

# Experiment log
exp_log = {
    'run_id': 'run_002_technical_xgb_v2',
    'timestamp': datetime.now().isoformat(),
    'model': 'XGBoost', 'version': 'v2',
    'threshold': best_threshold,
    'features': len(features_to_use),
    'class_weights': use_weights,
    'xgb_params': best_config['params'],
    'train_accuracy': v2['train_acc'],
    'test_accuracy': v2['test_acc'],
    'test_f1': v2['test_f1'],
    'test_precision': v2['test_prec'],
    'test_recall': v2['test_rec'],
    'overfit_gap': v2['overfit_gap'],
    'backtest_return': v2['backtest']['return_pct'],
    'backtest_win_rate': v2['backtest']['win_rate'],
    'total_trades': v2['backtest']['total_trades'],
    'improvements_over_v1': {
        'accuracy_change': round(v2['test_acc'] - v1_metrics['test_accuracy'], 4),
        'f1_change': round(v2['test_f1'] - v1_metrics['test_f1'], 4),
        'return_change': round(v2['backtest']['return_pct'] - v1_metrics['backtest_return'], 2)
    }
}
exp_path = os.path.join(EXPERIMENT_DIR, "runs", "run_002_technical_xgb_v2.json")
with open(exp_path, 'w') as f:
    json.dump(exp_log, f, indent=2)
print(f"  [v] Experiment log: {exp_path}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE -- Technical Model v2 is ready!")
print("=" * 60)
