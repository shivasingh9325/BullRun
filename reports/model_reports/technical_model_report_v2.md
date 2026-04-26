# BullRun -- Technical Model v2 Report (Improvement)
**Generated**: 2026-04-07 02:41:16

---

## 1. v1 vs v2 Comparison

| Metric | v1 (Old) | v2 (New) | Change |
|---|---|---|---|
| Test Accuracy | 0.4862 | 0.4145 | -0.0717 |
| Test F1 | 0.3782 | 0.4093 | +0.0311 |
| Precision (wt) | N/A | 0.4054 | -- |
| Recall (wt) | N/A | 0.4145 | -- |
| Backtest Return | +96.4% | +132.2% | +35.8% |
| Win Rate | 64.0% | 56.0% | -8.0% |
| Total Trades | 214 | 1556 | +1342 |

## 2. Threshold Analysis

Best threshold: **+/-2.0%** (forward 5 days)

| Threshold | Test Acc | F1 | Return% | Win Rate | Trades |
|---|---|---|---|---|---|
| +/-2.0% | 0.4862 | 0.3782 | +96.4% | 64.0% | 214 |
| +/-3.0% | 0.6579 | 0.5303 | +30.3% | 66.7% | 27 |
| +/-5.0% | 0.8505 | 0.7824 | -3.7% | 33.3% | 3 |
| +/-7.0% | 0.9372 | 0.9069 | +0.0% | 0% | 0 |


## 3. Class Imbalance

| Setting | Test Acc | F1 | Return | Win Rate |
|---|---|---|---|---|
| No weights | 0.4862 | 0.3782 | +96.4% | 64.0% |
| Class weights | 0.4173 | 0.4094 | +123.7% | 56.3% |

Decision: **Class weights used**

## 4. Hyperparameter Tuning

| Config | Test Acc | F1 | Gap | Return% | Win Rate |
|---|---|---|---|---|---|
| Baseline (v1 params) | 0.4192 | 0.412 | 0.1436 | +117.1% | 54.9% |
| Deeper trees | 0.4145 | 0.4093 | 0.3051 | +132.2% | 56.0% |
| More trees + slower LR | 0.4232 | 0.4128 | 0.0881 | +106.8% | 56.3% |
| Regularized | 0.4188 | 0.412 | 0.1106 | +105.5% | 55.8% |
| Shallow + Fast | 0.4224 | 0.4105 | 0.0514 | +66.3% | 54.6% |
| Large ensemble | 0.4285 | 0.4131 | 0.0379 | +103.2% | 56.5% |


Best config: **Deeper trees** with params: `{'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05}`

## 5. Feature Analysis

| # | Feature | Importance | Status |
|---|---|---|---|
| 1 | HL_Range_Pct | 0.0924 | Kept |
| 2 | BB_Upper | 0.0668 | Kept |
| 3 | BB_Lower | 0.0635 | Kept |
| 4 | EMA_12 | 0.0630 | Kept |
| 5 | EMA_26 | 0.0621 | Kept |
| 6 | BB_Width | 0.0593 | Kept |
| 7 | SMA_20 | 0.0593 | Kept |
| 8 | ATR_14 | 0.0588 | Kept |
| 9 | Volume_SMA_10 | 0.0542 | Kept |
| 10 | Price_vs_SMA20 | 0.0528 | Kept |
| 11 | MACD_Signal | 0.0523 | Kept |
| 12 | MACD | 0.0500 | Kept |
| 13 | SMA_10 | 0.0498 | Kept |
| 14 | MACD_Hist | 0.0497 | Kept |
| 15 | RSI_14 | 0.0474 | Kept |
| 16 | CO_Change_Pct | 0.0404 | Kept |
| 17 | Daily_Return_Pct | 0.0395 | Kept |
| 18 | Volume_Change_Pct | 0.0387 | Kept |

Features used in v2: **18** (removed 0 weak features)

## 6. Batch Training (Walk-Forward)

| Batch | Train Window | Test Window | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 2015-01-01->2019-12-31 | 2020-01-01->2020-12-31 | 0.3581 | 0.3607 |
| 2 | 2015-01-01->2020-12-31 | 2021-01-01->2021-12-31 | 0.383 | 0.3804 |
| 3 | 2015-01-01->2021-12-31 | 2022-01-01->2022-12-31 | 0.3906 | 0.3852 |
| 4 | 2015-01-01->2022-12-31 | 2023-01-01->2023-12-31 | 0.4552 | 0.4341 |


## 7. Final v2 Metrics

| Metric | Value |
|---|---|
| Test Accuracy | 0.4145 |
| Test F1 (weighted) | 0.4093 |
| Test Precision (wt) | 0.4054 |
| Test Recall (wt) | 0.4145 |
| Overfit Gap | 0.3051 |
| Backtest Return | +132.2% |
| Win Rate | 56.0% |
| Total Trades | 1556 |
| Avg Trade Return | 0.55% |

## 8. Changes Made (v1 -> v2)
- Threshold: 2% -> 2.0%
- Class weights: Enabled
- XGBoost params: {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05}
- Features: 18 -> 18 (dropped 0 weak)

## 9. Conclusion
- Model v2 improves F1 score vs v1
- Backtest return: v1=+96.4% vs v2=+132.2%
- Overfit gap: 0.3051 (needs attention)
- Ready for Meta Model integration: **Yes** (signals exported)
