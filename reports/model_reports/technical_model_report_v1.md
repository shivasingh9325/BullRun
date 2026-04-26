# BullRun -- Technical Model Report (v1)
**Generated**: 2026-04-07 02:25:13

---

## 1. Data Summary
| Item | Value |
|---|---|
| Total Rows (raw) | 129,472 |
| Total Rows (cleaned) | 128,320 |
| Stocks | 49 |
| Date Range | 2015-01-29 to 2026-04-02 |
| Columns Dropped | Adj Close (all NaN), Source |

## 2. Features Used (18 total)
| # | Feature |
|---|---|
| 1 | SMA_10 |
| 2 | SMA_20 |
| 3 | EMA_12 |
| 4 | EMA_26 |
| 5 | MACD |
| 6 | MACD_Signal |
| 7 | MACD_Hist |
| 8 | RSI_14 |
| 9 | BB_Upper |
| 10 | BB_Lower |
| 11 | BB_Width |
| 12 | ATR_14 |
| 13 | Volume_SMA_10 |
| 14 | Volume_Change_Pct |
| 15 | Daily_Return_Pct |
| 16 | HL_Range_Pct |
| 17 | CO_Change_Pct |
| 18 | Price_vs_SMA20 |

## 3. Target Configuration
| Parameter | Value |
|---|---|
| Forward Window | 5 days |
| Threshold | +/-2.0% |
| BUY count | 38,882 |
| HOLD count | 58,274 |
| SELL count | 31,164 |

## 4. Model Configuration
| Parameter | Value |
|---|---|
| Algorithm | XGBoost (XGBClassifier) |
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| Train Period | up to 2023-12-31 |
| Test Period | 2024-01-01 onwards |

## 5. Performance
| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.5391 | 0.4862 |
| F1 (weighted) | -- | 0.3782 |
| Precision (wt) | -- | 0.4107 |
| Recall (wt) | -- | 0.4862 |
| Overfit Gap | 0.0529 | -- |

## 6. Batch Training Results
| Batch | Train Window | Test Window | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 2015-01-01->2019-12-31 | 2020-01-01->2020-12-31 | 0.3829 | 0.3225 |
| 2 | 2015-01-01->2020-12-31 | 2021-01-01->2021-12-31 | 0.4309 | 0.3431 |
| 3 | 2015-01-01->2021-12-31 | 2022-01-01->2022-12-31 | 0.4234 | 0.3243 |
| 4 | 2015-01-01->2022-12-31 | 2023-01-01->2023-12-31 | 0.53 | 0.4033 |

## 7. Backtesting Results
| Metric | Value |
|---|---|
| Initial Capital | Rs.100,000 |
| Final Capital | Rs.196,407 |
| Total Return | +96.41% |
| Total Trades | 214 |
| Win Rate | 64.0% |

## 8. Top Features
| Feature | Importance |
|---|---|
| HL_Range_Pct | 0.0978 |
| EMA_26 | 0.0639 |
| BB_Upper | 0.0637 |
| BB_Width | 0.0613 |
| BB_Lower | 0.0608 |
| EMA_12 | 0.0602 |
| SMA_20 | 0.0574 |
| ATR_14 | 0.0573 |
| Price_vs_SMA20 | 0.0546 |
| Volume_SMA_10 | 0.0536 |

## 9. Insights
- Model uses 18 technical indicators as input features
- Walk-forward batch training validates consistency across multiple time periods
- Chronological train/test split ensures no data leakage
- Backtest simulates realistic trading with 10% capital allocation per trade

## 10. Recommended Next Steps
- Tune hyperparameters with Optuna
- Add class weighting to handle HOLD imbalance
- Integrate sentiment signals from Notebook 02
- Consider ensemble with LSTM for sequence memory
