# BullRun -- Meta Model Integrator Report
**Generated**: 2026-04-07 03:18:24

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
- **Tech_Prob_SELL**: 0.2953
- **Tech_Prob_HOLD**: 0.2804
- **Tech_Prob_BUY**: 0.2516
- **Volatility_14d**: 0.0774
- **Pred_Return_5d**: 0.0517
- **Max_Drawdown_30d**: 0.0437
- **Sentiment_Score**: 0.0000
- **News_Volume**: 0.0000

## 4. Final Meta Backtest Performance (Test Set: 2024+)
| Metric | Result |
|---|---|
| **Total Trades** | 32 |
| **Win Rate** | 62.5% |
| **System Return %** | +178.50% |
| **Avg Return / Trade** | +6.22% |

**Conclusion:** The Meta Engine is the final fully-integrated output of the BullRun pipeline. All signals are exported to `e:/Bull_Run\data\processed\meta_signals.csv`. 
