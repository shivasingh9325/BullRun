# BullRun -- Risk Model Report
**Generated**: 2026-04-07 03:17:00

---

## 1. Context
The Risk Model generates two critical continuous features that capture local market conditions for each stock. This acts as a dampening shield for the Meta Model.

## 2. Risk Metrics
- **Volatility_14d:** Annualized trailing 14-day volatility (%). If volatility is exploding, the Meta Model can either scale down position size or veto trades to avoid chop.
  - *Average Volatility in dataset:* 27.32%
- **Max_Drawdown_30d:** Measures the largest peak-to-trough drop over the last 30 trading days. Helps the system identify if it's trading into a falling knife or a gentle retracement.
  - *Average Drawdown in dataset:* -5.25%

## 3. Pipeline Output
- Final dataset serialized to: `e:/Bull_Run\data\processed\risk_features.csv`
- Shape: (128320, 4)

**Status:** Ready for Meta layer integration.
