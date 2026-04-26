# BullRun -- Price Prediction Model Report
**Generated**: 2026-04-07 03:16:16

---

## 1. Context
The Price Prediction Model uses a `LightGBM Regressor` to forecast the continuous 5-day forward return percentage. This provides the Meta Model with a quantitative target expectation rather than just a categorical Buy/Sell assumption.

## 2. Evaluation Metrics (Test Set: 2024+)
| Metric | Value | Interpretation |
|---|---|---|
| **RMSE** | 3.6648% | Standard deviation of prediction errors. |
| **MAE** | 2.7268% | Average absolute error in percentage points. |
| **R2 Score** | -0.0029 | Explained variance (typically very low in finance). |
| **Directional Accuracy** | 52.07% | How often the model correctly predicts the sign (+/-). |

## 3. Top 5 Drivers (Feature Importance)
- **MACD_Signal**: 13
- **RSI_14**: 11
- **Volume_SMA_10**: 11
- **HL_Range_Pct**: 9
- **Price_vs_SMA20**: 6

## 4. Pipeline Outputs
- Predictions saved to: `e:/Bull_Run\data\processed\price_prediction_features.csv`
- Model serialized to: `e:/Bull_Run\models\price_model\lgb_price_model.pkl`

**Status:** Ready for Meta layer integration.
