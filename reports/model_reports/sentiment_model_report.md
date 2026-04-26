# BullRun -- Sentiment Model Report (v1)
**Generated**: 2026-04-07 03:03:33

---

## 1. Data Summary
| Metric | Value |
|---|---|
| Total Headlines Processed | 2,500 |
| Time Alignment Shift | +1 Trading Day (Zero Look-Ahead Bias) |
| Stocks Covered | 49 |
| Target Output | e:/Bull_Run\data\processed\sentiment_features.csv |

## 2. Sentiment Distribution (Raw News)
| Tone | Count | Percentage |
|---|---|---|
| **Positive (+)** | 848 | 33.9% |
| **Negative (-)** | 983 | 39.3% |
| **Neutral (0)** | 669 | 26.8% |

## 3. Aggregation & Coverage
- Average sentiment calculation: `Mean of FinBERT scores per Day/Stock`
- Alignment logic: News on Day `T` is mapped to technical signals on Day `T+1`.
- Valid Market Periods (from Tech Model): 128,320
- Periods with News Volume > 0: **1,788 (1.4%)**
- Periods with no news (Filled with 0.0 Neutral): 126,532

## 4. Sample Outputs (Aggregated & Shifted)
```text
      Date       Stock  Sentiment_Score  News_Volume
2016-02-25 ADANIENT.NS        -0.517623            1
2016-04-13 ADANIENT.NS         0.000000            1
2016-12-07 ADANIENT.NS         0.000000            1
2017-04-13 ADANIENT.NS         0.957495            1
2017-08-03 ADANIENT.NS         0.927758            1
```

## 5. Key Observations
1. **MVP Simplicity**: We utilized the pre-trained `ProsusAI/finbert` model, avoiding complex proprietary NLP training while yielding highly robust financial sentiment identification.
2. **Neutrality Assumption**: Days without news are explicitly marked as 0.0. The downstream Meta-Model must be trained on this sparse feature space (i.e., treating sentiment as highly valuable when rare, but ignoring it when neutral).
3. **Integration Readiness**: The final dataset outputs EXACTLY on the `[Date, Stock]` primary keys matching the technical features.

**Status:** Sentiment Model NLP pipeline is fully complete and ready to be merged directly with the Technical Model signals into the **Meta Model**.
