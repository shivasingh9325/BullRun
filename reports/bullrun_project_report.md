# BullRun Project Report
Generated: 2026-04-15

## 1) Project Overview
BullRun is an AI-assisted trading stack focused on the NIFTY 50 universe, with a modular architecture spanning:
- Backend inference and execution engine (`backend/bullrun/`)
- FastAPI service layer (`backend/app/`)
- React dashboard frontend (`frontend/`)
- Reporting/evaluation artifacts (`reports/`, `backend/logs/`)

Core strategy flow follows a gated chain:
1. Technical signal generation
2. Sentiment augmentation (currently baseline/stub fallback in live path)
3. Meta-model validation
4. RL-driven allocation sizing
5. Risk and exposure controls
6. Broker execution with persistent portfolio state

## 2) Architecture & Operational Flow
Based on the daily pipeline implementation (`backend/bullrun/core/pipeline.py`):
- Loads production config from `backend/configs/prod_params.yaml`
- Runs timed/idempotent daily process (unless forced)
- Fetches ~60d market data
- Computes technical features
- Runs Technical model then Meta model
- Applies RL action for position weight proposal
- Enforces trade gates: confidence floor, sector cap, trade/day cap, drawdown scaling
- Executes BUY orders through mock broker and persists run state
- Writes audit events to DB (`INFERENCE_START` / `INFERENCE_COMPLETE`)

## 3) Configuration Snapshot (Current)
From `backend/configs/prod_params.yaml`:
- **Mode:** `dry_run: true`
- **Execution profile:** `CONSERVATIVE`
- **Risk controls:**
  - Max trades/day: `7`
  - Min trade confidence: `0.55`
  - Sector exposure cap: `30%`
  - Portfolio drawdown limit: `15%`
  - Soft drawdown scaling tiers configured
- **Universe in production config:** 5 symbols (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`)
- **Idempotency marker:** last run date set to `2026-04-08`

## 4) Health of Key Assets
Project contains required production artifacts:
- Trained technical, meta, and RL model binaries under `backend/models/`
- Processed feature/signal datasets under `backend/data/processed/`
- Runtime and evaluation logs under `backend/logs/`
- Historical research notebooks under `backend/archive/` and converted scripts under `notebooks/`

This indicates an end-to-end train → evaluate → infer pipeline exists and has been executed previously.

## 5) Performance & Evaluation Summary
### A) Weekly summary (`reports/weekly_summary.md`)
- Weekly ROI: **-15.66%**
- Current portfolio value: **INR 84,674.36**
- Sharpe ratio (annualized): **-6.83**
- Max drawdown (lifetime): **16.00%**
- Trades executed: **6**
- Avg return/trade: **-2.61%**

Interpretation: recent period shows material drawdown and weak risk-adjusted return.

### B) Strategy benchmark summary (`backend/logs/evaluation_summary.json`)
- **Buy & Hold:** +9.82% return, max DD -27.18%
- **Meta model strategy:** -1.32% return, max DD -2.66%
- **RL agent strategy:** +1.19% return, max DD -3.03%

Interpretation: model strategies reduce drawdown significantly versus buy/hold, but currently underperform buy/hold on raw return.

### C) Multi-asset evaluation (`backend/logs/multi_asset_evaluation.json`)
- Stocks evaluated: **50**
- Average return across universe: **+0.51%**
- Median return: **+0.77%**
- Worst stock: `ADANIENT.NS` at **-6.94%**
- Best stock: `BHARTIARTL.NS` at **+4.51%**
- Average win rate: **~38.74%**
- Average trade count: **~462**

Interpretation: broad performance is mildly positive but inconsistent, with low average hit rate and notable tail-risk on weak tickers.

## 6) Strengths Identified
- Strong modular code separation (core, logic, models, infra, API, frontend)
- Safety-focused execution gates (confidence checks, exposure caps, drawdown scaling)
- Reproducible CLI manager (`backend/manage.py`) for daily run, evaluation, health, reporting, training
- Good reporting footprint (weekly summaries, model reports, evaluation JSON)
- Mock broker persistence and audit trail integration

## 7) Risks / Gaps Identified
- Recent live-like weekly performance is negative and near drawdown threshold.
- Production inference path currently injects random sentiment/news fallback values when missing, which may add noise in gating decisions.
- Documentation references are slightly inconsistent (`backend/README.md` references `src/bullrun/` structure while active tree is `backend/bullrun/`).
- Conservative profile plus strict filtering can suppress trade opportunities (seen in prior final execution report with low/zero throughput under strict constraints).

## 8) Actionable Recommendations
1. Replace sentiment/news stubs with deterministic feature sources in daily inference.
2. Add rolling validation report combining return, drawdown, turnover, and hit-rate drift.
3. Tune confidence thresholds and tiered allocation caps using walk-forward optimization.
4. Expand production universe gradually from 5 symbols to broader basket with per-sector throttles.
5. Align documentation paths and operational runbooks with the current repository structure.
6. Add CI checks for model artifact presence/version compatibility before deployment.

## 9) Overall Status
BullRun is a **functionally complete, production-structured paper-trading platform** with a robust engineering foundation and strong risk controls. Current evidence suggests **risk containment is working better than return generation** in recent periods. The immediate priority is **signal-quality hardening and threshold calibration** to improve alpha capture without materially increasing drawdown.
