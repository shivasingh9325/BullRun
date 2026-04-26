# BullRun AI: Autonomous Trading Ecosystem

<p align="center">
  <img src="https://img.shields.io/badge/Stack-FastAPI%20%2B%20React%20%2B%20XGBoost-purple?style=for-the-badge&logo=python" alt="Tech Stack">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge" alt="Version">
</p>

---

BullRun AI is a hierarchical, multi-agent autonomous trading system designed to navigate modern market volatility with institutional-grade precision. It combines deep technical analysis, social sentiment extraction, and reinforcement learning to execute automated trades on the NIFTY 50 universe.

---

## Key Features

- **Multi-Layer ML Pipeline**: Technical Model → Meta Model → RL Agent → Risk Layer → Execution
- **XGBoost-Powered Signals**: 18 technical indicators with probabilistic outputs (SELL/HOLD/BUY)
- **Reinforcement Learning Allocation**: PPO agent for optimal position sizing
- **Real-Time Sentiment**: Finnhub API integration with keyword-based NLP scoring
- **Risk Management**: Sector exposure caps (30%), drawdown halts (15%), confidence thresholds
- **Persistent State**: Supabase/SQLite integration for portfolio tracking and audit logs
- **Modern Dashboard**: React + Vite with 3D visual effects and real-time updates
- **Paper Trading**: Mock broker with realistic slippage, fees, and order execution

---

## System Architecture

```
Market Data (yfinance)
        │
        ▼
┌───────────────────┐
│ Technical Model   │  XGBoost Classifier (18 indicators)
│ (Base Signals)    │  RSI, MACD, Bollinger Bands, ATR, etc.
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Sentiment Model   │  Finnhub API + Keyword NLP
│ (News Injection)  │  Sentiment Score [-1, 1], News Volume
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Meta Model        │  XGBoost Gate (validates signals)
│ (Signal Filter)   │  Checks volatility, drawdown, sentiment
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ RL Agent (PPO)    │  Stable-Baselines3
│ (Position Sizing) │  Determines allocation % (0-100%)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Risk Layer        │  Sector caps, drawdown scaling,
│ (Safety Guards)   │  max trades/day, confidence floors
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Mock Broker       │  Atomic execution, persistence,
│ (Execution)       │  trade logging to database
└───────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, Python 3.12, Uvicorn |
| **ML/AI** | XGBoost, Scikit-learn, PyTorch, Stable-Baselines3 |
| **Data** | yfinance, Pandas, NumPy |
| **Database** | Supabase (PostgreSQL), SQLite (local fallback) |
| **Frontend** | React 19, Vite, TailwindCSS 4, Lucide React |
| **APIs** | Finnhub (news/sentiment) |

---

## Folder Structure

```
BullRun/
├── backend/
│   ├── api/
│   │   ├── main.py           # FastAPI app entry, health, predict endpoints
│   │   ├── portfolio.py      # Portfolio & trade history endpoints
│   │   └── schemas.py        # Pydantic request/response models
│   ├── broker/
│   │   ├── broker_mock.py    # Mock execution broker with persistence
│   │   └── environment.py    # Gymnasium environment for RL training
│   ├── core/
│   │   ├── fetcher.py        # yfinance data fetcher
│   │   └── technical.py      # Technical indicator calculations
│   ├── db/
│   │   ├── models.py         # SQLAlchemy ORM models
│   │   ├── session.py        # Database session management
│   │   └── init_db.py        # Database initialization
│   ├── models/
│   │   ├── agent.py          # PortfolioManagerAgent (PPO RL)
│   │   ├── base.py           # Base predictor class
│   │   ├── meta.py           # MetaModel (XGBoost gate)
│   │   ├── sentiment_model.py # Sentiment predictor (Finnhub)
│   │   └── technical_model.py # Technical predictor (XGBoost)
│   ├── pipeline/
│   │   └── pipeline.py       # DailyInferencePipeline orchestrator
│   ├── backtesting/
│   │   └── engine.py         # Backtesting engine
│   └── utils/
│       ├── logger.py         # Logging utilities
│       └── logging.py        # Logging configuration
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main dashboard application
│   │   ├── main.jsx          # React entry point
│   │   ├── index.css         # Global styles
│   │   ├── components/
│   │   │   ├── GooeyInput.jsx
│   │   │   ├── MagicBento.jsx
│   │   │   ├── PillNav.jsx
│   │   │   └── Prism.jsx
│   │   └── assets/
│   │       ├── hero.png
│   │       ├── react.svg
│   │       └── vite.svg
│   ├── package.json
│   └── vite.config.js
├── configs/
│   ├── prod_params.yaml      # Production configuration
│   └── sector_map.json       # Stock sector mappings
├── data/
│   ├── processed/            # Processed feature datasets
│   ├── raw/                  # Raw market data
│   └── snapshots/            # Portfolio snapshots
├── logs/
│   ├── bullrun.log           # Main application log
│   └── equity_curve.csv      # Portfolio equity history
├── models/                   # Trained model binaries (not in repo)
├── notebooks/                # Research & model training notebooks
├── reports/
│   ├── bullrun_project_report.md
│   ├── model_reports/        # Individual model evaluations
│   └── weekly_summary.md
├── scripts/
│   ├── trading_daemon.py    # Live trading daemon
│   ├── train_multi_stock.py # Multi-stock RL training
│   └── prepare_rl_dataset.py
├── pipelines/
│   └── daily_inference.py
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── package.json              # Node.js dependencies
└── README.md
```

---

## Prerequisites

- **Python 3.12+**
- **Node.js 18+**
- **TA-Lib** (install manually: https://ta-lib.org/)

---

## Installation

### Backend

```bash
# Navigate to project root
cd BullRun

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install Python dependencies
pip install -r requirements.txt

# Note: TA-Lib must be installed separately
# Windows: Download from https://ta-lib.org/ and place in C:\ta-lib
# Linux: sudo apt-get install ta-lib && pip install ta-lib
```

### Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install npm dependencies
npm install
```

---

## Running the Application

### 1. Start Backend (FastAPI)

```bash
# From project root
# Windows PowerShell
$env:PYTHONPATH="backend"
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python directly
python -c "import uvicorn; uvicorn.run('backend.api.main:app', host='0.0.0.0', port=8000, reload=True)"
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 2. Start Frontend (React + Vite)

```bash
# From frontend directory
cd frontend
npm run dev
```

The dashboard will be available at: **http://localhost:5173**

### 3. Production Build (Frontend)

```bash
cd frontend
npm run build
npx serve dist -l 5173
```

---

## Environment Variables

Create a `.env` file in the project root (already provided):

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key
SUPABASE_DB_URL=sqlite:///bullrun.db
FINNHUB_API_KEY=your-finnhub-api-key
REDIS_URL=redis://localhost:6379/0
```

---

## API Endpoints

### Health Check

**GET** `/api/v1/health`

Returns system status, uptime, and model readiness.

**Response:**
```json
{
  "status": "Operational",
  "timestamp": 1713468000,
  "uptime_seconds": 3600,
  "engine": "BullRun 1.0.0",
  "model_inference": "Ready",
  "database": "Connected"
}
```

### Run Prediction

**POST** `/api/v1/predict`

Triggers the full daily inference pipeline.

**Request Body:**
```json
{
  "symbol": "RELIANCE.NS",
  "capital": 100000.0,
  "risk_preference": "BALANCED"
}
```

**Response:**
```json
{
  "status": "Success",
  "run_id": "20260418_143022",
  "mode": "DRY_RUN",
  "timestamp": 1713468022,
  "trades_executed": 2,
  "decisions": [
    {
      "symbol": "RELIANCE.NS",
      "signal": "BUY",
      "confidence": 0.73,
      "allocation": 0.15,
      "price": 2985.50,
      "timestamp": "2026-04-18T14:30:22",
      "sentiment_score": 0.25,
      "rl_weight": 0.75,
      "risk_flags": [],
      "reason": "All filters passed"
    }
  ],
  "errors": []
}
```

### Get Portfolio Status

**GET** `/api/v1/portfolio`

Returns current portfolio balances, holdings, and P&L.

**Response:**
```json
{
  "fiat_balance": 85000.00,
  "total_value": 98750.00,
  "unrealized_pnl": 3250.00,
  "holdings": {
    "RELIANCE.NS": 15.0,
    "TCS.NS": 8.0
  },
  "utilization_pct": 0.15
}
```

### Get Trade History

**GET** `/api/v1/portfolio/trades?limit=50`

Returns recent trade history.

**Response:**
```json
[
  {
    "ticker": "RELIANCE.NS",
    "action": "BUY",
    "price": 2985.50,
    "quantity": 15.0,
    "timestamp": "2026-04-18T14:30:22",
    "confidence": 0.73,
    "pnl": null,
    "reason": "All filters passed"
  }
]
```

### Run Backtest

**POST** `/api/v1/backtest`

Runs backtesting on historical data.

**Query Parameters:**
- `tickers` (optional): List of symbols, defaults to universe
- `period` (optional): Data period (e.g., "6mo", "1y")

**Response:**
```json
{
  "results": {
    "RELIANCE.NS": {
      "total_return": 0.12,
      "max_drawdown": -0.08,
      "sharpe_ratio": 1.45,
      "win_rate": 0.58,
      "trades": 24
    }
  }
}
```

### Evaluate Model

**GET** `/api/v1/model/evaluate`

Evaluates the MetaModel on synthetic test data.

**Response:**
```json
{
  "evaluation": {
    "accuracy": 0.8667,
    "precision_weighted": 0.88,
    "recall_weighted": 0.8667,
    "confusion_matrix": [[4, 1, 0], [0, 5, 0], [0, 1, 4]],
    "labels": ["SELL", "HOLD", "BUY"],
    "n_samples": 15
  },
  "note": "Synthetic test set — for diagnostic purposes only"
}
```

---

## Configuration

Key settings in `configs/prod_params.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dry_run` | true | Paper trading mode |
| `min_trade_confidence` | 0.55 | Minimum signal confidence |
| `max_trades_per_day` | 7 | Maximum trades per day |
| `sector_exposure_cap` | 0.30 | Max 30% in single sector |
| `portfolio_drawdown_limit` | 0.15 | Halt at 15% drawdown |
| `market_close_hour` | 16 | Only trade after 4 PM |

---

## Pipeline Flow

1. **Data Fetching**: Pull 60 days of OHLCV data via yfinance
2. **Feature Engineering**: Calculate 18 technical indicators
3. **Sentiment Injection**: Fetch news from Finnhub, compute sentiment score
4. **Technical Prediction**: XGBoost outputs SELL/HOLD/BUY probabilities
5. **Meta Validation**: Second XGBoost validates against volatility/drawdown
6. **RL Allocation**: PPO agent determines position size
7. **Risk Check**: Apply sector caps, drawdown scaling, trade limits
8. **Execution**: Mock broker submits orders, persists to database

---

## Current System Status

### Working
- ✅ Full ML pipeline (Technical → Meta → RL → Risk → Execution)
- ✅ FastAPI backend with all endpoints
- ✅ React dashboard with real-time updates
- ✅ Database persistence (SQLite/Supabase)
- ✅ Paper trading with realistic execution
- ✅ Backtesting engine

### Partially Implemented
- 🔶 Sentiment model uses Finnhub API with keyword fallback
- 🔶 RL model includes rule-based fallback when PPO unavailable

### Missing
- ⚠️ Trained model binaries (not included in repository)
- ⚠️ Live broker integration (using mock broker)

---

## Performance Metrics

Based on recent evaluations:

| Strategy | Return | Max Drawdown | Sharpe |
|----------|--------|--------------|--------|
| Buy & Hold (NIFTY 50) | +9.82% | -27.18% | 0.42 |
| Meta Model Strategy | -1.32% | -2.66% | -0.31 |
| RL Agent Strategy | +1.19% | -3.03% | 0.28 |

---

## Model Handling Strategy

To ensure a smooth and optimized GitHub repository experience, we implement the following handling strategy for large files:
- **Mock Models**: By default, the `models/` directory contains small `.pkl` mock models used for development and CI testing. These are perfectly safe to upload.
- **Production Models**: Deep learning models (`.pt`, `.h5`, `.onnx`) can often exceed 100MB. The `.gitignore` prevents these from being uploaded to standard Git history.
- **Data Sets**: All processed and raw data (`data/**/*.csv`) are ignored to comply with size constraints.
- **Cloud Fetching (Future)**: We plan to implement external object storage (e.g., AWS S3, Google Drive) using a `download_models.sh` script to fetch real weights during deployment. If you wish to version models directly, please set up **Git LFS**.

---

## Future Improvements

1. **Live Broker Integration**: Connect to Zerodha/Interactive Brokers API
2. **Model Retraining Pipeline**: Automated weekly model retraining
3. **Sentiment Enhancement**: Integrate Twitter/X social sentiment
4. **Multi-Asset Support**: Expand beyond NIFTY 50 to F&O segment
5. **Options Trading**: Add options strategy module
6. **Portfolio Rebalancing**: Automated weekly portfolio optimization
7. **Cloud Data Fetcher**: Auto-download large models from S3 buckets.

---

## License

MIT License - See LICENSE file for details.

---

**BullRun AI: Trade with Mathematical Certainty. 🚀**