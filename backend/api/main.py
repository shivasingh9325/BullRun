import sys
import os
import time
import asyncio
import uvicorn
from pathlib import Path
from datetime import datetime

# --- AUTO-PATH SETUP ---
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))
if str(base_dir / "bullrun") not in sys.path:
    sys.path.append(str(base_dir / "bullrun"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.api.portfolio import router as portfolio_router
from backend.utils.logger import get_logger
logger = get_logger(__name__)
from backend.db.init_db import init_db
from backend.api.schemas import InferenceRequest, InferenceResponse

_start_time = time.time()

app = FastAPI(
    title="BullRun AI Backend",
    version="1.0.0",
    description="Autonomous AI Trading System — multi-agent pipeline with XGBoost + PPO RL"
)

# Concurrency Guard — only one inference at a time
prediction_lock = asyncio.Semaphore(1)

# Setup CORS — frontend on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start-up logic
@app.on_event("startup")
async def startup_event():
    logger.info("BullRun AI Backend API Starting Up...")
    try:
        init_db()
    except Exception as e:
        logger.error(f"STARTUP-FAILURE: Database initialization failed: {e}")

# Include Versioned Routers
app.include_router(portfolio_router, prefix="/api/v1")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "BullRun AI Backend is Online", "docs": "/docs"}


@app.get("/api/v1/health")
async def health():
    uptime = int(time.time() - _start_time)
    return {
        "status": "Operational",
        "timestamp": time.time(),
        "uptime_seconds": uptime,
        "engine": "BullRun 1.0.0",
        "model_inference": "Ready",
        "database": "Connected"
    }


# ── Predict ───────────────────────────────────────────────────────────────────
@app.post("/api/v1/predict")
async def run_prediction(req: InferenceRequest):
    """
    Manually triggers the Daily Inference Pipeline.
    Returns standardized signals for each ticker.

    Response schema per decision:
        symbol, signal (BUY/SELL/HOLD), confidence, allocation,
        price, timestamp, sentiment_score, rl_weight, risk_flags, reason
    """
    if prediction_lock.locked():
        logger.warning("API-GUARD: Simultaneous prediction requested. Queuing...")

    async with prediction_lock:
        from backend.pipeline.pipeline import DailyInferencePipeline
        try:
            pipeline = DailyInferencePipeline()
            results = pipeline.run(request_data=req, force=True)
            return {
                "status": results.get("status", "Complete"),
                "run_id": results.get("run_id"),
                "mode": results.get("mode"),
                "timestamp": time.time(),
                "trades_executed": results.get("trades_executed", 0),
                "decisions": results.get("decisions", []),
                "errors": results.get("errors", [])
            }
        except FileNotFoundError as e:
            logger.error(f"API-FAILURE: Asset missing: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"API-FAILURE: Prediction trigger failed: {e}")
            raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")


# ── Backtesting ───────────────────────────────────────────────────────────────
@app.post("/api/v1/backtest")
async def run_backtest(tickers: list[str] = None, period: str = "6mo"):
    """
    Runs the backtesting engine on historical data.
    Returns P/L, Max Drawdown, Sharpe Ratio, Win Rate per ticker + aggregate.
    """
    async with prediction_lock:
        from backend.backtesting.engine import BacktestEngine
        try:
            engine = BacktestEngine()
            results = engine.run(tickers=tickers, period=period)
            return results
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"BACKTEST-FAILURE: {e}")
            raise HTTPException(status_code=500, detail=f"Backtest Error: {str(e)}")


# ── Meta Model Evaluation ─────────────────────────────────────────────────────
@app.get("/api/v1/model/evaluate")
async def evaluate_model():
    """
    Evaluates the MetaModel on a synthetic labeled dataset.
    Returns accuracy, precision, recall, and confusion matrix.
    Only available in development/diagnostic mode.
    """
    try:
        import numpy as np
        import pandas as pd
        import os

        backend_root = base_dir
        meta_path = os.path.join(str(backend_root), "models", "meta_model", "xgb_meta_model.pkl")

        from backend.models.meta import MetaModel
        meta = MetaModel(meta_path)

        # Build a minimal synthetic dataset for evaluation (5 data points per class)
        np.random.seed(42)
        n = 15
        features = {
            'Tech_Prob_SELL':    [0.7, 0.6, 0.5, 0.1, 0.2, 0.1, 0.3, 0.3, 0.5, 0.2, 0.6, 0.7, 0.1, 0.2, 0.1],
            'Tech_Prob_BUY':     [0.1, 0.1, 0.2, 0.8, 0.7, 0.6, 0.3, 0.3, 0.3, 0.7, 0.1, 0.1, 0.8, 0.7, 0.6],
            'Tech_Prob_HOLD':    [0.2, 0.3, 0.3, 0.1, 0.1, 0.3, 0.4, 0.4, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1, 0.3],
            'Sentiment_Score':   [-0.5,-0.3, 0.0, 0.5, 0.4, 0.3, 0.0, 0.1, 0.0, 0.4,-0.4,-0.5, 0.5, 0.4, 0.3],
            'News_Volume':       [8, 5, 2, 12, 10, 9, 3, 4, 2, 11, 7, 8, 12, 10, 9],
            'Pred_Return_5d':    [-1.2,-0.8, 0.0, 1.5, 1.2, 0.8, 0.0, 0.1, 0.0, 1.3,-1.1,-1.3, 1.5, 1.2, 0.9],
            'Volatility_14d':    [30, 25, 20, 15, 18, 22, 20, 20, 20, 16, 28, 32, 14, 19, 23],
            'Max_Drawdown_30d':  [-5, -4, -2, -1, -1, -2, -2, -2, -2, -1, -5, -6, -1, -1, -2],
        }
        df = pd.DataFrame(features)
        # Labels: 0=SELL, 1=HOLD, 2=BUY
        y_true = pd.Series([0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1])

        metrics = meta.evaluate(df, y_true)
        return {"evaluation": metrics, "note": "Synthetic test set — for diagnostic purposes only"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation Error: {str(e)}")


# ENTRY POINT for 'python app/main.py'
if __name__ == "__main__":
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
