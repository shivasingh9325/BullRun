from fastapi import APIRouter, HTTPException
from backend.api.schemas import PortfolioStatus, TradeInfo
from backend.db.session import SessionLocal
from backend.db.models import DBPortfolioState, DBTrade
from backend.broker.broker_mock import MockBroker
import pandas as pd
import os
import yaml
from typing import List

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


def _get_config():
    """Helper to load config relative to backend root."""
    config_path = "configs/prod_params.yaml"
    if not os.path.exists(config_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        config_path = os.path.join(base_dir, "configs", "prod_params.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@router.get("", response_model=PortfolioStatus)
async def get_status():
    """Return the latest portfolio state: balances, holdings, P&L."""
    db = SessionLocal()
    try:
        latest_state = (
            db.query(DBPortfolioState)
            .order_by(DBPortfolioState.timestamp.desc())
            .first()
        )
        if not latest_state:
            # Fallback to broker init defaults
            broker = MockBroker()
            return {
                "fiat_balance": broker.fiat_balance,
                "total_value": broker.fiat_balance,
                "unrealized_pnl": 0.0,
                "holdings": {},
                "utilization_pct": 0.0
            }

        return {
            "fiat_balance": latest_state.fiat_balance,
            "total_value": latest_state.total_value,
            "unrealized_pnl": latest_state.unrealized_pnl,
            "holdings": latest_state.holdings or {},
            "utilization_pct": latest_state.utilization_pct
        }
    finally:
        db.close()


@router.get("/trades", response_model=List[TradeInfo])
async def get_trades(limit: int = 50):
    """Return the most recent trade history."""
    db = SessionLocal()
    try:
        trades = (
            db.query(DBTrade)
            .order_by(DBTrade.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "ticker": t.symbol,
                "action": t.action,
                "price": t.price,
                "quantity": t.quantity,
                "timestamp": t.timestamp,
                "confidence": t.confidence,
                "pnl": t.pnl,
                "reason": t.reason,
            }
            for t in trades
        ]
    finally:
        db.close()


@router.get("/equity-curve")
async def get_equity_curve():
    """Return the equity curve data as a time series list."""
    try:
        cfg = _get_config()
        prod = cfg.get("production", {})
        equity_path = prod.get("equity_curve_path", "logs/equity_curve.csv")

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if not os.path.isabs(equity_path):
            equity_path = os.path.join(base_dir, equity_path)

        if not os.path.exists(equity_path):
            return {"data": [], "message": "No equity curve data yet"}

        df = pd.read_csv(equity_path)
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
