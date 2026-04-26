from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional
from datetime import datetime


class PortfolioStatus(BaseModel):
    fiat_balance: float
    total_value: float
    unrealized_pnl: float
    holdings: Dict[str, float]
    utilization_pct: float

    model_config = ConfigDict(from_attributes=True)


class TradeInfo(BaseModel):
    ticker: str
    action: str
    price: float
    quantity: float
    timestamp: datetime
    confidence: Optional[float] = None
    pnl: Optional[float] = None
    reason: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TradeSignal(BaseModel):
    """Standardized signal response — returned by /predict for each ticker."""
    symbol: str
    signal: str                  # "BUY" | "SELL" | "HOLD"
    confidence: float            # Meta_Prob_BUY [0.0, 1.0]
    allocation: float            # Fraction of portfolio to allocate [0.0, 1.0]
    price: float                 # Latest close price
    timestamp: str               # ISO 8601
    sentiment_score: float       # [-1.0, 1.0]
    rl_weight: float             # Raw RL output [0.0, 1.0]
    risk_flags: List[str]        # e.g. ["SECTOR_CAP", "NEAR_NEUTRAL_SIGNAL"]
    reason: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class InferenceResponse(BaseModel):
    status: str
    run_id: str
    mode: str
    timestamp: float
    trades_executed: int
    decisions: List[TradeSignal]
    errors: List[str]


class SystemHealth(BaseModel):
    status: str
    uptime_seconds: int
    version: str
    last_inference: Optional[datetime] = None
    errors_count: int


class StressTestRequest(BaseModel):
    scenario: str  # "CRASH", "BLEED", "FLASH"


class InferenceRequest(BaseModel):
    symbol: str = "RELIANCE.NS"
    capital: float = 100000.0
    risk_preference: str = "BALANCED"  # CONSERVATIVE, BALANCED, AGGRESSIVE
