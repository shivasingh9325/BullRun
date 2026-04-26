from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from .session import Base

class DBPortfolioState(Base):
    __tablename__ = "portfolio_state"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    fiat_balance = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    holdings = Column(JSON, nullable=False) # Store symbol:qty map
    utilization_pct = Column(Float, default=0.0)

class DBTrade(Base):
    __tablename__ = "trade_history"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    run_id = Column(String, index=True)
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False) # BUY, SELL
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    confidence = Column(Float)
    pnl = Column(Float, nullable=True)
    reason = Column(String)

class DBSystemAudit(Base):
    __tablename__ = "system_audit"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    event_type = Column(String) # INFERENCE, HEALTH_CHECK, ERROR
    status = Column(String)
    message = Column(String)
    details = Column(JSON, nullable=True) # Full inference result or error trace
