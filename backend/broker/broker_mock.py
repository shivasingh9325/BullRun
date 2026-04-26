import os
import yaml
from datetime import datetime
from backend.utils.logger import get_logger
from backend.db.session import SessionLocal
from backend.db.models import DBPortfolioState, DBTrade

logger = get_logger(__name__)

class MockBroker:
    def __init__(self, initial_balance=100000.0, slippage_pct=0.0005, fee_pct=0.00025, config_path="configs/prod_params.yaml"):
        self.fiat_balance = initial_balance
        self.holdings = {}
        self.slippage = slippage_pct
        self.fee = fee_pct
        
        # We now synchronize with the database on init
        self.load_state()

    def submit_order(self, symbol: str, quantity: float, price: float, order_type: str, run_id: str = "MANUAL", confidence: float = 0.0, reason: str = ""):
        db = SessionLocal()
        try:
            cost_per_leg = self.slippage + self.fee
            
            if order_type == 'BUY':
                total_cost = (quantity * price) * (1 + cost_per_leg)
                if self.fiat_balance >= total_cost:
                    self.fiat_balance -= total_cost
                    self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                    
                    # Record Trade
                    trade = DBTrade(
                        run_id=run_id,
                        symbol=symbol,
                        action='BUY',
                        price=price,
                        quantity=quantity,
                        confidence=confidence,
                        reason=reason
                    )
                    db.add(trade)
                    self.save_state_to_db(db)
                    db.commit()
                    logger.info(f"SUPABASE-BROKER: BUY {quantity:.2f} of {symbol} @ {price:.2f}")
                    return True
                else:
                    logger.warning(f"SUPABASE-BROKER: REJECTED BUY {symbol} - Insufficient Fiat.")
                    return False
                    
            elif order_type == 'SELL':
                if symbol in self.holdings and self.holdings[symbol] >= quantity:
                    total_return = (quantity * price) * (1 - cost_per_leg)
                    self.fiat_balance += total_return
                    self.holdings[symbol] -= quantity
                    if self.holdings[symbol] <= 1e-8:
                        del self.holdings[symbol]
                    
                    # Record Trade
                    trade = DBTrade(
                        run_id=run_id,
                        symbol=symbol,
                        action='SELL',
                        price=price,
                        quantity=quantity,
                        confidence=confidence,
                        reason=reason
                    )
                    db.add(trade)
                    self.save_state_to_db(db)
                    db.commit()
                    logger.info(f"SUPABASE-BROKER: SELL {quantity:.2f} of {symbol} @ {price:.2f}")
                    return True
                else:
                    logger.warning(f"SUPABASE-BROKER: REJECTED SELL {symbol} - Insufficient Shares.")
                    return False
        except Exception as e:
            logger.error(f"SUPABASE-BROKER: Error during order submission: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def get_portfolio_value(self, current_prices: dict):
        value = self.fiat_balance
        for sym, qty in self.holdings.items():
            if sym in current_prices:
                value += qty * current_prices[sym]
        return value

    def save_state_to_db(self, db):
        # We maintain a single record for 'latest' or insert a new snapshot?
        # For simplicity in this validation, we'll insert a new snapshot to keep history.
        state = DBPortfolioState(
            fiat_balance=self.fiat_balance,
            total_value=self.fiat_balance + sum([qty for qty in self.holdings.values()]), # Simplified
            holdings=self.holdings,
            utilization_pct=0.0 # Will be updated by pipeline
        )
        db.add(state)

    def load_state(self):
        db = SessionLocal()
        try:
            # Load the most recent state
            latest_state = db.query(DBPortfolioState).order_by(DBPortfolioState.timestamp.desc()).first()
            if latest_state:
                self.fiat_balance = latest_state.fiat_balance
                self.holdings = latest_state.holdings or {}
                logger.info("SUPABASE-BROKER: State successfully loaded from Cloud DB.")
            else:
                logger.warning("SUPABASE-BROKER: No state found in Supabase. Initializing default balance.")
                self.save_state_to_db(db)
                db.commit()
        except Exception as e:
            logger.error(f"SUPABASE-BROKER: Failed to load state: {e}")
        finally:
            db.close()
