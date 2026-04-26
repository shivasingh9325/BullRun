import os
import yaml
import json
import random
from datetime import datetime
import pandas as pd
import numpy as np
from backend.core.fetcher import DataFetcher
from backend.core.technical import calculate_technical_features
from backend.models.agent import PortfolioManagerAgent
from backend.models.meta import MetaModel
from backend.models.technical_model import TechnicalPredictor
from backend.models.sentiment_model import SentimentPredictor
from backend.broker.broker_mock import MockBroker
from backend.utils.logger import get_logger
from backend.db.session import SessionLocal
from backend.db.models import DBSystemAudit

logger = get_logger(__name__)


class DailyInferencePipeline:
    def __init__(self, config_path: str = "configs/prod_params.yaml"):
        # Resolve config path robustly
        self.config_path = config_path
        if not os.path.exists(self.config_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            alt_path = os.path.join(base_dir, "configs", "prod_params.yaml")
            if os.path.exists(alt_path):
                self.config_path = alt_path
            else:
                raise FileNotFoundError(
                    f"CRITICAL: Configuration file not found at {self.config_path} or {alt_path}"
                )

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.prod_cfg = self.config["production"]
        self.rl_cfg = self.config["rl"]
        self.paths_cfg = self.config.get("paths", {})
        self.broker = MockBroker()

        # Configurable confidence threshold (from YAML, not hardcoded)
        self.conf_threshold = self.prod_cfg.get("min_trade_confidence", 0.55)

        # Load Predictors
        models_dir = self.paths_cfg.get("models", "models")
        if not os.path.isabs(models_dir):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            models_dir = os.path.join(base_dir, "models")

        self.models_dir = models_dir

        tech_model_dir = os.path.abspath(os.path.join(self.models_dir, "technical_model"))
        self.tech_predictor = TechnicalPredictor(tech_model_dir)

        meta_path = os.path.abspath(os.path.join(self.models_dir, "meta_model", "xgb_meta_model.pkl"))
        self.meta_model = MetaModel(meta_path, confidence_threshold=self.conf_threshold)

        # Real Sentiment Predictor (Finnhub API + keyword NLP, with graceful fallback)
        self.sentiment_predictor = SentimentPredictor(days_back=7)

        # Load Nifty50 universe mapping for RL One-Hot
        nifty_path = os.path.join(os.path.dirname(os.path.abspath(self.config_path)), "nifty50.json")
        if os.path.exists(nifty_path):
            with open(nifty_path, "r") as f:
                self.universe = json.load(f)
            self.ticker_map = {t: i for i, t in enumerate(self.universe)}
        else:
            self.universe = []
            self.ticker_map = {}

    def check_market_timing(self):
        now = datetime.now()
        close_hour = self.prod_cfg.get("market_close_hour", 16)
        if now.hour < close_hour:
            logger.warning(
                f"MARKET-GUARD: It is currently {now.strftime('%H:%M')}. "
                f"Pipeline runs only after {close_hour}:00."
            )
            return False
        return True

    def check_idempotency(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self.prod_cfg.get("last_run_date") == today:
            logger.info(f"IDEMPOTENCY: Pipeline already executed for {today}. Skipping.")
            return False
        return True

    def get_drawdown_scaler(self):
        path = self.prod_cfg.get("equity_curve_path", "logs/equity_curve.csv")
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.abspath(self.config_path))
            path = os.path.abspath(os.path.join(base_dir, "..", path))

        if not os.path.exists(path):
            return 1.0

        try:
            df = pd.read_csv(path)
            if len(df) < 2:
                return 1.0

            values = df["portfolio_value"].tolist()
            peak = max(values)
            current = values[-1]
            drawdown = (peak - current) / peak if peak > 0 else 0

            soft_cfg = self.prod_cfg.get("soft_drawdown", {})
            t1 = soft_cfg.get("tier2", {"threshold": 0.10, "scale": 0.50})
            halt = soft_cfg.get("halt", {"threshold": 0.15, "scale": 0.00})

            if drawdown >= halt["threshold"]:
                return halt["scale"]
            if drawdown >= t1["threshold"]:
                return t1["scale"]

            return 1.0
        except Exception as e:
            logger.error(f"RISK-GUARD: Drawdown calculation failed: {e}")
            return 1.0

    def check_sector_exposure(self, symbol: str, prices: dict):
        sector_map_path = self.prod_cfg.get("sector_map_path", "configs/sector_map.json")
        if not os.path.isabs(sector_map_path):
            base_dir = os.path.dirname(os.path.abspath(self.config_path))
            sector_map_path = os.path.abspath(os.path.join(base_dir, "..", sector_map_path))

        if not os.path.exists(sector_map_path):
            return True, "No Sector Map"

        with open(sector_map_path, "r") as f:
            sector_map = json.load(f)

        target_sector = sector_map.get(symbol, "Unknown")
        cap = self.prod_cfg.get("sector_exposure_cap", 0.30)

        portfolio_value = self.broker.get_portfolio_value(prices)
        if portfolio_value == 0:
            return True, "Empty Portfolio"

        sector_value = 0
        for sym, qty in self.broker.holdings.items():
            if sector_map.get(sym) == target_sector:
                sector_value += qty * prices.get(sym, 0)

        current_sector_pct = sector_value / portfolio_value
        if current_sector_pct >= cap:
            return False, f"Sector Cap reached: {target_sector} ({current_sector_pct:.1%})"
        return True, "OK"

    def update_last_run(self):
        today = datetime.now().strftime("%Y-%m-%d")
        self.config["production"]["last_run_date"] = today
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.config, f)

    def run(self, request_data=None, force=False):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "DRY_RUN" if self.prod_cfg.get("dry_run", True) else "LIVE"
        logger.info(f"--- STEP 0: STARTING DAILY INFERENCE [RunID: {run_id} | Mode: {mode}] ---")

        # Log Start of Inference to DB
        db = SessionLocal()
        try:
            audit = DBSystemAudit(
                event_type="INFERENCE_START",
                status="Healthy",
                message=f"Starting Run {run_id}",
                details={"mode": mode}
            )
            db.add(audit)
            db.commit()
        except Exception as e:
            logger.error(f"DB-AUDIT: Failed to log start: {e}")
        finally:
            db.close()

        results = {
            "run_id": run_id,
            "mode": mode,
            "status": "Success",
            "trades_executed": 0,
            "decisions": [],
            "errors": []
        }

        if not force:
            if not self.check_market_timing():
                results["status"] = "Skipped (Timing)"
                return results
            if not self.check_idempotency():
                results["status"] = "Skipped (Idempotency)"
                return results

        # 1. Fetch Latest Data
        logger.info("--- STEP 1: DATA FETCHING ---")
        ticker_universe = self.prod_cfg.get("ticker_universe", ["RELIANCE.NS"])

        # If a specific symbol was requested via API, use that
        if request_data and hasattr(request_data, "symbol") and request_data.symbol:
            if request_data.symbol not in ticker_universe:
                ticker_universe = [request_data.symbol] + ticker_universe[:4]

        fetcher = DataFetcher(config_path=self.config_path)
        try:
            raw_df = fetcher.fetch_all(ticker_universe, period="60d")
            if raw_df.empty:
                logger.error("DATA-FAILURE: No data retrieved.")
                results["status"] = "Error (Data)"
                return results
        except Exception as e:
            logger.error(f"DATA-FAILURE: {e}")
            results["status"] = "Error (Data)"
            results["errors"].append(str(e))
            return results

        # 2. Generate Technical Features
        logger.info("--- STEP 2: FEATURE ENGINEERING ---")
        feat_df = calculate_technical_features(raw_df)

        # 3. Real Sentiment Injection (replaces random stub)
        logger.info("--- STEP 2b: SENTIMENT INJECTION ---")
        try:
            feat_df = self.sentiment_predictor.predict(feat_df)
            logger.info(f"SENTIMENT: Injected for tickers. Sample: {feat_df['Sentiment_Score'].mean():.4f}")
        except Exception as e:
            logger.warning(f"SENTIMENT-FAILURE: {e}. Using neutral fallback.")
            feat_df["Sentiment_Score"] = 0.0
            feat_df["News_Volume"] = 0

        # 4. Model Inference (Tech -> Meta)
        logger.info("--- STEP 3: MODEL INFERENCE ---")
        try:
            logger.info("PIPELINE: Running Technical Predictor...")
            tech_df = self.tech_predictor.predict(feat_df)

            # Audit log for first ticker
            audit_stock = ticker_universe[0]
            audit_row = tech_df[tech_df["Stock"] == audit_stock].tail(1)
            if not audit_row.empty:
                logger.info(
                    f"META-AUDIT {audit_stock} (Features): "
                    f"{audit_row[self.meta_model.features].to_dict('records')[0]}"
                )

            logger.info("PIPELINE: Running Meta Predictor...")
            inference_df = self.meta_model.predict(tech_df)
        except Exception as e:
            logger.error(f"MODEL-FAILURE: Inference chain failed: {e}")
            results["status"] = "Error (Inference)"
            results["errors"].append(f"Inference Error: {e}")
            return results

        # 5. RL Decision & Risk Scaling
        logger.info("--- STEP 4 & 5: RL DECISION & RISK SCALING ---")
        drawdown_scaler = self.get_drawdown_scaler()
        agent = PortfolioManagerAgent(inference_df, config_path=self.config_path)
        try:
            rl_path = os.path.abspath(os.path.join(self.models_dir, "rl", "ppo_portfolio_manager"))
            agent.load(path=rl_path)
        except Exception as e:
            logger.warning(f"RL-LOAD-WARNING: {e}. Using rule-based fallback.")
            agent._rule_based_mode = True

        latest_records = inference_df.groupby("Stock").tail(1).copy()
        current_prices = {row["Stock"]: row["Close"] for _, row in latest_records.iterrows()}
        portfolio_value = self.broker.get_portfolio_value(current_prices)

        trades_count = 0
        max_trades = self.prod_cfg.get("max_trades_per_day", 7)

        # 6. Order Execution
        logger.info("--- STEP 6: EXECUTION & PERSISTENCE ---")
        for idx, row in latest_records.iterrows():
            symbol = row["Stock"]
            price = row["Close"]
            meta_prob = row["Meta_Prob_BUY"]
            sentiment = row.get("Sentiment_Score", 0.0)

            # Format Observation for RL Agent
            exposure = (
                (self.broker.holdings.get(symbol, 0) * price) / portfolio_value
                if portfolio_value > 0
                else 0
            )
            core_obs = [
                self.broker.fiat_balance / 100000.0,
                exposure,
                (portfolio_value - 100000.0) / 100000.0,
                float(meta_prob),
                row["Volatility_14d"] / 100.0,
                row["Pred_Return_5d"]
            ]
            one_hot = [0.0] * len(self.universe)
            if symbol in self.ticker_map:
                one_hot[self.ticker_map[symbol]] = 1.0

            obs = np.array(core_obs, dtype=np.float32)
            rl_action = agent.predict_action(obs)
            rl_weight = float(rl_action[0]) if isinstance(rl_action, (list, np.ndarray)) else float(rl_action)

            decision = {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": round(float(meta_prob), 4),
                "allocation": 0.0,
                "price": round(float(price), 2),
                "timestamp": datetime.now().isoformat(),
                "sentiment_score": round(float(sentiment), 4),
                "rl_weight": round(rl_weight, 4),
                "risk_flags": []
            }

            logger.debug(f"AUDIT {symbol}: Meta={meta_prob:.3f} | RL={rl_weight:.2f} | Risk={drawdown_scaler:.2f}")

            # Near-neutral warn
            if 0.49 <= meta_prob <= 0.51:
                logger.warning(
                    f"CRITICAL: Meta output for {symbol} is nearly neutral ({meta_prob:.4f})."
                )
                decision["risk_flags"].append("NEAR_NEUTRAL_SIGNAL")

            # Decision Gate Logic
            sector_ok, sector_reason = self.check_sector_exposure(symbol, current_prices)

            if not sector_ok:
                decision["reason"] = sector_reason
                decision["risk_flags"].append("SECTOR_CAP")
            elif trades_count >= max_trades:
                decision["reason"] = "Daily trade limit reached"
            elif meta_prob < self.conf_threshold:
                decision["reason"] = f"Meta Rejection: {meta_prob:.2f} < {self.conf_threshold:.2f}"
            elif rl_weight <= 0.05:
                decision["reason"] = f"RL Rejection: Weight too low ({rl_weight:.2f})"
            else:
                target_alloc = rl_weight * drawdown_scaler
                fiat_to_risk = self.broker.fiat_balance * target_alloc
                qty = fiat_to_risk / price

                if qty > 0:
                    logger.info(
                        f"### EXECUTING BUY: {symbol} | Qty: {qty:.2f} | Conf: {meta_prob:.2f} ###"
                    )
                    success = self.broker.submit_order(
                        symbol, qty, price, "BUY",
                        run_id=run_id, confidence=float(meta_prob), reason="All filters passed"
                    )
                    if success:
                        trades_count += 1
                        decision["signal"] = "BUY"
                        decision["allocation"] = round(float(target_alloc), 4)
                        decision["reason"] = "All filters passed"

            results["decisions"].append(decision)

        results["trades_executed"] = trades_count
        self.update_last_run()

        # Log Completion to DB
        db = SessionLocal()
        try:
            audit = DBSystemAudit(
                event_type="INFERENCE_COMPLETE",
                status="Success",
                message=f"Completed Run {run_id} with {trades_count} trades.",
                details={"run_id": run_id, "mode": mode, "trades": trades_count}
            )
            db.add(audit)
            db.commit()
        except Exception as e:
            logger.error(f"DB-AUDIT: Failed to log completion: {e}")
        finally:
            db.close()

        logger.info(f"--- STEP 7: INFERENCE COMPLETE [Trades: {trades_count}] ---")
        return results


if __name__ == "__main__":
    pipeline = DailyInferencePipeline()
    pipeline.run(force=True)
