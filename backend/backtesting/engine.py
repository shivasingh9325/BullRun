"""
BullRun AI — Backtesting Engine
=================================
Simulates the full inference pipeline on historical data and reports:
  - Profit / Loss (absolute and %)
  - Max Drawdown (%)
  - Sharpe Ratio (annualized)
  - Total Trades
  - Win Rate (%)

Usage:
    from backend.backtesting.engine import BacktestEngine

    engine = BacktestEngine()
    results = engine.run(tickers=["RELIANCE.NS", "TCS.NS"], period="1y")
    engine.print_report(results)

CLI:
    python -m bullrun.backtesting.engine
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

# Allow running directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from backend.core.fetcher import DataFetcher
from backend.core.technical import calculate_technical_features
from backend.models.technical_model import TechnicalPredictor
from backend.models.meta import MetaModel
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Simulates BullRun AI pipeline on historical OHLCV data.

    Assumptions:
        - Paper trading: starts with initial_balance cash.
        - A BUY signal allocates a fixed fraction of available cash.
        - Positions are held until a SELL/HOLD signal falls below threshold.
        - No partial fills; simple market orders at next-day Open.
        - Transaction cost = slippage + fee (from config).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        initial_balance: float = 100_000.0
    ):
        # Resolve config
        if config_path is None:
            config_path = os.path.join(_BACKEND_ROOT, "configs", "prod_params.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.initial_balance = initial_balance
        self.prod_cfg = self.config.get("production", {})
        self.cost_cfg = self.config.get("costs", {})

        # Cost parameters
        self.slippage_pct = self.cost_cfg.get("slippage_pct", 0.05) / 100.0  # config is in %
        self.fee_pct = self.cost_cfg.get("brokerage_fee_pct", 0.025) / 100.0
        self.total_cost_pct = self.slippage_pct + self.fee_pct

        self.conf_threshold = self.prod_cfg.get("min_trade_confidence", 0.55)
        self.alloc_per_trade = self.prod_cfg.get("max_alloc_per_trade", 0.15)

        # Load Models
        models_dir = os.path.join(_BACKEND_ROOT, "models")
        tech_model_dir = os.path.join(models_dir, "technical_model")
        meta_model_path = os.path.join(models_dir, "meta_model", "xgb_meta_model.pkl")

        if not os.path.exists(tech_model_dir):
            raise FileNotFoundError(f"Technical model directory not found: {tech_model_dir}")
        if not os.path.exists(meta_model_path):
            raise FileNotFoundError(f"Meta model not found: {meta_model_path}")

        self.tech_predictor = TechnicalPredictor(tech_model_dir)
        self.meta_model = MetaModel(meta_model_path, confidence_threshold=self.conf_threshold)

        logger.info(f"BacktestEngine initialized | Balance={initial_balance:,.0f} | Threshold={self.conf_threshold}")

    def run(
        self,
        tickers: Optional[list] = None,
        period: str = "1y"
    ) -> dict:
        """
        Execute the backtest simulation.

        Args:
            tickers: List of ticker symbols. Defaults to prod config universe.
            period: yfinance period string ('6mo', '1y', '2y').

        Returns:
            dict with per-ticker and aggregate metrics.
        """
        if tickers is None:
            tickers = self.prod_cfg.get("ticker_universe", ["RELIANCE.NS"])

        logger.info(f"BACKTEST: Starting simulation for {tickers} over {period}")

        # 1. Fetch Data
        config_path = os.path.join(_BACKEND_ROOT, "configs", "prod_params.yaml")
        fetcher = DataFetcher(config_path=config_path)
        raw_df = fetcher.fetch_all(tickers, period=period)

        if raw_df.empty:
            logger.error("BACKTEST: No data fetched. Aborting.")
            return {"error": "No data fetched", "tickers": tickers}

        # 2. Feature Engineering
        feat_df = calculate_technical_features(raw_df)

        # 3. Add neutral sentiment baseline for backtesting
        feat_df["Sentiment_Score"] = 0.0
        feat_df["News_Volume"] = 5

        # 4. Technical + Meta Model Inference
        tech_df = self.tech_predictor.predict(feat_df)
        inference_df = self.meta_model.predict(tech_df)

        # 5. Simulate trading per ticker
        ticker_results = {}
        for ticker in tickers:
            ticker_df = inference_df[inference_df["Stock"] == ticker].copy().reset_index(drop=True)
            if len(ticker_df) < 5:
                logger.warning(f"BACKTEST: Insufficient data for {ticker}. Skipping.")
                continue
            result = self._simulate_ticker(ticker, ticker_df)
            ticker_results[ticker] = result

        # 6. Aggregate Metrics
        aggregate = self._aggregate_metrics(ticker_results)
        aggregate["ticker_results"] = ticker_results
        aggregate["backtest_period"] = period
        aggregate["tickers"] = tickers
        aggregate["run_timestamp"] = datetime.now().isoformat()

        return aggregate

    def _simulate_ticker(self, ticker: str, df: pd.DataFrame) -> dict:
        """Simulate buy/hold/sell decisions on a single ticker's history."""
        cash = self.initial_balance
        shares = 0.0
        avg_cost = 0.0
        equity_curve = []
        trades = []

        for i, row in df.iterrows():
            price = float(row.get("Close", 0))
            if price <= 0:
                equity_curve.append(cash + shares * price)
                continue

            meta_buy_prob = float(row.get("Meta_Prob_BUY", 0.0))
            meta_sell_prob = float(row.get("Meta_Prob_SELL", 0.0))
            portfolio_value = cash + shares * price

            # --- SELL LOGIC ---
            if shares > 0 and meta_sell_prob > self.conf_threshold:
                proceeds = shares * price * (1 - self.total_cost_pct)
                pnl = proceeds - (shares * avg_cost)
                trades.append({
                    "date": str(row.get("Date", i)),
                    "action": "SELL",
                    "price": price,
                    "shares": shares,
                    "pnl": round(pnl, 2)
                })
                cash += proceeds
                shares = 0.0
                avg_cost = 0.0

            # --- BUY LOGIC ---
            elif meta_buy_prob >= self.conf_threshold and shares == 0:
                fiat_to_invest = cash * self.alloc_per_trade
                cost_adjusted = fiat_to_invest / (1 + self.total_cost_pct)
                qty = cost_adjusted / price
                if qty > 0:
                    shares = qty
                    avg_cost = price * (1 + self.total_cost_pct)
                    cash -= (qty * price * (1 + self.total_cost_pct))
                    trades.append({
                        "date": str(row.get("Date", i)),
                        "action": "BUY",
                        "price": price,
                        "shares": qty,
                        "pnl": 0.0
                    })

            equity_curve.append(cash + shares * price)

        # Liquidate at end
        final_price = float(df.iloc[-1]["Close"])
        if shares > 0:
            cash += shares * final_price * (1 - self.total_cost_pct)
            shares = 0.0

        return self._compute_metrics(
            ticker=ticker,
            equity_curve=equity_curve,
            trades=trades,
            final_cash=cash
        )

    def _compute_metrics(
        self,
        ticker: str,
        equity_curve: list,
        trades: list,
        final_cash: float
    ) -> dict:
        """Compute P/L, Max Drawdown, Sharpe Ratio, Win Rate."""
        if not equity_curve:
            return {}

        eq = np.array(equity_curve, dtype=float)

        # --- P&L ---
        pnl_abs = final_cash - self.initial_balance
        pnl_pct = (pnl_abs / self.initial_balance) * 100.0

        # --- Max Drawdown ---
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / np.where(peak > 0, peak, 1)
        max_drawdown_pct = float(np.min(drawdown)) * 100.0

        # --- Sharpe Ratio (annualized, assuming daily data) ---
        daily_returns = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # --- Win Rate ---
        completed_sells = [t for t in trades if t["action"] == "SELL"]
        wins = [t for t in completed_sells if t["pnl"] > 0]
        win_rate = (len(wins) / len(completed_sells)) * 100.0 if completed_sells else 0.0

        logger.info(
            f"BACKTEST {ticker}: P/L={pnl_pct:.2f}% | "
            f"MaxDD={max_drawdown_pct:.2f}% | Sharpe={sharpe:.2f} | "
            f"Trades={len(trades)} | WinRate={win_rate:.1f}%"
        )

        return {
            "ticker": ticker,
            "initial_balance": self.initial_balance,
            "final_value": round(final_cash, 2),
            "pnl_abs": round(pnl_abs, 2),
            "pnl_pct": round(pnl_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 4),
            "total_trades": len(trades),
            "win_rate_pct": round(win_rate, 2),
            "trades": trades
        }

    def _aggregate_metrics(self, ticker_results: dict) -> dict:
        """Compute portfolio-level aggregate metrics."""
        if not ticker_results:
            return {}

        all_pnl = [r.get("pnl_pct", 0) for r in ticker_results.values()]
        all_dd = [r.get("max_drawdown_pct", 0) for r in ticker_results.values()]
        all_sharpe = [r.get("sharpe_ratio", 0) for r in ticker_results.values()]
        all_trades = [r.get("total_trades", 0) for r in ticker_results.values()]
        all_wr = [r.get("win_rate_pct", 0) for r in ticker_results.values() if r.get("win_rate_pct", 0) > 0]

        return {
            "avg_pnl_pct": round(float(np.mean(all_pnl)), 2),
            "worst_max_drawdown_pct": round(float(np.min(all_dd)), 2),
            "avg_sharpe_ratio": round(float(np.mean(all_sharpe)), 4),
            "total_trades": int(sum(all_trades)),
            "avg_win_rate_pct": round(float(np.mean(all_wr)), 2) if all_wr else 0.0
        }

    def print_report(self, results: dict):
        """Pretty-print the backtest results."""
        sep = "=" * 65
        print(f"\n{sep}")
        print("  BULLRUN AI — BACKTEST REPORT")
        print(f"  Period : {results.get('backtest_period', 'N/A')}")
        print(f"  Tickers: {', '.join(results.get('tickers', []))}")
        print(f"  Run At : {results.get('run_timestamp', 'N/A')}")
        print(sep)

        for ticker, r in results.get("ticker_results", {}).items():
            print(f"\n  [{ticker}]")
            print(f"    Initial Balance : ₹{r['initial_balance']:>12,.2f}")
            print(f"    Final Value     : ₹{r['final_value']:>12,.2f}")
            sign = "+" if r['pnl_abs'] >= 0 else ""
            print(f"    Net P/L         : {sign}₹{r['pnl_abs']:,.2f}  ({sign}{r['pnl_pct']:.2f}%)")
            print(f"    Max Drawdown    : {r['max_drawdown_pct']:.2f}%")
            print(f"    Sharpe Ratio    : {r['sharpe_ratio']:.4f}")
            print(f"    Total Trades    : {r['total_trades']}")
            print(f"    Win Rate        : {r['win_rate_pct']:.1f}%")

        print(f"\n{sep}")
        print("  PORTFOLIO AGGREGATE")
        print(f"    Avg P/L          : {results.get('avg_pnl_pct', 0):+.2f}%")
        print(f"    Worst Drawdown   : {results.get('worst_max_drawdown_pct', 0):.2f}%")
        print(f"    Avg Sharpe Ratio : {results.get('avg_sharpe_ratio', 0):.4f}")
        print(f"    Total Trades     : {results.get('total_trades', 0)}")
        print(f"    Avg Win Rate     : {results.get('avg_win_rate_pct', 0):.1f}%")
        print(f"{sep}\n")


# ── CLI Entry Point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BullRun AI Backtesting Engine")
    parser.add_argument("--period", default="6mo", help="yfinance period: 3mo, 6mo, 1y, 2y")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        help="List of ticker symbols"
    )
    args = parser.parse_args()

    engine = BacktestEngine()
    results = engine.run(tickers=args.tickers, period=args.period)
    engine.print_report(results)
