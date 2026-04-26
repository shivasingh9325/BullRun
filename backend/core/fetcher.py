import pandas as pd
import yfinance as yf
import os
import yaml
from pathlib import Path
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class DataFetcher:
    def __init__(self, config_path: str = 'configs/prod_params.yaml'):
        # Resolve config path robustly
        resolved_config_path = config_path
        if not os.path.exists(resolved_config_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            alt_path = os.path.join(base_dir, "configs", "prod_params.yaml")
            if os.path.exists(alt_path):
                resolved_config_path = alt_path
            else:
                raise FileNotFoundError(f"CRITICAL: Configuration file not found at {config_path} or {alt_path}")

        with open(resolved_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = self.config.get('paths', {}).get('raw_data', 'data/raw')
        
        # Ensure raw_dir is absolute or relative to resolved_config_path
        if not os.path.isabs(self.raw_dir):
            base_dir = os.path.dirname(os.path.abspath(resolved_config_path))
            self.raw_dir = os.path.abspath(os.path.join(base_dir, "..", self.raw_dir))
            
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def fetch_historical_nifty50(self):
        """
        Uses the existing combined_data.csv for MVP speed, 
        or falls back to yfinance if missing.
        """
        csv_path = 'e:/Bull_Run/dataset/combined_data.csv' # Known historical dump
        if os.path.exists(csv_path):
            logger.info(f"Loading historical baseline from {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            if 'Adj Close' in df.columns:
                df = df.drop(columns=['Adj Close']) # Known cleanup
            return df
        else:
            logger.warning("Local baseline not found. Live yfinance fetch not fully implemented for MVP backtester.")
            return pd.DataFrame()
            
    def fetch_all(self, tickers: list, period: str = "2y", interval: str = "1d"):
        """
        Fetches and flattens data for multiple tickers.
        """
        logger.info(f"FETCH: Grabbing {period} history for {tickers}")
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker", progress=False)
        
        temp_list = []
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                ticker_df = data[ticker].copy()
                ticker_df['Stock'] = ticker
                ticker_df = ticker_df.reset_index()
                temp_list.append(ticker_df)
        
        if not temp_list:
            return pd.DataFrame()
            
        combined = pd.concat(temp_list).sort_values(by=['Date', 'Stock']).reset_index(drop=True)
        return combined
