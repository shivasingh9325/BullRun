import sys
import os
import pandas as pd
from bullrun.utils.logger import get_logger
from bullrun.data.fetcher import DataFetcher
from bullrun.features.technical import calculate_technical_features
from bullrun.features.derived import calculate_derived_features
from bullrun.models.meta import MetaModel
from bullrun.execution.backtester import Backtester

logger = get_logger(__name__)

def run_daily_inference():
    logger.info("--- BullRun AI Pipeline Started ---")
    
    # 1. Fetch Data
    fetcher = DataFetcher()
    data = fetcher.fetch_historical_nifty50()
    if data.empty:
        logger.error("No data fetched. Pipeline halting.")
        sys.exit(1)
        
    # 2. Tech Features
    logger.info("Computing Technical Features...")
    tech_df = calculate_technical_features(data)
    
    # 3. Derived Features
    logger.info("Computing Risk & Price Features...")
    derived_df = calculate_derived_features(tech_df)
    
    # For MVP Integration we assume Meta output already exists locally 
    # to avoid loading 4 separate raw .pkl files in the pipeline integration
    # Here we mock the merged dataset for the Backtester to use
    meta_path = 'e:/Bull_Run/data/processed/meta_signals.csv'
    if os.path.exists(meta_path):
        logger.info(f"Loading merged Meta signals from {meta_path}...")
        final_df = pd.read_csv(meta_path, parse_dates=['Date'])
        
        # 4. Execution / Paper Trace
        bt = Backtester()
        bt.run(final_df)
    else:
        logger.warning("Meta signals missing. Please train base models first.")
        
    logger.info("--- BullRun AI Pipeline Completed ---")

if __name__ == "__main__":
    run_daily_inference()
