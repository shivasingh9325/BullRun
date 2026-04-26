import pandas as pd
import numpy as np

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive feature extractor for the Technical and Meta models.
    """
    temp_list = []
    
    for stock, group in df.groupby('Stock'):
        stock_df = group.copy().sort_values('Date')
        
        # 1. Moving Averages
        stock_df['SMA_10'] = stock_df['Close'].rolling(window=10).mean()
        stock_df['SMA_20'] = stock_df['Close'].rolling(window=20).mean()
        stock_df['EMA_12'] = stock_df['Close'].ewm(span=12, adjust=False).mean()
        stock_df['EMA_26'] = stock_df['Close'].ewm(span=26, adjust=False).mean()
        
        # 2. MACD
        stock_df['MACD'] = stock_df['EMA_12'] - stock_df['EMA_26']
        stock_df['MACD_Signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
        stock_df['MACD_Hist'] = stock_df['MACD'] - stock_df['MACD_Signal']
        
        # 3. RSI
        delta = stock_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 4. Bollinger Bands
        std_20 = stock_df['Close'].rolling(window=20).std()
        stock_df['BB_Upper'] = stock_df['SMA_20'] + (std_20 * 2)
        stock_df['BB_Lower'] = stock_df['SMA_20'] - (std_20 * 2)
        stock_df['BB_Width'] = (stock_df['BB_Upper'] - stock_df['BB_Lower']) / stock_df['SMA_20']
        
        # 5. ATR (Average True Range)
        high_low = stock_df['High'] - stock_df['Low']
        high_cp = np.abs(stock_df['High'] - stock_df['Close'].shift())
        low_cp = np.abs(stock_df['Low'] - stock_df['Close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        stock_df['ATR_14'] = tr.rolling(window=14).mean()
        
        # 6. Volume
        stock_df['Volume_SMA_10'] = stock_df['Volume'].rolling(window=10).mean()
        stock_df['Volume_Change_Pct'] = stock_df['Volume'].pct_change() * 100
        
        # 7. Price Action
        stock_df['Daily_Return_Pct'] = stock_df['Close'].pct_change() * 100
        stock_df['HL_Range_Pct'] = ((stock_df['High'] - stock_df['Low']) / stock_df['Low']) * 100
        stock_df['CO_Change_Pct'] = ((stock_df['Close'] - stock_df['Open']) / stock_df['Open']) * 100
        stock_df['Price_vs_SMA20'] = stock_df['Close'] / stock_df['SMA_20']
        
        # 8. Decision Features (For MetaModel)
        stock_df['Volatility_14d'] = stock_df['Daily_Return_Pct'].rolling(window=14).std() * np.sqrt(252)
        # Simplified Max Drawdown
        rolling_max = stock_df['Close'].rolling(window=30, min_periods=1).max()
        stock_df['Max_Drawdown_30d'] = ((stock_df['Close'] - rolling_max) / rolling_max) * 100
        # Dummy proxy for Pred_Return if not using a dedicated return predictor yet
        stock_df['Pred_Return_5d'] = stock_df['Daily_Return_Pct'].rolling(window=5).mean()
        
        # FINAL SAFETY: Fill NaNs/Infs for this stock's block
        stock_df = stock_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        temp_list.append(stock_df)
        
    if not temp_list:
        return pd.DataFrame()
        
    return pd.concat(temp_list).reset_index(drop=True)
