"""
Sentiment Analysis Service for BullRun AI.
Uses Finnhub API for real news sentiment. Falls back gracefully if unavailable.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from backend.models.base import BasePredictor
from backend.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1/company-news"

# --- Symbol normalization: remove exchange suffix for Finnhub ---
def _normalize_symbol(symbol: str) -> str:
    """Convert 'RELIANCE.NS' → 'RELIANCE'."""
    return symbol.split(".")[0]


def _score_headline(headline: str) -> float:
    """
    Simple keyword-based scorer as a lightweight NLP proxy.
    Returns a score in [-1.0, 1.0].
    """
    headline_lower = headline.lower()

    positive_kw = [
        "surge", "rally", "gain", "profit", "beat", "growth", "upgrade",
        "bullish", "strong", "buy", "record", "high", "positive", "boost",
        "outperform", "exceed"
    ]
    negative_kw = [
        "fall", "drop", "loss", "decline", "miss", "downgrade", "bearish",
        "weak", "sell", "crash", "low", "negative", "risk", "underperform",
        "concern", "cut", "warning"
    ]

    pos_hits = sum(1 for kw in positive_kw if kw in headline_lower)
    neg_hits = sum(1 for kw in negative_kw if kw in headline_lower)

    total = pos_hits + neg_hits
    if total == 0:
        return 0.0

    score = (pos_hits - neg_hits) / total
    return round(score, 4)


def _fetch_news_sentiments(symbol: str, days_back: int = 7) -> tuple[float, int]:
    """
    Fetch recent news for a symbol from Finnhub and compute sentiment.
    Returns (sentiment_score, news_volume).
    """
    if not FINNHUB_API_KEY:
        logger.warning("SENTIMENT: FINNHUB_API_KEY not set. Using neutral fallback.")
        return 0.0, 0

    clean_symbol = _normalize_symbol(symbol)
    today = datetime.now()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    params = {
        "symbol": clean_symbol,
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }

    try:
        response = requests.get(FINNHUB_BASE_URL, params=params, timeout=8)
        response.raise_for_status()
        news_items = response.json()

        if not news_items or not isinstance(news_items, list):
            logger.info(f"SENTIMENT: No news found for {symbol}. Using neutral.")
            return 0.0, 0

        headlines = [item.get("headline", "") for item in news_items if item.get("headline")]
        scores = [_score_headline(h) for h in headlines]
        news_volume = len(scores)
        sentiment_score = round(float(np.mean(scores)), 4) if scores else 0.0

        logger.info(f"SENTIMENT: {symbol} | Score={sentiment_score:.4f} | Volume={news_volume}")
        return sentiment_score, news_volume

    except requests.exceptions.RequestException as e:
        logger.warning(f"SENTIMENT: API failure for {symbol}: {e}. Using neutral fallback.")
        return 0.0, 0


class SentimentPredictor(BasePredictor):
    """
    Real sentiment analysis service.
    Calls Finnhub News API and applies keyword-based NLP scoring.
    Falls back to neutral (0.0) if API is unavailable.
    """
    def __init__(self, model_path: str = "finnhub", days_back: int = 7):
        super().__init__(model_path)
        self.days_back = days_back
        self._cache: dict[str, tuple[float, int, float]] = {}  # symbol -> (score, volume, timestamp)
        self._cache_ttl_seconds = 3600  # Refresh sentiment every hour

    def load_model(self):
        logger.info("SentimentPredictor: Using Finnhub live API + keyword NLP scorer.")
        self.model = "FINNHUB_KEYWORD_NLP"

    def _get_cached_or_fetch(self, symbol: str) -> tuple[float, int]:
        now = time.time()
        if symbol in self._cache:
            score, volume, ts = self._cache[symbol]
            if now - ts < self._cache_ttl_seconds:
                logger.debug(f"SENTIMENT: Cache hit for {symbol}.")
                return score, volume

        score, volume = _fetch_news_sentiments(symbol, self.days_back)
        self._cache[symbol] = (score, volume, now)
        return score, volume

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Injects 'Sentiment_Score' and 'News_Volume' columns into the DataFrame.
        Processes per-stock and applies sentiment to all rows per stock.
        """
        if self.model is None:
            self.load_model()

        df = df.copy()
        df["Sentiment_Score"] = 0.0
        df["News_Volume"] = 0

        if "Stock" not in df.columns:
            logger.warning("SENTIMENT: No 'Stock' column found. Applying global fetch for single ticker.")
            score, volume = _fetch_news_sentiments("UNKNOWN", self.days_back)
            df["Sentiment_Score"] = score
            df["News_Volume"] = volume
            return df

        for symbol in df["Stock"].unique():
            score, volume = self._get_cached_or_fetch(symbol)
            mask = df["Stock"] == symbol
            df.loc[mask, "Sentiment_Score"] = score
            df.loc[mask, "News_Volume"] = volume

        return df
