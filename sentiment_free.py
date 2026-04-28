# sentiment_free.py — Fear & Greed Index + SPY RSI (no LLM / no paid API)
import requests
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
from config import FEAR_GREED_URL, FEAR_GREED_TIMEOUT, SPY_TICKER

logger = logging.getLogger(__name__)


# ── Fear & Greed Index ───────────────────────────────────────────────────────

def get_fear_greed() -> dict:
    """
    Fetches CNN Fear & Greed Index (free, no API key required).
    Returns:
        {
          "score": float (0–100),
          "label": str  ("Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"),
          "normalized": float (0.0–1.0),   # ready to use as ML feature
          "timestamp": str
        }
    Falls back to neutral (50) on any error.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://edition.cnn.com/markets/fear-and-greed",
        "Accept":  "application/json",
    }
    try:
        resp = requests.get(FEAR_GREED_URL, headers=headers, timeout=FEAR_GREED_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        score = float(data["fear_and_greed"]["score"])
        label = data["fear_and_greed"]["rating"]

        return {
            "score":      score,
            "label":      label,
            "normalized": score / 100.0,
            "timestamp":  datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed ({e}), using neutral fallback.")
        return {
            "score":      50.0,
            "label":      "Neutral (fallback)",
            "normalized": 0.50,
            "timestamp":  datetime.utcnow().isoformat(),
        }


def classify_fear_greed(score: float) -> str:
    """Human-readable label from raw score."""
    if score <= 25:
        return "Extreme Fear"
    elif score <= 45:
        return "Fear"
    elif score <= 55:
        return "Neutral"
    elif score <= 75:
        return "Greed"
    else:
        return "Extreme Greed"


# ── SPY RSI as Market Sentiment Proxy ────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_spy_rsi(period: int = 14, lookback_days: int = 60) -> dict:
    """
    Downloads SPY daily closes and returns latest RSI.
    Returns:
        {
          "rsi": float,
          "normalized": float (0.0–1.0),
          "signal": str ("oversold" | "neutral" | "overbought")
        }
    Falls back to 50 on any error.
    """
    try:
        end   = datetime.today()
        start = end - timedelta(days=lookback_days + 10)
        df    = yf.download(SPY_TICKER, start=start, end=end, progress=False, auto_adjust=True)

        if df.empty or len(df) < period + 1:
            raise ValueError("Insufficient SPY data")

        rsi_series = _rsi(df["Close"].squeeze(), period)
        latest_rsi = float(rsi_series.iloc[-1])

        if latest_rsi <= 30:
            signal = "oversold"
        elif latest_rsi >= 70:
            signal = "overbought"
        else:
            signal = "neutral"

        return {
            "rsi":        round(latest_rsi, 2),
            "normalized": latest_rsi / 100.0,
            "signal":     signal,
        }
    except Exception as e:
        logger.warning(f"SPY RSI calculation failed ({e}), using neutral fallback.")
        return {"rsi": 50.0, "normalized": 0.50, "signal": "neutral"}


# ── News Sentiment via yfinance + VADER ──────────────────────────────────────

def get_news_sentiment(ticker: str, max_headlines: int = 10) -> dict:
    """
    Fetches recent headlines for ticker via yfinance.Ticker.news and scores
    them with VADER (rule-based, no model download required).

    Returns:
        {
          "compound":       float  (-1.0 to +1.0, VADER compound average),
          "normalized":     float  (0.0 to 1.0, ready to use as ML feature),
          "label":          str    ("Bullish" | "Neutral" | "Bearish"),
          "headline_count": int,
        }
    Falls back to neutral (0.5 / 0.0 compound) on any error or no news.
    """
    _neutral = {"compound": 0.0, "normalized": 0.5, "label": "Neutral", "headline_count": 0}
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        news = yf.Ticker(ticker).news or []
        if not news:
            logger.info(f"[{ticker}] No news found via yfinance.")
            return _neutral

        headlines = []
        for item in news[:max_headlines]:
            # yfinance >=0.2.x nests title inside content dict
            title = (item.get("content") or {}).get("title") or item.get("title", "")
            if title:
                headlines.append(title)

        if not headlines:
            return _neutral

        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        avg = sum(scores) / len(scores)
        normalized = (avg + 1.0) / 2.0

        if avg >= 0.05:
            label = "Bullish"
        elif avg <= -0.05:
            label = "Bearish"
        else:
            label = "Neutral"

        logger.info(f"[{ticker}] News sentiment: {label} ({avg:+.3f}, {len(headlines)} headlines)")
        return {
            "compound":       round(avg, 4),
            "normalized":     round(normalized, 4),
            "label":          label,
            "headline_count": len(headlines),
        }
    except Exception as e:
        logger.warning(f"[{ticker}] News sentiment error ({e}), using neutral fallback.")
        return _neutral


# ── Combined Sentiment Package ────────────────────────────────────────────────

def get_sentiment_features() -> dict:
    """
    Returns a unified sentiment feature dict used by both the scorer and ML model.

    Feature range notes:
    - fear_greed_score  : 0–100
    - fear_greed_norm   : 0.0–1.0
    - spy_rsi           : 0–100
    - spy_rsi_norm      : 0.0–1.0
    - combined_sentiment: 0.0–1.0  (average of both normalized values)
    - market_bullish    : bool
    """
    fg  = get_fear_greed()
    spy = get_spy_rsi()

    combined = (fg["normalized"] + spy["normalized"]) / 2.0
    bullish  = combined >= 0.50

    return {
        "fear_greed_score":   fg["score"],
        "fear_greed_label":   fg["label"],
        "fear_greed_norm":    fg["normalized"],
        "spy_rsi":            spy["rsi"],
        "spy_rsi_norm":       spy["normalized"],
        "spy_rsi_signal":     spy["signal"],
        "combined_sentiment": round(combined, 4),
        "market_bullish":     bullish,
        "fetched_at":         fg["timestamp"],
    }
