# scorer.py — 50% Technical + 50% ML scoring engine (no LLM)
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from config import (
    WEIGHT_TECHNICAL, WEIGHT_ML,
    ETF_UNIVERSE, HISTORY_DAYS, MIN_SCORE_BUY,
    LONGTERM_UNIVERSE, MIN_SCORE_LONGTERM_BUY, STOCKS_UNIVERSE,
    ATR_PERIOD, ADX_PERIOD, ADX_MIN_TREND, ADX_STRONG_TREND,
    SPY_SMA_FILTER, CONSEC_CANDLES,
)
from sentiment_free import get_sentiment_features
from ml_model import predict_all, get_model

logger = logging.getLogger(__name__)


# ── Technical Score (0–100) ───────────────────────────────────────────────────

def _rsi(s: pd.Series, p: int = 14) -> float:
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    loss  = (-delta).clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    rs    = gain / loss.replace(0, 1e-9)
    rsi   = (100 - 100 / (1 + rs)).iloc[-1]
    return float(rsi)


def _macd_signal(close: pd.Series) -> float:
    """Returns MACD histogram (positive = bullish)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    return float((macd - sig).iloc[-1])


def _sma_trend(close: pd.Series) -> float:
    """Score 0–100: golden cross quality above/below SMAs."""
    sma50  = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    last   = float(close.iloc[-1])

    score = 0.0
    if last > sma50:   score += 35
    if last > sma200:  score += 35
    if sma50 > sma200: score += 30   # golden cross
    return score


def _bollinger_position(close: pd.Series) -> float:
    """Score 0–100: position within Bollinger Bands (50 = mid)."""
    sma = close.rolling(20).mean().iloc[-1]
    std = close.rolling(20).std().iloc[-1]
    if std == 0:
        return 50.0
    z = (float(close.iloc[-1]) - sma) / std
    # z in [-2, +2] → scale to [0, 100]
    return float(np.clip((z + 2) / 4 * 100, 0, 100))


def _consecutive_bullish(close: pd.Series, n: int = CONSEC_CANDLES) -> bool:
    """True si las ultimas n velas cerraron al alza (confirmacion de tendencia)."""
    if len(close) < n + 1:
        return True  # sin datos suficientes, no bloquear
    diffs = close.diff().iloc[-n:]
    return bool((diffs > 0).all())


def _spy_above_sma50(spy_close: pd.Series) -> bool:
    """True si SPY esta por encima de su SMA50 (mercado en tendencia alcista)."""
    if len(spy_close) < 50:
        return True
    return float(spy_close.iloc[-1]) > float(spy_close.rolling(50).mean().iloc[-1])


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Average True Range — mide la volatilidad real del mercado."""
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(com=period - 1, min_periods=period).mean().iloc[-1])


def _adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> tuple[float, float, float]:
    """
    Average Directional Index — mide la fuerza de la tendencia (no la direccion).
    Retorna (adx, plus_di, minus_di).
      ADX < 20 : mercado lateral   → penalizar entrada
      ADX 20-25: tendencia moderada → neutral
      ADX >= 25: tendencia fuerte   → bonificar entrada
    """
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm  = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # When both positive, keep only the larger one
    mask      = plus_dm >= minus_dm
    plus_dm   = plus_dm.where(mask, 0.0)
    minus_dm  = minus_dm.where(~mask, 0.0)

    atr_s    = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di  = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_s.replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_s.replace(0, 1e-9)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    adx = float(dx.ewm(com=period - 1, min_periods=period).mean().iloc[-1])
    return adx, float(plus_di.iloc[-1]), float(minus_di.iloc[-1])


def _volume_signal(volume: pd.Series) -> float:
    """Score 0–100: recent volume vs 20-day average."""
    ratio = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
    # ratio 0→0, 1→50, 2→100
    return float(min(ratio * 50, 100))


def compute_technical_score(df: pd.DataFrame) -> dict:
    """
    Returns {
      "score": float 0–100,
      "atr":   float (volatilidad absoluta),
      "components": {...}
    }
    ADX actua como multiplicador del score tecnico:
      ADX < 20  → x0.75 (mercado lateral, penalizar)
      ADX 20-25 → x1.00 (tendencia moderada)
      ADX >= 25 → x1.10 (tendencia fuerte, bonificar)
    """
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    rsi   = _rsi(close)
    macd  = _macd_signal(close)
    trend = _sma_trend(close)
    bb    = _bollinger_position(close)
    vol   = _volume_signal(volume)
    atr   = _atr(df)
    adx, plus_di, minus_di = _adx(df)

    # RSI scoring: sweet spot 40–70
    if rsi < 30:
        rsi_score = 20.0
    elif rsi <= 50:
        rsi_score = 40.0 + (rsi - 30) * 1.0
    elif rsi <= 70:
        rsi_score = 60.0 + (rsi - 50) * 1.0
    else:
        rsi_score = max(0.0, 90.0 - (rsi - 70) * 2.0)

    # MACD normalized relative to price (0.2% of price as scale unit)
    price = float(close.iloc[-1])
    macd_norm = float(np.clip(50 + (macd / max(price * 0.002, 1e-6)) * 25, 0, 100))

    # Weighted technical sub-scores (weights unchanged)
    score = (
        rsi_score * 0.25 +
        macd_norm * 0.25 +
        trend     * 0.30 +
        bb        * 0.10 +
        vol       * 0.10
    )

    # ADX multiplier — filtra mercados laterales
    if adx < ADX_MIN_TREND:
        adx_mult  = 0.75
        adx_label = "lateral"
    elif adx >= ADX_STRONG_TREND:
        adx_mult  = 1.10
        adx_label = "fuerte"
    else:
        adx_mult  = 1.00
        adx_label = "moderada"

    score = float(np.clip(score * adx_mult, 0, 100))

    return {
        "score": round(score, 2),
        "atr":   round(atr, 4),
        "components": {
            "rsi":       round(rsi, 2),
            "rsi_score": round(rsi_score, 2),
            "macd_norm": round(macd_norm, 2),
            "sma_trend": round(trend, 2),
            "bb_pos":    round(bb, 2),
            "vol_ratio": round(vol, 2),
            "adx":       round(adx, 2),
            "adx_label": adx_label,
            "plus_di":   round(plus_di, 2),
            "minus_di":  round(minus_di, 2),
        },
    }


# ── ML Score (0–100) ─────────────────────────────────────────────────────────

def compute_ml_score(ticker: str, df: pd.DataFrame, sentiment: dict) -> dict:
    """
    Returns {
      "score": float 0–100,
      "proba": float 0–1
    }
    """
    model  = get_model(ticker)
    proba  = model.predict_proba(df, sentiment)
    score  = round(proba * 100, 2)
    return {"score": score, "proba": proba}


# ── Composite Score ───────────────────────────────────────────────────────────

def score_etf(ticker: str, df: pd.DataFrame, sentiment: dict, spy_above_sma: bool = True) -> dict:
    """
    Compute the full 0–100 composite score for one ETF.

    Returns:
    {
      "ticker":    str,
      "score":     float,   # 0–100 composite
      "signal":    str,     # "BUY" | "HOLD" | "SELL"
      "technical": dict,
      "ml":        dict,
      "sentiment": dict,
      "timestamp": str,
    }
    """
    tech = compute_technical_score(df)
    ml   = compute_ml_score(ticker, df, sentiment)

    composite = round(
        tech["score"] * WEIGHT_TECHNICAL +
        ml["score"]   * WEIGHT_ML,
        2
    )

    close = df["Close"].squeeze()

    if composite >= MIN_SCORE_BUY:
        signal = "BUY"
    elif composite <= 35:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Filtro 1: mercado bajista — SPY bajo SMA50
    if signal == "BUY" and SPY_SMA_FILTER and not spy_above_sma:
        signal = "HOLD"
        composite = min(composite, 55)

    # Filtro 2: confirmacion de velas consecutivas alcistas
    if signal == "BUY" and not _consecutive_bullish(close, CONSEC_CANDLES):
        signal = "HOLD"
        composite = min(composite, 60)

    # Filtro 3: bloquear en miedo extremo a menos que RSI sea oversold (oportunidad contrarian)
    # spy_rsi > 30: mercado en miedo extremo pero NO oversold → bloquear entrada
    if sentiment["fear_greed_score"] <= 20 and sentiment["spy_rsi"] > 30:
        signal = "HOLD"
        composite = min(composite, 50)

    close = df["Close"].squeeze()
    current_price = float(close.iloc[-1])

    adx_value = tech["components"].get("adx", 0)

    return {
        "ticker":        ticker,
        "score":         composite,
        "signal":        signal,
        "current_price": round(current_price, 4),
        "atr":           tech["atr"],
        "adx":           round(adx_value, 1),
        "spy_above_sma": spy_above_sma,
        "technical": tech,
        "ml":        ml,
        "sentiment": {
            "fear_greed":        round(sentiment["fear_greed_score"], 1),
            "fear_greed_label":  sentiment["fear_greed_label"],
            "spy_rsi":           sentiment["spy_rsi"],
            "spy_rsi_signal":    sentiment["spy_rsi_signal"],
            "combined":          round(sentiment["combined_sentiment"] * 100, 1),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Long-Term Scoring ────────────────────────────────────────────────────────

def score_longterm(ticker: str, df: pd.DataFrame, spy_6m_return: float = 0.0) -> dict:
    """
    Long-term score (0–100) for a 6–24 month investment horizon.

    Components:
      - Momentum 12 meses  (25%)
      - Momentum  6 meses  (15%)
      - Tendencia SMA 50/100/200 (20%)
      - Fuerza relativa vs SPY 6m (15%)
      - Dividend yield     (15%)
      - Tendencia volumen 50d/200d (10%)
    """
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    # 1. Momentum 12 meses
    if len(close) >= 252:
        mom_12m = (float(close.iloc[-1]) / float(close.iloc[-252]) - 1) * 100
    else:
        mom_12m = 0.0
    # -20% → 0 pts, 0% → 40 pts, +30% → 100 pts
    mom_12m_score = float(np.clip((mom_12m + 20) / 50 * 100, 0, 100))

    # 2. Momentum 6 meses
    if len(close) >= 126:
        mom_6m = (float(close.iloc[-1]) / float(close.iloc[-126]) - 1) * 100
    else:
        mom_6m = 0.0
    mom_6m_score = float(np.clip((mom_6m + 10) / 30 * 100, 0, 100))

    # 3. Tendencia SMA largo plazo
    last   = float(close.iloc[-1])
    sma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else last
    sma100 = float(close.rolling(100).mean().iloc[-1]) if len(close) >= 100 else last
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else last

    sma_score = 0.0
    if last > sma50:   sma_score += 30.0
    if last > sma100:  sma_score += 30.0
    if last > sma200:  sma_score += 40.0

    # 4. Fuerza relativa vs SPY (outperformance en 6 meses)
    rs_diff = mom_6m - spy_6m_return * 100           # % difference vs SPY
    rs_score = float(np.clip(50 + rs_diff * 2, 0, 100))

    # 5. Dividend yield
    try:
        info      = yf.Ticker(ticker).info
        div_yield = float(info.get("dividendYield") or 0)
        # 0% → 0, 2% → 50, 4%+ → 100
        # yfinance a veces devuelve el yield como decimal (0.035) o como pct (3.5)
        if div_yield > 1:
            div_yield = div_yield / 100
        div_score     = float(np.clip(div_yield / 0.04 * 100, 0, 100))
        div_yield_pct = round(div_yield * 100, 2)
    except Exception:
        div_score     = 50.0
        div_yield_pct = 0.0

    # 6. Tendencia de volumen (50d vs 200d)
    vol_50  = float(volume.rolling(50).mean().iloc[-1])  if len(volume) >= 50  else float(volume.mean())
    vol_200 = float(volume.rolling(200).mean().iloc[-1]) if len(volume) >= 200 else float(volume.mean())
    if vol_200 > 0:
        vol_ratio = vol_50 / vol_200
        vol_score = float(np.clip((vol_ratio - 0.5) / 1.5 * 100, 0, 100))
    else:
        vol_score = 50.0

    composite = round(
        mom_12m_score * 0.25 +
        mom_6m_score  * 0.15 +
        sma_score     * 0.20 +
        rs_score      * 0.15 +
        div_score     * 0.15 +
        vol_score     * 0.10,
        2,
    )

    if composite >= MIN_SCORE_LONGTERM_BUY:
        signal = "STRONG BUY"
    elif composite >= 50:
        signal = "ACCUMULATE"
    else:
        signal = "WAIT"

    return {
        "ticker":    ticker,
        "score":     composite,
        "signal":    signal,
        "horizon":   "largo plazo (6–24 meses)",
        "components": {
            "momentum_12m_pct":   round(mom_12m, 2),
            "momentum_12m_score": round(mom_12m_score, 2),
            "momentum_6m_pct":    round(mom_6m, 2),
            "momentum_6m_score":  round(mom_6m_score, 2),
            "sma_score":          round(sma_score, 2),
            "rs_vs_spy_score":    round(rs_score, 2),
            "dividend_yield_pct": div_yield_pct,
            "dividend_score":     round(div_score, 2),
            "volume_trend_score": round(vol_score, 2),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


def score_all_longterm() -> list[dict]:
    """
    Descarga 2 anos de datos y calcula el score de largo plazo
    para todos los ETFs en LONGTERM_UNIVERSE.
    Devuelve lista ordenada por score descendente.
    """
    end   = datetime.today()
    start = end - timedelta(days=504)   # ~2 anos

    # SPY como benchmark para fuerza relativa
    spy_6m_return = 0.0
    try:
        spy_df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if len(spy_df) >= 126:
            spy_close = spy_df["Close"].squeeze()
            spy_6m_return = float(spy_close.iloc[-1]) / float(spy_close.iloc[-126]) - 1
    except Exception as e:
        logger.warning(f"No se pudo obtener SPY para benchmark: {e}")

    results = []
    for ticker in LONGTERM_UNIVERSE:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty or len(df) < 100:
                logger.warning(f"[{ticker}] Datos insuficientes para largo plazo, omitiendo.")
                continue
            result = score_longterm(ticker, df, spy_6m_return=spy_6m_return)
            results.append(result)
            logger.info(
                f"[{ticker}] Score LP: {result['score']:.1f} | Senal: {result['signal']} | "
                f"Mom12m: {result['components']['momentum_12m_pct']:.1f}% | "
                f"Div: {result['components']['dividend_yield_pct']:.2f}%"
            )
        except Exception as e:
            logger.error(f"[{ticker}] Score largo plazo fallido: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def score_all_stocks() -> list[dict]:
    """
    Analisis de largo plazo para acciones individuales en STOCKS_UNIVERSE.
    Misma logica que score_all_longterm() pero sobre stocks.
    """
    end   = datetime.today()
    start = end - timedelta(days=504)

    spy_6m_return = 0.0
    try:
        spy_df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if len(spy_df) >= 126:
            spy_close = spy_df["Close"].squeeze()
            spy_6m_return = float(spy_close.iloc[-1]) / float(spy_close.iloc[-126]) - 1
    except Exception as e:
        logger.warning(f"SPY benchmark error: {e}")

    results = []
    for ticker in STOCKS_UNIVERSE:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty or len(df) < 100:
                logger.warning(f"[{ticker}] Datos insuficientes, omitiendo.")
                continue
            result = score_longterm(ticker, df, spy_6m_return=spy_6m_return)
            results.append(result)
            logger.info(
                f"[{ticker}] Score LP: {result['score']:.1f} | {result['signal']} | "
                f"Mom12m: {result['components']['momentum_12m_pct']:.1f}%"
            )
        except Exception as e:
            logger.error(f"[{ticker}] Score stocks fallido: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── Batch Scoring ─────────────────────────────────────────────────────────────

def score_all_etfs() -> list[dict]:
    """
    Downloads data, fetches sentiment, and scores every ETF in the universe.
    Returns list of result dicts, sorted by composite score descending.
    """
    logger.info("Fetching sentiment features...")
    sentiment = get_sentiment_features()
    logger.info(
        f"Fear & Greed: {sentiment['fear_greed_score']:.0f} ({sentiment['fear_greed_label']}) | "
        f"SPY RSI: {sentiment['spy_rsi']:.1f} ({sentiment['spy_rsi_signal']})"
    )

    end   = datetime.today()
    start = end - timedelta(days=HISTORY_DAYS + 50)

    # Verificar si SPY esta sobre SMA50 (filtro de mercado bajista)
    spy_ok = True
    try:
        spy_df    = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        spy_ok    = _spy_above_sma50(spy_df["Close"].squeeze())
        spy_state = "ALCISTA" if spy_ok else "BAJISTA"
        logger.info(f"SPY vs SMA50: {spy_state} — {'entradas permitidas' if spy_ok else 'entradas BLOQUEADAS'}")
        if not spy_ok:
            logger.warning("SPY bajo SMA50 — filtro activado, BUY signals convertidos a HOLD")
    except Exception as e:
        logger.warning(f"SPY SMA filter error ({e}) — allowing entries")

    results = []
    for ticker in ETF_UNIVERSE:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty or len(df) < 60:
                logger.warning(f"[{ticker}] Insufficient data, skipping.")
                continue
            result = score_etf(ticker, df, sentiment, spy_above_sma=spy_ok)
            results.append(result)
            logger.info(
                f"[{ticker}] Score: {result['score']:.1f} | "
                f"Signal: {result['signal']} | "
                f"Tech: {result['technical']['score']:.1f} | "
                f"ML: {result['ml']['score']:.1f}"
            )
        except Exception as e:
            logger.error(f"[{ticker}] Scoring failed: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
