# config.py — ETF Trader Configuration (No LLM)
import os
from dotenv import load_dotenv

load_dotenv()

# ── Broker ─────────────────────────────────────────────────────────────────
BROKER = os.getenv("BROKER", "robinhood")          # "robinhood" | "alpaca"

ROBINHOOD_USERNAME = os.getenv("ROBINHOOD_USERNAME", "")
ROBINHOOD_PASSWORD = os.getenv("ROBINHOOD_PASSWORD", "")
ROBINHOOD_MFA_CODE = os.getenv("ROBINHOOD_MFA_CODE", "")   # TOTP if 2FA enabled

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ── Telegram ────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Circuit Breaker ─────────────────────────────────────────────────────────
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "false").lower() == "true"
CIRCUIT_BREAKER_URL     = os.getenv("CIRCUIT_BREAKER_URL", "http://localhost:5030")

# ── ETF Universe ────────────────────────────────────────────────────────────
ETF_UNIVERSE = [
    "SPY", "SCHD", "IFRA", "SCHF",
    "GLD", "IWM", "XLE", "QQQ", "VTI",
    # XLV eliminado: win rate 20%, PF 0.80 en backtest (2026-04-19)
]
SPY_TICKER   = "SPY"           # used as market-sentiment proxy

# ── Long-Term Universe (6–24 month horizon) ──────────────────────────────────
LONGTERM_UNIVERSE = [
    "SPY",   # S&P 500 — referencia del mercado
    "QQQ",   # Nasdaq 100 — tecnologia
    "VTI",   # Total US Market
    "SCHD",  # Dividendos alta calidad
    "IFRA",  # Infraestructura
    "GLD",   # Oro — cobertura inflacion
    "IWM",   # Small caps US
    "SCHF",  # Mercados desarrollados internacionales
    "XLE",   # Energia
]
MIN_SCORE_LONGTERM_BUY = 65   # score minimo para STRONG BUY largo plazo

# ── Stocks Universe (largo plazo — acciones individuales) ────────────────────
STOCKS_UNIVERSE = [
    # Tecnologia
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    # Finanzas
    "JPM", "V", "MA", "BRK-B",
    # Salud
    "UNH", "JNJ", "LLY",
    # Consumo / Defensivo
    "COST", "WMT", "PG",
    # Energia / Industrial
    "XOM", "NEE", "CAT",
]

# ── Scoring Weights ──────────────────────────────────────────────────────────
WEIGHT_TECHNICAL  = 0.50       # Technical indicators
WEIGHT_ML         = 0.50       # XGBoost + RF ensemble (includes F&G + SPY RSI)

# ── Trade Rules ─────────────────────────────────────────────────────────────
MIN_SCORE_BUY      = 60        # minimo global — por simbolo usa MIN_SCORE_BY_TICKER

# Umbrales optimizados por backtest (500 dias) — maximizan PF con WR >= 50%
MIN_SCORE_BY_TICKER = {
    "SPY":  69,   # WR=75%  PF=3.59
    "SCHD": 84,   # WR=67%  PF=1.45  (filtro alto)
    "IFRA": 69,   # WR=60%  PF=2.07
    "SCHF": 60,   # WR=100% PF=4.11  (muy selectivo a score bajo)
    "GLD":  84,   # WR=78%  PF=4.55  (mejor ETF del universo)
    "IWM":  78,   # WR=75%  PF=2.78
    "XLE":  60,   # WR=64%  PF=1.36
    "QQQ":  84,   # WR=67%  PF=1.36  (filtro alto)
    "VTI":  69,   # WR=75%  PF=3.11
}
STOP_LOSS_PCT      = 0.05      # 5 % below entry (initial hard stop, Alpaca bracket)
TAKE_PROFIT_PCT    = 0.12      # 12 % above entry (base)
TAKE_PROFIT_HIGH   = 0.18      # 18 % when ADX >= ADX_HIGH_TREND (tendencia fuerte)

# ── Trailing Stop ────────────────────────────────────────────────────────────
TRAILING_ACTIVATE_PCT = 0.06   # activate trailing stop once gain reaches 6%
TRAILING_STOP_PCT     = 0.03   # trail 3% below position high once activated

# ── Live Trading Progression (cambiar manualmente al pasar a real) ────────────
# Semana 1-2 live ($500 cuenta): MAX_POSITION_PCT = 0.05  MAX_OPEN_POSITIONS = 3
# Semana 3-4 live ($500 cuenta): MAX_POSITION_PCT = 0.10  MAX_OPEN_POSITIONS = 3
# Mes 2+ live  ($500 cuenta):    MAX_POSITION_PCT = 0.20  MAX_OPEN_POSITIONS = 3
_is_live           = "paper" not in os.getenv("ALPACA_BASE_URL", "paper-api.alpaca.markets").lower()
MAX_POSITION_PCT   = 0.05 if _is_live else 0.10   # live: 5%  | paper: 10%
MAX_OPEN_POSITIONS = 3    if _is_live else 4       # live: 3   | paper: 4

# ── Scan Schedule ────────────────────────────────────────────────────────────
SCAN_INTERVAL_MIN  = 30        # minutes between scans
COOLDOWN_HOURS     = 24        # horas de cooldown entre trades del mismo símbolo
COOLDOWN_SL_HOURS  = 144       # cooldown extra after stop-loss (6 days)

# ── ATR / ADX Parameters ─────────────────────────────────────────────────────
ATR_MULTIPLIER     = 1.5       # dynamic SL = ATR * multiplier / price (min STOP_LOSS_PCT)
ATR_PERIOD         = 14        # ATR lookback
ADX_PERIOD         = 14        # ADX lookback
ADX_MIN_TREND      = 20        # ADX below this = sideways market, score penalized 25%
ADX_STRONG_TREND   = 25        # ADX above this = strong trend, score boosted 10%
ADX_HIGH_TREND     = 30        # ADX above this = take profit ampliado (TAKE_PROFIT_HIGH)

# ── Fear & Greed Position Sizing ─────────────────────────────────────────────
FEAR_GREED_REDUCE_THRESHOLD = 30   # F&G below this = mercado en miedo extremo
FEAR_GREED_REDUCE_PCT       = 0.05  # reducir posicion a 5% cuando F&G < threshold

# ── Filtros de Entrada ────────────────────────────────────────────────────────
SPY_SMA_FILTER     = True      # bloquear BUY cuando SPY < SMA50 (mercado bajista)
CONSEC_CANDLES     = 2         # velas alcistas consecutivas requeridas para confirmar

# ── Risk Filters ──────────────────────────────────────────────────────────────
VIX_MAX            = 25        # block entries when VIX >= this level
VIX_TICKER         = "^VIX"    # Yahoo Finance VIX symbol
NY_OPEN_HOUR       = 9
NY_OPEN_MIN        = 30
NY_CLOSE_HOUR      = 16
NY_CLOSE_MIN       = 0
MARKET_TIMEZONE    = "America/New_York"

# ── Auto-Trade ────────────────────────────────────────────────────────────────
AUTO_TRADE = os.getenv("AUTO_TRADE", "false").lower() == "true"
# True  → BUY/SELL se ejecutan automáticamente sin aprobación humana.
# False → comportamiento original: espera /approve_TICKER por Telegram.

# ── Flask ────────────────────────────────────────────────────────────────────
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = 5010
DEBUG      = os.getenv("DEBUG", "false").lower() == "true"

# ── Data ─────────────────────────────────────────────────────────────────────
HISTORY_DAYS     = 252          # 1 year for ML training
FEATURE_LOOKBACK = 60           # days for rolling features

# ── Fear & Greed ─────────────────────────────────────────────────────────────
FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
FEAR_GREED_TIMEOUT = 10         # seconds

# ── ML Model ─────────────────────────────────────────────────────────────────
ML_MODEL_PATH      = "ml_models/"
RETRAIN_ON_STARTUP = True
MIN_TRAIN_SAMPLES  = 100
