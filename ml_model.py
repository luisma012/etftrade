# ml_model.py — XGBoost + Random Forest ensemble (no LLM)
import os
import logging
import pickle
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb

from config import (
    ETF_UNIVERSE, ML_MODEL_PATH, HISTORY_DAYS,
    MIN_TRAIN_SAMPLES
)

logger = logging.getLogger(__name__)

os.makedirs(ML_MODEL_PATH, exist_ok=True)


# ── Feature Engineering ───────────────────────────────────────────────────────

def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    loss  = (-delta).clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _macd(s: pd.Series, fast=12, slow=26, sig=9):
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    signal   = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal


def build_features(df: pd.DataFrame, sentiment: dict | None = None) -> pd.DataFrame:
    """
    Builds feature matrix from OHLCV DataFrame.
    Optionally injects sentiment features (Fear & Greed, SPY RSI).
    """
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # Price returns
    feat["ret_1d"]  = close.pct_change(1)
    feat["ret_5d"]  = close.pct_change(5)
    feat["ret_20d"] = close.pct_change(20)

    # Volatility
    feat["vol_10d"] = feat["ret_1d"].rolling(10).std()
    feat["vol_30d"] = feat["ret_1d"].rolling(30).std()

    # Trend
    feat["sma20"]    = close.rolling(20).mean() / close - 1
    feat["sma50"]    = close.rolling(50).mean() / close - 1
    feat["sma200"]   = close.rolling(200, min_periods=50).mean() / close - 1
    feat["above_50"] = (close > close.rolling(50).mean()).astype(int)

    # Momentum
    feat["rsi14"] = _rsi(close, 14) / 100
    feat["rsi7"]  = _rsi(close, 7)  / 100

    macd_line, macd_sig = _macd(close)
    feat["macd_diff"] = (macd_line - macd_sig) / close

    # Volume ratio
    feat["vol_ratio"] = volume / volume.rolling(20).mean()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    feat["atr_pct"] = atr_14 / close

    # Bollinger Band position
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["bb_pos"] = (close - bb_mid) / (2 * bb_std + 1e-9)

    # ADX — fuerza de tendencia
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    plus_dm    = (high - prev_high).clip(lower=0)
    minus_dm   = (prev_low - low).clip(lower=0)
    mask       = plus_dm >= minus_dm
    plus_dm    = plus_dm.where(mask, 0.0)
    minus_dm   = minus_dm.where(~mask, 0.0)
    atr_s      = atr_14.replace(0, 1e-9)
    plus_di    = 100 * plus_dm.ewm(com=13, min_periods=14).mean() / atr_s
    minus_di   = 100 * minus_dm.ewm(com=13, min_periods=14).mean() / atr_s
    dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    feat["adx_norm"]    = dx.ewm(com=13, min_periods=14).mean() / 100
    feat["plus_di_norm"]  = plus_di / 100
    feat["minus_di_norm"] = minus_di / 100

    # Momentum largo plazo
    feat["ret_60d"] = close.pct_change(60)

    # Distancia al maximo de 6 meses (min_periods=30 para evitar dropna masivo)
    high_6m = close.rolling(126, min_periods=30).max()
    feat["dist_6m_high"] = (close - high_6m) / high_6m.replace(0, 1e-9)

    # Precio vs SMA200 (con min_periods para datos cortos)
    feat["above_200"] = (close > close.rolling(200, min_periods=50).mean()).astype(int)

    # Sentiment features (broadcast as constants for the latest bar)
    if sentiment:
        feat["fear_greed_norm"]   = sentiment.get("fear_greed_norm",   0.5)
        feat["spy_rsi_norm"]      = sentiment.get("spy_rsi_norm",      0.5)
        feat["combined_sentiment"] = sentiment.get("combined_sentiment", 0.5)
    else:
        feat["fear_greed_norm"]    = 0.5
        feat["spy_rsi_norm"]       = 0.5
        feat["combined_sentiment"] = 0.5

    return feat.dropna()


def build_labels(close: pd.Series, forward_days: int = 5, threshold: float = 0.01) -> pd.Series:
    """
    Label: 1 if price rises > threshold in next `forward_days`, else 0.
    """
    future_ret = close.shift(-forward_days) / close - 1
    return (future_ret > threshold).astype(int)


# ── ETFModel ──────────────────────────────────────────────────────────────────

class ETFModel:
    """XGBoost + RandomForest voting ensemble for a single ETF."""

    def __init__(self, ticker: str):
        self.ticker    = ticker
        self.model     = None
        self.scaler    = StandardScaler()
        self.feature_cols: list[str] = []
        self._model_file  = os.path.join(ML_MODEL_PATH, f"{ticker}_model.pkl")
        self._scaler_file = os.path.join(ML_MODEL_PATH, f"{ticker}_scaler.pkl")
        self._cols_file   = os.path.join(ML_MODEL_PATH, f"{ticker}_cols.pkl")
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            files = [self._model_file, self._scaler_file, self._cols_file]
            if all(os.path.exists(f) for f in files):
                with open(self._model_file,  "rb") as f: self.model        = pickle.load(f)
                with open(self._scaler_file, "rb") as f: self.scaler       = pickle.load(f)
                with open(self._cols_file,   "rb") as f: self.feature_cols = pickle.load(f)
                logger.info(f"[{self.ticker}] ML model loaded from disk.")
        except Exception as e:
            logger.warning(f"[{self.ticker}] Could not load model: {e}")

    def _save(self):
        with open(self._model_file,  "wb") as f: pickle.dump(self.model,        f)
        with open(self._scaler_file, "wb") as f: pickle.dump(self.scaler,       f)
        with open(self._cols_file,   "wb") as f: pickle.dump(self.feature_cols, f)
        logger.info(f"[{self.ticker}] Model saved.")

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, days: int = HISTORY_DAYS):
        logger.info(f"[{self.ticker}] Fetching {days} days of history...")
        end   = datetime.today()
        start = end - timedelta(days=days + 50)

        df = yf.download(self.ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < MIN_TRAIN_SAMPLES:
            logger.error(f"[{self.ticker}] Not enough data to train.")
            return False

        feat   = build_features(df)
        labels = build_labels(df["Close"].squeeze())
        labels = labels.reindex(feat.index).dropna()
        feat   = feat.loc[labels.index]

        if len(feat) < MIN_TRAIN_SAMPLES:
            logger.warning(f"[{self.ticker}] Too few labeled samples ({len(feat)}).")
            return False

        self.feature_cols = feat.columns.tolist()
        X = feat.values
        y = labels.values

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0
        )
        rf_clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        ensemble = VotingClassifier(
            estimators=[("xgb", xgb_clf), ("rf", rf_clf)],
            voting="soft"
        )
        ensemble.fit(X_train_s, y_train)

        preds = ensemble.predict(X_test_s)
        report = classification_report(y_test, preds, output_dict=True)
        acc = report.get("accuracy", 0)
        logger.info(f"[{self.ticker}] Training done — accuracy: {acc:.2%}")

        self.model = ensemble
        self._save()
        return True

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame, sentiment: dict | None = None) -> float:
        """
        Returns probability (0.0–1.0) that the ETF will rise > 1 % in the next 5 days.
        """
        if self.model is None:
            logger.warning(f"[{self.ticker}] No model trained yet — returning 0.5.")
            return 0.5

        feat = build_features(df, sentiment)
        if feat.empty:
            return 0.5

        # Align to training features
        for col in self.feature_cols:
            if col not in feat.columns:
                feat[col] = 0.0
        feat = feat[self.feature_cols].iloc[[-1]]

        try:
            X_s   = self.scaler.transform(feat.values)
            proba = float(self.model.predict_proba(X_s)[0][1])
            return round(proba, 4)
        except Exception as e:
            logger.error(f"[{self.ticker}] Prediction error: {e}")
            return 0.5


# ── Model Registry ────────────────────────────────────────────────────────────

_registry: dict[str, ETFModel] = {}


def get_model(ticker: str) -> ETFModel:
    if ticker not in _registry:
        _registry[ticker] = ETFModel(ticker)
    return _registry[ticker]


def train_all(retrain: bool = False):
    """Train (or load) models for all ETFs in the universe."""
    for ticker in ETF_UNIVERSE:
        m = get_model(ticker)
        if retrain or m.model is None:
            m.train()


def predict_all(dfs: dict[str, pd.DataFrame], sentiment: dict | None = None) -> dict[str, float]:
    """
    Returns {ticker: proba} for each ETF.
    `dfs` is {ticker: OHLCV DataFrame}.
    """
    results = {}
    for ticker, df in dfs.items():
        results[ticker] = get_model(ticker).predict_proba(df, sentiment)
    return results
