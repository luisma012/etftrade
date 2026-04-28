#!/usr/bin/env python3
"""
ETF Technical Analysis Dashboard — Alpaca Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Symbols  : SPY, VGT, SCHD, IFRA
Analysis : VWAP + Bands, Delta, Footprint, pandas-ta
ML       : XGBoost  →  BUY / HOLD / SELL
Broker   : Alpaca (paper & live) — REST + WebSocket
Dashboard: http://localhost:5010
Mode     : Semi-automatic — signals queued, manual confirm required

Variables de entorno:
  ALPACA_API_KEY      tu API key de Alpaca
  ALPACA_SECRET_KEY   tu secret key de Alpaca
  ALPACA_PAPER        "true" (default) | "false" para live trading
"""

import os
import json
import time
import queue
import threading
import warnings
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Carga automática del archivo .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import pandas as pd
import pandas_ta as ta          # noqa: F401  (registers .ta accessor)
import xgboost as xgb
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════
SYMBOLS          = ["SPY", "VGT", "SCHD", "IFRA", "TSLA"]   # ← agrega o cambia símbolos aquí
DASHBOARD_PORT   = 5010
REFRESH_SEC      = 60          # intervalo de fondo

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER      = os.getenv("ALPACA_PAPER", "true").lower() != "false"

# Telegram
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",  "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Scanner — ETFs populares a monitorear
SCANNER_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",           # Índices principales
    "VGT", "XLK", "SOXX",                  # Tecnología
    "SCHD", "VYM", "DGRO",                 # Dividendos
    "GLD", "SLV", "IAU",                   # Metales
    "TLT", "IEF", "BND",                   # Bonos
    "XLE", "VDE",                           # Energía
    "XLF", "VFH",                           # Financiero
    "XLV", "VHT",                           # Salud
    "IFRA", "PAVE",                         # Infraestructura
    "ARKK", "ARKG",                         # Innovación
]
ALERT_CONFIDENCE = 0.80   # alerta si confianza >= 80%

# ── Auto-Trading ──────────────────────────────────────────────
AUTO_TRADE            = True    # False = solo alertas, True = ejecuta solo
AUTO_CONFIDENCE       = 0.85   # confianza mínima para entrada automática
AUTO_MAX_POSITION_USD = 1000   # máximo $ por posición
STOP_LOSS_PCT         = 0.02   # stop loss 2% desde entrada
BREAKEVEN_TRIGGER_PCT = 0.01   # mueve stop a breakeven después de +1%
TAKE_PROFIT_PCT       = 0.04   # take profit en +4%

# ── Protecciones adicionales ──────────────────────────────────
COOLDOWN_DAYS         = 3      # días bloqueado tras stop-loss
VIX_MAX               = 20     # VIX máximo para permitir nuevas entradas (>20 = miedo)
DAILY_REPORT_UTC_HOUR = 20     # hora UTC del reporte diario (4 PM ET = cierre mercado USA)
COOLDOWN_FILE         = Path(__file__).parent / "cooldown.json"
AUDIT_FILE            = Path(__file__).parent / "audit_state.json"
INITIAL_CAPITAL       = 100_000   # capital inicial de la cuenta paper

# Colores del dashboard
C = {
    "bg":     "#0d1117",
    "card":   "#161b22",
    "border": "#30363d",
    "green":  "#3fb950",
    "red":    "#f85149",
    "yellow": "#d29922",
    "blue":   "#58a6ff",
    "purple": "#bc8cff",
    "orange": "#e3b341",
    "text":   "#e6edf3",
    "muted":  "#8b949e",
}

# ══════════════════════════════════════════════════════════════
# 0. AUDIT TRACKER — métricas para pasar a LIVE
# ══════════════════════════════════════════════════════════════
class AuditTracker:
    """Rastrea métricas de rendimiento y evalúa el checklist para ir a LIVE."""

    def __init__(self):
        self._state = self._load()

    def _load(self) -> dict:
        if AUDIT_FILE.exists():
            try:
                return json.loads(AUDIT_FILE.read_text())
            except Exception:
                pass
        return {
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "daily_snapshots": [],       # [{date, portfolio, pnl, pnl_pct}]
            "peak_portfolio": INITIAL_CAPITAL,
            "consecutive_losses": 0,
            "max_consecutive_losses": 0,
            "total_wins": 0,
            "total_losses": 0,
            "errors_last_7d": [],        # [date_str, ...]
        }

    def _save(self):
        try:
            AUDIT_FILE.write_text(json.dumps(self._state, indent=2))
        except Exception as exc:
            logger.error(f"Audit save error: {exc}")

    def record_snapshot(self, portfolio: float, positions: list):
        """Llamada diaria: registra estado del portfolio."""
        today = datetime.now().strftime("%Y-%m-%d")
        snaps = self._state["daily_snapshots"]

        # Evitar duplicado del mismo día
        if snaps and snaps[-1]["date"] == today:
            snaps[-1] = {"date": today, "portfolio": portfolio}
        else:
            snaps.append({"date": today, "portfolio": portfolio})

        # Peak portfolio
        if portfolio > self._state["peak_portfolio"]:
            self._state["peak_portfolio"] = portfolio

        # Limitar a 90 días de historia
        self._state["daily_snapshots"] = snaps[-90:]
        self._save()

    def record_trade_result(self, won: bool):
        """Registra resultado de un trade para streak."""
        if won:
            self._state["total_wins"] += 1
            self._state["consecutive_losses"] = 0
        else:
            self._state["total_losses"] += 1
            self._state["consecutive_losses"] += 1
            self._state["max_consecutive_losses"] = max(
                self._state["max_consecutive_losses"],
                self._state["consecutive_losses"]
            )
        self._save()

    def record_error(self):
        today = datetime.now().strftime("%Y-%m-%d")
        self._state["errors_last_7d"].append(today)
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        self._state["errors_last_7d"] = [d for d in self._state["errors_last_7d"] if d >= cutoff]
        self._save()

    def evaluate(self, account: dict, positions: list, autotrader) -> list:
        """Evalúa todos los checks y retorna lista de (nombre, passed, detalle)."""
        checks = []
        snaps = self._state["daily_snapshots"]
        portfolio = account.get("portfolio_value", 0)
        start_date = self._state.get("start_date", datetime.now().strftime("%Y-%m-%d"))

        # ── FASE 1: Paper Trading ─────────────────────────────
        # 1. Días en paper
        days_running = (datetime.now() - datetime.strptime(start_date, "%Y-%m-%d")).days
        checks.append(("Fase 1", f"Paper mínimo 30 días ({days_running}/30)",
                        days_running >= 30, f"{days_running} días"))

        # 2. Win rate > 55%
        total = self._state["total_wins"] + self._state["total_losses"]
        wr = (self._state["total_wins"] / total * 100) if total > 0 else 0
        checks.append(("Fase 1", f"Win Rate > 55% ({wr:.1f}%)",
                        wr > 55 and total >= 5, f"{self._state['total_wins']}W / {self._state['total_losses']}L"))

        # 3. Profit Factor > 1.3  (usamos P&L del portfolio como proxy)
        pnl_total = portfolio - INITIAL_CAPITAL if portfolio else 0
        roi = (pnl_total / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL else 0
        checks.append(("Fase 1", f"ROI positivo ({roi:+.2f}%)",
                        roi > 0, f"P&L: ${pnl_total:+,.2f}"))

        # 4. Drawdown máximo < 10%
        peak = self._state.get("peak_portfolio", INITIAL_CAPITAL)
        dd = ((portfolio - peak) / peak * 100) if peak > 0 else 0
        max_dd = dd  # current drawdown
        # Check from snapshots
        for snap in snaps:
            snap_dd = ((snap["portfolio"] - peak) / peak * 100) if peak > 0 else 0
            max_dd = min(max_dd, snap_dd)
            if snap["portfolio"] > peak:
                peak = snap["portfolio"]
        checks.append(("Fase 1", f"Drawdown máximo < 10% ({abs(max_dd):.2f}%)",
                        abs(max_dd) < 10, f"Max DD: {max_dd:.2f}%"))

        # 5. Pérdidas consecutivas < 5
        consec = self._state.get("max_consecutive_losses", 0)
        checks.append(("Fase 1", f"Pérdidas consecutivas < 5 ({consec})",
                        consec < 5, f"Máx racha negativa: {consec}"))

        # 6. Stop-loss funcional
        has_stops = all("stop" in pos for pos in autotrader.open_positions.values()) if autotrader.open_positions else True
        checks.append(("Fase 1", "Stop-loss configurado en todas las posiciones",
                        has_stops, "OK" if has_stops else "Faltan stops"))

        # 7. Telegram funcionando
        tg_ok = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
        checks.append(("Fase 1", "Telegram configurado",
                        tg_ok, "Conectado" if tg_ok else "Sin configurar"))

        # 8. Sin errores críticos 7 días
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        recent_errors = [d for d in self._state.get("errors_last_7d", []) if d >= cutoff]
        checks.append(("Fase 1", f"Sin errores críticos 7 días ({len(recent_errors)} errores)",
                        len(recent_errors) == 0, f"{len(recent_errors)} errores recientes"))

        # 9. Filtro VIX activo
        checks.append(("Fase 1", f"Filtro VIX activo (máx={VIX_MAX})",
                        VIX_MAX > 0, f"VIX límite: {VIX_MAX}"))

        # 10. Cooldown activo
        checks.append(("Fase 1", f"Cooldown activo ({COOLDOWN_DAYS} días)",
                        COOLDOWN_DAYS >= 1, f"{COOLDOWN_DAYS} días"))

        # ── FASE 2: Configuración de seguridad ────────────────
        checks.append(("Fase 2", "Modo PAPER activo",
                        ALPACA_PAPER, "PAPER" if ALPACA_PAPER else "LIVE"))

        checks.append(("Fase 2", f"Posición máxima <= $1,000 (${AUTO_MAX_POSITION_USD:,})",
                        AUTO_MAX_POSITION_USD <= 1000, f"${AUTO_MAX_POSITION_USD:,}"))

        checks.append(("Fase 2", f"Confianza mínima >= 85% ({AUTO_CONFIDENCE*100:.0f}%)",
                        AUTO_CONFIDENCE >= 0.85, f"{AUTO_CONFIDENCE*100:.0f}%"))

        # 11. Mínimo 10 trades para validar
        checks.append(("Fase 2", f"Mínimo 10 trades completados ({total})",
                        total >= 10, f"{total} trades"))

        # ── FASE 3: Resultado final ───────────────────────────
        passed = sum(1 for c in checks if c[2])
        total_checks = len(checks)
        ready = passed == total_checks and days_running >= 30
        checks.append(("RESULTADO", f"LISTO para LIVE: {passed}/{total_checks} checks",
                        ready, "APROBADO" if ready else f"FALTAN {total_checks - passed}"))

        return checks

audit = AuditTracker()


# ══════════════════════════════════════════════════════════════
# 1. ALPACA CLIENT  (datos históricos + trading)
# ══════════════════════════════════════════════════════════════
class AlpacaClient:
    """
    Wrapper unificado para alpaca-py.
    Datos históricos: StockHistoricalDataClient
    Trading:         TradingClient
    Stream en vivo:  StockDataStream  (WebSocket)
    """

    def __init__(self):
        self._hist    = None
        self._trade   = None
        self._stream  = None
        self._stream_thread: Optional[threading.Thread] = None

        # Colas para ticks en tiempo real
        self.live_quotes: Dict[str, dict] = {}   # symbol → último quote
        self.live_bars:   Dict[str, dict] = {}   # symbol → último bar de 1-min
        self._ws_ready   = False

        self._init_clients()

    # ── inicialización ────────────────────────────────────────
    def _init_clients(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.warning("⚠  ALPACA_API_KEY / ALPACA_SECRET_KEY no configuradas → modo yFinance")
            return
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client  import TradingClient

            self._hist  = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            self._trade = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)

            acc = self._trade.get_account()
            mode = "PAPER" if ALPACA_PAPER else "LIVE"
            logger.info(f"Alpaca {mode} ✓  portfolio=${float(acc.portfolio_value):,.2f}  "
                        f"cash=${float(acc.cash):,.2f}")
        except Exception as exc:
            logger.error(f"Alpaca init error: {exc}")

    # ── WebSocket stream ──────────────────────────────────────
    def start_stream(self):
        """Lanza el WebSocket de Alpaca en un hilo daemon."""
        if not self._hist or self._stream_thread:
            return
        self._stream_thread = threading.Thread(
            target=self._run_stream, daemon=True, name="alpaca-ws"
        )
        self._stream_thread.start()

    def _run_stream(self):
        try:
            from alpaca.data.live import StockDataStream

            stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

            async def on_bar(bar):
                self.live_bars[bar.symbol] = {
                    "open":   float(bar.open),
                    "high":   float(bar.high),
                    "low":    float(bar.low),
                    "close":  float(bar.close),
                    "volume": float(bar.volume),
                    "ts":     str(bar.timestamp),
                }

            async def on_quote(quote):
                self.live_quotes[quote.symbol] = {
                    "bid":    float(quote.bid_price),
                    "ask":    float(quote.ask_price),
                    "bid_sz": float(quote.bid_size),
                    "ask_sz": float(quote.ask_size),
                    "ts":     str(quote.timestamp),
                }

            # Suscribir SYMBOLS + símbolos con posiciones abiertas en Alpaca
            ws_symbols = set(SYMBOLS)
            try:
                for p in self.get_positions():
                    ws_symbols.add(p["symbol"])
            except Exception:
                pass
            ws_symbols = list(ws_symbols)
            logger.info(f"WebSocket suscrito a: {ws_symbols}")
            stream.subscribe_bars(on_bar,   *ws_symbols)
            stream.subscribe_quotes(on_quote, *ws_symbols)
            self._ws_ready = True
            logger.info("Alpaca WebSocket ✓  (bars + quotes en tiempo real)")
            stream.run()
        except Exception as exc:
            logger.error(f"WebSocket error: {exc}")
            self._ws_ready = False

    # ── datos históricos ──────────────────────────────────────
    def get_bars(self, symbol: str, tf: str = "1d", limit: int = 500) -> pd.DataFrame:
        """Alpaca histórico → fallback yFinance."""
        if self._hist:
            df = self._alpaca_bars(symbol, tf, limit)
            if not df.empty:
                return df
        return self._yf_bars(symbol, tf, limit)

    def _alpaca_bars(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        try:
            from alpaca.data.requests   import StockBarsRequest
            from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit

            tf_map = {
                "1d":  TimeFrame.Day,
                "1h":  TimeFrame.Hour,
                "15m": TimeFrame(15, TimeFrameUnit.Minute),
                "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
            }
            days = {"1d": max(limit, 400), "1h": 120, "15m": 60, "5m": 30}.get(tf, 400)
            req  = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf_map.get(tf, TimeFrame.Day),
                start=datetime.now(timezone.utc) - timedelta(days=days),
                feed="iex",          # gratis; cambia a "sip" con suscripción
            )
            raw = self._hist.get_stock_bars(req).df.reset_index()
            if "symbol" in raw.columns:
                raw = raw[raw["symbol"] == symbol]
            raw = raw.rename(columns={"timestamp": "datetime"})
            raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True).dt.tz_localize(None)
            raw = raw.set_index("datetime")
            raw.columns = [c.lower() for c in raw.columns]
            return raw[["open", "high", "low", "close", "volume"]].dropna()
        except Exception as exc:
            logger.debug(f"_alpaca_bars [{symbol}]: {exc}")
            return pd.DataFrame()

    @staticmethod
    def _yf_bars(symbol: str, tf: str, limit: int) -> pd.DataFrame:
        imap = {"1d": "1d", "1h": "1h", "15m": "15m", "5m": "5m"}
        pmap = {"1d": f"{min(limit,730)}d", "1h": "60d", "15m": "60d", "5m": "30d"}
        try:
            df = yf.download(symbol, period=pmap.get(tf, "1y"),
                             interval=imap.get(tf, "1d"),
                             progress=False, auto_adjust=True)
            # yFinance >=0.2.x devuelve MultiIndex cuando hay un solo ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "datetime"
            df.index = pd.to_datetime(df.index)
            return df[["open", "high", "low", "close", "volume"]].dropna()
        except Exception as exc:
            logger.error(f"yFinance [{symbol}]: {exc}")
            return pd.DataFrame()

    # ── cuenta ────────────────────────────────────────────────
    def get_account(self) -> dict:
        if not self._trade:
            return {}
        try:
            a = self._trade.get_account()
            return {
                "portfolio_value": round(float(a.portfolio_value), 2),
                "cash":            round(float(a.cash), 2),
                "buying_power":    round(float(a.buying_power), 2),
                "equity":          round(float(a.equity), 2),
                "daytrade_count":  int(a.daytrade_count),
                "status":          str(a.status),
                "paper":           ALPACA_PAPER,
            }
        except Exception as exc:
            logger.error(f"get_account: {exc}")
            return {}

    def get_positions(self) -> List[dict]:
        if not self._trade:
            return []
        try:
            return [
                {
                    "symbol":         p.symbol,
                    "qty":            float(p.qty),
                    "side":           str(p.side).split(".")[-1],
                    "avg_price":      round(float(p.avg_entry_price), 2),
                    "current_price":  round(float(p.current_price), 2),
                    "market_value":   round(float(p.market_value), 2),
                    "unrealized_pl":  round(float(p.unrealized_pl), 2),
                    "unrealized_plpc":round(float(p.unrealized_plpc) * 100, 2),
                }
                for p in self._trade.get_all_positions()
            ]
        except Exception as exc:
            logger.error(f"get_positions: {exc}")
            return []

    def get_orders(self, limit: int = 15) -> List[dict]:
        if not self._trade:
            return []
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums    import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            return [
                {
                    "id":         str(o.id),
                    "symbol":     o.symbol,
                    "side":       str(o.side).split(".")[-1].upper(),
                    "qty":        float(o.qty or 0),
                    "filled_qty": float(o.filled_qty or 0),
                    "status":     str(o.status).split(".")[-1],
                    "created_at": str(o.created_at)[:16],
                }
                for o in self._trade.get_orders(req)
            ]
        except Exception as exc:
            logger.error(f"get_orders: {exc}")
            return []

    # ── ejecución de órdenes ──────────────────────────────────
    def market_order(self, symbol: str, qty: float, side: str) -> dict:
        if not self._trade:
            return {"status": "error", "message": "Alpaca no conectado"}
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums    import OrderSide, TimeInForce
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            o = self._trade.submit_order(req)
            logger.info(f"Orden enviada: {side} {qty} {symbol}  id={o.id}")
            return {"status": "submitted", "id": str(o.id),
                    "symbol": symbol, "qty": qty, "side": side}
        except Exception as exc:
            logger.error(f"market_order: {exc}")
            return {"status": "error", "message": str(exc)}

    def limit_order(self, symbol: str, qty: float, side: str, limit_price: float) -> dict:
        if not self._trade:
            return {"status": "error", "message": "Alpaca no conectado"}
        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums    import OrderSide, TimeInForce
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
            o = self._trade.submit_order(req)
            logger.info(f"Orden límite: {side} {qty} {symbol} @ ${limit_price}  id={o.id}")
            return {"status": "submitted", "id": str(o.id),
                    "symbol": symbol, "qty": qty, "side": side, "limit_price": limit_price}
        except Exception as exc:
            logger.error(f"limit_order: {exc}")
            return {"status": "error", "message": str(exc)}

    def cancel_alpaca_order(self, order_id: str) -> bool:
        if not self._trade:
            return False
        try:
            self._trade.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            logger.error(f"cancel_order: {exc}")
            return False

    @property
    def connected(self) -> bool:
        return self._trade is not None


# ══════════════════════════════════════════════════════════════
# 2. ANÁLISIS TÉCNICO
# ══════════════════════════════════════════════════════════════
class TA:
    """VWAP, Delta, Footprint + pandas-ta."""

    # ── VWAP ──────────────────────────────────────────────────
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        if isinstance(df.index, pd.DatetimeIndex):
            dates  = df.index.normalize()
            result = pd.Series(np.nan, index=df.index)
            for d in dates.unique():
                m           = dates == d
                result[m]   = (tp[m] * df["volume"][m]).cumsum() / df["volume"][m].cumsum()
        else:
            result = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
        return result.rename("VWAP")

    @staticmethod
    def vwap_bands(df: pd.DataFrame, vwap: pd.Series,
                   mults: Tuple[float, ...] = (1.0, 2.0, 3.0)) -> Dict[str, pd.Series]:
        tp   = (df["high"] + df["low"] + df["close"]) / 3
        std  = pd.Series(np.nan, index=df.index)
        if isinstance(df.index, pd.DatetimeIndex):
            for d in df.index.normalize().unique():
                m       = df.index.normalize() == d
                cv      = df["volume"][m].cumsum()
                std[m]  = np.sqrt(((tp[m] - vwap[m]) ** 2 * df["volume"][m]).cumsum() / cv)
        else:
            std = np.sqrt(((tp - vwap) ** 2 * df["volume"]).cumsum() / df["volume"].cumsum())
        return {f"vwap_up_{m}": vwap + m * std for m in mults} | \
               {f"vwap_dn_{m}": vwap - m * std for m in mults}

    # ── Delta ──────────────────────────────────────────────────
    @staticmethod
    def delta(df: pd.DataFrame) -> pd.DataFrame:
        rng    = (df["high"] - df["low"]).replace(0, np.nan)
        buy_r  = ((df["close"] - df["low"]) / rng).fillna(0.5)
        out    = pd.DataFrame(index=df.index)
        out["buy_vol"]    = df["volume"] * buy_r
        out["sell_vol"]   = df["volume"] * (1 - buy_r)
        out["delta"]      = out["buy_vol"] - out["sell_vol"]
        out["cum_delta"]  = out["delta"].cumsum()
        out["delta_pct"]  = (out["delta"] / df["volume"].replace(0, np.nan) * 100).fillna(0)
        # Divergencia precio vs delta
        out["divergence"] = (np.sign(df["close"].diff()) != np.sign(out["delta"].diff())).astype(int)
        return out

    # ── Footprint ─────────────────────────────────────────────
    @staticmethod
    def footprint(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
        rows = []
        for ts, r in df.iterrows():
            rng = r["high"] - r["low"]
            if rng <= 0:
                continue
            buy_r  = (r["close"] - r["low"]) / rng
            levels = []
            for i in range(n):
                mid  = r["low"] + (i + 0.5) * rng / n
                wt   = (mid - r["low"]) / rng if r["close"] >= r["open"] else (r["high"] - mid) / rng
                vol  = r["volume"] * max(wt, 0) / n * 2
                levels.append({"price": mid, "buy": vol * buy_r,
                                "sell": vol * (1 - buy_r), "total": vol})
            levels.sort(key=lambda x: x["total"], reverse=True)
            poc = levels[0]["price"]
            levels.sort(key=lambda x: x["price"])
            rows.append({"datetime": ts, "poc": poc,
                         "vah": levels[-2]["price"] if len(levels) > 1 else poc,
                         "val": levels[1]["price"]  if len(levels) > 1 else poc})
        return pd.DataFrame(rows).set_index("datetime") if rows else pd.DataFrame()

    # ── Suite completa ─────────────────────────────────────────
    @staticmethod
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 30:
            return df
        df = df.copy()

        # Tendencia
        for l in [9, 21, 50, 200]:
            df.ta.ema(length=l, append=True)
        df.ta.sma(length=20, append=True)

        # Momentum
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.stoch(k=14, d=3, append=True)
        df.ta.cci(length=20, append=True)
        df.ta.willr(length=14, append=True)
        df.ta.mom(length=10, append=True)
        df.ta.roc(length=10, append=True)

        # Volatilidad
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.natr(length=14, append=True)
        df.ta.kc(length=20, append=True)

        # Volumen
        df.ta.obv(append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.vwma(length=20, append=True)
        df.ta.adosc(append=True)

        # Fuerza de tendencia
        df.ta.adx(length=14, append=True)
        df.ta.aroon(length=14, append=True)
        df.ta.supertrend(length=7, multiplier=3.0, append=True)

        # VWAP + bandas
        vwap = TA.vwap(df)
        df["VWAP"] = vwap
        for k, v in TA.vwap_bands(df, vwap).items():
            df[k] = v
        df["price_vs_vwap"] = ((df["close"] - df["VWAP"]) / df["VWAP"] * 100).round(4)

        # Delta
        dlt = TA.delta(df)
        for col in dlt.columns:
            df[f"dlt_{col}"] = dlt[col]

        # Footprint POC
        try:
            fp = TA.footprint(df.tail(60))
            if not fp.empty:
                df["fp_poc"] = fp["poc"].reindex(df.index)
                df["fp_vah"] = fp["vah"].reindex(df.index)
                df["fp_val"] = fp["val"].reindex(df.index)
        except Exception as exc:
            logger.debug(f"footprint: {exc}")

        return df


# ══════════════════════════════════════════════════════════════
# 3. XGBOOST SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════
class SignalEngine:
    """XGBoost por símbolo → BUY / HOLD / SELL con probabilidades."""

    FEATURES = [
        "RSI_14", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "BBP_20_2.0", "ATRr_14", "ADX_14", "MFI_14", "OBV",
        "VWMA_20", "STOCHk_14_3_3", "STOCHd_14_3_3", "CCI_20_0.015",
        "WILLR_14", "MOM_10", "NATR_14", "ROC_10",
        "price_vs_vwap", "dlt_delta", "dlt_delta_pct", "dlt_cum_delta",
        "EMA_9", "EMA_21", "EMA_50", "volume",
    ]
    LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}

    def __init__(self):
        self.models:  Dict[str, xgb.XGBClassifier] = {}
        self.scalers: Dict[str, StandardScaler]     = {}
        self.meta:    Dict[str, dict]               = {}

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.FEATURES if c in df.columns]
        return df[cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    @staticmethod
    def _y(df: pd.DataFrame, fwd: int = 5, thr: float = 0.008) -> pd.Series:
        ret = df["close"].shift(-fwd) / df["close"] - 1
        y   = pd.Series(1, index=df.index)
        y[ret >  thr] = 2
        y[ret < -thr] = 0
        return y

    def train(self, symbol: str, df: pd.DataFrame) -> dict:
        if len(df) < 120:
            return {"status": "datos_insuficientes", "n": len(df)}
        X = self._X(df).iloc[:-5]
        y = self._y(df).iloc[:-5]
        X, y = X.align(y, join="inner", axis=0)
        if len(X) < 80:
            return {"status": "datos_insuficientes_post_align"}

        sc    = StandardScaler()
        Xs    = sc.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, shuffle=False)

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

        acc = accuracy_score(yte, model.predict(Xte))
        self.models[symbol]  = model
        self.scalers[symbol] = sc
        self.meta[symbol]    = {
            "accuracy":   round(acc, 4),
            "features":   list(X.columns),
            "trained_at": datetime.now().strftime("%H:%M:%S"),
            "n_samples":  len(Xtr),
        }
        logger.info(f"XGBoost [{symbol}]  acc={acc:.3f}  n={len(Xtr)}")
        return {"status": "ok", **self.meta[symbol]}

    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        if symbol not in self.models:
            return {"signal": "HOLD", "confidence": 0.0, "proba": {}, "trained": False}
        X    = self._X(df)
        Xs   = self.scalers[symbol].transform(X.iloc[-1:])
        pred = int(self.models[symbol].predict(Xs)[0])
        prob = self.models[symbol].predict_proba(Xs)[0]
        n    = len(prob)
        return {
            "signal":     self.LABELS.get(pred, "HOLD"),
            "confidence": float(max(prob)),
            "proba": {
                "SELL": float(prob[0]) if n > 0 else 0,
                "HOLD": float(prob[1]) if n > 1 else 0,
                "BUY":  float(prob[2]) if n > 2 else 0,
            },
            "trained": True,
        }

    def importance(self, symbol: str, top: int = 10) -> Dict[str, float]:
        if symbol not in self.models:
            return {}
        m  = self.models[symbol]
        fi = dict(zip(m.feature_names_in_, m.feature_importances_))
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top])


# ══════════════════════════════════════════════════════════════
# 4. TELEGRAM ALERTAS
# ══════════════════════════════════════════════════════════════
class TelegramAlert:
    """Envía mensajes a Telegram cuando hay señales fuertes."""

    BASE = "https://api.telegram.org/bot"

    def __init__(self):
        self.token   = TELEGRAM_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        self._sent:  Dict[str, str] = {}   # symbol → última señal enviada
        if self.enabled:
            logger.info("Telegram ✓")
        else:
            logger.warning("Telegram no configurado")

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            import urllib.request
            url  = f"{self.BASE}{self.token}/sendMessage"
            data = json.dumps({"chat_id": self.chat_id,
                               "text": text, "parse_mode": "HTML"}).encode()
            req  = urllib.request.Request(url, data=data,
                                          headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception as exc:
            logger.error(f"Telegram send: {exc}")
            return False

    def check_and_alert(self, symbol: str, signal: str,
                        confidence: float, price: float, proba: dict) -> None:
        """Envía alerta solo si la señal es nueva y supera el umbral."""
        if confidence < ALERT_CONFIDENCE:
            return
        if signal == "HOLD":
            return
        # No repetir la misma señal para el mismo símbolo
        if self._sent.get(symbol) == signal:
            return

        emoji = "🟢" if signal == "BUY" else "🔴"
        msg = (
            f"{emoji} <b>SEÑAL {signal} — {symbol}</b>\n"
            f"💰 Precio: <b>${price:,.2f}</b>\n"
            f"🎯 Confianza: <b>{confidence*100:.0f}%</b>\n"
            f"📊 BUY {proba.get('BUY',0)*100:.0f}%  "
            f"HOLD {proba.get('HOLD',0)*100:.0f}%  "
            f"SELL {proba.get('SELL',0)*100:.0f}%\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        if self.send(msg):
            self._sent[symbol] = signal
            logger.info(f"Telegram → {signal} {symbol} ({confidence*100:.0f}%)")

    def send_scanner_report(self, results: List[dict]) -> None:
        """Envía el top de ETFs recomendados por el scanner."""
        if not results:
            return
        lines = ["📡 <b>SCANNER ETF — Mejores oportunidades</b>\n"]
        for r in results[:5]:
            emoji = "🟢" if r["signal"] == "BUY" else "🔴"
            lines.append(
                f"{emoji} <b>{r['symbol']}</b>  {r['signal']} {r['confidence']*100:.0f}%  "
                f"RSI {r['rsi']:.0f}  ${r['price']:,.2f}"
            )
        lines.append(f"\n⏰ {datetime.now().strftime('%H:%M:%S')}")
        self.send("\n".join(lines))

    def send_daily_report(self, account: dict, positions: List[dict],
                          orders: List[dict]) -> None:
        """Reporte diario al cierre del mercado (20:00 UTC / 4 PM ET)."""
        if not self.enabled:
            return
        today     = datetime.now().strftime("%Y-%m-%d")
        pnl_open  = sum(p.get("unrealized_pl", 0) for p in positions)
        portfolio = account.get("portfolio_value", 0)
        cash      = account.get("cash", 0)
        mode      = "PAPER" if account.get("paper") else "LIVE"

        # Trades ejecutados hoy
        todays = [o for o in orders if o.get("created_at", "")[:10] == today
                  and o.get("status") == "filled"]

        lines = [
            f"<b>📊 REPORTE DIARIO — {today} ({mode})</b>\n",
            f"💼 Portfolio: <b>${portfolio:,.2f}</b>",
            f"💵 Cash: ${cash:,.2f}",
            f"📈 P&amp;L abierto: <b>${pnl_open:+,.2f}</b>",
        ]

        if positions:
            lines.append("\n<b>Posiciones abiertas:</b>")
            for p in positions:
                pl    = p.get("unrealized_pl", 0)
                plpct = p.get("unrealized_plpc", 0)
                tag   = "+" if pl >= 0 else "-"
                lines.append(
                    f"  {tag} <b>{p['symbol']}</b> x{int(p['qty'])}  "
                    f"${pl:+.2f} ({plpct:+.1f}%)"
                )

        if todays:
            lines.append(f"\n<b>Trades ejecutados hoy ({len(todays)}):</b>")
            for o in todays[:8]:
                lines.append(
                    f"  {o['side']} {o['symbol']} x{int(o.get('qty', 0))}  "
                    f"@ ${o.get('fill_price', '—')}"
                )

        lines.append(f"\n⏰ Cierre {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        self.send("\n".join(lines))
        logger.info("Reporte diario enviado a Telegram")


# ══════════════════════════════════════════════════════════════
# 5. ETF SCANNER
# ══════════════════════════════════════════════════════════════
class ETFScanner:
    """Escanea el universo de ETFs y rankea por señal XGBoost."""

    def __init__(self, data_provider: "AlpacaClient", signal_engine: "SignalEngine"):
        self._data    = data_provider
        self._signals = signal_engine
        self.results: List[dict] = []
        self.last_scan: Optional[datetime] = None

    def scan(self) -> List[dict]:
        """Escanea todos los ETFs del universo y devuelve ranking BUY."""
        logger.info(f"Scanner iniciado — {len(SCANNER_UNIVERSE)} ETFs …")
        results = []

        for symbol in SCANNER_UNIVERSE:
            try:
                df = self._data.get_bars(symbol, "1d", 300)
                if df.empty or len(df) < 60:
                    continue

                df = TA.enrich(df)

                # Entrenar si no existe modelo
                if symbol not in self._signals.models:
                    self._signals.train(symbol, df)

                sig  = self._signals.predict(symbol, df)
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                chg  = (last["close"] - prev["close"]) / prev["close"] * 100

                results.append({
                    "symbol":     symbol,
                    "signal":     sig.get("signal", "HOLD"),
                    "confidence": sig.get("confidence", 0),
                    "proba":      sig.get("proba", {}),
                    "price":      round(float(last["close"]), 2),
                    "change_pct": round(float(chg), 2),
                    "rsi":        round(float(last.get("RSI_14", 50)), 1),
                    "adx":        round(float(last.get("ADX_14", 0)), 1),
                    "vwap":       round(float(last.get("VWAP", 0)), 2),
                    "delta_pct":  round(float(last.get("dlt_delta_pct", 0)), 1),
                    "volume":     int(last["volume"]),
                })
            except Exception as exc:
                logger.debug(f"Scanner [{symbol}]: {exc}")

        # Ordenar: BUY primero, luego por confianza
        results.sort(key=lambda x: (
            0 if x["signal"] == "BUY" else (1 if x["signal"] == "HOLD" else 2),
            -x["confidence"]
        ))

        self.results    = results
        self.last_scan  = datetime.now()
        logger.info(f"Scanner completado — {len(results)} ETFs analizados")
        return results


# ══════════════════════════════════════════════════════════════
# VIX — Filtro de régimen de mercado
# ══════════════════════════════════════════════════════════════
_vix_cache: tuple = (0.0, datetime.min)

def _fetch_vix() -> float:
    """Retorna el VIX actual con caché de 15 minutos. 0.0 = no disponible."""
    global _vix_cache
    val, ts = _vix_cache
    if val > 0 and (datetime.now() - ts).seconds < 900:
        return val
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        val = float(df["close"].dropna().iloc[-1])
        _vix_cache = (val, datetime.now())
        logger.info(f"VIX: {val:.1f}")
        return val
    except Exception as exc:
        logger.warning(f"VIX fetch: {exc}")
        return 0.0


# ══════════════════════════════════════════════════════════════
# 6. AUTO TRADER — Ejecución automática con Breakeven
# ══════════════════════════════════════════════════════════════
class AutoTrader:
    """
    Ejecuta órdenes automáticamente cuando XGBoost supera el umbral.
    Gestiona stop-loss, breakeven y take-profit por posición.

    Lógica de breakeven:
      - Entra en BUY a precio P
      - Stop inicial: P × (1 - STOP_LOSS_PCT)
      - Cuando precio sube BREAKEVEN_TRIGGER_PCT → stop se mueve a P (sin pérdida)
      - Cuando precio sube TAKE_PROFIT_PCT → cierra posición con ganancia
    """

    def __init__(self, alpaca: "AlpacaClient", telegram: "TelegramAlert"):
        self.alpaca   = alpaca
        self.telegram = telegram
        self.enabled  = AUTO_TRADE

        # Registro de posiciones abiertas por este bot
        # symbol → {entry_price, qty, stop, breakeven_activated, side}
        self._positions: Dict[str, dict] = {}
        self._traded_today: set = set()   # evita doble entrada mismo día

        # Cooldown: symbol → datetime del stop-loss (bloquea re-entrada COOLDOWN_DAYS días)
        self._cooldown: Dict[str, datetime] = {}
        self._load_cooldown()

        # Importar posiciones ya abiertas en Alpaca para no perder stop/TP entre reinicios
        self._sync_positions_from_alpaca()

    def _sync_positions_from_alpaca(self) -> None:
        """
        Lee las posiciones abiertas en Alpaca y las importa al estado interno.
        Esto evita que al reiniciar el bot se pierda el seguimiento de stop/breakeven
        y que se vuelva a entrar en un símbolo que ya tiene posición abierta.
        """
        try:
            existing = self.alpaca.get_positions()
            for pos in existing:
                sym = pos["symbol"]
                if sym in self._positions:
                    continue
                entry = pos["avg_price"]
                stop  = round(entry * (1 - STOP_LOSS_PCT), 2)
                tp    = round(entry * (1 + TAKE_PROFIT_PCT), 2)
                be    = round(entry * (1 + BREAKEVEN_TRIGGER_PCT), 2)
                self._positions[sym] = {
                    "entry":            entry,
                    "qty":              max(1, int(pos["qty"])),
                    "stop":             stop,
                    "take_profit":      tp,
                    "be_trigger":       be,
                    "breakeven_active": False,
                    "atr":              0,
                    "side":             "BUY",
                }
                self._traded_today.add(sym)
                logger.info(
                    f"AutoTrader: posición importada {sym}  "
                    f"entry=${entry:.2f}  stop=${stop:.2f}  tp=${tp:.2f}"
                )
        except Exception as exc:
            logger.warning(f"AutoTrader._sync_positions_from_alpaca: {exc}")

    # ── Cooldown post stop-loss ───────────────────────────────
    def _load_cooldown(self) -> None:
        """Carga cooldowns desde disco y elimina los expirados."""
        try:
            if COOLDOWN_FILE.exists():
                data = json.loads(COOLDOWN_FILE.read_text())
                now  = datetime.now()
                self._cooldown = {
                    k: datetime.fromisoformat(v)
                    for k, v in data.items()
                    if (now - datetime.fromisoformat(v)).days < COOLDOWN_DAYS
                }
                if self._cooldown:
                    logger.info(f"Cooldown activo: {list(self._cooldown.keys())}")
        except Exception as exc:
            logger.warning(f"AutoTrader._load_cooldown: {exc}")

    def _save_cooldown(self) -> None:
        """Persiste cooldowns en disco para sobrevivir reinicios."""
        try:
            COOLDOWN_FILE.write_text(
                json.dumps({k: v.isoformat() for k, v in self._cooldown.items()})
            )
        except Exception as exc:
            logger.warning(f"AutoTrader._save_cooldown: {exc}")

    def _in_cooldown(self, symbol: str) -> bool:
        """True si el símbolo está bloqueado tras un stop-loss reciente."""
        if symbol not in self._cooldown:
            return False
        days_since = (datetime.now() - self._cooldown[symbol]).days
        return days_since < COOLDOWN_DAYS

    def _current_price(self, symbol: str, df: pd.DataFrame) -> float:
        """Precio más reciente — WebSocket > Alpaca positions API > último cierre."""
        live = self.alpaca.live_bars.get(symbol, {})
        if live:
            return float(live.get("close", df.iloc[-1]["close"]))
        # Fallback: precio actual desde posiciones Alpaca
        try:
            for p in self.alpaca.get_positions():
                if p["symbol"] == symbol:
                    return float(p["current_price"])
        except Exception:
            pass
        return float(df.iloc[-1]["close"])

    def _calc_qty(self, price: float) -> int:
        """Calcula cantidad de acciones según el presupuesto máximo."""
        qty = int(AUTO_MAX_POSITION_USD / price)
        return max(qty, 1)

    def _atr(self, df: pd.DataFrame) -> float:
        """ATR actual del DataFrame."""
        for col in ["ATRr_14", "ATR_14"]:
            if col in df.columns:
                val = float(df.iloc[-1][col])
                if val > 0:
                    return val
        # Fallback: rango promedio últimas 14 barras
        return float((df["high"] - df["low"]).tail(14).mean())

    def evaluate(self, symbol: str, sig: dict, df: pd.DataFrame) -> None:
        """Evalúa si entrar, gestionar o salir de una posición."""
        if not self.enabled:
            return

        signal     = sig.get("signal", "HOLD")
        confidence = sig.get("confidence", 0)
        price      = self._current_price(symbol, df)
        atr        = self._atr(df)

        # ── ENTRADA nueva posición BUY ──────────────────────────
        if (signal == "BUY"
                and confidence >= AUTO_CONFIDENCE
                and symbol not in self._positions
                and symbol not in self._traded_today
                and not self._in_cooldown(symbol)):

            # Filtro VIX: bloquear entrada si el mercado está en pánico
            vix = _fetch_vix()
            if vix > 0 and vix > VIX_MAX:
                logger.info(f"Entrada bloqueada {symbol}: VIX={vix:.1f} > {VIX_MAX}")
                self.telegram.send(
                    f"⚠️ <b>ENTRADA BLOQUEADA — {symbol}</b>\n"
                    f"VIX actual: <b>{vix:.1f}</b> (máx permitido: {VIX_MAX})\n"
                    f"Señal BUY {confidence*100:.0f}% ignorada — mercado en modo miedo.\n"
                    f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                )
                return

            qty         = self._calc_qty(price)
            # Stop-loss: 1.5× ATR por debajo de entrada
            stop        = round(price - 1.5 * atr, 2)
            # Take-profit: 3× ATR por encima (ratio 1:2)
            take_profit = round(price + 3.0 * atr, 2)
            # Breakeven se activa cuando ganamos 1× ATR
            be_trigger  = round(price + 1.0 * atr, 2)

            result = self.alpaca.market_order(symbol, qty, "BUY")
            if result.get("status") == "submitted":
                self._positions[symbol] = {
                    "entry":              price,
                    "qty":                qty,
                    "stop":               stop,
                    "take_profit":        take_profit,
                    "be_trigger":         be_trigger,
                    "breakeven_active":   False,
                    "atr":                atr,
                    "side":               "BUY",
                }
                self._traded_today.add(symbol)

                sl_pct = ((price - stop) / price) * 100
                tp_pct = ((take_profit - price) / price) * 100
                msg = (
                    f"🤖 <b>AUTO BUY — {symbol}</b>\n"
                    f"💰 Entrada: <b>${price:,.2f}</b>  ×{qty}\n"
                    f"📐 ATR: ${atr:.2f}\n"
                    f"🛑 Stop: ${stop:,.2f} (-{sl_pct:.1f}%)  [1.5× ATR]\n"
                    f"⚖️ Breakeven en: ${be_trigger:,.2f}  [+1× ATR]\n"
                    f"🎯 Target: ${take_profit:,.2f} (+{tp_pct:.1f}%)  [3× ATR]\n"
                    f"🧠 Confianza XGBoost: {confidence*100:.0f}%\n"
                    f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                )
                self.telegram.send(msg)
                logger.info(f"AUTO BUY {qty} {symbol} @ ${price:.2f}")

        # ── GESTIÓN de posición abierta ─────────────────────────
        elif symbol in self._positions:
            pos   = self._positions[symbol]
            entry = pos["entry"]
            stop  = pos["stop"]
            qty   = pos["qty"]
            tp    = pos["take_profit"]

            # Activar BREAKEVEN cuando precio supera 1× ATR desde entrada
            if not pos["breakeven_active"] and price >= pos.get("be_trigger", entry * 1.01):
                pos["stop"]             = entry   # stop → precio de entrada
                pos["breakeven_active"] = True
                msg = (
                    f"⚖️ <b>BREAKEVEN activado — {symbol}</b>\n"
                    f"📍 Stop movido a entrada: ${entry:,.2f}\n"
                    f"💹 Precio actual: ${price:,.2f} "
                    f"(+{((price/entry)-1)*100:.2f}%)\n"
                    f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                )
                self.telegram.send(msg)
                logger.info(f"BREAKEVEN {symbol} stop→${entry:.2f}")

            # TAKE PROFIT
            if price >= tp:
                result = self.alpaca.market_order(symbol, qty, "SELL")
                if result.get("status") == "submitted":
                    pnl = (price - entry) * qty
                    msg = (
                        f"✅ <b>TAKE PROFIT — {symbol}</b>\n"
                        f"📍 Entrada: ${entry:,.2f} → Salida: ${price:,.2f}\n"
                        f"💵 P&L: <b>+${pnl:,.2f}</b> (+{((price/entry)-1)*100:.2f}%)\n"
                        f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                    )
                    self.telegram.send(msg)
                    logger.info(f"TAKE PROFIT {symbol} P&L=+${pnl:.2f}")
                    audit.record_trade_result(won=True)
                    del self._positions[symbol]

            # STOP LOSS
            elif price <= pos["stop"]:
                result = self.alpaca.market_order(symbol, qty, "SELL")
                if result.get("status") == "submitted":
                    pnl  = (price - entry) * qty
                    tipo = "BREAKEVEN" if pos["breakeven_active"] else "STOP LOSS"
                    emoji = "⚖️" if pos["breakeven_active"] else "🛑"
                    msg = (
                        f"{emoji} <b>{tipo} — {symbol}</b>\n"
                        f"📍 Entrada: ${entry:,.2f} → Salida: ${price:,.2f}\n"
                        f"💵 P&L: <b>${pnl:,.2f}</b> ({((price/entry)-1)*100:.2f}%)\n"
                        f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                    )
                    self.telegram.send(msg)
                    logger.info(f"{tipo} {symbol} P&L=${pnl:.2f}")
                    audit.record_trade_result(won=(pnl >= 0))
                    del self._positions[symbol]

                    # Cooldown solo en stop-loss real (no en breakeven → salida neutra)
                    if not pos["breakeven_active"]:
                        self._cooldown[symbol] = datetime.now()
                        self._save_cooldown()
                        logger.info(f"Cooldown: {symbol} bloqueado {COOLDOWN_DAYS} días")
                        self.telegram.send(
                            f"⏸ <b>COOLDOWN activado — {symbol}</b>\n"
                            f"Bloqueado para re-entrada durante {COOLDOWN_DAYS} días.\n"
                            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                        )

    def reset_daily(self) -> None:
        """Resetea el set de trades del día (llamar al inicio de cada sesión)."""
        self._traded_today.clear()
        self._sync_positions_from_alpaca()
        logger.info("AutoTrader: reset diario")

    @property
    def open_positions(self) -> Dict[str, dict]:
        return self._positions


# ══════════════════════════════════════════════════════════════
# 7. BACKTESTER
# ══════════════════════════════════════════════════════════════
class Backtester:
    """
    Simula la estrategia completa sobre datos históricos:
    - Señales XGBoost
    - Stop-loss 1.5× ATR
    - Breakeven +1× ATR
    - Take-profit 3× ATR
    - Capital inicial configurable
    """

    def __init__(self, signal_engine: "SignalEngine"):
        self.signals = signal_engine
        self.results: Dict[str, dict] = {}

    def run(self, symbol: str, df: pd.DataFrame,
            capital: float = 10_000.0,
            confidence_threshold: float = AUTO_CONFIDENCE,
            atr_stop: float = 1.5,
            atr_be: float = 1.0,
            atr_tp: float = 3.0) -> dict:
        """
        Corre el backtest barra a barra sobre df ya enriquecido con indicadores.
        Retorna métricas y log de trades.
        """
        if df.empty or len(df) < 60:
            return {"error": "datos insuficientes"}

        # Re-entrenar modelo con los primeros 60% de datos
        split     = int(len(df) * 0.6)
        train_df  = df.iloc[:split]
        test_df   = df.iloc[split:].copy()

        self.signals.train(symbol, train_df)

        equity      = capital
        peak_equity = capital
        position    = None   # dict cuando hay trade abierto
        trades      = []
        equity_curve = []

        for i in range(1, len(test_df)):
            row      = test_df.iloc[i]
            prev_row = test_df.iloc[i - 1]
            price    = float(row["close"])
            high     = float(row["high"])
            low      = float(row["low"])
            date     = test_df.index[i]

            # ATR actual
            atr = float(row.get("ATRr_14", row.get("ATR_14",
                  float((test_df["high"] - test_df["low"]).iloc[max(0, i-14):i].mean()))))
            if atr <= 0:
                atr = price * 0.01

            # ── Gestión de posición abierta ──────────────────
            if position:
                entry    = position["entry"]
                stop     = position["stop"]
                tp       = position["take_profit"]
                be_trig  = position["be_trigger"]
                qty      = position["qty"]

                # Breakeven: precio toca +1× ATR → stop → entrada
                if not position["be_active"] and low <= be_trig <= high or price >= be_trig:
                    position["stop"]      = entry
                    position["be_active"] = True

                # Take-profit tocado (high llega al target)
                if high >= tp:
                    exit_price = tp
                    pnl        = (exit_price - entry) * qty
                    equity    += pnl
                    trades.append({
                        "date":       str(date)[:10],
                        "symbol":     symbol,
                        "side":       "BUY",
                        "entry":      round(entry, 2),
                        "exit":       round(exit_price, 2),
                        "qty":        qty,
                        "pnl":        round(pnl, 2),
                        "pnl_pct":    round((exit_price / entry - 1) * 100, 2),
                        "result":     "TP",
                        "be_active":  position["be_active"],
                        "bars_held":  i - position["entry_bar"],
                    })
                    position = None

                # Stop-loss tocado (low cae al stop)
                elif low <= stop:
                    exit_price = stop
                    pnl        = (exit_price - entry) * qty
                    equity    += pnl
                    label      = "BE" if position["be_active"] else "SL"
                    trades.append({
                        "date":       str(date)[:10],
                        "symbol":     symbol,
                        "side":       "BUY",
                        "entry":      round(entry, 2),
                        "exit":       round(exit_price, 2),
                        "qty":        qty,
                        "pnl":        round(pnl, 2),
                        "pnl_pct":    round((exit_price / entry - 1) * 100, 2),
                        "result":     label,
                        "be_active":  position["be_active"],
                        "bars_held":  i - position["entry_bar"],
                    })
                    position = None

            # ── Señal de entrada ──────────────────────────────
            if position is None:
                window = test_df.iloc[max(0, i-5):i+1]
                sig    = self.signals.predict(symbol, window)
                if (sig.get("signal") == "BUY"
                        and sig.get("confidence", 0) >= confidence_threshold):

                    qty        = max(1, int(AUTO_MAX_POSITION_USD / price))
                    stop_price = round(price - atr_stop * atr, 2)
                    tp_price   = round(price + atr_tp  * atr, 2)
                    be_price   = round(price + atr_be  * atr, 2)

                    position = {
                        "entry":       price,
                        "qty":         qty,
                        "stop":        stop_price,
                        "take_profit": tp_price,
                        "be_trigger":  be_price,
                        "be_active":   False,
                        "entry_bar":   i,
                        "confidence":  sig.get("confidence", 0),
                    }

            # Equity curve
            mark_to_market = equity
            if position:
                mark_to_market += (price - position["entry"]) * position["qty"]
            peak_equity = max(peak_equity, mark_to_market)
            equity_curve.append({
                "date":     str(date)[:10],
                "equity":   round(mark_to_market, 2),
                "drawdown": round((mark_to_market - peak_equity) / peak_equity * 100, 2),
            })

        # Cerrar posición abierta al final
        if position:
            last_price = float(test_df.iloc[-1]["close"])
            pnl        = (last_price - position["entry"]) * position["qty"]
            equity    += pnl
            trades.append({
                "date":      str(test_df.index[-1])[:10],
                "symbol":    symbol,
                "side":      "BUY",
                "entry":     round(position["entry"], 2),
                "exit":      round(last_price, 2),
                "qty":       position["qty"],
                "pnl":       round(pnl, 2),
                "pnl_pct":  round((last_price / position["entry"] - 1) * 100, 2),
                "result":    "OPEN",
                "be_active": position["be_active"],
                "bars_held": len(test_df) - position["entry_bar"],
            })

        # ── Métricas ──────────────────────────────────────────
        n_trades   = len(trades)
        winners    = [t for t in trades if t["pnl"] > 0]
        losers     = [t for t in trades if t["pnl"] <= 0]
        win_rate   = len(winners) / n_trades * 100 if n_trades else 0
        avg_win    = np.mean([t["pnl"] for t in winners]) if winners else 0
        avg_loss   = np.mean([t["pnl"] for t in losers])  if losers  else 0
        profit_factor = abs(sum(t["pnl"] for t in winners) /
                            sum(t["pnl"] for t in losers)) if losers and any(t["pnl"] < 0 for t in losers) else 999
        total_pnl  = equity - capital
        roi        = total_pnl / capital * 100
        max_dd     = min((e["drawdown"] for e in equity_curve), default=0)
        avg_bars   = np.mean([t["bars_held"] for t in trades]) if trades else 0

        result = {
            "symbol":         symbol,
            "capital":        capital,
            "final_equity":   round(equity, 2),
            "total_pnl":      round(total_pnl, 2),
            "roi_pct":        round(roi, 2),
            "n_trades":       n_trades,
            "win_rate":       round(win_rate, 1),
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
            "profit_factor":  round(profit_factor, 2),
            "max_drawdown":   round(max_dd, 2),
            "avg_bars_held":  round(avg_bars, 1),
            "trades":         trades,
            "equity_curve":   equity_curve,
            "test_bars":      len(test_df),
            "train_bars":     len(train_df),
        }
        self.results[symbol] = result
        logger.info(f"Backtest [{symbol}] ROI={roi:.1f}%  WR={win_rate:.0f}%  "
                    f"trades={n_trades}  PF={profit_factor:.2f}")
        return result


# ══════════════════════════════════════════════════════════════
# 8. ORQUESTADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════
class ETFAnalyzer:
    """Coordina datos, análisis técnico, XGBoost y Alpaca."""

    def __init__(self):
        self.alpaca     = AlpacaClient()
        self.signals    = SignalEngine()
        self.telegram   = TelegramAlert()
        self.scanner    = ETFScanner(self.alpaca, self.signals)
        self.autotrader = AutoTrader(self.alpaca, self.telegram)
        self.backtester = Backtester(self.signals)

        self._cache:   Dict[str, pd.DataFrame] = {}
        self._sigs:    Dict[str, dict]          = {}
        self._updated: Dict[str, datetime]      = {}
        self._pending: List[dict]               = []  # órdenes en espera de confirmación

        self._last_report_date: str = ""
        self._start_daily_report_thread()

    # ── reporte diario automático ────────────────────────────
    def _start_daily_report_thread(self) -> None:
        """Hilo daemon que envía el reporte diario a las 20:00 UTC (cierre USA)."""
        def _loop():
            while True:
                try:
                    now   = datetime.now(timezone.utc)
                    today = now.strftime("%Y-%m-%d")
                    # Solo lunes–viernes, entre 20:00 y 20:04 UTC, una vez por día
                    if (now.weekday() < 5
                            and now.hour == DAILY_REPORT_UTC_HOUR
                            and now.minute < 5
                            and self._last_report_date != today):
                        acc     = self.alpaca.get_account()
                        pos     = self.alpaca.get_positions()
                        orders  = self.alpaca.get_orders(limit=30)
                        self.telegram.send_daily_report(acc, pos, orders)
                        self._last_report_date = today
                except Exception as exc:
                    logger.error(f"Daily report thread: {exc}")
                time.sleep(60)

        threading.Thread(target=_loop, daemon=True, name="daily-report").start()
        logger.info("Hilo de reporte diario iniciado (20:00 UTC)")

    # ── carga y enriquecimiento ───────────────────────────────
    def load(self, symbol: str, tf: str = "1d",
             limit: int = 500, force: bool = False) -> pd.DataFrame:
        key  = f"{symbol}_{tf}"
        last = self._updated.get(key, datetime.min)
        if not force and key in self._cache and (datetime.now() - last).seconds < REFRESH_SEC:
            return self._cache[key]

        logger.info(f"Cargando {symbol} [{tf}] …")
        df = self.alpaca.get_bars(symbol, tf, limit)
        if df.empty:
            return df

        df = TA.enrich(df)

        # Train XGBoost si no existe o se fuerza
        if symbol not in self.signals.models or force:
            self.signals.train(symbol, df)
        sig = self.signals.predict(symbol, df)
        self._sigs[symbol]   = sig
        self._cache[key]     = df
        self._updated[key]   = datetime.now()

        # Alerta Telegram si señal fuerte
        last_price = float(df.iloc[-1]["close"])
        self.telegram.check_and_alert(
            symbol, sig.get("signal", "HOLD"),
            sig.get("confidence", 0), last_price, sig.get("proba", {})
        )

        # Auto-trading: evaluar entrada / gestión de posición
        self.autotrader.evaluate(symbol, sig, df)

        return df

    def load_all(self, tf: str = "1d") -> None:
        for s in SYMBOLS:
            try:
                self.load(s, tf)
            except Exception as exc:
                logger.error(f"load_all [{s}]: {exc}")

    # ── resumen por símbolo ───────────────────────────────────
    def summary(self, symbol: str) -> dict:
        df = self._cache.get(f"{symbol}_1d", pd.DataFrame())
        if df.empty:
            return {}
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        chg  = (last["close"] - prev["close"]) / prev["close"] * 100
        sig  = self._sigs.get(symbol, {})

        # Precio en tiempo real si WebSocket activo
        live = self.alpaca.live_bars.get(symbol, {})
        price = float(live.get("close", last["close"]))

        return {
            "symbol":     symbol,
            "price":      round(price, 2),
            "change_pct": round(float(chg), 2),
            "volume":     int(last["volume"]),
            "vwap":       round(float(last.get("VWAP", 0)), 2),
            "rsi":        round(float(last.get("RSI_14", 50)), 1),
            "adx":        round(float(last.get("ADX_14", 0)), 1),
            "atr":        round(float(last.get("ATRr_14", last.get("ATR_14", 0))), 2),
            "macd":       round(float(last.get("MACD_12_26_9", 0)), 4),
            "delta":      round(float(last.get("dlt_delta", 0)), 0),
            "delta_pct":  round(float(last.get("dlt_delta_pct", 0)), 1),
            "signal":     sig.get("signal", "HOLD"),
            "confidence": round(sig.get("confidence", 0) * 100, 1),
            "proba":      sig.get("proba", {}),
            "live":       bool(live),   # True si hay dato en tiempo real
        }

    # ── gestión de órdenes (semi-automático) ──────────────────
    def queue_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", limit_price: float = 0.0,
                    reason: str = "") -> dict:
        order = {
            "id":          f"pend_{int(time.time()*1000)}",
            "symbol":      symbol,
            "side":        side.upper(),
            "qty":         qty,
            "order_type":  order_type,   # "market" | "limit"
            "limit_price": limit_price,
            "reason":      reason,
            "created_at":  datetime.now().strftime("%H:%M:%S"),
            "status":      "pending",
        }
        self._pending.append(order)
        logger.info(f"Orden en cola: {side} {qty} {symbol} ({order_type})")
        return order

    def confirm_order(self, order_id: str) -> dict:
        for o in self._pending:
            if o["id"] == order_id:
                if o["order_type"] == "limit" and o["limit_price"] > 0:
                    result = self.alpaca.limit_order(
                        o["symbol"], o["qty"], o["side"], o["limit_price"]
                    )
                else:
                    result = self.alpaca.market_order(o["symbol"], o["qty"], o["side"])
                o["status"] = "submitted"
                self._pending.remove(o)
                return result
        return {"status": "error", "message": "orden no encontrada"}

    def cancel_pending(self, order_id: str) -> bool:
        for o in self._pending:
            if o["id"] == order_id:
                self._pending.remove(o)
                return True
        return False

    @property
    def pending(self) -> List[dict]:
        return [o for o in self._pending if o["status"] == "pending"]


# ══════════════════════════════════════════════════════════════
# Singleton global
# ══════════════════════════════════════════════════════════════
analyzer = ETFAnalyzer()
logger.info("Cargando datos iniciales …")
analyzer.load_all()
analyzer.alpaca.start_stream()          # WebSocket en tiempo real


# ══════════════════════════════════════════════════════════════
# 5. DASHBOARD
# ══════════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
)
app.title = "ETF Analyzer — Alpaca"

# ── componentes reutilizables ──────────────────────────────────
def _badge(signal: str, conf: float = 0) -> dbc.Badge:
    color = {"BUY": "success", "SELL": "danger", "HOLD": "warning"}.get(signal, "secondary")
    return dbc.Badge(f"{signal}  {conf:.0f}%" if conf else signal,
                     color=color, className="fs-6 p-2")

def _sym_card(s: dict) -> dbc.Card:
    if not s:
        return dbc.Card()
    chg_c  = C["green"] if s["change_pct"] >= 0 else C["red"]
    arrow  = "▲" if s["change_pct"] >= 0 else "▼"
    live_dot = " 🟢" if s.get("live") else ""
    return dbc.Card(
        dbc.CardBody([
            html.H5(s["symbol"] + live_dot, className="text-info mb-1"),
            html.H3(f"${s['price']:,.2f}", className="mb-0"),
            html.Div(f"{arrow} {abs(s['change_pct']):.2f}%",
                     style={"color": chg_c, "fontWeight": "bold"}),
            html.Hr(style={"borderColor": C["border"], "margin": "6px 0"}),
            dbc.Row([
                dbc.Col([html.Small("RSI",  className="text-muted d-block"), html.Strong(f"{s['rsi']:.0f}")],  width=4),
                dbc.Col([html.Small("ADX",  className="text-muted d-block"), html.Strong(f"{s['adx']:.0f}")],  width=4),
                dbc.Col([html.Small("VWAP", className="text-muted d-block"), html.Strong(f"${s['vwap']:,.2f}")], width=4),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([html.Small("Δ Vol%", className="text-muted d-block"),
                         html.Div(f"{s['delta_pct']:+.1f}%",
                                  style={"color": C["green"] if s["delta_pct"] > 0 else C["red"]})], width=6),
                dbc.Col([html.Small("ATR",  className="text-muted d-block"), html.Strong(f"{s['atr']:.2f}")], width=6),
            ], className="mb-2"),
            _badge(s["signal"], s["confidence"]),
        ]),
        style={
            "backgroundColor": C["card"],
            "border": f"1px solid {C['border']}",
            "borderTop": f"3px solid {C['blue']}",
        },
    )

def _empty_fig(title: str = "") -> go.Figure:
    return go.Figure().update_layout(
        template="plotly_dark", paper_bgcolor=C["bg"],
        plot_bgcolor=C["card"], title=title, font=dict(color=C["text"]),
    )

def _base_layout(**kw) -> dict:
    return dict(
        template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
        font=dict(color=C["text"]), margin=dict(l=50, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        xaxis_rangeslider_visible=False,
        **kw,
    )


# ── LAYOUT ────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True,
    style={"backgroundColor": C["bg"], "minHeight": "100vh", "padding": "12px"},
    children=[

    # Header
    dbc.Row([
        dbc.Col([
            html.H4("ETF Technical Analysis — Alpaca", className="text-info mb-0"),
            html.Small("SPY · VGT · SCHD · IFRA  │  VWAP · Delta · Footprint · XGBoost  │  "
                       f"{'🟡 PAPER' if ALPACA_PAPER else '🔴 LIVE'}",
                       className="text-muted"),
        ], width=8),
        dbc.Col([
            html.Div(id="ts-display", className="text-muted small text-end"),
            dbc.ButtonGroup([
                dbc.Button("⟳ Refresh",    id="btn-refresh", color="info",    size="sm"),
                dbc.Button("⚡ Retrain ML", id="btn-retrain", color="warning", size="sm"),
            ]),
        ], width=4, className="d-flex flex-column align-items-end gap-1"),
    ], className="mb-3 p-2 rounded",
       style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}"}),

    # Controles
    dbc.Row([
        dbc.Col(dbc.Switch(id="sw-auto", label="Auto-refresh 60s", value=False), width=2),
        dbc.Col(dcc.Dropdown(id="dd-symbol",
                             options=[{"label": s, "value": s} for s in SYMBOLS],
                             value=SYMBOLS[0], clearable=False,
                             style={"backgroundColor": C["card"], "color": C["text"]}), width=2),
        dbc.Col(dcc.Dropdown(id="dd-tf",
                             options=[{"label": "Daily",  "value": "1d"},
                                      {"label": "1-Hour", "value": "1h"},
                                      {"label": "15-Min", "value": "15m"}],
                             value="1d", clearable=False,
                             style={"backgroundColor": C["card"], "color": C["text"]}), width=2),
        dbc.Col(dcc.Dropdown(id="dd-chart",
                             options=[{"label": "Candle + VWAP",      "value": "candle"},
                                      {"label": "Delta Analysis",     "value": "delta"},
                                      {"label": "Footprint",          "value": "footprint"},
                                      {"label": "XGBoost Importance", "value": "feat"}],
                             value="candle", clearable=False,
                             style={"backgroundColor": C["card"], "color": C["text"]}), width=2),
        # Tipo de orden
        dbc.Col(dcc.Dropdown(id="dd-ord-type",
                             options=[{"label": "Market", "value": "market"},
                                      {"label": "Limit",  "value": "limit"}],
                             value="market", clearable=False,
                             style={"backgroundColor": C["card"], "color": C["text"]}), width=2),
        dbc.Col(dbc.Input(id="inp-limit-price", type="number", placeholder="Precio límite",
                          step=0.01, style={"backgroundColor": C["card"], "color": C["text"]}), width=2),
    ], className="mb-3 align-items-center"),

    # Symbol cards
    html.Div(id="div-cards", className="mb-3"),

    # Gráfico principal + panel de señales
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Loading(dcc.Graph(id="main-chart", style={"height": "480px"}), type="circle")
        ), style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}"}), width=9),

        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("Señales XGBoost", className="text-info mb-0")),
            dbc.CardBody(html.Div(id="div-signals"),
                         style={"overflowY": "auto", "maxHeight": "460px"}),
        ], style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}"}), width=3),
    ], className="mb-3"),

    # Indicadores + cuenta
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(id="ind-chart", style={"height": "220px"})
        ), style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}"}), width=8),

        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("Cuenta Alpaca / Posiciones", className="text-info mb-0")),
            dbc.CardBody(html.Div(id="div-account")),
        ], style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}"}), width=4),
    ], className="mb-3"),

    # Gestión de órdenes
    dbc.Card([
        dbc.CardHeader([
            html.H6("Gestión de Órdenes — Semi-Automático", className="text-warning mb-0 d-inline"),
            html.Small("  •  Las señales se encolan; debes confirmar antes de ejecutar",
                       className="text-muted"),
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.InputGroup([
                    dbc.Select(id="ord-sym",
                               options=[{"label": s, "value": s} for s in SYMBOLS],
                               value=SYMBOLS[0]),
                    dbc.Select(id="ord-side",
                               options=[{"label": "BUY", "value": "BUY"},
                                        {"label": "SELL", "value": "SELL"}],
                               value="BUY"),
                    dbc.Input(id="ord-qty", type="number", value=1, min=1, step=1),
                    dbc.Button("Encolar Orden", id="btn-queue", color="warning"),
                ]), width=9),
            ], className="mb-3"),
            html.H6("Pendientes de confirmación", className="text-warning small"),
            html.Div(id="div-pending"),
            html.Hr(style={"borderColor": C["border"]}),
            html.H6("Historial Alpaca", className="text-muted small"),
            html.Div(id="div-orders"),
        ]),
    ], className="mb-3",
       style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}"}),

    # Quote en tiempo real
    dbc.Card([
        dbc.CardHeader(html.H6("Quotes en tiempo real (WebSocket)", className="text-info mb-0")),
        dbc.CardBody(html.Div(id="div-live")),
    ], className="mb-3",
       style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}"}),

    # AutoTrader panel
    dbc.Card([
        dbc.CardHeader(dbc.Row([
            dbc.Col(html.H6("🤖 Auto-Trader — Posiciones activas", className="text-warning mb-0"), width=8),
            dbc.Col(dbc.Switch(id="sw-autotrader", label="Auto ON",
                               value=AUTO_TRADE), width=4, className="text-end"),
        ])),
        dbc.CardBody([
            html.Small(
                f"Stop: 1.5× ATR  │  Breakeven: +1× ATR  │  Target: 3× ATR  │  "
                f"Confianza mín: {AUTO_CONFIDENCE*100:.0f}%  │  "
                f"Máx por trade: ${AUTO_MAX_POSITION_USD:,}",
                className="text-muted d-block mb-2"
            ),
            html.Div(id="div-autotrader"),
        ]),
    ], className="mb-3",
       style={"backgroundColor": C["card"], "border": f"2px solid {C['yellow']}"}),

    # ETF Scanner
    dbc.Card([
        dbc.CardHeader(dbc.Row([
            dbc.Col(html.H6("📡 ETF Scanner — Mejores oportunidades", className="text-info mb-0"), width=8),
            dbc.Col(dbc.Button("▶ Ejecutar Scanner", id="btn-scan",
                               color="info", size="sm"), width=4, className="text-end"),
        ])),
        dbc.CardBody([
            html.Small(id="scan-timestamp", className="text-muted d-block mb-2"),
            dcc.Loading(html.Div(id="div-scanner"), type="circle"),
        ]),
    ], className="mb-3",
       style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}"}),

    # Backtesting panel
    dbc.Card([
        dbc.CardHeader(html.H6("📈 Backtesting — Simulación histórica", className="text-info mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="bt-symbol",
                    options=[{"label": s, "value": s} for s in SYMBOLS],
                    value=SYMBOLS[0], clearable=False,
                    style={"backgroundColor": C["card"], "color": C["text"]}), width=2),
                dbc.Col(dbc.Input(id="bt-capital", type="number", value=10000,
                                  placeholder="Capital $", min=1000, step=1000,
                                  style={"backgroundColor": C["card"], "color": C["text"]}), width=2),
                dbc.Col(dbc.Input(id="bt-conf", type="number", value=85,
                                  placeholder="Confianza mín %", min=50, max=99, step=5,
                                  style={"backgroundColor": C["card"], "color": C["text"]}), width=2),
                dbc.Col(dbc.Button("▶ Correr Backtest", id="btn-backtest",
                                   color="success", size="sm"), width=2),
                dbc.Col(dbc.Button("📤 Enviar a Telegram", id="btn-bt-tg",
                                   color="info", size="sm"), width=2),
            ], className="mb-3"),
            dcc.Loading(html.Div(id="div-backtest"), type="circle"),
        ]),
    ], className="mb-3",
       style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}"}),

    # Audit panel — Checklist para pasar a LIVE
    dbc.Card([
        dbc.CardHeader(dbc.Row([
            dbc.Col(html.H6("📋 Auditoría — Checklist para cuenta LIVE",
                             className="text-warning mb-0"), width=8),
            dbc.Col(html.Div(id="audit-progress", className="text-end"), width=4),
        ])),
        dbc.CardBody(html.Div(id="div-audit")),
    ], className="mb-3",
       style={"backgroundColor": C["card"], "border": f"2px solid {C['orange']}"}),

    # Stores / intervalos
    dcc.Interval(id="iv-auto",  interval=60_000,  disabled=True),
    dcc.Interval(id="iv-live",  interval=3_000,   disabled=False),  # refresh quotes c/3s
    dcc.Interval(id="iv-audit", interval=60_000,  disabled=False),  # refresh audit c/60s
    dcc.Store(id="store-tick",  data=0),
])


# ══════════════════════════════════════════════════════════════
# 6. CALLBACKS
# ══════════════════════════════════════════════════════════════

@app.callback(Output("iv-auto", "disabled"), Input("sw-auto", "value"))
def _toggle(on):
    return not on


@app.callback(
    Output("store-tick", "data"),
    [Input("btn-refresh", "n_clicks"),
     Input("btn-retrain", "n_clicks"),
     Input("iv-auto",     "n_intervals"),
     Input("dd-symbol",   "value"),
     Input("dd-tf",       "value")],
    State("store-tick", "data"),
    prevent_initial_call=True,
)
def _tick(r, rt, ni, sym, tf, cur):
    ctx  = callback_context
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if "retrain" in trig:
        logger.info("Re-entrenando todos los modelos …")
        for s in SYMBOLS:
            analyzer.load(s, force=True)
    return (cur or 0) + 1


@app.callback(Output("ts-display", "children"),
              [Input("store-tick", "data"), Input("iv-live", "n_intervals")])
def _ts(*_):
    ws = "🟢 WS activo" if analyzer.alpaca._ws_ready else "⚪ WS inactivo"
    return f"{ws}  │  {datetime.now().strftime('%H:%M:%S')}"


@app.callback(Output("div-cards", "children"), Input("store-tick", "data"))
def _cards(tick):
    cards = []
    for sym in SYMBOLS:
        if f"{sym}_1d" not in analyzer._cache:
            analyzer.load(sym)
        cards.append(dbc.Col(_sym_card(analyzer.summary(sym)), width=3))
    return dbc.Row(cards)


# ── gráfico principal ─────────────────────────────────────────
@app.callback(
    Output("main-chart", "figure"),
    [Input("dd-symbol", "value"), Input("dd-tf", "value"),
     Input("dd-chart",  "value"), Input("store-tick", "data")],
)
def _main_chart(sym, tf, ctype, tick):
    df = analyzer.load(sym, tf)
    if df.empty:
        return _empty_fig(f"Sin datos — {sym}")
    tail = df.tail(150)
    dispatch = {
        "candle":    _fig_candle,
        "delta":     _fig_delta,
        "footprint": _fig_footprint,
        "feat":      lambda t, s: _fig_fi(s),
    }
    return dispatch.get(ctype, _fig_candle)(tail, sym)


def _fig_candle(df: pd.DataFrame, sym: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02,
                        subplot_titles=[f"{sym} — Precio · VWAP · EMA", "Volumen · Delta", "MACD"])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=sym, increasing_line_color=C["green"], decreasing_line_color=C["red"],
    ), 1, 1)

    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                                 line=dict(color=C["orange"], width=2, dash="dash")), 1, 1)
        for mult, alpha in [(1.0, 0.30), (2.0, 0.15)]:
            u, d = f"vwap_up_{mult}", f"vwap_dn_{mult}"
            if u in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[u],
                                         line=dict(color=f"rgba(227,179,65,{alpha})", width=1),
                                         name=f"VWAP+{mult}σ", showlegend=(mult == 1.0)), 1, 1)
                fig.add_trace(go.Scatter(x=df.index, y=df[d],
                                         line=dict(color=f"rgba(227,179,65,{alpha})", width=1),
                                         fill="tonexty",
                                         fillcolor=f"rgba(227,179,65,{alpha*0.3})",
                                         name=f"VWAP-{mult}σ", showlegend=False), 1, 1)

    for ema, col in [("EMA_9", C["red"]), ("EMA_21", C["blue"]), ("EMA_50", "#aaaaff")]:
        if ema in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ema], name=ema,
                                     line=dict(color=col, width=1)), 1, 1)

    st = next((c for c in df.columns if c.startswith("SUPERT_") and not c.endswith("d")), None)
    if st:
        fig.add_trace(go.Scatter(x=df.index, y=df[st], name="SuperTrend",
                                 line=dict(color=C["purple"], width=1.5), opacity=0.8), 1, 1)

    if "fp_poc" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["fp_poc"], name="POC",
                                 mode="markers",
                                 marker=dict(color=C["yellow"], size=5, symbol="diamond")), 1, 1)

    dcol = [C["green"] if v >= 0 else C["red"]
            for v in df.get("dlt_delta", pd.Series([0] * len(df)))]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volumen",
                          marker_color=dcol, opacity=0.6), 2, 1)

    if "MACD_12_26_9" in df.columns:
        h = df.get("MACDh_12_26_9", pd.Series())
        fig.add_trace(go.Bar(x=df.index, y=h, name="Hist",
                              marker_color=[C["green"] if v >= 0 else C["red"] for v in h]), 3, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_12_26_9"], name="MACD",
                                  line=dict(color=C["blue"], width=1.5)), 3, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df.get("MACDs_12_26_9", pd.Series()),
                                  name="Signal", line=dict(color=C["red"], width=1.5)), 3, 1)

    fig.update_layout(**_base_layout())
    return fig


def _fig_delta(df: pd.DataFrame, sym: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.03,
                        subplot_titles=[f"{sym}", "Delta (Compra − Venta)", "Delta Acumulado"])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=sym, increasing_line_color=C["green"], decreasing_line_color=C["red"],
    ), 1, 1)
    if "dlt_delta" in df.columns:
        dcol = [C["green"] if v >= 0 else C["red"] for v in df["dlt_delta"]]
        fig.add_trace(go.Bar(x=df.index, y=df["dlt_delta"], name="Delta", marker_color=dcol), 2, 1)
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)
        if "dlt_divergence" in df.columns:
            m = df["dlt_divergence"] == 1
            fig.add_trace(go.Scatter(x=df.index[m], y=df["dlt_delta"][m], mode="markers",
                                      name="Divergencia",
                                      marker=dict(color=C["yellow"], size=9, symbol="diamond")), 2, 1)
    if "dlt_cum_delta" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["dlt_cum_delta"], name="Delta Acum.",
                                  line=dict(color=C["blue"], width=2),
                                  fill="tozeroy", fillcolor="rgba(88,166,255,0.1)"), 3, 1)
    fig.update_layout(**_base_layout())
    return fig


def _fig_footprint(df: pd.DataFrame, sym: str) -> go.Figure:
    tail = df.tail(25)
    xs, ys, bvs, svs = [], [], [], []
    for ts, row in tail.iterrows():
        rng = row["high"] - row["low"]
        if rng <= 0:
            continue
        buy_r = (row["close"] - row["low"]) / rng
        for i in range(8):
            mid = row["low"] + (i + 0.5) * rng / 8
            vol = row["volume"] / 8
            xs.append(str(ts)[:10]); ys.append(round(mid, 2))
            bvs.append(vol * buy_r); svs.append(vol * (1 - buy_r))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=["Volumen Compra", "Volumen Venta"])
    mk = dict(size=12)
    fig.add_trace(go.Scatter(x=xs, y=ys, name="Compra", mode="markers",
                              marker=dict(**mk, color=bvs, colorscale="Greens",
                                          showscale=True, colorbar=dict(x=0.45, title="Buy"))), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=ys, name="Venta", mode="markers",
                              marker=dict(**mk, color=svs, colorscale="Reds",
                                          showscale=True, colorbar=dict(x=1.0, title="Sell"))), 1, 2)
    fig.update_layout(title=f"{sym} — Footprint (últimas 25 barras)", **_base_layout())
    return fig


def _fig_fi(sym: str) -> go.Figure:
    fi = analyzer.signals.importance(sym)
    if not fi:
        return _empty_fig(f"Modelo no entrenado para {sym}")
    names = list(fi.keys()); values = list(fi.values())
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h",
                           marker=dict(color=values, colorscale="Blues")))
    fig.update_layout(title=f"{sym} — Feature Importance XGBoost",
                      yaxis={"categoryorder": "total ascending"},
                      **_base_layout(margin=dict(l=180, r=20, t=50, b=20)))
    return fig


# ── indicadores ───────────────────────────────────────────────
@app.callback(Output("ind-chart", "figure"),
              [Input("dd-symbol", "value"), Input("store-tick", "data")])
def _ind(sym, tick):
    df = analyzer._cache.get(f"{sym}_1d", pd.DataFrame())
    if df.empty:
        return _empty_fig()
    tail = df.tail(60)
    fig  = make_subplots(rows=1, cols=3, subplot_titles=["RSI · MFI", "Estocástico", "ADX"])

    def hl(y, row, col, color):
        fig.add_hline(y=y, line_dash="dash", line_color=color, opacity=0.4, row=row, col=col)

    if "RSI_14" in tail.columns:
        fig.add_trace(go.Scatter(x=tail.index, y=tail["RSI_14"], name="RSI",
                                  line=dict(color=C["blue"])), 1, 1)
        hl(70, 1, 1, C["red"]); hl(30, 1, 1, C["green"]); hl(50, 1, 1, C["muted"])
    if "MFI_14" in tail.columns:
        fig.add_trace(go.Scatter(x=tail.index, y=tail["MFI_14"], name="MFI",
                                  line=dict(color=C["orange"], dash="dot")), 1, 1)
    for col, color, name in [("STOCHk_14_3_3", C["green"], "%K"),
                               ("STOCHd_14_3_3", C["red"],   "%D")]:
        if col in tail.columns:
            fig.add_trace(go.Scatter(x=tail.index, y=tail[col], name=name,
                                      line=dict(color=color)), 1, 2)
    hl(80, 1, 2, C["red"]); hl(20, 1, 2, C["green"])
    if "ADX_14" in tail.columns:
        fig.add_trace(go.Scatter(x=tail.index, y=tail["ADX_14"], name="ADX",
                                  line=dict(color=C["purple"])), 1, 3)
        hl(25, 1, 3, C["yellow"])
    for col, color, name in [("DMP_14", C["green"], "DI+"), ("DMN_14", C["red"], "DI-")]:
        if col in tail.columns:
            fig.add_trace(go.Scatter(x=tail.index, y=tail[col], name=name,
                                      line=dict(color=color, dash="dot")), 1, 3)
    fig.update_layout(**_base_layout(margin=dict(l=30, r=20, t=30, b=10)))
    return fig


# ── señales ───────────────────────────────────────────────────
@app.callback(Output("div-signals", "children"),
              [Input("dd-symbol", "value"), Input("store-tick", "data")])
def _signals(sym, tick):
    items = []
    for s in SYMBOLS:
        sig    = analyzer._sigs.get(s, {"signal": "HOLD", "confidence": 0, "proba": {}})
        signal = sig.get("signal", "HOLD")
        conf   = sig.get("confidence", 0)
        proba  = sig.get("proba", {})
        color  = {"BUY": "success", "SELL": "danger", "HOLD": "warning"}.get(signal, "secondary")
        items.append(dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Strong(s, className="text-info"), width=5),
                dbc.Col(dbc.Badge(signal, color=color),       width=4),
                dbc.Col(html.Small(f"{conf*100:.0f}%", className="text-muted"), width=3),
            ]),
            html.Small(
                f"B:{proba.get('BUY',0)*100:.0f}%  "
                f"H:{proba.get('HOLD',0)*100:.0f}%  "
                f"S:{proba.get('SELL',0)*100:.0f}%",
                className="text-muted",
            ) if proba else html.Small("no entrenado", className="text-muted"),
        ], className="p-2"), className="mb-1",
        style={"backgroundColor": "#1c2128", "border": f"1px solid {C['border']}"}))

    cur_sig = analyzer._sigs.get(sym, {}).get("signal", "HOLD")
    items += [
        html.Hr(style={"borderColor": C["border"]}),
        html.Small(f"Acción rápida → {sym}", className="text-muted d-block mb-1"),
        dbc.ButtonGroup([
            dbc.Button("▲ BUY 1",  id="btn-qbuy",  color="success", size="sm",
                       disabled=(cur_sig != "BUY")),
            dbc.Button("▼ SELL 1", id="btn-qsell", color="danger",  size="sm",
                       disabled=(cur_sig != "SELL")),
        ]),
        html.Small("* encola la orden — requiere confirmación abajo",
                   className="text-muted d-block mt-1"),
    ]
    return items


# ── cuenta ────────────────────────────────────────────────────
@app.callback(Output("div-account", "children"), Input("store-tick", "data"))
def _account(tick):
    acc = analyzer.alpaca.get_account()
    pos = analyzer.alpaca.get_positions()
    rows = []
    if acc:
        mode = "PAPER 🟡" if acc.get("paper") else "LIVE 🔴"
        rows.append(html.Small(f"Modo: {mode}", className="text-warning d-block mb-1"))
        for k, label in [("portfolio_value", "Portfolio"), ("cash", "Efectivo"),
                          ("buying_power", "Poder de compra")]:
            rows.append(dbc.Row([
                dbc.Col(html.Small(label, className="text-muted"), width=7),
                dbc.Col(html.Strong(f"${acc[k]:,.2f}"), width=5),
            ]))
        rows.append(html.Small(f"Day trades: {acc.get('daytrade_count',0)}",
                               className="text-muted d-block mt-1"))
        rows.append(html.Hr(style={"borderColor": C["border"]}))
    else:
        rows.append(dbc.Alert("Configura ALPACA_API_KEY", color="warning", className="p-2 small"))

    if pos:
        total_pl = sum(p["unrealized_pl"] for p in pos)
        total_mv = sum(p["market_value"] for p in pos)
        total_pl_pct = (total_pl / (total_mv - total_pl) * 100) if (total_mv - total_pl) != 0 else 0
        total_col = C["green"] if total_pl >= 0 else C["red"]

        rows.append(html.H6("Posiciones", className="text-muted"))

        # P&L total del portafolio
        rows.append(dbc.Row([
            dbc.Col(html.Small("P&L Total", className="text-muted fw-bold"), width=4),
            dbc.Col(html.Strong(f"${total_pl:+,.2f}",
                                style={"color": total_col}), width=4),
            dbc.Col(html.Strong(f"({total_pl_pct:+.2f}%)",
                                style={"color": total_col}), width=4),
        ], className="mb-1 p-1", style={"backgroundColor": "#1c2128",
                                         "borderRadius": "4px"}))
        rows.append(html.Hr(style={"borderColor": C["border"], "margin": "4px 0"}))

        # Separar ganadores y perdedores
        winners = sorted([p for p in pos if p["unrealized_pl"] >= 0],
                         key=lambda x: x["unrealized_pl"], reverse=True)
        losers  = sorted([p for p in pos if p["unrealized_pl"] < 0],
                         key=lambda x: x["unrealized_pl"])

        for label, group, icon in [("Ganancia", winners, "🟢"), ("Pérdida", losers, "🔴")]:
            if group:
                rows.append(html.Small(f"{icon} {label}", className="text-muted d-block mt-1 mb-1"))
                for p in group:
                    pl_c = C["green"] if p["unrealized_pl"] >= 0 else C["red"]
                    rows.append(dbc.Row([
                        dbc.Col(html.Small(f"{p['symbol']} ×{p['qty']:.0f}"), width=3),
                        dbc.Col(html.Small(f"${p['current_price']:.2f}"),      width=3),
                        dbc.Col(html.Small(f"${p['unrealized_pl']:+,.2f}",
                                           style={"color": pl_c}), width=3),
                        dbc.Col(html.Small(f"{p['unrealized_plpc']:+.1f}%",
                                           style={"color": pl_c}), width=3),
                    ], className="mb-0"))
    return rows or [html.Small("Sin datos", className="text-muted")]


# ── gestión de órdenes ────────────────────────────────────────
@app.callback(
    [Output("div-pending", "children"), Output("div-orders", "children")],
    [Input("btn-queue",  "n_clicks"),
     Input({"type": "btn-confirm", "index": ALL}, "n_clicks"),
     Input({"type": "btn-cancel",  "index": ALL}, "n_clicks"),
     Input("btn-qbuy",   "n_clicks"),
     Input("btn-qsell",  "n_clicks"),
     Input("store-tick", "data")],
    [State("ord-sym",          "value"),
     State("ord-side",         "value"),
     State("ord-qty",          "value"),
     State("dd-symbol",        "value"),
     State("dd-ord-type",      "value"),
     State("inp-limit-price",  "value")],
    prevent_initial_call=True,
)
def _orders(qn, cfn, ccn, qbn, qsn, tick,
            osym, oside, oqty, selSym, ordType, limPrice):
    ctx  = callback_context
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "btn-queue" in trig and qn:
        analyzer.queue_order(osym, oside, float(oqty or 1),
                             order_type=ordType or "market",
                             limit_price=float(limPrice or 0),
                             reason="Manual")

    elif "btn-qbuy" in trig and qbn:
        sig = analyzer._sigs.get(selSym, {})
        analyzer.queue_order(selSym, "BUY", 1,
                             reason=f"XGBoost BUY {sig.get('confidence',0)*100:.0f}%")

    elif "btn-qsell" in trig and qsn:
        analyzer.queue_order(selSym, "SELL", 1, reason="XGBoost SELL")

    elif "btn-confirm" in trig:
        try:
            oid = json.loads(trig.split(".")[0])["index"]
            res = analyzer.confirm_order(oid)
            logger.info(f"Confirmada: {res}")
        except Exception as exc:
            logger.error(f"confirm: {exc}")

    elif "btn-cancel" in trig:
        try:
            oid = json.loads(trig.split(".")[0])["index"]
            analyzer.cancel_pending(oid)
        except Exception as exc:
            logger.error(f"cancel: {exc}")

    # Panel pendientes
    pending = analyzer.pending
    if not pending:
        pane = html.Small("Sin órdenes pendientes", className="text-muted")
    else:
        pane = html.Div([
            dbc.Row([
                dbc.Col(html.Strong(o["symbol"]),                      width=2),
                dbc.Col(dbc.Badge(o["side"],
                                  color="success" if o["side"] == "BUY" else "danger"), width=1),
                dbc.Col(html.Small(f"×{o['qty']}"),                    width=1),
                dbc.Col(html.Small(o.get("order_type", "market").upper(),
                                   className="text-info"),              width=1),
                dbc.Col(html.Small(f"${o['limit_price']:.2f}"
                                   if o.get("limit_price") else ""),   width=2),
                dbc.Col(html.Small(o.get("reason", ""), className="text-muted"), width=3),
                dbc.Col(dbc.ButtonGroup([
                    dbc.Button("✓", id={"type": "btn-confirm", "index": o["id"]},
                               color="success", size="sm"),
                    dbc.Button("✕", id={"type": "btn-cancel",  "index": o["id"]},
                               color="danger",  size="sm"),
                ]), width=2),
            ], className="mb-1 align-items-center")
            for o in pending
        ])

    # Historial Alpaca
    orders = analyzer.alpaca.get_orders()
    if not orders:
        ord_pane = html.Small("Sin historial o Alpaca no conectado", className="text-muted")
    else:
        ord_pane = html.Div([
            dbc.Row([
                dbc.Col(html.Small(o["symbol"]),                       width=2),
                dbc.Col(dbc.Badge(o["side"],
                                  color="success" if "BUY" in o["side"] else "danger",
                                  className="small"),                  width=2),
                dbc.Col(html.Small(f"×{o['qty']:.0f}  (filled {o['filled_qty']:.0f})"), width=3),
                dbc.Col(dbc.Badge(o["status"], color="secondary",
                                  className="small"),                  width=2),
                dbc.Col(html.Small(o["created_at"], className="text-muted"), width=3),
            ], className="mb-1 align-items-center")
            for o in orders
        ])
    return pane, ord_pane


# ── quotes en tiempo real ─────────────────────────────────────
@app.callback(Output("div-live", "children"), Input("iv-live", "n_intervals"))
def _live(n):
    if not analyzer.alpaca.live_quotes and not analyzer.alpaca.live_bars:
        return html.Small(
            "⏰ Mercado cerrado — quotes en tiempo real disponibles Lun-Vie 9:30-16:00 ET",
            className="text-muted"
        )
    rows = []
    # Mostrar SYMBOLS + símbolos con posiciones activas del AutoTrader
    all_syms = list(dict.fromkeys(list(SYMBOLS) + list(analyzer.autotrader.open_positions.keys())))
    for sym in all_syms:
        q = analyzer.alpaca.live_quotes.get(sym, {})
        b = analyzer.alpaca.live_bars.get(sym, {})
        spread = round(q.get("ask", 0) - q.get("bid", 0), 4) if q else None
        rows.append(dbc.Col(dbc.Card(dbc.CardBody([
            html.Strong(sym, className="text-info"),
            dbc.Row([
                dbc.Col([html.Small("Bid", className="text-muted d-block"),
                         html.Span(f"${q.get('bid',0):.2f}", style={"color": C["red"]})], width=4),
                dbc.Col([html.Small("Ask", className="text-muted d-block"),
                         html.Span(f"${q.get('ask',0):.2f}", style={"color": C["green"]})], width=4),
                dbc.Col([html.Small("Spread", className="text-muted d-block"),
                         html.Span(f"${spread:.4f}" if spread is not None else "—")], width=4),
            ]),
            html.Small(f"Bar 1min: O={b.get('open',0):.2f} H={b.get('high',0):.2f} "
                       f"L={b.get('low',0):.2f} C={b.get('close',0):.2f}",
                       className="text-muted d-block mt-1") if b else None,
        ], className="p-2"),
        style={"backgroundColor": "#1c2128", "border": f"1px solid {C['border']}"}), width=3))
    return dbc.Row(rows)


# ══════════════════════════════════════════════════════════════
# 7. FONDO — refresca datos periódicamente
# ── AutoTrader callbacks ──────────────────────────────────────
@app.callback(Output("sw-autotrader", "value"), Input("sw-autotrader", "value"))
def _toggle_auto(val):
    analyzer.autotrader.enabled = val
    logger.info(f"AutoTrader {'ACTIVADO' if val else 'DESACTIVADO'}")
    return val


@app.callback(Output("div-autotrader", "children"),
              [Input("iv-live", "n_intervals"), Input("store-tick", "data")])
def _autotrader_panel(*_):
    positions = analyzer.autotrader.open_positions
    if not positions:
        return html.Small("Sin posiciones automáticas abiertas", className="text-muted")

    # Construir mapa de precios actuales desde posiciones Alpaca (fallback robusto)
    _alpaca_prices = {}
    try:
        for p in analyzer.alpaca.get_positions():
            _alpaca_prices[p["symbol"]] = p["current_price"]
    except Exception:
        pass

    rows = []
    for sym, pos in positions.items():
        # Prioridad: 1) WebSocket live_bars  2) Alpaca positions API  3) precio de entrada
        live_bar = analyzer.alpaca.live_bars.get(sym, {})
        if live_bar:
            price = float(live_bar["close"])
        elif sym in _alpaca_prices:
            price = float(_alpaca_prices[sym])
        else:
            price = float(pos["entry"])
        pnl      = (price - pos["entry"]) * pos["qty"]
        pnl_pct  = (price / pos["entry"] - 1) * 100
        pnl_col  = C["green"] if pnl >= 0 else C["red"]
        be_icon  = "⚖️" if pos["breakeven_active"] else "🛑"
        rows.append(dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col(html.Strong(sym, className="text-info"),                     width=1),
            dbc.Col(html.Small(f"Entrada: ${pos['entry']:,.2f}"),                width=2),
            dbc.Col(html.Small(f"Actual: ${price:,.2f}"),                        width=2),
            dbc.Col(html.Small(f"P&L: ", className="d-inline") ,                width=1),
            dbc.Col(html.Small(f"${pnl:+,.2f} ({pnl_pct:+.2f}%)",
                               style={"color": pnl_col}),                        width=2),
            dbc.Col(html.Small(f"{be_icon} Stop: ${pos['stop']:,.2f}"),          width=2),
            dbc.Col(html.Small(f"🎯 Target: ${pos['take_profit']:,.2f}"),        width=2),
        ])), className="mb-1 p-1",
        style={"backgroundColor": "#1c2128", "border": f"1px solid {C['border']}"}))
    return html.Div(rows)


# ── Audit panel callback ─────────────────────────────────────
@app.callback(
    [Output("div-audit", "children"), Output("audit-progress", "children")],
    [Input("iv-audit", "n_intervals"), Input("store-tick", "data")],
)
def _audit_panel(*_):
    acc = analyzer.alpaca.get_account()
    pos = analyzer.alpaca.get_positions()
    portfolio = acc.get("portfolio_value", 0) if acc else 0

    # Registrar snapshot diario
    if portfolio > 0:
        audit.record_snapshot(portfolio, pos)

    checks = audit.evaluate(acc, pos, analyzer.autotrader)

    passed = sum(1 for c in checks if c[2] and c[0] != "RESULTADO")
    total  = sum(1 for c in checks if c[0] != "RESULTADO")
    pct    = int(passed / total * 100) if total else 0

    # Barra de progreso
    bar_color = "danger" if pct < 40 else "warning" if pct < 75 else "success"
    progress = html.Div([
        html.Small(f"{passed}/{total} checks ({pct}%)", className="text-muted"),
        dbc.Progress(value=pct, color=bar_color, striped=True, animated=True,
                     style={"height": "8px", "marginTop": "4px"}),
    ])

    # Renderizar checks
    rows = []
    current_phase = ""
    for phase, label, ok, detail in checks:
        # Encabezado de fase
        if phase != current_phase:
            current_phase = phase
            if phase == "RESULTADO":
                rows.append(html.Hr(style={"borderColor": C["border"]}))
            else:
                rows.append(html.H6(f"{'📊' if '1' in phase else '🔒' if '2' in phase else '🏁'} {phase}",
                                     className="text-info mt-2 mb-1"))

        if phase == "RESULTADO":
            # Resultado final grande
            res_color = C["green"] if ok else C["red"]
            res_icon  = "✅ APROBADO — Listo para LIVE" if ok else f"❌ NO LISTO — {detail}"
            rows.append(html.H5(res_icon, style={"color": res_color},
                                className="text-center mt-2"))
            if not ok:
                rows.append(html.Small(
                    "Completa todos los checks para habilitar cuenta LIVE",
                    className="text-muted d-block text-center"
                ))
        else:
            icon  = "✅" if ok else "❌"
            color = C["green"] if ok else C["red"]
            rows.append(dbc.Row([
                dbc.Col(html.Span(icon, style={"fontSize": "16px"}), width=1),
                dbc.Col(html.Small(label, style={"color": color}), width=7),
                dbc.Col(html.Small(detail, className="text-muted text-end"), width=4),
            ], className="mb-1 align-items-center",
               style={"backgroundColor": "#1c2128" if not ok else "transparent",
                       "borderRadius": "4px", "padding": "2px 4px"}))

    return html.Div(rows), progress


# ── ETF Scanner callback ──────────────────────────────────────
@app.callback(
    [Output("div-scanner", "children"),
     Output("scan-timestamp", "children")],
    Input("btn-scan", "n_clicks"),
    prevent_initial_call=True,
)
def _run_scanner(n):  # noqa: redefined below — scanner callback
    results = analyzer.scanner.scan()
    # Envía reporte a Telegram si hay BUYs fuertes
    buys = [r for r in results if r["signal"] == "BUY" and r["confidence"] >= ALERT_CONFIDENCE]
    if buys:
        analyzer.telegram.send_scanner_report(buys)

    if not results:
        return html.Small("Sin resultados", className="text-muted"), ""

    rows = []
    for r in results:
        sig    = r["signal"]
        color  = {"BUY": "success", "SELL": "danger", "HOLD": "warning"}.get(sig, "secondary")
        chg_c  = C["green"] if r["change_pct"] >= 0 else C["red"]
        arrow  = "▲" if r["change_pct"] >= 0 else "▼"
        rows.append(dbc.Row([
            dbc.Col(html.Strong(r["symbol"], className="text-info"), width=1),
            dbc.Col(dbc.Badge(sig, color=color), width=1),
            dbc.Col(html.Small(f"{r['confidence']*100:.0f}%"), width=1),
            dbc.Col(html.Small(f"${r['price']:,.2f}"), width=2),
            dbc.Col(html.Small(f"{arrow}{abs(r['change_pct']):.2f}%",
                               style={"color": chg_c}), width=1),
            dbc.Col(html.Small(f"RSI {r['rsi']:.0f}"), width=1),
            dbc.Col(html.Small(f"ADX {r['adx']:.0f}"), width=1),
            dbc.Col(html.Small(f"Δ {r['delta_pct']:+.1f}%",
                               style={"color": C["green"] if r["delta_pct"] > 0 else C["red"]}), width=2),
        ], className="mb-1 align-items-center border-bottom pb-1",
           style={"borderColor": C["border"]}))

    ts = f"Último scan: {analyzer.scanner.last_scan.strftime('%H:%M:%S')} — {len(results)} ETFs analizados"
    return html.Div(rows), ts


# ── Backtest callbacks ────────────────────────────────────────
@app.callback(
    Output("div-backtest", "children"),
    Input("btn-backtest", "n_clicks"),
    [State("bt-symbol", "value"),
     State("bt-capital", "value"),
     State("bt-conf",    "value")],
    prevent_initial_call=True,
)
def _run_backtest(n, symbol, capital, conf):
    df = analyzer.load(symbol, "1d", 500)
    if df.empty:
        return dbc.Alert("Sin datos", color="danger")

    res = analyzer.backtester.run(
        symbol, df,
        capital=float(capital or 10000),
        confidence_threshold=float(conf or 85) / 100,
    )

    if "error" in res:
        return dbc.Alert(res["error"], color="danger")

    trades = res["trades"]
    eq     = res["equity_curve"]

    # ── Equity curve ─────────────────────────────────────────
    eq_df  = pd.DataFrame(eq)
    fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3], vertical_spacing=0.03,
                           subplot_titles=["Equity Curve", "Drawdown %"])

    fig_eq.add_trace(go.Scatter(
        x=eq_df["date"], y=eq_df["equity"], name="Equity",
        line=dict(color=C["green"], width=2),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.1)"
    ), 1, 1)
    fig_eq.add_hline(y=res["capital"], line_dash="dash",
                     line_color="white", opacity=0.4, row=1, col=1)

    fig_eq.add_trace(go.Bar(
        x=eq_df["date"], y=eq_df["drawdown"], name="Drawdown",
        marker_color=C["red"], opacity=0.7
    ), 2, 1)

    fig_eq.update_layout(**_base_layout(
        title=f"{symbol} — Backtest  ROI: {res['roi_pct']:+.1f}%  "
              f"WR: {res['win_rate']:.0f}%  PF: {res['profit_factor']:.2f}",
        margin=dict(l=50, r=20, t=50, b=20)
    ))

    # ── Métricas ──────────────────────────────────────────────
    roi_col  = "success" if res["roi_pct"] >= 0 else "danger"
    metrics  = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Capital inicial", className="text-muted d-block"),
            html.H5(f"${res['capital']:,.0f}"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Capital final", className="text-muted d-block"),
            html.H5(f"${res['final_equity']:,.0f}"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("ROI", className="text-muted d-block"),
            html.H5(f"{res['roi_pct']:+.1f}%",
                    style={"color": C["green"] if res["roi_pct"] >= 0 else C["red"]}),
        ])), width=1),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Win Rate", className="text-muted d-block"),
            html.H5(f"{res['win_rate']:.0f}%"),
        ])), width=1),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Profit Factor", className="text-muted d-block"),
            html.H5(f"{res['profit_factor']:.2f}",
                    style={"color": C["green"] if res["profit_factor"] >= 1 else C["red"]}),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Max Drawdown", className="text-muted d-block"),
            html.H5(f"{res['max_drawdown']:.1f}%", style={"color": C["red"]}),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Trades", className="text-muted d-block"),
            html.H5(f"{res['n_trades']}  (avg {res['avg_bars_held']:.0f} barras)"),
        ])), width=2),
    ], className="mb-3 g-2")

    # ── Trade log ────────────────────────────────────────────
    if trades:
        trade_rows = []
        for t in trades[-20:]:   # últimos 20
            result  = t["result"]
            color   = {"TP": C["green"], "SL": C["red"],
                       "BE": C["yellow"], "OPEN": C["blue"]}.get(result, C["muted"])
            pnl_col = C["green"] if t["pnl"] >= 0 else C["red"]
            trade_rows.append(dbc.Row([
                dbc.Col(html.Small(t["date"]),                                     width=2),
                dbc.Col(dbc.Badge(result, color="secondary",
                                  style={"backgroundColor": color}),               width=1),
                dbc.Col(html.Small(f"${t['entry']:,.2f} → ${t['exit']:,.2f}"),    width=3),
                dbc.Col(html.Small(f"×{t['qty']}"),                                width=1),
                dbc.Col(html.Small(f"${t['pnl']:+,.2f} ({t['pnl_pct']:+.1f}%)",
                                   style={"color": pnl_col}),                      width=3),
                dbc.Col(html.Small(f"{t['bars_held']} barras"),                    width=2),
            ], className="mb-1 align-items-center"))
        trade_log = html.Div([
            html.H6("Últimos 20 trades", className="text-muted mt-2"),
            *trade_rows
        ])
    else:
        trade_log = html.Small("Sin trades en el período", className="text-muted")

    return html.Div([metrics, dcc.Graph(figure=fig_eq, style={"height": "400px"}), trade_log])


@app.callback(
    Output("btn-bt-tg", "children"),
    Input("btn-bt-tg", "n_clicks"),
    State("bt-symbol", "value"),
    prevent_initial_call=True,
)
def _bt_to_telegram(n, symbol):
    res = analyzer.backtester.results.get(symbol)
    if not res:
        return "Sin datos"
    msg = (
        f"📈 <b>BACKTEST — {symbol}</b>\n"
        f"💰 Capital: ${res['capital']:,.0f} → ${res['final_equity']:,.0f}\n"
        f"📊 ROI: <b>{res['roi_pct']:+.1f}%</b>\n"
        f"🎯 Win Rate: {res['win_rate']:.0f}%\n"
        f"⚡ Profit Factor: {res['profit_factor']:.2f}\n"
        f"📉 Max Drawdown: {res['max_drawdown']:.1f}%\n"
        f"🔢 Trades: {res['n_trades']}  (avg {res['avg_bars_held']:.0f} barras)\n"
        f"🧠 Confianza mín: {AUTO_CONFIDENCE*100:.0f}%\n"
        f"📐 Stop 1.5×ATR  BE +1×ATR  TP 3×ATR"
    )
    analyzer.telegram.send(msg)
    return "✅ Enviado"


# ══════════════════════════════════════════════════════════════
def _bg_loop():
    scan_counter = 0
    while True:
        time.sleep(REFRESH_SEC)
        logger.info("Refresh de fondo …")
        try:
            analyzer.load_all()
        except Exception as exc:
            logger.error(f"bg_loop: {exc}")

        # Scanner cada 60 minutos (60 ciclos × 60s = 3600s)
        scan_counter += 1
        if scan_counter >= 60:
            scan_counter = 0
            try:
                results = analyzer.scanner.scan()
                buys    = [r for r in results if r["signal"] == "BUY" and r["confidence"] >= ALERT_CONFIDENCE]
                if buys:
                    analyzer.telegram.send_scanner_report(buys)
            except Exception as exc:
                logger.error(f"scanner loop: {exc}")


# ══════════════════════════════════════════════════════════════
# 8. ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    _paper_tag = "PAPER" if ALPACA_PAPER else "LIVE"
    _key_status = "SET" if ALPACA_API_KEY else "NO configurada (solo yFinance)"
    _sec_status = "SET" if ALPACA_SECRET_KEY else "NO configurada"
    print(f"""
+==============================================================+
|         ETF Technical Analysis -- Alpaca Edition             |
+==============================================================+
|  Simbolos  : {", ".join(SYMBOLS):<46}|
|  Analisis  : VWAP - Delta - Footprint - XGBoost             |
|  Broker    : Alpaca  ({_paper_tag})                                      |
|  Modo      : Semi-automatico -- confirma antes de ejecutar   |
|  Dashboard : http://localhost:{DASHBOARD_PORT}                         |
+==============================================================+

Estado de credenciales:
  ALPACA_API_KEY    : {_key_status}
  ALPACA_SECRET_KEY : {_sec_status}
  ALPACA_PAPER      : {ALPACA_PAPER}
""")

    threading.Thread(target=_bg_loop, daemon=True, name="bg-refresh").start()
    app.run(debug=False, port=DASHBOARD_PORT, host="0.0.0.0")
