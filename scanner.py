# scanner.py — Market scanner (30-min NY session loop)
import json
import logging
import time
import requests
import threading
from datetime import datetime, timezone
import pytz

from config import (
    SCAN_INTERVAL_MIN, NY_OPEN_HOUR, NY_OPEN_MIN,
    NY_CLOSE_HOUR, NY_CLOSE_MIN, MARKET_TIMEZONE,
    MIN_SCORE_BUY, MIN_SCORE_BY_TICKER, CIRCUIT_BREAKER_ENABLED, CIRCUIT_BREAKER_URL,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    COOLDOWN_HOURS, COOLDOWN_SL_HOURS, MAX_OPEN_POSITIONS,
    VIX_MAX, VIX_TICKER,
    TRAILING_STOP_PCT, TRAILING_ACTIVATE_PCT, TAKE_PROFIT_PCT, TAKE_PROFIT_HIGH,
    STOP_LOSS_PCT, ATR_MULTIPLIER, ADX_HIGH_TREND,
    FEAR_GREED_REDUCE_THRESHOLD, FEAR_GREED_REDUCE_PCT, MAX_POSITION_PCT,
    AUTO_TRADE,
)
import yfinance as yf
from scorer import score_all_etfs
from ml_model import train_all
from sentiment_free import get_news_sentiment

logger = logging.getLogger(__name__)

_tz           = pytz.timezone(MARKET_TIMEZONE)
_pending_trades: dict[str, dict] = {}   # ticker → signal dict
_scan_thread: threading.Thread | None = None
_running = False
_last_audit_date: str = ""              # fecha del ultimo chequeo diario (YYYY-MM-DD)

_TRAILING_STATE_FILE = "trailing_state.json"


# ── Telegram ──────────────────────────────────────────────────────────────────

def _tg_send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured.")
        return
    try:
        url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       text,
            "parse_mode": "HTML",
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


def _calc_dynamic_tp(result: dict) -> float:
    """
    Take-profit dinamico segun ADX:
      ADX >= ADX_HIGH_TREND (30): usar TAKE_PROFIT_HIGH (18%)
      ADX < 30: usar TAKE_PROFIT_PCT (12%)
    """
    adx = result.get("adx", 0)
    return TAKE_PROFIT_HIGH if adx >= ADX_HIGH_TREND else TAKE_PROFIT_PCT


def _calc_position_pct(sentiment: dict) -> float:
    """
    Reduce el tamano de posicion cuando Fear & Greed esta en miedo extremo.
    Normal: MAX_POSITION_PCT (10%)
    F&G < FEAR_GREED_REDUCE_THRESHOLD: FEAR_GREED_REDUCE_PCT (5%)
    """
    fg = sentiment.get("fear_greed_score", 50)
    if fg < FEAR_GREED_REDUCE_THRESHOLD:
        return FEAR_GREED_REDUCE_PCT
    return MAX_POSITION_PCT


def _calc_dynamic_sl(result: dict) -> float:
    """
    Calcula el stop-loss dinamico basado en ATR.
    SL = max(STOP_LOSS_PCT, ATR * ATR_MULTIPLIER / precio_actual)
    Maximo 10% para evitar SLs demasiado amplios.
    """
    atr   = result.get("atr", 0.0)
    price = result.get("current_price", 0.0)
    if atr > 0 and price > 0:
        atr_sl = (ATR_MULTIPLIER * atr) / price
        return round(max(STOP_LOSS_PCT, min(atr_sl, 0.10)), 4)
    return STOP_LOSS_PCT


def notify_signal(result: dict):
    """Send a BUY signal to Telegram for manual approval."""
    ticker     = result["ticker"]
    score      = result["score"]
    signal     = result["signal"]
    tech       = result["technical"]["score"]
    ml         = result["ml"]["score"]
    fg         = result["sentiment"]["fear_greed"]
    fgl        = result["sentiment"]["fear_greed_label"]
    spy_rsi    = result["sentiment"]["spy_rsi"]
    rsi_sig    = result["sentiment"]["spy_rsi_signal"]
    ts         = result["timestamp"]
    adx        = result["technical"]["components"].get("adx", 0)
    adx_label  = result["technical"]["components"].get("adx_label", "?")
    dynamic_sl = result.get("dynamic_sl", STOP_LOSS_PCT)
    dynamic_tp = result.get("dynamic_tp", TAKE_PROFIT_PCT)
    pos_pct    = result.get("position_pct", MAX_POSITION_PCT)
    spy_ok     = result.get("spy_above_sma", True)
    news       = result.get("news_sentiment", {})
    news_label = news.get("label", "N/A")
    news_n     = news.get("headline_count", 0)

    msg = (
        f"<b>ETF Signal -- {ticker}</b>\n"
        f"Action : <b>{signal}</b>\n"
        f"Score  : <b>{score:.1f}/100</b>\n"
        f"-----------------\n"
        f"Technical  : {tech:.1f}/100\n"
        f"ML (XGB+RF): {ml:.1f}/100\n"
        f"-----------------\n"
        f"ADX        : {adx:.1f} ({adx_label})\n"
        f"Stop-Loss  : {dynamic_sl*100:.1f}% (ATR)\n"
        f"Take-Profit: {dynamic_tp*100:.0f}% ({'alto' if dynamic_tp > TAKE_PROFIT_PCT else 'normal'})\n"
        f"Posicion   : {pos_pct*100:.0f}% capital\n"
        f"SPY        : {'OK' if spy_ok else 'BAJO SMA50'}\n"
        f"-----------------\n"
        f"Fear & Greed : {fg:.0f} ({fgl})\n"
        f"SPY RSI      : {spy_rsi:.1f} ({rsi_sig})\n"
        f"Noticias     : {news_label} ({news_n} titulares)\n"
        f"-----------------\n"
        f"{ts}\n\n"
        f"/approve_{ticker}  o  /reject_{ticker}"
    )
    _tg_send(msg)
    logger.info(f"[{ticker}] Signal sent to Telegram.")


# ── Circuit Breaker ───────────────────────────────────────────────────────────

def _circuit_breaker_ok() -> bool:
    """Returns True if circuit breaker allows trading.
    If CIRCUIT_BREAKER_ENABLED=False (default), always returns True (gate disabled).
    If enabled, fail-CLOSED: any non-CLOSED status (OPEN, UNKNOWN, missing field,
    bad JSON, timeout, connection error) blocks trades. The user explicitly
    enabled the breaker, so an unreachable service is treated as unsafe.
    """
    if not CIRCUIT_BREAKER_ENABLED:
        return True
    try:
        resp = requests.get(f"{CIRCUIT_BREAKER_URL}/status", timeout=5)
        if resp.status_code != 200:
            logger.warning(
                f"Circuit breaker returned HTTP {resp.status_code} — "
                f"fail-CLOSED, trades blocked."
            )
            return False
        data = resp.json()
        status = (data.get("status") or "UNKNOWN").upper()
        if status != "CLOSED":
            logger.warning(f"Circuit breaker is {status} — trades blocked.")
            return False
        return True
    except Exception as e:
        logger.warning(
            f"Circuit breaker unreachable ({e}) — fail-CLOSED, trades blocked."
        )
        return False


# ── NY Session Check ──────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    now = datetime.now(_tz)
    if now.weekday() >= 5:          # Saturday or Sunday
        return False
    open_time  = now.replace(hour=NY_OPEN_HOUR,  minute=NY_OPEN_MIN,  second=0)
    close_time = now.replace(hour=NY_CLOSE_HOUR, minute=NY_CLOSE_MIN, second=0)
    return open_time <= now < close_time


# ── Pending Signal Expiry ─────────────────────────────────────────────────────

def _expire_pending(max_age_min: int = 90):
    """Remove pending signals older than max_age_min minutes."""
    now = datetime.now(timezone.utc)
    expired = []
    for ticker, r in _pending_trades.items():
        ts = datetime.fromisoformat(r["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if (now - ts).total_seconds() / 60 > max_age_min:
            expired.append(ticker)
    for ticker in expired:
        logger.info(f"[{ticker}] Pending signal expired (>{max_age_min}m old), removing.")
        del _pending_trades[ticker]


# ── VIX Filter ───────────────────────────────────────────────────────────────

def _vix_ok() -> bool:
    """Returns True if VIX is below VIX_MAX (safe to enter new trades)."""
    try:
        df = yf.download(VIX_TICKER, period="2d", progress=False, auto_adjust=True)
        if df.empty:
            logger.warning("VIX data unavailable — allowing trades.")
            return True
        vix = float(df["Close"].squeeze().iloc[-1])
        if vix >= VIX_MAX:
            logger.warning(f"VIX={vix:.1f} >= {VIX_MAX} — new entries blocked.")
            _tg_send(f"⚠️ VIX={vix:.1f} — entradas bloqueadas (umbral {VIX_MAX})")
            return False
        logger.info(f"VIX={vix:.1f} — OK")
        return True
    except Exception as e:
        logger.warning(f"VIX check error ({e}) — allowing trades.")
        return True


# ── Cooldown ─────────────────────────────────────────────────────────────────

def _load_cooldown() -> dict:
    try:
        with open("cooldown.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _in_cooldown(ticker: str) -> bool:
    """Returns True if ticker is in normal or post-SL cooldown."""
    cd = _load_cooldown()
    now = datetime.now(timezone.utc)

    # Post stop-loss cooldown (longer)
    sl_key = f"{ticker}_sl"
    if sl_key in cd:
        sl_dt = datetime.fromisoformat(cd[sl_key]).replace(tzinfo=timezone.utc)
        elapsed_h = (now - sl_dt).total_seconds() / 3600
        if elapsed_h < COOLDOWN_SL_HOURS:
            logger.info(f"[{ticker}] Post-SL cooldown ({elapsed_h:.1f}h < {COOLDOWN_SL_HOURS}h), skipping.")
            return True

    # Normal trade cooldown
    last = cd.get(ticker)
    if not last:
        return False
    last_dt = datetime.fromisoformat(last).replace(tzinfo=timezone.utc)
    elapsed_h = (now - last_dt).total_seconds() / 3600
    if elapsed_h < COOLDOWN_HOURS:
        logger.info(f"[{ticker}] In cooldown ({elapsed_h:.1f}h < {COOLDOWN_HOURS}h), skipping.")
        return True
    return False


def write_cooldown(ticker: str, stop_loss: bool = False):
    """Record trade timestamp. If stop_loss=True, applies longer cooldown."""
    try:
        cd = _load_cooldown()
        now_iso = datetime.now(timezone.utc).isoformat()
        cd[ticker] = now_iso
        if stop_loss:
            cd[f"{ticker}_sl"] = now_iso
            logger.info(f"[{ticker}] Post-SL cooldown recorded ({COOLDOWN_SL_HOURS}h).")
        else:
            logger.info(f"[{ticker}] Cooldown recorded ({COOLDOWN_HOURS}h).")
        with open("cooldown.json", "w") as f:
            json.dump(cd, f, indent=2)
    except Exception as e:
        logger.error(f"Cooldown write error: {e}")


# ── Trailing Stop Monitor ────────────────────────────────────────────────────

def _load_trailing_state() -> dict:
    try:
        with open(_TRAILING_STATE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_trailing_state(state: dict):
    try:
        with open(_TRAILING_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Trailing state write error: {e}")


# ── Centralized Position Close ────────────────────────────────────────────────

VALID_CLOSE_REASONS = {
    "manual_sell",       # /api/sell/<ticker>
    "auto_sell_signal",  # AUTO_TRADE + score SELL
    "trailing_stop",     # client-side trailing trigger (scanner.py)
    "take_profit",       # client-side TP backup (scanner.py)
    "emergency_stop",    # /api/emergency-stop
    "close_non_etf",     # /api/close-non-etf
    "bracket_sl",        # bracket stop_loss leg filled by Alpaca server-side
    "bracket_tp",        # bracket take_profit leg filled by Alpaca server-side
}


def close_position(
    ticker: str,
    reason: str,
    qty: float | None = None,
    stop_loss: bool = False,
    extra_msg: str | None = None,
) -> dict:
    """Centralized position close. Single owner of: broker.sell, fill confirmation,
    cooldown write, trailing_state cleanup, and Telegram alert.

    Args:
        ticker:    symbol to close.
        reason:    one of VALID_CLOSE_REASONS.
        qty:       shares to sell. None = close entire position.
        stop_loss: True applies the extended SL cooldown (COOLDOWN_SL_HOURS).
        extra_msg: optional context appended to the Telegram alert.

    Returns:
        {"success": bool, "fill_price": float|None,
         "fill_qty": float|None, "error_msg": str|None}

    Contract:
        - State (cooldown + trailing_state) is mutated ONLY after broker.sell
          returns status=ok. If the order submit fails, local state is left
          intact so the next scan re-evaluates correctly.
        - Telegram is sent in BOTH success and error paths.
        - Fill price/qty are confirmed by polling get_order_status for up to
          5 seconds. For market orders this is near-instant in paper and
          typically <1s in live. If polling times out, the close is still
          considered successful (broker.sell returned ok) but fill_price/qty
          come back as None.
    """
    if reason not in VALID_CLOSE_REASONS:
        raise ValueError(
            f"close_position: invalid reason '{reason}'. "
            f"Must be one of {sorted(VALID_CLOSE_REASONS)}."
        )

    import broker as _broker

    logger.info(
        f"[{ticker}] close_position: reason={reason} qty={qty} stop_loss={stop_loss}"
    )

    sell_result = _broker.sell(ticker, qty)
    if sell_result.get("status") != "ok":
        err = sell_result.get("message") or "unknown broker error"
        logger.error(f"[{ticker}] close_position FAILED ({reason}): {err}")
        _tg_send(
            f"⚠️ <b>{ticker}</b> cierre FALLÓ\n"
            f"Razón: <code>{reason}</code>\n"
            f"Error: {err}"
        )
        return {
            "success":    False,
            "fill_price": None,
            "fill_qty":   None,
            "error_msg":  err,
        }

    # Poll for fill confirmation (paper: instant; live market: ~1s).
    fill_price: float | None = None
    fill_qty:   float | None = None
    order_id = sell_result.get("order_id")
    if order_id:
        for _ in range(10):  # 10 × 0.5s = 5s budget
            try:
                status = _broker.get_order_status(order_id)
                fp = status.get("filled_avg_price")
                if fp:
                    fill_price = float(fp)
                    fq = status.get("filled_qty")
                    fill_qty = float(fq) if fq else None
                    break
            except Exception as poll_err:  # noqa: BLE001 - polling, broker can flake
                logger.debug(f"[{ticker}] fill poll error: {poll_err}")
            time.sleep(0.5)

    # State cleanup — only reached if broker.sell succeeded.
    write_cooldown(ticker, stop_loss=stop_loss)
    state = _load_trailing_state()
    if ticker in state:
        del state[ticker]
        _save_trailing_state(state)

    # Telegram alert
    fill_str = f"${fill_price:.2f}" if fill_price is not None else "pendiente"
    if fill_qty is not None:
        qty_str = f"{fill_qty:g}"
    elif qty is not None:
        qty_str = f"{qty:g}"
    else:
        qty_str = "toda la posición"
    msg = (
        f"🔴 <b>{ticker}</b> posición cerrada\n"
        f"Razón: <code>{reason}</code>\n"
        f"Cantidad: {qty_str}  |  Fill: {fill_str}"
    )
    if extra_msg:
        msg += f"\n{extra_msg}"
    _tg_send(msg)

    logger.info(
        f"[{ticker}] close_position OK ({reason}): "
        f"fill_price={fill_price} fill_qty={fill_qty} order_id={order_id}"
    )

    return {
        "success":    True,
        "fill_price": fill_price,
        "fill_qty":   fill_qty,
        "error_msg":  None,
    }


def _monitor_positions():
    """
    Check all open positions for trailing stop and take-profit.
    Runs each scan cycle alongside the normal entry scan.

    Logic:
    - Track high-water mark per position in trailing_state.json
    - If gain >= TRAILING_ACTIVATE_PCT (6%): activate trailing mode
    - In trailing mode, if price drops TRAILING_STOP_PCT (3%) from high → sell
    - This is a scanner-side layer; Alpaca bracket orders (5% SL, 12% TP) remain
      as a server-side safety net. The trailing stop typically fires first when
      the position has recovered enough gain.
    """
    import broker as _broker

    try:
        positions = _broker.get_positions()
    except Exception as e:
        logger.warning(f"Trailing monitor: could not fetch positions ({e})")
        return

    if not positions:
        return

    state = _load_trailing_state()
    state_changed = False

    for ticker, pos in positions.items():
        entry  = pos.get("avg_buy_price", 0.0)
        current = pos.get("current_price", 0.0)
        if entry <= 0 or current <= 0:
            continue

        gain_pct = (current - entry) / entry

        # Update high-water mark
        ticker_state = state.setdefault(ticker, {"high_price": entry, "trailing_active": False})
        if current > ticker_state["high_price"]:
            ticker_state["high_price"] = current
            state_changed = True

        high = ticker_state["high_price"]

        # Activate trailing stop once gain threshold is crossed
        if not ticker_state["trailing_active"] and gain_pct >= TRAILING_ACTIVATE_PCT:
            ticker_state["trailing_active"] = True
            state_changed = True
            logger.info(f"[{ticker}] Trailing stop ACTIVATED at {gain_pct*100:.1f}% gain (high=${high:.2f})")
            _tg_send(
                f"📈 <b>{ticker}</b> trailing stop activado\n"
                f"Ganancia: +{gain_pct*100:.1f}%  |  Máximo: ${high:.2f}\n"
                f"Stop activo: ${high*(1-TRAILING_STOP_PCT):.2f} ({TRAILING_STOP_PCT*100:.0f}% bajo el máximo)"
            )

        # Check trailing stop trigger
        if ticker_state["trailing_active"]:
            trailing_sl = high * (1 - TRAILING_STOP_PCT)
            if current <= trailing_sl:
                logger.info(
                    f"[{ticker}] TRAILING STOP triggered: "
                    f"price=${current:.2f} <= ${trailing_sl:.2f} (high=${high:.2f})"
                )
                # Persist any in-memory updates before close_position writes its
                # own cleanup, so we don't lose high_price bumps from this loop.
                if state_changed:
                    _save_trailing_state(state)
                    state_changed = False
                close_position(
                    ticker,
                    reason="trailing_stop",
                    stop_loss=True,
                    extra_msg=(
                        f"Precio: ${current:.2f}  |  Stop: ${trailing_sl:.2f}  |  "
                        f"Máximo: ${high:.2f}\n"
                        f"Ganancia desde entrada: {gain_pct*100:+.1f}%"
                    ),
                )
                # close_position has rewritten the on-disk state on success.
                # Reload so the rest of this loop sees the truth.
                state = _load_trailing_state()
                continue

        # Take-profit check (backup for cases where Alpaca bracket fails)
        if gain_pct >= TAKE_PROFIT_PCT and not ticker_state["trailing_active"]:
            logger.info(
                f"[{ticker}] TAKE-PROFIT reached: "
                f"{gain_pct*100:.1f}% >= {TAKE_PROFIT_PCT*100:.0f}%"
            )
            if state_changed:
                _save_trailing_state(state)
                state_changed = False
            close_position(
                ticker,
                reason="take_profit",
                extra_msg=(
                    f"Ganancia: +{gain_pct*100:.1f}%  "
                    f"(objetivo: {TAKE_PROFIT_PCT*100:.0f}%)"
                ),
            )
            state = _load_trailing_state()
            continue

    # Defensive cleanup — sweeps trailing_state entries for tickers that are
    # no longer in `positions`. This is the safety net for exits the scanner
    # never observed locally: Alpaca bracket SL/TP legs filling server-side,
    # manual sells from the web UI, or any future close path that forgot to
    # call close_position(). close_position() itself already purges its
    # ticker, so this sweep is purely belt-and-suspenders.
    for ticker in list(state.keys()):
        if ticker not in positions:
            del state[ticker]
            state_changed = True

    if state_changed:
        _save_trailing_state(state)


# ── Scan Cycle ────────────────────────────────────────────────────────────────

def run_scan() -> list[dict]:
    """Execute one full scan cycle. Returns list of signal dicts."""
    if not _circuit_breaker_ok():
        logger.info("Scan aborted — circuit breaker active.")
        return []

    if not _vix_ok():
        logger.info("Scan aborted — VIX too high.")
        return []

    _monitor_positions()   # trailing stop + TP check on existing positions
    _expire_pending()

    logger.info("=== ETF SCAN STARTED ===")
    results = score_all_etfs()

    # Fetch current positions — also syncs in-memory state after bot restarts
    import broker as _broker
    try:
        current_positions = _broker.get_positions()
        # Remove pending signals for tickers already in portfolio (handles restart sync)
        for ticker in list(_pending_trades.keys()):
            if ticker in current_positions:
                logger.info(f"[{ticker}] Position detected on restart sync — clearing pending signal.")
                del _pending_trades[ticker]
    except Exception as e:
        logger.warning(f"Could not fetch positions ({e}), skipping position check.")
        current_positions = {}

    open_count = len(current_positions) + len(_pending_trades)

    signals = []
    for r in results:
        ticker = r["ticker"]

        # ── Auto-SELL: ETF en cartera con señal SELL ──────────────────────────
        # NOTE: this path goes through close_position() so trailing_state and
        # cooldown are cleaned consistently. Telegram alerts (success + error)
        # are emitted by close_position; the score context is appended via
        # extra_msg.
        if AUTO_TRADE and r["signal"] == "SELL" and ticker in current_positions:
            logger.info(f"[{ticker}] AUTO-SELL: score={r['score']:.1f} (señal SELL)")
            close_position(
                ticker,
                reason="auto_sell_signal",
                extra_msg=(
                    f"Score: {r['score']:.1f}/100 (señal SELL)\n"
                    f"Precio actual: ${r['current_price']:.2f}"
                ),
            )
            continue

        min_score = MIN_SCORE_BY_TICKER.get(ticker, MIN_SCORE_BUY)
        if r["signal"] != "BUY" or r["score"] < min_score:
            continue

        if ticker in _pending_trades:
            logger.info(f"[{ticker}] Already pending approval, skipping duplicate signal.")
            continue

        if ticker in current_positions:
            logger.info(f"[{ticker}] Already in portfolio, skipping signal.")
            continue

        if open_count >= MAX_OPEN_POSITIONS:
            logger.info(f"Max open positions ({MAX_OPEN_POSITIONS}) reached, skipping {ticker}.")
            continue

        if _in_cooldown(ticker):
            continue

        # ── News sentiment soft gate ──────────────────────────────────────────
        news_sent = get_news_sentiment(ticker)
        r["news_sentiment"] = news_sent
        if news_sent["compound"] <= -0.2:
            logger.info(
                f"[{ticker}] Blocked by negative news sentiment: "
                f"{news_sent['label']} ({news_sent['compound']:+.3f}, "
                f"{news_sent['headline_count']} headlines)"
            )
            _tg_send(
                f"📰 <b>{ticker}</b> señal bloqueada por noticias negativas\n"
                f"Sentimiento: {news_sent['label']} ({news_sent['compound']:+.3f})\n"
                f"Score técnico+ML: {r['score']:.1f}/100 (era BUY)"
            )
            continue

        r["dynamic_sl"]   = _calc_dynamic_sl(r)
        r["dynamic_tp"]   = _calc_dynamic_tp(r)
        _sent = {
            "fear_greed_score": r.get("sentiment", {}).get("fear_greed", 50),
        }
        r["position_pct"] = _calc_position_pct(_sent)

        if AUTO_TRADE:
            # ── Ejecución automática — sin aprobación humana ──────────────────
            buy_result = _broker.buy(
                ticker,
                stop_loss_pct=r["dynamic_sl"],
                take_profit_pct=r["dynamic_tp"],
                position_pct=r["position_pct"],
            )
            if buy_result.get("status") == "ok":
                write_cooldown(ticker)
                adx_label = r.get("technical", {}).get("components", {}).get("adx_label", "?")
                _tg_send(
                    f"✅ <b>AUTO-BUY {ticker}</b>\n"
                    f"Score : {r['score']:.1f}/100\n"
                    f"Técnico: {r['technical']['score']:.1f} | ML: {r['ml']['score']:.1f}\n"
                    f"ADX   : {r.get('adx', 0):.1f} ({adx_label})\n"
                    f"SL    : {r['dynamic_sl']*100:.1f}% | TP: {r['dynamic_tp']*100:.0f}%\n"
                    f"Posición: {r['position_pct']*100:.0f}% capital\n"
                    f"Precio: ${r['current_price']:.2f}"
                )
                signals.append(r)
                open_count += 1
                logger.info(f"[{ticker}] AUTO-BUY ejecutado: {buy_result}")
            else:
                logger.error(f"[{ticker}] AUTO-BUY fallido: {buy_result}")
                _tg_send(f"❌ <b>AUTO-BUY fallido {ticker}</b>: {buy_result.get('message','')}")
        else:
            # ── Flujo original: esperar aprobación por Telegram ───────────────
            _pending_trades[ticker] = r
            notify_signal(r)
            signals.append(r)
            open_count += 1

    logger.info(f"Scan complete — {len(signals)} new BUY signals.")
    return results          # return full list for /api/scan endpoint


# ── Daily Audit Check ─────────────────────────────────────────────────────────

def _run_daily_audit():
    """Runs audit once per day at market open. Sends Telegram alert on result."""
    global _last_audit_date
    today = datetime.now(_tz).strftime("%Y-%m-%d")
    if today == _last_audit_date:
        return
    _last_audit_date = today
    logger.info("Running daily live-readiness audit...")
    try:
        from check_audit import run_audit
        result = run_audit(verbose=False)
        n_ok    = result["passed"]
        total   = result["total"]
        n_fail  = result["failed"]

        if result["ready"]:
            msg = (
                "<b>El bot ETF esta listo para crear cuenta real</b>\n\n"
                f"Todos los criterios pasados ({n_ok}/{total})\n\n"
                "Pasos para activar:\n"
                "1. Genera API keys LIVE en app.alpaca.markets\n"
                "2. Cambia ALPACA_BASE_URL en .env a la URL live\n"
                "3. Reinicia el bot\n"
                "4. Empieza con MAX_POSITION_PCT=0.002 ($200/trade)"
            )
            logger.info("AUDIT PASSED — bot ready for live trading!")
        else:
            failed_txt = "\n".join(f"  XX {i}" for i in result["failed_items"])
            warn_txt   = "\n".join(f"  ?? {i}" for i in result["warn_items"])
            msg = (
                f"<b>Auditoria diaria — {today}</b>\n"
                f"Estado: {n_ok}/{total} criterios OK\n\n"
            )
            if result["failed_items"]:
                msg += f"Pendiente:\n{failed_txt}\n"
            if result["warn_items"]:
                msg += f"\nAdvertencias:\n{warn_txt}"
            logger.info(f"Daily audit: {n_ok}/{total} OK, {n_fail} failed.")

        _tg_send(msg)
    except Exception as e:
        logger.error(f"Daily audit error: {e}")


# ── Background Loop ───────────────────────────────────────────────────────────

def _loop():
    global _running
    logger.info("Scanner loop started.")
    while _running:
        if _is_market_open():
            try:
                _run_daily_audit()   # runs once per day at market open
                run_scan()
            except Exception as e:
                logger.error(f"Scan error: {e}")
        else:
            logger.info("Market closed — scanner sleeping.")

        # Sleep in 60-second chunks so we can stop cleanly
        for _ in range(SCAN_INTERVAL_MIN * 60 // 60):
            if not _running:
                break
            time.sleep(60)

    logger.info("Scanner loop stopped.")


def start_scanner():
    global _scan_thread, _running
    if _running:
        return
    _running = True
    _scan_thread = threading.Thread(target=_loop, daemon=True, name="ETFScanner")
    _scan_thread.start()


def stop_scanner():
    global _running
    _running = False


# ── Pending Trades API ────────────────────────────────────────────────────────

def get_pending() -> dict:
    return dict(_pending_trades)


def approve_trade(ticker: str) -> dict | None:
    return _pending_trades.pop(ticker, None)


def reject_trade(ticker: str) -> bool:
    if ticker in _pending_trades:
        del _pending_trades[ticker]
        return True
    return False


# ── Startup ───────────────────────────────────────────────────────────────────

def initialize():
    from config import RETRAIN_ON_STARTUP
    logger.info("Training / loading ML models...")
    train_all(retrain=RETRAIN_ON_STARTUP)
    logger.info("Models ready.")
