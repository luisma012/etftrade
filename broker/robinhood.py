# broker/robinhood.py — Robinhood broker via robin_stocks
import logging
import robin_stocks.robinhood as rh

from config import (
    ROBINHOOD_USERNAME, ROBINHOOD_PASSWORD, ROBINHOOD_MFA_CODE,
    MAX_POSITION_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT
)

logger = logging.getLogger(__name__)

_logged_in = False


# ── Auth ──────────────────────────────────────────────────────────────────────

def login():
    global _logged_in
    if _logged_in:
        return True
    try:
        kwargs = {
            "username":    ROBINHOOD_USERNAME,
            "password":    ROBINHOOD_PASSWORD,
            "store_session": True,
            "by_sms":      False,
        }
        if ROBINHOOD_MFA_CODE:
            kwargs["mfa_code"] = ROBINHOOD_MFA_CODE

        rh.login(**kwargs)
        _logged_in = True
        logger.info("Robinhood login successful.")
        return True
    except Exception as e:
        logger.error(f"Robinhood login failed: {e}")
        return False


def logout():
    global _logged_in
    try:
        rh.logout()
        _logged_in = False
    except Exception:
        pass


def _ensure_login():
    if not _logged_in:
        login()


# ── Portfolio Helpers ─────────────────────────────────────────────────────────

def get_portfolio_value() -> float:
    _ensure_login()
    try:
        profile = rh.profiles.load_portfolio_profile()
        return float(profile.get("equity", 0))
    except Exception as e:
        logger.error(f"get_portfolio_value error: {e}")
        return 0.0


def get_positions() -> dict[str, dict]:
    """Returns {ticker: {quantity, avg_buy_price, current_price, unrealized_pnl}}"""
    _ensure_login()
    try:
        raw      = rh.account.build_holdings()
        positions = {}
        for ticker, data in raw.items():
            positions[ticker] = {
                "quantity":        float(data.get("quantity",        0)),
                "avg_buy_price":   float(data.get("average_buy_price", 0)),
                "current_price":   float(data.get("price",           0)),
                "unrealized_pnl":  float(data.get("equity_change",   0)),
            }
        return positions
    except Exception as e:
        logger.error(f"get_positions error: {e}")
        return {}


def get_quote(ticker: str) -> float:
    _ensure_login()
    try:
        data  = rh.stocks.get_latest_price(ticker)
        return float(data[0])
    except Exception as e:
        logger.error(f"get_quote({ticker}) error: {e}")
        return 0.0


# ── Order Execution ───────────────────────────────────────────────────────────

def _calc_shares(ticker: str, portfolio_value: float) -> float:
    price = get_quote(ticker)
    if price <= 0:
        return 0.0
    max_spend = portfolio_value * MAX_POSITION_PCT
    shares    = max_spend / price
    return round(shares, 6)          # fractional shares supported on Robinhood


def buy(ticker: str, portfolio_value: float | None = None) -> dict:
    """Market buy with auto position sizing."""
    _ensure_login()
    try:
        pv     = portfolio_value or get_portfolio_value()
        shares = _calc_shares(ticker, pv)
        if shares <= 0:
            return {"status": "error", "message": "Cannot determine shares"}

        order = rh.orders.order_buy_fractional_by_quantity(
            symbol=ticker,
            quantity=shares,
            timeInForce="gfd",
        )
        logger.info(f"[{ticker}] BUY order placed: {shares:.4f} shares — {order}")
        return {"status": "ok", "ticker": ticker, "shares": shares, "order": order}
    except Exception as e:
        logger.error(f"[{ticker}] buy() error: {e}")
        return {"status": "error", "message": str(e)}


def sell(ticker: str, quantity: float | None = None) -> dict:
    """Market sell. Sells all shares if quantity not specified."""
    _ensure_login()
    try:
        if quantity is None:
            positions = get_positions()
            quantity  = positions.get(ticker, {}).get("quantity", 0)
        if quantity <= 0:
            return {"status": "error", "message": f"No position in {ticker}"}

        order = rh.orders.order_sell_fractional_by_quantity(
            symbol=ticker,
            quantity=quantity,
            timeInForce="gfd",
        )
        logger.info(f"[{ticker}] SELL order placed: {quantity:.4f} shares — {order}")
        return {"status": "ok", "ticker": ticker, "shares": quantity, "order": order}
    except Exception as e:
        logger.error(f"[{ticker}] sell() error: {e}")
        return {"status": "error", "message": str(e)}


def cancel_order(order_id: str) -> dict:
    _ensure_login()
    try:
        result = rh.orders.cancel_stock_order(order_id)
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_order_status(order_id: str) -> dict:
    _ensure_login()
    try:
        info = rh.orders.get_stock_order_info(order_id)
        avg = info.get("average_price")
        cum = info.get("cumulative_quantity")
        return {
            "id":               str(info.get("id", order_id)),
            "status":           str(info.get("state", "unknown")).lower(),
            "symbol":           info.get("symbol"),
            "qty":              info.get("quantity"),
            "filled_avg_price": float(avg) if avg else None,
            "filled_qty":       float(cum) if cum else 0.0,
            "filled_at":        info.get("last_transaction_at"),
            "_raw":             info,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
