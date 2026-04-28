# broker/alpaca.py — Alpaca Markets broker (paper + live)
import logging
from alpaca.trading.client   import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, GetOrdersRequest,
    TakeProfitRequest, StopLossRequest,
)
from alpaca.trading.enums    import OrderSide, TimeInForce, QueryOrderStatus, OrderClass
from alpaca.data.historical  import StockHistoricalDataClient
from alpaca.data.requests    import StockLatestQuoteRequest

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    MAX_POSITION_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT, TAKE_PROFIT_HIGH
)

logger = logging.getLogger(__name__)

_paper = "paper" in ALPACA_BASE_URL.lower()

_client: TradingClient | None = None
_data_client: StockHistoricalDataClient | None = None


def _get_client() -> TradingClient:
    global _client
    if _client is None:
        _client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=_paper,
        )
    return _client


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )
    return _data_client


# ── Portfolio Helpers ─────────────────────────────────────────────────────────

def get_portfolio_value() -> float:
    try:
        acct = _get_client().get_account()
        return float(acct.equity)
    except Exception as e:
        logger.error(f"Alpaca get_portfolio_value error: {e}")
        return 0.0


def get_positions() -> dict[str, dict]:
    try:
        raw = _get_client().get_all_positions()
        return {
            p.symbol: {
                "quantity":       float(p.qty),
                "avg_buy_price":  float(p.avg_entry_price),
                "current_price":  float(p.current_price),
                "unrealized_pnl": float(p.unrealized_pl),
            }
            for p in raw
        }
    except Exception as e:
        logger.error(f"Alpaca get_positions error: {e}")
        return {}


def get_quote(ticker: str) -> float:
    try:
        req    = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        data   = _get_data_client().get_stock_latest_quote(req)
        quote  = data[ticker]
        # mid-price of bid/ask
        return float((quote.bid_price + quote.ask_price) / 2)
    except Exception as e:
        logger.error(f"Alpaca get_quote({ticker}) error: {e}")
        return 0.0


# ── Order Execution ───────────────────────────────────────────────────────────

def _calc_notional(portfolio_value: float, position_pct: float | None = None) -> float:
    """Dollar amount = position_pct (or MAX_POSITION_PCT) * portfolio."""
    pct = position_pct if position_pct is not None else MAX_POSITION_PCT
    return round(portfolio_value * pct, 2)


def buy(ticker: str, portfolio_value: float | None = None, stop_loss_pct: float | None = None, take_profit_pct: float | None = None, position_pct: float | None = None) -> dict:
    """
    Market buy with server-side bracket order (stop-loss + take-profit on Alpaca).
    stop_loss_pct: dynamic SL from ATR (overrides STOP_LOSS_PCT if provided).
    Falls back to simple market order if bracket order fails.
    """
    try:
        pv       = portfolio_value or get_portfolio_value()
        notional = _calc_notional(pv, position_pct)
        if notional <= 0:
            return {"status": "error", "message": "Zero notional"}

        sl_pct = stop_loss_pct if stop_loss_pct is not None else STOP_LOSS_PCT
        tp_pct = take_profit_pct if take_profit_pct is not None else TAKE_PROFIT_PCT

        # Get current price to calculate bracket levels
        quote = get_quote(ticker)
        if quote <= 0:
            logger.warning(f"[{ticker}] No se pudo obtener precio — orden SIN bracket (sin SL/TP)")
        if quote > 0:
            sl_price = round(quote * (1 - sl_pct), 2)
            tp_price = round(quote * (1 + tp_pct), 2)
            try:
                req = MarketOrderRequest(
                    symbol=ticker,
                    notional=notional,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    stop_loss=StopLossRequest(stop_price=sl_price),
                    take_profit=TakeProfitRequest(limit_price=tp_price),
                )
                order = _get_client().submit_order(req)
                logger.info(
                    f"[{ticker}] Alpaca BRACKET BUY ${notional:.2f} "
                    f"SL={sl_price} ({sl_pct*100:.1f}%) TP={tp_price} — order_id={order.id}"
                )
                return {
                    "status":    "ok",
                    "ticker":    ticker,
                    "notional":  notional,
                    "order_id":  str(order.id),
                    "sl_price":  sl_price,
                    "sl_pct":    round(sl_pct * 100, 2),
                    "tp_price":  tp_price,
                    "bracket":   True,
                }
            except Exception as bracket_err:
                logger.warning(f"[{ticker}] Bracket order failed ({bracket_err}), falling back to market order.")

        # Fallback: simple market order (paper trading or bracket not supported)
        req = MarketOrderRequest(
            symbol=ticker,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = _get_client().submit_order(req)
        logger.info(f"[{ticker}] Alpaca BUY ${notional:.2f} — order_id={order.id}")
        return {
            "status":   "ok",
            "ticker":   ticker,
            "notional": notional,
            "order_id": str(order.id),
            "bracket":  False,
        }
    except Exception as e:
        logger.error(f"[{ticker}] Alpaca buy() error: {e}")
        return {"status": "error", "message": str(e)}


def sell(ticker: str, quantity: float | None = None) -> dict:
    """Market sell. If quantity is None, closes entire position."""
    try:
        if quantity is None:
            result = _get_client().close_position(ticker)
            logger.info(f"[{ticker}] Alpaca CLOSE POSITION — {result}")
            return {"status": "ok", "ticker": ticker, "order_id": str(result.id)}

        req = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = _get_client().submit_order(req)
        logger.info(f"[{ticker}] Alpaca SELL {quantity} shares — order_id={order.id}")
        return {"status": "ok", "ticker": ticker, "shares": quantity, "order_id": str(order.id)}
    except Exception as e:
        logger.error(f"[{ticker}] Alpaca sell() error: {e}")
        return {"status": "error", "message": str(e)}


def cancel_order(order_id: str) -> dict:
    try:
        _get_client().cancel_order_by_id(order_id)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_order_status(order_id: str) -> dict:
    try:
        order = _get_client().get_order_by_id(order_id)
        return {
            "id":               str(order.id),
            "status":           str(order.status).split(".")[-1].lower(),
            "symbol":           order.symbol,
            "qty":              str(order.qty) if order.qty is not None else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "filled_qty":       float(order.filled_qty) if order.filled_qty else 0.0,
            "filled_at":        order.filled_at.isoformat() if order.filled_at else None,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_open_orders() -> list[dict]:
    try:
        req    = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = _get_client().get_orders(req)
        return [
            {"id": str(o.id), "symbol": o.symbol, "qty": str(o.qty), "side": str(o.side)}
            for o in orders
        ]
    except Exception as e:
        logger.error(f"Alpaca list_open_orders error: {e}")
        return []


def list_closed_orders(days: int = 90, limit: int = 500) -> list[dict]:
    """Returns filled/closed orders in the last N days, sorted oldest → newest."""
    from datetime import datetime, timedelta, timezone
    try:
        after = datetime.now(timezone.utc) - timedelta(days=days)
        req   = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=after,
            limit=limit,
            direction="asc",
        )
        orders = _get_client().get_orders(req)
        return [
            {
                "id":                str(o.id),
                "symbol":            o.symbol,
                "side":              str(o.side).split(".")[-1].lower(),
                "qty":               float(o.filled_qty) if o.filled_qty else 0.0,
                "filled_avg_price":  float(o.filled_avg_price) if o.filled_avg_price else 0.0,
                "filled_at":         o.filled_at.isoformat() if o.filled_at else None,
                "status":            str(o.status).split(".")[-1].lower(),
                "order_type":        str(o.order_type).split(".")[-1].lower() if o.order_type else None,
            }
            for o in orders
            if o.filled_qty and float(o.filled_qty) > 0
        ]
    except Exception as e:
        logger.error(f"Alpaca list_closed_orders error: {e}")
        return []
