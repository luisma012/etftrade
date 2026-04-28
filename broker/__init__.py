# broker/__init__.py — Broker factory
from config import BROKER

if BROKER == "alpaca":
    from broker.alpaca import (
        buy, sell, cancel_order, get_order_status,
        get_portfolio_value, get_positions, get_quote,
        list_open_orders, list_closed_orders,
    )
else:
    from broker.robinhood import (
        buy, sell, cancel_order, get_order_status,
        get_portfolio_value, get_positions, get_quote,
        login, logout,
    )
    login()
