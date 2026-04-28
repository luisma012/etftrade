# app.py — Flask API — ETF Trader (no LLM) — port 5010
import logging
import os
from datetime import datetime
from flask import Flask, jsonify, request, abort, render_template

from config import FLASK_HOST, FLASK_PORT, DEBUG, ETF_UNIVERSE, MIN_SCORE_BUY
from scanner import (
    initialize, start_scanner, stop_scanner,
    run_scan, get_pending, approve_trade, reject_trade,
    write_cooldown, _is_market_open, close_position,
)
import broker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
_start_time = datetime.utcnow()


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    return jsonify({
        "status":       "ok",
        "service":      "etf-trader",
        "port":         FLASK_PORT,
        "market_open":  _is_market_open(),
        "uptime_sec":   int(uptime),
        "timestamp":    datetime.utcnow().isoformat(),
    })


# ── Manual Scan ───────────────────────────────────────────────────────────────

@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Trigger an immediate scan regardless of schedule."""
    try:
        results = run_scan()
        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        logger.error(f"/api/scan error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/scores", methods=["GET"])
def api_scores():
    """Run a fresh scoring pass (no trade signals) and return all scores."""
    from scorer import score_all_etfs
    results = score_all_etfs()
    return jsonify({"status": "ok", "scores": results})


# ── Pending Trades ────────────────────────────────────────────────────────────

@app.route("/api/pending", methods=["GET"])
def api_pending():
    return jsonify({"pending": get_pending()})


@app.route("/api/approve/<ticker>", methods=["POST"])
def api_approve(ticker: str):
    ticker = ticker.upper()
    if ticker not in ETF_UNIVERSE:
        abort(400, f"{ticker} not in ETF universe")

    trade = approve_trade(ticker)
    if not trade:
        return jsonify({"status": "error", "message": f"No pending trade for {ticker}"}), 404

    dynamic_sl  = trade.get("dynamic_sl")
    dynamic_tp  = trade.get("dynamic_tp")
    result = broker.buy(ticker, stop_loss_pct=dynamic_sl, take_profit_pct=dynamic_tp)
    if result.get("status") == "ok":
        write_cooldown(ticker)
    logger.info(f"APPROVED & EXECUTED: {ticker} SL={dynamic_sl} TP={dynamic_tp} — {result}")
    return jsonify({"status": "ok", "ticker": ticker, "broker_result": result})


@app.route("/api/reject/<ticker>", methods=["POST"])
def api_reject(ticker: str):
    ticker = ticker.upper()
    if reject_trade(ticker):
        return jsonify({"status": "ok", "message": f"{ticker} trade rejected"})
    return jsonify({"status": "error", "message": f"No pending trade for {ticker}"}), 404


# ── Telegram Webhook (approve / reject via chat commands) ────────────────────

@app.route("/telegram/webhook", methods=["POST"])
def telegram_webhook():
    """
    Handles incoming Telegram bot messages.
    Commands: /approve_SPY  /reject_SPY
    Set webhook URL in BotFather: https://your-host/telegram/webhook
    """
    data = request.json or {}
    message = data.get("message", {})
    text    = message.get("text", "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))

    from config import TELEGRAM_CHAT_ID
    if chat_id != str(TELEGRAM_CHAT_ID):
        logger.warning(f"Telegram message from unknown chat_id: {chat_id}")
        return jsonify({"ok": True})

    if text.startswith("/approve_"):
        ticker = text.split("_", 1)[1].upper()
        trade  = approve_trade(ticker)
        if trade:
            dynamic_sl = trade.get("dynamic_sl")
            result = broker.buy(ticker, stop_loss_pct=dynamic_sl)
            if result.get("status") == "ok":
                write_cooldown(ticker)
                sl_info = f"SL={result.get('sl_pct', '')}%" if result.get('sl_pct') else ""
                _tg_reply(chat_id, f"✅ {ticker} BUY executed. {sl_info}")
            else:
                _tg_reply(chat_id, f"❌ {ticker} BUY failed: {result.get('message','')}")
        else:
            _tg_reply(chat_id, f"❌ No pending trade for {ticker}")

    elif text.startswith("/reject_"):
        ticker = text.split("_", 1)[1].upper()
        if reject_trade(ticker):
            _tg_reply(chat_id, f"🚫 {ticker} trade rejected.")
        else:
            _tg_reply(chat_id, f"❌ No pending trade for {ticker}")

    elif text == "/pending":
        pending = get_pending()
        if pending:
            msg = "Pending trades:\n" + "\n".join(
                f"• {t}: score={d['score']:.1f}" for t, d in pending.items()
            )
        else:
            msg = "No pending trades."
        _tg_reply(chat_id, msg)

    elif text == "/status":
        _tg_reply(chat_id, f"ETF Trader running. Market open: {_is_market_open()}")

    return jsonify({"ok": True})


def _tg_reply(chat_id: str, text: str):
    import requests as req
    from config import TELEGRAM_TOKEN
    if not TELEGRAM_TOKEN:
        return
    try:
        req.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=5,
        )
    except Exception as e:
        logger.error(f"Telegram reply error: {e}")


# ── Broker / Portfolio ────────────────────────────────────────────────────────

@app.route("/api/portfolio", methods=["GET"])
def api_portfolio():
    try:
        value     = broker.get_portfolio_value()
        positions = broker.get_positions()
        return jsonify({"portfolio_value": value, "positions": positions})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/trade-history", methods=["GET"])
def api_trade_history():
    """Historial de trades cerrados desde Alpaca. ?days=30 (default 90)."""
    days = int(request.args.get("days", 90))
    try:
        orders = broker.list_closed_orders(days=days)

        # Pair BUYs with SELLs per ticker (FIFO) to build round-trips
        from collections import defaultdict
        queues     = defaultdict(list)   # ticker → list of open BUY lots
        roundtrips = []

        for o in orders:
            if o["side"] == "buy":
                queues[o["symbol"]].append(o)
            elif o["side"] == "sell":
                remaining_qty = o["qty"]
                while remaining_qty > 1e-6 and queues[o["symbol"]]:
                    lot      = queues[o["symbol"]][0]
                    matched  = min(lot["qty"], remaining_qty)
                    entry    = lot["filled_avg_price"]
                    exit_p   = o["filled_avg_price"]
                    pnl      = (exit_p - entry) * matched
                    pnl_pct  = (exit_p - entry) / entry * 100 if entry else 0

                    from datetime import datetime as _dt
                    hold_days = None
                    if lot["filled_at"] and o["filled_at"]:
                        d1 = _dt.fromisoformat(lot["filled_at"].replace("Z", "+00:00"))
                        d2 = _dt.fromisoformat(o["filled_at"].replace("Z", "+00:00"))
                        hold_days = round((d2 - d1).total_seconds() / 86400, 1)

                    roundtrips.append({
                        "ticker":       o["symbol"],
                        "entry_date":   lot["filled_at"],
                        "exit_date":    o["filled_at"],
                        "entry_price":  round(entry, 4),
                        "exit_price":   round(exit_p, 4),
                        "qty":          round(matched, 4),
                        "pnl":          round(pnl, 2),
                        "pnl_pct":      round(pnl_pct, 2),
                        "hold_days":    hold_days,
                    })
                    lot["qty"]    -= matched
                    remaining_qty -= matched
                    if lot["qty"] <= 1e-6:
                        queues[o["symbol"]].pop(0)

        # Stats
        wins   = [r for r in roundtrips if r["pnl"] > 0]
        losses = [r for r in roundtrips if r["pnl"] < 0]
        gp     = sum(r["pnl"] for r in wins)
        gl     = abs(sum(r["pnl"] for r in losses))

        stats = {
            "total_trades":  len(roundtrips),
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      round(len(wins) / len(roundtrips) * 100, 1) if roundtrips else 0,
            "net_pnl":       round(sum(r["pnl"] for r in roundtrips), 2),
            "gross_profit":  round(gp, 2),
            "gross_loss":    round(gl, 2),
            "profit_factor": round(gp / gl, 2) if gl > 0 else None,
            "avg_win":       round(gp / len(wins), 2) if wins else 0,
            "avg_loss":      round(gl / len(losses), 2) if losses else 0,
        }

        # Open positions still in queue (not yet closed)
        open_lots = []
        for ticker, lots in queues.items():
            for lot in lots:
                if lot["qty"] > 1e-6:
                    open_lots.append({
                        "ticker":      ticker,
                        "entry_date":  lot["filled_at"],
                        "entry_price": round(lot["filled_avg_price"], 4),
                        "qty":         round(lot["qty"], 4),
                    })

        return jsonify({
            "status":     "ok",
            "period_days": days,
            "stats":      stats,
            "trades":     sorted(roundtrips, key=lambda x: x["exit_date"], reverse=True),
            "open_lots":  open_lots,
        })
    except Exception as e:
        logger.error(f"/api/trade-history error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sell/<ticker>", methods=["POST"])
def api_sell(ticker: str):
    ticker    = ticker.upper()
    body      = request.json or {}
    qty       = body.get("quantity")
    stop_loss = body.get("stop_loss", False)   # True → apply extended SL cooldown
    result    = close_position(
        ticker,
        reason="manual_sell",
        qty=qty,
        stop_loss=stop_loss,
    )
    return jsonify({
        "status":     "ok" if result["success"] else "error",
        "message":    result["error_msg"],
        "fill_price": result["fill_price"],
        "fill_qty":   result["fill_qty"],
    }), (200 if result["success"] else 502)


# ── Non-ETF Position Management ──────────────────────────────────────────────

@app.route("/api/non-etf-positions", methods=["GET"])
def api_non_etf_positions():
    """List positions that are NOT in the ETF universe."""
    try:
        positions = broker.get_positions()
        non_etf = {k: v for k, v in positions.items() if k not in ETF_UNIVERSE}
        etf_pos = {k: v for k, v in positions.items() if k in ETF_UNIVERSE}
        return jsonify({
            "etf_positions":     etf_pos,
            "non_etf_positions": non_etf,
            "non_etf_count":     len(non_etf),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/close-non-etf", methods=["POST"])
def api_close_non_etf():
    """Close all positions that are NOT in the ETF universe."""
    try:
        positions = broker.get_positions()
        non_etf = [k for k in positions if k not in ETF_UNIVERSE]
        results = {}
        for ticker in non_etf:
            r = close_position(ticker, reason="close_non_etf")
            results[ticker] = {
                "status":     "ok" if r["success"] else "error",
                "fill_price": r["fill_price"],
                "fill_qty":   r["fill_qty"],
                "error_msg":  r["error_msg"],
            }
            if r["success"]:
                logger.info(f"Closed non-ETF position: {ticker}")
            else:
                logger.error(f"Failed to close {ticker}: {r['error_msg']}")
        return jsonify({"status": "ok", "closed": results, "count": len(non_etf)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Circuit Breaker passthrough ───────────────────────────────────────────────

@app.route("/api/circuit", methods=["GET"])
def api_circuit():
    """Proxy check to circuit breaker service."""
    import requests as req
    from config import CIRCUIT_BREAKER_URL
    try:
        resp = req.get(f"{CIRCUIT_BREAKER_URL}/status", timeout=5)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"status": "UNKNOWN", "error": str(e)}), 502


# ── ML Model Control ──────────────────────────────────────────────────────────

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Manually trigger ML model retraining."""
    from ml_model import train_all
    try:
        train_all(retrain=True)
        return jsonify({"status": "ok", "message": "Models retrained."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Long-Term Analysis ───────────────────────────────────────────────────────

@app.route("/api/longterm", methods=["GET"])
def api_longterm():
    """
    Analisis de largo plazo (6-24 meses) para todos los ETFs en LONGTERM_UNIVERSE.
    Score 0-100 basado en: momentum 12m/6m, SMA 50/100/200,
    fuerza relativa vs SPY, dividendo yield y tendencia de volumen.

    Senales:
      STRONG BUY  → score >= 65
      ACCUMULATE  → score >= 50
      WAIT        → score < 50
    """
    from scorer import score_all_longterm
    try:
        results = score_all_longterm()
        strong_buys   = [r for r in results if r["signal"] == "STRONG BUY"]
        accumulates   = [r for r in results if r["signal"] == "ACCUMULATE"]
        top_pick      = results[0] if results else None
        return jsonify({
            "status":       "ok",
            "horizon":      "largo plazo (6-24 meses)",
            "top_pick":     top_pick["ticker"] if top_pick else None,
            "strong_buys":  [r["ticker"] for r in strong_buys],
            "accumulates":  [r["ticker"] for r in accumulates],
            "scores":       results,
        })
    except Exception as e:
        logger.error(f"/api/longterm error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stocks", methods=["GET"])
def api_stocks():
    """
    Analisis de largo plazo para acciones individuales (STOCKS_UNIVERSE).
    Mismos criterios que /api/longterm: momentum, SMA, RS vs SPY, dividendo, volumen.
    """
    from scorer import score_all_stocks
    try:
        results = score_all_stocks()
        strong_buys = [r for r in results if r["signal"] == "STRONG BUY"]
        accumulates = [r for r in results if r["signal"] == "ACCUMULATE"]
        return jsonify({
            "status":      "ok",
            "horizon":     "largo plazo (6-24 meses)",
            "top_pick":    results[0]["ticker"] if results else None,
            "strong_buys": [r["ticker"] for r in strong_buys],
            "accumulates": [r["ticker"] for r in accumulates],
            "scores":      results,
        })
    except Exception as e:
        logger.error(f"/api/stocks error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Sentiment Status ──────────────────────────────────────────────────────────

@app.route("/api/sentiment", methods=["GET"])
def api_sentiment():
    from sentiment_free import get_sentiment_features
    return jsonify(get_sentiment_features())


# ── Scanner Control ───────────────────────────────────────────────────────────

@app.route("/api/scanner/start", methods=["POST"])
def api_scanner_start():
    start_scanner()
    return jsonify({"status": "ok", "message": "Scanner started."})


@app.route("/api/scanner/stop", methods=["POST"])
def api_scanner_stop():
    stop_scanner()
    return jsonify({"status": "ok", "message": "Scanner stopped."})


# ── Emergency Kill Switch ─────────────────────────────────────────────────────

@app.route("/api/emergency-stop", methods=["POST"])
def api_emergency_stop():
    import requests as _req
    from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

    ts        = datetime.utcnow().isoformat()
    log_lines = [f"=== EMERGENCY STOP {ts} ==="]
    results   = {
        "timestamp":        ts,
        "scanner":          None,
        "orders_cancelled": [],
        "positions_closed": [],
        "errors":           [],
    }

    # 1. Stop scanner
    try:
        stop_scanner()
        log_lines.append("SCANNER stopped")
        results["scanner"] = "stopped"
    except Exception as e:
        msg = f"SCANNER error: {e}"
        log_lines.append(msg)
        results["errors"].append(msg)

    # 2. Cancel all open orders
    try:
        orders = broker.list_open_orders()
        for o in orders:
            res    = broker.cancel_order(o["id"])
            status = res.get("status", "error")
            log_lines.append(f"CANCEL {o['symbol']} order {o['id']}: {status}")
            results["orders_cancelled"].append({"id": o["id"], "symbol": o["symbol"], "status": status})
    except Exception as e:
        msg = f"CANCEL ORDERS error: {e}"
        log_lines.append(msg)
        results["errors"].append(msg)

    # 3. Sell all open positions via close_position so trailing_state and
    # cooldown stay consistent. Note: unrealized_pnl logged here is a
    # snapshot at command time; the realized P&L is the difference between
    # avg_entry and the actual fill_price returned below (relevant when
    # market is closed and the order queues for next-open fill).
    try:
        positions = broker.get_positions()
        for ticker, pos in positions.items():
            r        = close_position(ticker, reason="emergency_stop")
            status   = "ok" if r["success"] else "error"
            pnl      = pos.get("unrealized_pnl", 0)
            fp       = r["fill_price"]
            fp_str   = f"${fp:.2f}" if fp is not None else "pending"
            log_lines.append(
                f"SELL {ticker} (P&L unrealized ${pnl:.2f}, fill={fp_str}): {status}"
            )
            results["positions_closed"].append({
                "ticker":         ticker,
                "status":         status,
                "unrealized_pnl": pnl,
                "fill_price":     fp,
                "fill_qty":       r["fill_qty"],
                "error_msg":      r["error_msg"],
            })
    except Exception as e:
        msg = f"SELL POSITIONS error: {e}"
        log_lines.append(msg)
        results["errors"].append(msg)

    # 4. Send Telegram alert
    n_orders    = len(results["orders_cancelled"])
    n_positions = len(results["positions_closed"])
    tg_lines    = [
        "🚨 KILL SWITCH ACTIVADO",
        f"Scanner: detenido",
        f"Órdenes canceladas: {n_orders}",
        f"Posiciones cerradas: {n_positions}",
    ]
    for p in results["positions_closed"]:
        tg_lines.append(f"  • {p['ticker']} → {p['status']}  P&L ${p['unrealized_pnl']:.2f}")
    if results["errors"]:
        tg_lines.append(f"⚠️ Errores: {len(results['errors'])}")
    tg_lines.append(f"Hora UTC: {ts}")

    try:
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            _req.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": "\n".join(tg_lines), "parse_mode": "HTML"},
                timeout=10,
            )
            log_lines.append("TELEGRAM sent")
    except Exception as e:
        log_lines.append(f"TELEGRAM error: {e}")

    # 5. Write to logs/emergency.log
    os.makedirs("logs", exist_ok=True)
    with open("logs/emergency.log", "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n\n")

    logger.warning(f"EMERGENCY STOP — positions closed: {n_positions}, orders cancelled: {n_orders}")
    return jsonify({"status": "ok", **results})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Initializing ETF Trader (no LLM)...")
    initialize()        # train / load ML models
    start_scanner()     # background 30-min scan loop

    logger.info(f"Starting Flask on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG, use_reloader=False)
