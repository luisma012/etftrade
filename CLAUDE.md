# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

ETFTRADE is a semi-autonomous ETF trading bot with a Flask web dashboard. It uses a 50% technical + 50% ML composite score to identify trade signals in a 9-ETF universe (SPY, SCHD, IFRA, SCHF, GLD, IWM, XLE, QQQ, VTI — see `ETF_UNIVERSE` in `config.py`; XLV was eliminated 2026-04-19, VGT is no longer scanned despite stale pickle files on disk), then requires Telegram-based human approval before executing any order. Currently running in paper trading mode on Alpaca.

## Running the Project

```bash
# Install dependencies (note: numpy<2.3 required for xgboost/numba compatibility)
pip install -r requirements.txt

# Start Flask API server (main entry point, port 5010)
python app.py

# Alternative: Dash-based dashboard (full-featured, same port)
python analyzer.py
```

Windows launchers: `run.bat` (app.py), `start.bat` (analyzer.py), `menu.bat` (interactive menu for all operations).

On startup, `app.py` calls `initialize()` (defined in `scanner.py`, trains/loads ML models) then `start_scanner()` (30-min background scan thread). First run trains 9 XGBoost+RF models — one per `ETF_UNIVERSE` symbol (~10–30 seconds); subsequent runs load from `ml_models/` instantly.

## Testing / Manual API Interaction

There are no automated tests. Verify behavior via the Flask API:

```bash
# Health check (uptime, market open status)
curl http://localhost:5010/health

# Check ETF scores without triggering trades
curl http://localhost:5010/api/scores

# Manually trigger a market scan (populates pending trades)
curl -X POST http://localhost:5010/api/scan

# View pending signals awaiting approval
curl http://localhost:5010/api/pending

# Approve a trade (executes the buy order)
curl -X POST http://localhost:5010/api/approve/SPY

# Reject a pending trade
curl -X POST http://localhost:5010/api/reject/SPY

# Manually sell a position (optional body: {"quantity": N, "stop_loss": true})
curl -X POST http://localhost:5010/api/sell/SPY

# Force ML model retraining
curl -X POST http://localhost:5010/api/retrain

# Current portfolio / P&L (returns acct.equity from Alpaca; returns 0.0 on API error)
curl http://localhost:5010/api/portfolio

# Current Fear & Greed + SPY RSI sentiment
curl http://localhost:5010/api/sentiment

# Long-term ETF scores (6–24 month horizon)
curl http://localhost:5010/api/longterm

# Long-term individual stock scores
curl http://localhost:5010/api/stocks

# Non-ETF positions (should be empty; use /api/close-non-etf to clean up)
curl http://localhost:5010/api/non-etf-positions
curl -X POST http://localhost:5010/api/close-non-etf

# Scanner control
curl -X POST http://localhost:5010/api/scanner/start
curl -X POST http://localhost:5010/api/scanner/stop
```

Telegram commands also work: `/approve_SPY`, `/reject_SPY`, `/pending`, `/status`.

## Architecture Overview

```
yfinance / Alpaca WebSocket
        │
   scanner.py  ← background thread, 30-min intervals, market hours only (9:30–16:00 NY)
        │
   scorer.py   ← composite 0–100 score
   ├── Technical (50%): RSI, MACD, SMA cross, Bollinger Bands, Volume
   │   └── ADX multiplier: lateral (×0.75) / moderate (×1.00) / strong (×1.10)
   └── ml_model.py (50%): XGBoost + Random Forest VotingClassifier
              └── sentiment_free.py: Fear & Greed Index + SPY RSI (free, no API key)
        │
   Entry gates (all must pass):
   ├── Score ≥ MIN_SCORE_BY_TICKER[symbol] (per-ticker backtest-optimized thresholds)
   ├── VIX < VIX_MAX (25)
   ├── SPY above SMA50 (SPY_SMA_FILTER)
   ├── CONSEC_CANDLES (2) consecutive bullish closes
   ├── Fear & Greed > 20 OR SPY RSI ≤ 40 (block extreme-fear non-oversold entries)
   ├── Circuit breaker CLOSED (optional, disabled by default)
   └── Cooldown ok? Position limits ok?
        │
   → Telegram notification → human approval (AUTO_TRADE=false by default)
        │
   broker/__init__.py  ← factory: loads Alpaca or Robinhood based on BROKER env var
   ├── alpaca.py  ← primary: paper + live via alpaca-py SDK (bracket orders: 5% SL / TP)
   └── robinhood.py  ← secondary: robin-stocks SDK; no bracket order support (loses SL/TP guarantees)
```

**Key files:**
- `config.py` — single source of truth for all trading parameters (thresholds, limits, weights)
- `app.py` — Flask API routes + dashboard; starts scanner thread
- `scanner.py` — scan loop, gate checks, `_pending_trades` dict, trailing stop monitor, daily audit
- `scorer.py` — composite scoring + long-term scoring (ETFs and individual stocks)
- `ml_model.py` — feature engineering (25 features), ensemble training, model persistence in `ml_models/`
- `sentiment_free.py` — free sentiment signals (CNN Fear & Greed, SPY RSI)
- `analyzer.py` — 2,700-line Dash dashboard with VWAP, Bollinger Bands, Footprint, XGBoost BUY/HOLD/SELL signals, and real-time WebSocket quotes. Parallel to app.py; both run on port 5010.
- `check_audit.py` — live-readiness checklist (runs daily at market open; also `python check_audit.py`)
- `backtest_bot.py` — simulates the exact live bot logic on historical data
- `backtest_thresholds.py` — compares entry thresholds 50–70 to optimize Profit Factor
- `AUDIT_BOT_LIVE.md` — formal 3-phase live-readiness checklist (28-item; min 30 days paper)

## Configuration

Copy `.env.template` to `.env` and fill in credentials:

```
BROKER=alpaca                          # or robinhood
ALPACA_API_KEY=pk_...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # change for live
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
CIRCUIT_BREAKER_URL=http://localhost:5030

# Robinhood (if BROKER=robinhood)
ROBINHOOD_USERNAME=...
ROBINHOOD_PASSWORD=...
ROBINHOOD_MFA_CODE=...                 # TOTP if 2FA enabled
```

All trading rule constants are in `config.py`. Current values (verify before changing):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MIN_SCORE_BUY` | 60 | Global floor — per-ticker thresholds override this |
| `MIN_SCORE_BY_TICKER` | see below | Backtest-optimized per symbol |
| `MAX_POSITION_PCT` | 0.05 live / 0.10 paper | Auto-detected from ALPACA_BASE_URL; drops to 5% when Fear & Greed < 30 |
| `MAX_OPEN_POSITIONS` | 3 live / 4 paper | Auto-detected |
| `STOP_LOSS_PCT` | 0.05 | Floor; ATR-based SL (`ATR_MULTIPLIER=1.5`) may be higher |
| `TAKE_PROFIT_PCT` | 0.12 | Base; expands to 0.18 when ADX ≥ 30 (`TAKE_PROFIT_HIGH`) |
| `TRAILING_ACTIVATE_PCT` | 0.06 | Trailing stop activates at +6% |
| `TRAILING_STOP_PCT` | 0.03 | Trails 3% below position high |
| `SCAN_INTERVAL_MIN` | 30 | |
| `COOLDOWN_HOURS` | 24 | Normal cooldown after a trade |
| `COOLDOWN_SL_HOURS` | 144 | Extended cooldown (6 days) after stop-loss |
| `VIX_MAX` | 25 | Block new entries when VIX ≥ this |
| `AUTO_TRADE` | false | Set to true to skip Telegram approval (not recommended) |
| `RETRAIN_ON_STARTUP` | True | Set to False to skip ML retraining on restart |

**Per-ticker entry thresholds** (backtest-optimized, 500-day window):

| Ticker | Threshold | Win Rate | Profit Factor |
|--------|-----------|----------|---------------|
| GLD | 84 | 78% | 4.55 |
| SCHF | 60 | 100% | 4.11 |
| SPY | 69 | 75% | 3.59 |
| VTI | 69 | 75% | 3.11 |
| IWM | 78 | 75% | 2.78 |
| IFRA | 69 | 60% | 2.07 |
| SCHD | 84 | 67% | 1.45 |
| QQQ | 84 | 67% | 1.36 |
| XLE | 60 | 64% | 1.36 |

The circuit breaker is optional and disabled by default (`CIRCUIT_BREAKER_ENABLED=false`).

## Trailing Stop Logic

`_monitor_positions()` in `scanner.py` runs each scan cycle. It tracks a high-water mark per open position in `trailing_state.json`:
1. Once a position gains ≥ 6%, trailing mode activates; sends Telegram alert.
2. In trailing mode, if price drops ≥ 3% from the high, the position is sold automatically.
3. Alpaca bracket orders (5% SL / 12% TP) remain as a server-side safety net; the trailing stop typically fires first once the gain threshold is met.

Two independent exit mechanisms run concurrently: client-side trailing stop (scanner.py) and server-side bracket order (Alpaca). Trailing state survives process restarts via `trailing_state.json`.

## Persistent State

| File | Purpose |
|------|---------|
| `ml_models/` | Pickled XGBoost + RF models (auto-created on first run) |
| `cooldown.json` | Last trade timestamp per symbol; SL cooldown stored as `{TICKER}_sl` |
| `trailing_state.json` | High-water marks and trailing-active flags for open positions |
| `audit_state.json` | Paper trading P&L snapshot |
| `backtest_results.csv` | Trade log from backtests (used by audit to validate WR/PF) |

## Non-Obvious Architectural Constraints

**Portfolio value returns 0.0 on broker error**: `broker/alpaca.py get_portfolio_value()` calls `acct.equity` and returns `0.0` on any exception. If Alpaca credentials are missing or invalid, the dashboard silently shows $0.00 — check `.env` and the Flask logs.

**Startup blocks on ML training**: `initialize()` (defined in `scanner.py`, called from `app.py`) calls `train_all()` synchronously before Flask starts. First run can take 10–30 seconds. Subsequent runs load from `ml_models/` instantly. The directory currently holds 33 pickle files (3 per ticker × 11 tickers including stale VGT and XLV), but only 27 are actively used — `train_all()` iterates `ETF_UNIVERSE` (9 symbols).

**Mixed fail policies**: VIX check is fail-OPEN — if yfinance is unreachable, trades continue (`_vix_ok()` in `scanner.py`). Circuit breaker is fail-CLOSED when enabled — if the CB service at `CIRCUIT_BREAKER_URL` is unreachable, returns non-200, returns non-JSON, or returns `status != "CLOSED"` (including UNKNOWN/missing), trades are blocked. When `CIRCUIT_BREAKER_ENABLED=false` (default), the gate is fully bypassed.

**Alpaca coupling despite broker abstraction**: Robinhood does not support native bracket orders. The trailing stop logic and bracket SL/TP guarantees only work correctly with Alpaca. Switching to `BROKER=robinhood` silently loses server-side protection.

**`BROKER` defaults to Robinhood, not Alpaca**: `config.py:8` is `os.getenv("BROKER", "robinhood")`. If `BROKER` is unset in `.env`, `broker/__init__.py` imports the Robinhood adapter and calls `login()` at import time — which means startup fails fast with bad credentials, and even with good ones you silently lose bracket-order protection (see above). Despite the rest of this doc treating Alpaca as primary, you must explicitly set `BROKER=alpaca` to get the Alpaca path.

**Backtest uses technical score only (not ML)**: `backtest_bot.py` calls a technical-only scoring function because retraining 11 ML models per backtest day is too expensive. Live trading uses 50% ML weight. Backtest results are therefore conservative relative to live performance.

**Entry gates are not equally enforced**: Hard gates abort the signal entirely (per-ticker score threshold, VIX, cooldown, position limits, circuit breaker). Soft gates reduce the score via ADX multipliers (±25%). A symbol with score 80/100 can still be blocked if soft gates all fire.

**Pending signals auto-expire after 90 minutes**: `_expire_pending()` runs each scan cycle. Signals not approved/rejected via Telegram within 90 minutes are silently dropped.

**`AUTO_TRADE` flag defaults to false**: `config.py` has `AUTO_TRADE = os.getenv("AUTO_TRADE", "false")`. When set to true, the scan loop executes `broker.buy()` directly without Telegram approval.

**`/api/scores` is synchronous and slow**: Downloads OHLCV for all 11 symbols + runs full scoring in the Flask request thread. Latency is 3–10 seconds. A slow scores request blocks approvals and rejections.

**Three scoring universes**: The bot operates across three symbol sets:
- Short-term swing trades (30-min scans): 9 ETFs in `ETF_UNIVERSE`
- Long-term ETF accumulation (6–24 month): same ETFs via `/api/longterm` — uses momentum scoring (SMA 50/100/200, 6m/12m returns), not composite score
- Long-term individual stocks: 21 symbols via `/api/stocks`

**Sentiment is also an ML feature, not just a filter**: `sentiment_free.py` provides `fear_greed_norm`, `spy_rsi_norm`, and `combined_sentiment` which are 3 of the 25 ML features — broadcast as market-wide constants to every symbol's model input on each scan.

**Cooldown is two-tiered**: Normal exit → 24h cooldown. Stop-loss exit → 144h cooldown. The SL cooldown key is stored as `{TICKER}_sl` in `cooldown.json`.

**Fear & Greed reduces position size**: When F&G < 30 (extreme fear), `MAX_POSITION_PCT` is overridden to `FEAR_GREED_REDUCE_PCT = 0.05` regardless of live/paper mode.

## Backtesting

```bash
# Full backtest using the bot's actual scorer logic (~2 years, all symbols)
python backtest_bot.py

# Run standard strategy comparisons (MACD, SMA crossover, RSI, Bollinger)
python run_backtest_all.py   # writes results to backtest_results.csv + backtest_reports/

# Optimize score thresholds across range 50–70
python backtest_thresholds.py

# Screen candidate ETFs/stocks not in the main universe
python backtest_candidates.py
```

`backtest_bot.py` uses 252-day training window + 500-day test window, `INITIAL_CAPITAL=100_000`, and mirrors live rules (stop-loss, take-profit, cooldowns). Results are printed to stdout.

## Live Readiness Audit

```bash
python check_audit.py   # manual run, validates 8 sections: system, portfolio, risk, CB, cooldown, bracket orders, backtest metrics, security
```

Runs automatically once per day at market open (via `scanner._run_daily_audit()`). See `AUDIT_BOT_LIVE.md` for the full 3-phase checklist. Minimum thresholds for going live: win rate >55%, profit factor >1.3, max drawdown <10%.

## Switching to Live Trading

Change `ALPACA_BASE_URL` in `.env` from paper to `https://api.alpaca.markets`. Live trading auto-sets `MAX_POSITION_PCT=0.05` and `MAX_OPEN_POSITIONS=3`. Follow the progression in `config.py` comments (Week 1–2: 5%, Week 3–4: 10%, Month 2+: 20%).
