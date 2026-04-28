# backtest_bot.py — Backtesting real del ETFTRADE usando scorer.py
# Simula exactamente la logica del bot: composite score >= MIN_SCORE_BUY → BUY
# Stop-loss 5%, Take-profit 12%, Position size 10%, max 4 posiciones
#
# Uso: python backtest_bot.py

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from config import (
    ETF_UNIVERSE, MIN_SCORE_BUY,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT, TRAILING_ACTIVATE_PCT,
    MAX_POSITION_PCT, MAX_OPEN_POSITIONS,
    HISTORY_DAYS, WEIGHT_TECHNICAL, WEIGHT_ML,
)
from scorer import compute_technical_score
from ml_model import build_features, build_labels

# ── Parámetros del backtest ────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000
BACKTEST_DAYS   = 500   # ~2 años de trading days
TRAIN_DAYS      = 252   # datos de entrenamiento antes del backtest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    loss  = (-delta).clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def compute_score_series(df: pd.DataFrame) -> pd.Series:
    """
    Computes daily composite scores across all available dates.
    Uses ONLY technical score (ML would require daily retraining which is expensive).
    Normalizes to 0-100.
    """
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    # RSI
    rsi_s = _rsi(close, 14)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    # SMA trend
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # Bollinger
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std().replace(0, 1e-9)
    bb_pos = ((close - bb_mid) / bb_std + 2) / 4 * 100

    # Volume ratio
    vol_ratio = (volume / volume.rolling(20).mean()).clip(0, 2) * 50

    scores = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 200:
            scores.iloc[i] = np.nan
            continue

        rsi = float(rsi_s.iloc[i])
        if rsi < 30:
            rsi_score = 20.0
        elif rsi <= 50:
            rsi_score = 40.0 + (rsi - 30) * 1.0
        elif rsi <= 70:
            rsi_score = 60.0 + (rsi - 50) * 1.0
        else:
            rsi_score = max(0.0, 90.0 - (rsi - 70) * 2.0)

        price = float(close.iloc[i])
        macd  = float(macd_hist.iloc[i])
        macd_norm = float(np.clip(50 + (macd / max(price * 0.002, 1e-6)) * 25, 0, 100))

        trend = 0.0
        if price > float(sma50.iloc[i]):   trend += 35
        if price > float(sma200.iloc[i]):  trend += 35
        if float(sma50.iloc[i]) > float(sma200.iloc[i]): trend += 30

        bb  = float(np.clip(bb_pos.iloc[i], 0, 100))
        vol = float(min(float(vol_ratio.iloc[i]), 100))

        composite = (
            rsi_score * 0.25 +
            macd_norm * 0.25 +
            trend     * 0.30 +
            bb        * 0.10 +
            vol       * 0.10
        )
        scores.iloc[i] = composite

    return scores


# ── Backtester ────────────────────────────────────────────────────────────────

def run_backtest(ticker: str, df: pd.DataFrame) -> dict:
    scores = compute_score_series(df)
    close  = df["Close"].squeeze()

    capital    = INITIAL_CAPITAL
    position   = None   # {"entry_price", "entry_date", "shares", "cost"}
    trades     = []
    equity     = []

    max_shares = lambda: (capital * MAX_POSITION_PCT) / float(close.iloc[i])

    for i in range(len(df)):
        date  = df.index[i]
        price = float(close.iloc[i])
        score = scores.iloc[i]

        # Manage open position: check trailing stop / take-profit
        if position is not None:
            # Track high-water mark for trailing stop
            if price > position["high_price"]:
                position["high_price"] = price

            pnl_pct = (price - position["entry_price"]) / position["entry_price"]
            gain_from_entry = (position["high_price"] - position["entry_price"]) / position["entry_price"]

            exit_reason = None
            if pnl_pct >= TAKE_PROFIT_PCT:
                exit_reason = "TAKE_PROFIT"
            elif gain_from_entry >= TRAILING_ACTIVATE_PCT:
                # Trailing stop activated: 3% below high-water mark
                trailing_sl = position["high_price"] * (1 - TRAILING_STOP_PCT)
                if price <= trailing_sl:
                    exit_reason = "TRAILING_STOP"
            elif pnl_pct <= -STOP_LOSS_PCT:
                # Hard stop (before trailing activates)
                exit_reason = "STOP_LOSS"

            if exit_reason:
                proceeds = position["shares"] * price
                pnl      = proceeds - position["cost"]
                capital += proceeds
                trades.append({
                    "ticker":       ticker,
                    "entry_date":   position["entry_date"],
                    "exit_date":    date,
                    "entry_price":  position["entry_price"],
                    "exit_price":   price,
                    "shares":       position["shares"],
                    "pnl":          round(pnl, 2),
                    "pnl_pct":      round(pnl_pct * 100, 2),
                    "exit_reason":  exit_reason,
                    "hold_days":    (date - position["entry_date"]).days,
                })
                position = None

        # Entry: score >= threshold, no open position
        if position is None and not pd.isna(score) and score >= MIN_SCORE_BUY:
            invest   = capital * MAX_POSITION_PCT
            shares   = invest / price
            capital -= invest
            position = {
                "entry_price": price,
                "entry_date":  date,
                "shares":      shares,
                "cost":        invest,
                "high_price":  price,   # high-water mark for trailing stop
            }

        # Equity = cash + open position value
        pos_value = (position["shares"] * price) if position else 0.0
        equity.append(capital + pos_value)

    # Close any remaining position at end
    if position is not None:
        price = float(close.iloc[-1])
        proceeds = position["shares"] * price
        pnl_pct  = (price - position["entry_price"]) / position["entry_price"]
        capital += proceeds
        trades.append({
            "ticker":       ticker,
            "entry_date":   position["entry_date"],
            "exit_date":    df.index[-1],
            "entry_price":  position["entry_price"],
            "exit_price":   price,
            "shares":       position["shares"],
            "pnl":          round(proceeds - position["cost"], 2),
            "pnl_pct":      round(pnl_pct * 100, 2),
            "exit_reason":  "END_OF_PERIOD",
            "hold_days":    (df.index[-1] - position["entry_date"]).days,
        })
        equity[-1] = capital

    if not trades:
        return {"ticker": ticker, "trades": 0, "no_signals": True}

    eq = pd.Series(equity, index=df.index)
    total_return = (eq.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr  = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1 / max(years, 0.01)) - 1) * 100

    rolling_max = eq.expanding().max()
    drawdown    = (eq - rolling_max) / rolling_max * 100
    max_dd      = drawdown.min()

    wins      = [t for t in trades if t["pnl"] > 0]
    losses    = [t for t in trades if t["pnl"] <= 0]
    win_rate  = len(wins) / len(trades) * 100
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss= abs(sum(t["pnl"] for t in losses)) or 1e-9
    pf        = gross_win / gross_loss

    daily_ret = eq.pct_change().dropna()
    sharpe    = 0.0
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() * 252 - 0.02) / (daily_ret.std() * np.sqrt(252))

    avg_hold  = np.mean([t["hold_days"] for t in trades])

    return {
        "ticker":         ticker,
        "period":         f"{df.index[0].date()} → {df.index[-1].date()}",
        "trades":         len(trades),
        "win_rate":       round(win_rate, 1),
        "profit_factor":  round(pf, 2),
        "total_return":   round(total_return, 2),
        "cagr":           round(cagr, 2),
        "sharpe":         round(sharpe, 2),
        "max_drawdown":   round(max_dd, 2),
        "avg_hold_days":  round(avg_hold, 1),
        "final_capital":  round(eq.iloc[-1], 2),
        "trade_log":      trades,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    end   = datetime.today()
    start = end - timedelta(days=BACKTEST_DAYS + TRAIN_DAYS + 10)

    print(f"\nETFTRADE Backtest  {(end - timedelta(days=BACKTEST_DAYS)).date()} to {end.date()}")
    print(f"Config: capital=${INITIAL_CAPITAL:,}  entry_score>={MIN_SCORE_BUY}  SL={STOP_LOSS_PCT*100:.0f}%  TP={TAKE_PROFIT_PCT*100:.0f}%  pos={MAX_POSITION_PCT*100:.0f}%\n")

    all_results = []

    for ticker in ETF_UNIVERSE:
        print(f"Downloading {ticker}...", end=" ", flush=True)
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty or len(df) < 300:
                print("insufficient data, skip")
                continue
            # Use only backtest period (last BACKTEST_DAYS bars)
            df_bt = df.tail(BACKTEST_DAYS + TRAIN_DAYS)
            r = run_backtest(ticker, df_bt)
            all_results.append(r)

            if r.get("no_signals"):
                print(f"no signals generated")
            else:
                print(f"done — {r['trades']} trades")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'Ticker':<7} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Return':>9} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'AvgHold':>9}")
    print("-" * 95)

    all_trades = []
    for r in all_results:
        if r.get("no_signals"):
            print(f"{r['ticker']:<7} {'—':>7}  No BUY signals generated")
            continue
        print(
            f"{r['ticker']:<7} {r['trades']:>7} {r['win_rate']:>7.1f}% {r['profit_factor']:>6.2f}"
            f" {r['total_return']:>+8.1f}% {r['cagr']:>+7.1f}% {r['sharpe']:>7.2f}"
            f" {r['max_drawdown']:>+7.1f}% {r['avg_hold_days']:>8.1f}d"
        )
        all_trades.extend(r.get("trade_log", []))

    print("=" * 95)

    if all_trades:
        df_trades = pd.DataFrame(all_trades)

        # Save CSV
        out_csv = os.path.join(os.path.dirname(__file__), "backtest_results.csv")
        df_trades.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\nTrade log saved: {out_csv}")

        # Exit reason breakdown
        reasons = df_trades["exit_reason"].value_counts()
        print("\nExit reasons:")
        for reason, count in reasons.items():
            pct = count / len(df_trades) * 100
            avg_pnl = df_trades[df_trades["exit_reason"] == reason]["pnl_pct"].mean()
            print(f"  {reason:<20} {count:>4} trades ({pct:>5.1f}%)  avg P&L: {avg_pnl:>+5.1f}%")

        total_pnl = df_trades["pnl"].sum()
        print(f"\nTotal net P&L across all tickers: ${total_pnl:>+,.0f}")


if __name__ == "__main__":
    main()
