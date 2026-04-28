# backtest_thresholds.py — Compara umbrales de entrada para ETFTRADE
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from config import ETF_UNIVERSE

INITIAL_CAPITAL = 100_000
STOP_LOSS_PCT   = 0.05
TAKE_PROFIT_PCT = 0.12
MAX_POSITION_PCT= 0.10
BACKTEST_DAYS   = 500
TRAIN_DAYS      = 252
THRESHOLDS      = [50, 55, 60, 65, 70]

def _rsi(s, p=14):
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    loss  = (-delta).clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

def compute_score_series(df):
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    rsi_s  = _rsi(close, 14)
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd_h = ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std().replace(0, 1e-9)
    bb_pos = ((close - bb_mid) / bb_std + 2) / 4 * 100
    vol_r  = (volume / volume.rolling(20).mean()).clip(0, 2) * 50

    scores = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 200:
            scores.iloc[i] = np.nan
            continue
        rsi = float(rsi_s.iloc[i])
        if rsi < 30:    rsi_score = 20.0
        elif rsi <= 50: rsi_score = 40.0 + (rsi - 30)
        elif rsi <= 70: rsi_score = 60.0 + (rsi - 50)
        else:           rsi_score = max(0.0, 90.0 - (rsi - 70) * 2.0)

        price     = float(close.iloc[i])
        macd_norm = float(np.clip(50 + (float(macd_h.iloc[i]) / max(price * 0.002, 1e-9)) * 25, 0, 100))
        trend = (35 if price > float(sma50.iloc[i]) else 0) + \
                (35 if price > float(sma200.iloc[i]) else 0) + \
                (30 if float(sma50.iloc[i]) > float(sma200.iloc[i]) else 0)
        bb  = float(np.clip(bb_pos.iloc[i], 0, 100))
        vol = float(min(float(vol_r.iloc[i]), 100))
        scores.iloc[i] = rsi_score*0.25 + macd_norm*0.25 + trend*0.30 + bb*0.10 + vol*0.10
    return scores

def run_backtest(df, threshold):
    close  = df["Close"].squeeze()
    scores = compute_score_series(df)
    capital, position, trades, equity = INITIAL_CAPITAL, None, [], []

    for i in range(len(df)):
        date, price, score = df.index[i], float(close.iloc[i]), scores.iloc[i]

        if position is not None:
            pnl_pct = (price - position["entry_price"]) / position["entry_price"]
            if pnl_pct <= -STOP_LOSS_PCT:
                exit_r = "SL"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                exit_r = "TP"
            else:
                exit_r = None

            if exit_r:
                proceeds = position["shares"] * price
                capital += proceeds
                trades.append({"pnl": proceeds - position["cost"], "pnl_pct": pnl_pct*100, "exit": exit_r})
                position = None

        if position is None and not pd.isna(score) and score >= threshold:
            invest   = capital * MAX_POSITION_PCT
            capital -= invest
            position = {"entry_price": price, "shares": invest/price, "cost": invest}

        equity.append(capital + (position["shares"]*price if position else 0))

    if position is not None:
        price    = float(close.iloc[-1])
        proceeds = position["shares"] * price
        capital += proceeds
        trades.append({"pnl": proceeds - position["cost"],
                        "pnl_pct": (price - position["entry_price"])/position["entry_price"]*100,
                        "exit": "END"})
        equity[-1] = capital

    if not trades:
        return None

    eq  = pd.Series(equity, index=df.index)
    tr  = (eq.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    cagr= ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1/max(yrs,0.01)) - 1) * 100
    mdd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()
    wr  = len([t for t in trades if t["pnl"] > 0]) / len(trades) * 100
    gw  = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl  = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0)) or 1e-9
    dr  = eq.pct_change().dropna()
    sh  = ((dr.mean()*252 - 0.02) / (dr.std()*np.sqrt(252))) if dr.std() > 0 else 0
    sl_count = sum(1 for t in trades if t["exit"] == "SL")
    tp_count = sum(1 for t in trades if t["exit"] == "TP")

    return {
        "trades": len(trades), "win_rate": round(wr,1), "pf": round(gw/gl,2),
        "total_return": round(tr,2), "cagr": round(cagr,2),
        "sharpe": round(sh,2), "max_dd": round(mdd,2),
        "final": round(eq.iloc[-1],0),
        "sl": sl_count, "tp": tp_count,
    }

def main():
    end   = datetime.today()
    start = end - timedelta(days=BACKTEST_DAYS + TRAIN_DAYS + 10)

    print(f"\nDescargando datos ({', '.join(ETF_UNIVERSE)})...\n")
    dfs = {}
    for t in ETF_UNIVERSE:
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if not df.empty:
            dfs[t] = df.tail(BACKTEST_DAYS + TRAIN_DAYS)
            print(f"  {t}: {len(dfs[t])} barras")

    print(f"\n{'Umbral':<8} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Return':>9} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'SL/TP':>8} {'Capital Final':>14}")
    print("-" * 100)

    best_pf = {}

    for threshold in THRESHOLDS:
        all_trades, all_returns, total_final = 0, [], INITIAL_CAPITAL * len(dfs)
        wins, gross_w, gross_l = 0, 0, 0
        all_sl, all_tp = 0, 0
        all_eq = []

        for ticker, df in dfs.items():
            r = run_backtest(df, threshold)
            if r is None:
                continue
            all_trades  += r["trades"]
            gross_w     += r["pf"] * (abs(gross_l) or 1) * r["trades"] * 0.1  # approx
            all_sl      += r["sl"]
            all_tp      += r["tp"]

            # Aggregate equity approximation
            close = df["Close"].squeeze()
            scores = compute_score_series(df)
            all_returns.append(r["total_return"])
            total_final += (r["final"] - INITIAL_CAPITAL)

        if all_trades == 0:
            print(f"{threshold:<8} {'sin senales':>75}")
            continue

        avg_return = np.mean(all_returns) if all_returns else 0
        avg_cagr   = avg_return / (BACKTEST_DAYS/365.25)
        pnl_total  = total_final - INITIAL_CAPITAL * len(dfs)

        # Per-ticker details
        results_by_ticker = {}
        for ticker, df in dfs.items():
            results_by_ticker[ticker] = run_backtest(df, threshold)

        all_wr  = np.mean([r["win_rate"]  for r in results_by_ticker.values() if r])
        all_pf  = np.mean([r["pf"]        for r in results_by_ticker.values() if r])
        all_sh  = np.mean([r["sharpe"]    for r in results_by_ticker.values() if r])
        all_mdd = np.min ([r["max_dd"]    for r in results_by_ticker.values() if r])
        total_cap = sum(r["final"] for r in results_by_ticker.values() if r)

        marker = " <-- actual" if threshold == 65 else ""
        print(f"{threshold:<8} {all_trades:>7} {all_wr:>7.1f}% {all_pf:>6.2f} {avg_return:>+8.1f}% {avg_cagr:>+7.1f}% {all_sh:>7.2f} {all_mdd:>+7.1f}% {all_sl:>4}SL/{all_tp:<3}TP  ${total_cap:>12,.0f}{marker}")

    print("-" * 100)
    print(f"Capital inicial total ({len(dfs)} ETFs x $100K): ${INITIAL_CAPITAL * len(dfs):,}\n")

    # Detalle por ETF para cada umbral
    print("\nDetalle por ETF:")
    print(f"{'Ticker':<7}", end="")
    for th in THRESHOLDS:
        print(f"  {'TH='+str(th):>10}", end="")
    print()
    print("-" * 65)

    for ticker, df in dfs.items():
        print(f"{ticker:<7}", end="")
        for th in THRESHOLDS:
            r = run_backtest(df, th)
            if r:
                print(f"  {r['total_return']:>+9.1f}%", end="")
            else:
                print(f"  {'sin sig':>10}", end="")
        print()

    print("-" * 65)
    print("(Retorno total por ticker en el periodo)")

if __name__ == "__main__":
    main()
