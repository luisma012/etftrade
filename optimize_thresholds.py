# optimize_thresholds.py — Encuentra el MIN_SCORE_BUY optimo por ETF
# Prueba umbrales de 60 a 85 y elige el que maximiza PF con WR>50% y >2 trades

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from config import (
    ETF_UNIVERSE, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT, TRAILING_ACTIVATE_PCT,
    MAX_POSITION_PCT, HISTORY_DAYS,
)

INITIAL_CAPITAL = 100_000
BACKTEST_DAYS   = 500
THRESHOLDS      = list(range(60, 86, 3))   # 60,63,66,69,72,75,78,81,84
MIN_TRADES      = 3


def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d).clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-9))


def compute_score_series(df):
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    rsi_s  = _rsi(close)
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd_h = ema12 - ema26 - (ema12-ema26).ewm(span=9, adjust=False).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std().replace(0, 1e-9)
    bb_pos = ((close - bb_mid) / bb_std + 2) / 4 * 100
    vol_r  = (volume / volume.rolling(20).mean()).clip(0, 2) * 50

    scores = pd.Series(np.nan, index=df.index)
    for i in range(200, len(df)):
        rsi = float(rsi_s.iloc[i])
        rs  = 20 if rsi < 30 else (40+(rsi-30) if rsi<=50 else (60+(rsi-50) if rsi<=70 else max(0, 90-(rsi-70)*2)))
        p   = float(close.iloc[i])
        mn  = float(np.clip(50 + (float(macd_h.iloc[i]) / max(p*0.002, 1e-6))*25, 0, 100))
        tr  = (35 if p > float(sma50.iloc[i]) else 0) + \
              (35 if p > float(sma200.iloc[i]) else 0) + \
              (30 if float(sma50.iloc[i]) > float(sma200.iloc[i]) else 0)
        bb  = float(np.clip(bb_pos.iloc[i], 0, 100))
        vl  = float(min(float(vol_r.iloc[i]), 100))
        scores.iloc[i] = rs*0.25 + mn*0.25 + tr*0.30 + bb*0.10 + vl*0.10
    return scores


def run_backtest(ticker, df, threshold):
    scores  = compute_score_series(df)
    close   = df["Close"].squeeze()
    capital = INITIAL_CAPITAL
    pos     = None
    trades  = []

    for i in range(len(df)):
        price = float(close.iloc[i])
        score = scores.iloc[i]

        if pos is not None:
            if price > pos["high"]:
                pos["high"] = price
            pnl_pct = (price - pos["entry"]) / pos["entry"]
            gain    = (pos["high"] - pos["entry"]) / pos["entry"]
            reason  = None
            if pnl_pct >= TAKE_PROFIT_PCT:
                reason = "TP"
            elif gain >= TRAILING_ACTIVATE_PCT and price <= pos["high"]*(1-TRAILING_STOP_PCT):
                reason = "TRAIL"
            elif pnl_pct <= -STOP_LOSS_PCT:
                reason = "SL"

            if reason:
                pnl = pos["shares"] * price - pos["cost"]
                capital += pos["shares"] * price
                trades.append({"pnl": pnl, "pnl_pct": pnl_pct, "reason": reason})
                pos = None

        if pos is None and not pd.isna(score) and score >= threshold:
            invest  = capital * MAX_POSITION_PCT
            pos = {"entry": price, "shares": invest/price,
                   "cost": invest, "high": price}
            capital -= invest

    if pos is not None:
        price = float(close.iloc[-1])
        pnl_pct = (price - pos["entry"]) / pos["entry"]
        trades.append({"pnl": pos["shares"]*price - pos["cost"],
                       "pnl_pct": pnl_pct, "reason": "END"})
        capital += pos["shares"] * price

    if len(trades) < MIN_TRADES:
        return None

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses)) or 1e-6
    wr     = len(wins) / len(trades)
    pf     = gp / gl
    ret    = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return {"trades": len(trades), "wr": wr, "pf": pf, "ret": ret}


# ── Main ─────────────────────────────────────────────────────────────────────

end   = datetime.today()
start = end - timedelta(days=BACKTEST_DAYS + HISTORY_DAYS + 60)

print(f"\nOptimizando umbrales para {len(ETF_UNIVERSE)} ETFs...")
print(f"Umbrales probados: {THRESHOLDS}\n")

optimal = {}

for ticker in ETF_UNIVERSE:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        print(f"{ticker}: sin datos")
        continue

    best = None
    best_score = -999

    for thr in THRESHOLDS:
        res = run_backtest(ticker, df, thr)
        if res is None:
            continue
        if res["wr"] < 0.45:
            continue
        # Score compuesto: ponderamos PF (60%) y WR (40%)
        combined = res["pf"] * 0.6 + res["wr"] * 100 * 0.4
        if combined > best_score:
            best_score = combined
            best = {"threshold": thr, **res}

    if best:
        optimal[ticker] = best
        print(f"{ticker:5}  thr={best['threshold']}  trades={best['trades']}  "
              f"WR={best['wr']*100:.0f}%  PF={best['pf']:.2f}  ret={best['ret']:+.1f}%")
    else:
        optimal[ticker] = {"threshold": 72}  # fallback al default
        print(f"{ticker:5}  sin umbral optimo — usando default 72")

# ── Resultado final ───────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RESULTADO: MIN_SCORE_BUY optimo por ETF")
print("="*55)
for t, r in sorted(optimal.items(), key=lambda x: x[1].get("threshold", 72)):
    thr = r.get("threshold", 72)
    wr  = r.get("wr", 0) * 100
    pf  = r.get("pf", 0)
    print(f"  {t:5} → {thr}   (WR={wr:.0f}%  PF={pf:.2f})")

# ── Generar bloque para config.py ─────────────────────────────────────────────
print("\n" + "="*55)
print("Copia esto en config.py:")
print("="*55)
print("\nMIN_SCORE_BY_TICKER = {")
for t, r in sorted(optimal.items()):
    print(f'    "{t}": {r.get("threshold", 72)},')
print("}")
print(f'\nMIN_SCORE_BUY = {min(r.get("threshold",72) for r in optimal.values())}  # minimo global')
