# backtest_candidates.py — Evalua ETFs candidatos para agregar al universo
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT, MIN_SCORE_BUY

INITIAL_CAPITAL  = 100_000
MAX_POSITION_PCT = 0.10
BACKTEST_DAYS    = 500
TRAIN_DAYS       = 252

CANDIDATES = {
    "QQQ":  "Nasdaq 100 (tech amplio)",
    "IWM":  "Russell 2000 (small cap)",
    "GLD":  "Oro (hedge)",
    "XLV":  "Healthcare (defensivo)",
    "XLE":  "Energia",
    "DIA":  "Dow Jones",
    "EFA":  "Mercados desarrollados ex-US",
    "AGG":  "Bonos agregados (bajo riesgo)",
    "ARKK": "Innovacion (alta volatilidad)",
    "XLK":  "Tecnologia pura",
}

# Universo actual para comparar
CURRENT = ["SPY", "VGT", "SCHD", "IFRA", "SCHF"]

def _rsi(s, p=14):
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    loss  = (-delta).clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

def compute_scores(df):
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
        if   rsi < 30:    rs = 20.0
        elif rsi <= 50:   rs = 40.0 + (rsi - 30)
        elif rsi <= 70:   rs = 60.0 + (rsi - 50)
        else:             rs = max(0.0, 90.0 - (rsi - 70) * 2.0)
        price     = float(close.iloc[i])
        macd_norm = float(np.clip(50 + (float(macd_h.iloc[i]) / max(price * 0.002, 1e-9)) * 25, 0, 100))
        trend     = (35 if price > float(sma50.iloc[i]) else 0) + \
                    (35 if price > float(sma200.iloc[i]) else 0) + \
                    (30 if float(sma50.iloc[i]) > float(sma200.iloc[i]) else 0)
        bb  = float(np.clip(bb_pos.iloc[i], 0, 100))
        vol = float(min(float(vol_r.iloc[i]), 100))
        scores.iloc[i] = rs*0.25 + macd_norm*0.25 + trend*0.30 + bb*0.10 + vol*0.10
    return scores

def backtest(df):
    close  = df["Close"].squeeze()
    scores = compute_scores(df)
    capital, position, trades, equity = INITIAL_CAPITAL, None, [], []

    for i in range(len(df)):
        date, price, score = df.index[i], float(close.iloc[i]), scores.iloc[i]
        if position is not None:
            pnl = (price - position["entry"]) / position["entry"]
            if pnl <= -STOP_LOSS_PCT:
                capital += position["shares"] * price
                trades.append({"pnl": position["shares"]*price - position["cost"], "pnl_pct": pnl*100})
                position = None
            elif pnl >= TAKE_PROFIT_PCT:
                capital += position["shares"] * price
                trades.append({"pnl": position["shares"]*price - position["cost"], "pnl_pct": pnl*100})
                position = None

        if position is None and not pd.isna(score) and score >= MIN_SCORE_BUY:
            inv = capital * MAX_POSITION_PCT
            capital -= inv
            position = {"entry": price, "shares": inv/price, "cost": inv}

        equity.append(capital + (position["shares"]*price if position else 0))

    if position:
        price = float(close.iloc[-1])
        pnl   = (price - position["entry"]) / position["entry"]
        capital += position["shares"] * price
        trades.append({"pnl": position["shares"]*price - position["cost"], "pnl_pct": pnl*100})
        equity[-1] = capital

    if not trades:
        return None

    eq  = pd.Series(equity, index=df.index)
    tr  = (eq.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    cagr= ((eq.iloc[-1]/INITIAL_CAPITAL)**(1/max(yrs,0.01))-1)*100
    mdd = ((eq - eq.expanding().max())/eq.expanding().max()*100).min()
    wins= [t for t in trades if t["pnl"] > 0]
    wr  = len(wins)/len(trades)*100
    gw  = sum(t["pnl"] for t in wins)
    gl  = abs(sum(t["pnl"] for t in trades if t["pnl"]<=0)) or 1e-9
    dr  = eq.pct_change().dropna()
    sh  = ((dr.mean()*252-0.02)/(dr.std()*np.sqrt(252))) if dr.std()>0 else 0
    return {"trades": len(trades), "wr": round(wr,1), "pf": round(gw/gl,2),
            "ret": round(tr,2), "cagr": round(cagr,2), "mdd": round(mdd,2), "sharpe": round(sh,2)}

def main():
    end   = datetime.today()
    start = end - timedelta(days=BACKTEST_DAYS + TRAIN_DAYS + 10)

    all_tickers = list(CANDIDATES.keys()) + CURRENT
    print(f"\nDescargando {len(all_tickers)} activos...\n")

    print(f"{'Ticker':<6} {'Descripcion':<32} {'Trades':>7} {'WinRate':>8} {'PF':>5} {'Return':>9} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8}  Recomendacion")
    print("-" * 115)

    current_avg_wr  = []
    current_avg_ret = []

    for ticker in CURRENT:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty: continue
        r = backtest(df.tail(BACKTEST_DAYS + TRAIN_DAYS))
        if r:
            current_avg_wr.append(r["wr"])
            current_avg_ret.append(r["ret"])

    avg_wr  = np.mean(current_avg_wr)  if current_avg_wr  else 52
    avg_ret = np.mean(current_avg_ret) if current_avg_ret else 1.2

    print(f"  [Universo actual — promedio: WinRate={avg_wr:.1f}%  Return={avg_ret:+.1f}%]")
    print()

    for ticker, desc in CANDIDATES.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty or len(df) < 300:
                print(f"{ticker:<6} {desc:<32}  sin datos suficientes")
                continue
            r = backtest(df.tail(BACKTEST_DAYS + TRAIN_DAYS))
            if not r:
                print(f"{ticker:<6} {desc:<32}  sin senales")
                continue

            # Recomendacion
            if r["wr"] >= 55 and r["pf"] >= 1.5 and r["mdd"] > -15:
                rec = "AGREGAR"
            elif r["wr"] >= 50 and r["pf"] >= 1.2:
                rec = "Considerar"
            elif r["mdd"] < -20:
                rec = "Muy volatil"
            else:
                rec = "No recomendado"

            print(f"{ticker:<6} {desc:<32} {r['trades']:>7} {r['wr']:>7.1f}% {r['pf']:>5.2f}"
                  f" {r['ret']:>+8.1f}% {r['cagr']:>+7.1f}% {r['sharpe']:>7.2f} {r['mdd']:>+7.1f}%  {rec}")
        except Exception as e:
            print(f"{ticker:<6} {desc:<32}  ERROR: {e}")

    print("-" * 115)
    print("\nCriterios: WinRate>=55% + PF>=1.5 + MaxDD>-15% = AGREGAR")

if __name__ == "__main__":
    main()
