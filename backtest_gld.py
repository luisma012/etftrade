"""
Backtest de GLD — Momentum strategy con parámetros del bot
SL=5%, TP=12%, Trailing 6%/3%, capital=$100,000
Produce: métricas completas + equity curve (PNG)
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Parámetros del bot ─────────────────────────────────────────────────────────
SYMBOL        = "GLD"
CAPITAL       = 100_000
POSITION_PCT  = 0.10
STOP_LOSS     = 0.05
TAKE_PROFIT   = 0.12
TRAIL_ACTIVATE = 0.06
TRAIL_PCT     = 0.03
COMMISSION    = 0.001   # 0.1% por trade
PERIOD_DAYS   = 500
RSI_PERIOD    = 14
SMA_FAST      = 20
SMA_SLOW      = 50

# ── Descarga de datos ──────────────────────────────────────────────────────────
print(f"Descargando {SYMBOL}...")
end   = datetime.today()
start = end - timedelta(days=PERIOD_DAYS + 60)
df = yf.download(SYMBOL, start=start, end=end, progress=False, auto_adjust=True)
df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
df = df[["open", "high", "low", "close", "volume"]].dropna()
print(f"  {len(df)} barras | {df.index[0].date()} a {df.index[-1].date()}")

# ── Indicadores ───────────────────────────────────────────────────────────────
close = df["close"]

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d).clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-9))

df["rsi"]      = rsi(close)
df["sma_fast"] = close.rolling(SMA_FAST).mean()
df["sma_slow"] = close.rolling(SMA_SLOW).mean()
df["macd"]     = close.ewm(span=12).mean() - close.ewm(span=26).mean()
df["macd_sig"] = df["macd"].ewm(span=9).mean()
df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
df = df.dropna()

# ── Señales (replica lógica técnica del bot) ───────────────────────────────────
def score(row):
    s = 0
    rsi_v = row["rsi"]
    if   rsi_v < 30: s += 25
    elif rsi_v < 45: s += 20
    elif rsi_v < 55: s += 10
    elif rsi_v < 70: s += 5
    if row["macd"] > row["macd_sig"]: s += 20
    if row["close"] > row["sma_fast"]: s += 17
    if row["close"] > row["sma_slow"]: s += 18
    if row["sma_fast"] > row["sma_slow"]: s += 10
    if row["vol_ratio"] > 1.2: s += 10
    return min(s, 100)

df["score"] = df.apply(score, axis=1)
df["signal"] = (df["score"] >= 72).astype(int)

# ── Backtest ──────────────────────────────────────────────────────────────────
equity      = CAPITAL
position    = 0.0
entry_price = 0.0
high_price  = 0.0
trailing    = False
equity_curve = []
trades      = []

for i, (dt, row) in enumerate(df.iterrows()):
    price = float(row["close"])

    if position > 0:
        gain = (price - entry_price) / entry_price

        # Trailing stop activation
        if gain >= TRAIL_ACTIVATE:
            trailing = True
        if price > high_price:
            high_price = price

        # Exit conditions
        exit_reason = None
        if trailing and price <= high_price * (1 - TRAIL_PCT):
            exit_reason = "TRAILING_STOP"
        elif gain >= TAKE_PROFIT:
            exit_reason = "TAKE_PROFIT"
        elif gain <= -STOP_LOSS:
            exit_reason = "STOP_LOSS"

        if exit_reason:
            pnl = position * (price - entry_price) - (position * price * COMMISSION)
            equity += pnl
            trades.append({
                "exit_date": dt, "exit_price": price,
                "pnl": pnl, "pnl_pct": gain * 100,
                "reason": exit_reason,
            })
            position = 0.0
            trailing = False

    elif row["signal"] == 1 and position == 0:
        size = equity * POSITION_PCT
        shares = size / price
        cost = shares * price * COMMISSION
        position = shares
        entry_price = price
        high_price = price
        trades.append({
            "entry_date": dt, "entry_price": price,
            "exit_date": None, "exit_price": None,
            "pnl": None, "pnl_pct": None, "reason": None,
        })

    equity_curve.append({"date": dt, "equity": equity + (position * price if position > 0 else 0)})

eq = pd.DataFrame(equity_curve).set_index("date")["equity"]

# ── Métricas ──────────────────────────────────────────────────────────────────
completed = [t for t in trades if t.get("reason")]
n_trades  = len(completed)
wins      = [t for t in completed if t["pnl"] > 0]
win_rate  = len(wins) / n_trades * 100 if n_trades else 0
gross_p   = sum(t["pnl"] for t in wins)
gross_l   = abs(sum(t["pnl"] for t in completed if t["pnl"] <= 0))
pf        = gross_p / gross_l if gross_l > 0 else float("inf")
total_ret = (eq.iloc[-1] - CAPITAL) / CAPITAL * 100
days      = (eq.index[-1] - eq.index[0]).days
cagr      = ((eq.iloc[-1] / CAPITAL) ** (365 / days) - 1) * 100 if days > 0 else 0

daily_ret = eq.pct_change().dropna()
sharpe    = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
neg_ret   = daily_ret[daily_ret < 0]
sortino   = (daily_ret.mean() / neg_ret.std()) * np.sqrt(252) if len(neg_ret) > 0 else 0
roll_max  = eq.cummax()
drawdown  = (eq - roll_max) / roll_max
max_dd    = drawdown.min() * 100
calmar    = cagr / abs(max_dd) if max_dd != 0 else 0
var_95    = np.percentile(daily_ret, 5) * 100

exit_counts = {}
for t in completed:
    exit_counts[t["reason"]] = exit_counts.get(t["reason"], 0) + 1

# ── Imprimir resultados ────────────────────────────────────────────────────────
print(f"""
{'='*70}
  BACKTEST: {SYMBOL}  |  {df.index[0].date()} → {df.index[-1].date()}
  Capital: ${CAPITAL:,}  |  Score mínimo: 72  |  SL={STOP_LOSS*100:.0f}%  TP={TAKE_PROFIT*100:.0f}%
{'='*70}
  RENDIMIENTO                        RIESGO
  Total Return : {total_ret:+.2f}%              Max Drawdown : {max_dd:.2f}%
  CAGR         : {cagr:+.2f}%              VaR (95%)    : {var_95:.2f}%
  Sharpe Ratio : {sharpe:.2f}               Volatilidad  : {daily_ret.std()*np.sqrt(252)*100:.1f}% anual
  Sortino Ratio: {sortino:.2f}               Calmar Ratio : {calmar:.2f}
{'-'*70}
  TRADES
  Total trades : {n_trades}                Win Rate     : {win_rate:.1f}%
  Profit Factor: {pf:.2f}               Capital final: ${eq.iloc[-1]:,.0f}
{'-'*70}
  Salidas:""")
for reason, count in exit_counts.items():
    avg_pnl = np.mean([t["pnl_pct"] for t in completed if t["reason"] == reason])
    print(f"    {reason:<20} {count:>3} trades  avg P&L: {avg_pnl:+.1f}%")
print('='*70)

# ── Equity Curve ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(f"GLD Backtest — Sharpe: {sharpe:.2f} | Return: {total_ret:+.1f}% | MaxDD: {max_dd:.1f}%",
             fontsize=13, fontweight="bold")

ax1.plot(eq.index, eq.values, color="#2196F3", linewidth=1.5, label="Equity")
ax1.axhline(CAPITAL, color="gray", linestyle="--", linewidth=0.8, label="Capital inicial")
ax1.fill_between(eq.index, CAPITAL, eq.values,
                 where=(eq.values >= CAPITAL), alpha=0.15, color="green")
ax1.fill_between(eq.index, CAPITAL, eq.values,
                 where=(eq.values < CAPITAL), alpha=0.15, color="red")
ax1.set_ylabel("Equity ($)")
ax1.legend(loc="upper left")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.grid(alpha=0.3)

ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="red", alpha=0.5)
ax2.set_ylabel("Drawdown (%)")
ax2.set_xlabel("Fecha")
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.tight_layout()
out = "backtest_gld_equity.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nGráfico guardado: {out}")
plt.show()
